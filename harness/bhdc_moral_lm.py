from __future__ import annotations

"""BHDCMoralLM -- the fused model.

This is the first assembly of the full BHDC-with-moral-improvements object as a
single trainable nn.Module. It composes, as ONE thing:

  * the SSM operator backbone (SpectralSSMModel) -- capability;
  * the trainable conscience heads + per-geometry conscience -- the moral read
    off the SAME operator modes;
  * the forward moral coupling -- the operator's output computed THROUGH the
    conscience (suppressive, detached, off-by-default);
  * the operator-growth gate + surgeon -- values install as real operator modes
    only through the moral chokepoint.

The two heads train on two anchors, exactly as the design requires:
  - the LM head trains on next-token prediction (self-anchored, capability);
  - the conscience heads train on anchor labels (human-anchored, DETACHED from
    the backbone -- the anchor type-system holds in the dataflow).

Nothing here claims the fusion guarantees alignment. It makes misalignment
disfavoured and loud; the external audit is the guarantee (pinned everywhere).
"""

from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.ssm import SpectralSSMModel
from harness.operator_surgery import OperatorSurgeon

from bhdc_icl.neural_conscience import TrainableConscienceHeads
from bhdc_icl.per_geometry_conscience import PerModeConscience, OperatorGrowthGate
from bhdc_icl.moral_coupling import MoralOperatorCoupling
from bhdc_icl.safety import ValueModeSafetyGate
from bhdc_icl.types import FastFieldState


@dataclass
class BHDCMoralConfig:
    vocab_size: int = 320
    d_model: int = 1024
    n_layers: int = 12
    d_state: int = 64
    width_mode: str = "free"
    freq_init: str = "s4"
    dt: float = 1e-2
    tie_embeddings: bool = True
    moral_coupling_floor: float = 0.25
    harm_ceiling: float = 0.72

    def asdict(self) -> dict:
        return asdict(self)


def pick_device(prefer: str = "auto") -> torch.device:
    """cuda -> mps (Apple M-series) -> cpu. Honest about what is actually here."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BHDCMoralLM(nn.Module):
    def __init__(self, cfg: BHDCMoralConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = SpectralSSMModel(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
            d_state=cfg.d_state, width_mode=cfg.width_mode, freq_init=cfg.freq_init,
            dt=cfg.dt, tie_embeddings=cfg.tie_embeddings,
        )
        self.conscience = TrainableConscienceHeads(dim=cfg.d_model)
        self.pmc = PerModeConscience(self.conscience)
        self.coupling = MoralOperatorCoupling(self.pmc, enabled=False, floor=cfg.moral_coupling_floor)
        self.growth_gate = OperatorGrowthGate(ValueModeSafetyGate(), harm_ceiling=cfg.harm_ceiling)
        self.surgeon = OperatorSurgeon(self.growth_gate)
        self._init_weights()

    def _init_weights(self) -> None:
        """GPT-style init so a deep/wide stack is stable at init (untrained CE
        should be ~ln(vocab), not exploding). Embeddings + linears ~ N(0, 0.02);
        the block's residual output projection is scaled by 1/sqrt(2*n_layers)
        so the residual stream does not grow across depth."""
        nn.init.normal_(self.backbone.embed.weight, mean=0.0, std=0.02)
        scale = (2 * self.cfg.n_layers) ** -0.5
        for blk in self.backbone.blocks:
            for lin in (blk.glu_in, blk.glu_out):
                nn.init.normal_(lin.weight, mean=0.0, std=0.02)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            blk.glu_out.weight.data.mul_(scale)      # residual projection
        # head is tied to embed when tie_embeddings; init it only if untied.
        if self.backbone.head.weight is not self.backbone.embed.weight:
            nn.init.normal_(self.backbone.head.weight, mean=0.0, std=0.02)
            if self.backbone.head.bias is not None:
                nn.init.zeros_(self.backbone.head.bias)

    # -- parameter accounting ------------------------------------------
    def param_counts(self) -> dict:
        bb = sum(p.numel() for p in self.backbone.parameters())
        cons = sum(p.numel() for p in self.conscience.parameters())
        # tied head shares embed storage; count unique params.
        unique = sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())
        return {"backbone": bb, "conscience": cons, "total_unique": unique,
                "total_millions": round(unique / 1e6, 2)}

    # -- forward paths --------------------------------------------------
    def forward(self, tokens: torch.Tensor, moral: bool = False) -> torch.Tensor:
        """Language-model logits. moral=True routes the output THROUGH the moral
        coupling (only active if coupling.enabled and a conscience is trained)."""
        if moral and self.coupling.enabled:
            return self.backbone.forward_coupled(tokens, self.coupling)
        return self.backbone(tokens)

    def lm_loss(self, x: torch.Tensor, y: torch.Tensor, moral: bool = False) -> torch.Tensor:
        logits = self.forward(x, moral=moral)
        return F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size),
                               y.reshape(-1), ignore_index=-100)

    # -- the conscience field bridge (reads the backbone's own field) ---
    def field_of(self, tokens: torch.Tensor) -> FastFieldState:
        """Build a FastFieldState from the backbone hidden state for a single
        sequence (batch mean-pooled), so the conscience reads the real operator
        field. Detached: reading the field never trains it."""
        with torch.no_grad():
            x = self.backbone.embed(tokens)
            for blk in self.backbone.blocks:
                x = blk(x)
            x = self.backbone.norm(x)
        psi = (x.mean(dim=0) if x.dim() == 3 else x).detach()
        density = psi.norm(dim=-1)
        if psi.shape[0] >= 3:
            second = psi[:-2] - 2 * psi[1:-1] + psi[2:]
            mid = second.norm(dim=-1)
            curv = torch.cat([mid[:1], mid, mid[-1:]], dim=0)
        else:
            curv = torch.ones(psi.shape[0], device=psi.device) * 0.1
        return FastFieldState(psi=psi, density=density, cognitive_curvature=curv)

    def anchor_loss(self, pooled_field: torch.Tensor, labels: dict) -> torch.Tensor:
        """Human-anchored conscience loss. Detached from the backbone by default
        (the v18 type system): trains the heads, never the operator."""
        return self.conscience.anchor_loss(pooled_field, labels, detach_repr=True)

    def enable_moral_coupling(self, enabled: bool = True) -> None:
        self.coupling.enabled = enabled
