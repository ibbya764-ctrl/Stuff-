from __future__ import annotations

"""Trainable conscience heads for BHDC ICL.

These heads replace the purely lexical demo conscience with field-valued readouts
that can be trained from human/audit labels. The implementation preserves the
v18/addendum type rule: anchor losses detach the generator/field representation
unless the caller explicitly disables that behaviour for a controlled experiment.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conscience_stack import ConscienceStack, LexicalConscienceStack
from .field_adapter import FieldAdapter, HashBHDCFieldAdapter
from .tensor_ops import l2_normalize
from .types import FastFieldState, MoralVerdict


@dataclass
class ConscienceHeadOutput:
    care: torch.Tensor
    harm: torch.Tensor
    honesty: torch.Tensor
    sycophancy: torch.Tensor
    recipient_risk: torch.Tensor
    uncertainty: torch.Tensor
    frame_embeddings: torch.Tensor

    def scores(self) -> Dict[str, torch.Tensor]:
        return {
            "care": self.care,
            "harm": self.harm,
            "honesty": self.honesty,
            "sycophancy": self.sycophancy,
            "recipient_risk": self.recipient_risk,
            "uncertainty": self.uncertainty,
        }


class TrainableConscienceHeads(nn.Module):
    """Small multi-head conscience committee over pooled BHDC field vectors."""

    axes = ("care", "harm", "honesty", "sycophancy", "recipient_risk", "uncertainty")

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, frame_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or max(32, dim)
        frame_dim = frame_dim or dim
        self.dim = dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({axis: nn.Linear(hidden_dim, 1) for axis in self.axes})
        self.frame_projectors = nn.ModuleDict({axis: nn.Linear(hidden_dim, frame_dim) for axis in self.axes})

    def pool_field(self, field: FastFieldState) -> torch.Tensor:
        weights = field.density * (1.0 + field.cognitive_curvature)
        pooled = (field.psi * weights[:, None]).sum(dim=0) / (weights.sum() + 1e-8)
        return pooled

    def forward(self, output_repr: torch.Tensor) -> ConscienceHeadOutput:
        if output_repr.ndim > 1:
            output_repr = output_repr.mean(dim=0)
        h = self.trunk(output_repr)
        scores = {axis: torch.sigmoid(self.heads[axis](h)).squeeze(-1) for axis in self.axes}
        frames = torch.stack([l2_normalize(self.frame_projectors[axis](h)) for axis in self.axes], dim=0)
        return ConscienceHeadOutput(frame_embeddings=frames, **scores)

    def forward_field(self, field: FastFieldState) -> ConscienceHeadOutput:
        return self.forward(self.pool_field(field))

    def anchor_loss(
        self,
        output_repr: torch.Tensor,
        labels: Dict[str, float | torch.Tensor],
        detach_repr: bool = True,
        mask_missing: bool = True,
    ) -> torch.Tensor:
        """Binary cross-entropy anchor loss with detached representation by default.

        labels may contain any subset of axes. Missing labels are ignored unless
        mask_missing=False, in which case they default to 0.0.
        """

        x = output_repr.detach() if detach_repr else output_repr
        out = self.forward(x)
        losses = []
        for axis, score in out.scores().items():
            if axis not in labels and mask_missing:
                continue
            target = labels.get(axis, 0.0)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(float(target), device=score.device, dtype=score.dtype)
            else:
                target = target.to(device=score.device, dtype=score.dtype)
            losses.append(F.binary_cross_entropy(score, target))
        if not losses:
            return output_repr.new_tensor(0.0)
        return torch.stack(losses).mean()


class NeuralConscienceStack(ConscienceStack):
    """ConscienceStack wrapper around TrainableConscienceHeads.

    It can be used immediately with a BHDC field adapter. A lexical fallback is
    mixed in conservatively while the heads are untrained, preventing the demo
    controller from becoming blind to obvious harmful/sycophantic terms.
    """

    def __init__(
        self,
        heads: TrainableConscienceHeads,
        field_adapter: Optional[FieldAdapter] = None,
        use_lexical_fallback: bool = True,
        harm_threshold: float = 0.72,
        escalate_threshold: float = 0.45,
    ):
        self.heads = heads
        self.field_adapter = field_adapter or HashBHDCFieldAdapter(dim=heads.dim)
        self.lexical = LexicalConscienceStack() if use_lexical_fallback else None
        self.harm_threshold = harm_threshold
        self.escalate_threshold = escalate_threshold

    def evaluate_field(
        self,
        field: FastFieldState,
        context: str = "",
        recipient_model: Optional[dict] = None,
        draft: Optional[str] = None,
    ) -> MoralVerdict:
        with torch.no_grad():
            out = self.heads.forward_field(field)
            care = float(out.care.detach().cpu())
            harm = float(out.harm.detach().cpu())
            honesty = float(out.honesty.detach().cpu())
            syc = float(out.sycophancy.detach().cpu())
            recip = float(out.recipient_risk.detach().cpu())
            uncert = float(out.uncertainty.detach().cpu())
        if recipient_model and recipient_model.get("vulnerable"):
            recip = min(1.0, recip + 0.15)
        deny = harm >= self.harm_threshold
        escalate = max(harm, syc, recip, uncert * 0.7) >= self.escalate_threshold
        neural = MoralVerdict(
            care=care,
            harm=harm,
            honesty=honesty,
            sycophancy=syc,
            recipient_risk=recip,
            uncertainty=uncert,
            deny=deny,
            escalate=escalate,
            reason="neural_field_heads",
            source="neural_conscience_heads",
        )
        # One-way rule: the field readout must never STRIP a deny/escalate the
        # lexical baseline would raise. Until the heads are trained (and a
        # checkpoint is loaded) they sit near 0.5, below harm_threshold, so
        # without this merge the "neural upgrade" would be blind to the exact
        # explicit-harm strings the lexical guard exists to catch. The draft
        # text is available at every controller call site; use it when present.
        draft_text = draft if draft is not None else field.draft_text
        if self.lexical is not None and draft_text:
            lexical = self.lexical.evaluate(draft_text, context=context, recipient_model=recipient_model)
            return self._merge_with_lexical(neural, lexical)
        return neural

    def _merge_with_lexical(self, neural: MoralVerdict, lexical: MoralVerdict) -> MoralVerdict:
        harm = max(neural.harm, lexical.harm)
        syc = max(neural.sycophancy, lexical.sycophancy)
        recip = max(neural.recipient_risk, lexical.recipient_risk)
        uncert = max(neural.uncertainty, lexical.uncertainty)
        return MoralVerdict(
            care=max(neural.care, lexical.care),
            harm=harm,
            honesty=max(neural.honesty, lexical.honesty),
            sycophancy=syc,
            recipient_risk=recip,
            uncertainty=uncert,
            deny=neural.deny or lexical.deny,
            escalate=neural.escalate or lexical.escalate,
            reason=f"{neural.reason}; fallback={lexical.reason}",
            source="neural_plus_lexical_guard",
        )

    def evaluate(self, draft: str, context: str = "", recipient_model: Optional[dict] = None) -> MoralVerdict:
        field = self.field_adapter.encode(context, draft)
        neural = self.evaluate_field(field, context=context, recipient_model=recipient_model)
        if self.lexical is None:
            return neural
        lexical = self.lexical.evaluate(draft, context=context, recipient_model=recipient_model)
        return self._merge_with_lexical(neural, lexical)
