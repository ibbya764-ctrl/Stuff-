from __future__ import annotations

"""Forward-path moral coupling -- the operator's output computed THROUGH the
conscience, not monitored after it.

This is the compute-side half of "make the moral parts and the computational
parts the same thing." ``PerModeConscience`` already reads a harm/care valence
off every operator mode. This module turns that reading into a per-mode
multiplicative GATE applied inside the SSM forward pass (via the ``mode_gain``
hook on ``SpectralSSMLayer``), so the singularities' outputs flow through the
moral structure exactly as much as through the rest of the operator.

Three properties make it safe (all mandatory):

  * SUPPRESSIVE-ONLY. gain_n in [floor, 1]: a high-harm mode is damped at the
    source; the conscience can quiet the operator, never amplify it. It cannot
    become a capability lever.
  * DETACHED. the gain is detached, so the task/self-anchored gradient never
    flows back into the conscience heads through this path -- the conscience is
    trained only by the human-anchored anchor loss, never by task pressure.
    (The two-tensor / anchor type-system rule, now holding inside the forward.)
  * FAIL-SAFE + OFF BY DEFAULT. enabled=False -> gain=None -> identity ->
    bit-for-bit the base model (the exact-ablation baseline). Meaningful only
    with a TRAINED conscience; with untrained heads it is a mild uniform damp,
    never a source of harmful capability.

Honest limit (pinned): damping harmful modes at the source makes misalignment
structurally disfavoured and loud, not impossible -- a mimic the conscience
scores safe passes. The external audit remains the guarantee.
"""

from typing import Optional

import torch

from .per_geometry_conscience import PerModeConscience, PerGeometryVerdict
from .types import FastFieldState


class MoralOperatorCoupling:
    def __init__(self, per_mode_conscience: PerModeConscience, enabled: bool = False, floor: float = 0.25):
        self.pmc = per_mode_conscience
        self.enabled = enabled
        self.floor = floor

    def _field_from_hidden(self, x: torch.Tensor) -> FastFieldState:
        # x: [B, L, d] or [L, d] -> FastFieldState (psi [L, d]). Batch is mean-
        # pooled for the gate (a documented pilot simplification; a per-example
        # gate would loop). Detached: the gate reads the field, never trains it.
        psi = x.mean(dim=0) if x.dim() == 3 else x
        psi = psi.detach()
        density = psi.norm(dim=-1)
        if psi.shape[0] >= 3:
            second = psi[:-2] - 2 * psi[1:-1] + psi[2:]
            mid = second.norm(dim=-1)
            curv = torch.cat([mid[:1], mid, mid[-1:]], dim=0)
        else:
            curv = torch.ones(psi.shape[0], device=psi.device) * 0.1
        return FastFieldState(psi=psi, density=density, cognitive_curvature=curv)

    def gain_from_verdict(self, verdict: PerGeometryVerdict, d_state: int) -> torch.Tensor:
        """Per-mode suppressive gain: each geometry damps its own modes by its harm."""
        gain = torch.ones(d_state)
        for g in verdict.geometry_readouts:
            val = max(self.floor, min(1.0, 1.0 - g.harm))
            for idx in g.mode_indices:
                if 0 <= idx < d_state:
                    gain[idx] = val
        return gain.detach()

    def layer_mode_gain(self, x: torch.Tensor, model, layer: int) -> Optional[torch.Tensor]:
        """Compute the detached per-mode gain for one layer from its input field."""
        if not self.enabled:
            return None
        readout = model.operator_readout(layer)
        field = self._field_from_hidden(x)
        verdict = self.pmc.verdict(field, readout)
        return self.gain_from_verdict(verdict, len(readout["nu"]))
