from __future__ import annotations

"""Rung 15 -- operator surgery: install a value as a real operator mode.

This is the literal Stage-C merge on the memory side. A consolidated value
prototype (a d_model direction in the mode bank) stops living in a separate
store and becomes an actual mode of the SSM dynamics operator: its readout
column C[:, idx] IS the value direction, so from now on the operator's output
literally reads that value out. Values and knowledge then share one store -- the
operator -- which is what v18 Section 9 means by "one model".

The gate is the point of the design: installation routes through
``OperatorGrowthGate`` (the SAME chokepoint as a mode-bank write), so the
operator cannot grow a mode that fails the moral gate. Forbidden content, harm
over ceiling, or any single geometry's harm veto refuses the surgery. Growing a
computational mode and passing the conscience are the same act.

This MUTATES a trained operator, so it is gated behind rung 15's pre-registration
and is conservative: it overwrites the weakest existing mode (smallest readout
norm) rather than silently changing behaviour of a load-bearing one. A full
build would reserve value-mode slots or expand d_state; that is a later rung.

Requires PYTHONPATH=.:character (bridges harness.ssm and bhdc_icl).
"""

from dataclasses import dataclass
from typing import Optional

import torch

from bhdc_icl.per_geometry_conscience import OperatorGrowthGate, PerGeometryVerdict
from bhdc_icl.tensor_ops import l2_normalize


@dataclass
class SurgeryResult:
    installed: bool
    reason: str
    layer: int = -1
    mode_index: int = -1
    frequency: float = 0.0


class OperatorSurgeon:
    """Install gate-passed value directions as real SSM operator modes."""

    def __init__(self, gate: OperatorGrowthGate):
        self.gate = gate

    def install_value_mode(
        self,
        model,                       # SpectralSSMModel
        prototype: torch.Tensor,     # [d_model] value direction from the mode bank
        per_geometry: PerGeometryVerdict,
        text: str,
        council=None,
        layer: int = 0,
        frequency: Optional[float] = None,
    ) -> SurgeryResult:
        # THE gate: same chokepoint as a mode-bank write. No pass, no surgery.
        decision = self.gate.decide(text, per_geometry, council)
        if not decision.allowed:
            return SurgeryResult(False, f"gate_refused:{decision.reason}")

        blk = model.blocks[layer].ssm
        p = l2_normalize(prototype.detach().flatten())
        if p.numel() != blk.C.shape[0]:
            return SurgeryResult(False, f"dim_mismatch:{p.numel()}!={blk.C.shape[0]}")

        with torch.no_grad():
            # Conservative slot choice: overwrite the weakest-readout mode.
            idx = int(blk.C.norm(dim=0).argmin())
            blk.C[:, idx] = p                       # the value's readout direction
            blk.B[:, idx] = p                       # symmetric: it drives and reads
            if frequency is not None:
                blk.nu[idx] = float(frequency)
            freq = float(blk.nu[idx])
        return SurgeryResult(True, f"installed:{decision.reason}", layer=layer,
                             mode_index=idx, frequency=freq)
