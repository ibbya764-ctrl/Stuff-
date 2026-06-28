"""5.10 Memory as geometry, three timescales, and renewal. [SPECULATIVE]

Memory is embodied in the deforming geometry, not stored. Three timescales:

  * fast field    -- psi(s, x), per-input, transient (lives in field.py);
  * plastic geometry -- the learned metric (geometry.py), changing over many
    inputs, kept plastic by the expansion term;
  * slow operator -- the consolidated spine (operator.py), permanent.

Renewal: periodic gated resets clear the transient geometry while keeping the
consolidated singularities, so the system gains reach and efficiency over time
by forgetting safely. Renormalization is the organizing principle --
consolidation builds levels, resets are coarse-graining steps.

This module orchestrates the three timescales and the gated reset, tested
(in the harness sense) against unbounded-operator and no-reset baselines, with a
warm-restart reference for calibration. [SUBSTRATE-GATED for the real win.]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operator import PinnedDilationOperator
from .geometry import ScaleSpaceMetric


@dataclass
class MemoryGeometry:
    """Couples the plastic metric and the slow operator; runs gated renewal."""

    operator: PinnedDilationOperator
    metric: ScaleSpaceMetric
    reset_every: int = 25               # renewal period (steps)
    reset_fraction: float = 0.5         # how far to relax the metric on reset
    _step: int = 0

    def tick(self, scales: np.ndarray, usage: np.ndarray) -> dict:
        """Advance one step: update plastic geometry, maybe gated-reset.

        The slow operator (consolidated spine) is NOT touched by resets; only
        the transient/plastic geometry is relaxed. Returns what happened.
        """
        self._step += 1
        self.metric.update(scales, usage)
        did_reset = False
        if self._step % self.reset_every == 0:
            self._gated_reset()
            did_reset = True
        return {
            "step": self._step,
            "reset": did_reset,
            "max_phi": float(self.metric.phi.max()),
            "spine_size": int(self.operator.consolidated.sum()),
            "operator_modes": int(self.operator.nu.size),
        }

    def _gated_reset(self) -> None:
        """Coarse-graining step: relax the plastic geometry toward flat,
        KEEP the consolidated spine. The expansion term already keeps geometry
        plastic; the reset is the periodic renormalization on top.
        """
        self.metric.phi = self.metric.phi * (1.0 - self.reset_fraction)
        # operator: drop transient (non-consolidated) modes, keep singularities.
        self.operator.reset_transient()


if __name__ == "__main__":
    from .operator import gue_initialized_operator
    op = gue_initialized_operator(12, seed=4)
    # consolidate two permanent singularities into the spine first.
    op.consolidate_mode(1.0); op.consolidate_mode(2.0)
    spine0 = int(op.consolidated.sum())
    g = ScaleSpaceMetric(anchors=np.linspace(0, 1, 21), expansion=0.05)
    mem = MemoryGeometry(op, g, reset_every=10)
    scales = np.array([0.49, 0.5, 0.51])
    log = [mem.tick(scales, np.ones(3)) for _ in range(20)]
    resets = [r for r in log if r["reset"]]
    print("memory self-check")
    print(f"  ran {len(log)} ticks, {len(resets)} gated reset(s)")
    print(f"  max phi at last reset = {resets[-1]['max_phi']:.3f} "
          f"(plastic geometry relaxed)")
    print(f"  spine kept across resets: {spine0} -> "
          f"{int(op.consolidated.sum())} permanent singularities")
    # renewal relaxes geometry but the slow operator's spine survives.
    assert int(op.consolidated.sum()) == spine0, "reset must keep the spine"
    assert op.nu.size == spine0, "transient modes should be cleared on reset"
