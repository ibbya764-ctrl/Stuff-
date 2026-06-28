"""5. Collapse becomes gradual consolidation. [SPECULATIVE]

Collapse is no longer a final global event: it is gradual consolidation along
the scale-landscape.

    dense, agreeing, operator-coupled regions are observed (unitarily, 3) and
    CRYSTALLIZE -> permanent "singularities", fixed modes folded into the spine,
    -1/2 pinned;
    sparse, disagreeing regions stay LIQUID and keep evolving.

The crystallized regions are the consolidated singularities of the earlier
self-organizing loop -- repeatedly-useful structure folded into the operator as
fixed critical-line modes, so the model stops re-deriving them. The -1/2 stays
pinned through ALL crystallization (consolidation may add/reshape omega-modes,
never move the real part). This module decides which samples crystallize and
folds their dominant frequency into the operator spine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .operator import PinnedDilationOperator, CRITICAL_LINE


@dataclass
class Consolidation:
    """Tracks liquid/solid state per sample and folds solids into the spine.

    A sample crystallizes when it is (i) dense/agreeing enough -- low liquidity
    -- and (ii) operator-coupled (high occupation mass on a single mode). On
    crystallization its dominant frequency is consolidated into the operator at
    Re=-1/2.
    """

    operator: PinnedDilationOperator
    liquidity_thresh: float = 0.35      # below this -> eligible to solidify
    coupling_thresh: float = 0.15       # peak mode occupation needed
    crystallized_modes: list = field(default_factory=list)

    def step(self, occupations: np.ndarray, liquidity: np.ndarray,
             nu_of_sample: np.ndarray) -> dict:
        """One consolidation pass.

        `nu_of_sample` is each sample's dominant operator frequency (the omega
        it would fold in). Returns which samples crystallized this pass.
        """
        occ = np.asarray(occupations, dtype=float)
        peak = occ.max(axis=1)                       # operator-coupling strength
        solid = (liquidity < self.liquidity_thresh) & (peak > self.coupling_thresh)
        newly = []
        for i in np.where(solid)[0]:
            idx = self.operator.consolidate_mode(float(nu_of_sample[i]))
            # the non-negotiable: the folded mode sits on the critical line.
            assert self.operator.eigenvalues.real[idx] == CRITICAL_LINE
            self.crystallized_modes.append(idx)
            newly.append(int(i))
        return {
            "newly_crystallized": newly,
            "n_solid": int(solid.sum()),
            "n_liquid": int((~solid).sum()),
            "spine_size": len(self.crystallized_modes),
        }

    def state(self, liquidity: np.ndarray) -> np.ndarray:
        """Per-sample label: 'solid' (consolidating) vs 'liquid' (evolving)."""
        return np.where(liquidity < self.liquidity_thresh, "solid", "liquid")


if __name__ == "__main__":
    from .operator import gue_initialized_operator
    op = gue_initialized_operator(16, seed=3)
    n0 = op.nu.size
    con = Consolidation(op)
    # toy: 6 samples, a couple dense+coupled (low liquidity, peaked occupation).
    n, N = 6, 16
    occ = np.full((n, N), 1.0 / N)
    occ[0, 3] = 0.6; occ[0] /= occ[0].sum()         # sample 0 peaked -> coupled
    occ[1, 7] = 0.5; occ[1] /= occ[1].sum()         # sample 1 peaked
    liquidity = np.array([0.1, 0.2, 0.9, 0.8, 0.95, 0.7])
    nu_dom = op.nu[np.argmax(occ, axis=1)]
    out = con.step(occ, liquidity, nu_dom)
    print("consolidation self-check")
    print(f"  states          = {con.state(liquidity).tolist()}")
    print(f"  newly crystallized = {out['newly_crystallized']}")
    print(f"  spine grew {n0} -> {op.nu.size} modes (all Re=-1/2: "
          f"{np.allclose(op.eigenvalues.real, CRITICAL_LINE)})")
    assert op.nu.size > n0 and np.allclose(op.eigenvalues.real, CRITICAL_LINE)
