"""5.7 -- Anti-collapse as a first-class subsystem. [ENGINEERING]

The same failure mode has appeared four times in the program -- shared pole
banks collapsing to one timescale, routed geometries to one expert, low-rank
branches to one identity, moving samples to one mode -- and the fix is always
the same primitive: a pressure that prevents everything collapsing into one
mode. Diversity loss, MoE load-balancing, and sample repulsion are that one
mechanism wearing different clothes.

This module is that mechanism made first-class: ONE controller that every
module calls, exposing a `strength` and a `mode` per call (the one design
requirement from the note -- the right strength differs by module), and the
collapse diagnostic (the spread/variance check) implemented ONCE so it is
watched everywhere.

Numpy-only. No GPU, no model. Self-test: `python3 -m continuum.anti_collapse`.

Modes
-----
* ``soft_spread``  : structured spread for sample spacing -- cover, do not
                     scatter. Gaussian/RBF-kernel repulsion (the generic SVGD
                     primitive; this is the A/B *baseline* of 5.6).
* ``hard_balance`` : load-balancing for routed geometries -- use ALL slots.
                     Pushes a usage distribution toward uniform.
* ``merge_or_repel``: redundancy force anchored to operator-mode overlap
                     (5.6 refinement) -- repel samples occupying *different*
                     operator modes, merge (attract) those occupying the
                     *same* ones. Repels by mode-difference, not geometric
                     distance, so it is not circular. [SPECULATIVE here: the
                     "operator modes" are passed in as occupation vectors; in
                     the real architecture they come from the shared spine.]

The GUE-targeted force law of 5.6 is deliberately NOT a mode here: it is the
experimental upgrade that scale_dynamics A/B's *against* this baseline, so the
GUE claim is measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# Collapse diagnostic -- implemented ONCE, watched everywhere.
# ----------------------------------------------------------------------

def collapse_diagnostic(
    positions: np.ndarray,
    *,
    floor: float = 1e-3,
    target_low: float | None = None,
    target_high: float | None = None,
) -> dict:
    """The single spread/variance collapse check (5.6 'The diagnostics').

    The governing diagnostic is Var(positions). If it goes to zero the
    population has collapsed. But the target is not *maximum* spread -- it is
    *structured* spread: high enough to cover, low enough to focus. So this
    reports both the raw collapse flag (var < floor) and, if a
    [target_low, target_high] band is supplied, whether the spread sits in the
    structured-spread band (the galaxy picture as a measurable target).

    Accepts 1-D positions or (n, d) -- variance is summed over dims (total
    spread), matching "Var(s_1, ..., s_N)" for the scalar scale axis.
    """
    p = np.asarray(positions, dtype=float)
    var = float(np.var(p, axis=0).sum())
    out = {
        "var": var,
        "collapsed": bool(var < floor),
        "n": int(p.shape[0]),
    }
    if target_low is not None and target_high is not None:
        out["structured_spread"] = bool(target_low <= var <= target_high)
        out["target_band"] = (float(target_low), float(target_high))
    return out


# ----------------------------------------------------------------------
# The controller.
# ----------------------------------------------------------------------

@dataclass
class AntiCollapseController:
    """One primitive serving geometry routing, sample spacing, branch
    diversity, adaptive rank and adaptive sampling, with one set of knobs.

    `force(...)` returns a per-element push (same shape as the input) that the
    caller integrates / adds to a loss gradient. `loss(...)` returns the scalar
    penalty form for modules that want a regularizer rather than a force. The
    two are consistent: force == -d loss / d position for soft_spread.
    """

    bandwidth: float = 1.0          # kernel length-scale h (soft_spread)
    eps: float = 1e-9               # numerical floor

    # -- the per-call interface: strength + mode --------------------------

    def force(
        self,
        positions: np.ndarray,
        *,
        strength: float = 1.0,
        mode: str = "soft_spread",
        modes_occupied: np.ndarray | None = None,
    ) -> np.ndarray:
        """Anti-collapse push for each element. Shape matches `positions`
        (or, for hard_balance, matches the usage vector passed as positions)."""
        if mode == "soft_spread":
            return strength * self._rbf_repulsion(positions)
        if mode == "hard_balance":
            return strength * self._load_balance(positions)
        if mode == "merge_or_repel":
            if modes_occupied is None:
                raise ValueError(
                    "merge_or_repel needs `modes_occupied` (operator-mode "
                    "occupation per sample); geometric distance cannot tell "
                    "redundant-far from distinct-near apart."
                )
            return strength * self._redundancy_force(positions, modes_occupied)
        raise ValueError(f"unknown anti-collapse mode: {mode!r}")

    # -- mode implementations --------------------------------------------

    def _rbf_repulsion(self, positions: np.ndarray) -> np.ndarray:
        """Generic Gaussian/RBF-kernel repulsion -- the SVGD primitive.

        F_i = sum_j (x_i - x_j) / h^2 * exp(-|x_i - x_j|^2 / 2h^2).
        Smooth, bounded, spreads particles to *represent a distribution*.
        This is the safe default and the 5.6 A/B baseline.
        """
        x = np.asarray(positions, dtype=float)
        x2 = x[:, None] if x.ndim == 1 else x
        diff = x2[:, None, :] - x2[None, :, :]          # (n, n, d)
        d2 = np.sum(diff ** 2, axis=-1)                 # (n, n)
        k = np.exp(-d2 / (2.0 * self.bandwidth ** 2))   # RBF kernel
        f = np.sum(k[:, :, None] * diff, axis=1) / self.bandwidth ** 2
        return f.reshape(np.shape(positions))

    def _load_balance(self, usage: np.ndarray) -> np.ndarray:
        """Hard load-balancing: push a usage distribution toward uniform.

        Returns the negative gradient of the load-balance penalty
        sum_e (u_e - 1/E)^2, i.e. a pull on under/over-used slots toward 1/E.
        Geometry routing wants this hard ('use all experts').
        """
        u = np.asarray(usage, dtype=float)
        target = 1.0 / u.size
        return -(u - target)

    def _redundancy_force(
        self, positions: np.ndarray, modes_occupied: np.ndarray
    ) -> np.ndarray:
        """Repel by operator-mode DIFFERENCE, merge by mode OVERLAP (5.6).

        modes_occupied[i] is sample i's occupation over the shared operator's
        modes. Overlap o_ij = <m_i, m_j> / (|m_i||m_j|) in [0, 1]. Samples that
        overlap (redundant) attract/merge; samples that are orthogonal
        (distinct) repel. The push acts along the geometric axis but is
        *gated by information overlap*, so two samples far apart in scale but
        encoding the same modes are pulled together, and two close samples
        encoding different modes are kept apart -- exactly what geometric
        distance alone cannot do. [SPECULATIVE; build after geometric stable.]
        """
        x = np.asarray(positions, dtype=float)
        x2 = x[:, None] if x.ndim == 1 else x
        m = np.asarray(modes_occupied, dtype=float)
        norm = np.linalg.norm(m, axis=1, keepdims=True) + self.eps
        mhat = m / norm
        overlap = mhat @ mhat.T                          # (n, n) in [-1, 1]
        # repel when distinct (overlap -> 0): weight = (1 - overlap),
        # merge when redundant (overlap -> 1): weight goes negative.
        weight = 1.0 - 2.0 * np.clip(overlap, 0.0, 1.0)  # +1 distinct .. -1 same
        np.fill_diagonal(weight, 0.0)
        diff = x2[:, None, :] - x2[None, :, :]
        d = np.linalg.norm(diff, axis=-1, keepdims=True) + self.eps
        direction = diff / d                             # unit, away from j
        f = np.sum(weight[:, :, None] * direction, axis=1)
        return f.reshape(np.shape(positions))

    # -- the regularizer form (for loss-based callers) -------------------

    def loss(self, positions: np.ndarray, *, strength: float = 1.0) -> float:
        """Scalar diversity penalty (soft_spread): mean pairwise RBF overlap.

        Minimizing this spreads the population; -d/dx of it is _rbf_repulsion
        up to the kernel-derivative factor, so loss- and force-based callers
        get the same anti-collapse pressure.
        """
        x = np.asarray(positions, dtype=float)
        x2 = x[:, None] if x.ndim == 1 else x
        diff = x2[:, None, :] - x2[None, :, :]
        d2 = np.sum(diff ** 2, axis=-1)
        k = np.exp(-d2 / (2.0 * self.bandwidth ** 2))
        n = x2.shape[0]
        off = (np.sum(k) - n) / max(n * (n - 1), 1)      # mean off-diagonal
        return float(strength * off)


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def _selftest(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    ctrl = AntiCollapseController(bandwidth=0.5)

    # soft_spread: a clump should feel net outward pressure (spread grows).
    clump = rng.normal(scale=0.05, size=(40, 1))
    f = ctrl.force(clump, mode="soft_spread", strength=1.0)
    moved = clump + 0.1 * f
    assert np.var(moved) > np.var(clump), "soft_spread did not spread a clump"

    # collapse diagnostic: clump flagged collapsed, spread not.
    assert collapse_diagnostic(clump, floor=1e-2)["collapsed"]
    spread = rng.normal(scale=1.0, size=(40, 1))
    assert not collapse_diagnostic(spread, floor=1e-2)["collapsed"]

    # structured-spread band.
    diag = collapse_diagnostic(spread, target_low=0.5, target_high=2.0)
    assert "structured_spread" in diag

    # hard_balance: an over-used slot gets pushed down, under-used up.
    usage = np.array([0.7, 0.1, 0.1, 0.1])
    push = ctrl.force(usage, mode="hard_balance")
    assert push[0] < 0 < push[1], "hard_balance did not move toward uniform"

    # merge_or_repel: isolate each behaviour with a clean 2-sample pair.
    pair = np.array([[0.0], [1.0]])
    redundant = np.array([[1.0, 0.0], [1.0, 0.0]])   # same modes -> merge
    distinct = np.array([[1.0, 0.0], [0.0, 1.0]])    # orthogonal -> repel
    rf_red = ctrl.force(pair, mode="merge_or_repel", modes_occupied=redundant)
    rf_dis = ctrl.force(pair, mode="merge_or_repel", modes_occupied=distinct)
    # redundant: sample 0 pulled toward sample 1 (+x); distinct: pushed away (-x).
    assert rf_red[0, 0] > 0, "merge_or_repel did not merge a redundant pair"
    assert rf_dis[0, 0] < 0, "merge_or_repel did not repel a distinct pair"

    # loss form is non-negative and decreases as a clump spreads.
    assert ctrl.loss(clump) > ctrl.loss(spread)

    print("anti_collapse self-test PASSED")
    print(f"  soft_spread   : Var {np.var(clump):.4f} -> {np.var(moved):.4f} "
          f"after one push")
    print(f"  collapse diag : clump collapsed={collapse_diagnostic(clump, floor=1e-2)['collapsed']}, "
          f"spread collapsed={collapse_diagnostic(spread, floor=1e-2)['collapsed']}")
    print(f"  hard_balance  : push[0]={push[0]:+.3f} (over-used down), "
          f"push[1]={push[1]:+.3f} (under-used up)")
    print(f"  merge_or_repel: redundant force[0]={rf_red[0,0]:+.3f} (merges), "
          f"distinct force[0]={rf_dis[0,0]:+.3f} (repels)")


if __name__ == "__main__":
    _selftest()
