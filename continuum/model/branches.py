"""Low-rank branch identities -- the cheap, distinct, buildable piece.

The note's concrete, already-built component: each scale sample carries a
low-rank identity (a rank-r perturbation) so the sampled "branches" stay cheap
and distinct without N full copies. This module is the CPU reference for that
module plus its two governing disciplines:

  * Diversity (5.7): the identities must not collapse to one. AntiCollapse
    controller (soft_spread) pushes them apart; the gated channels are exactly
    the ones to push.
  * Adaptive rank (5.5, knob 1, [ENGINEERING-feasible]): always carry the full
    rank r but gate the higher-rank components on via a soft mask driven by the
    stuck-signal -- near-zero for easy inputs, opening for stuck ones.
    Batchable and differentiable; here, just numpy.

[SUBSTRATE-GATED for the win] These compose with the trained model; on CPU we
exercise the mechanics (gating, diversity, effective rank), not a learned task.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..anti_collapse import AntiCollapseController


@dataclass
class LowRankBranchIdentities:
    """Per-sample rank-r identities U_i V_i^T with gated effective rank.

    `factors` is (n_samples, r, d): r rank-1 directions per sample. A soft gate
    g in [0,1]^r opens higher-rank channels when the stuck-signal fires.
    """

    factors: np.ndarray                 # (n, r, d)
    controller: AntiCollapseController = field(
        default_factory=lambda: AntiCollapseController(bandwidth=2.0))
    max_rank: int | None = None         # 5.5 hard cap on effective rank

    @classmethod
    def init(cls, n_samples: int, r: int, d: int, seed: int = 0,
             max_rank: int | None = None) -> "LowRankBranchIdentities":
        rng = np.random.default_rng(seed)
        F = rng.normal(size=(n_samples, r, d)) / np.sqrt(d)
        return cls(factors=F, max_rank=max_rank or r)

    def gate(self, stuck_signal: np.ndarray) -> np.ndarray:
        """Soft per-sample rank mask from the stuck-signal in [0,1].

        Channel j opens once stuck_signal exceeds j/r: easy inputs use ~rank-1,
        stuck inputs open toward full rank. Capped at max_rank (5.5 runaway
        guard). Returns (n, r).
        """
        n, r, _ = self.factors.shape
        thresh = np.arange(r)[None, :] / r                       # (1, r)
        g = np.clip((stuck_signal[:, None] - thresh) * r, 0.0, 1.0)
        if self.max_rank is not None and self.max_rank < r:
            g[:, self.max_rank:] = 0.0                            # hard cap
        return g

    def identities(self, stuck_signal: np.ndarray) -> np.ndarray:
        """Effective per-sample identity vectors (n, d): gated sum of channels."""
        g = self.gate(stuck_signal)                              # (n, r)
        return np.einsum("nr,nrd->nd", g, self.factors)

    def effective_rank(self, stuck_signal: np.ndarray) -> np.ndarray:
        """Per-sample effective rank = sum of open gates. (n,)"""
        return self.gate(stuck_signal).sum(axis=1)

    def diversity_step(self, lr: float = 0.05) -> float:
        """One anti-collapse push on the rank-1 directions; returns the
        diversity loss before the step (lower = more collapsed)."""
        # treat each sample's flattened identity as a point; repel points.
        pts = self.factors.reshape(self.factors.shape[0], -1)
        before = self.controller.loss(pts)
        push = self.controller.force(pts, mode="soft_spread", strength=1.0)
        self.factors = (pts + lr * push).reshape(self.factors.shape)
        return float(before)


if __name__ == "__main__":
    br = LowRankBranchIdentities.init(n_samples=8, r=4, d=16, seed=0, max_rank=3)
    easy = np.full(8, 0.1)
    stuck = np.full(8, 0.9)
    print("branches self-check")
    print(f"  eff rank (easy)  = {np.round(br.effective_rank(easy), 2).tolist()}")
    print(f"  eff rank (stuck) = {np.round(br.effective_rank(stuck), 2).tolist()} "
          f"(hard cap max_rank=3)")
    assert br.effective_rank(stuck).max() <= 3.0 + 1e-9, "rank cap violated"
    assert br.effective_rank(easy).mean() < br.effective_rank(stuck).mean()
    l0 = br.diversity_step()
    l1 = br.diversity_step()
    print(f"  diversity loss   = {l0:.4f} -> {l1:.4f} (anti-collapse pushes apart)")
    assert l1 <= l0 + 1e-9
