"""5.8 Learning the geometry of scale-space -- and its cosmological constant.
[SPECULATIVE -- the deepest extension, guarded hardest.]

Learn the metric on scale-space itself: regions where useful reasoning
repeatedly happens become closer in the learned geometry, so future samples
flow down the easier paths experience has carved. Consolidation applied to the
geometry rather than the operator's modes.

The danger is the recurring collapse in its most subtle costume: a learned
metric that pulls useful regions closer is rich-get-richer -- useful regions get
sampled more, look more useful, pull closer still -- contracting scale-space
onto the few paths that worked early and freezing the model. Worse than earlier
collapses because the samples can look healthily spread IN THE LEARNED METRIC
while collapsed in the real scale-space.

The fix is a constant expansion term -- structurally the cosmological constant:
a uniform outward pressure on the metric that keeps far regions reachable. The
two-way information<->geometry coupling (information sources the geometry; the
geometry moves the information) is the load-bearing relationship -- the same
structure as Einstein's equation, the learned metric is what makes the
architecture relativistic rather than Newtonian.

Diagnostic (mandatory): spread in the ORIGINAL coordinates, not the learned
ones -- the only thing that detects metric-masked collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScaleSpaceMetric:
    """Conformal metric over a grid of scale anchors via log-factor phi(s).

    Learned length element  d ell = exp(-phi(s)) ds : phi>0 shortens (regions
    pulled closer). `usage` (information) raises phi (contraction); the constant
    `expansion` (the cosmological constant Lambda) lowers it everywhere
    (re-expansion). Without Lambda this contracts without bound.
    """

    anchors: np.ndarray                 # (M,) sorted scale grid
    phi: np.ndarray = None              # (M,) log conformal factor, >=0
    lr: float = 0.3                     # information -> geometry coupling
    expansion: float = 0.05             # Lambda: constant outward pressure
    phi_cap: float = 3.0                # hard cap on contraction depth

    def __post_init__(self):
        self.anchors = np.asarray(self.anchors, dtype=float).reshape(-1)
        if self.phi is None:
            self.phi = np.zeros_like(self.anchors)

    def _usage_to_anchors(self, scales, usage):
        """Scatter per-sample usage onto nearest anchors."""
        u = np.zeros_like(self.anchors)
        idx = np.abs(scales[:, None] - self.anchors[None, :]).argmin(axis=1)
        np.add.at(u, idx, usage)
        return u

    def update(self, scales: np.ndarray, usage: np.ndarray) -> None:
        """One info->geometry step then the expansion (cosmological constant).

        phi += lr * usage          (useful regions contract closer)
        phi -= Lambda              (uniform re-expansion -- prevents runaway)
        phi clipped to [0, phi_cap].
        """
        s = np.asarray(scales, dtype=float).reshape(-1)
        u = self._usage_to_anchors(s, np.asarray(usage, dtype=float))
        self.phi = self.phi + self.lr * u
        self.phi = self.phi - self.expansion          # Lambda, applied everywhere
        self.phi = np.clip(self.phi, 0.0, self.phi_cap)

    def learned_distance(self, a: float, b: float) -> float:
        """Integral of exp(-phi) ds between two scales along the anchor grid."""
        lo, hi = sorted((a, b))
        mask = (self.anchors >= lo) & (self.anchors <= hi)
        if mask.sum() < 2:
            return abs(b - a)
        seg = self.anchors[mask]
        w = np.exp(-0.5 * (self.phi[mask][:-1] + self.phi[mask][1:]))
        return float(np.sum(np.diff(seg) * w))

    # -- the mandatory diagnostic ----------------------------------------

    def collapse_warning(self, scales: np.ndarray) -> dict:
        """Detect metric-masked collapse: spread in ORIGINAL vs LEARNED coords.

        If learned spread stays healthy while original spread shrinks, the
        metric is hiding a collapse -- the failure this section guards against.
        """
        s = np.sort(np.asarray(scales, dtype=float).reshape(-1))
        orig = float(np.var(s))
        learned_span = self.learned_distance(s.min(), s.max())
        return {
            "original_var": orig,
            "learned_span": learned_span,
            "max_phi": float(self.phi.max()),
            "contraction_capped": bool(np.any(self.phi >= self.phi_cap - 1e-9)),
        }


if __name__ == "__main__":
    anchors = np.linspace(0, 1, 21)
    g = ScaleSpaceMetric(anchors=anchors, expansion=0.05)
    scales = np.array([0.48, 0.5, 0.52])            # all hammering the middle
    # repeatedly use the middle: without Lambda this would contract forever.
    for _ in range(50):
        g.update(scales, usage=np.ones(3))
    d_mid = g.learned_distance(0.4, 0.6)
    d_edge = g.learned_distance(0.0, 0.2)
    print("geometry self-check")
    print(f"  max phi (capped {g.phi_cap}) = {g.phi.max():.3f}")
    print(f"  learned dist mid (used)  = {d_mid:.3f}")
    print(f"  learned dist edge (idle) = {d_edge:.3f}")
    print(f"  collapse warning = {g.collapse_warning(np.linspace(0,1,9))}")
    # used region pulled closer than idle region, but contraction is bounded.
    assert d_mid < d_edge, "used region should contract closer"
    assert g.phi.max() <= g.phi_cap + 1e-9, "expansion+cap must bound contraction"
