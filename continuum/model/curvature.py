"""5.9 Information curvature -- where computation should go. [SPECULATIVE]

importance = density x curvature: the corner-detector that closes the loop with
5.8 (information curves the geometry; the geometry's curvature drives the
sampling). Density says where the field has mass; curvature says where it is
*bending* -- a flat dense plateau needs few samples, a sharply curving ridge
needs many. Their product marks the informative corners (heading toward
decision curvature).

This is A/B'd against density-alone in the build order (note: "density x
curvature ... A/B'd against density alone as an early, cheap, high-value test").
This module supplies both signals so that A/B is a one-line change.
"""

from __future__ import annotations

import numpy as np


def field_curvature(scales: np.ndarray, values: np.ndarray) -> np.ndarray:
    """|second derivative| of the field along scale -- how sharply it bends.

    `values` is a scalar-per-sample summary of psi(s) (e.g. feature norm).
    Curvature is estimated by a local finite-difference on the (sorted) scale
    axis; endpoints reuse their neighbour's value.
    """
    s = np.asarray(scales, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(s)
    s, v = s[order], v[order]
    curv = np.zeros_like(v)
    for i in range(1, len(v) - 1):
        h1, h2 = s[i] - s[i - 1], s[i + 1] - s[i]
        if h1 > 0 and h2 > 0:
            curv[i] = abs((v[i + 1] - v[i]) / h2 - (v[i] - v[i - 1]) / h1) / (0.5 * (h1 + h2))
    if len(v) > 2:
        curv[0], curv[-1] = curv[1], curv[-2]
    # undo the sort
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return curv[inv]


def importance(density: np.ndarray, curvature: np.ndarray,
               use_curvature: bool = True) -> np.ndarray:
    """Sampling importance per sample.

    use_curvature=True : importance = density * (eps + curvature)  (5.9)
    use_curvature=False: importance = density                      (A/B baseline)
    Normalized to sum 1 (a sampling distribution over scale-space).
    """
    rho = np.asarray(density, dtype=float)
    if use_curvature:
        imp = rho * (1e-6 + np.asarray(curvature, dtype=float))
    else:
        imp = rho.copy()
    tot = imp.sum()
    return imp / tot if tot > 0 else np.full_like(imp, 1.0 / imp.size)


if __name__ == "__main__":
    scales = np.linspace(0, 1, 9)
    # a field that is flat then sharply kinked: curvature should spike at kink.
    vals = np.concatenate([np.zeros(4), [0.0, 1.0], np.full(3, 1.0)])
    curv = field_curvature(scales, vals)
    dens = np.ones_like(scales)
    imp_c = importance(dens, curv, use_curvature=True)
    imp_d = importance(dens, curv, use_curvature=False)
    print("curvature self-check")
    print(f"  curvature   = {np.round(curv, 2).tolist()}")
    print(f"  imp (dxc)   = {np.round(imp_c, 3).tolist()} (mass at the kink)")
    print(f"  imp (d only)= {np.round(imp_d, 3).tolist()} (uniform)")
    assert np.argmax(imp_c) in (4, 5), "importance should peak at the kink"
    assert np.allclose(imp_d, imp_d[0]), "density-only baseline should be flat"
