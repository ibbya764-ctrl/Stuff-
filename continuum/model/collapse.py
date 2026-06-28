"""Collapse assembles from many regions -- not one.

Because the samples form a distributed orbital structure and the collapse
coefficient can be local, the final answer is assembled from many locally-useful
regions rather than selected from one:

    Psi(x) = integral c(s, x) psi(s, x) ds

where c(s, x) is LOCAL -- different parts x of the representation take
contributions from different scale regions. If region A has the best structure
for one part and region B for another, the collapse includes both. This is the
precise answer to "it does not have to collapse into one singular branch": one
coherent answer assembled from many locally-useful regions of the field.

The local coefficient is driven by importance (density x curvature, 5.9) and by
each sample's per-feature fit, so a sample only contributes where it is locally
best.
"""

from __future__ import annotations

import numpy as np


def assemble(psi: np.ndarray, importance: np.ndarray,
             local_fit: np.ndarray | None = None) -> np.ndarray:
    """Psi(x) = sum_s c(s, x) psi(s, x), c local and normalized per feature x.

    psi          : (n_scales, d) field features per sample.
    importance   : (n_scales,) global sampling importance (5.9).
    local_fit    : (n_scales, d) optional per-feature suitability; if None,
                   coefficients are global (importance broadcast over features).

    c(s, x) propto importance(s) * local_fit(s, x), normalized over s for each
    feature x so every part of the answer is a convex combination of samples.
    """
    psi = np.asarray(psi)
    imp = np.asarray(importance, dtype=float).reshape(-1, 1)
    if local_fit is None:
        c = np.broadcast_to(imp, psi.shape).copy()
    else:
        c = imp * np.asarray(local_fit, dtype=float)
    c = c / (c.sum(axis=0, keepdims=True) + 1e-12)    # normalize over scales
    return np.sum(c * psi, axis=0), c


if __name__ == "__main__":
    # two samples, each best on a different half of the features.
    d = 6
    psiA = np.concatenate([np.ones(d // 2), np.zeros(d - d // 2)])
    psiB = np.concatenate([np.zeros(d // 2), 2 * np.ones(d - d // 2)])
    psi = np.stack([psiA, psiB])                      # (2, d)
    importance = np.array([0.5, 0.5])
    local_fit = np.stack([                            # A fits left, B fits right
        np.concatenate([np.ones(d // 2), 0.01 * np.ones(d - d // 2)]),
        np.concatenate([0.01 * np.ones(d // 2), np.ones(d - d // 2)]),
    ])
    out, c = assemble(psi, importance, local_fit)
    print("collapse self-check")
    print(f"  assembled Psi = {np.round(out, 3).tolist()}")
    print(f"  coeff sample A = {np.round(c[0], 2).tolist()}")
    print(f"  coeff sample B = {np.round(c[1], 2).tolist()}")
    # left features come from A (~1), right from B (~2): one answer, two regions.
    assert out[0] > 0.9 and out[-1] > 1.9, "assembly should take best-of-each"
