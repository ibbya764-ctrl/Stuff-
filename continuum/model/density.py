"""2. The density field rho(s, x) shapes the scale-landscape. [SPECULATIVE,
building on the field's [ENGINEERING] substrate]

rho turns the bare continuum psi(s) into a landscape rather than a flat family:
high-density regions become valleys/attractors, low-density flat. It decides
compute allocation (dense regions stay liquid and keep evolving; sparse ones
solidify) and sets the sampling rate (5.6).

Built here from two readable signals, both computable from what the model has:
  * agreement: scale samples whose operator-mode occupations overlap are in a
    dense, mutually-supporting region (the thermofield-double regime).
  * occupancy mass: total modal amplitude present at a scale.

This is the scalar foundation-stone density promoted to live over scale-space.
The attraction force the samples feel (scale_dynamics) is +grad log rho.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DensityField:
    """rho(s) over scale-space, estimated from sample occupations + positions.

    Kernel-smoothed agreement density: each pair of samples contributes mass
    where they sit, weighted by operator-mode overlap (agreement). Dense =
    many agreeing samples nearby.
    """

    bandwidth: float = 0.3

    def density(self, scales: np.ndarray, occupations: np.ndarray) -> np.ndarray:
        """rho at each sample's scale. Shape (n_scales,).

        rho(s_i) = sum_j overlap(i, j) * K(s_i - s_j), K Gaussian. Overlap is
        cosine of mode-occupation vectors (the operator-anchored agreement).
        """
        s = np.asarray(scales, dtype=float).reshape(-1)
        occ = np.asarray(occupations, dtype=float)
        norm = np.linalg.norm(occ, axis=1, keepdims=True) + 1e-12
        agree = (occ / norm) @ (occ / norm).T            # (n, n) in [0, 1]
        d2 = (s[:, None] - s[None, :]) ** 2
        K = np.exp(-d2 / (2.0 * self.bandwidth ** 2))
        return (agree * K).sum(axis=1)

    def log_grad(self, scales: np.ndarray, occupations: np.ndarray) -> np.ndarray:
        """d/ds_i log rho(s_i) -- the attraction direction each sample climbs.

        Analytic gradient perturbing ONLY s_i (other samples held fixed): for
        rho(s_i) = sum_j agree(i,j) K(s_i - s_j),
            d rho_i/d s_i = sum_j agree(i,j) * (-(s_i - s_j)/bw^2) * K(s_i - s_j).
        (A finite difference that shifts the whole population leaves every
        pairwise gap unchanged and gives an identically-zero gradient -- the
        per-coordinate analytic form is the correct attraction.)
        """
        s = np.asarray(scales, dtype=float).reshape(-1)
        occ = np.asarray(occupations, dtype=float)
        norm = np.linalg.norm(occ, axis=1, keepdims=True) + 1e-12
        agree = (occ / norm) @ (occ / norm).T
        diff = s[:, None] - s[None, :]
        K = np.exp(-diff ** 2 / (2.0 * self.bandwidth ** 2))
        base = (agree * K).sum(axis=1)
        drho = (agree * (-diff / self.bandwidth ** 2) * K).sum(axis=1)
        return drho / (base + 1e-12)

    def liquidity(self, scales: np.ndarray, occupations: np.ndarray) -> np.ndarray:
        """Per-sample 'still liquid?' signal in [0, 1].

        5.2 / 5: dense (uncertain or rich) regions stay liquid and keep
        evolving; sparse regions solidify early. Here liquidity rises with
        local density (rich terrain deserves more computation) -- normalized
        to [0, 1] across the population.
        """
        rho = self.density(scales, occupations)
        lo, hi = rho.min(), rho.max()
        return (rho - lo) / (hi - lo + 1e-12)


if __name__ == "__main__":
    from .operator import gue_initialized_operator
    from .field import ContinuousHypothesisField
    op = gue_initialized_operator(32, seed=2)
    fld = ContinuousHypothesisField.from_input(op, np.arange(8.0))
    scales = np.linspace(0.0, 1.0, 8)
    occ = fld.mode_occupations(scales)
    df = DensityField(bandwidth=0.3)
    rho = df.density(scales, occ)
    print("density self-check")
    print(f"  rho(s)      = {np.round(rho, 3).tolist()}")
    print(f"  log-grad    = {np.round(df.log_grad(scales, occ), 3).tolist()}")
    print(f"  liquidity   = {np.round(df.liquidity(scales, occ), 3).tolist()}")
    assert rho.shape == scales.shape
