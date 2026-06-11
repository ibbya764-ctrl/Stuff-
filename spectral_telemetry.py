"""Spectral telemetry for the CP2 AI program (v3 record, Part 4).

Numpy-only diagnostics, designed to run per training checkpoint:

  1. spacing_ratio        -- <r~> level-repulsion statistic (unfolding-free).
                             GUE ~ 0.5996, GOE ~ 0.5359, Poisson ~ 0.3863.
  2. spectral_form_factor -- SFF K(t) of a dynamics operator, with a
                             dip-ramp-plateau readout. The ramp is the
                             operational GUE/chaos fingerprint.
  3. hill_alpha           -- Hill/power-law density exponent of a weight
                             matrix ESD tail (HT-SR / SETOL signal; healthy
                             band alpha in [2, 4], optimum ~ 2). Reports
                             k-range sensitivity instead of a single number
                             (backgrounds are modeled, not assumed flat).
  4. equivariance_error   -- max relative deviation |f(g.x) - g.f(x)| for a
                             layer claiming a symmetry (machine-precision gate).

Two-spectral-targets doctrine (binding): GUE targets apply to the DYNAMICS
operator only; trained WEIGHT matrices are read for heavy-tailed alpha ~ 2
and must not be GUE-regularized.

Run `python spectral_telemetry.py` for the self-test.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "spacing_ratio",
    "spectral_form_factor",
    "ramp_readout",
    "hill_alpha",
    "equivariance_error",
    "R_GUE",
    "R_GOE",
    "R_POISSON",
]

R_GUE = 0.5996
R_GOE = 0.5359
R_POISSON = 2.0 * np.log(2.0) - 1.0  # ~ 0.3863


# ----------------------------------------------------------------------
# 1. Spacing-ratio statistic
# ----------------------------------------------------------------------

def spacing_ratio(eigenvalues: np.ndarray) -> float:
    """Mean adjacent-gap ratio <r~> = <min(r, 1/r)> of a real spectrum.

    Unfolding-free level-repulsion diagnostic (Oganesyan-Huse / ABGR).
    For complex spectra pass e.g. ``eigs.real`` or ``np.abs(eigs)``
    explicitly -- the caller decides which axis carries the physics.
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=float).ravel())
    s = np.diff(lam)
    s = s[s > 0]
    if s.size < 2:
        return float("nan")
    r = s[1:] / s[:-1]
    return float(np.mean(np.minimum(r, 1.0 / r)))


# ----------------------------------------------------------------------
# 2. Spectral form factor and ramp readout
# ----------------------------------------------------------------------

def spectral_form_factor(
    eigenvalue_sets: np.ndarray | list,
    times: np.ndarray | None = None,
    connected: bool = True,
):
    """SFF K(t) = <|sum_j exp(i t lam_j)|^2> / N over an ensemble.

    ``eigenvalue_sets``: (n_samples, N) array (or list of 1-D spectra; a
    single spectrum is accepted but the readout will be noisy -- prefer
    checkpoints-as-ensemble or block-resampled operators).

    If ``connected``, the disconnected part |<Z(t)>|^2/N is subtracted:
    this is the "model the smooth background, don't assume it flat" rule.
    Returns (times, K).
    """
    sets = np.atleast_2d(np.asarray(eigenvalue_sets, dtype=float))
    n_samp, n_dim = sets.shape
    # Unfold: map each eigenvalue to its normalized rank in the pooled
    # ensemble density. This removes the smooth density profile (the
    # "model the background" rule), leaving uniform density with mean
    # spacing 1, so the GUE ramp K ~ t/(2*pi) ends at Heisenberg time 2*pi.
    pooled = np.sort(sets.ravel())
    sets = np.searchsorted(pooled, sets).astype(float) / n_samp
    if times is None:
        times = np.logspace(-2, np.log10(8 * np.pi), 200)
    Z = np.exp(1j * times[None, :, None] * sets[:, None, :]).sum(axis=2)
    K = np.mean(np.abs(Z) ** 2, axis=0) / n_dim
    if connected and n_samp > 1:
        K = K - np.abs(Z.mean(axis=0)) ** 2 / n_dim
        K = np.maximum(K, 1e-12)
    return times, K


def ramp_readout(times: np.ndarray, K: np.ndarray) -> dict:
    """Fit the ramp region of an SFF curve.

    Heuristic: plateau level is the late-time median; the ramp window is
    where K rises from 10% to 80% of plateau after the dip. Returns the
    log-log slope there (GUE linear ramp => slope ~ 1) and a boolean
    ``ramp_present`` (slope in [0.6, 1.4] with decent correlation).
    """
    K = np.asarray(K, float)
    t = np.asarray(times, float)
    plateau = float(np.median(K[int(0.9 * len(K)):]))
    dip_idx = int(np.argmin(K[: int(0.9 * len(K))]))
    sel = (np.arange(len(K)) > dip_idx) & (K > 0.10 * plateau) & (K < 0.80 * plateau)
    out = {"plateau": plateau, "slope": float("nan"),
           "r2": float("nan"), "ramp_present": False}
    if sel.sum() >= 5:
        x, y = np.log(t[sel]), np.log(K[sel])
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
        out.update(slope=float(slope), r2=r2,
                   ramp_present=bool(0.6 <= slope <= 1.4 and r2 > 0.8))
    return out


# ----------------------------------------------------------------------
# 3. Hill / power-law alpha for weight ESDs
# ----------------------------------------------------------------------

def hill_alpha(weight_or_eigs: np.ndarray, k_fracs=(0.05, 0.10, 0.20)) -> dict:
    """Power-law DENSITY exponent alpha of an ESD tail, p(lam) ~ lam^-alpha.

    Accepts a weight matrix W (uses eigenvalues of W W^T / n, the HT-SR
    convention) or a 1-D array of eigenvalues. Hill tail index gamma on the
    top-k order statistics gives alpha = 1 + 1/gamma_hat ... reported as
    alpha = 1 + k / sum log(lam_i / lam_(k+1)).

    Reports alpha at several tail fractions plus the spread, because a
    single-k Hill fit is exactly the kind of unmodeled background the
    program has been burned by. Healthy band: alpha in [2, 4], optimum ~ 2.
    """
    a = np.asarray(weight_or_eigs, dtype=float)
    if a.ndim == 2:
        n = max(a.shape)
        m = a @ a.T if a.shape[0] <= a.shape[1] else a.T @ a
        eigs = np.linalg.eigvalsh(m / n)
    else:
        eigs = a
    eigs = np.sort(eigs[eigs > 0])[::-1]
    alphas = {}
    for f in k_fracs:
        k = max(int(f * eigs.size), 5)
        if k >= eigs.size:
            continue
        tail = eigs[:k]
        gamma = np.mean(np.log(tail / eigs[k]))
        alphas[f] = float(1.0 + 1.0 / gamma)
    vals = np.array(list(alphas.values()))
    return {
        "alpha": float(np.median(vals)) if vals.size else float("nan"),
        "alpha_by_kfrac": alphas,
        "spread": float(vals.max() - vals.min()) if vals.size else float("nan"),
        "in_healthy_band": bool(vals.size and 2.0 <= np.median(vals) <= 4.0),
    }


# ----------------------------------------------------------------------
# 4. Equivariance gate
# ----------------------------------------------------------------------

def equivariance_error(layer, group_action_in, group_action_out,
                       inputs: np.ndarray) -> float:
    """Max relative error  |f(g.x) - g.f(x)| / |f(x)|  over a batch.

    ``layer`` maps arrays to arrays; the two actions apply a group element
    on input/output space. The v2 II.4 gate: this must sit at machine
    precision through depth, or the symmetry claim is removed.
    """
    worst = 0.0
    for x in inputs:
        fx = layer(x)
        lhs = layer(group_action_in(x))
        rhs = group_action_out(fx)
        denom = float(np.linalg.norm(fx)) or 1.0
        worst = max(worst, float(np.linalg.norm(lhs - rhs)) / denom)
    return worst


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def _selftest(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n, n_samp = 200, 64

    # --- GUE ensemble: spacing ratio + SFF ramp -----------------------
    gue = []
    for _ in range(n_samp):
        h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        gue.append(np.linalg.eigvalsh((h + h.conj().T) / 2))
    gue = np.array(gue)
    r_gue = np.mean([spacing_ratio(e) for e in gue])
    assert abs(r_gue - R_GUE) < 0.02, f"GUE <r~> off: {r_gue:.4f}"

    t, K = spectral_form_factor(gue)
    ramp = ramp_readout(t, K)
    assert ramp["ramp_present"], f"GUE ramp not detected: {ramp}"

    # --- Poisson control: no repulsion, no GUE ramp claim --------------
    poi = np.cumsum(rng.exponential(size=(n_samp, n)), axis=1)
    r_poi = np.mean([spacing_ratio(e) for e in poi])
    assert abs(r_poi - R_POISSON) < 0.02, f"Poisson <r~> off: {r_poi:.4f}"

    # --- Heavy-tailed ESD: Hill alpha recovery -------------------------
    alpha_true = 2.5
    lam = (1.0 - rng.uniform(size=20000)) ** (-1.0 / (alpha_true - 1.0))
    est = hill_alpha(lam)
    assert abs(est["alpha"] - alpha_true) < 0.15, f"Hill alpha off: {est}"

    # --- Equivariance gate: rotation-commuting linear layer ------------
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    layer = lambda x: 3.0 * x                      # commutes with SO(2)
    err = equivariance_error(layer, lambda x: rot @ x, lambda y: rot @ y,
                             rng.normal(size=(8, 2)))
    assert err < 1e-12, f"equivariance gate broken: {err:.2e}"

    print("spectral_telemetry self-test PASSED")
    print(f"  <r~> GUE     = {r_gue:.4f}   (target {R_GUE})")
    print(f"  <r~> Poisson = {r_poi:.4f}   (target {R_POISSON:.4f})")
    print(f"  SFF ramp     : slope {ramp['slope']:.3f}, r2 {ramp['r2']:.3f}, "
          f"plateau {ramp['plateau']:.3f}")
    print(f"  Hill alpha   = {est['alpha']:.3f} (true {alpha_true}), "
          f"spread {est['spread']:.3f}")
    print(f"  equivariance = {err:.2e}")


if __name__ == "__main__":
    _selftest()
