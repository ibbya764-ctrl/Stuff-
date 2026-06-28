"""5.6 force-law A/B harness -- the pre-registered spacing experiment.

Runs three arms (GUE / generic-RBF / no-repulsion) at matched confinement,
noise and sample count over several seeds, measures the level-spacing <r~> of
the post-burn-in sample positions (via spectral_telemetry), and reports
SUPPORT / FALSIFY against the bands committed in PREREGISTRATION.md.

This tests the SPACING half of the note's 5.6 falsification block only; the
downstream-task-performance half needs the built model and is out of scope
here (logged below, not silently dropped).

    python3 -m continuum.eval_force_law           # default multi-seed run
    python3 -m continuum.eval_force_law --quick   # fewer seeds/steps (smoke)
"""

from __future__ import annotations

import argparse

import numpy as np

from spectral_telemetry import R_GUE, R_POISSON, spacing_ratio

from .scale_dynamics import GaussianMixtureLandscape, ScaleSampleDynamics

# Pre-registered bands (see PREREGISTRATION.md -- committed before running).
GUE_BAND = (0.56, 0.63)
POISSON_BAND = (0.36, 0.42)

ARMS = {
    "gue": dict(repulsion="gue", beta=2.0),
    "rbf": dict(repulsion="rbf", beta=2.0),
    "none": dict(repulsion="none", beta=0.0),
}


def run_arm(name: str, *, n_samples, steps, burn_in, seeds, dt) -> dict:
    """Run one arm across seeds; return mean <r~>, Var(s), collapse flag."""
    cfg = ARMS[name]
    land = GaussianMixtureLandscape(
        centers=np.array([0.0]), weights=np.array([1.0]), widths=np.array([1.0]))
    r_vals, var_vals, collapsed = [], [], []
    for seed in range(seeds):
        dyn = ScaleSampleDynamics(
            landscape=land, repulsion=cfg["repulsion"],
            alpha=2.0, beta=cfg["beta"], temperature=1.0, dt=dt)
        out = dyn.run(n_samples=n_samples, steps=steps, burn_in=burn_in,
                      seed=seed, record_every=50)
        # <r~> per snapshot, averaged (each snapshot is one "spectrum").
        ens = out["ensemble"]
        if ens.size:
            r_vals.append(np.mean([spacing_ratio(snap) for snap in ens]))
        var_vals.append(out["diagnostic"]["var"])
        collapsed.append(out["diagnostic"]["collapsed"])
    return {
        "r_mean": float(np.mean(r_vals)) if r_vals else float("nan"),
        "r_std": float(np.std(r_vals)) if r_vals else float("nan"),
        "var_mean": float(np.mean(var_vals)),
        "collapsed_any": bool(any(collapsed)),
    }


def _in(x, band) -> bool:
    return band[0] <= x <= band[1]


def evaluate(results: dict) -> dict:
    """Apply the pre-registered prediction + kill condition."""
    gue_r = results["gue"]["r_mean"]
    rbf_r = results["rbf"]["r_mean"]
    none_r = results["none"]["r_mean"]

    gue_is_gue = _in(gue_r, GUE_BAND)
    rbf_is_gue = _in(rbf_r, GUE_BAND)
    none_is_poisson = _in(none_r, POISSON_BAND)

    # Kill conditions from PREREGISTRATION.md.
    if not gue_is_gue:
        verdict = "FALSIFY-GUE-CLAIM"
        reason = (f"GUE arm <r~>={gue_r:.3f} missed the GUE band {GUE_BAND}: "
                  f"the log-gas force law did not produce GUE spacing even in "
                  f"the clean toy. Generic repulsion is the right default.")
    elif rbf_is_gue:
        verdict = "FALSIFY-GUE-CLAIM"
        reason = (f"Generic RBF arm <r~>={rbf_r:.3f} also landed in the GUE "
                  f"band {GUE_BAND}: the GUE-targeted choice earns nothing over "
                  f"the baseline (note: 'matches or underperforms the generic "
                  f"kernel').")
    else:
        verdict = "SUPPORT"
        reason = (f"GUE arm <r~>={gue_r:.3f} in GUE band {GUE_BAND}; generic "
                  f"RBF arm <r~>={rbf_r:.3f} distinguishable (outside it). The "
                  f"GUE-targeted force law produces GUE spacing the generic "
                  f"kernel does not.")
    return {
        "verdict": verdict,
        "reason": reason,
        "gue_in_band": gue_is_gue,
        "rbf_in_gue_band": rbf_is_gue,
        "none_is_poisson": none_is_poisson,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: fewer seeds/steps")
    args = ap.parse_args()

    if args.quick:
        kw = dict(n_samples=40, steps=4000, burn_in=1500, seeds=2, dt=1e-3)
    else:
        kw = dict(n_samples=48, steps=10000, burn_in=4000, seeds=4, dt=1e-3)

    print("=" * 68)
    print("5.6 force-law spacing experiment  (pre-registered; see "
          "PREREGISTRATION.md)")
    print(f"  matched: alpha=2, sigma=1, T=1, N={kw['n_samples']}, "
          f"seeds={kw['seeds']}, steps={kw['steps']}")
    print(f"  targets: R_GUE={R_GUE}  R_POISSON={R_POISSON:.4f}")
    print(f"  bands  : GUE {GUE_BAND}   Poisson {POISSON_BAND}")
    print("=" * 68)

    results = {name: run_arm(name, **kw) for name in ARMS}
    for name, r in results.items():
        print(f"  {name:<5} <r~> = {r['r_mean']:.3f} +/- {r['r_std']:.3f}   "
              f"Var(s) = {r['var_mean']:7.3f}   "
              f"collapsed={r['collapsed_any']}")

    verdict = evaluate(results)
    print("-" * 68)
    print(f"  none ~ Poisson? {verdict['none_is_poisson']}   "
          f"(collapse control reads Poisson local statistics)")
    print(f"  VERDICT: {verdict['verdict']}")
    print(f"  {verdict['reason']}")
    print("-" * 68)
    print("  COVERAGE LIMIT: spacing half of the 5.6 block only. The "
          "downstream-")
    print("  task-performance half needs the built model (GPU/Groq/Scaffold) "
          "and")
    print("  is NOT tested here. Green here is necessary, not sufficient.")
    print("=" * 68)


if __name__ == "__main__":
    main()
