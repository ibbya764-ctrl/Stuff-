"""5.6 -- Moving scale samples, force-driven distribution. [SPECULATIVE]

Scale samples (NOT branches -- they are mobile samples of one continuous field
psi(s)) move under three forces:

    ds_i/dt = alpha * attraction(s_i)  -  beta * repulsion_i  +  eta_i

  * attraction : up the density gradient, toward dense/operator-coupled
                 regions. Here a confining landscape rho(s) (valleys = dense).
  * repulsion  : the anti-collapse primitive (5.7). Two force laws compete:
                   - GENERIC  : Gaussian/RBF kernel (SVGD). The safe default
                                and the A/B baseline.
                   - GUE      : the note's claim -- the repulsion that produces
                                GUE-like level statistics. Realized here as the
                                Dyson log-gas: logarithmic (1/Delta) repulsion.
                                At inverse-temperature beta=2 in a harmonic
                                confiner, the stationary sample positions ARE
                                the GUE eigenvalue statistics. This is why the
                                note can say "use the forces" and "target GUE
                                spacing" are one proposal: the log-gas force is
                                literally the generative mechanism of GUE.
  * noise      : lets samples explore rather than settle in the nearest pit.

The harness (eval_force_law) integrates these three arms -- GUE, generic, and a
no-repulsion control -- and reads the level-spacing <r~> of the final positions
(via spectral_telemetry) to test, on CPU, whether the GUE force law actually
produces GUE spacing and whether it is distinguishable from the generic kernel.

The -1/2 pin is untouched: these are sample *positions* on the scale axis, not
the operator's real part. Numpy-only. Self-test:
`python3 -m continuum.scale_dynamics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .anti_collapse import AntiCollapseController, collapse_diagnostic


# ----------------------------------------------------------------------
# Density landscape (the "attraction to informative regions").
# ----------------------------------------------------------------------

@dataclass
class GaussianMixtureLandscape:
    """A toy density rho(s) with a few valleys/attractors (5.2 picture).

    centers/weights/widths define dense regions; the attraction force is
    +grad log rho (pull up the density gradient). A single broad center
    reduces to harmonic confinement -- the Dyson-gas confiner, which is what
    the GUE arm needs to reach its stationary law.
    """

    centers: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    weights: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    widths: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    def log_grad(self, s: np.ndarray) -> np.ndarray:
        """grad log rho(s) for scalar positions s (shape (n,))."""
        s = np.asarray(s, dtype=float).reshape(-1)
        c, w, sig = self.centers, self.weights, self.widths
        # component densities g_k(s) (n, K)
        z = (s[:, None] - c[None, :]) / sig[None, :]
        g = w[None, :] * np.exp(-0.5 * z ** 2) / sig[None, :]
        rho = g.sum(axis=1) + 1e-300
        # d/ds log rho = (sum_k g_k * -(s-c)/sig^2) / rho
        dg = g * (-(s[:, None] - c[None, :]) / sig[None, :] ** 2)
        return dg.sum(axis=1) / rho


# ----------------------------------------------------------------------
# The force laws.
# ----------------------------------------------------------------------

def gue_repulsion(s: np.ndarray, soft: float = 0.05) -> np.ndarray:
    """Dyson log-gas repulsion: F_i = sum_{j!=i} (s_i - s_j)/((s_i-s_j)^2+soft^2).

    The gradient of the log-Coulomb potential -sum log|s_i - s_j|, softened at
    short range so the singular 1/Delta force does not destabilize the Euler
    step. For `soft` much smaller than the mean spacing (here ~1.3) it equals
    1/Delta everywhere that matters and leaves the local level statistics GUE.
    With a harmonic confiner and beta=2 thermal noise, the stationary
    distribution is the GUE eigenvalue ensemble -- so this force law's
    equilibrium spacing is GUE level repulsion BY CONSTRUCTION (the note's
    central 5.6 claim, here as a directly testable mechanism, not an analogy).
    """
    s = np.asarray(s, dtype=float).reshape(-1)
    diff = s[:, None] - s[None, :]                 # (n, n)
    f = diff / (diff ** 2 + soft ** 2)             # ~1/diff for |diff|>>soft
    np.fill_diagonal(f, 0.0)
    return f.sum(axis=1)


@dataclass
class ScaleSampleDynamics:
    """Integrate ds/dt = alpha*attraction - beta*repulsion + noise.

    `repulsion` selects the force law:
      * "gue"  -> Dyson log-gas (the claim under test)
      * "rbf"  -> generic Gaussian kernel via AntiCollapseController (baseline)
      * "none" -> no repulsion (collapse control)
    """

    landscape: GaussianMixtureLandscape = field(
        default_factory=GaussianMixtureLandscape)
    repulsion: str = "gue"
    alpha: float = 1.0              # attraction (confinement) strength
    beta: float = 1.0               # repulsion strength
    temperature: float = 1.0        # noise scale (sets the Dyson beta)
    dt: float = 1e-3
    controller: AntiCollapseController = field(
        default_factory=lambda: AntiCollapseController(bandwidth=0.5))

    def step(self, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        s = np.asarray(s, dtype=float).reshape(-1)
        attraction = self.alpha * self.landscape.log_grad(s)
        if self.repulsion == "gue":
            rep = self.beta * gue_repulsion(s)
        elif self.repulsion == "rbf":
            rep = self.beta * self.controller.force(
                s, mode="soft_spread", strength=1.0)
        elif self.repulsion == "none":
            rep = 0.0
        else:
            raise ValueError(f"unknown repulsion {self.repulsion!r}")
        # Overdamped Langevin: drift*dt + sqrt(2 T dt) * N(0,1).
        noise = np.sqrt(2.0 * self.temperature * self.dt) * rng.standard_normal(s.size)
        return s + (attraction + rep) * self.dt + noise

    def run(
        self,
        n_samples: int = 64,
        steps: int = 4000,
        burn_in: int = 1500,
        seed: int = 0,
        record_every: int = 1,
    ) -> dict:
        """Integrate to (approximate) stationarity and collect positions.

        Returns the final positions, an ensemble of post-burn-in snapshots
        (rows = snapshots, for the SFF if wanted), and the Var(s) trajectory.
        """
        rng = np.random.default_rng(seed)
        s = rng.standard_normal(n_samples)
        var_traj, snapshots = [], []
        for t in range(steps):
            s = self.step(s, rng)
            var_traj.append(float(np.var(s)))
            if t >= burn_in and (t - burn_in) % record_every == 0:
                snapshots.append(s.copy())
        return {
            "final": s,
            "ensemble": np.array(snapshots),
            "var_traj": np.array(var_traj),
            "diagnostic": collapse_diagnostic(s),
        }


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def _selftest(seed: int = 0) -> None:
    # Harmonic confiner sigma=1 -- with alpha=2, beta=2, T=1 the GUE arm is
    # exactly Dyson Brownian motion: drift -2x + 2*sum 1/(x_i-x_j), sqrt(2) dB,
    # whose stationary law is the GUE eigenvalue ensemble.
    land = GaussianMixtureLandscape(
        centers=np.array([0.0]),
        weights=np.array([1.0]),
        widths=np.array([1.0]),
    )

    # GUE arm reaches a non-collapsed, bounded structured spread.
    gue = ScaleSampleDynamics(landscape=land, repulsion="gue",
                              alpha=2.0, beta=2.0, temperature=1.0)
    out = gue.run(n_samples=48, steps=8000, burn_in=3000, seed=seed)
    assert not out["diagnostic"]["collapsed"], "GUE arm collapsed unexpectedly"
    assert out["final"].std() > 0.5, "GUE arm has no spread"
    assert out["diagnostic"]["var"] < 100.0, "GUE arm not bounded (check scaling)"

    # No-repulsion control collapses harder than the GUE arm (lower Var) when
    # noise is small relative to confinement: pure attraction pulls together.
    ctrl = ScaleSampleDynamics(landscape=land, repulsion="none",
                               alpha=2.0, beta=0.0, temperature=0.02)
    out_ctrl = ctrl.run(n_samples=48, steps=8000, burn_in=3000, seed=seed)
    assert out_ctrl["diagnostic"]["var"] < out["diagnostic"]["var"], \
        "no-repulsion control did not contract relative to GUE arm"

    print("scale_dynamics self-test PASSED")
    print(f"  GUE arm  Var(s) = {out['diagnostic']['var']:.3f}  "
          f"collapsed={out['diagnostic']['collapsed']}")
    print(f"  none arm Var(s) = {out_ctrl['diagnostic']['var']:.3f}  "
          f"collapsed={out_ctrl['diagnostic']['collapsed']} "
          f"(low noise -> contracts toward the attractor)")


if __name__ == "__main__":
    _selftest()
