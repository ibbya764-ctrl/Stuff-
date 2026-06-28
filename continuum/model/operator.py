"""The shared dilation operator -- the spine. [IMPLEMENTED structure / PROPOSED science]

Grounded in CP2_AI_Architecture_v3.md Part 3.2 + 3b.1: the dynamics operator is
a diagonal linear SSM x' = A x + B u with eigenvalues

    lambda_k = -w_k + i*nu_k      (w_k = half-width / decay rate, nu_k = freq)

parametrized by its spectrum directly (S4/S5 practice). Two disciplines from
the v3 record are load-bearing and enforced here:

  * Critical-line / -1/2 pin (3b.1): "all resonances decay at the same rate" =
    maximal spectral rigidity. The note's -1/2 is this real part. We pin
    Re(lambda_k) = -1/2 for every mode and NEVER move it -- consolidation may
    add/reshape omega-modes (frequencies), never the real part.
  * GUE / level repulsion on the DYNAMICS operator only (Part 4 table): the
    frequencies nu_k should show level repulsion, <r~> ~ 0.603. Diagnosed via
    spectral_telemetry; never applied to trained weights (two-spectral-targets
    firewall).

D = t d/dt is the generator of the scale continuum: dilating by scale s acts on
mode k by an effective evolution time tau(s). psi(s) (the continuous hypothesis
field) is the orbit of this operator, sampled at representative scales.

Numpy-only; this is the real, runnable structure. The *training* of the
spectrum (and the SFF-ramp validation at scale) needs the GPU substrate and is
out of scope for a CPU session -- flagged where it bites.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spectral_telemetry import spacing_ratio, R_GUE

CRITICAL_LINE = -0.5  # the -1/2 pin: Re(lambda_k) for every mode. Never moved.


@dataclass
class PinnedDilationOperator:
    """Diagonal SSM spine with critical-line-pinned eigenvalues.

    Modes carry a frequency nu_k (free) and a fixed width fixing the real part
    at CRITICAL_LINE. ``consolidated`` marks modes folded in by consolidation
    (the permanent singularities) so resets can keep them (5.10).
    """

    nu: np.ndarray                      # (N,) frequencies, free
    consolidated: np.ndarray = None     # (N,) bool: folded into the spine
    seed: int = 0

    def __post_init__(self):
        self.nu = np.asarray(self.nu, dtype=float).reshape(-1)
        if self.consolidated is None:
            self.consolidated = np.zeros(self.nu.size, dtype=bool)

    # -- spectrum ---------------------------------------------------------

    @property
    def eigenvalues(self) -> np.ndarray:
        """lambda_k = -1/2 + i*nu_k. Real part pinned by construction."""
        return CRITICAL_LINE + 1j * self.nu

    def enforce_critical_line(self, eigs: np.ndarray) -> np.ndarray:
        """Project any candidate eigenvalues back onto the critical line.

        The single non-negotiable: consolidation may move frequencies (imag)
        but the real part is forced to -1/2. Use this anywhere the operator
        could be touched.
        """
        return CRITICAL_LINE + 1j * np.asarray(eigs).imag

    # -- dilation flow: psi(s) is the operator's orbit --------------------

    def dilate(self, amplitudes: np.ndarray, s: float) -> np.ndarray:
        """Evolve modal amplitudes to scale s.

        D = t d/dt => scale s maps to evolution time tau = s (log-time). Mode k
        evolves by exp(lambda_k * tau): amplitude decays as exp(-tau/2) (the
        pinned width) and rotates by nu_k. This is the unitary-up-to-uniform-
        decay flow that makes every mode share one decay rate (rigidity).
        """
        return amplitudes * np.exp(self.eigenvalues * s)

    # -- consolidation hook (5 / 5.10): grow the spine, keep the pin ------

    def consolidate_mode(self, nu_new: float) -> int:
        """Fold a new omega-mode into the spine (a permanent singularity).

        Adds a frequency at Re=-1/2 and marks it consolidated. Returns its
        index. Never touches existing modes' real parts.
        """
        self.nu = np.append(self.nu, float(nu_new))
        self.consolidated = np.append(self.consolidated, True)
        return self.nu.size - 1

    def reset_transient(self) -> None:
        """5.10 gated renewal: drop non-consolidated modes, keep the spine."""
        keep = self.consolidated
        self.nu = self.nu[keep]
        self.consolidated = self.consolidated[keep]

    # -- diagnostics (Part 4) --------------------------------------------

    def level_spacing(self) -> float:
        """<r~> of the frequencies -- the GUE/level-repulsion target on the
        DYNAMICS operator (target ~ 0.603). Weights are never measured this
        way (firewall)."""
        return spacing_ratio(self.nu)

    def gue_gap(self) -> float:
        """|<r~> - R_GUE|: how far the frequency spectrum is from GUE."""
        r = self.level_spacing()
        return float("nan") if np.isnan(r) else abs(r - R_GUE)


def gue_initialized_operator(n_modes: int = 64, seed: int = 0,
                             freq_scale: float = 10.0) -> PinnedDilationOperator:
    """Initialize the spine with GUE-distributed frequencies (level repulsion).

    Frequencies = eigenvalues of a GUE matrix (which have <r~> ~ 0.603 by
    construction), scaled. Gives the operator the mixing/no-mode-collapse
    spectrum the v3 table wants, with the real part pinned at -1/2.
    """
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    nu = np.linalg.eigvalsh((h + h.conj().T) / 2)
    nu = freq_scale * nu / np.std(nu)
    return PinnedDilationOperator(nu=nu, seed=seed)


if __name__ == "__main__":
    op = gue_initialized_operator(64, seed=0)
    print("operator self-check")
    print(f"  modes        = {op.nu.size}")
    print(f"  Re(lambda)   = {set(np.round(op.eigenvalues.real, 6).tolist())} "
          f"(must be {{{CRITICAL_LINE}}})")
    assert np.allclose(op.eigenvalues.real, CRITICAL_LINE)
    print(f"  <r~> freqs   = {op.level_spacing():.3f} (GUE target {R_GUE})")
    idx = op.consolidate_mode(3.14)
    assert op.eigenvalues.real[idx] == CRITICAL_LINE, "pin violated on consolidate"
    print(f"  consolidated mode {idx}: Re still {op.eigenvalues.real[idx]}")
    op.reset_transient()
    print(f"  after reset_transient: {op.nu.size} mode(s) kept (the spine)")
