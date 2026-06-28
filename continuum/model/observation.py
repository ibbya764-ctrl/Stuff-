"""3. Observation without decoherence -- the load-bearing new idea.
[SPECULATIVE -- strong claim, needs its own experiment. SUBSTRATE-GATED.]

The tension this resolves: partial/local collapse needs observation, but
observation is decoherence, and the one empirical result says decoherence kills
the architecture. The note's claim: the operator re-coupling in a dense region
is UNITARY (a thermofield-double entangling op), not a projection -- so it makes
the agreement-structure of nearby scales available WITHOUT projecting, and
entanglement is coherence-preserving. "Measurement information without
measurement decoherence."

This is "the kind of too-good-to-be-true that must be checked before anything is
built on it." The gating experiment (note's falsification block): does an
entanglement-mediated local readout preserve the coherence telemetry while still
letting regions specialize, versus a projective local readout that should
collapse it? That needs the BUILT complex-amplitude model and is NOT runnable
here.

What this module IS: a faithful CPU demonstration that the two readout *types*
have the claimed signatures on a toy multi-sample state -- unitary coupling
preserves coherence/purity, projection destroys them. This shows the mechanism
is coherent as stated; it does NOT show the claim holds in the real
architecture. The distinction (entangle-and-read vs measure-against-outside-
basis) is real and demonstrable; whether it gives "the collapse signal without
the collapse cost" for actual reasoning is the experiment that gates this whole
branch. Do not let a green here be read as the claim confirmed.
"""

from __future__ import annotations

import numpy as np


def l1_coherence(rho: np.ndarray) -> float:
    """l1 measure of coherence: sum of off-diagonal magnitudes of rho.

    This is the 'interference/coherence telemetry' that carried the only
    positive signal so far. Zero iff rho is diagonal (fully decohered).
    """
    off = rho - np.diag(np.diag(rho))
    return float(np.sum(np.abs(off)))


def purity(rho: np.ndarray) -> float:
    """Tr(rho^2): 1 for a pure (coherent) state, <1 once mixed by measurement."""
    return float(np.real(np.trace(rho @ rho)))


def coherent_state(coeffs: np.ndarray) -> np.ndarray:
    """Density matrix of the coherent superposition Psi = sum_b c_b |b>.

    The branches in a dense region as one pure, coherent superposition.
    """
    c = np.asarray(coeffs, dtype=complex).reshape(-1)
    c = c / (np.linalg.norm(c) + 1e-12)
    return np.outer(c, c.conj())


def projective_readout(rho: np.ndarray) -> np.ndarray:
    """Measure the branches against an outside basis: dephase to the diagonal.

    This is the magnitude-collapse that strangled the additive task -- you
    square, pick, discard phase. Off-diagonals (coherence) -> 0; state -> mixed.
    """
    return np.diag(np.diag(rho)).astype(complex)


def thermofield_coupling(n: int, theta: float = 0.6) -> np.ndarray:
    """A unitary that entangles nearby samples through a shared operator mode.

    Toy thermofield-double / entangling op: nearest-neighbour 'beamsplitter'
    rotations coupling sample b with b+1 (the operator entering the state). It
    is unitary by construction, so it cannot decohere -- it rotates the basis
    in which agreement is read, making correlations available without
    projection.
    """
    U = np.eye(n, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    for b in range(n - 1):
        g = np.eye(n, dtype=complex)
        g[b, b], g[b + 1, b + 1] = c, c
        g[b, b + 1], g[b + 1, b] = -1j * s, -1j * s
        U = g @ U
    return U


def unitary_observation(rho: np.ndarray, theta: float = 0.6) -> np.ndarray:
    """Coherence-preserving observation: rho -> U rho U^dagger (U entangling).

    The agreement information becomes available through the entanglement; purity
    is preserved (unitary), so the coherence telemetry survives.
    """
    U = thermofield_coupling(rho.shape[0], theta)
    return U @ rho @ U.conj().T


def demo_distinction(n: int = 5, seed: int = 0) -> dict:
    """Show the two readouts' signatures on one coherent multi-sample state."""
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=n) + 1j * rng.normal(size=n)
    rho = coherent_state(coeffs)
    proj = projective_readout(rho)
    unit = unitary_observation(rho)
    return {
        "start":     {"coherence": l1_coherence(rho),  "purity": purity(rho)},
        "projective": {"coherence": l1_coherence(proj), "purity": purity(proj)},
        "unitary":    {"coherence": l1_coherence(unit), "purity": purity(unit)},
    }


if __name__ == "__main__":
    d = demo_distinction()
    print("observation self-check  [SUBSTRATE-GATED demo of the mechanism, "
          "NOT the claim]")
    for k, v in d.items():
        print(f"  {k:<11} coherence={v['coherence']:.3f}  purity={v['purity']:.3f}")
    # projective destroys coherence & purity; unitary preserves purity.
    assert d["projective"]["coherence"] < 1e-9, "projection should fully dephase"
    assert d["projective"]["purity"] < d["start"]["purity"] - 1e-3
    assert abs(d["unitary"]["purity"] - 1.0) < 1e-9, "unitary must preserve purity"
    assert d["unitary"]["coherence"] > 1e-3, "unitary should keep coherence"
    print("  -> unitary keeps purity=1 (coherent); projection collapses it. The")
    print("     DISTINCTION is real; whether it gives the collapse signal for")
    print("     real reasoning is the gating experiment (needs the built model).")
