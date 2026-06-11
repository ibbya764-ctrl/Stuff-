"""
hermitian_gue.py
================

GUE testing for Hermitian directed co-occurrence graphs.

The core prediction: a technique library that has accumulated rich
*directed* reasoning structure (A tends to set up B, B requires A
first) will have a Hermitian adjacency matrix whose eigenvalue spacing
statistics converge toward GUE. A library with only symmetric co-
occurrence (no directional information) produces a real symmetric
matrix whose eigenvalues follow GOE.

The transition is tracked by:
  - imaginary_fraction: how much of the off-diagonal weight is in the
    imaginary part (0 = real/GOE, 1 = maximally complex/GUE)
  - eigenvalue_gue_score: KS-distance score against GUE Wigner surmise
  - ensemble_identified: "gue", "goe", or "poisson"
  - goe_to_gue_shift: difference in GUE score between current state and
    what it would be if all phases were zeroed (pure symmetric baseline)

This module imports from directed_graph.py and gue_engine.py.
No changes to either of those files required.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from directed_graph import DirectedCoOccurrenceGraph
from gue_domain_tester import preprocess_spectral
from gue_engine import identify_ensemble, gue_score


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class HermitianGUEResult:
    """
    Full GUE measurement from one DirectedCoOccurrenceGraph.
    """

    n_techniques:         int
    n_runs:               int
    imaginary_fraction:   float        # 0 = GOE, 1 = maximally GUE
    goe_fraction:         float        # fraction of pairs with no directional data

    # Eigenvalue statistics (from Hermitian matrix)
    gue_score:            Optional[float]
    goe_score:            Optional[float]
    poisson_score:        Optional[float]
    ensemble_identified:  str

    # Baseline comparison: what would the scores be with phases zeroed?
    baseline_gue_score:   Optional[float]   # GOE baseline (imaginary = 0)
    goe_to_gue_shift:     Optional[float]   # how much GUE score improved

    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "n_techniques":         self.n_techniques,
            "n_runs":               self.n_runs,
            "imaginary_fraction":   round(self.imaginary_fraction, 4),
            "goe_fraction":         round(self.goe_fraction, 4),
            "gue_score":            self.gue_score,
            "goe_score":            self.goe_score,
            "poisson_score":        self.poisson_score,
            "ensemble_identified":  self.ensemble_identified,
            "baseline_gue_score":   self.baseline_gue_score,
            "goe_to_gue_shift":     self.goe_to_gue_shift,
            "notes":                self.notes,
        }

    def summary(self) -> str:
        lines = [
            f"Hermitian GUE Result ({self.n_techniques} techniques, "
            f"{self.n_runs} runs)",
            f"  Imaginary fraction:  {self.imaginary_fraction:.3f}  "
            f"(0=GOE, 1=max-GUE)",
            f"  GOE-fraction:        {self.goe_fraction:.3f}  "
            f"(pairs without directional data)",
        ]
        if self.gue_score is not None:
            marker = "★" if self.ensemble_identified == "gue" else " "
            lines.append(
                f"  {marker} Eigenvalue stats: "
                f"GUE={self.gue_score:.3f}  "
                f"GOE={self.goe_score:.3f}  "
                f"Poisson={self.poisson_score:.3f}  "
                f"→ {self.ensemble_identified.upper()}"
            )
            if self.goe_to_gue_shift is not None:
                shift_str = f"{self.goe_to_gue_shift:+.3f}"
                lines.append(
                    f"    Shift vs symmetric baseline: {shift_str}  "
                    f"({'↑ GUE' if self.goe_to_gue_shift > 0 else '↓ away from GUE'})"
                )
        if self.notes:
            lines.append(f"  Note: {self.notes}")
        return "\n".join(lines)


# ============================================================
# Core measurement
# ============================================================

def measure_hermitian_gue(
    directed_graph: DirectedCoOccurrenceGraph,
    technique_ids: Optional[list] = None,
    min_techniques: int = 4,
    verbose: bool = True,
) -> HermitianGUEResult:
    """
    Build the Hermitian adjacency matrix from directed_graph and
    measure its eigenvalue spacing statistics.

    Also runs a symmetric baseline (all phases zeroed) to quantify
    the GOE → GUE shift from accumulated directional structure.

    Returns a HermitianGUEResult.
    """
    summary = directed_graph.summary()
    n_tech  = summary["n_techniques"]
    n_runs  = summary["n_runs"]
    im_frac = summary["imaginary_fraction"]
    goe_frac = summary["goe_fraction"]

    if n_tech < min_techniques:
        result = HermitianGUEResult(
            n_techniques=n_tech, n_runs=n_runs,
            imaginary_fraction=im_frac, goe_fraction=goe_frac,
            gue_score=None, goe_score=None, poisson_score=None,
            ensemble_identified="insufficient_data",
            baseline_gue_score=None, goe_to_gue_shift=None,
            notes=f"Need ≥ {min_techniques} techniques, have {n_tech}.",
        )
        if verbose:
            print(result.summary())
        return result

    # Build full Hermitian matrix
    H, tids = directed_graph.build_hermitian_matrix(
        technique_ids=technique_ids, normalise=True,
    )

    # Eigenvalues of Hermitian matrix are real
    eigenvalues_full = np.linalg.eigvalsh(H).real

    # GUE test on full (directed) matrix
    positions_full = preprocess_spectral(H)
    ens_full = identify_ensemble(positions_full)

    # --- Baseline: zero all phases → real symmetric matrix ---
    H_symmetric = H.real.copy()
    # Re-symmetrise (should already be symmetric in the real part, but be safe)
    H_symmetric = (H_symmetric + H_symmetric.T) / 2

    positions_sym = preprocess_spectral(H_symmetric)
    ens_sym = identify_ensemble(positions_sym)
    base_gue = ens_sym.get("gue")
    full_gue = ens_full.get("gue")

    shift: Optional[float] = None
    if base_gue is not None and full_gue is not None:
        shift = full_gue - base_gue

    notes = ""
    if n_tech < 20:
        notes = f"Small matrix (n={n_tech}); statistics are approximate."

    result = HermitianGUEResult(
        n_techniques=n_tech, n_runs=n_runs,
        imaginary_fraction=im_frac, goe_fraction=goe_frac,
        gue_score=full_gue,
        goe_score=ens_full.get("goe"),
        poisson_score=ens_full.get("poisson"),
        ensemble_identified=ens_full.get("closest", "unknown"),
        baseline_gue_score=base_gue,
        goe_to_gue_shift=shift,
        notes=notes,
    )

    if verbose:
        print(result.summary())
    return result


# ============================================================
# Transition tracker
# ============================================================

def track_goe_to_gue_transition(
    directed_graph: DirectedCoOccurrenceGraph,
    technique_ids: Optional[list] = None,
    n_steps: int = 10,
    verbose: bool = True,
) -> list[dict]:
    """
    Measure how the eigenvalue statistics evolve as the imaginary
    component is gradually increased from 0 → full strength.

    Useful for visualising and verifying the GOE → GUE transition.
    Returns a list of dicts: [{alpha, imaginary_fraction, gue_score,
    goe_score, ensemble}, ...] for alpha = 0, 0.1, ..., 1.0.
    """
    H, tids = directed_graph.build_hermitian_matrix(
        technique_ids=technique_ids, normalise=True,
    )
    H_real = H.real.copy()
    H_imag = H.imag.copy()

    steps = np.linspace(0, 1, n_steps + 1)
    trajectory = []

    for alpha in steps:
        H_alpha = H_real + 1j * alpha * H_imag
        # Restore Hermitian property
        H_alpha = (H_alpha + H_alpha.conj().T) / 2

        positions = preprocess_spectral(H_alpha)
        ens = identify_ensemble(positions)

        # Imaginary fraction at this alpha
        off_mask = ~np.eye(H_alpha.shape[0], dtype=bool)
        off_entries = H_alpha[off_mask]
        magnitudes = np.abs(off_entries)
        im_frac = float(
            np.mean(np.abs(off_entries.imag) / np.where(magnitudes > 0, magnitudes, 1))
        ) if len(off_entries) > 0 else 0.0

        point = {
            "alpha":               round(float(alpha), 2),
            "imaginary_fraction":  round(im_frac, 4),
            "gue_score":           ens.get("gue"),
            "goe_score":           ens.get("goe"),
            "ensemble":            ens.get("closest", "unknown"),
        }
        trajectory.append(point)

        if verbose:
            g = ens.get("gue") or 0
            bar = "█" * int(g * 25)
            print(
                f"  α={alpha:.1f}  Im={im_frac:.3f}  "
                f"GUE={str(ens.get('gue')):<7}  "
                f"GOE={str(ens.get('goe')):<7}  "
                f"→ {(ens.get('closest') or '?').upper():<7}  {bar}"
            )

    return trajectory


# ============================================================
# Integration: drop-in for TechniqueComposer
# ============================================================

def build_directed_composer(
    library,
    directed_graph_path: str = "./directed_cooccurrence.json",
):
    """
    Helper that builds a TechniqueComposer whose co-occurrence graph is
    a DirectedCoOccurrenceGraph. The composer gets all the same
    within-domain functionality plus Hermitian GUE tracking.

    Usage:
        from hermitian_gue import build_directed_composer
        composer = build_directed_composer(library)
        # Use exactly like TechniqueComposer
        composer.update_from_run(ordered_technique_ids)
        # Plus:
        result = measure_hermitian_gue(composer.graph)
    """
    from technique_composer import TechniqueComposer
    dgraph = DirectedCoOccurrenceGraph(directed_graph_path)
    composer = TechniqueComposer(library, cooccurrence_path=directed_graph_path)
    # Swap the symmetric graph for the directed one
    composer.graph = dgraph
    return composer
