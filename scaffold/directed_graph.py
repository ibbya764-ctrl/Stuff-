"""
directed_graph.py
=================

Drop-in replacement for CoOccurrenceGraph that additionally tracks
the temporal ordering of techniques within each run.

The key insight: symmetric co-occurrence matrices are real → GOE
statistics. When we encode *directionality* (does A tend to come
before B?) as a phase angle via the Euler formula, we get a
Hermitian matrix → GUE statistics emerge naturally as the system
accumulates directed structure.

The Hermitian adjacency matrix is:

    H_{ij} = |w_{ij}| · exp(i · φ_{ij}) / √N

where:
    |w_{ij}| = normalised co-occurrence strength
    φ_{ij}  = (π/2) · (n_{i→j} − n_{j→i}) / (n_{i→j} + n_{j→i} + 1)

    φ = 0     → A and B always co-occur simultaneously (symmetric)
    φ = +π/2  → A always precedes B (maximum directionality)
    φ = −π/2  → B always precedes A

Hermiticity is automatic: H_{ji} = conj(H_{ij}) because φ_{ji} = −φ_{ij}.

The transition from GOE to GUE statistics in the eigenvalues tracks
a real property of the system: as the technique library accumulates
richer directed causal structure (A tends to set up B, B tends to
require A first), the Hamiltonian becomes more complex-Hermitian and
its eigenvalue spacings converge toward GUE.
"""

import os
import json
import math
import numpy as np
from typing import Iterable, Optional


class DirectedCoOccurrenceGraph:
    """
    Tracks co-occurrence AND temporal ordering of techniques.

    Compatible with CoOccurrenceGraph: all the same methods work
    (update_from_run, co_count, lift, companions, summary, save).

    New methods:
        directed_count(a, b)      → how many runs had a before b
        directionality(a, b)      → float in [−1, 1]; +1 = a always first
        euler_phase(a, b)         → φ_{ij} in [−π/2, π/2]
        build_hermitian_matrix()  → complex Hermitian adjacency matrix
        goe_fraction()            → how much of the graph is still symmetric
    """

    def __init__(self, path: str = "./directed_cooccurrence.json"):
        self.path = path
        self._data: dict = {
            "version":   2,          # distinguish from old symmetric-only files
            "n_runs":    0,
            "n_uses":    {},          # technique_id → int
            "co_counts": {},          # f"{a}|{b}" (sorted) → int
            "directed":  {},          # f"{a}→{b}" (ordered) → int
        }
        self._load()

    # -------- I/O --------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                loaded = json.load(f)
            # Migrate from version-1 (symmetric-only) files
            if "directed" not in loaded:
                loaded["directed"] = {}
            if "version" not in loaded:
                loaded["version"] = 2
            self._data = loaded
        except (json.JSONDecodeError, OSError) as e:
            print(f"[directed_graph] Warning: could not load: {e}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    # -------- Key helpers --------

    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        """Symmetric key (sorted) for co-occurrence counts."""
        return "|".join(sorted([a, b]))

    @staticmethod
    def _directed_key(a: str, b: str) -> str:
        """Directed key preserving order."""
        return f"{a}→{b}"

    # -------- Updates --------

    def update_from_run(
        self,
        technique_ids,
        success: bool = True,
    ) -> None:
        """
        Record a successful run. Accepts either:
          - A list: order is preserved → directed counts extracted.
          - A set / unordered iterable: treated as simultaneous, no
            directional information extracted.

        Only successful runs update the graph (failed runs would be
        anti-evidence and are skipped, same as CoOccurrenceGraph).
        """
        if not success:
            return

        # Preserve order if list; deduplicate while keeping first occurrence
        if isinstance(technique_ids, list):
            seen = set()
            ordered = []
            for tid in technique_ids:
                if tid not in seen:
                    seen.add(tid)
                    ordered.append(tid)
        else:
            ordered = list(set(technique_ids))

        if not ordered:
            return

        self._data["n_runs"] += 1

        # Usage counts
        for tid in ordered:
            self._data["n_uses"][tid] = (
                self._data["n_uses"].get(tid, 0) + 1
            )

        # Symmetric co-occurrence
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                key = self._pair_key(a, b)
                self._data["co_counts"][key] = (
                    self._data["co_counts"].get(key, 0) + 1
                )

        # Directed counts (only when order is meaningful, i.e. list input)
        if isinstance(technique_ids, list):
            for i, a in enumerate(ordered):
                for b in ordered[i + 1:]:
                    dkey = self._directed_key(a, b)
                    self._data["directed"][dkey] = (
                        self._data["directed"].get(dkey, 0) + 1
                    )

        self.save()

    # -------- Symmetric queries (same as CoOccurrenceGraph) --------

    def co_count(self, a: str, b: str) -> int:
        return self._data["co_counts"].get(self._pair_key(a, b), 0)

    def lift(self, a: str, b: str) -> float:
        n_runs = self._data["n_runs"]
        if n_runs == 0:
            return 0.0
        n_a  = self._data["n_uses"].get(a, 0)
        n_b  = self._data["n_uses"].get(b, 0)
        n_ab = self.co_count(a, b)
        if n_a == 0 or n_b == 0 or n_ab == 0:
            return 0.0
        return (n_ab / n_runs) / ((n_a / n_runs) * (n_b / n_runs))

    def companions(
        self, tid: str, top_n: int = 5, min_lift: float = 1.2,
    ) -> list[tuple[str, float]]:
        scored = []
        for key, count in self._data["co_counts"].items():
            a, b = key.split("|", 1)
            if tid not in (a, b):
                continue
            other = b if a == tid else a
            if count < 2:
                continue
            l = self.lift(tid, other)
            if l >= min_lift:
                scored.append((other, l))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    # -------- Directed queries (new) --------

    def directed_count(self, a: str, b: str) -> int:
        """How many runs had technique a appearing before b."""
        return self._data["directed"].get(self._directed_key(a, b), 0)

    def directionality(self, a: str, b: str) -> float:
        """
        Signed directionality in [−1, +1].
          +1 → a always precedes b
          −1 → b always precedes a
           0 → equal or no directional data
        """
        n_ab = self.directed_count(a, b)
        n_ba = self.directed_count(b, a)
        total = n_ab + n_ba
        if total == 0:
            return 0.0
        return (n_ab - n_ba) / total

    def euler_phase(self, a: str, b: str) -> float:
        """
        Phase angle φ_{ab} ∈ [−π/2, +π/2] for H_{ab} = |w| exp(iφ).

        Uses a Laplace-smoothed directionality so that zero observations
        give φ = 0 (symmetric, no directional information) rather than
        an undefined value.
        """
        n_ab = self.directed_count(a, b)
        n_ba = self.directed_count(b, a)
        # Smoothed: divide by (n_ab + n_ba + 1) so zero observations → 0
        return (math.pi / 2.0) * (n_ab - n_ba) / (n_ab + n_ba + 1)

    # -------- Hermitian matrix construction --------

    def build_hermitian_matrix(
        self,
        technique_ids: Optional[list] = None,
        normalise: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build the complex Hermitian adjacency matrix.

        H_{ij} = strength_{ij} · exp(i · φ_{ij})
        H_{ii} = normalised usage count (real, on diagonal)

        Returns (H, technique_ids) where H is an (N×N) complex Hermitian
        numpy array and technique_ids is the ordered list of technique IDs
        corresponding to rows/columns.

        Normalisation: off-diagonal entries divided by √N (standard
        Wigner semicircle scaling for RMT universality class tests).
        """
        if technique_ids is None:
            technique_ids = sorted(self._data["n_uses"].keys())

        n = len(technique_ids)
        if n < 2:
            return np.zeros((n, n), dtype=complex), technique_ids

        idx = {tid: i for i, tid in enumerate(technique_ids)}

        # Maximum co-occurrence for normalisation
        max_co = max(self._data["co_counts"].values(), default=1)
        max_use = max(self._data["n_uses"].values(), default=1)

        H = np.zeros((n, n), dtype=complex)

        # Diagonal: normalised usage (real)
        for tid in technique_ids:
            i = idx[tid]
            H[i, i] = self._data["n_uses"].get(tid, 0) / max_use

        # Off-diagonal: Hermitian with Euler phases
        for ti in technique_ids:
            i = idx[ti]
            for tj in technique_ids:
                j = idx[tj]
                if i >= j:
                    continue
                co = self.co_count(ti, tj)
                if co == 0:
                    continue
                strength = co / max_co
                phi      = self.euler_phase(ti, tj)
                entry    = strength * np.exp(1j * phi)
                H[i, j]  = entry
                H[j, i]  = np.conj(entry)  # Hermitian conjugate

        # RMT normalisation: scale off-diagonal by 1/√N
        if normalise and n > 1:
            off_diag_mask = ~np.eye(n, dtype=bool)
            H[off_diag_mask] /= math.sqrt(n)

        return H, technique_ids

    def goe_fraction(self) -> float:
        """
        Fraction of technique pairs that have NO directional data —
        i.e. are still purely symmetric / GOE-contributing.
        Ranges from 1.0 (all symmetric) to 0.0 (all directed).
        """
        total_pairs = len(self._data["co_counts"])
        if total_pairs == 0:
            return 1.0
        directed_pairs = sum(
            1 for key in self._data["co_counts"]
            if any(
                self._data["directed"].get(f"{a}→{b}", 0) > 0 or
                self._data["directed"].get(f"{b}→{a}", 0) > 0
                for a, b in [key.split("|", 1)]
            )
        )
        return 1.0 - (directed_pairs / total_pairs)

    def imaginary_fraction(
        self, technique_ids: Optional[list] = None,
    ) -> float:
        """
        Mean |Im(H_{ij})| / |H_{ij}| over non-zero off-diagonal entries.
        0.0 = purely real (GOE), 1.0 = purely imaginary (maximally GUE).
        Useful as a quick diagnostic without full eigenvalue computation.
        """
        H, _ = self.build_hermitian_matrix(technique_ids, normalise=False)
        n = H.shape[0]
        if n < 2:
            return 0.0
        off = [(i, j) for i in range(n) for j in range(i + 1, n)
               if H[i, j] != 0]
        if not off:
            return 0.0
        fracs = []
        for i, j in off:
            mag = abs(H[i, j])
            fracs.append(abs(H[i, j].imag) / mag if mag > 0 else 0.0)
        return float(np.mean(fracs))

    # -------- Inspection --------

    def summary(self) -> dict:
        n_directed_pairs = sum(
            1 for k in self._data["directed"] if self._data["directed"][k] > 0
        )
        return {
            "n_runs":             self._data["n_runs"],
            "n_techniques":       len(self._data["n_uses"]),
            "n_pairs":            len(self._data["co_counts"]),
            "n_directed_pairs":   n_directed_pairs,
            "goe_fraction":       round(self.goe_fraction(), 3),
            "imaginary_fraction": round(
                self.imaginary_fraction(), 3
            ) if len(self._data["n_uses"]) >= 2 else 0.0,
        }
