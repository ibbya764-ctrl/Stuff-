"""
gue_engine.py
=============

GUE (Gaussian Unitary Ensemble) distribution engine.

The GUE is defined by its nearest-neighbor spacing distribution —
the Wigner surmise:

    P_GUE(s) = (32/π²) s² exp(−4s²/π)

Two properties distinguish it from simpler distributions:

  - Level repulsion: P(s) ~ s² for small s. Spacings near zero are
    quadratically suppressed — elements avoid crowding together.
  - Soft upper bound: very large gaps are also suppressed. The
    distribution favours an intermediate, characteristic spacing.

This module:

  1. Provides exact GUE, GOE, and Poisson distributions for comparison.

  2. Measures how far any distribution of values is from GUE, using
     the KS statistic on unfolded nearest-neighbour spacings.

  3. For any existing distribution, computes what value added next
     would best improve GUE convergence — giving the system a target.

  4. Integrates with each component of the reasoning system:
       - TechniqueLibrary     (co-occurrence matrix eigenvalue spacing)
       - ObligationStore      (similarity matrix eigenvalue spacing)
       - Branch scores        (spacing of scores across competing branches)
       - Structure mapping    (GUE-based prior over unmapped entity relevance)

  5. Provides a SystemGUEMonitor that runs after each pipeline pass
     and logs the convergence trajectory over time.

No ML dependencies. Uses numpy and scipy only.
"""

import os
import json
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


PI = math.pi

# ============================================================
# Core GUE mathematics
# ============================================================
# Pre-compute CDFs on a fine grid at import time.
# All CDF lookups are O(log N) interpolation thereafter.

_N_GRID = 2000
_S_MAX  = 5.0
_S_GRID = np.linspace(0.0, _S_MAX, _N_GRID)


def _pdf_gue(s: np.ndarray) -> np.ndarray:
    """Wigner surmise for GUE."""
    s = np.asarray(s, dtype=float)
    return (32.0 / PI**2) * s**2 * np.exp(-4.0 * s**2 / PI)


def _pdf_goe(s: np.ndarray) -> np.ndarray:
    """Wigner surmise for GOE."""
    s = np.asarray(s, dtype=float)
    return (PI / 2.0) * s * np.exp(-PI * s**2 / 4.0)


def _pdf_poisson(s: np.ndarray) -> np.ndarray:
    """Poisson — no correlations (baseline)."""
    s = np.asarray(s, dtype=float)
    return np.exp(-s)


def _build_cdf(pdf_fn, grid: np.ndarray) -> np.ndarray:
    vals = pdf_fn(grid)
    cdf = np.zeros_like(grid)
    _integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    for i in range(1, len(grid)):
        cdf[i] = _integrate(vals[:i+1], grid[:i+1])
    mx = cdf[-1]
    return cdf / mx if mx > 0 else cdf


_CDF_GUE     = _build_cdf(_pdf_gue,     _S_GRID)
_CDF_GOE     = _build_cdf(_pdf_goe,     _S_GRID)
_CDF_POISSON = _build_cdf(_pdf_poisson, _S_GRID)


def _cdf_gue(s)     -> np.ndarray: return np.interp(s, _S_GRID, _CDF_GUE)
def _cdf_goe(s)     -> np.ndarray: return np.interp(s, _S_GRID, _CDF_GOE)
def _cdf_poisson(s) -> np.ndarray: return np.interp(s, _S_GRID, _CDF_POISSON)


def _icdf_gue(p: np.ndarray) -> np.ndarray:
    """Inverse CDF (quantile function) of GUE via interpolation."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return np.interp(p, _CDF_GUE, _S_GRID)


# ============================================================
# Unfolding and KS distance
# ============================================================

def unfold_spacings(values) -> np.ndarray:
    """
    Compute unfolded nearest-neighbour spacings from a set of values.
    Unfolding normalises by local density (here: mean spacing) so the
    resulting spacings have mean ≈ 1 and are comparable across systems.

    Returns an empty array if fewer than 3 values are provided.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.sort(arr)
    if len(arr) < 3:
        return np.array([])
    spacings = np.diff(arr)
    spacings = spacings[spacings > 0]
    if len(spacings) == 0 or np.mean(spacings) == 0:
        return np.array([])
    return spacings / np.mean(spacings)


def ks_distance(spacings: np.ndarray, ensemble: str = "gue") -> float:
    """
    Kolmogorov-Smirnov distance between the sample spacing distribution
    and the theoretical ensemble CDF.

    ensemble: "gue" | "goe" | "poisson"
    Returns value in [0, 1]; 0 = perfect match, 1 = maximally different.
    """
    if len(spacings) == 0:
        return 1.0
    cdf_fn = {"gue": _cdf_gue, "goe": _cdf_goe,
               "poisson": _cdf_poisson}[ensemble]
    s = np.sort(spacings)
    n = len(s)
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    theory  = cdf_fn(s)
    return float(np.max(np.maximum(
        np.abs(ecdf_hi - theory),
        np.abs(ecdf_lo - theory),
    )))


def gue_score(values, min_values: int = 4) -> float:
    """
    Convenience: how close is a set of values to GUE spacing?
    Returns a score in [0, 1] where 1 = perfect GUE, 0 = maximally far.
    Also computes GOE and Poisson distances so the caller can see
    which distribution the data is closest to.
    """
    spacings = unfold_spacings(values)
    if len(spacings) < min_values - 1:
        return float("nan")
    return 1.0 - ks_distance(spacings, "gue")


def identify_ensemble(values, min_values: int = 4) -> dict:
    """
    Measure KS distance from GUE, GOE, and Poisson.
    Returns the closest ensemble and distances.
    """
    spacings = unfold_spacings(values)
    if len(spacings) < min_values - 1:
        return {"closest": "insufficient_data",
                "gue": None, "goe": None, "poisson": None}
    d_gue     = ks_distance(spacings, "gue")
    d_goe     = ks_distance(spacings, "goe")
    d_poisson = ks_distance(spacings, "poisson")
    closest   = min(
        [("gue", d_gue), ("goe", d_goe), ("poisson", d_poisson)],
        key=lambda x: x[1],
    )[0]
    return {
        "closest": closest,
        "gue":     round(1.0 - d_gue,     3),
        "goe":     round(1.0 - d_goe,     3),
        "poisson": round(1.0 - d_poisson, 3),
    }


# ============================================================
# Optimal next value
# ============================================================

def optimal_next_value(
    values,
    search_range: Optional[tuple] = None,
    n_candidates: int = 500,
) -> dict:
    """
    Given existing values, find the scalar value whose addition best
    improves GUE convergence (minimises KS distance from GUE).

    search_range: (low, high) for the search. Defaults to the current
    value range, expanded by 20% on each side.

    Returns {'value': float, 'new_gue_score': float,
             'old_gue_score': float, 'improvement': float}
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return {"value": None, "note": "insufficient data"}

    if search_range is None:
        lo, hi = float(arr.min()), float(arr.max())
        margin = (hi - lo) * 0.2 if hi > lo else 0.5
        lo, hi = lo - margin, hi + margin
    else:
        lo, hi = search_range

    candidates = np.linspace(lo, hi, n_candidates)
    old_score  = gue_score(arr)

    best_val   = candidates[0]
    best_score = -1.0
    for c in candidates:
        trial    = np.append(arr, c)
        score    = gue_score(trial)
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_val   = c

    return {
        "value":         float(best_val),
        "new_gue_score": round(best_score, 4),
        "old_gue_score": round(old_score, 4) if not math.isnan(old_score) else None,
        "improvement":   round(best_score - (old_score if not math.isnan(old_score) else 0), 4),
    }


# ============================================================
# GUE-based entity relevance prior
# ============================================================

class GUEEntityPrior:
    """
    Assigns relevance probabilities to unmapped entities in a partial
    structure mapping, using GUE level repulsion as the prior.

    Mapped entities are clearly relevant (probability near 1.0).
    Unmapped entities get GUE-spaced probabilities below 0.5.

    The level repulsion property (P(s) ~ s²) means entities receive
    distinct probabilities — the prior resists assigning equal
    ambiguity to all unmapped entities and instead spreads them.
    """

    def __init__(self, mapped_floor: float = 0.80, unmapped_ceiling: float = 0.45):
        self.mapped_floor    = mapped_floor
        self.unmapped_ceiling = unmapped_ceiling

    def assign(
        self,
        mapped_entity_ids: list,
        unmapped_entity_ids: list,
        type_compatibility: Optional[dict] = None,
    ) -> dict:
        """
        Returns {entity_id: relevance_probability}.

        type_compatibility: optional {entity_id: float in [0,1]} giving
        a type-compatibility hint for unmapped entities. Higher means
        the entity type is more compatible with the mapped entities.
        """
        result = {}

        # Mapped entities: GUE-spaced in [mapped_floor, 1.0]
        n_mapped = len(mapped_entity_ids)
        if n_mapped > 0:
            lo, hi = self.mapped_floor, 1.0
            mapped_probs = self._gue_spaced_in_range(n_mapped, lo, hi)
            for eid, prob in zip(mapped_entity_ids, mapped_probs):
                result[eid] = float(prob)

        # Unmapped entities: GUE-spaced in [0, unmapped_ceiling]
        n_unmapped = len(unmapped_entity_ids)
        if n_unmapped > 0:
            lo, hi = 0.0, self.unmapped_ceiling
            unmapped_probs = self._gue_spaced_in_range(n_unmapped, lo, hi)

            # If type compatibility hints are provided, re-rank by them
            if type_compatibility:
                order = sorted(
                    unmapped_entity_ids,
                    key=lambda e: type_compatibility.get(e, 0.0),
                )
            else:
                order = unmapped_entity_ids

            for eid, prob in zip(order, sorted(unmapped_probs)):
                result[eid] = float(prob)

        return result

    @staticmethod
    def _gue_spaced_in_range(n: int, lo: float, hi: float) -> np.ndarray:
        """
        Generate n values with GUE-like spacing in [lo, hi].
        Uses GUE quantiles to place the values.
        """
        if n == 1:
            return np.array([(lo + hi) / 2])
        # Uniform quantile points, then map through GUE inverse CDF
        # to get GUE-spaced positions in [0,1], then rescale to [lo, hi].
        q = np.linspace(0.05, 0.95, n)
        gue_positions = _icdf_gue(q)
        # Normalise to [0, 1]
        mn, mx = gue_positions.min(), gue_positions.max()
        if mx > mn:
            gue_positions = (gue_positions - mn) / (mx - mn)
        else:
            gue_positions = np.linspace(0, 1, n)
        return lo + gue_positions * (hi - lo)


# ============================================================
# Component monitors
# ============================================================

@dataclass
class GUEMeasurement:
    """Snapshot of GUE statistics for one component."""
    component:      str
    n_values:       int
    gue_score:      Optional[float]      # 1 = perfect GUE
    goe_score:      Optional[float]
    poisson_score:  Optional[float]
    closest:        str
    optimal_next:   Optional[dict] = None
    timestamp:      float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "component":     self.component,
            "n_values":      self.n_values,
            "gue_score":     self.gue_score,
            "goe_score":     self.goe_score,
            "poisson_score": self.poisson_score,
            "closest":       self.closest,
            "optimal_next":  self.optimal_next,
            "timestamp":     self.timestamp,
        }


def _measure(values, component: str, include_optimal: bool = True) -> GUEMeasurement:
    arr = [v for v in values if v is not None and math.isfinite(float(v))]
    ens = identify_ensemble(arr)
    opt = optimal_next_value(arr) if include_optimal and len(arr) >= 4 else None
    return GUEMeasurement(
        component=component,
        n_values=len(arr),
        gue_score=ens.get("gue"),
        goe_score=ens.get("goe"),
        poisson_score=ens.get("poisson"),
        closest=ens.get("closest", "unknown"),
        optimal_next=opt,
    )


class BranchGUEScorer:
    """
    Measures GUE statistics over the distribution of branch scores in
    a pipeline run, and computes a GUE-convergence bonus for each
    candidate branch score.
    """

    def __init__(self, weight: float = 0.15):
        """weight: how much the GUE bonus contributes to total_score."""
        self.weight = weight

    def measure(self, branch_scores: list) -> GUEMeasurement:
        return _measure(branch_scores, component="branch_scores")

    def gue_bonus(
        self,
        candidate_score: float,
        existing_scores: list,
    ) -> float:
        """
        How much would adding a branch with `candidate_score` improve
        the branch score distribution's GUE convergence?

        Returns a bonus in [0, weight] to add to the branch's total_score.
        """
        if len(existing_scores) < 3:
            return 0.0
        old_ks  = ks_distance(unfold_spacings(existing_scores), "gue")
        trial   = list(existing_scores) + [candidate_score]
        new_ks  = ks_distance(unfold_spacings(trial),           "gue")
        improvement = old_ks - new_ks          # positive = improvement
        # Scale to [0, weight]; improvement can be at most 1.
        return float(np.clip(improvement * self.weight, 0.0, self.weight))

    def optimal_branch_score(self, existing_scores: list) -> Optional[float]:
        """What score should the next branch aim for?"""
        if len(existing_scores) < 2:
            return None
        result = optimal_next_value(existing_scores)
        return result.get("value")


class TechniqueLibraryGUEMonitor:
    """
    Measures GUE statistics over:
      (a) the distribution of technique success rates
      (b) eigenvalue spacing of the co-occurrence matrix
    """

    def measure_success_rates(self, library) -> GUEMeasurement:
        rates = []
        for t in library._data.get("techniques", {}).values():
            n_success = t.get("n_success", 0)
            n_fail    = t.get("n_fail",    0)
            total = n_success + n_fail
            if total > 0:
                rates.append(n_success / total)
        return _measure(rates, component="technique_success_rates")

    def measure_cooccurrence_eigenvalues(
        self, library, graph: "CoOccurrenceGraph",
    ) -> GUEMeasurement:
        """Build co-occurrence matrix and check eigenvalue spacing."""
        tids = list(library._data.get("techniques", {}).keys())
        n = len(tids)
        if n < 4:
            return GUEMeasurement(
                component="cooccurrence_eigenvalues",
                n_values=n, gue_score=None, goe_score=None,
                poisson_score=None, closest="insufficient_data",
            )
        idx = {tid: i for i, tid in enumerate(tids)}
        mat = np.zeros((n, n))
        for key, count in graph._data.get("co_counts", {}).items():
            a, b = key.split("|", 1)
            if a in idx and b in idx:
                mat[idx[a], idx[b]] = count
                mat[idx[b], idx[a]] = count
        eigenvalues = np.linalg.eigvalsh(mat)
        return _measure(eigenvalues, component="cooccurrence_eigenvalues")


class ObligationStoreGUEMonitor:
    """
    Measures GUE statistics over the distribution of obligation
    occurrence counts (how often each gap recurs).
    """

    def measure(self, store) -> GUEMeasurement:
        try:
            gaps = store.query_persistent_gaps(top_n=200)
            counts = [g.n_occurrences for g in gaps]
        except Exception:
            return GUEMeasurement(
                component="obligation_counts",
                n_values=0, gue_score=None, goe_score=None,
                poisson_score=None, closest="insufficient_data",
            )
        return _measure(counts, component="obligation_counts")

    def gue_optimal_target_score(self, store) -> Optional[float]:
        """
        Which obligation-count value, if added via a new run, would
        best improve GUE convergence of the obligation distribution?
        This tells the curiosity engine what *frequency* of target
        to prioritise.
        """
        try:
            gaps = store.query_persistent_gaps(top_n=200)
            counts = [float(g.n_occurrences) for g in gaps]
        except Exception:
            return None
        result = optimal_next_value(counts)
        return result.get("value")


# ============================================================
# System-level monitor
# ============================================================

@dataclass
class GUEReport:
    """Full GUE health report for one pipeline run."""
    run_id: str
    timestamp: float
    branch_scores: GUEMeasurement
    technique_rates: GUEMeasurement
    obligation_counts: GUEMeasurement
    overall_gue_score: float        # mean of component GUE scores
    trend: Optional[float] = None   # vs. previous run; + = improving

    def to_dict(self) -> dict:
        return {
            "run_id":            self.run_id,
            "timestamp":         self.timestamp,
            "overall_gue_score": round(self.overall_gue_score, 4),
            "trend":             round(self.trend, 4) if self.trend else None,
            "components": {
                "branch_scores":     self.branch_scores.to_dict(),
                "technique_rates":   self.technique_rates.to_dict(),
                "obligation_counts": self.obligation_counts.to_dict(),
            },
        }

    def summary(self) -> str:
        lines = [
            f"GUE Report [{self.run_id}]",
            f"  Overall GUE score: {self.overall_gue_score:.3f}"
            + (f" (trend: {'+' if self.trend >= 0 else ''}{self.trend:.3f})"
               if self.trend is not None else ""),
            "",
        ]
        for m in [self.branch_scores, self.technique_rates,
                  self.obligation_counts]:
            score_str = (f"{m.gue_score:.3f}" if m.gue_score is not None
                         else "n/a")
            lines.append(
                f"  {m.component:<35} GUE={score_str:<7} "
                f"closest={m.closest} (n={m.n_values})"
            )
            if m.optimal_next and m.optimal_next.get("improvement", 0) > 0.01:
                lines.append(
                    f"    → optimal next value: "
                    f"{m.optimal_next['value']:.4f} "
                    f"(improvement {m.optimal_next['improvement']:+.3f})"
                )
        return "\n".join(lines)


class SystemGUEMonitor:
    """
    Top-level monitor. Call after each pipeline run. Aggregates all
    component measurements and logs the convergence trajectory.
    """

    def __init__(
        self,
        log_path: str = "./gue_log.json",
        verbose: bool = True,
    ):
        self.log_path = log_path
        self.verbose  = verbose
        self._log: list[dict] = []
        self._branch_scorer   = BranchGUEScorer()
        self._tech_monitor    = TechniqueLibraryGUEMonitor()
        self._obl_monitor     = ObligationStoreGUEMonitor()
        self._entity_prior    = GUEEntityPrior()
        self._load()

    # ---- I/O ----

    def _load(self) -> None:
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path) as f:
                self._log = json.load(f).get("runs", [])
        except (json.JSONDecodeError, OSError):
            self._log = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump({"runs": self._log}, f, indent=2)

    # ---- Main entry point ----

    def report(
        self,
        run_id: str,
        branch_scores: list,
        library=None,
        cooccurrence_graph=None,
        obligation_store=None,
    ) -> GUEReport:
        """
        Compute a full GUE report for this run. Pass in whatever
        components are available; unavailable ones are skipped.
        """
        b_meas = self._branch_scorer.measure(branch_scores)

        t_meas = (
            self._tech_monitor.measure_success_rates(library)
            if library is not None
            else GUEMeasurement(
                component="technique_rates", n_values=0,
                gue_score=None, goe_score=None,
                poisson_score=None, closest="unavailable",
            )
        )

        o_meas = (
            self._obl_monitor.measure(obligation_store)
            if obligation_store is not None
            else GUEMeasurement(
                component="obligation_counts", n_values=0,
                gue_score=None, goe_score=None,
                poisson_score=None, closest="unavailable",
            )
        )

        # Overall score: mean of available component scores
        available = [
            m.gue_score for m in [b_meas, t_meas, o_meas]
            if m.gue_score is not None
        ]
        overall = float(np.mean(available)) if available else float("nan")

        # Trend vs. previous run
        trend = None
        if self._log:
            prev = self._log[-1].get("overall_gue_score")
            if prev is not None and not math.isnan(overall):
                trend = overall - prev

        report = GUEReport(
            run_id=run_id,
            timestamp=time.time(),
            branch_scores=b_meas,
            technique_rates=t_meas,
            obligation_counts=o_meas,
            overall_gue_score=overall,
            trend=trend,
        )

        self._log.append(report.to_dict())
        self._save()

        if self.verbose:
            print(report.summary())

        return report

    # ---- Accessors for other modules ----

    @property
    def branch_scorer(self) -> BranchGUEScorer:
        return self._branch_scorer

    @property
    def entity_prior(self) -> GUEEntityPrior:
        return self._entity_prior

    @property
    def obligation_monitor(self) -> ObligationStoreGUEMonitor:
        return self._obl_monitor

    def convergence_trend(self) -> list[float]:
        """Historical GUE scores across all runs."""
        return [r.get("overall_gue_score", 0.0) for r in self._log
                if r.get("overall_gue_score") is not None]
