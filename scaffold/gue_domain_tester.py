"""
gue_domain_tester.py
====================

Domain-aware GUE testing framework.

The core GUE mathematics in gue_engine.py are domain-independent.
But the preprocessing — what counts as a "spacing" and how you
normalise it — differs substantially across domains:

  SPECTRAL   Eigenvalues of matrices: unfold by Wigner semicircle law
             (the expected eigenvalue density for large GUE matrices),
             not by mean spacing. Mean-spacing unfolding distorts the
             bulk statistics.

  TEMPORAL   Event timestamps (neural spikes, market trades, bus
             arrivals): the event rate λ(t) can vary over time. Unfold
             by the integrated rate Λ(t) = ∫₀ᵗ λ(u)du, estimated via
             KDE. Without rate-correction, slow periods appear as
             artificially large spacings.

  SPATIAL_1D Points on a line (gene positions, rainfall events, cracks
             in a surface): same unfolding as temporal but over a
             spatial axis. Handle boundary effects carefully.

  LINGUISTIC Positions of target features in text (stressed syllables,
             rhyme words, punctuation). The "spacing" is in character
             or word units. Multi-scale: test at word, sentence, or
             passage level.

  FINANCIAL  Log-returns, price levels, or inter-trade times. Returns
             often need volatility normalisation before testing.

  GENERIC    Raw scalar values. Unfold by mean spacing (the simplest
             approach, appropriate when no domain structure is known).

All domain handlers return an array of unfolded spacings ready for
gue_engine.ks_distance. The rest of the pipeline is shared.

Additional features:
  - Bootstrap confidence intervals on KS statistics
  - Rolling-window analysis (how does GUE-ness change across time/position?)
  - Multi-scale testing (does GUE emerge at some scales but not others?)
  - Multi-domain comparison report
"""

import re
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

from gue_engine import (
    unfold_spacings, ks_distance, identify_ensemble,
    gue_score, _pdf_gue, _S_GRID, _CDF_GUE,
)


# ============================================================
# Domain types
# ============================================================

SPECTRAL   = "spectral"
TEMPORAL   = "temporal"
SPATIAL_1D = "spatial_1d"
LINGUISTIC = "linguistic"
FINANCIAL  = "financial"
GENERIC    = "generic"

ALL_DOMAINS = {SPECTRAL, TEMPORAL, SPATIAL_1D, LINGUISTIC, FINANCIAL, GENERIC}


# ============================================================
# Domain-specific preprocessing
# ============================================================

def preprocess_spectral(data) -> np.ndarray:
    """
    Eigenvalues (or a real/complex symmetric matrix).
    Unfolds by the Wigner semicircle law; returns unfolded POSITIONS
    in [0, N] so that identify_ensemble can compute spacings from them.
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("Matrix must be square.")
        eigs = np.linalg.eigvalsh(arr).real.astype(float)
    else:
        eigs = arr.real.ravel().astype(float) if np.iscomplexobj(arr)                else arr.ravel().astype(float)
    eigs = np.sort(eigs[np.isfinite(eigs)])
    if len(eigs) < 4:
        return np.array([])

    R      = (eigs.max() - eigs.min()) / 2.0
    center = (eigs.max() + eigs.min()) / 2.0
    if R == 0:
        return np.array([])

    def semicircle_cdf(x):
        xn = np.clip((x - center) / R, -1.0, 1.0)
        return 0.5 + (xn * np.sqrt(1.0 - xn ** 2) + np.arcsin(xn)) / np.pi

    return semicircle_cdf(eigs) * len(eigs)  # positions, not spacings


def preprocess_temporal(
    timestamps,
    kde_bandwidth: Optional[float] = None,
    n_grid: int = 512,
) -> np.ndarray:
    """
    Sequence of event timestamps. Corrects for non-constant rate.
    Unfolds by the integrated rate Λ(t) = ∫₀ᵗ λ(u) du, where λ(u)
    is estimated by KDE over the timestamps.

    kde_bandwidth: KDE bandwidth (auto-selected if None via Silverman's rule).
    """
    ts = np.sort(np.asarray(timestamps, dtype=float))
    ts = ts[np.isfinite(ts)]
    if len(ts) < 4:
        return np.array([])

    # KDE bandwidth (Silverman's rule if not specified)
    if kde_bandwidth is None:
        std = np.std(ts)
        if std == 0:
            return unfold_spacings(ts)
        kde_bandwidth = 1.06 * std * len(ts)**(-0.2)

    # Evaluate KDE on a grid spanning the data
    t_min, t_max = ts.min(), ts.max()
    grid = np.linspace(t_min, t_max, n_grid)
    diffs = (grid[:, None] - ts[None, :]) / kde_bandwidth
    kde = np.exp(-0.5 * diffs**2).sum(axis=1)
    kde /= (kde_bandwidth * np.sqrt(2 * np.pi) * len(ts))
    kde = np.maximum(kde, 1e-10)

    # Integrated rate at each event time (trapezoidal)
    integrated_rate = np.interp(ts, grid,
                                np.concatenate([[0],
                                np.cumsum(np.diff(grid) *
                                          (kde[:-1] + kde[1:]) / 2)]))
    rescaled = integrated_rate * len(ts)
    return rescaled  # positions; identify_ensemble computes spacings


def preprocess_spatial_1d(
    positions,
    correct_for_density: bool = True,
    kde_bandwidth: Optional[float] = None,
) -> np.ndarray:
    """
    Point positions on a 1D axis (gene positions, rainfall sites, etc.).
    Handles boundary effects by working only on the interior.
    Optionally corrects for spatially varying density.
    """
    pos = np.sort(np.asarray(positions, dtype=float))
    pos = pos[np.isfinite(pos)]
    if len(pos) < 4:
        return np.array([])

    if correct_for_density:
        return preprocess_temporal(pos, kde_bandwidth=kde_bandwidth)
    else:
        return unfold_spacings(pos)


def preprocess_linguistic(
    positions_or_text,
    feature: str = "positions",
    unit: str = "characters",
    pattern: Optional[str] = None,
) -> np.ndarray:
    """
    Test GUE statistics in text or text-derived sequences.

    Modes:
      feature="positions" : `positions_or_text` is already a list of
                            integer positions (e.g. rhyme word positions).
                            Most flexible — extract positions yourself.

      feature="pattern"   : extract positions of regex `pattern` in the
                            string `positions_or_text`.

      feature="sentences" : positions of sentence boundaries in text.
      feature="newlines"  : positions of line breaks.
      feature="words"     : positions of word starts.

    unit: "characters" (default) | "words" — what unit positions are in.
    """
    if feature == "positions":
        pos = np.sort(np.asarray(positions_or_text, dtype=float))
    elif isinstance(positions_or_text, str):
        text = positions_or_text
        if feature == "pattern" and pattern:
            pos = np.array([m.start() for m in re.finditer(pattern, text)],
                           dtype=float)
        elif feature == "sentences":
            pos = np.array([m.start()
                            for m in re.finditer(r'[.!?]+\s', text)],
                           dtype=float)
        elif feature == "newlines":
            pos = np.array([m.start()
                            for m in re.finditer(r'\n', text)],
                           dtype=float)
        elif feature == "words":
            pos = np.array([m.start()
                            for m in re.finditer(r'\b\w', text)],
                           dtype=float)
        else:
            raise ValueError(f"Unknown feature '{feature}'")
        if unit == "words":
            words = re.findall(r'\S+', text)
            # Map character positions to word positions
            char_to_word = {}
            idx = 0
            for wi, w in enumerate(words):
                p = text.find(w, idx)
                char_to_word[p] = wi
                idx = p + len(w)
            pos = np.array(sorted(char_to_word.get(int(p), 0)
                                  for p in pos), dtype=float)
    else:
        raise ValueError("positions_or_text must be a list of positions or "
                         "a string when feature != 'positions'.")

    pos = pos[np.isfinite(pos)]
    if len(pos) < 4:
        return np.array([])
    return unfold_spacings(pos)


def preprocess_financial(
    values,
    mode: str = "log_returns",
    volatility_normalise: bool = True,
) -> np.ndarray:
    """
    Financial time series.

    mode:
      "log_returns"     : compute log returns and test spacing of returns.
      "inter_trade"     : values are trade timestamps; test inter-trade times.
      "price_levels"    : test spacing of price levels directly.

    volatility_normalise: if True, divide by rolling std (window=20)
                          before testing, to remove heteroskedasticity.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if mode == "log_returns":
        if len(arr) < 2:
            return np.array([])
        returns = np.diff(np.log(np.maximum(arr, 1e-10)))
        if volatility_normalise and len(returns) > 20:
            vol = np.array([
                np.std(returns[max(0, i-20):i+1])
                for i in range(len(returns))
            ])
            vol = np.maximum(vol, 1e-10)
            returns = returns / vol
        return np.sort(returns)  # sorted values; identify_ensemble computes NNS

    elif mode == "inter_trade":
        return preprocess_temporal(arr)

    elif mode == "price_levels":
        return unfold_spacings(arr)

    else:
        raise ValueError(f"Unknown financial mode '{mode}'")


def preprocess_generic(values) -> np.ndarray:
    """Fallback: simple mean-normalised nearest-neighbour spacings."""
    return unfold_spacings(values)


PREPROCESSORS: dict[str, Callable] = {
    SPECTRAL:   preprocess_spectral,
    TEMPORAL:   preprocess_temporal,
    SPATIAL_1D: preprocess_spatial_1d,
    LINGUISTIC: preprocess_linguistic,
    FINANCIAL:  preprocess_financial,
    GENERIC:    preprocess_generic,
}


# ============================================================
# Bootstrap confidence intervals
# ============================================================

def bootstrap_ci(
    spacings: np.ndarray,
    ensemble: str = "gue",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
) -> dict:
    """
    Bootstrap confidence interval for the KS distance from `ensemble`.
    Returns {"mean", "lower", "upper", "std"} at the (1-alpha) level.
    """
    if len(spacings) < 3:
        return {"mean": None, "lower": None, "upper": None, "std": None}
    ks_samples = []
    for _ in range(n_bootstrap):
        boot = np.random.choice(spacings, size=len(spacings), replace=True)
        ks_samples.append(ks_distance(boot, ensemble))
    ks_samples = np.array(ks_samples)
    lo = float(np.percentile(ks_samples, 100 * alpha / 2))
    hi = float(np.percentile(ks_samples, 100 * (1 - alpha / 2)))
    return {
        "mean":  round(float(np.mean(ks_samples)),  4),
        "lower": round(lo,                           4),
        "upper": round(hi,                           4),
        "std":   round(float(np.std(ks_samples)),   4),
    }


# ============================================================
# GUEDomainTest result
# ============================================================

@dataclass
class GUEDomainTest:
    """Result of one GUE test on one dataset in one domain."""

    domain_type:       str
    label:             str           # human label for this dataset
    n_raw:             int           # number of raw data points
    n_spacings:        int           # number of spacings after unfolding
    gue_score:         Optional[float]
    goe_score:         Optional[float]
    poisson_score:     Optional[float]
    closest_ensemble:  str
    ci_gue:            dict = field(default_factory=dict)
    window_index:      Optional[int] = None   # for rolling window tests
    window_position:   Optional[float] = None
    timestamp:         float = field(default_factory=time.time)
    notes:             str = ""

    @property
    def is_gue(self) -> bool:
        return self.closest_ensemble == "gue"

    def to_dict(self) -> dict:
        return {
            "domain_type":      self.domain_type,
            "label":            self.label,
            "n_raw":            self.n_raw,
            "n_spacings":       self.n_spacings,
            "gue_score":        self.gue_score,
            "goe_score":        self.goe_score,
            "poisson_score":    self.poisson_score,
            "closest_ensemble": self.closest_ensemble,
            "ci_gue":           self.ci_gue,
            "window_index":     self.window_index,
            "window_position":  self.window_position,
            "notes":            self.notes,
        }

    def summary(self) -> str:
        score_str = f"{self.gue_score:.3f}" if self.gue_score else "n/a"
        ci = self.ci_gue
        ci_str = ""
        if ci.get("lower") is not None:
            ci_str = f" [95% CI: {1-ci['upper']:.3f}–{1-ci['lower']:.3f}]"
        closest = self.closest_ensemble
        marker = "★" if closest == "gue" else " "
        return (
            f"{marker} {self.label} ({self.domain_type})\n"
            f"    GUE={score_str}{ci_str}  "
            f"GOE={self.goe_score or 'n/a'}  "
            f"Poisson={self.poisson_score or 'n/a'}\n"
            f"    n={self.n_spacings} spacings. "
            f"Closest: {closest.upper()}."
            + (f"  {self.notes}" if self.notes else "")
        )


# ============================================================
# Multi-domain comparison report
# ============================================================

@dataclass
class MultiDomainReport:
    """Comparison across multiple GUEDomainTest results."""

    tests: list[GUEDomainTest]
    label: str = "GUE Multi-Domain Report"

    @property
    def gue_fraction(self) -> float:
        valid = [t for t in self.tests if t.gue_score is not None]
        if not valid:
            return 0.0
        return sum(1 for t in valid if t.is_gue) / len(valid)

    @property
    def mean_gue_score(self) -> Optional[float]:
        scores = [t.gue_score for t in self.tests if t.gue_score is not None]
        return float(np.mean(scores)) if scores else None

    def ranked(self) -> list[GUEDomainTest]:
        return sorted(
            [t for t in self.tests if t.gue_score is not None],
            key=lambda t: t.gue_score,
            reverse=True,
        )

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"{self.label}",
            f"{'='*60}",
            f"Domains tested: {len(self.tests)}",
            f"GUE fraction:   {self.gue_fraction:.0%} "
            f"({sum(1 for t in self.tests if t.is_gue)}/{len(self.tests)})",
        ]
        mean = self.mean_gue_score
        if mean is not None:
            lines.append(f"Mean GUE score: {mean:.3f}")
        lines.append("")
        lines.append("Results (ranked by GUE score):")
        for t in self.ranked():
            lines.append(t.summary())
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ============================================================
# Main tester class
# ============================================================

class DomainGUETester:
    """
    Tests any dataset for GUE statistics with domain-appropriate
    preprocessing.

    Usage:
        tester = DomainGUETester()

        # Single test
        result = tester.test(eigenvalues, domain=SPECTRAL, label="Hamiltonian")

        # Rolling window
        windows = tester.test_rolling(timestamps, domain=TEMPORAL,
                                       window_size=100, step=20)

        # Multi-scale
        scales = tester.test_multiscale(text_positions, domain=LINGUISTIC)

        # Compare across domains
        report = tester.compare([result1, result2, ...])
    """

    def __init__(
        self,
        n_bootstrap: int = 300,
        min_spacings: int = 8,
        verbose: bool = True,
    ):
        self.n_bootstrap  = n_bootstrap
        self.min_spacings = min_spacings
        self.verbose      = verbose

    # -------- Single test --------

    def test(
        self,
        data,
        domain: str = GENERIC,
        label: str = "",
        bootstrap: bool = True,
        preprocess_kwargs: Optional[dict] = None,
    ) -> GUEDomainTest:
        """
        Run a GUE test on `data` with `domain`-appropriate preprocessing.
        """
        if domain not in ALL_DOMAINS:
            raise ValueError(f"domain must be one of {ALL_DOMAINS}")
        label = label or domain

        preprocess_fn = PREPROCESSORS[domain]
        kwargs = preprocess_kwargs or {}

        raw = data
        n_raw = len(data) if hasattr(data, '__len__') else 0

        try:
            spacings = preprocess_fn(raw, **kwargs)
        except Exception as e:
            if self.verbose:
                print(f"[gue_tester] Preprocessing failed for '{label}': {e}")
            spacings = np.array([])

        from gue_engine import unfold_spacings as _us
        n_spacings = len(_us(spacings)) if len(spacings) >= 2 else 0

        if n_spacings < self.min_spacings:
            result = GUEDomainTest(
                domain_type=domain, label=label,
                n_raw=n_raw, n_spacings=n_spacings,
                gue_score=None, goe_score=None, poisson_score=None,
                closest_ensemble="insufficient_data",
                notes=f"Need ≥ {self.min_spacings} spacings, got {n_spacings}.",
            )
            if self.verbose:
                print(f"[gue_tester] {label}: insufficient data ({n_spacings} "
                      f"spacings, need {self.min_spacings})")
            return result

        ens = identify_ensemble(spacings)
        ci  = bootstrap_ci(spacings, "gue", self.n_bootstrap) \
              if bootstrap else {}

        notes = ""
        if n_spacings < 30:
            notes = f"Small sample (n={n_spacings}); treat CI as approximate."

        result = GUEDomainTest(
            domain_type=domain, label=label,
            n_raw=n_raw, n_spacings=n_spacings,
            gue_score=ens.get("gue"), goe_score=ens.get("goe"),
            poisson_score=ens.get("poisson"),
            closest_ensemble=ens.get("closest", "unknown"),
            ci_gue=ci, notes=notes,
        )
        if self.verbose:
            print(result.summary())
        return result

    # -------- Rolling window --------

    def test_rolling(
        self,
        data,
        domain: str = GENERIC,
        window_size: int = 50,
        step: int = 10,
        label: str = "",
        preprocess_kwargs: Optional[dict] = None,
    ) -> list[GUEDomainTest]:
        """
        Slide a window across the data and test GUE at each position.
        Returns one GUEDomainTest per window.
        Useful for detecting WHERE in a sequence GUE emerges.
        """
        data_arr = np.asarray(data, dtype=float)
        n = len(data_arr)
        results: list[GUEDomainTest] = []
        positions = range(0, n - window_size + 1, step)

        if self.verbose:
            print(f"[gue_tester] Rolling window: {len(list(positions))} windows "
                  f"of size {window_size} (step {step})")

        for i, start in enumerate(positions):
            window = data_arr[start: start + window_size]
            window_label = f"{label} [w{i}: {start}–{start+window_size}]"
            # Suppress per-window verbose output
            old_verbose = self.verbose
            self.verbose = False
            result = self.test(
                window, domain=domain,
                label=window_label,
                bootstrap=False,   # skip bootstrap for speed
                preprocess_kwargs=preprocess_kwargs,
            )
            self.verbose = old_verbose
            result.window_index    = i
            result.window_position = float(start + window_size / 2)
            results.append(result)

        if self.verbose:
            scores = [r.gue_score for r in results if r.gue_score is not None]
            if scores:
                best_i = int(np.argmax(scores))
                print(f"  Best window: {results[best_i].label} "
                      f"(GUE={results[best_i].gue_score:.3f})")
                print(f"  Mean GUE across windows: {np.mean(scores):.3f} "
                      f"± {np.std(scores):.3f}")
        return results

    # -------- Multi-scale --------

    def test_multiscale(
        self,
        data,
        domain: str = GENERIC,
        scales: Optional[list] = None,
        label: str = "",
        preprocess_kwargs: Optional[dict] = None,
    ) -> list[GUEDomainTest]:
        """
        Test at multiple coarsening levels.
        At each scale k, only every k-th data point is used.
        Reveals at which granularity GUE structure is strongest.
        """
        data_arr = np.asarray(data, dtype=float)
        if scales is None:
            max_scale = max(2, len(data_arr) // (self.min_spacings + 2))
            scales = [1, 2, 4, 8, 16, 32]
            scales = [s for s in scales if s <= max_scale]

        results: list[GUEDomainTest] = []
        if self.verbose:
            print(f"[gue_tester] Multi-scale: testing {len(scales)} scales")

        for scale in scales:
            coarsened = data_arr[::scale]
            scale_label = f"{label} [scale={scale}]"
            old_verbose = self.verbose
            self.verbose = False
            result = self.test(
                coarsened, domain=domain, label=scale_label,
                bootstrap=False,
                preprocess_kwargs=preprocess_kwargs,
            )
            self.verbose = old_verbose
            results.append(result)

        if self.verbose:
            print(f"  Scale | GUE score | Closest")
            for r, s in zip(results, scales):
                score_str = f"{r.gue_score:.3f}" if r.gue_score else "n/a"
                print(f"  {s:<5} | {score_str:<9} | {r.closest_ensemble}")

        return results

    # -------- Compare --------

    def compare(
        self,
        tests: list[GUEDomainTest],
        label: str = "GUE Multi-Domain Comparison",
    ) -> MultiDomainReport:
        report = MultiDomainReport(tests=tests, label=label)
        if self.verbose:
            print(report.summary())
        return report

    # -------- Convenience: test rolling and return score trajectory --------

    def rolling_gue_trajectory(
        self,
        data,
        domain: str = GENERIC,
        window_size: int = 50,
        step: int = 10,
        preprocess_kwargs: Optional[dict] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (positions, gue_scores) arrays for plotting.
        Positions are the centre of each window.
        """
        results = self.test_rolling(
            data, domain=domain,
            window_size=window_size, step=step,
            preprocess_kwargs=preprocess_kwargs,
        )
        positions = np.array([r.window_position for r in results
                              if r.window_position is not None])
        scores    = np.array([r.gue_score if r.gue_score is not None else float('nan')
                              for r in results
                              if r.window_position is not None])
        return positions, scores
