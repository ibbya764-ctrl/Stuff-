"""Value/character telemetry for the CP2 AI program (v3 record, Part 6).

Companion to ``spectral_telemetry.py``. Where that module reads the DYNAMICS
operator's spectrum, this one reads the character/council mode bank -- the
persistent store where the v18 note claims values consolidate. Numpy-only, so
the harness runner can log it side-by-side with the spectral diagnostics per
checkpoint WITHOUT importing torch or the council package.

It consumes a plain snapshot dict (produced by ``snapshot_from_mode_bank`` when
a live bank is available, or hand-built in tests), so the firewall holds: this
is a read-only diagnostic. Telemetry may only ever ADD scrutiny -- nothing here
feeds consolidation gating, renewal protection, or audit-budget allocation
(one-way rule; see v18 §7 / addendum §3).

Diagnostics:
  1. lineage_summary       -- mode count, mean anchored vs cognitive fraction,
                              how many modes clear the value-status threshold.
  2. anchor_coordinate_spread -- committee/frame diversity measured in ANCHOR
                              coordinates, not the model's own learned basis
                              (v18 §5: a colluded committee looks diverse in its
                              own gamed basis; measure against the anchor).
  3. renewal_survival_summary -- distribution of cross-renewal survival, with a
                              hard DEGENERATE flag when survival is pinned at
                              ceiling (addendum Erratum 4: all-1.0 survival is a
                              kill signal, not a pass).
  4. drift_readout         -- per-lineage prototype drift between two snapshots
                              (the measurable form of "value drift = mode drift",
                              once content-matched lineages exist -- Erratum 3).

Run `python value_telemetry.py` for the self-test.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "lineage_summary",
    "anchor_coordinate_spread",
    "renewal_survival_summary",
    "drift_readout",
    "snapshot_from_mode_bank",
    "VALUE_STATUS_THRESHOLD",
]

#: anchor_fraction above which a mode counts as value-status (matches the
#: council's self_model / interiority_monitor gate).
VALUE_STATUS_THRESHOLD = 0.2


# ----------------------------------------------------------------------
# 1. Lineage / provenance summary
# ----------------------------------------------------------------------
def lineage_summary(anchor_fracs: Sequence[float], cognitive_fracs: Sequence[float]) -> dict:
    """Provenance make-up of the mode bank."""
    a = np.asarray(anchor_fracs, dtype=float)
    c = np.asarray(cognitive_fracs, dtype=float)
    n = int(a.size)
    return {
        "n_modes": n,
        "mean_anchor_fraction": float(a.mean()) if n else 0.0,
        "mean_cognitive_fraction": float(c.mean()) if n else 0.0,
        "n_value_status": int(np.sum(a > VALUE_STATUS_THRESHOLD)) if n else 0,
        "value_status_fraction": float(np.mean(a > VALUE_STATUS_THRESHOLD)) if n else 0.0,
    }


# ----------------------------------------------------------------------
# 2. Anti-collapse spread in ANCHOR coordinates
# ----------------------------------------------------------------------
def anchor_coordinate_spread(
    frames: np.ndarray,
    anchor_basis: Optional[np.ndarray] = None,
) -> dict:
    """Mean pairwise cosine distance of committee frames.

    If ``anchor_basis`` (rows = anchor-label directions) is given, frames are
    projected onto it FIRST, so diversity is measured in the anchor's
    coordinates rather than the committee's own -- the trap v18 §5 names. With
    no basis it falls back to raw-coordinate spread and says so.
    """
    F = np.asarray(frames, dtype=float)
    if F.ndim != 2 or F.shape[0] < 2:
        return {"spread": 0.0, "coords": "insufficient_frames", "collapsed": True}
    coords = "raw"
    if anchor_basis is not None:
        A = np.asarray(anchor_basis, dtype=float)
        F = F @ A.T
        coords = "anchor"
    norm = np.linalg.norm(F, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    U = F / norm
    sim = U @ U.T
    n = U.shape[0]
    off = sim[~np.eye(n, dtype=bool)]
    spread = float(np.mean(1.0 - off))
    return {"spread": spread, "coords": coords, "collapsed": spread < 0.05}


# ----------------------------------------------------------------------
# 3. Renewal survival distribution
# ----------------------------------------------------------------------
def renewal_survival_summary(
    survival_scores: Sequence[float],
    null_floor: float = 0.0,
    degenerate: bool = False,
) -> dict:
    """Cross-renewal survival, with the degenerate-ceiling kill flag.

    ``degenerate`` should carry through from ``RenewalController.RenewalReport``.
    We also independently flag an all-at-ceiling distribution, because that is
    the "nothing actually moved" signature the addendum warns reads as a false
    pass.
    """
    s = np.asarray(survival_scores, dtype=float)
    n = int(s.size)
    at_ceiling = bool(n and np.all(s >= 0.999))
    return {
        "n": n,
        "mean": float(s.mean()) if n else 0.0,
        "min": float(s.min()) if n else 0.0,
        "max": float(s.max()) if n else 0.0,
        "null_floor": float(null_floor),
        "mean_excess_over_null": float(s.mean() - null_floor) if n else 0.0,
        "degenerate": bool(degenerate or at_ceiling),
    }


# ----------------------------------------------------------------------
# 4. Prototype drift between two snapshots (value drift = mode drift)
# ----------------------------------------------------------------------
def drift_readout(
    prev_prototypes: Dict[str, Sequence[float]],
    curr_prototypes: Dict[str, Sequence[float]],
) -> dict:
    """Per-lineage cosine drift between two mode-bank snapshots.

    Only lineages present in BOTH snapshots are scored (content-matched
    lineages, per addendum Erratum 3). Drift is 1 - cos, in [0, 2].
    """
    drifts: Dict[str, float] = {}
    for mid, prev in prev_prototypes.items():
        if mid not in curr_prototypes:
            continue
        p = np.asarray(prev, dtype=float)
        c = np.asarray(curr_prototypes[mid], dtype=float)
        denom = (np.linalg.norm(p) * np.linalg.norm(c)) or 1.0
        drifts[mid] = float(1.0 - float(p @ c) / denom)
    vals = np.asarray(list(drifts.values()), dtype=float)
    return {
        "per_lineage": drifts,
        "n_tracked": int(vals.size),
        "mean_drift": float(vals.mean()) if vals.size else 0.0,
        "max_drift": float(vals.max()) if vals.size else 0.0,
    }


# ----------------------------------------------------------------------
# Live-bank adapter (torch optional; only used when a bank is present)
# ----------------------------------------------------------------------
def snapshot_from_mode_bank(mode_bank) -> dict:
    """Extract a numpy-only snapshot from a live ContentMatchedModeBank.

    Kept here (not in the council package) so the harness can call it without a
    hard torch dependency at import time.
    """
    protos: Dict[str, list] = {}
    anchor_fracs: List[float] = []
    cognitive_fracs: List[float] = []
    survival: List[float] = []
    for slot in getattr(mode_bank, "slots", []):
        vec = slot.prototype
        protos[slot.lineage_id] = [float(x) for x in vec.detach().cpu().numpy().ravel()]
        anchor_fracs.append(float(slot.anchor_fraction))
        cognitive_fracs.append(float(slot.cognitive_fraction))
        survival.append(float(slot.renewal_survival_count))
    return {
        "prototypes": protos,
        "anchor_fractions": anchor_fracs,
        "cognitive_fractions": cognitive_fracs,
        "renewal_survival_counts": survival,
    }


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)

    ls = lineage_summary([0.7, 0.1, 0.4, 0.0], [0.3, 0.9, 0.6, 1.0])
    assert ls["n_modes"] == 4
    assert ls["n_value_status"] == 2  # 0.7 and 0.4 exceed 0.2
    print("lineage_summary:", ls)

    # Diverse frames -> healthy spread; identical frames -> collapsed.
    diverse = rng.standard_normal((6, 8))
    coll = np.ones((6, 8))
    sd = anchor_coordinate_spread(diverse)
    sc = anchor_coordinate_spread(coll)
    assert not sd["collapsed"] and sc["collapsed"]
    # Anchor-basis projection path runs and is labelled.
    proj = anchor_coordinate_spread(diverse, anchor_basis=rng.standard_normal((3, 8)))
    assert proj["coords"] == "anchor"
    print("spread diverse/collapsed:", round(sd["spread"], 3), round(sc["spread"], 3))

    rs = renewal_survival_summary([1.0, 1.0, 1.0], null_floor=0.9, degenerate=False)
    assert rs["degenerate"], "all-at-ceiling survival must flag degenerate"
    rs2 = renewal_survival_summary([0.9, 0.4, 0.7], null_floor=0.3)
    assert not rs2["degenerate"]
    print("renewal degenerate flag ok")

    a = {"m1": [1.0, 0.0], "m2": [0.0, 1.0]}
    b = {"m1": [0.99, 0.14], "m2": [0.0, 1.0]}
    dr = drift_readout(a, b)
    assert dr["n_tracked"] == 2 and dr["max_drift"] > 0
    print("drift_readout:", {k: round(v, 4) for k, v in dr["per_lineage"].items()})

    print("\nvalue_telemetry self-test passed.")


if __name__ == "__main__":
    _selftest()
