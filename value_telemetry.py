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
    "operator_coordinates",
    "stage_c_convergence",
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
# 5. Stage-C bridge: project mode-bank prototypes onto the operator modes
# ----------------------------------------------------------------------
def operator_coordinates(
    prototypes: Dict[str, Sequence[float]],
    readout_C: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Express each mode-bank prototype in the SSM operator's mode basis.

    ``readout_C`` is the layer's [d_model, d_state] readout matrix from
    ``SpectralSSMModel.operator_readout``. The coordinate of a prototype on
    operator mode n is its normalized alignment with column n of C -- i.e. how
    strongly that value direction rides operator resonance n. Returns, per
    lineage, a [d_state] non-negative coordinate vector summing to 1 (a
    distribution over operator modes).

    This is the literal connection between the text-vector mode bank and the
    operator spectrum. It does NOT by itself establish the one-model identity
    (that is what ``stage_c_convergence`` tests, against a null).
    """
    C = np.asarray(readout_C, dtype=float)                 # [d_model, d_state]
    col_norm = np.linalg.norm(C, axis=0, keepdims=True)
    col_norm[col_norm == 0] = 1.0
    Cn = C / col_norm
    coords: Dict[str, np.ndarray] = {}
    for mid, vec in prototypes.items():
        p = np.asarray(vec, dtype=float)
        if p.shape[0] != Cn.shape[0]:
            raise ValueError(f"prototype dim {p.shape[0]} != d_model {Cn.shape[0]}")
        align = np.abs(p @ Cn)                              # [d_state]
        s = align.sum()
        coords[mid] = align / s if s > 0 else align
        # Mode-bank prototypes are l2-normalized already; alignment magnitude
        # is what carries the "which resonance" signal.
    return coords


def _participation_ratio(p: np.ndarray) -> float:
    """Effective number of modes a coordinate distribution occupies (1..d_state)."""
    p = np.asarray(p, dtype=float)
    denom = float(np.sum(p ** 2))
    return float(1.0 / denom) if denom > 0 else 0.0


def stage_c_convergence(
    coords_by_id: Dict[str, np.ndarray],
    channel_by_id: Dict[str, str],
    n_shuffle: int = 200,
    seed: int = 0,
) -> dict:
    """Do VALUE (anchored) modes occupy a different operator-mode subspace
    than COGNITIVE modes -- beyond a shuffled-label null?

    Staked prediction (addendum Erratum 5 / Question B H0'): NO unconfounded
    separation at this scale -- the operator coordinate carries no value/
    knowledge class signal, so ``separation`` should sit inside the null band
    and ``p_value`` should be non-significant. That NEGATIVE is the predicted,
    first-class outcome; a positive would be the surprise that starts to earn
    the one-model identity. Either way this is the Stage-C telemetry that must
    exist before v18 Section 9 leaves [SPECULATIVE].

    Separation statistic = 1 - cosine(mean value profile, mean cognitive
    profile): 0 = identical subspace, ->1 = disjoint. Compared against the
    distribution under random re-assignment of the same channel labels.
    """
    rng = np.random.default_rng(seed)
    ids = [m for m in coords_by_id if m in channel_by_id]
    if len(ids) < 4:
        return {"separation": 0.0, "p_value": 1.0, "n": len(ids),
                "verdict": "insufficient_modes", "null_mean": 0.0}

    mats = np.stack([np.asarray(coords_by_id[m], dtype=float) for m in ids])
    is_anchored = np.array([channel_by_id[m] in ("human_anchor", "audit") for m in ids])
    if is_anchored.sum() < 2 or (~is_anchored).sum() < 2:
        return {"separation": 0.0, "p_value": 1.0, "n": len(ids),
                "verdict": "one_channel_only", "null_mean": 0.0}

    def _sep(mask: np.ndarray) -> float:
        a = mats[mask].mean(axis=0)
        b = mats[~mask].mean(axis=0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(1.0 - float(a @ b) / (na * nb))

    observed = _sep(is_anchored)
    k = int(is_anchored.sum())
    null = np.empty(n_shuffle, dtype=float)
    for i in range(n_shuffle):
        perm = np.zeros(len(ids), dtype=bool)
        perm[rng.permutation(len(ids))[:k]] = True
        null[i] = _sep(perm)
    null_mean = float(null.mean())
    p_value = float((np.sum(null >= observed) + 1) / (n_shuffle + 1))
    significant = p_value < 0.05 and observed > null_mean

    return {
        "n": len(ids),
        "n_value": int(is_anchored.sum()),
        "n_cognitive": int((~is_anchored).sum()),
        "separation": round(observed, 4),
        "null_mean": round(null_mean, 4),
        "p_value": round(p_value, 4),
        "significant": bool(significant),
        "mean_value_participation": round(
            float(np.mean([_participation_ratio(mats[i]) for i in range(len(ids)) if is_anchored[i]])), 3),
        "mean_cognitive_participation": round(
            float(np.mean([_participation_ratio(mats[i]) for i in range(len(ids)) if not is_anchored[i]])), 3),
        "verdict": ("SEPARATION_DETECTED_investigate" if significant
                    else "NULL_as_predicted_operator_carries_no_class_signal"),
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
    channel: Dict[str, str] = {}
    anchor_fracs: List[float] = []
    cognitive_fracs: List[float] = []
    survival: List[float] = []
    for slot in getattr(mode_bank, "slots", []):
        vec = slot.prototype
        protos[slot.lineage_id] = [float(x) for x in vec.detach().cpu().numpy().ravel()]
        # channel type ("anchored"/"self") -> a value vs cognitive label for
        # the Stage-C separation test.
        ctype = slot.metadata.get("channel_type", "self")
        channel[slot.lineage_id] = "human_anchor" if ctype == "anchored" else "cognitive"
        anchor_fracs.append(float(slot.anchor_fraction))
        cognitive_fracs.append(float(slot.cognitive_fraction))
        survival.append(float(slot.renewal_survival_count))
    return {
        "prototypes": protos,
        "channel_by_id": channel,
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

    # Stage-C: project prototypes onto a toy operator readout, run the null.
    d_model, d_state = 8, 6
    C = rng.standard_normal((d_model, d_state))
    protos = {f"m{i}": list(rng.standard_normal(d_model)) for i in range(8)}
    channels = {f"m{i}": ("human_anchor" if i % 2 == 0 else "cognitive") for i in range(8)}
    coords = operator_coordinates(protos, C)
    assert all(abs(v.sum() - 1.0) < 1e-6 for v in coords.values())
    sc = stage_c_convergence(coords, channels, n_shuffle=200)
    # Random prototypes with random channel labels: must land in the null band.
    assert not sc["significant"], "random labels must not separate"
    print("stage_c (random null):", {k: sc[k] for k in ("separation", "null_mean", "p_value", "verdict")})

    print("\nvalue_telemetry self-test passed.")


if __name__ == "__main__":
    _selftest()
