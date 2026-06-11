"""
self_adjuster.py
================

Empirically-driven self-adjustment for the reasoning scaffold.

The system observes its own performance through the episodic store,
compares it against its current configuration, and adjusts parameters
that are demonstrably miscalibrated.

Three adjustment types, in order of safety:

  ThresholdCalibrator   — adjusts the consolidation and retrieval
                          thresholds (obligation merge threshold,
                          technique merge threshold, GUE min_lift)
                          based on whether current thresholds are
                          actually improving downstream verification.

  ScoringWeightLearner  — learns which branch scoring weights
                          (similarity vs coverage vs cohesion) produced
                          the most verified runs, and updates the
                          composer's weights accordingly.

  VocabularyExpander    — discovers structural patterns in the
                          accumulated relational graphs that don't match
                          any existing abstract structure, and proposes
                          new entries to the cross-domain vocabulary.

All adjustments go through the same cycle:
  1. Propose (based on empirical analysis)
  2. Sandbox (record current state, apply tentatively)
  3. Measure (wait N runs, check performance delta)
  4. Adopt or rollback (based on measured improvement)
  5. Log (record what was tried and whether it worked)

The log feeds back into the introspection module — the system knows
what it has tried to change about itself and whether it worked.

SAFETY NOTE: self-modification that bypasses measurement is dangerous.
Every adjustment in this module requires a positive performance delta
before being permanently adopted. The sandbox and rollback mechanisms
are not optional.
"""

import os
import re
import json
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any


# ============================================================
# Adjustment record
# ============================================================

@dataclass
class AdjustmentRecord:
    """Logs one proposed and measured adjustment."""

    timestamp:       float
    adjuster_type:   str         # "threshold" | "weight" | "vocabulary"
    parameter_name:  str
    old_value:       Any
    new_value:       Any
    rationale:       str
    status:          str  = "proposed"  # "proposed" | "sandboxed" | "adopted" | "rejected"
    performance_before: Optional[float] = None
    performance_after:  Optional[float] = None
    delta:              Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "timestamp":          self.timestamp,
            "adjuster_type":      self.adjuster_type,
            "parameter_name":     self.parameter_name,
            "old_value":          self.old_value,
            "new_value":          self.new_value,
            "rationale":          self.rationale,
            "status":             self.status,
            "performance_before": self.performance_before,
            "performance_after":  self.performance_after,
            "delta":              self.delta,
        }


# ============================================================
# Performance measurement
# ============================================================

def _measure_performance(episodic_store, window: int = 20) -> dict:
    """
    Compute a composite performance score from recent episodic records.

    Components:
      - verification_rate:   fraction of recent runs that verified
      - structural_fraction: average structural fraction (LLM reliance)
      - gue_score:           average GUE score across recent runs
    """
    recent = episodic_store.query_recent(n=window)
    if not recent:
        return {"composite": 0.0, "n_runs": 0,
                "verification_rate": 0.0, "structural_fraction": 0.0}

    n = len(recent)
    ver_rate   = sum(1 for r in recent if r.verified) / n
    struct_avg = sum(r.structural_fraction for r in recent) / n
    gue_scores = [r.gue_score for r in recent if r.gue_score is not None]
    gue_avg    = sum(gue_scores) / len(gue_scores) if gue_scores else 0.5

    # Composite: weighted sum
    composite = 0.5 * ver_rate + 0.3 * struct_avg + 0.2 * gue_avg

    return {
        "composite":          round(composite, 4),
        "n_runs":             n,
        "verification_rate":  round(ver_rate, 4),
        "structural_fraction": round(struct_avg, 4),
        "gue_avg":            round(gue_avg, 4),
    }


# ============================================================
# ThresholdCalibrator
# ============================================================

class ThresholdCalibrator:
    """
    Adjusts consolidation and retrieval thresholds based on whether
    current values are producing good outcomes.

    Logic: if the obligation store is growing fast but verification
    rate is not improving, the merge threshold may be too conservative
    (not merging enough). If techniques are being merged too aggressively,
    performance drops. The calibrator finds the empirical sweet spot.
    """

    # Thresholds and their valid ranges
    ADJUSTABLE = {
        "obligation_sim_threshold": (0.65, 0.95, 0.03),  # (min, max, step)
        "technique_sim_threshold":  (0.75, 0.97, 0.03),
        "gue_min_lift":             (1.05, 1.50, 0.05),
        "graph_min_lift":           (1.05, 1.40, 0.05),
    }

    def __init__(self, consolidator, episodic_store):
        self.consolidator = consolidator
        self.episodic     = episodic_store

    def propose_adjustments(self) -> list[AdjustmentRecord]:
        """
        Analyse current performance and propose threshold changes.
        """
        perf    = _measure_performance(self.episodic)
        records = []

        if perf["n_runs"] < 10:
            return []   # not enough data

        # Obligation merge threshold:
        # If verification rate is low AND obligation store is large →
        # try merging more aggressively (lower threshold)
        try:
            n_obls = len(self.episodic.obligations._data.get("obligations", {}))
        except Exception:
            n_obls = 0

        current_obl = self.consolidator.obl_threshold
        if perf["verification_rate"] < 0.5 and n_obls > 30:
            new_val = max(
                self.ADJUSTABLE["obligation_sim_threshold"][0],
                current_obl - self.ADJUSTABLE["obligation_sim_threshold"][2],
            )
            if new_val != current_obl:
                records.append(AdjustmentRecord(
                    timestamp=time.time(),
                    adjuster_type="threshold",
                    parameter_name="obligation_sim_threshold",
                    old_value=current_obl,
                    new_value=new_val,
                    rationale=(
                        f"Verification rate {perf['verification_rate']:.0%} is low "
                        f"and obligation store has {n_obls} entries. "
                        f"Merging more aggressively may reduce noise."
                    ),
                    performance_before=perf["composite"],
                ))

        # Technique merge threshold:
        # If structural fraction is low → techniques may be too similar,
        # not providing diverse compositional options
        current_tech = self.consolidator.tech_threshold
        if perf["structural_fraction"] < 0.3 and perf["n_runs"] >= 15:
            new_val = min(
                self.ADJUSTABLE["technique_sim_threshold"][1],
                current_tech + self.ADJUSTABLE["technique_sim_threshold"][2],
            )
            if new_val != current_tech:
                records.append(AdjustmentRecord(
                    timestamp=time.time(),
                    adjuster_type="threshold",
                    parameter_name="technique_sim_threshold",
                    old_value=current_tech,
                    new_value=new_val,
                    rationale=(
                        f"Structural fraction {perf['structural_fraction']:.0%} is low. "
                        f"Raising technique merge threshold preserves more distinct "
                        f"techniques, giving the sampler more compositional diversity."
                    ),
                    performance_before=perf["composite"],
                ))

        return records

    def apply(self, record: AdjustmentRecord) -> None:
        """Apply a threshold adjustment."""
        name = record.parameter_name
        val  = record.new_value
        if name == "obligation_sim_threshold":
            self.consolidator.obl_threshold = val
        elif name == "technique_sim_threshold":
            self.consolidator.tech_threshold = val
        elif name in ("gue_min_lift", "graph_min_lift"):
            self.consolidator.graph_min_lift = val
        record.status = "sandboxed"


# ============================================================
# ScoringWeightLearner
# ============================================================

class ScoringWeightLearner:
    """
    Learns which branch scoring weights produced the most verified runs.

    The TechniqueComposer uses:
      score = w_seed * seed_score + w_coverage * coverage + w_cohesion * cohesion

    Current weights: (0.45, 0.35, 0.20)
    This learner estimates better weights from the episodic store.
    """

    DEFAULT_WEIGHTS = (0.45, 0.35, 0.20)
    WEIGHT_STEP     = 0.05
    MIN_WEIGHT      = 0.10

    def __init__(self, episodic_store, embedded_composer):
        self.episodic  = episodic_store
        self.composer  = embedded_composer
        self._current  = list(self.DEFAULT_WEIGHTS)

    def estimate_better_weights(self) -> Optional[tuple]:
        """
        Simple gradient estimate: for each weight dimension, perturb
        and estimate the direction that would have improved verification rate.

        Returns new (w_seed, w_coverage, w_cohesion) if improvement found,
        else None.
        """
        recent = self.episodic.query_recent(n=30)
        if len(recent) < 10:
            return None

        verified   = [r for r in recent if r.verified]
        unverified = [r for r in recent if not r.verified]

        if not verified or not unverified:
            return None

        # For verified runs: their structural_fraction is a proxy for
        # how much the coverage/cohesion weights contributed.
        # High structural fraction + verified → coverage/cohesion weights are working.
        # Low structural fraction + verified → seed similarity is dominating.
        avg_struct_verified   = sum(r.structural_fraction for r in verified) / len(verified)
        avg_struct_unverified = sum(r.structural_fraction for r in unverified) / len(unverified)

        w = list(self._current)

        # If verified runs have higher structural fraction:
        # coverage and cohesion weights should increase
        if avg_struct_verified > avg_struct_unverified + 0.1:
            w[1] = min(0.70, w[1] + self.WEIGHT_STEP)
            w[2] = min(0.40, w[2] + self.WEIGHT_STEP)
            w[0] = max(self.MIN_WEIGHT, 1.0 - w[1] - w[2])

        # If verified runs have LOWER structural fraction:
        # seed similarity is more important
        elif avg_struct_unverified > avg_struct_verified + 0.1:
            w[0] = min(0.70, w[0] + self.WEIGHT_STEP)
            w[1] = max(self.MIN_WEIGHT, w[1] - self.WEIGHT_STEP / 2)
            w[2] = max(self.MIN_WEIGHT, 1.0 - w[0] - w[1])

        else:
            return None   # no clear signal

        # Normalise
        total = sum(w)
        w = [wi / total for wi in w]

        if w == self._current:
            return None

        return tuple(round(wi, 3) for wi in w)

    def propose_adjustment(self) -> Optional[AdjustmentRecord]:
        new_weights = self.estimate_better_weights()
        if new_weights is None:
            return None

        return AdjustmentRecord(
            timestamp=time.time(),
            adjuster_type="weight",
            parameter_name="branch_scoring_weights",
            old_value=tuple(self._current),
            new_value=new_weights,
            rationale=(
                f"Analysis of {self.episodic.n_records} episodic records "
                f"suggests shifting scoring weights toward "
                f"{'coverage/cohesion' if new_weights[1] > self._current[1] else 'seed similarity'}."
            ),
            performance_before=_measure_performance(self.episodic)["composite"],
        )

    def apply(self, record: AdjustmentRecord) -> None:
        """Apply weight adjustment to the embedded composer."""
        new_w = record.new_value
        self._current = list(new_w)
        # Patch the composer's scoring (if the composer exposes weights)
        # The TechniqueComposer.compose_for_problem uses hardcoded weights;
        # we store the learned weights here and the pipeline can read them.
        record.status = "sandboxed"

    @property
    def current_weights(self) -> tuple:
        return tuple(self._current)


# ============================================================
# VocabularyExpander
# ============================================================

class VocabularyExpander:
    """
    Discovers new abstract structural patterns in the accumulated
    relational graphs and proposes additions to the cross-domain vocabulary.

    The 13 hand-curated structures in cross_domain_composer.py are a
    starting set. As the system accumulates relational graphs from
    verified techniques across multiple domains, it may encounter
    structural patterns that don't fit any existing category.

    This module clusters relational graph fingerprints, identifies
    clusters with no matching abstract structure, and proposes new
    vocabulary entries.
    """

    def __init__(self, embedded_composer, cross_domain_composer=None):
        self.composer      = embedded_composer
        self.cross_domain  = cross_domain_composer
        self._proposed_structures: list[dict] = []

    def find_unmatched_patterns(
        self, min_cluster_size: int = 3, similarity_threshold: float = 0.7,
    ) -> list[dict]:
        """
        Find technique clusters whose abstract structure doesn't match
        any existing vocabulary entry.

        Returns list of {cluster_techniques, proposed_name,
        trigger_words, description}.
        """
        techs  = self.composer.library._data.get("techniques", {})
        if len(techs) < min_cluster_size * 2:
            return []

        # Get embeddings for all techniques
        embeddings: dict[str, np.ndarray] = {}
        for tid, t in techs.items():
            try:
                e = self.composer.embedder.embed(tid, t)
                embeddings[tid] = e
            except Exception:
                pass

        if len(embeddings) < min_cluster_size * 2:
            return []

        # Cluster by cosine similarity
        tids   = list(embeddings.keys())
        E      = np.stack([embeddings[tid] for tid in tids])
        sim    = E @ E.T   # cosine similarity matrix (embeddings are normalised)

        # Simple greedy clustering
        clusters: list[list[str]] = []
        clustered: set[str] = set()
        for i, tid in enumerate(tids):
            if tid in clustered:
                continue
            cluster = [tid]
            for j, tid2 in enumerate(tids):
                if tid2 in clustered or j == i:
                    continue
                if sim[i, j] >= similarity_threshold:
                    cluster.append(tid2)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
                clustered.update(cluster)

        if not clusters:
            return []

        # Check each cluster against existing abstract structures
        proposals: list[dict] = []
        existing_keywords: set[str] = set()

        if self.cross_domain is not None:
            try:
                from cross_domain_composer import ABSTRACT_STRUCTURES
                for spec in ABSTRACT_STRUCTURES.values():
                    existing_keywords.update(spec.get("triggers", set()))
            except Exception:
                pass

        for cluster in clusters:
            # Extract distinctive words from this cluster's techniques
            cluster_words: dict[str, int] = {}
            for tid in cluster:
                t = techs.get(tid, {})
                text = " ".join([t.get("name",""), t.get("description",""),
                                  t.get("when_to_use","")])
                for word in re.findall(r"[a-z]{4,}", text.lower()):
                    cluster_words[word] = cluster_words.get(word, 0) + 1

            # Words that appear in most cluster techniques but not in
            # existing vocabulary are potential new trigger keywords
            distinctive = [
                w for w, count in cluster_words.items()
                if count >= len(cluster) * 0.6
                and w not in existing_keywords
            ][:10]

            if not distinctive:
                continue

            # Propose a name from the most common distinctive words
            top_words = sorted(distinctive,
                               key=lambda w: cluster_words[w], reverse=True)[:3]
            proposed_name = "_".join(top_words[:2])

            # Get example technique names from the cluster
            example_names = [
                techs[tid].get("name", "")[:50]
                for tid in cluster[:3]
            ]

            proposals.append({
                "proposed_name":    proposed_name,
                "trigger_words":    distinctive,
                "cluster_size":     len(cluster),
                "example_techniques": example_names,
                "description":      (
                    f"Pattern discovered in {len(cluster)} co-occurring techniques. "
                    f"Characteristic vocabulary: {', '.join(top_words)}."
                ),
            })

        self._proposed_structures.extend(proposals)
        return proposals

    def generate_addition_record(
        self, proposal: dict,
    ) -> AdjustmentRecord:
        return AdjustmentRecord(
            timestamp=time.time(),
            adjuster_type="vocabulary",
            parameter_name="abstract_structures",
            old_value="existing_13",
            new_value=proposal["proposed_name"],
            rationale=(
                f"Cluster of {proposal['cluster_size']} techniques with "
                f"shared structural pattern not in current vocabulary. "
                f"Key terms: {', '.join(proposal['trigger_words'][:5])}. "
                f"Examples: {', '.join(proposal['example_techniques'][:2])}."
            ),
        )


# ============================================================
# SelfAdjuster — top-level coordinator
# ============================================================

class SelfAdjuster:
    """
    Coordinates all self-adjustment types with sandbox, measure,
    adopt/reject, and log.

    Usage:
        adjuster = SelfAdjuster(
            consolidator=consolidator,
            episodic_store=episodic,
            embedded_composer=composer,
            log_path="./self_adjustment_log.json",
        )

        # Run a full adjustment cycle
        report = adjuster.run_cycle()

        # After N more runs, evaluate sandboxed adjustments
        adjuster.evaluate_sandboxed()
    """

    # Minimum improvement delta to adopt an adjustment
    MIN_IMPROVEMENT_DELTA = 0.02

    # How many runs to wait before evaluating a sandboxed adjustment
    SANDBOX_WINDOW = 10

    def __init__(
        self,
        consolidator,
        episodic_store,
        embedded_composer,
        cross_domain_composer  = None,
        log_path:        str   = "./self_adjustment_log.json",
        verbose:         bool  = True,
        human_approval:  bool  = True,   # require human approval for vocab changes
    ):
        self.consolidator   = consolidator
        self.episodic       = episodic_store
        self.composer       = embedded_composer
        self.verbose        = verbose
        self.human_approval = human_approval
        self.log_path       = log_path

        self.threshold_calibrator = ThresholdCalibrator(consolidator, episodic_store)
        self.weight_learner       = ScoringWeightLearner(episodic_store, embedded_composer)
        self.vocab_expander       = VocabularyExpander(embedded_composer, cross_domain_composer)

        self._log: list[dict]              = self._load_log()
        self._sandboxed: list[AdjustmentRecord] = []
        self._runs_since_sandbox: int      = 0

    # ---- Main cycle ----

    def run_cycle(self) -> dict:
        """
        Propose, sandbox, and (if safe) apply adjustments.
        Returns a summary of what was proposed and what was applied.
        """
        if self.verbose:
            print("\n[self_adjuster] Running adjustment cycle...")

        perf = _measure_performance(self.episodic)
        if perf["n_runs"] < 10:
            if self.verbose:
                print(f"  Insufficient data ({perf['n_runs']} runs). "
                      f"Need ≥ 10.")
            return {"status": "insufficient_data"}

        proposed: list[AdjustmentRecord] = []
        applied:  list[AdjustmentRecord] = []
        pending:  list[AdjustmentRecord] = []

        # 1. Threshold proposals
        for rec in self.threshold_calibrator.propose_adjustments():
            proposed.append(rec)
            # Threshold changes are low-risk — apply immediately to sandbox
            self.threshold_calibrator.apply(rec)
            self._sandboxed.append(rec)
            pending.append(rec)
            if self.verbose:
                print(f"  [threshold] {rec.parameter_name}: "
                      f"{rec.old_value} → {rec.new_value}")
                print(f"    Rationale: {rec.rationale[:80]}")

        # 2. Weight proposals
        wrec = self.weight_learner.propose_adjustment()
        if wrec:
            proposed.append(wrec)
            self.weight_learner.apply(wrec)
            self._sandboxed.append(wrec)
            pending.append(wrec)
            if self.verbose:
                print(f"  [weights] branch_scoring: "
                      f"{wrec.old_value} → {wrec.new_value}")
                print(f"    Rationale: {wrec.rationale[:80]}")

        # 3. Vocabulary proposals — always need human approval
        vocab_proposals = self.vocab_expander.find_unmatched_patterns()
        for vp in vocab_proposals[:2]:   # cap at 2 per cycle
            vrec = self.vocab_expander.generate_addition_record(vp)
            proposed.append(vrec)
            if self.human_approval:
                vrec.status = "pending_approval"
                if self.verbose:
                    print(f"  [vocabulary] Proposed new structure: "
                          f"'{vrec.new_value}' (needs human approval)")
                    print(f"    {vrec.rationale[:120]}")
            else:
                # Auto-apply only if human_approval=False
                vrec.status = "sandboxed"
                self._sandboxed.append(vrec)
                pending.append(vrec)

        # Log all proposals
        for rec in proposed:
            self._log.append(rec.to_dict())
        self._save_log()

        summary = {
            "n_proposed":   len(proposed),
            "n_sandboxed":  len(pending),
            "n_vocab":      len(vocab_proposals),
            "performance_before": perf,
            "current_weights": self.weight_learner.current_weights,
            "current_obl_threshold": self.consolidator.obl_threshold,
        }

        if self.verbose and not proposed:
            print("  No adjustments needed at current performance level.")

        return summary

    # ---- Evaluation of sandboxed adjustments ----

    def notify_run_completed(self) -> None:
        """Call after each pipeline run to track the sandbox window."""
        self._runs_since_sandbox += 1

    def evaluate_sandboxed(self) -> list[AdjustmentRecord]:
        """
        After SANDBOX_WINDOW runs, evaluate all sandboxed adjustments.
        Adopt those that improved performance; roll back the rest.
        """
        if self._runs_since_sandbox < self.SANDBOX_WINDOW:
            return []
        if not self._sandboxed:
            return []

        perf_now = _measure_performance(self.episodic)
        evaluated: list[AdjustmentRecord] = []

        for rec in self._sandboxed:
            rec.performance_after = perf_now["composite"]
            rec.delta = (
                (rec.performance_after - rec.performance_before)
                if rec.performance_before is not None else None
            )

            if rec.delta is not None and rec.delta >= self.MIN_IMPROVEMENT_DELTA:
                rec.status = "adopted"
                if self.verbose:
                    print(f"  [adopt] {rec.parameter_name}: "
                          f"{rec.old_value} → {rec.new_value} "
                          f"(Δ={rec.delta:+.4f})")
            else:
                rec.status = "rejected"
                self._rollback(rec)
                if self.verbose:
                    print(f"  [reject] {rec.parameter_name}: "
                          f"rolled back (Δ={rec.delta:+.4f if rec.delta else 'n/a'})")

            evaluated.append(rec)

        self._sandboxed = [r for r in self._sandboxed
                           if r.status == "sandboxed"]
        self._runs_since_sandbox = 0

        # Update log
        for rec in evaluated:
            for entry in self._log:
                if (entry.get("parameter_name") == rec.parameter_name
                        and entry.get("timestamp") == rec.timestamp):
                    entry.update(rec.to_dict())
                    break
            else:
                self._log.append(rec.to_dict())
        self._save_log()

        return evaluated

    def _rollback(self, record: AdjustmentRecord) -> None:
        """Revert an adjustment to its old value."""
        name = record.parameter_name
        val  = record.old_value
        if name == "obligation_sim_threshold":
            self.consolidator.obl_threshold = val
        elif name == "technique_sim_threshold":
            self.consolidator.tech_threshold = val
        elif name in ("gue_min_lift", "graph_min_lift"):
            self.consolidator.graph_min_lift = val
        elif name == "branch_scoring_weights":
            self.weight_learner._current = list(val)

    # ---- Inspection ----

    def adjustment_history(self) -> list[dict]:
        return list(self._log)

    def n_adopted(self) -> int:
        return sum(1 for r in self._log if r.get("status") == "adopted")

    def n_rejected(self) -> int:
        return sum(1 for r in self._log if r.get("status") == "rejected")

    # ---- Persistence ----

    def _load_log(self) -> list[dict]:
        if not os.path.exists(self.log_path):
            return []
        try:
            with open(self.log_path) as f:
                return json.load(f).get("adjustments", [])
        except (json.JSONDecodeError, OSError):
            return []

    def _save_log(self) -> None:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump({"adjustments": self._log}, f, indent=2, default=str)
