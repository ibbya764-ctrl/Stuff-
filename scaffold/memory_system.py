"""
memory_system.py
================

Multi-timescale memory architecture for the reasoning scaffold.

Four components:

  EpisodicStore        — compact records of successful reasoning chains.
                         The system's autobiography. Persists every verified
                         run as a compact record: what was asked, what method
                         won, what was verified, what obligations were
                         discharged, what the structural fraction was.

  MemoryConsolidator   — moves learning from short-term to long-term.
                         Finds near-duplicate obligations and merges them.
                         Finds near-duplicate techniques and merges them.
                         Prunes directed graph edges with no predictive value.
                         Tracks whether the system is getting more efficient.

  UnifiedRetrieval     — one interface across all stores simultaneously.
                         Given a question, returns: relevant techniques,
                         similar past runs, related obligations, and the
                         historically successful sequence to follow.
                         This is what the structural branch sampler calls
                         instead of querying each store separately.

  MemorySystem         — top-level coordinator.
                         record_run() after each pipeline run.
                         consolidate() periodically (overnight/between sessions).
                         query() for unified retrieval.
                         health_report() for compression tracking.

The compression metric is the key diagnostic: storage bytes per verified
fact, tracked over time. When it decreases, the system is getting more
efficient per unit of knowledge learned. When it plateaus, the technique
library and obligation store have captured the useful structure and it's
time to consider distilling into a smaller fine-tuned model.
"""

import os
import re
import json
import math
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# EpisodicStore
# ============================================================

@dataclass
class EpisodicRecord:
    """One successful reasoning run, stored compactly."""
    run_id:               str
    timestamp:            float
    domain:               str
    question:             str
    method_name:          str
    technique_id:         str
    verified:             bool
    structural_fraction:  float
    obligations_discharged: int
    obligations_raised:   int
    gue_score:            Optional[float]
    key_steps:            list[str]   # first 3 steps (compressed)

    def to_dict(self) -> dict:
        return {
            "run_id":               self.run_id,
            "timestamp":            self.timestamp,
            "domain":               self.domain,
            "question":             self.question[:200],
            "method_name":          self.method_name,
            "technique_id":         self.technique_id,
            "verified":             self.verified,
            "structural_fraction":  self.structural_fraction,
            "obligations_discharged": self.obligations_discharged,
            "obligations_raised":   self.obligations_raised,
            "gue_score":            self.gue_score,
            "key_steps":            [s[:120] for s in self.key_steps[:3]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodicRecord":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class EpisodicStore:
    """
    Compact episodic memory of successful reasoning runs.
    Stores the trajectory — not just the extracted technique — so the
    full context of each discovery is preserved.
    """

    def __init__(self, path: str = "./episodic_store.json"):
        self.path = path
        self._records: dict[str, EpisodicRecord] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for d in data.get("records", []):
                try:
                    r = EpisodicRecord.from_dict(d)
                    self._records[r.run_id] = r
                except Exception:
                    pass
        except (json.JSONDecodeError, OSError):
            pass

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "version": 1,
                "records": [r.to_dict() for r in self._records.values()],
            }, f, indent=2)

    def record(
        self,
        pipeline_result: dict,
        gue_score:        Optional[float] = None,
    ) -> Optional[EpisodicRecord]:
        """
        Extract and store a compact episodic record from a pipeline result.
        Only records if a branch was selected (i.e. the run produced output).
        """
        selected = pipeline_result.get("selected_branch", {})
        if not selected:
            return None

        run_id = pipeline_result.get("run_id", f"run-{int(time.time())}")
        if run_id in self._records:
            return self._records[run_id]   # already recorded

        # Count discharged obligations
        resolution  = pipeline_result.get("resolution", {})
        obligations = resolution.get("obligations", []) or []
        discharged  = sum(1 for o in obligations
                          if o.get("status") == "discharged")
        raised      = len(obligations)

        record = EpisodicRecord(
            run_id=run_id,
            timestamp=time.time(),
            domain=pipeline_result.get("domain", ""),
            question=pipeline_result.get("question", ""),
            method_name=selected.get("method", ""),
            technique_id=selected.get("provenance", {}).get(
                "technique_id", ""
            ),
            verified=bool(
                selected.get("verification_report", {})
                .get("verdict", {})
                .get("verified")
            ),
            structural_fraction=float(
                selected.get("structural_fraction", 0.0)
            ),
            obligations_discharged=discharged,
            obligations_raised=raised,
            gue_score=gue_score,
            key_steps=list(selected.get("steps", []))[:3],
        )
        self._records[run_id] = record
        self.save()
        return record

    def query_recent(self, n: int = 20) -> list[EpisodicRecord]:
        return sorted(
            self._records.values(),
            key=lambda r: r.timestamp,
            reverse=True,
        )[:n]

    def query_domain(self, domain: str) -> list[EpisodicRecord]:
        return [r for r in self._records.values() if r.domain == domain]

    def query_similar(
        self, question: str, top_n: int = 5,
    ) -> list[tuple[EpisodicRecord, float]]:
        """
        Find past runs with questions semantically similar to the query.
        Uses word-overlap similarity (fast, no embedder required).
        """
        q_words = set(_tokenise(question))
        scored: list[tuple[EpisodicRecord, float]] = []
        for r in self._records.values():
            r_words = set(_tokenise(r.question))
            if not q_words or not r_words:
                continue
            overlap = len(q_words & r_words) / len(q_words | r_words)
            if overlap > 0.1:
                scored.append((r, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    @property
    def n_records(self) -> int:
        return len(self._records)

    def statistics(self) -> dict:
        if not self._records:
            return {"n_records": 0}
        verified = sum(1 for r in self._records.values() if r.verified)
        fracs    = [r.structural_fraction for r in self._records.values()]
        gue_scores = [r.gue_score for r in self._records.values()
                      if r.gue_score is not None]
        return {
            "n_records":          self.n_records,
            "verified_fraction":  verified / self.n_records,
            "avg_struct_fraction": sum(fracs) / len(fracs) if fracs else 0,
            "avg_gue_score":      sum(gue_scores) / len(gue_scores)
                                  if gue_scores else None,
            "domains":            list({r.domain for r in self._records.values()}),
        }


# ============================================================
# MemoryConsolidator
# ============================================================

@dataclass
class ConsolidationReport:
    obligations_merged:   int = 0
    techniques_merged:    int = 0
    graph_edges_pruned:   int = 0
    storage_bytes_before: int = 0
    storage_bytes_after:  int = 0
    compression_delta:    float = 0.0   # negative = got smaller (good)
    notes:                list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Consolidation Report:"]
        lines.append(f"  Obligations merged:  {self.obligations_merged}")
        lines.append(f"  Techniques merged:   {self.techniques_merged}")
        lines.append(f"  Graph edges pruned:  {self.graph_edges_pruned}")
        delta = self.storage_bytes_after - self.storage_bytes_before
        lines.append(
            f"  Storage delta:       {delta:+,} bytes "
            f"({'smaller ✓' if delta < 0 else 'larger'})"
        )
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)


def _tokenise(text: str) -> list[str]:
    STOP = {
        "the", "a", "an", "is", "are", "be", "to", "of", "and", "or",
        "for", "in", "on", "at", "by", "with", "from", "as", "this",
        "that", "it", "any", "all", "some", "when", "where",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", text.lower())
    return [t for t in tokens if t not in STOP and len(t) > 2]


def _tfidf_cosine(text_a: str, text_b: str, idf: Optional[dict] = None) -> float:
    """Fast TF-IDF cosine similarity between two texts."""
    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    vocab = set(tokens_a) | set(tokens_b)
    vec_a = {t: tokens_a.count(t) for t in vocab}
    vec_b = {t: tokens_b.count(t) for t in vocab}
    dot   = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vocab)
    na    = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    nb    = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    return dot / (na * nb + 1e-8)


class MemoryConsolidator:
    """
    Periodically cleans and compresses all memory stores.

    Run this between sessions (overnight is ideal). It:
      - Finds near-duplicate obligations and merges them
      - Finds near-duplicate techniques and merges them
      - Prunes directed graph edges that have no predictive value
      - Reports how much storage was saved
    """

    def __init__(
        self,
        obligation_sim_threshold: float = 0.82,
        technique_sim_threshold:  float = 0.88,
        graph_min_lift:           float = 1.15,
        graph_min_runs_before_prune: int = 15,
        verbose: bool = True,
    ):
        self.obl_threshold   = obligation_sim_threshold
        self.tech_threshold  = technique_sim_threshold
        self.graph_min_lift  = graph_min_lift
        self.graph_min_runs  = graph_min_runs_before_prune
        self.verbose         = verbose

    def consolidate(
        self,
        obligation_store,
        technique_library,
        directed_graph,
    ) -> ConsolidationReport:
        """
        Full consolidation pass. Returns a ConsolidationReport.
        """
        report = ConsolidationReport()
        report.storage_bytes_before = self._measure_storage(
            obligation_store, technique_library, directed_graph
        )

        if self.verbose:
            print("[consolidator] Starting consolidation pass...")

        # 1. Merge duplicate obligations
        n_obl = self._merge_obligations(obligation_store)
        report.obligations_merged = n_obl
        if self.verbose and n_obl > 0:
            print(f"  Merged {n_obl} duplicate obligations.")

        # 2. Merge duplicate techniques
        n_tech = self._merge_techniques(technique_library)
        report.techniques_merged = n_tech
        if self.verbose and n_tech > 0:
            print(f"  Merged {n_tech} near-duplicate techniques.")

        # 3. Prune weak directed graph edges
        n_pruned = self._prune_graph(directed_graph)
        report.graph_edges_pruned = n_pruned
        if self.verbose and n_pruned > 0:
            print(f"  Pruned {n_pruned} weak graph edges.")

        report.storage_bytes_after = self._measure_storage(
            obligation_store, technique_library, directed_graph
        )
        report.compression_delta = (
            (report.storage_bytes_after - report.storage_bytes_before)
            / max(1, report.storage_bytes_before)
        )

        if self.verbose:
            print(report.summary())

        return report

    # ---- Obligation merging ----

    def _merge_obligations(self, obligation_store) -> int:
        """
        Find pairs of obligations with text similarity > threshold
        and merge the lower-occurrence one into the higher-occurrence one.
        """
        obls = obligation_store._data.get("obligations", {})
        ids  = list(obls.keys())
        n    = len(ids)
        if n < 2:
            return 0

        merged = 0
        to_remove: set[str] = set()

        for i in range(n):
            if ids[i] in to_remove:
                continue
            for j in range(i + 1, n):
                if ids[j] in to_remove:
                    continue
                oa = obls[ids[i]]
                ob = obls[ids[j]]
                sim = _tfidf_cosine(oa.get("text", ""), ob.get("text", ""))
                if sim < self.obl_threshold:
                    continue

                # Merge: keep the one with higher occurrence, update its stats
                if oa.get("n_occurrences", 0) >= ob.get("n_occurrences", 0):
                    survivor, duplicate = ids[i], ids[j]
                else:
                    survivor, duplicate = ids[j], ids[i]

                # Transfer occurrence data from duplicate to survivor
                obls[survivor]["n_occurrences"] = (
                    obls[survivor].get("n_occurrences", 0)
                    + obls[duplicate].get("n_occurrences", 0)
                )
                for sb in obls[duplicate].get("source_branches", []):
                    if sb not in obls[survivor].get("source_branches", []):
                        obls[survivor].setdefault("source_branches", []).append(sb)
                for run in obls[duplicate].get("discharged_in_runs", []):
                    if run not in obls[survivor].get("discharged_in_runs", []):
                        obls[survivor].setdefault("discharged_in_runs", []).append(run)

                to_remove.add(duplicate)
                merged += 1

        for key in to_remove:
            del obls[key]

        if to_remove:
            obligation_store.save()

        return merged

    # ---- Technique merging ----

    def _merge_techniques(self, technique_library) -> int:
        """
        Find near-duplicate techniques (high TF-IDF cosine) and merge.
        Keeps the one with the better success rate.
        """
        techs = technique_library._data.get("techniques", {})
        ids   = list(techs.keys())
        n     = len(ids)
        if n < 2:
            return 0

        merged      = 0
        to_remove:  set[str] = set()

        def success_rate(t: dict) -> float:
            total = t.get("n_success", 0) + t.get("n_fail", 0)
            return t.get("n_success", 0) / max(1, total)

        for i in range(n):
            if ids[i] in to_remove:
                continue
            for j in range(i + 1, n):
                if ids[j] in to_remove:
                    continue
                ta = techs[ids[i]]
                tb = techs[ids[j]]
                # Only merge within same domain
                if ta.get("domain") != tb.get("domain"):
                    continue
                text_a = " ".join([ta.get("name", ""), ta.get("description", "")])
                text_b = " ".join([tb.get("name", ""), tb.get("description", "")])
                sim = _tfidf_cosine(text_a, text_b)
                if sim < self.tech_threshold:
                    continue

                # Keep the one with better success rate
                if success_rate(ta) >= success_rate(tb):
                    survivor, duplicate = ids[i], ids[j]
                else:
                    survivor, duplicate = ids[j], ids[i]

                # Combine example texts and run history
                existing_ex = techs[survivor].get("example_text", "")
                dup_ex      = techs[duplicate].get("example_text", "")
                if dup_ex and dup_ex not in existing_ex:
                    techs[survivor]["example_text"] = (
                        existing_ex + " | " + dup_ex
                    )[:1000]

                techs[survivor]["n_success"] = (
                    techs[survivor].get("n_success", 0)
                    + techs[duplicate].get("n_success", 0)
                )
                techs[survivor]["n_fail"] = (
                    techs[survivor].get("n_fail", 0)
                    + techs[duplicate].get("n_fail", 0)
                )

                to_remove.add(duplicate)
                merged += 1

        for key in to_remove:
            del techs[key]

        if to_remove:
            technique_library.save()

        return merged

    # ---- Graph pruning ----

    def _prune_graph(self, directed_graph) -> int:
        """
        Remove directed graph edges whose lift is below the threshold,
        but only after enough runs to have reliable statistics.
        """
        n_runs = directed_graph._data.get("n_runs", 0)
        if n_runs < self.graph_min_runs:
            return 0

        to_remove: list[str] = []
        for key in list(directed_graph._data.get("co_counts", {}).keys()):
            a, b = key.split("|", 1)
            lift = directed_graph.lift(a, b)
            if 0 < lift < self.graph_min_lift:
                to_remove.append(key)

        for key in to_remove:
            directed_graph._data["co_counts"].pop(key, None)
            a, b = key.split("|", 1)
            directed_graph._data["directed"].pop(f"{a}→{b}", None)
            directed_graph._data["directed"].pop(f"{b}→{a}", None)

        if to_remove:
            directed_graph.save()

        return len(to_remove)

    # ---- Storage measurement ----

    def _measure_storage(
        self, obligation_store, technique_library, directed_graph,
    ) -> int:
        total = 0
        for obj in [obligation_store, technique_library, directed_graph]:
            path = getattr(obj, "path", None)
            if path and os.path.exists(path):
                total += os.path.getsize(path)
        return total


# ============================================================
# UnifiedRetrieval
# ============================================================

@dataclass
class UnifiedMemoryResult:
    """Everything the pipeline needs to know about a question."""
    question:              str
    domain:                str
    relevant_techniques:   list[tuple[str, float]]   # (technique_id, score)
    similar_past_runs:     list[tuple[EpisodicRecord, float]]
    related_obligations:   list[tuple[str, int]]     # (obligation_id, n_occurrences)
    suggested_sequence:    list[str]                 # technique_ids in order
    freshness_score:       float                     # 0=never seen, 1=well-explored

    def has_prior_experience(self) -> bool:
        return len(self.similar_past_runs) > 0 or len(self.related_obligations) > 0

    def best_technique(self) -> Optional[str]:
        return self.relevant_techniques[0][0] if self.relevant_techniques else None

    def summary(self) -> str:
        lines = [
            f"Memory query: '{self.question[:60]}' [{self.domain}]",
            f"  Relevant techniques:   {len(self.relevant_techniques)}",
            f"  Similar past runs:     {len(self.similar_past_runs)}",
            f"  Related obligations:   {len(self.related_obligations)}",
            f"  Suggested sequence:    {self.suggested_sequence}",
            f"  Freshness score:       {self.freshness_score:.2f} "
            f"({'familiar' if self.freshness_score > 0.5 else 'novel'})",
        ]
        return "\n".join(lines)


class UnifiedRetrieval:
    """
    Single interface across all memory stores.
    Replaces separate calls to technique library, obligation store,
    and episodic store with one unified query.
    """

    def __init__(
        self,
        episodic_store:     EpisodicStore,
        obligation_store,
        embedded_composer,
        directed_graph,
    ):
        self.episodic     = episodic_store
        self.obligations  = obligation_store
        self.composer     = embedded_composer
        self.directed     = directed_graph

    def query(
        self,
        question:  str,
        domain:    str,
        top_k:     int = 5,
    ) -> UnifiedMemoryResult:
        """
        Query all memory stores and return a unified result.
        """
        # 1. Relevant techniques from embedding space
        techniques: list[tuple[str, float]] = []
        if self.composer.embedder.is_initialised:
            techniques = self.composer.find_similar_techniques(
                question, domain=domain, top_n=top_k,
            )

        # 2. Similar past runs from episodic memory
        similar_runs = self.episodic.query_similar(question, top_n=top_k)

        # 3. Related obligations (open gaps that might be relevant)
        related_obls: list[tuple[str, int]] = []
        try:
            gaps = self.obligations.query_persistent_gaps(top_n=30)
            for gap in gaps:
                sim = _tfidf_cosine(question, gap.text)
                if sim > 0.15:
                    related_obls.append((gap.obligation_id, gap.n_occurrences))
            related_obls.sort(key=lambda x: x[1], reverse=True)
            related_obls = related_obls[:top_k]
        except Exception:
            pass

        # 4. Suggested sequence from directed graph
        sequence: list[str] = []
        if techniques:
            seed_id = techniques[0][0]
            companions = self.directed.companions(seed_id, top_n=3)
            sequence = [seed_id] + [c[0] for c in companions]

        # 5. Freshness score — how well explored is this question type?
        freshness = min(1.0,
            len(similar_runs) / 5.0 * 0.5
            + len(techniques) / top_k * 0.5
        )

        return UnifiedMemoryResult(
            question=question,
            domain=domain,
            relevant_techniques=techniques,
            similar_past_runs=similar_runs,
            related_obligations=related_obls,
            suggested_sequence=sequence,
            freshness_score=freshness,
        )


# ============================================================
# CompressionMetrics
# ============================================================

@dataclass
class CompressionMetrics:
    timestamp:            float
    n_verified_facts:     int     # episodic records that verified
    total_storage_bytes:  int
    bytes_per_fact:       float   # key metric — want this to decrease
    technique_count:      int
    obligation_count:     int
    avg_gue_score:        Optional[float]

    def to_dict(self) -> dict:
        return {
            "timestamp":           self.timestamp,
            "n_verified_facts":    self.n_verified_facts,
            "total_storage_bytes": self.total_storage_bytes,
            "bytes_per_fact":      round(self.bytes_per_fact, 2),
            "technique_count":     self.technique_count,
            "obligation_count":    self.obligation_count,
            "avg_gue_score":       self.avg_gue_score,
        }


# ============================================================
# MemorySystem — top-level coordinator
# ============================================================

class MemorySystem:
    """
    Top-level memory coordinator.

    Holds references to all stores and provides:
      - record_run()   — called after each pipeline run
      - consolidate()  — called periodically (overnight recommended)
      - query()        — unified retrieval
      - health_report() — compression metrics + GUE scores

    Usage:
        from memory_system import MemorySystem

        memory = MemorySystem(
            episodic_path      = "./episodic_store.json",
            obligation_store   = store,
            embedded_composer  = composer,
            directed_graph     = dgraph,
            metrics_log_path   = "./memory_metrics.json",
        )

        # After each pipeline run
        memory.record_run(result)

        # Between sessions
        memory.consolidate()

        # Before generating branches
        mem = memory.query("derive MOND coefficient", domain="physics")
        print(mem.summary())
    """

    def __init__(
        self,
        episodic_path:     str,
        obligation_store,
        embedded_composer,
        directed_graph,
        metrics_log_path:  str  = "./memory_metrics.json",
        verbose:           bool = True,
    ):
        self.episodic   = EpisodicStore(episodic_path)
        self.obligations = obligation_store
        self.composer    = embedded_composer
        self.directed    = directed_graph
        self.metrics_log = metrics_log_path
        self.verbose     = verbose

        self.consolidator = MemoryConsolidator(verbose=verbose)
        self.retrieval    = UnifiedRetrieval(
            self.episodic, obligation_store, embedded_composer, directed_graph,
        )
        self._metrics_history: list[dict] = self._load_metrics()

    # ---- Record a run ----

    def record_run(
        self,
        pipeline_result: dict,
        gue_score:        Optional[float] = None,
    ) -> Optional[EpisodicRecord]:
        """
        Record a completed pipeline run to episodic memory.
        Also triggers embedder update if the run had a selected branch.
        """
        record = self.episodic.record(pipeline_result, gue_score)

        # Update embedder on successful, verified runs
        selected = pipeline_result.get("selected_branch", {})
        if record and record.verified and record.technique_id:
            try:
                # Get technique ids used in this run
                tids_used = [record.technique_id]
                self.composer.update_from_run(tids_used, success=True)
            except Exception:
                pass

        return record

    # ---- Consolidation ----

    def consolidate(self) -> ConsolidationReport:
        """
        Run a full consolidation pass. Recommended between sessions.
        """
        return self.consolidator.consolidate(
            self.obligations, self.composer.library, self.directed,
        )

    # ---- Retrieval ----

    def query(self, question: str, domain: str, top_k: int = 5) -> UnifiedMemoryResult:
        return self.retrieval.query(question, domain, top_k)

    # ---- Health reporting ----

    def health_report(self) -> CompressionMetrics:
        """
        Compute and log current compression metrics.
        The key signal: bytes_per_fact over time. Decreasing = good.
        """
        stats       = self.episodic.statistics()
        n_verified  = sum(
            1 for r in self.episodic._records.values() if r.verified
        )

        # Measure total storage
        total_bytes = 0
        for obj in [self.obligations, self.composer.library, self.directed]:
            path = getattr(obj, "path", None)
            if path and os.path.exists(path):
                total_bytes += os.path.getsize(path)
        if os.path.exists(self.episodic.path):
            total_bytes += os.path.getsize(self.episodic.path)

        bytes_per_fact = total_bytes / max(1, n_verified)

        # GUE score across embedding space
        avg_gue = stats.get("avg_gue_score")

        metrics = CompressionMetrics(
            timestamp=time.time(),
            n_verified_facts=n_verified,
            total_storage_bytes=total_bytes,
            bytes_per_fact=bytes_per_fact,
            technique_count=len(
                self.composer.library._data.get("techniques", {})
            ),
            obligation_count=len(
                self.obligations._data.get("obligations", {})
            ),
            avg_gue_score=avg_gue,
        )

        self._metrics_history.append(metrics.to_dict())
        self._save_metrics()

        if self.verbose:
            self._print_health(metrics)

        return metrics

    def compression_trend(self) -> list[float]:
        """
        Historical bytes_per_fact values — want this decreasing over time.
        """
        return [m.get("bytes_per_fact", 0) for m in self._metrics_history]

    # ---- Persistence ----

    def _load_metrics(self) -> list[dict]:
        if not os.path.exists(self.metrics_log):
            return []
        try:
            with open(self.metrics_log) as f:
                return json.load(f).get("history", [])
        except (json.JSONDecodeError, OSError):
            return []

    def _save_metrics(self) -> None:
        os.makedirs(os.path.dirname(self.metrics_log) or ".", exist_ok=True)
        with open(self.metrics_log, "w") as f:
            json.dump({"history": self._metrics_history}, f, indent=2)

    def _print_health(self, m: CompressionMetrics) -> None:
        trend = self.compression_trend()
        trend_str = ""
        if len(trend) >= 2:
            delta = trend[-1] - trend[-2]
            trend_str = f" ({'+' if delta > 0 else ''}{delta:.0f} vs last)"
        print(
            f"\n[memory] Health report:\n"
            f"  Verified facts:   {m.n_verified_facts}\n"
            f"  Techniques:       {m.technique_count}\n"
            f"  Obligations:      {m.obligation_count}\n"
            f"  Storage:          {m.total_storage_bytes:,} bytes\n"
            f"  Bytes per fact:   {m.bytes_per_fact:.0f}{trend_str} "
            f"{'↓ good' if len(trend) >= 2 and trend[-1] < trend[-2] else ''}\n"
            + (f"  Avg GUE score:    {m.avg_gue_score:.3f}\n"
               if m.avg_gue_score else "")
        )
