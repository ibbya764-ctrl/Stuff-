"""
brain_modules.py
================

Three brain-analog modules that complete the cognitive architecture:

  WorkingMemory      — persistent slot-based scratchpad. Holds the
                       active reasoning context across multiple turns.
                       Distinct from long-term episodic store.
                       7 slots (matching human working memory capacity).
                       Relevance decays; overflow consolidates to episodic.

  AffectiveSignal    — urgency scores for unresolved obligations.
                       Not just whether something is unresolved but
                       how much the system cares about resolving it.
                       Four components: recurrence, age, blocking,
                       failure rate. Feeds curiosity engine and
                       attention director prioritisation.

  ProceduralMemory   — reasoning patterns that automatise with use.
                       Below threshold (7 verified uses): full reasoning.
                       Above threshold: pattern applied directly, no
                       full chain needed. Unlearns on failures.
                       The cerebellum analog.

  BrainModulesBundle — wires all three together and integrates with
                       ThoughtAwareBrainPipeline and EnhancedPipeline.
"""

import os
import re
import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# WorkingMemory
# ============================================================

@dataclass
class WorkingMemorySlot:
    """One item in active working memory."""
    slot_id:       str
    content:       str
    slot_type:     str     # "hypothesis"|"result"|"context"|"obligation"|"constraint"
    relevance:     float   # 0-1, decays over time
    created_at:    float
    last_accessed: float
    domain:        str = ""
    source_run:    str = ""

    def decay(self, rate: float = 0.05) -> None:
        """Reduce relevance slightly each turn."""
        self.relevance = max(0.0, self.relevance - rate)

    def access(self) -> None:
        """Accessing a slot boosts its relevance."""
        self.relevance    = min(1.0, self.relevance + 0.2)
        self.last_accessed = time.time()

    def to_dict(self) -> dict:
        return {
            "slot_id":      self.slot_id,
            "content":      self.content,
            "slot_type":    self.slot_type,
            "relevance":    round(self.relevance, 3),
            "created_at":   self.created_at,
            "last_accessed": self.last_accessed,
            "domain":       self.domain,
            "source_run":   self.source_run,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkingMemorySlot":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class WorkingMemory:
    """
    Active reasoning scratchpad — the brain's working memory.

    Holds currently relevant information across turns.
    Limited to MAX_SLOTS (7) active items.
    Items decay in relevance when not accessed.
    Overflow consolidates to the episodic store.

    Persists to disk — survives closing the terminal and coming back.
    """

    MAX_SLOTS    = 7
    DECAY_RATE   = 0.05    # per turn
    MIN_RELEVANCE = 0.1    # below this, slot is evicted

    def __init__(self, path: str = "./scaffold_data/working_memory.json"):
        self.path   = path
        self._slots: dict[str, WorkingMemorySlot] = {}
        self._turn:  int = 0
        self._load()

    # ---- Core operations ----

    def add(
        self,
        content:    str,
        slot_type:  str,
        domain:     str = "",
        source_run: str = "",
        relevance:  float = 1.0,
    ) -> str:
        """
        Add an item to working memory.
        If at capacity, evicts the lowest-relevance slot first.
        """
        if len(self._slots) >= self.MAX_SLOTS:
            self._evict_lowest()

        slot_id = f"wm-{int(time.time() * 1000) % 100000}"
        self._slots[slot_id] = WorkingMemorySlot(
            slot_id=slot_id,
            content=content,
            slot_type=slot_type,
            relevance=relevance,
            created_at=time.time(),
            last_accessed=time.time(),
            domain=domain,
            source_run=source_run,
        )
        self._save()
        return slot_id

    def query(self, question: str, top_n: int = 3) -> list[WorkingMemorySlot]:
        """
        Retrieve the most relevant slots for the current question.
        Uses word-overlap similarity, boosted by relevance score.
        """
        q_words = set(_wm_tokenise(question))
        scored: list[tuple[float, WorkingMemorySlot]] = []

        for slot in self._slots.values():
            s_words  = set(_wm_tokenise(slot.content))
            overlap  = (
                len(q_words & s_words) / max(1, len(q_words | s_words))
            )
            combined = 0.5 * overlap + 0.5 * slot.relevance
            if combined > 0.05:
                scored.append((combined, slot))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [s for _, s in scored[:top_n]]

        # Accessing these slots boosts their relevance
        for slot in results:
            slot.access()

        if results:
            self._save()
        return results

    def update(self, slot_id: str, new_content: str) -> bool:
        if slot_id in self._slots:
            self._slots[slot_id].content      = new_content
            self._slots[slot_id].last_accessed = time.time()
            self._slots[slot_id].relevance     = min(
                1.0, self._slots[slot_id].relevance + 0.1
            )
            self._save()
            return True
        return False

    def advance_turn(self, episodic_store=None) -> int:
        """
        Decay all slots. Evict expired ones to episodic store.
        Call after each reasoning turn.
        Returns number of slots evicted.
        """
        self._turn += 1
        evicted = 0

        for slot in list(self._slots.values()):
            slot.decay(self.DECAY_RATE)
            if slot.relevance < self.MIN_RELEVANCE:
                if episodic_store and slot.slot_type == "result":
                    # Push completed results to long-term memory
                    try:
                        episodic_store.add(slot.content, slot.domain)
                    except Exception:
                        pass
                del self._slots[slot.slot_id]
                evicted += 1

        self._save()
        return evicted

    def get_context_string(self, max_slots: int = 4) -> str:
        """
        Format active working memory as context for the Reasoner.
        Call this before each reasoning step.
        """
        if not self._slots:
            return ""

        # Get highest-relevance slots
        top_slots = sorted(
            self._slots.values(),
            key=lambda s: s.relevance,
            reverse=True,
        )[:max_slots]

        lines = ["[WORKING MEMORY — active context from recent reasoning]"]
        for slot in top_slots:
            lines.append(
                f"  [{slot.slot_type.upper()}] {slot.content[:120]}"
                f" (relevance: {slot.relevance:.2f})"
            )
        lines.append("[/WORKING MEMORY]")
        return "\n".join(lines)

    def update_from_result(
        self,
        reasoning_output,
        question: str,
        domain:   str = "",
    ) -> None:
        """
        After a reasoning run, update working memory with:
        - The result (as a 'result' slot)
        - Any new obligations raised (as 'obligation' slots)
        - The active method (as 'context' slot)
        """
        if reasoning_output.result:
            self.add(
                content=f"Q: {question[:60]} → {reasoning_output.result[:120]}",
                slot_type="result",
                domain=domain,
                source_run=reasoning_output.run_id,
                relevance=1.0,
            )

        for obl in reasoning_output.obligations_raised[:2]:
            self.add(
                content=obl[:120],
                slot_type="obligation",
                domain=domain,
                source_run=reasoning_output.run_id,
                relevance=0.8,
            )

        if reasoning_output.method != "parse_failed":
            self.add(
                content=f"Recent method: {reasoning_output.method}",
                slot_type="context",
                domain=domain,
                source_run=reasoning_output.run_id,
                relevance=0.7,
            )

    # ---- Internals ----

    def _evict_lowest(self) -> Optional[WorkingMemorySlot]:
        if not self._slots:
            return None
        lowest_id = min(self._slots, key=lambda k: self._slots[k].relevance)
        slot = self._slots.pop(lowest_id)
        return slot

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._turn  = data.get("turn", 0)
            self._slots = {
                d["slot_id"]: WorkingMemorySlot.from_dict(d)
                for d in data.get("slots", [])
                if "slot_id" in d
            }
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "turn":  self._turn,
                "slots": [s.to_dict() for s in self._slots.values()],
            }, f, indent=2)

    @property
    def n_slots(self) -> int:
        return len(self._slots)


# ============================================================
# AffectiveSignal
# ============================================================

@dataclass
class SalienceRecord:
    """Urgency record for one unresolved obligation."""
    obligation_id:    str
    obligation_text:  str
    urgency:          float     # 0-1 composite score
    recurrence:       float     # how often has this appeared
    age_score:        float     # how long unresolved (normalised)
    blocking_score:   float     # how much does this block other reasoning
    failure_score:    float     # rate of failed resolution attempts
    n_occurrences:    int
    n_failed_attempts: int
    first_seen:       float
    last_attempted:   float
    domain:           str

    def to_dict(self) -> dict:
        return {
            "obligation_id":    self.obligation_id,
            "obligation_text":  self.obligation_text[:200],
            "urgency":          round(self.urgency, 4),
            "recurrence":       round(self.recurrence, 4),
            "age_score":        round(self.age_score, 4),
            "blocking_score":   round(self.blocking_score, 4),
            "failure_score":    round(self.failure_score, 4),
            "n_occurrences":    self.n_occurrences,
            "n_failed_attempts": self.n_failed_attempts,
            "domain":           self.domain,
        }


class AffectiveSignal:
    """
    Urgency scoring for unresolved obligations.

    Combines four components into a single urgency score:
      - Recurrence: how many times has this appeared?
      - Age:        how long has this been unresolved?
      - Blocking:   how much does this block other reasoning?
      - Failure:    how many attempts to resolve this have failed?

    High urgency = the system should prioritise this gap.
    Urgency increases with repeated failure (frustration signal).
    Urgency also increases with blocking score (importance signal).
    """

    # Weights for urgency components
    W_RECURRENCE = 0.25
    W_AGE        = 0.20
    W_BLOCKING   = 0.35
    W_FAILURE    = 0.20

    # Normalisation constants
    MAX_OCCURRENCES    = 15
    MAX_AGE_DAYS       = 30
    MAX_FAILED         = 8

    def __init__(self, path: str = "./scaffold_data/affective_signal.json"):
        self.path      = path
        self._records: dict[str, SalienceRecord] = {}
        self._load()

    # ---- Core operations ----

    def update_from_obligation_store(
        self,
        obligation_store,
        blocking_scores: Optional[dict] = None,
    ) -> int:
        """
        Sync with the obligation store and recompute urgency scores.
        blocking_scores: {obligation_id: float} from curiosity engine.
        Returns number of records updated.
        """
        try:
            gaps = obligation_store.query_persistent_gaps(top_n=100)
        except Exception:
            return 0

        now     = time.time()
        updated = 0

        for gap in gaps:
            obl_id    = gap.obligation_id
            age_days  = (now - getattr(gap, "first_seen_time", now)) / 86400
            n_occ     = gap.n_occurrences

            rec  = self._records.get(obl_id)
            n_fail = rec.n_failed_attempts if rec else 0
            last_att = rec.last_attempted if rec else now

            recurrence = min(1.0, n_occ / self.MAX_OCCURRENCES)
            age_score  = min(1.0, age_days / self.MAX_AGE_DAYS)
            blocking   = min(1.0, (blocking_scores or {}).get(obl_id, 0.3))
            failure    = min(1.0, n_fail / self.MAX_FAILED)

            urgency = (
                self.W_RECURRENCE * recurrence
                + self.W_AGE      * age_score
                + self.W_BLOCKING * blocking
                + self.W_FAILURE  * failure
            )

            self._records[obl_id] = SalienceRecord(
                obligation_id=obl_id,
                obligation_text=gap.text,
                urgency=round(urgency, 4),
                recurrence=recurrence,
                age_score=age_score,
                blocking_score=blocking,
                failure_score=failure,
                n_occurrences=n_occ,
                n_failed_attempts=n_fail,
                first_seen=now - age_days * 86400,
                last_attempted=last_att,
                domain=self._infer_domain(gap.text),
            )
            updated += 1

        self._save()
        return updated

    def notify_attempt(self, obligation_id: str, success: bool) -> None:
        """
        Update urgency after a resolution attempt.
        Success decreases urgency. Failure increases it.
        """
        if obligation_id not in self._records:
            return

        rec = self._records[obligation_id]
        rec.last_attempted = time.time()

        if success:
            # Success: drop urgency significantly
            rec.urgency          = max(0.0, rec.urgency - 0.3)
            rec.failure_score    = max(0.0, rec.failure_score - 0.1)
        else:
            # Failure: increase failure component
            rec.n_failed_attempts += 1
            rec.failure_score     = min(
                1.0, rec.n_failed_attempts / self.MAX_FAILED
            )
            # Recompute urgency
            rec.urgency = (
                self.W_RECURRENCE * rec.recurrence
                + self.W_AGE      * rec.age_score
                + self.W_BLOCKING * rec.blocking_score
                + self.W_FAILURE  * rec.failure_score
            )

        self._save()

    def top_urgent(self, n: int = 5, domain: str = "") -> list[SalienceRecord]:
        """Return top N most urgent obligations."""
        records = list(self._records.values())
        if domain:
            records = [r for r in records if r.domain == domain] or records
        records.sort(key=lambda r: r.urgency, reverse=True)
        return records[:n]

    def get_context_for_curiosity(self, n: int = 3) -> str:
        """
        Format high-urgency obligations as context for curiosity engine.
        """
        top = self.top_urgent(n)
        if not top:
            return ""
        lines = ["[HIGH URGENCY GAPS — system most wants to resolve these]"]
        for r in top:
            lines.append(
                f"  [{r.urgency:.2f}] {r.obligation_text[:100]}"
                f" (failed {r.n_failed_attempts}× — domain: {r.domain})"
            )
        lines.append("[/URGENCY]")
        return "\n".join(lines)

    def urgency_for(self, obligation_id: str) -> float:
        rec = self._records.get(obligation_id)
        return rec.urgency if rec else 0.0

    # ---- Internals ----

    @staticmethod
    def _infer_domain(text: str) -> str:
        text = text.lower()
        if any(w in text for w in ["physics","field","closure","mond","grav"]):
            return "physics"
        if any(w in text for w in ["econ","agent","utility","market","wage"]):
            return "economics"
        if any(w in text for w in ["bio","gene","protein","cell","neural"]):
            return "biology"
        return "general"

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for d in data.get("records", []):
                try:
                    rec = SalienceRecord(**d)
                    self._records[rec.obligation_id] = rec
                except Exception:
                    pass
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "records": [r.to_dict() for r in self._records.values()]
            }, f, indent=2)


# ============================================================
# ProceduralMemory
# ============================================================

@dataclass
class ProceduralPattern:
    """
    A reasoning pattern that has automatised through repeated verification.
    """
    pattern_id:          str
    trigger_description: str
    trigger_keywords:    list[str]
    action_steps:        list[str]    # verified reasoning steps
    result_template:     str          # the expected result
    verified_count:      int          # number of successful verifications
    failed_count:        int          # number of failures
    domain:              str
    first_verified:      float
    last_verified:       float
    technique_id:        str = ""     # linked technique if any

    @property
    def success_rate(self) -> float:
        total = self.verified_count + self.failed_count
        return self.verified_count / max(1, total)

    @property
    def confidence(self) -> str:
        if self.verified_count >= 10 and self.success_rate > 0.85:
            return "HIGH"
        if self.verified_count >= 5 and self.success_rate > 0.70:
            return "MODERATE"
        return "LOW"

    @property
    def is_automatic(self) -> bool:
        """
        Pattern is automatic when verified enough times at sufficient
        success rate — analogous to cerebellum automatisation.
        """
        return (
            self.verified_count >= ProceduralMemory.AUTOMATISATION_THRESHOLD
            and self.success_rate >= 0.75
        )

    def to_dict(self) -> dict:
        return {
            "pattern_id":          self.pattern_id,
            "trigger_description": self.trigger_description,
            "trigger_keywords":    self.trigger_keywords,
            "action_steps":        self.action_steps,
            "result_template":     self.result_template,
            "verified_count":      self.verified_count,
            "failed_count":        self.failed_count,
            "domain":              self.domain,
            "first_verified":      self.first_verified,
            "last_verified":       self.last_verified,
            "technique_id":        self.technique_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProceduralPattern":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class ProceduralMemory:
    """
    Reasoning patterns that automatise with verified use.

    The cerebellum analog: patterns practised enough times no longer
    need conscious (full-chain) reasoning — they fire directly.

    Below AUTOMATISATION_THRESHOLD uses → full reasoning chain.
    Above threshold + sufficient success rate → automatic application.

    Patterns deteriorate with failures (unlearning is possible).
    """

    AUTOMATISATION_THRESHOLD = 7     # verified uses to become automatic
    DETERIORATION_THRESHOLD  = 3     # consecutive failures before demotion
    SIMILARITY_THRESHOLD     = 0.35  # word-overlap to match a pattern

    def __init__(self, path: str = "./scaffold_data/procedural_memory.json"):
        self.path      = path
        self._patterns: dict[str, ProceduralPattern] = {}
        self._load()

    # ---- Learning ----

    def learn_from_run(
        self,
        reasoning_output,
        technique_id: str = "",
    ) -> Optional[str]:
        """
        After a verified run, contribute to a procedural pattern.
        Creates a new pattern or strengthens an existing one.
        Returns pattern_id if a pattern was updated.
        """
        if not reasoning_output.verified:
            return None
        if not reasoning_output.steps:
            return None

        # Find matching existing pattern
        pattern_id = self._find_matching_pattern(
            reasoning_output.question,
            reasoning_output.domain,
        )

        if pattern_id:
            # Strengthen existing pattern
            self._patterns[pattern_id].verified_count += 1
            self._patterns[pattern_id].last_verified   = time.time()
            was_auto = self._patterns[pattern_id].is_automatic
            if not was_auto and self._patterns[pattern_id].is_automatic:
                print(f"  [procedural] Pattern '{pattern_id}' became AUTOMATIC "
                      f"({self._patterns[pattern_id].verified_count} verifications)")
        else:
            # Create new pattern
            keywords  = list(set(_wm_tokenise(reasoning_output.question)))[:8]
            steps     = [
                f"[{s.label}] {s.content}"
                for s in reasoning_output.steps
            ]
            pattern_id = f"proc-{int(time.time()) % 100000}"
            self._patterns[pattern_id] = ProceduralPattern(
                pattern_id=pattern_id,
                trigger_description=reasoning_output.question[:150],
                trigger_keywords=keywords,
                action_steps=steps,
                result_template=reasoning_output.result[:200],
                verified_count=1,
                failed_count=0,
                domain=reasoning_output.domain,
                first_verified=time.time(),
                last_verified=time.time(),
                technique_id=technique_id,
            )

        self._save()
        return pattern_id

    def notify_failure(self, pattern_id: str) -> bool:
        """
        A pattern failed. Increment failure count.
        If too many failures, demote from automatic.
        Returns True if pattern was demoted.
        """
        if pattern_id not in self._patterns:
            return False

        p = self._patterns[pattern_id]
        p.failed_count += 1

        # Check if consecutive failures should demote
        consecutive_failures = p.failed_count
        if consecutive_failures >= self.DETERIORATION_THRESHOLD:
            # Demote: reduce verified_count to just below threshold
            p.verified_count = max(
                0, self.AUTOMATISATION_THRESHOLD - 1
            )
            print(f"  [procedural] Pattern '{pattern_id}' DEMOTED "
                  f"({consecutive_failures} failures)")
            self._save()
            return True

        self._save()
        return False

    # ---- Retrieval ----

    def try_automatic(
        self,
        question: str,
        domain:   str = "",
    ) -> Optional[tuple[ProceduralPattern, "ReasonerOutput"]]:
        """
        If a matching automatic pattern exists, return a fast-path
        ReasonerOutput without running the full reasoning chain.

        Returns (pattern, reasoner_output) or None if no automatic
        pattern matches.
        """
        pattern_id = self._find_matching_pattern(question, domain, automatic_only=True)
        if not pattern_id:
            return None

        pattern = self._patterns[pattern_id]

        from brain_pipeline import ReasonerOutput, ReasonerStep
        steps = []
        for step_text in pattern.action_steps:
            match = re.match(r"\[(\w+)\]\s*(.*)", step_text)
            if match:
                steps.append(ReasonerStep(
                    label=match.group(1),
                    content=match.group(2),
                    source=f"procedural:{pattern_id}",
                ))
            else:
                steps.append(ReasonerStep(
                    label="STRUCTURAL",
                    content=step_text,
                    source=f"procedural:{pattern_id}",
                ))

        output = ReasonerOutput(
            run_id=f"proc-{int(time.time())}",
            domain=domain or pattern.domain,
            question=question,
            method=f"[AUTOMATIC] {pattern.trigger_description[:60]}",
            steps=steps,
            assumptions=[],
            result=pattern.result_template,
            verified=True,
            confidence=pattern.confidence,
            structural_fraction=1.0,   # all structural — no LLM derivation
            obligations_raised=[],
            obligations_discharged=[],
        )

        print(f"  [procedural] Automatic pattern matched "
              f"({pattern.verified_count} prior verifications, "
              f"{pattern.success_rate:.0%} success rate)")

        return pattern, output

    def get_relevant_patterns(
        self,
        question: str,
        domain:   str = "",
        top_n:    int = 3,
    ) -> list[ProceduralPattern]:
        """
        Get relevant patterns (including non-automatic) as context.
        """
        scored: list[tuple[float, ProceduralPattern]] = []
        q_words = set(_wm_tokenise(question))

        for p in self._patterns.values():
            if domain and p.domain != domain:
                continue
            p_words = set(p.trigger_keywords)
            overlap = len(q_words & p_words) / max(1, len(q_words | p_words))
            if overlap >= self.SIMILARITY_THRESHOLD * 0.5:
                score = overlap * p.success_rate * math.log1p(p.verified_count)
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_n]]

    def stats(self) -> dict:
        automatic = sum(1 for p in self._patterns.values() if p.is_automatic)
        return {
            "n_patterns":      len(self._patterns),
            "n_automatic":     automatic,
            "n_learning":      len(self._patterns) - automatic,
            "avg_verified":    (
                sum(p.verified_count for p in self._patterns.values())
                / max(1, len(self._patterns))
            ),
        }

    # ---- Internals ----

    def _find_matching_pattern(
        self,
        question:       str,
        domain:         str = "",
        automatic_only: bool = False,
    ) -> Optional[str]:
        q_words   = set(_wm_tokenise(question))
        best_id   = None
        best_score = 0.0

        for pid, p in self._patterns.items():
            if domain and p.domain != domain:
                continue
            if automatic_only and not p.is_automatic:
                continue
            p_words = set(p.trigger_keywords)
            overlap = len(q_words & p_words) / max(1, len(q_words | p_words))
            score   = overlap * p.success_rate
            if score > best_score and overlap >= self.SIMILARITY_THRESHOLD:
                best_score = score
                best_id    = pid

        return best_id

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for d in data.get("patterns", []):
                try:
                    p = ProceduralPattern.from_dict(d)
                    self._patterns[p.pattern_id] = p
                except Exception:
                    pass
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "patterns": [p.to_dict() for p in self._patterns.values()]
            }, f, indent=2)


# ============================================================
# BrainModulesBundle — wires everything together
# ============================================================

class BrainModulesBundle:
    """
    Convenience wrapper that holds all three modules and provides
    integration hooks for ThoughtAwareBrainPipeline and EnhancedPipeline.

    Usage:
        from brain_modules import BrainModulesBundle

        brain = BrainModulesBundle(base_dir="./scaffold_data")

        # Before reasoning
        context = brain.pre_reasoning_context(question, domain)

        # After reasoning
        brain.post_reasoning_update(reasoning_output, question, domain)

        # Periodic (between sessions)
        brain.consolidate(obligation_store)
    """

    def __init__(self, base_dir: str = "./scaffold_data"):
        os.makedirs(base_dir, exist_ok=True)
        self.working_memory  = WorkingMemory(
            os.path.join(base_dir, "working_memory.json")
        )
        self.affective       = AffectiveSignal(
            os.path.join(base_dir, "affective_signal.json")
        )
        self.procedural      = ProceduralMemory(
            os.path.join(base_dir, "procedural_memory.json")
        )

    def pre_reasoning_context(self, question: str, domain: str = "") -> str:
        """
        Assemble all brain module context for injection before reasoning.
        Returns a formatted context string.
        """
        parts = []

        wm_ctx = self.working_memory.get_context_string(max_slots=3)
        if wm_ctx:
            parts.append(wm_ctx)

        affect_ctx = self.affective.get_context_for_curiosity(n=2)
        if affect_ctx:
            parts.append(affect_ctx)

        # Add procedural patterns as hints
        patterns = self.procedural.get_relevant_patterns(question, domain, top_n=2)
        if patterns:
            lines = ["[PROCEDURAL HINTS — verified patterns for this type of problem]"]
            for p in patterns:
                auto_str = " [AUTOMATIC]" if p.is_automatic else f" [{p.verified_count} uses]"
                lines.append(f"  {p.trigger_description[:80]}{auto_str}")
            lines.append("[/PROCEDURAL]")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def try_automatic(
        self, question: str, domain: str = ""
    ) -> Optional[tuple]:
        """
        Check if a procedural shortcut exists for this question.
        Returns (pattern, ReasonerOutput) if automatic pattern found.
        """
        return self.procedural.try_automatic(question, domain)

    def post_reasoning_update(
        self,
        reasoning_output,
        question:    str,
        domain:      str = "",
        pattern_id:  Optional[str] = None,
        success:     bool = True,
    ) -> None:
        """
        Update all brain modules after a reasoning run.
        Call this after every pipeline run.
        """
        # Update working memory
        self.working_memory.update_from_result(reasoning_output, question, domain)
        self.working_memory.advance_turn()

        # Update procedural memory
        if reasoning_output.verified:
            pid = self.procedural.learn_from_run(reasoning_output)
        elif pattern_id:
            self.procedural.notify_failure(pattern_id)

        # Update affective signal for newly raised obligations
        # (full sync happens in consolidate())

    def consolidate(self, obligation_store=None, episodic_store=None) -> dict:
        """
        Periodic sync — update affective signal from obligation store,
        consolidate working memory overflow to episodic.
        """
        report = {}

        if obligation_store:
            n = self.affective.update_from_obligation_store(obligation_store)
            report["affective_updated"] = n

        if episodic_store:
            evicted = self.working_memory.advance_turn(episodic_store)
            report["wm_evicted_to_episodic"] = evicted

        report["procedural_stats"] = self.procedural.stats()
        return report

    def status(self) -> str:
        wm    = self.working_memory
        proc  = self.procedural.stats()
        top_u = self.affective.top_urgent(n=3)

        lines = [
            "Brain Modules Status:",
            f"  Working Memory:  {wm.n_slots}/{WorkingMemory.MAX_SLOTS} slots active",
            f"  Procedural:      {proc['n_patterns']} patterns "
            f"({proc['n_automatic']} automatic)",
            f"  Affective (top urgency):",
        ]
        for r in top_u:
            lines.append(
                f"    [{r.urgency:.2f}] {r.obligation_text[:70]}"
            )
        return "\n".join(lines)


# ============================================================
# Helpers
# ============================================================

_STOP = {
    "the","a","an","is","are","be","to","of","and","or","for",
    "in","on","at","by","with","from","as","this","that","it",
}

def _wm_tokenise(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z]{3,}", text.lower())
        if t not in _STOP
    ]
