"""
compute_router.py
=================

Routes questions to the appropriate level of compute based on
what they actually need. Not every question requires the full
pipeline — running self-review and back-verification on "what
is the speed of light" wastes time and degrades the experience.

Four levels:

  FAST    — procedural shortcut check + direct Communicator call.
            No Reasoner chain at all. For questions the system has
            answered many times and has automatic patterns for.
            Target: < 15 seconds.

  LIGHT   — Reasoner → Communicator. No self-review, no back-verify.
            For straightforward factual or explanatory questions.
            Target: 30-60 seconds.

  FULL    — Reasoner → self-review → Communicator → back-verify.
            The standard pipeline. For reasoning-intensive questions.
            Target: 90-180 seconds.

  DEEP    — Full pipeline + thought streaming (step-by-step with
            real-time self-checking). For the hardest questions where
            catching mid-stream mistakes matters most.
            Target: 4-8 minutes.

Routing is two-stage:

  1. Pre-routing: classify complexity from the question text.
     Simple heuristics — length, question type markers, domain signals.

  2. Dynamic escalation: if the response at the current level
     fails (low confidence, verification failure, low psychological
     quality), escalate to the next level automatically.
     The user sees a note: "Going deeper on this one."

Also provides time estimates before processing so users know
what to expect, and records actual times to improve estimates.
"""

import re
import time
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# Complexity signals
# ============================================================

FAST_SIGNALS = [
    r"\bwhat is\b", r"\bwhat are\b", r"\bdefine\b", r"\bdefinition\b",
    r"\bhow many\b", r"\bwhen did\b", r"\bwho is\b", r"\bwhere is\b",
    r"\bwhat does .+ stand for\b", r"\bwhat does .+ mean\b",
]

LIGHT_SIGNALS = [
    r"\bexplain\b", r"\bdescribe\b", r"\bsummarise\b", r"\bsummarize\b",
    r"\bcompare\b", r"\bwhat is the difference\b", r"\boverview\b",
    r"\bintroduce\b", r"\bgive me an idea\b", r"\btell me about\b",
]

FULL_SIGNALS = [
    r"\bderive\b", r"\bprove\b", r"\bshow that\b", r"\bdemonstrate\b",
    r"\banalyse\b", r"\banalyze\b", r"\bwhy does\b", r"\bhow does\b",
    r"\bwhat causes\b", r"\bwork through\b", r"\bcalculate\b",
    r"\bverify\b", r"\bcheck\b", r"\bwhat are the implications\b",
]

DEEP_SIGNALS = [
    r"\bthink through\b", r"\bif you had to\b", r"\bfrom first principles\b",
    r"\bfundamentally\b", r"\bphilosophically\b", r"\bdeeply\b",
    r"\bwhat would it mean\b", r"\bwhat is the relationship between\b",
    r"\bhow might we reconcile\b", r"\bwhat are the assumptions\b",
]

# Complexity markers in the domain
FORMAL_DOMAINS = {"physics", "mathematics", "logic", "chemistry", "philosophy"}
LIGHT_DOMAINS  = {"conversation", "general", "planning", "writing"}


@dataclass
class RouteDecision:
    level:            str        # "fast" | "light" | "full" | "deep"
    confidence:       float      # how confident in this routing decision
    reason:           str        # why this level was chosen
    estimated_seconds: int       # time estimate for user
    can_escalate:     bool = True


@dataclass
class RouteRecord:
    question_hash:    str
    domain:           str
    chosen_level:     str
    final_level:      str        # may differ if escalated
    estimated_sec:    int
    actual_sec:       float
    verified:         bool
    escalated:        bool
    timestamp:        float = field(default_factory=time.time)


# ============================================================
# ComputeRouter
# ============================================================

class ComputeRouter:
    """
    Classifies question complexity and routes to the right pipeline.

    Learns from experience: if a domain consistently requires full
    pipeline, its baseline route moves up. If questions at a domain
    almost always verify at light level, light becomes the baseline.
    """

    # Escalation path
    LEVELS = ["fast", "light", "full", "deep"]

    # Time estimates per level (seconds)
    TIME_ESTIMATES = {
        "fast":  12,
        "light": 45,
        "full":  120,
        "deep":  300,
    }

    # Escalation threshold: escalate if result quality below this
    ESCALATION_CONFIDENCE_THRESHOLD = 0.35

    def __init__(
        self,
        path:    str  = "./scaffold_data/compute_router.json",
        verbose: bool = True,
    ):
        self.path    = path
        self.verbose = verbose
        self._records: list[dict] = []
        self._domain_baselines: dict[str, str] = {}
        self._load()

    # ── Routing ────────────────────────────────────────────

    def classify(self, question: str, domain: str = "") -> RouteDecision:
        """
        Classify a question and decide which compute level to use.
        """
        lower = question.lower().strip()
        words = len(lower.split())

        # Very short questions are usually simple
        if words <= 4 and not any(re.search(p, lower) for p in FULL_SIGNALS + DEEP_SIGNALS):
            return RouteDecision(
                level="fast", confidence=0.8,
                reason="short question, likely factual",
                estimated_seconds=self.TIME_ESTIMATES["fast"],
            )

        # Signal scoring
        scores = {
            "fast":  sum(1 for p in FAST_SIGNALS  if re.search(p, lower)),
            "light": sum(1 for p in LIGHT_SIGNALS if re.search(p, lower)),
            "full":  sum(1 for p in FULL_SIGNALS  if re.search(p, lower)),
            "deep":  sum(1 for p in DEEP_SIGNALS  if re.search(p, lower)),
        }

        # Domain modifier
        domain_lower = domain.lower()
        if any(d in domain_lower for d in FORMAL_DOMAINS):
            scores["full"]  += 1
            scores["deep"]  += 1
        elif any(d in domain_lower for d in LIGHT_DOMAINS):
            scores["light"] += 1
            scores["fast"]  += 1

        # Question length modifier: longer = more complex
        if words > 30:
            scores["full"] += 1
        if words > 60:
            scores["deep"] += 1

        # Multiple sub-questions
        if lower.count("?") > 1 or "and" in lower and words > 20:
            scores["full"] += 1

        # Domain baseline from experience
        baseline = self._domain_baselines.get(domain_lower)
        if baseline:
            scores[baseline] += 1

        # Find winner
        best  = max(scores, key=lambda k: scores[k])
        total = sum(scores.values()) or 1
        conf  = scores[best] / total

        # If no strong signal, use light as default
        if conf < 0.3 or total < 2:
            best = "light"
            conf = 0.5
            reason = "no strong signal — defaulting to light"
        else:
            reason = f"signal score: {scores}"

        if self.verbose:
            print(f"  [router] {question[:50]}... → {best.upper()} "
                  f"(confidence {conf:.2f})")

        return RouteDecision(
            level=best,
            confidence=conf,
            reason=reason,
            estimated_seconds=self._estimate_time(best, domain),
        )

    def should_escalate(
        self,
        current_level:  str,
        verified:       bool,
        confidence_str: str,
        quality_score:  float = 0.5,
    ) -> tuple[bool, str]:
        """
        After a routing level runs, decide whether to escalate.
        Returns (should_escalate, reason).
        """
        if current_level == "deep":
            return False, "already at deepest level"

        conf_map = {"HIGH": 0.85, "MODERATE": 0.6, "LOW": 0.35, "UNCERTAIN": 0.2}
        conf_val = conf_map.get(confidence_str, 0.5)

        if not verified and current_level == "fast":
            return True, "fast path failed to verify — trying light"

        if not verified and current_level == "light":
            return True, "light path failed — escalating to full reasoning"

        if conf_val < 0.4 and current_level in ("fast", "light"):
            return True, f"confidence {confidence_str} too low — escalating"

        if quality_score < self.ESCALATION_CONFIDENCE_THRESHOLD:
            return True, f"psychological quality {quality_score:.2f} below threshold"

        return False, "result acceptable at this level"

    def escalate(self, current_level: str) -> str:
        """Move to the next compute level."""
        idx = self.LEVELS.index(current_level)
        if idx < len(self.LEVELS) - 1:
            return self.LEVELS[idx + 1]
        return current_level

    def route_and_run(
        self,
        question:       str,
        domain:         str,
        pipeline,
        run_at_level:   Callable,    # fn(question, domain, level) → result
        max_escalations: int = 2,
    ) -> tuple[dict, RouteRecord]:
        """
        Full routing + running + escalation loop.

        run_at_level(question, domain, level) → result dict

        Returns (result, record).
        """
        decision = self.classify(question, domain)
        level    = decision.level
        t0       = time.time()
        escalated = False
        n_escalations = 0

        result = {}
        while n_escalations <= max_escalations:
            result = run_at_level(question, domain, level)

            reasoning = result.get("reasoning")
            verified  = result.get("verified", False)
            confidence = result.get("confidence", "UNCERTAIN")
            quality   = result.get("quality", 0.5)

            should_esc, esc_reason = self.should_escalate(
                level, verified, confidence, quality
            )

            if should_esc and n_escalations < max_escalations:
                new_level = self.escalate(level)
                if new_level != level:
                    if self.verbose:
                        print(f"  [router] Escalating {level} → {new_level}: {esc_reason}")
                    result["escalation_note"] = f"Escalated to {new_level}: {esc_reason}"
                    level = new_level
                    escalated = True
                    n_escalations += 1
                    continue
            break

        elapsed = time.time() - t0

        # Record this run
        import hashlib
        q_hash = hashlib.sha256(question.encode()).hexdigest()[:10]
        record = RouteRecord(
            question_hash=q_hash,
            domain=domain,
            chosen_level=decision.level,
            final_level=level,
            estimated_sec=decision.estimated_seconds,
            actual_sec=round(elapsed, 1),
            verified=result.get("verified", False),
            escalated=escalated,
        )
        self._record(record, domain)

        result["route"] = {
            "level":     level,
            "escalated": escalated,
            "elapsed":   round(elapsed, 1),
            "estimated": decision.estimated_seconds,
        }
        return result, record

    # ── Time estimation ────────────────────────────────────

    def _estimate_time(self, level: str, domain: str = "") -> int:
        """Estimate seconds based on level and domain history."""
        base = self.TIME_ESTIMATES[level]

        # Refine from domain history
        domain_records = [
            r for r in self._records[-50:]
            if r.get("domain") == domain
            and r.get("final_level") == level
        ]
        if len(domain_records) >= 3:
            avg_actual = sum(r["actual_sec"] for r in domain_records) / len(domain_records)
            # Blend base estimate with observed average
            base = int(0.4 * base + 0.6 * avg_actual)

        return base

    def estimate_display(self, question: str, domain: str) -> str:
        """Human-readable time estimate for display before processing."""
        decision = self.classify(question, domain)
        sec = decision.estimated_seconds
        if sec < 60:
            return f"~{sec}s ({decision.level})"
        return f"~{sec//60}m ({decision.level})"

    # ── Domain learning ────────────────────────────────────

    def _record(self, record: RouteRecord, domain: str) -> None:
        self._records.append(vars(record))
        if len(self._records) > 200:
            self._records = self._records[-200:]

        # Update domain baseline
        domain_records = [
            r for r in self._records[-30:]
            if r.get("domain") == domain
        ]
        if len(domain_records) >= 5:
            level_counts: dict[str, int] = {}
            for r in domain_records:
                l = r.get("final_level", "light")
                level_counts[l] = level_counts.get(l, 0) + 1
            most_common = max(level_counts, key=level_counts.__getitem__)
            self._domain_baselines[domain] = most_common

        self._save()

    def stats(self) -> dict:
        """Routing statistics."""
        if not self._records:
            return {}
        recent = self._records[-50:]
        avg_time = sum(r.get("actual_sec", 0) for r in recent) / len(recent)
        escalation_rate = sum(1 for r in recent if r.get("escalated")) / len(recent)
        level_dist: dict[str, int] = {}
        for r in recent:
            l = r.get("final_level", "?")
            level_dist[l] = level_dist.get(l, 0) + 1
        return {
            "total_routed":     len(self._records),
            "avg_time_sec":     round(avg_time, 1),
            "escalation_rate":  round(escalation_rate, 3),
            "level_distribution": level_dist,
            "domain_baselines": self._domain_baselines,
        }

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._records          = d.get("records", [])
            self._domain_baselines = d.get("domain_baselines", {})
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "records":          self._records[-200:],
                "domain_baselines": self._domain_baselines,
            }, f, indent=2)
