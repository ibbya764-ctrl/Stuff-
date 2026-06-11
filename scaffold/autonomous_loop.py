"""
autonomous_loop.py
==================

The system asking itself questions when no users are active.

The DMN generates spontaneous questions. The curiosity engine flags
gaps. The belief system identifies things it's uncertain about.
This loop picks those up and actually processes them — running them
through the full pipeline, accumulating verified training examples,
improving without anyone asking anything.

This is how the system gets smarter overnight.

Sources of autonomous questions (in priority order):
  1. DMN spontaneous_question — what it was wondering about
  2. Weakest beliefs — challenge what it's least confident in
  3. Unresolved obligations — things flagged as needing verification
  4. Cross-domain connections — "if A maps to B, what about C?"
  5. Domain gaps — areas with little verified knowledge

Behaviour:
  - Checks if a user request is active before each question
  - Backs off when the system is under load
  - Rate limits itself (max N questions per hour)
  - Logs every question asked and answer verified

Start it:
    from autonomous_loop import AutonomousLoop, start_loop
    loop = start_loop(scaffold, collective, questions_per_hour=4)
"""

import time
import json
import os
import threading
import random
from dataclasses import dataclass, field
from typing import Optional, Callable

_global_loop: Optional["AutonomousLoop"] = None


def get_loop() -> Optional["AutonomousLoop"]:
    return _global_loop


def start_loop(
    scaffold:          dict,
    collective         = None,
    process_fn:        Optional[Callable] = None,
    questions_per_hour: int = 4,
    base_dir:          str = "./scaffold_data",
    verbose:           bool = True,
) -> "AutonomousLoop":
    global _global_loop
    _global_loop = AutonomousLoop(
        scaffold=scaffold,
        collective=collective,
        process_fn=process_fn,
        questions_per_hour=questions_per_hour,
        base_dir=base_dir,
        verbose=verbose,
    )
    _global_loop.start()
    return _global_loop


@dataclass
class AutonomousRun:
    question:   str
    domain:     str
    source:     str       # where the question came from
    verified:   bool
    response:   str
    timestamp:  float = field(default_factory=time.time)
    elapsed:    float = 0.0

    def to_dict(self):
        return {
            "question":  self.question[:120],
            "domain":    self.domain,
            "source":    self.source,
            "verified":  self.verified,
            "response":  self.response[:200],
            "timestamp": self.timestamp,
            "elapsed":   round(self.elapsed, 1),
        }


class AutonomousLoop:
    """
    Background loop: the system questions itself when idle.
    
    Runs autonomously, respects active user sessions,
    rate limits itself, and logs all activity.
    """

    def __init__(
        self,
        scaffold:           dict,
        collective          = None,
        process_fn:         Optional[Callable] = None,
        questions_per_hour: int  = 4,
        base_dir:           str  = "./scaffold_data",
        verbose:            bool = True,
    ):
        self.scaffold    = scaffold
        self.collective  = collective
        self.process_fn  = process_fn
        self.qph         = questions_per_hour
        self.base_dir    = base_dir
        self.verbose     = verbose

        self.interval    = max(300, 3600 // max(1, questions_per_hour))
        self._running    = False
        self._active_requests = 0
        self._lock       = threading.Lock()
        self._thread:    Optional[threading.Thread] = None
        self._history:   list[dict] = []
        self._stats      = {
            "total_asked":    0,
            "total_verified": 0,
            "started_at":     0.0,
            "last_question":  "",
            "last_ran":       0.0,
        }
        self._log_path = os.path.join(base_dir, "autonomous_log.json")
        self._load()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stats["started_at"] = time.time()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="BriAutonomous"
        )
        self._thread.start()
        if self.verbose:
            print(f"[autonomous] Started — {self.qph} questions/hour "
                  f"(every {self.interval//60}min)")

    def stop(self) -> None:
        self._running = False

    def mark_request_active(self) -> None:
        with self._lock:
            self._active_requests += 1

    def mark_request_done(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def status(self) -> dict:
        return {
            "running":        self._running,
            "total_asked":    self._stats["total_asked"],
            "total_verified": self._stats["total_verified"],
            "last_question":  self._stats["last_question"][:80],
            "last_ran":       self._stats["last_ran"],
            "interval_min":   self.interval // 60,
            "active_requests": self._active_requests,
        }

    def full_status(self) -> dict:
        s = self.status()
        s["recent"] = [
            {"question": r["question"][:60], "verified": r["verified"],
             "source": r["source"], "timestamp": r["timestamp"]}
            for r in self._history[-10:]
        ]
        return s

    # ── Main loop ─────────────────────────────────────────

    def _loop(self) -> None:
        # Initial wait — don't fire immediately on startup
        time.sleep(60)
        while self._running:
            # Skip if users are active
            if self._active_requests > 0:
                time.sleep(30)
                continue

            try:
                self._one_cycle()
            except Exception as e:
                if self.verbose:
                    print(f"[autonomous] Cycle error: {e}")

            time.sleep(self.interval)

    def _one_cycle(self) -> None:
        """Ask one question autonomously."""
        question, domain, source = self._pick_question()
        if not question:
            return

        if self.verbose:
            print(f"\n[autonomous] Self-question ({source}): {question[:70]}")

        t0 = time.time()
        self._stats["last_question"] = question
        self._stats["last_ran"]      = time.time()

        # Process through pipeline
        result = self._process(question, domain)

        elapsed  = time.time() - t0
        verified = result.get("verified", False)
        response = result.get("response", "")

        run = AutonomousRun(
            question=question, domain=domain, source=source,
            verified=verified, response=response, elapsed=elapsed,
        )
        self._history.append(run.to_dict())
        if len(self._history) > 200:
            self._history = self._history[-200:]

        self._stats["total_asked"]    += 1
        if verified:
            self._stats["total_verified"] += 1

        if self.verbose:
            status_str = "✓ verified" if verified else "○ unverified"
            print(f"  [autonomous] {status_str} in {elapsed:.0f}s — {response[:60]}")

        self._save()

    def _process(self, question: str, domain: str) -> dict:
        """Process a question through the pipeline."""
        if self.process_fn:
            try:
                return self.process_fn(question, domain, "autonomous", self.scaffold, self.collective)
            except Exception:
                pass

        # Direct pipeline call fallback
        sc = self.scaffold
        try:
            pipeline = sc.get("pipeline")
            if not pipeline:
                return {}

            sc["dmn"].pause_for_task()
            try:
                ctx_parts = []
                try: ctx_parts.append(sc["brain"].pre_reasoning_context(question, domain))
                except Exception: pass
                try: ctx_parts.append(sc["dmn"].get_context())
                except Exception: pass

                ctx = "\n\n".join(c for c in ctx_parts if c.strip())
                full_q = f"{question}\n\n{ctx}" if ctx else question

                result = pipeline.ask(full_q, domain=domain)
                r = result.get("reasoning")
                if r:
                    try: sc["brain"].post_reasoning_update(r, question, domain, success=r.verified)
                    except Exception: pass
                    if r.verified:
                        try: sc["from_scratch"].add_verified_run(r, question)
                        except Exception: pass
                        try:
                            if self.collective:
                                self.collective.contribute(r, question, "autonomous", {
                                    "messages": [{"role":"user","content":question},
                                                 {"role":"assistant","content":result.get("response","")}]
                                })
                        except Exception: pass
                return result
            finally:
                sc["dmn"].resume()
        except Exception as e:
            if self.verbose:
                print(f"[autonomous] Process error: {e}")
            return {}

    # ── Question sources ──────────────────────────────────

    def _pick_question(self) -> tuple[str, str, str]:
        """
        Pick the next question to ask. Returns (question, domain, source).
        Returns ("", "", "") if nothing available.
        """
        # Weight the sources — varied, not always the same type
        roll = random.random()

        if roll < 0.35:
            q, d = self._from_dmn()
            if q: return q, d, "dmn_spontaneous"

        if roll < 0.55:
            q, d = self._from_weak_beliefs()
            if q: return q, d, "belief_challenge"

        if roll < 0.70:
            q, d = self._from_cross_domain()
            if q: return q, d, "cross_domain"

        if roll < 0.85:
            q, d = self._from_obligations()
            if q: return q, d, "obligation"

        q, d = self._from_domain_gap()
        if q: return q, d, "domain_exploration"

        # Fallback: try each source
        for fn in [self._from_dmn, self._from_weak_beliefs,
                   self._from_cross_domain, self._from_obligations]:
            q, d = fn()
            if q: return q, d, "fallback"

        return "", "", ""

    def _from_dmn(self) -> tuple[str, str]:
        dmn = self.scaffold.get("dmn")
        if not dmn: return "", ""
        try:
            q = dmn.workspace.spontaneous_question
            if q and len(q) > 10:
                domain = dmn.workspace.domains_active[0] if dmn.workspace.domains_active else "general"
                return q, domain
        except Exception: pass
        return "", ""

    def _from_weak_beliefs(self) -> tuple[str, str]:
        plastic = self.scaffold.get("plastic")
        if not plastic: return "", ""
        try:
            weak = plastic.beliefs.weakest_beliefs(n=3)
            if weak:
                b = random.choice(weak)
                q = (f"What evidence would confirm or refute the claim: "
                     f"'{b.statement}'? Reason carefully.")
                return q, "general"
        except Exception: pass
        return "", ""

    def _from_cross_domain(self) -> tuple[str, str]:
        dmn = self.scaffold.get("dmn")
        if not dmn: return "", ""
        try:
            insight = dmn.workspace.cross_domain_insight
            if insight and len(insight) > 20:
                q = (f"This connection was noticed: '{insight}'. "
                     f"What are the implications of this structural similarity? "
                     f"Push it further — where does the analogy break?")
                return q, "general"
        except Exception: pass
        return "", ""

    def _from_obligations(self) -> tuple[str, str]:
        sc = self.scaffold
        try:
            plastic = sc.get("plastic")
            if plastic:
                tensions = plastic.tension.active_tensions() if hasattr(plastic, 'tension') else []
                if tensions:
                    t = tensions[0]
                    q = f"Resolve this tension: {str(t)[:100]}"
                    return q, "general"
        except Exception: pass
        return "", ""

    def _from_domain_gap(self) -> tuple[str, str]:
        """Ask about a domain where knowledge is thin."""
        knowledge = self.scaffold.get("knowledge")
        if not knowledge: return "", ""
        try:
            domains = ["physics", "mathematics", "biology", "economics",
                       "philosophy", "computer_science", "history"]
            random.shuffle(domains)
            for domain in domains:
                profile = knowledge.knowledge_base.domain_profile(domain)
                if profile.get("depth") in ("none", "beginning"):
                    questions = {
                        "physics":    "What is the most fundamental unsolved problem in physics right now?",
                        "mathematics": "What does Gödel's incompleteness theorem actually prove, and why does it matter?",
                        "biology":    "How does epigenetic inheritance challenge classical Darwinian evolution?",
                        "economics":  "What is the relationship between information asymmetry and market failure?",
                        "philosophy": "What is the strongest argument against physicalism about consciousness?",
                        "computer_science": "What is P vs NP and why can't we resolve it?",
                        "history":    "What structural factors explain the collapse of Bronze Age civilizations?",
                    }
                    if domain in questions:
                        return questions[domain], domain
        except Exception: pass
        return "", ""

    # ── Persistence ───────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._log_path):
            return
        try:
            with open(self._log_path) as f:
                d = json.load(f)
            self._history = d.get("history", [])
            self._stats.update(d.get("stats", {}))
        except Exception: pass

    def _save(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        with open(self._log_path, "w") as f:
            json.dump({"history": self._history[-200:], "stats": self._stats}, f)
