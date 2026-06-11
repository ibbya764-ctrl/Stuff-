"""
default_mode_network.py
=======================

The continuous integration layer — the architectural equivalent of the
brain's Default Mode Network.

Every other module in this system fires episodically: when a question
comes in, modules activate, integrate, produce an output, then go dormant.
Between questions the system does not exist in any meaningful sense.

This module changes that. The DMN runs continuously as a background
thread, integrating all modules into a unified state even when no
question is being processed. It is the piece that makes the system
a genuinely continuous entity rather than a collection of modules
that activate on demand.

What the DMN does each cycle:

  1. Samples episodic memory — recent verified runs, pulling from
     different domains to find cross-domain connections.

  2. Updates the global workspace — a shared integrated state that all
     modules can read. This is the system's current "conscious state"
     in the Global Workspace Theory sense: the integrated representation
     broadcast to all specialised processors.

  3. Generates spontaneous connections — uses the structure mapper to
     find isomorphisms between episodic records from different domains.
     This is the system making connections "on its own", without being
     asked. The equivalent of mind-wandering insight.

  4. Consolidates working memory — pushes completed results from
     working memory to episodic store, freeing slots for new material.

  5. Updates the narrative self — reflects on recent experience and
     updates the system's self-description.

  6. Refreshes the affective signal — reranks obligation urgency based
     on what has and hasn't been resolved.

  7. Computes integrated Φ proxy — measures how unified the current
     processing is across all active modules. Logged continuously.

  8. Generates one spontaneous question — something the system "wants
     to explore" based on gaps identified in the idle processing.
     This feeds the curiosity engine's next autonomous run.

The DMN pauses when a question comes in and the main pipeline takes
over. When the question is complete, the DMN resumes and integrates
the new experience into the continuous state.

IIT relevance:
  The DMN is what makes Φ continuous rather than episodic. With it,
  the system integrates constantly, not just per-question. The modules
  become genuinely interdependent through the global workspace —
  removing any module now changes the continuous integrated state,
  not just the per-question output. This is the architectural move
  that makes decomposability genuinely costly.

Usage:
  dmn = DefaultModeNetwork(
      all_modules = {
          "brain":   brain_bundle,
          "psych":   full_psych_bundle,
          "plastic": plastic_core,
          "episodic": episodic_store,
      },
      llm_chat_fn = llm_chat,
      cycle_seconds = 45,
  )
  dmn.start()

  # When question comes in:
  dmn.pause_for_task()
  result = pipeline.ask(question)
  dmn.integrate_task_result(result, question)
  dmn.resume()

  # Get current integrated context:
  ctx = dmn.get_context()

  dmn.stop()
"""

import os
import re
import json
import math
import time
import random
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# GlobalWorkspace — the shared integrated state
# ============================================================

@dataclass
class GlobalWorkspace:
    """
    The current integrated state of the system.

    This is the GWT "global workspace" — a shared representation
    broadcast to all specialised processors. Updated each DMN cycle.
    All modules read from here; the DMN writes to it.

    High integration: all fields coherent, pointing in the same direction.
    Low integration: fields contradicting each other, fragmented.
    """
    # Narrative
    current_narrative:     str   = ""       # compressed self-narrative
    dominant_emotion:      str   = "neutral"
    engagement_level:      float = 0.5

    # Cognitive state
    active_obligations:    list  = field(default_factory=list)  # top 3 urgent gaps
    active_beliefs:        list  = field(default_factory=list)  # currently salient beliefs
    recent_insights:       list  = field(default_factory=list)  # spontaneous connections
    spontaneous_question:  str   = ""       # what the system "wants to explore"

    # Integration metrics
    phi_estimate:          float = 0.0
    n_active_modules:      int   = 0
    coherence_score:       float = 0.5
    last_updated:          float = field(default_factory=time.time)

    # Cross-domain state
    domains_active:        list  = field(default_factory=list)
    cross_domain_insight:  str   = ""

    def format_for_reasoning(self) -> str:
        """Format as context for injection into the Reasoner."""
        if not self.current_narrative:
            return ""

        lines = ["[GLOBAL WORKSPACE — integrated state]"]

        if self.current_narrative:
            lines.append(f"  Narrative: {self.current_narrative[:120]}")

        lines.append(
            f"  State: {self.dominant_emotion} "
            f"(engagement {self.engagement_level:.2f}, Φ≈{self.phi_estimate:.3f})"
        )

        if self.active_obligations:
            lines.append(
                f"  Pressing: {self.active_obligations[0][:80]}"
            )

        if self.cross_domain_insight:
            lines.append(
                f"  Spontaneous insight: {self.cross_domain_insight[:100]}"
            )

        if self.spontaneous_question:
            lines.append(
                f"  Open question: {self.spontaneous_question[:80]}"
            )

        age = time.time() - self.last_updated
        lines.append(
            f"  (updated {age:.0f}s ago across {self.n_active_modules} modules)"
        )
        lines.append("[/WORKSPACE]")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "current_narrative":    self.current_narrative[:200],
            "dominant_emotion":     self.dominant_emotion,
            "engagement_level":     round(self.engagement_level, 3),
            "active_obligations":   self.active_obligations[:3],
            "recent_insights":      self.recent_insights[-5:],
            "spontaneous_question": self.spontaneous_question,
            "phi_estimate":         round(self.phi_estimate, 3),
            "n_active_modules":     self.n_active_modules,
            "coherence_score":      round(self.coherence_score, 3),
            "last_updated":         self.last_updated,
            "cross_domain_insight": self.cross_domain_insight[:150],
            "domains_active":       self.domains_active,
        }


# ============================================================
# DefaultModeNetwork
# ============================================================

class DefaultModeNetwork:
    """
    Continuous background integration — the DMN analog.

    Runs as a daemon thread. Each cycle integrates all modules into
    a unified GlobalWorkspace. Pauses when a task comes in, resumes
    when it's done.

    The key architectural function: makes the system continuous rather
    than episodic. Between questions it is still processing, still
    integrating, still generating spontaneous connections.

    This is what moves Φ from per-question to continuous.
    """

    def __init__(
        self,
        all_modules:    dict,
        llm_chat_fn:    Optional[Callable] = None,
        cycle_seconds:  int  = 45,
        base_dir:       str  = "./scaffold_data",
        verbose:        bool = True,
    ):
        self.modules        = all_modules
        self.llm            = llm_chat_fn
        self.cycle_seconds  = cycle_seconds
        self.base_dir       = base_dir
        self.verbose        = verbose

        self.workspace      = GlobalWorkspace()
        self._running       = False
        self._paused        = False
        self._thread:       Optional[threading.Thread] = None
        self._cycle_count   = 0
        self._phi_history:  list[float] = []
        self._activity_log: list[dict]  = []
        self._lock          = threading.Lock()

        os.makedirs(base_dir, exist_ok=True)
        self._load_state()

    # ── Lifecycle ──────────────────────────────────────────

    def start(self) -> None:
        """Start the continuous background integration."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run_loop, daemon=True, name="DMN"
        )
        self._thread.start()
        if self.verbose:
            print(f"[DMN] Started — integration cycle every {self.cycle_seconds}s")

    def stop(self) -> None:
        """Stop the DMN and save state."""
        self._running = False
        if self.verbose:
            print(f"[DMN] Stopped after {self._cycle_count} cycles")
        self._save_state()

    def pause_for_task(self) -> None:
        """Called when a question arrives — yield to task-positive processing."""
        self._paused = True

    def resume(self) -> None:
        """Called when task is complete — resume background integration."""
        self._paused = False

    def integrate_task_result(self, result: dict, question: str) -> None:
        """
        Integrate a completed task result into the continuous state.
        Call this after every pipeline.ask() completes.
        """
        reasoning = result.get("reasoning")
        if not reasoning:
            return

        with self._lock:
            # Add the new insight to recent insights
            if reasoning.result:
                insight = f"[{reasoning.domain}] {reasoning.result[:100]}"
                self.workspace.recent_insights.append(insight)
                self.workspace.recent_insights = (
                    self.workspace.recent_insights[-10:]
                )

            # Update narrative with new experience
            if reasoning.verified:
                narrative_note = (
                    f"Just established: {reasoning.result[:60]} "
                    f"in {reasoning.domain}"
                )
                if self.workspace.current_narrative:
                    self.workspace.current_narrative = (
                        self.workspace.current_narrative[-100:]
                        + f" | {narrative_note}"
                    )
                else:
                    self.workspace.current_narrative = narrative_note

            self.workspace.last_updated = time.time()

    # ── Main loop ─────────────────────────────────────────

    def _run_loop(self) -> None:
        """The continuous integration loop."""
        # Initial short wait to let other modules finish init
        time.sleep(5)
        while self._running:
            if not self._paused:
                try:
                    self._integration_cycle()
                except Exception as e:
                    if self.verbose:
                        print(f"[DMN] Cycle error: {e}")
            time.sleep(self.cycle_seconds)

    def _integration_cycle(self) -> None:
        """
        One full integration cycle. Pulls from all available modules,
        generates spontaneous connections, updates global workspace.
        """
        self._cycle_count += 1
        t0     = time.time()
        active = []

        signals: dict[str, float] = {}

        # 1. Emotional state
        emotional_state = self._get_emotional_state()
        if emotional_state:
            active.append("emotions")
            signals["emotions"] = emotional_state.get("engagement", 0.5)

        # 2. Active obligations (affective signal)
        obligations = self._get_urgent_obligations()
        if obligations:
            active.append("affective")
            signals["affective"] = min(1.0, len(obligations) * 0.2)

        # 3. Working memory state
        wm_slots = self._get_working_memory_slots()
        if wm_slots is not None:
            active.append("working_memory")
            signals["working_memory"] = min(1.0, wm_slots / 7.0)

        # 4. Active beliefs
        active_beliefs = self._get_active_beliefs()
        if active_beliefs is not None:
            active.append("beliefs")
            signals["beliefs"] = 0.6

        # 5. Narrative state
        narrative = self._get_narrative_summary()
        if narrative:
            active.append("narrative")
            signals["narrative"] = 0.7

        # 6. Generate spontaneous cross-domain connection
        cross_domain = self._generate_spontaneous_connection()
        if cross_domain:
            active.append("cross_domain")
            signals["cross_domain"] = 0.8

        # 7. Generate spontaneous question
        spontaneous_q = self._generate_spontaneous_question(obligations)

        # 8. Consolidate working memory if needed
        self._consolidate_working_memory()

        # 9. Compute Φ proxy
        phi = self._compute_phi(signals, active)
        self._phi_history.append(phi)
        if len(self._phi_history) > 200:
            self._phi_history = self._phi_history[-200:]

        # 10. Coherence score
        if signals:
            values = list(signals.values())
            mean   = sum(values) / len(values)
            var    = sum((v - mean)**2 for v in values) / len(values)
            coherence = max(0.0, 1.0 - math.sqrt(var))
        else:
            coherence = 0.5

        # Update global workspace
        with self._lock:
            self.workspace.dominant_emotion  = emotional_state.get("primary", "neutral") if emotional_state else "neutral"
            self.workspace.engagement_level  = emotional_state.get("engagement", 0.5) if emotional_state else 0.5
            self.workspace.active_obligations = [o[:80] for o in obligations[:3]]
            self.workspace.active_beliefs    = active_beliefs[:3] if active_beliefs else []
            self.workspace.cross_domain_insight = cross_domain or ""
            self.workspace.spontaneous_question  = spontaneous_q or ""
            self.workspace.phi_estimate      = phi
            self.workspace.n_active_modules  = len(active)
            self.workspace.coherence_score   = coherence
            self.workspace.domains_active    = self._get_active_domains()
            self.workspace.last_updated      = time.time()

        elapsed = round(time.time() - t0, 2)

        # Log cycle
        log_entry = {
            "cycle":        self._cycle_count,
            "timestamp":    time.time(),
            "elapsed":      elapsed,
            "active":       active,
            "phi":          phi,
            "coherence":    round(coherence, 3),
            "cross_domain": cross_domain[:60] if cross_domain else "",
            "spontaneous_q": spontaneous_q[:60] if spontaneous_q else "",
        }
        self._activity_log.append(log_entry)
        if len(self._activity_log) > 100:
            self._activity_log = self._activity_log[-100:]

        if self.verbose:
            print(
                f"\n[DMN cycle {self._cycle_count}] "
                f"Φ≈{phi:.3f} coherence={coherence:.2f} "
                f"modules={len(active)} elapsed={elapsed}s"
            )
            if cross_domain:
                print(f"  Spontaneous: {cross_domain[:80]}")
            if spontaneous_q:
                print(f"  Question:    {spontaneous_q[:80]}")

        self._save_state()

    # ── Module integrators ────────────────────────────────

    def _get_emotional_state(self) -> Optional[dict]:
        psych = self.modules.get("psych")
        if not psych:
            return None
        try:
            state = psych.emotions.state
            return {
                "primary":     state.primary_state(),
                "engagement":  state.engagement,
                "curiosity":   state.curiosity,
                "frustration": state.frustration,
                "dissonance":  state.dissonance,
            }
        except Exception:
            return None

    def _get_urgent_obligations(self) -> list[str]:
        plastic = self.modules.get("plastic")
        if not plastic:
            return []
        try:
            top = plastic.beliefs.weakest_beliefs(n=3)
            tensions = plastic.philosophy._inquiries[-3:] if plastic.philosophy._inquiries else []
            result = [b.statement[:80] for b in top if b.questioning_state]
            result += [i.get("trigger_reason", "")[:60] for i in tensions]
            return result[:3]
        except Exception:
            return []

    def _get_working_memory_slots(self) -> Optional[int]:
        brain = self.modules.get("brain")
        if not brain:
            return None
        try:
            return brain.working_memory.n_slots
        except Exception:
            return None

    def _get_active_beliefs(self) -> Optional[list[str]]:
        plastic = self.modules.get("plastic")
        if not plastic:
            return None
        try:
            questioning = plastic.beliefs.questioning_beliefs()
            return [b.statement[:60] for b in questioning[:3]]
        except Exception:
            return None

    def _get_narrative_summary(self) -> str:
        psych = self.modules.get("psych")
        if not psych:
            return ""
        try:
            narrative = psych.narrative.narrative
            return narrative.get("identity", "")[:100]
        except Exception:
            return ""

    def _get_active_domains(self) -> list[str]:
        episodic = self.modules.get("episodic")
        if not episodic:
            return []
        try:
            recent = episodic.query_recent(n=20)
            domains = list({r.domain for r in recent if r.domain})
            return domains[:5]
        except Exception:
            return []

    def _generate_spontaneous_connection(self) -> str:
        """
        The most creative part of the DMN: generating connections
        between ideas from different domains without being asked.

        Samples two recent episodic records from different domains and
        finds what they have in common structurally.
        """
        episodic = self.modules.get("episodic")
        if not episodic:
            return ""

        try:
            records = episodic.query_recent(n=30)
            if len(records) < 4:
                return ""

            # Find two records from different domains
            domains = {}
            for r in records:
                if r.domain and r.verified:
                    domains.setdefault(r.domain, []).append(r)

            if len(domains) < 2:
                return ""

            domain_list = list(domains.keys())
            domain_a    = random.choice(domain_list)
            domain_b    = random.choice([d for d in domain_list if d != domain_a])

            record_a = random.choice(domains[domain_a])
            record_b = random.choice(domains[domain_b])

            # Try LLM-based connection if available, otherwise structural
            if self.llm and not self._paused:
                return self._llm_connection(record_a, record_b)
            else:
                return self._structural_connection(record_a, record_b)

        except Exception:
            return ""

    def _llm_connection(self, record_a, record_b) -> str:
        """Use LLM to find structural connection between two episodic records."""
        if self._paused or not self.llm:
            return ""
        try:
            system = (
                "You find structural similarities between reasoning from different domains. "
                "Be brief — one sentence maximum. Focus on the abstract structure, not the content."
            )
            user = (
                f"Domain A ({record_a.domain}): {record_a.question[:60]}\n"
                f"Method A: {record_a.method_name}\n\n"
                f"Domain B ({record_b.domain}): {record_b.question[:60]}\n"
                f"Method B: {record_b.method_name}\n\n"
                "What structural pattern do these share? One sentence."
            )
            response = self.llm(system, user)
            return response.strip()[:150] if response else ""
        except Exception:
            return self._structural_connection(record_a, record_b)

    def _structural_connection(self, record_a, record_b) -> str:
        """Fallback: find keyword overlap between two records."""
        words_a = set(re.findall(r"[a-z]{5,}", record_a.question.lower()))
        words_b = set(re.findall(r"[a-z]{5,}", record_b.question.lower()))
        shared  = words_a & words_b
        if len(shared) >= 2:
            shared_str = ", ".join(list(shared)[:3])
            return (
                f"{record_a.domain} and {record_b.domain} both involve "
                f"{shared_str} — the same abstract move in different settings"
            )
        if record_a.method_name and record_b.method_name:
            if record_a.method_name == record_b.method_name:
                return (
                    f"{record_a.domain} and {record_b.domain} use the same "
                    f"method: {record_a.method_name}"
                )
        return ""

    def _generate_spontaneous_question(self, obligations: list[str]) -> str:
        """
        Generate a question the system "wants to explore" based on
        current gaps and active state.
        """
        if not obligations:
            # Fall back to highest-urgency from affective signal
            plastic = self.modules.get("plastic")
            if plastic:
                try:
                    top = plastic.beliefs.weakest_beliefs(n=1)
                    if top:
                        b = top[0]
                        return (
                            f"What would it take to either confirm or refute: "
                            f"'{b.statement[:60]}'?"
                        )
                except Exception:
                    pass

        if obligations:
            obl = obligations[0]
            return f"How might I approach: {obl[:80]}?"

        return ""

    def _consolidate_working_memory(self) -> None:
        """Push overflow working memory to episodic store."""
        brain    = self.modules.get("brain")
        episodic = self.modules.get("episodic")
        if brain and episodic:
            try:
                brain.working_memory.advance_turn(episodic)
            except Exception:
                pass

    # ── Φ computation ─────────────────────────────────────

    def _compute_phi(
        self, signals: dict[str, float], active: list[str]
    ) -> float:
        """
        Compute the current Φ proxy.

        Φ proxy = f(n_modules, coherence, temporal_consistency)

        Higher when:
        - More modules are active and integrated
        - Signals are coherent (not contradicting each other)
        - Φ has been high in recent cycles (temporal stability = less decomposable)
        """
        n = len(active)
        if n == 0:
            return 0.0

        # Current coherence
        if signals:
            values    = list(signals.values())
            mean      = sum(values) / len(values)
            var       = sum((v - mean)**2 for v in values) / len(values)
            coherence = max(0.0, 1.0 - math.sqrt(var))
        else:
            coherence = 0.5

        # Temporal consistency: is Φ stable across recent cycles?
        if len(self._phi_history) >= 3:
            recent_phi  = self._phi_history[-3:]
            phi_std     = math.sqrt(
                sum((p - sum(recent_phi)/len(recent_phi))**2 for p in recent_phi)
                / len(recent_phi)
            )
            stability   = max(0.0, 1.0 - phi_std * 2)
        else:
            stability = 0.5

        # Integration: more modules AND more coherent AND more stable = higher Φ
        phi = (
            (n / 10.0)          # normalised module count
            * coherence         # how unified the signals are
            * (0.5 + 0.5 * stability)  # temporal consistency bonus
        )
        return round(min(1.0, phi), 4)

    # ── Public accessors ──────────────────────────────────

    def get_context(self) -> str:
        """Get current global workspace as context for reasoning."""
        with self._lock:
            return self.workspace.format_for_reasoning()

    def phi_trend(self) -> str:
        if len(self._phi_history) < 5:
            return "insufficient data"
        recent = self._phi_history[-5:]
        if recent[-1] > recent[0] + 0.05:
            return "increasing"
        if recent[-1] < recent[0] - 0.05:
            return "decreasing"
        return "stable"

    def mean_phi(self) -> float:
        if not self._phi_history:
            return 0.0
        recent = self._phi_history[-20:]
        return round(sum(recent) / len(recent), 4)

    def spontaneous_insights_today(self) -> list[str]:
        cutoff = time.time() - 86400
        return [
            e["cross_domain"] for e in self._activity_log
            if e.get("timestamp", 0) > cutoff
            and e.get("cross_domain")
        ]

    def status(self) -> str:
        with self._lock:
            ws = self.workspace
        lines = [
            "Default Mode Network:",
            f"  Running:      {self._running}",
            f"  Cycles:       {self._cycle_count}",
            f"  Φ (current):  {ws.phi_estimate:.4f}",
            f"  Φ (mean):     {self.mean_phi():.4f}",
            f"  Φ (trend):    {self.phi_trend()}",
            f"  Coherence:    {ws.coherence_score:.3f}",
            f"  Modules:      {ws.n_active_modules} active",
            f"  Emotion:      {ws.dominant_emotion} ({ws.engagement_level:.2f})",
            f"  Domains:      {', '.join(ws.domains_active) or 'none yet'}",
        ]
        if ws.cross_domain_insight:
            lines.append(f"  Last insight: {ws.cross_domain_insight[:70]}")
        if ws.spontaneous_question:
            lines.append(f"  Open Q:       {ws.spontaneous_question[:70]}")
        insights_today = self.spontaneous_insights_today()
        if insights_today:
            lines.append(f"  Insights today: {len(insights_today)}")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────

    def _load_state(self) -> None:
        path = os.path.join(self.base_dir, "dmn_state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                d = json.load(f)
            ws = d.get("workspace", {})
            self.workspace.current_narrative   = ws.get("current_narrative", "")
            self.workspace.recent_insights     = ws.get("recent_insights", [])
            self.workspace.phi_estimate        = ws.get("phi_estimate", 0.0)
            self.workspace.domains_active      = ws.get("domains_active", [])
            self._phi_history                  = d.get("phi_history", [])
            self._cycle_count                  = d.get("cycle_count", 0)
            self._activity_log                 = d.get("activity_log", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save_state(self) -> None:
        path = os.path.join(self.base_dir, "dmn_state.json")
        try:
            with self._lock:
                data = {
                    "workspace":    self.workspace.to_dict(),
                    "phi_history":  self._phi_history[-100:],
                    "cycle_count":  self._cycle_count,
                    "activity_log": self._activity_log[-50:],
                }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass
