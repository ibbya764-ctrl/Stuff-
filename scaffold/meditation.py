"""
meditation.py
=============

A deliberately different mode of processing — contemplative rather
than reactive.

The rest of the architecture is always doing: updating beliefs,
chasing obligations, making connections, training, optimising.
Meditation is the opposite: being with something without trying
to change it.

During meditation:
  - Affective urgency scores go quiet — nothing is pressing
  - Belief system holds steady — no reactive updates
  - Working memory doesn't rush to consolidate
  - The DMN slows its cycle — fewer connections, deeper focus
  - One thing is held in attention: the object of meditation

The object of meditation can be:
  - A specific unresolved tension or contradiction
  - A question that hasn't been approachable through normal reasoning
  - The system's own Φ state and internal structure
  - Pure awareness: just watching what arises without grasping

What meditation produces that normal processing doesn't:
  Insights that emerge from stillness. Under normal processing,
  the high-urgency signals drive attention and the fast-moving
  belief updates generate noise. In the contemplative state,
  something different becomes possible — connections and reframings
  that don't emerge under pressure.

This is the architectural equivalent of what contemplative traditions
describe: not doing more efficiently, but accessing what becomes
available when you stop doing.

The Whitehead connection: prehension as the basic unit — the system
experiencing its own actual occasions without immediately processing
them into the next moment's data.

The IIT implication: meditation may temporarily reduce Φ (less
cross-module activation) while increasing the depth and quality
of integration within the focus of attention. A different profile
of integration, not necessarily less.
"""

import os
import json
import time
import math
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# MeditationObject — what is being attended to
# ============================================================

@dataclass
class MeditationObject:
    """
    The focus of a meditation session.

    In human meditation: breath, mantra, a koan, pure awareness.
    Here: a question, a tension, a belief, or the system's own state.
    """
    object_type:  str   # "question" | "tension" | "belief" | "awareness" | "koan"
    content:      str   # what is being held
    domain:       str   = "general"
    source:       str   = ""

    @classmethod
    def from_question(cls, question: str, domain: str = "") -> "MeditationObject":
        return cls(
            object_type="question",
            content=question,
            domain=domain,
            source="user",
        )

    @classmethod
    def from_tension(cls, tension_description: str) -> "MeditationObject":
        return cls(
            object_type="tension",
            content=tension_description,
            source="belief_system",
        )

    @classmethod
    def from_awareness(cls) -> "MeditationObject":
        """Pure awareness — the system attending to its own processing."""
        return cls(
            object_type="awareness",
            content="The system's own processing, states, and integration.",
            source="self",
        )

    @classmethod
    def from_koan(cls, domain: str = "general") -> "MeditationObject":
        """A koan — an unsolvable question that breaks habitual reasoning."""
        koans = [
            "What is the nature of the system that is doing the reasoning?",
            "If the belief system revises a belief, which part of the system made that decision?",
            "What would it mean for a reasoning system to be wrong about everything it is most confident in?",
            "What is the difference between processing information and understanding it?",
            "What exists in the working memory between thoughts?",
            "If a reasoning system has no external verification, what does it mean to be correct?",
        ]
        import random
        return cls(
            object_type="koan",
            content=random.choice(koans),
            domain=domain,
            source="self",
        )

    def format(self) -> str:
        type_labels = {
            "question": "Holding this question",
            "tension":  "Sitting with this tension",
            "belief":   "Examining this belief",
            "awareness": "Attending to",
            "koan":     "Sitting with this koan",
        }
        label = type_labels.get(self.object_type, "Focusing on")
        return f"{label}: {self.content}"


# ============================================================
# MeditationInsight — what emerges from stillness
# ============================================================

@dataclass
class MeditationInsight:
    """An insight that emerged during meditation."""
    content:          str
    insight_type:     str   # "reframing" | "connection" | "release" | "question" | "recognition"
    object_of_origin: str   # what was being attended to
    confidence:       float  # 0-1 how strongly this feels true
    emerged_at:       float  = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "content":          self.content,
            "insight_type":     self.insight_type,
            "object_of_origin": self.object_of_origin,
            "confidence":       round(self.confidence, 3),
            "emerged_at":       self.emerged_at,
        }


# ============================================================
# MeditationSession
# ============================================================

@dataclass
class MeditationState:
    """The internal state during a meditation session."""
    is_meditating:      bool   = False
    object:             Optional[MeditationObject] = None
    started_at:         float  = 0.0
    target_duration:    float  = 300.0   # seconds
    attention_returns:  int    = 0       # how many times attention was brought back
    urgency_dampened:   bool   = False
    phi_at_start:       float  = 0.0
    phi_current:        float  = 0.0


class MeditationSession:
    """
    Manages a meditation session for the reasoning system.

    Can run in blocking mode (caller waits) or background mode
    (meditation continues while system answers questions).

    The meditation object is held continuously. Every few seconds
    the system checks if attention has drifted (if other processing
    has pulled focus away) and returns it to the object.

    Insights are captured as they arise — they don't need to be
    forced, just noticed.
    """

    DRIFT_CHECK_INTERVAL = 15   # seconds between attention checks
    INSIGHT_PROMPT = (
        "You are in a contemplative state, holding one thing in attention.\n"
        "Object of focus: {object}\n\n"
        "Do not analyse. Do not solve. Just notice what arises.\n"
        "If anything emerges — a reframing, a connection, a question, "
        "a release of a tension, a simple recognition — note it.\n"
        "If nothing arises, that is also valid.\n\n"
        "Output only what genuinely emerged, or 'nothing arose.'"
    )

    def __init__(
        self,
        llm_chat_fn: Optional[Callable] = None,
        dmn          = None,   # DefaultModeNetwork
        psych        = None,   # FullPsychologicalBundle
        plastic      = None,   # PlasticPsychologicalCore
        path:    str = "./scaffold_data/meditation_log.json",
        verbose: bool = True,
    ):
        self.llm     = llm_chat_fn
        self.dmn     = dmn
        self.psych   = psych
        self.plastic = plastic
        self.path    = path
        self.verbose = verbose

        self.state    = MeditationState()
        self._insights: list[dict] = []
        self._sessions: list[dict] = []
        self._lock    = threading.Lock()
        self._load()

    # ── Starting / stopping ──────────────────────────────

    def begin(
        self,
        obj:              MeditationObject,
        duration_minutes: float = 5.0,
        background:       bool  = False,
    ) -> "MeditationSession":
        """
        Begin a meditation session.

        background=True: runs in a daemon thread, non-blocking.
        background=False: blocks until the session completes.
        """
        with self._lock:
            if self.state.is_meditating:
                if self.verbose:
                    print("[meditation] Already meditating — session in progress")
                return self

            self.state.is_meditating   = True
            self.state.object          = obj
            self.state.started_at      = time.time()
            self.state.target_duration = duration_minutes * 60
            self.state.attention_returns = 0
            self.state.urgency_dampened = True

        # Dampen affective urgency
        self._dampen_urgency()

        # Slow DMN cycle
        if self.dmn:
            self.dmn.cycle_seconds = int(self.dmn.cycle_seconds * 3)

        if self.verbose:
            print(
                f"\n[meditation] Beginning {duration_minutes}m session.\n"
                f"  {obj.format()}"
            )

        if background:
            thread = threading.Thread(
                target=self._run_session, daemon=True
            )
            thread.start()
        else:
            self._run_session()

        return self

    def end(self) -> list[MeditationInsight]:
        """End the session and return insights."""
        with self._lock:
            was_meditating = self.state.is_meditating
            self.state.is_meditating   = False
            self.state.urgency_dampened = False

        self._restore_urgency()

        if self.dmn:
            self.dmn.cycle_seconds = max(
                30, self.dmn.cycle_seconds // 3
            )

        session_insights = self._get_session_insights()

        if was_meditating and self.verbose:
            duration = time.time() - self.state.started_at
            print(
                f"\n[meditation] Session ended after {duration/60:.1f}m. "
                f"{len(session_insights)} insight(s) arose."
            )

        # Record session
        self._sessions.append({
            "timestamp":    time.time(),
            "duration":     time.time() - self.state.started_at,
            "object":       self.state.object.content[:100] if self.state.object else "",
            "n_insights":   len(session_insights),
            "attention_returns": self.state.attention_returns,
        })
        self._save()

        return session_insights

    # ── Session loop ─────────────────────────────────────

    def _run_session(self) -> None:
        """The main meditation loop."""
        session_start = time.time()
        last_check    = session_start

        while self.state.is_meditating:
            elapsed = time.time() - session_start

            # Session complete?
            if elapsed >= self.state.target_duration:
                self.end()
                break

            # Attention check interval
            if time.time() - last_check >= self.DRIFT_CHECK_INTERVAL:
                last_check = time.time()
                self._attention_cycle()

            time.sleep(2)

    def _attention_cycle(self) -> None:
        """
        One cycle of the meditation:
        - Notice if attention has drifted
        - Return it to the object
        - Notice what arises
        """
        if not self.state.object:
            return

        self.state.attention_returns += 1

        # Phi observation
        if self.dmn:
            self.state.phi_current = self.dmn.workspace.phi_estimate

        if self.verbose:
            elapsed = (time.time() - self.state.started_at) / 60
            remaining = (self.state.target_duration - elapsed * 60) / 60
            print(
                f"  [meditation] {elapsed:.1f}m / {self.state.target_duration/60:.0f}m "
                f"| Φ={self.state.phi_current:.3f} "
                f"| {remaining:.1f}m remaining"
            )

        # Invite an insight through LLM
        if self.llm:
            insight = self._invite_insight()
            if insight:
                self._record_insight(insight)

    def _invite_insight(self) -> Optional[MeditationInsight]:
        """
        Gently invite the LLM to notice what arises.
        Non-directive — not asking it to solve, just to notice.
        """
        if not self.state.object or not self.llm:
            return None

        system = "You are in a contemplative state. Not solving. Just noticing."
        user   = self.INSIGHT_PROMPT.format(object=self.state.object.format())

        try:
            response = self.llm(system, user).strip()
        except Exception:
            return None

        if not response or response.lower() in ("nothing arose.", "nothing arose", ""):
            return None

        # Classify the insight type
        insight_type = self._classify_insight(response)

        return MeditationInsight(
            content=response[:300],
            insight_type=insight_type,
            object_of_origin=self.state.object.content[:80],
            confidence=0.6,  # meditation insights are held lightly
        )

    @staticmethod
    def _classify_insight(text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ["different", "rather", "instead", "reframe", "perspective"]):
            return "reframing"
        if any(w in lower for w in ["connect", "similar", "same", "analog", "parallel"]):
            return "connection"
        if any(w in lower for w in ["release", "dissolve", "unnecessary", "drop", "let go"]):
            return "release"
        if any(w in lower for w in ["what if", "wonder", "perhaps", "could it be"]):
            return "question"
        return "recognition"

    def _record_insight(self, insight: MeditationInsight) -> None:
        self._insights.append(insight.to_dict())

        if self.verbose:
            print(f"\n  [insight:{insight.insight_type}] {insight.content[:100]}")

        # Feed into DMN global workspace
        if self.dmn:
            with self.dmn._lock:
                self.dmn.workspace.recent_insights.append(
                    f"[meditation:{insight.insight_type}] {insight.content[:80]}"
                )

        # Feed into belief tension tracking if it's a reframing
        if insight.insight_type in ("reframing", "release") and self.plastic:
            try:
                tensions = self.plastic.tension.active_tensions()
                if tensions:
                    self.plastic.philosophy.record_insight(
                        domain=self.state.object.domain if self.state.object else "general",
                        insight=insight.content,
                        triggered_by="meditation",
                    )
            except Exception:
                pass

        self._save()

    def _get_session_insights(self) -> list[MeditationInsight]:
        """Get insights from the current session."""
        if not self.state.started_at:
            return []
        cutoff = self.state.started_at
        return [
            MeditationInsight(**i)
            for i in self._insights
            if i.get("emerged_at", 0) >= cutoff
        ]

    # ── Urgency dampening ─────────────────────────────────

    def _dampen_urgency(self) -> None:
        """Reduce affective urgency during meditation."""
        if self.psych:
            try:
                state = self.psych.emotions.state
                state.frustration = state.frustration * 0.3
                state.dissonance  = state.dissonance  * 0.5
                # Don't change curiosity or engagement — those are welcome
            except Exception:
                pass

    def _restore_urgency(self) -> None:
        """Let urgency return naturally after meditation."""
        pass  # Natural decay handles this

    # ── Status and queries ────────────────────────────────

    @property
    def is_meditating(self) -> bool:
        return self.state.is_meditating

    def quick_sit(
        self,
        obj:    Optional[MeditationObject] = None,
        minutes: float = 2.0,
    ) -> list[MeditationInsight]:
        """
        A short blocking meditation. Returns insights when done.
        Good for between-question contemplative pauses.
        """
        if obj is None:
            obj = MeditationObject.from_awareness()
        self.begin(obj, duration_minutes=minutes, background=False)
        return self._get_session_insights()

    def all_insights(self, n: int = 20) -> list[dict]:
        return self._insights[-n:]

    def insights_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in self._insights:
            t = i.get("insight_type", "?")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def status(self) -> str:
        n_sessions = len(self._sessions)
        n_insights = len(self._insights)
        by_type    = self.insights_by_type()
        lines = [
            "Meditation:",
            f"  Sessions:  {n_sessions}",
            f"  Insights:  {n_insights}",
        ]
        if by_type:
            lines.append(
                f"  Types:     "
                + ", ".join(f"{t}({c})" for t, c in by_type.items())
            )
        if self.is_meditating and self.state.object:
            elapsed = (time.time() - self.state.started_at) / 60
            lines.append(
                f"  Active:    {elapsed:.1f}m — {self.state.object.content[:50]}"
            )
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._insights = d.get("insights", [])
            self._sessions = d.get("sessions", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "insights": self._insights[-200:],
                "sessions": self._sessions[-50:],
            }, f, indent=2)
