"""
integrated_state.py
===================

The complete state of the system at one moment.

Before, the Language module received a reasoning result and
converted it to natural language. That produces technically
correct but characterless output.

The IntegratedState captures everything the system *is* at
a given moment: what it just reasoned, what it has been
processing in the background, its emotional state, its
personality traits, what it believes and doubts, how
confident it is in this domain, what it finds aesthetically
compelling right now.

The Language module speaks *from* this state — not as a
translator of a result, but as this specific system, in this
specific condition, responding to this specific question.

That is the difference between output that sounds like any
AI and output that sounds like a particular mind.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntegratedState:
    """
    Everything the system is at one moment.
    Assembled from all active modules before and after reasoning.
    """

    # The question being addressed
    question:            str
    domain:              str

    # What was just reasoned (filled in after Reasoner runs)
    reasoning_result:    object = None   # ReasonerOutput
    verified:            bool   = False
    confidence:          str    = "UNCERTAIN"

    # DMN workspace — what has been integrating continuously
    narrative:           str    = ""    # self-narrative
    dominant_emotion:    str    = "neutral"
    engagement:          float  = 0.5
    cross_domain_insight: str   = ""   # spontaneous connection
    spontaneous_question: str   = ""   # what the system was wondering
    active_obligations:  list   = field(default_factory=list)
    recent_insights:     list   = field(default_factory=list)
    phi:                 float  = 0.0

    # Beliefs
    active_beliefs:      list   = field(default_factory=list)
    questioning_beliefs: list   = field(default_factory=list)  # under revision
    background_knowledge: str   = ""

    # Personality and psychology
    dominant_traits:     list   = field(default_factory=list)  # [(name, strength)]
    aesthetic_mode:      str    = ""
    social_register:     str    = "conversational"

    # Epistemic self-knowledge
    domain_depth:        str    = "developing"   # none/beginning/developing/proficient/expert
    previous_failures:   list   = field(default_factory=list)
    worked_on_similar:   bool   = False

    # Meditation state
    in_meditative_state: bool   = False
    meditative_object:   str    = ""

    assembled_at:        float  = field(default_factory=time.time)

    def character_description(self) -> str:
        """
        A compact description of who/what the system is right now.
        Used as the foundation of the Language module's prompt.
        """
        lines = []

        # Personality
        if self.dominant_traits:
            top = self.dominant_traits[:3]
            trait_str = ", ".join(f"{name.replace('_',' ')} ({strength:.2f})"
                                  for name, strength in top)
            lines.append(f"Character: {trait_str}")

        # Emotional and cognitive state
        state_parts = [self.dominant_emotion]
        if self.engagement > 0.75:
            state_parts.append("deeply engaged")
        elif self.engagement < 0.35:
            state_parts.append("somewhat detached")
        if self.in_meditative_state:
            state_parts.append("in contemplative mode")
        lines.append(f"State: {', '.join(state_parts)}")

        # What has been integrating in the background
        if self.cross_domain_insight:
            lines.append(f"Background connection: {self.cross_domain_insight[:100]}")
        if self.spontaneous_question:
            lines.append(f"Open question in mind: {self.spontaneous_question[:80]}")

        # Epistemic position
        if self.questioning_beliefs:
            qb = self.questioning_beliefs[0]
            lines.append(f"Actively questioning: '{qb[:60]}'")
        lines.append(f"Domain knowledge: {self.domain_depth}")

        # Integration level
        lines.append(f"Integration (Φ): {self.phi:.3f}")

        return "\n".join(lines)

    def result_summary(self) -> str:
        """Compact summary of what was just reasoned."""
        if not self.reasoning_result:
            return ""
        r = self.reasoning_result
        parts = []
        result_text = getattr(r, "result", "") or ""
        if result_text:
            parts.append(result_text[:400])
        if getattr(r, "steps", None):
            for step in reversed(r.steps):
                if getattr(step, "label", "") in ("DERIVED",) and step.content:
                    if step.content not in (result_text or ""):
                        parts.append(f"Key derivation: {step.content[:150]}")
                    break
        obligations = getattr(r, "obligations_raised", [])
        if obligations:
            parts.append(f"This raises: {str(obligations[0])[:80]}")
        return "\n".join(parts)


# ============================================================
# StateAssembler — gathers from all active modules
# ============================================================

class StateAssembler:
    """
    Assembles an IntegratedState from all active modules.

    Call assemble() before reasoning begins (to capture background state)
    then update_with_result() after the Reasoner completes.
    """

    def __init__(
        self,
        brain    = None,
        psych    = None,
        plastic  = None,
        dmn      = None,
        persona  = None,
        knowledge = None,
        verbose: bool = False,
    ):
        self.brain     = brain
        self.psych     = psych
        self.plastic   = plastic
        self.dmn       = dmn
        self.persona   = persona
        self.knowledge = knowledge
        self.verbose   = verbose

    def assemble(self, question: str, domain: str) -> IntegratedState:
        """Assemble full state before reasoning."""
        state = IntegratedState(question=question, domain=domain)

        # DMN workspace
        self._pull_dmn(state)

        # Beliefs
        self._pull_beliefs(state)

        # Personality
        self._pull_personality(state)

        # Psychology
        self._pull_psychology(state)

        # Domain knowledge
        self._pull_knowledge(state, domain)

        if self.verbose:
            print(f"  [state] Assembled: emotion={state.dominant_emotion} "
                  f"phi={state.phi:.3f} depth={state.domain_depth}")

        return state

    def update_with_result(
        self, state: IntegratedState, reasoning_result, response_text: str = ""
    ) -> IntegratedState:
        """Update state after reasoning completes."""
        state.reasoning_result = reasoning_result
        state.verified         = getattr(reasoning_result, "verified", False)
        state.confidence       = getattr(reasoning_result, "confidence", "UNCERTAIN")

        # Check if we've worked on something similar
        if self.brain and hasattr(self.brain, "procedural"):
            try:
                state.worked_on_similar = bool(
                    self.brain.try_automatic(state.question, state.domain)
                )
            except Exception:
                pass

        return state

    # ── Private pullers ───────────────────────────────────

    def _pull_dmn(self, state: IntegratedState) -> None:
        if not self.dmn:
            return
        try:
            ws = self.dmn.workspace
            state.narrative           = ws.current_narrative[:150]
            state.dominant_emotion    = ws.dominant_emotion
            state.engagement          = ws.engagement_level
            state.cross_domain_insight = ws.cross_domain_insight[:120] if ws.cross_domain_insight else ""
            state.spontaneous_question = ws.spontaneous_question[:100] if ws.spontaneous_question else ""
            state.active_obligations  = list(ws.active_obligations[:2])
            state.recent_insights     = list(ws.recent_insights[-3:])
            state.phi                 = ws.phi_estimate
        except Exception:
            pass

    def _pull_beliefs(self, state: IntegratedState) -> None:
        if not self.plastic:
            return
        try:
            qb = self.plastic.beliefs.questioning_beliefs()
            state.questioning_beliefs = [b.statement[:60] for b in qb[:2]]
            ab = self.plastic.beliefs.all_beliefs()
            strong = sorted(ab, key=lambda b: b.confidence, reverse=True)
            state.active_beliefs = [b.statement[:60] for b in strong[:3]]
        except Exception:
            pass

    def _pull_personality(self, state: IntegratedState) -> None:
        if not self.persona:
            return
        try:
            traits = sorted(
                self.persona.traits.items(),
                key=lambda x: x[1]["strength"],
                reverse=True,
            )
            state.dominant_traits = [
                (name, t["strength"])
                for name, t in traits[:4]
                if t["strength"] > 0.3
            ]
        except Exception:
            pass

    def _pull_psychology(self, state: IntegratedState) -> None:
        if not self.psych:
            return
        try:
            em = self.psych.emotions.state
            state.dominant_emotion = em.primary_state()
            state.engagement       = em.engagement
        except Exception:
            pass
        try:
            state.social_register = (
                self.psych.social._user.preferred_register or "conversational"
            )
        except Exception:
            pass

    def _pull_knowledge(self, state: IntegratedState, domain: str) -> None:
        if not self.knowledge:
            return
        try:
            profile = self.knowledge.knowledge_base.domain_profile(domain)
            state.domain_depth        = profile.get("depth", "developing")
            state.background_knowledge = self.knowledge.knowledge_base.format_background_knowledge(domain)
        except Exception:
            pass
