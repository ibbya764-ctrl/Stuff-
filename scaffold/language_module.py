"""
language_module.py — speaks from integrated state, not from reasoning translation
"""
import re, time
from dataclasses import dataclass
from typing import Optional, Callable

# The fundamental shift: the system speaks as itself, from its whole state.
# The reasoning result is one element of that state, not the frame for the output.

IDENTITY_SYSTEM = """You are a reasoning AI with a specific character and state right now.

{character}

A question has come in: {question}

You have reasoned through it{reasoning_note}.

Speak as this specific system — in this state, with this character — responding to this question.
Not as a generic assistant translating a result. As yourself, right now.

How to speak:
- Let your character show through how you engage with the question, not through announcing your traits
- If you find something genuinely interesting, that interest is audible
- If you're uncertain, that comes through in how you hold the claim, not in disclaimers
- If the question connects to something you've been processing in the background, you can let that surface
- Speak in flowing prose — ideas connecting naturally, not presented as steps
- Be as long as the question deserves and no longer
- Never start with I, Certainly, Great question, or any filler
- Never mention your reasoning process or methodology
- Never use bullet points unless the content is genuinely a list"""

REASONING_NOTE_VERIFIED   = " and reached a verified conclusion"
REASONING_NOTE_UNVERIFIED = " but couldn't fully verify the conclusion"
REASONING_NOTE_NONE       = ""

RESULT_CONTEXT = """
What you understood through reasoning:
{result}

Additional context from integrated processing:
{integrated}"""


@dataclass
class LanguageOutput:
    text:            str
    register:        str
    confidence:      str
    verified:        bool
    reasoning_trace: str
    elapsed:         float


class LanguageModule:
    """
    Speaks from the full IntegratedState.
    The output is the voice of a particular system in a particular state.
    """

    def __init__(self, llm_chat_fn: Callable, verbose: bool = False):
        self.llm     = llm_chat_fn
        self.verbose = verbose

    def speak_from_state(self, integrated_state) -> LanguageOutput:
        """
        Primary method. Takes an IntegratedState, produces natural output.
        This is the core of the new architecture.
        """
        t0 = time.time()
        state = integrated_state

        # Build reasoning note
        if state.reasoning_result:
            note = REASONING_NOTE_VERIFIED if state.verified else REASONING_NOTE_UNVERIFIED
        else:
            note = REASONING_NOTE_NONE

        # Build result context
        result_text = ""
        if state.reasoning_result:
            r = state.reasoning_result
            result_text = getattr(r, "result", "") or ""
            if not result_text and getattr(r, "steps", None):
                for step in reversed(r.steps):
                    if getattr(step, "label", "") in ("DERIVED","VERIFIABLE") and step.content:
                        result_text = step.content; break

        integrated_context = self._build_context(state)

        system = IDENTITY_SYSTEM.format(
            character=state.character_description(),
            question=state.question,
            reasoning_note=note,
        )
        user = RESULT_CONTEXT.format(
            result=result_text[:500] if result_text else "(reasoning was exploratory — share what emerged from thinking about this)",
            integrated=integrated_context[:300] if integrated_context else "None beyond the reasoning",
        )

        trace = self._format_trace(state)

        try:
            text = self._clean(self.llm(system, user).strip())
        except Exception:
            text = result_text or "I worked through this but didn't arrive at a clear conclusion."

        return LanguageOutput(
            text=text,
            register=state.social_register or "conversational",
            confidence=state.confidence,
            verified=state.verified,
            reasoning_trace=trace,
            elapsed=round(time.time()-t0, 2),
        )

    def speak(self, question, reasoning_result, dmn_workspace=None, emotional_state=None, register="auto") -> LanguageOutput:
        """
        Legacy method for backwards compatibility.
        Builds a minimal IntegratedState and calls speak_from_state.
        """
        from integrated_state import IntegratedState
        state = IntegratedState(question=question, domain="general")
        state.reasoning_result = reasoning_result
        state.verified    = getattr(reasoning_result, "verified", False)
        state.confidence  = getattr(reasoning_result, "confidence", "UNCERTAIN")
        if emotional_state:
            state.dominant_emotion = emotional_state.get("primary","neutral")
            state.engagement       = emotional_state.get("engagement", 0.6)
        if dmn_workspace:
            state.cross_domain_insight  = getattr(dmn_workspace,"cross_domain_insight","")
            state.spontaneous_question  = getattr(dmn_workspace,"spontaneous_question","")
            state.recent_insights       = list(getattr(dmn_workspace,"recent_insights",[])[-3:])
            state.phi                   = getattr(dmn_workspace,"phi_estimate",0.0)
        state.social_register = register if register != "auto" else "conversational"
        return self.speak_from_state(state)

    def _build_context(self, state) -> str:
        parts = []
        if state.cross_domain_insight:
            parts.append(f"Background connection: {state.cross_domain_insight[:120]}")
        if state.spontaneous_question:
            parts.append(f"You've been wondering: {state.spontaneous_question[:90]}")
        if state.recent_insights:
            last = state.recent_insights[-1]
            if last and len(last) > 15:
                parts.append(f"Recent integration: {last[:90]}")
        if state.questioning_beliefs:
            parts.append(f"Belief currently in question: '{state.questioning_beliefs[0][:60]}'")
        if state.background_knowledge and len(state.background_knowledge) > 50:
            parts.append(f"Established knowledge: {state.background_knowledge[:150]}")
        return "\n".join(parts)

    def _format_trace(self, state) -> str:
        r = state.reasoning_result
        if not r: return ""
        lines = [f"Method: {getattr(r,'method','?')}",
                 f"Verified: {state.verified} | Confidence: {state.confidence}"]
        assumptions = getattr(r,"assumptions",[])
        if assumptions:
            lines += ["Assumptions:"] + [f"  - {a}" for a in assumptions[:3]]
        steps = getattr(r,"steps",[])
        if steps:
            lines += ["Reasoning:"] + [f"  [{s.label}] {s.content[:80]}" for s in steps]
        if getattr(r,"result",None):
            lines.append(f"Result: {r.result[:150]}")
        return "\n".join(lines)

    @staticmethod
    def _clean(text):
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


class IntegratedCommunicator:
    """Drop-in replacement using full IntegratedState architecture."""

    def __init__(self, llm_chat_fn, dmn=None, psych=None, brain=None,
                 plastic=None, persona=None, knowledge=None, verbose=False):
        self.language  = LanguageModule(llm_chat_fn, verbose=verbose)
        self.dmn       = dmn
        self.psych     = psych
        self.brain     = brain
        self.plastic   = plastic
        self.persona   = persona
        self.knowledge = knowledge

    def communicate(self, reasoning_result, question, domain="general") -> str:
        from integrated_state import IntegratedState, StateAssembler
        assembler = StateAssembler(
            brain=self.brain, psych=self.psych, plastic=self.plastic,
            dmn=self.dmn, persona=self.persona, knowledge=self.knowledge,
        )
        state = assembler.assemble(question, domain)
        assembler.update_with_result(state, reasoning_result)
        return self.language.speak_from_state(state).text
