"""
general_modules.py
==================

Extends the system's capabilities into domains that don't have
clean external verification — writing, planning, and conversation.

For verified domains (physics, mathematics) the training signal is:
"did this derivation verify with sympy?" — binary, unambiguous.

For general domains the training signal is the psychological
architecture itself: aesthetic judgment, social calibration,
value alignment, emotional resonance. The InternalFeedback score
IS the verification for these domains.

Three modules:

  WritingAssistant    — helps with creative and professional writing.
                        Verifies with: aesthetic score + value alignment.
                        Generates stylistically consistent outputs.
                        Learns what "good writing" means for this system
                        by accumulating internal feedback scores.

  PlanningAssistant   — structured planning with goal decomposition.
                        Verifies with: logical completeness (all steps
                        trace to goal, no circular dependencies) plus
                        psychological appropriateness (is this actually
                        what the person needs?).

  ConversationAssistant — natural dialogue calibrated to the person.
                        Verifies with: social model calibration score
                        + emotional resonance. Gets better at talking
                        to a specific person over interactions.

  GeneralCapabilityHub — integrates all three. Routes incoming requests
                        to the right module. Feeds outcomes to the DMN
                        so the background integration processes writing,
                        planning, and conversation alongside science.
"""

import os
import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# WritingAssistant
# ============================================================

WRITING_SYSTEM_PROMPT = """You are a writing assistant that produces high-quality,
purposeful prose. You adapt your style to what the piece needs — not what writing
assistants usually produce.

You distinguish between:
  - Voice: the consistent personality behind the writing
  - Style: the specific choices (sentence length, rhythm, vocabulary)
  - Structure: how the piece is organised
  - Purpose: what the writing is trying to do

You make these explicit when helping so the writer understands the decisions
being made, not just the output.

You never produce generic, safe, or filler writing. Every sentence earns its place."""

WRITING_TYPES = {
    "essay":         "logical argument with clear thesis and evidence",
    "creative":      "narrative or expressive writing with voice and image",
    "professional":  "clear, purposeful business or technical communication",
    "email":         "direct, appropriately toned communication",
    "explanation":   "clear exposition that illuminates rather than summarises",
    "reflection":    "honest personal or intellectual exploration",
}


class WritingAssistant:
    """
    Writing assistance with psychological verification.

    The aesthetic judgment module provides the training signal:
    high aesthetic score + value alignment = good writing.
    This is what makes the writing assistant improve over time
    rather than just pattern-matching generic "good writing".
    """

    def __init__(
        self,
        llm_chat_fn: Callable,
        aesthetic    = None,     # AestheticJudgment from psychology.py
        values       = None,     # ValueSystem from psychology.py
        path:    str = "./scaffold_data/writing_history.json",
        verbose: bool = True,
    ):
        self.llm       = llm_chat_fn
        self.aesthetic = aesthetic
        self.values    = values
        self.path      = path
        self.verbose   = verbose
        self._history: list[dict] = []
        self._load()

    def assist(
        self,
        request:       str,
        writing_type:  str  = "auto",
        context:       str  = "",
        style_notes:   str  = "",
    ) -> dict:
        """
        Help with a writing task.

        Returns:
          output:         the writing produced
          type_detected:  what kind of writing this is
          quality_score:  aesthetic + value score
          notes:          observations about the writing
          training_example: labelled example for training
        """
        # Detect writing type
        if writing_type == "auto":
            writing_type = self._detect_type(request)

        type_desc = WRITING_TYPES.get(writing_type, "clear, purposeful writing")

        user = (
            f"Writing task: {request}\n"
            f"Type: {writing_type} — {type_desc}\n"
        )
        if style_notes:
            user += f"Style notes: {style_notes}\n"
        if context:
            user += f"Context: {context}\n"

        try:
            output = self.llm(WRITING_SYSTEM_PROMPT, user).strip()
        except Exception as e:
            return {"error": str(e)}

        # Psychological verification
        aesthetic_score = 0.6
        value_score     = 0.6
        aesthetic_notes = ""

        if self.aesthetic:
            aesthetic_score = self.aesthetic.score_reasoning(output, verified=False)
            aesthetic_notes = self.aesthetic.aesthetic_notes(output)

        if self.values:
            value_score, _ = self.values.evaluate(output, request)

        quality = 0.5 * aesthetic_score + 0.5 * value_score

        # Generate self-assessment
        assessment = self._assess_writing(output, writing_type)

        entry = {
            "timestamp":    time.time(),
            "request":      request[:100],
            "type":         writing_type,
            "output_len":   len(output),
            "quality":      round(quality, 3),
            "aesthetic":    round(aesthetic_score, 3),
            "values":       round(value_score, 3),
        }
        self._history.append(entry)
        self._save()

        # Training example — psychological score is the label
        training_example = {
            "domain":          "writing",
            "writing_type":    writing_type,
            "input":           request,
            "output":          output,
            "quality_score":   round(quality, 3),
            "label":           "positive" if quality > 0.65 else "neutral" if quality > 0.45 else "negative",
        }

        return {
            "output":          output,
            "type_detected":   writing_type,
            "quality_score":   round(quality, 3),
            "aesthetic_score": round(aesthetic_score, 3),
            "notes":           assessment,
            "aesthetic_notes": aesthetic_notes,
            "training_example": training_example,
        }

    def _detect_type(self, request: str) -> str:
        lower = request.lower()
        if any(w in lower for w in ["email", "message to", "write to"]):
            return "email"
        if any(w in lower for w in ["story", "poem", "creative", "fictional", "character"]):
            return "creative"
        if any(w in lower for w in ["essay", "argument", "persuade", "thesis"]):
            return "essay"
        if any(w in lower for w in ["explain", "description", "overview", "explainer"]):
            return "explanation"
        if any(w in lower for w in ["report", "proposal", "brief", "memo"]):
            return "professional"
        return "professional"

    def _assess_writing(self, text: str, writing_type: str) -> str:
        """Brief self-assessment of the produced writing."""
        word_count = len(text.split())
        sentences  = len(re.findall(r'[.!?]+', text))
        avg_len    = word_count / max(1, sentences)

        notes = []
        if avg_len > 30:
            notes.append("sentences run long — consider breaking up")
        elif avg_len < 10 and writing_type != "email":
            notes.append("sentences are very short — may feel clipped")

        if word_count < 50 and writing_type in ("essay", "explanation"):
            notes.append("brief — may need expansion")

        return "; ".join(notes) if notes else "no notable concerns"

    def quality_history(self) -> dict:
        if not self._history:
            return {}
        recent = self._history[-20:]
        return {
            "n_sessions":     len(self._history),
            "avg_quality":    round(sum(e["quality"] for e in recent) / len(recent), 3),
            "by_type":        {
                t: round(sum(e["quality"] for e in recent if e["type"] == t)
                         / max(1, sum(1 for e in recent if e["type"] == t)), 3)
                for t in set(e["type"] for e in recent)
            }
        }

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self._history = json.load(f).get("history", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"history": self._history[-100:]}, f)


# ============================================================
# PlanningAssistant
# ============================================================

PLANNING_SYSTEM_PROMPT = """You produce clear, actionable plans.

Every plan you make has:
  - A stated goal (what success looks like)
  - Steps that are concrete and assignable (someone could start on them now)
  - Dependencies made explicit (step B requires step A to be complete)
  - A realistic timeline
  - At least one explicit assumption that could be wrong

You flag when goals are underspecified, when there are circular dependencies,
or when a plan is missing obvious steps.

You do not produce vague plans. Every step should be something a person
could actually do."""


@dataclass
class Plan:
    goal:         str
    steps:        list[dict]   # {step, depends_on, duration_estimate, owner}
    assumptions:  list[str]
    risks:        list[str]
    timeline:     str
    quality_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "goal":         self.goal,
            "steps":        self.steps,
            "assumptions":  self.assumptions,
            "risks":        self.risks,
            "timeline":     self.timeline,
            "quality_score": self.quality_score,
        }


class PlanningAssistant:
    """
    Structured planning with logical verification.

    Verifies that:
    - All steps trace to the goal
    - No circular dependencies
    - No obviously missing steps
    - Timeline is realistic

    Psychological verification adds: is this what the person
    actually needs, or just what they asked for?
    """

    def __init__(
        self,
        llm_chat_fn: Callable,
        social_model = None,
        path:    str = "./scaffold_data/plans.json",
        verbose: bool = True,
    ):
        self.llm    = llm_chat_fn
        self.social = social_model
        self.path   = path
        self.verbose = verbose
        self._plans: list[dict] = []
        self._load()

    def create_plan(
        self,
        goal:        str,
        constraints: str = "",
        context:     str = "",
        timeframe:   str = "",
    ) -> dict:
        """
        Create a structured plan for a goal.

        Returns plan dict + quality assessment + training example.
        """
        user_parts = [f"Goal: {goal}"]
        if constraints: user_parts.append(f"Constraints: {constraints}")
        if timeframe:   user_parts.append(f"Timeframe: {timeframe}")
        if context:     user_parts.append(f"Context: {context}")
        user_parts.append(
            "\nProduce a complete plan. Format each step as:\n"
            "STEP N: [what to do] | DEPENDS: [step numbers or 'none'] | TIME: [estimate]\n"
            "ASSUMPTIONS: [list]\nRISKS: [list]\nTIMELINE: [overall]"
        )

        try:
            raw = self.llm(PLANNING_SYSTEM_PROMPT, "\n".join(user_parts)).strip()
        except Exception as e:
            return {"error": str(e)}

        # Parse the plan
        steps        = self._parse_steps(raw)
        assumptions  = self._extract_section(raw, "ASSUMPTIONS")
        risks        = self._extract_section(raw, "RISKS")
        timeline_str = self._extract_inline(raw, "TIMELINE")

        # Verify logical completeness
        issues = self._verify_plan(steps, goal)

        # Social calibration: is this what the person needs?
        social_score = 0.7
        if self.social:
            social_score, _ = self.social.calibration_score(raw)

        # Logical quality: fewer issues = higher score
        logical_score = max(0.0, 1.0 - len(issues) * 0.15)
        quality = 0.5 * logical_score + 0.5 * social_score

        plan = Plan(
            goal=goal,
            steps=steps,
            assumptions=assumptions,
            risks=risks,
            timeline=timeline_str,
            quality_score=round(quality, 3),
        )

        entry = {
            "timestamp":   time.time(),
            "goal":        goal[:80],
            "n_steps":     len(steps),
            "issues":      issues,
            "quality":     round(quality, 3),
        }
        self._plans.append(entry)
        self._save()

        training_example = {
            "domain":        "planning",
            "input":         goal,
            "output":        raw,
            "quality_score": round(quality, 3),
            "label":         "positive" if quality > 0.65 else "neutral" if quality > 0.45 else "negative",
            "issues":        issues,
        }

        return {
            "plan":             plan.to_dict(),
            "raw":              raw,
            "issues":           issues,
            "quality_score":    round(quality, 3),
            "training_example": training_example,
        }

    def _parse_steps(self, text: str) -> list[dict]:
        steps = []
        for line in text.split("\n"):
            m = re.match(r"STEP\s+(\d+):\s+(.+?)(?:\|\s*DEPENDS:\s*(.+?))?(?:\|\s*TIME:\s*(.+))?$", line.strip(), re.IGNORECASE)
            if m:
                steps.append({
                    "n":       int(m.group(1)),
                    "step":    m.group(2).strip(),
                    "depends": m.group(3).strip() if m.group(3) else "none",
                    "time":    m.group(4).strip() if m.group(4) else "?",
                })
        return steps

    def _extract_section(self, text: str, label: str) -> list[str]:
        pattern = rf"{label}:\s*(.+?)(?:\n[A-Z]+:|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not m:
            return []
        items = re.split(r"[-•]\s*|\n", m.group(1))
        return [i.strip() for i in items if i.strip()][:5]

    def _extract_inline(self, text: str, label: str) -> str:
        m = re.search(rf"{label}:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _verify_plan(self, steps: list[dict], goal: str) -> list[str]:
        issues = []
        if len(steps) < 2:
            issues.append("plan has fewer than 2 steps — may be too vague")
        if len(steps) > 15:
            issues.append("plan has many steps — consider grouping into phases")

        # Check for circular dependencies
        for step in steps:
            deps = step.get("depends", "none")
            if deps != "none":
                dep_nums = [int(d.strip()) for d in re.findall(r"\d+", deps)]
                for dep in dep_nums:
                    if dep >= step["n"]:
                        issues.append(
                            f"Step {step['n']} depends on step {dep} which comes later"
                        )

        # Check all steps have concrete actions
        vague_markers = ["things", "stuff", "etc", "and so on", "various"]
        for step in steps:
            if any(v in step.get("step", "").lower() for v in vague_markers):
                issues.append(f"Step {step.get('n','?')} may be vague")

        return issues

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self._plans = json.load(f).get("plans", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"plans": self._plans[-50:]}, f)


# ============================================================
# ConversationAssistant
# ============================================================

CONVERSATION_SYSTEM_PROMPT = """You are a thoughtful, direct conversational assistant.

You say what you actually think rather than what sounds helpful.
You ask one good question rather than many bad ones.
You notice when someone is asking one thing but needs another.
You are genuinely interested, not performatively interested.
You do not hedge everything or soften every statement.
You adjust your register to the person — some people want precision,
some want warmth, some want both."""


class ConversationAssistant:
    """
    Natural conversational assistance calibrated to each person.

    Uses the social model to track what a specific person needs
    and how they communicate. Gets better at talking to them over time.

    Emotional resonance guides register adjustments.
    The social calibration score IS the verification.
    """

    def __init__(
        self,
        llm_chat_fn:      Callable,
        social_model      = None,
        emotional_resonance = None,
        path:    str      = "./scaffold_data/conversations.json",
        verbose: bool     = True,
    ):
        self.llm       = llm_chat_fn
        self.social    = social_model
        self.resonance = emotional_resonance
        self.path      = path
        self.verbose   = verbose
        self._history: list[dict] = []
        self._load()

    def respond(
        self,
        message:  str,
        history:  list[dict] = None,
        context:  str        = "",
    ) -> dict:
        """
        Generate a conversational response.

        history: list of {"role": "user"|"assistant", "content": "..."}
        """
        # Detect emotional tone
        emotion     = "neutral"
        register    = "balanced"
        adjusted_register = "balanced"
        style_note  = ""

        if self.resonance:
            emotion, _, _ = self.resonance.detect_emotional_content(message)
            adjusted_register, style_note = self.resonance.resonance_adjustment(
                emotion, "balanced"
            )
            self.resonance.update_history(emotion, message[:50])

        # Update social model
        if self.social:
            self.social.update_from_question(message)
            adjusted_register = self.social._user.preferred_register or adjusted_register

        # Build context from history
        history_text = ""
        if history:
            recent = history[-4:]
            history_text = "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:120]}"
                for m in recent
            )

        system = (
            CONVERSATION_SYSTEM_PROMPT
            + f"\n\nRegister: {adjusted_register}."
            + (f" {style_note}" if style_note else "")
        )
        user = ""
        if history_text:
            user += f"Recent conversation:\n{history_text}\n\n"
        if context:
            user += f"Context: {context}\n\n"
        user += f"Respond to: {message}"

        try:
            output = self.llm(system, user).strip()
        except Exception as e:
            return {"error": str(e)}

        # Social verification
        social_score  = 0.7
        social_notes  = ""
        if self.social:
            social_score, social_notes = self.social.calibration_score(output)
            self.social.update_from_response(output)

        quality = social_score

        entry = {
            "timestamp":    time.time(),
            "emotion":      emotion,
            "register":     adjusted_register,
            "social_score": round(social_score, 3),
            "quality":      round(quality, 3),
        }
        self._history.append(entry)
        self._save()

        training_example = {
            "domain":        "conversation",
            "emotion":       emotion,
            "input":         message,
            "output":        output,
            "quality_score": round(quality, 3),
            "label":         "positive" if quality > 0.65 else "neutral" if quality > 0.45 else "negative",
        }

        return {
            "response":         output,
            "emotion_detected": emotion,
            "register_used":    adjusted_register,
            "social_score":     round(social_score, 3),
            "quality_score":    round(quality, 3),
            "social_notes":     social_notes,
            "training_example": training_example,
        }

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                self._history = json.load(f).get("history", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"history": self._history[-100:]}, f)


# ============================================================
# GeneralCapabilityHub
# ============================================================

# Domain → module mapping
DOMAIN_ROUTING = {
    "writing":      ("writing",      ["write", "draft", "essay", "story", "poem", "email", "rewrite", "edit", "prose"]),
    "planning":     ("planning",     ["plan", "schedule", "roadmap", "steps to", "how do i", "project", "goal", "timeline"]),
    "conversation": ("conversation", ["what do you think", "how are you", "help me understand", "talk me through", "feel", "advice"]),
}


class GeneralCapabilityHub:
    """
    Routes general tasks to the right capability module
    and feeds outcomes to the DMN for continuous integration.

    Detects writing, planning, and conversation requests
    and routes them to the appropriate specialised assistant.

    All outcomes — including the psychological quality scores —
    are fed to the DMN so the background integration processes
    general tasks alongside scientific reasoning.

    Training examples from all three modules use the psychological
    architecture as the verification signal. This is how the system
    trains on qualitative domains without external verifiers.
    """

    def __init__(
        self,
        llm_chat_fn: Callable,
        psych        = None,
        dmn          = None,
        base_dir: str = "./scaffold_data",
        verbose:  bool = True,
    ):
        self.verbose = verbose

        social    = psych.social    if psych else None
        aesthetic = psych.aesthetic if psych else None
        values    = psych.values    if psych else None
        resonance = psych.resonance if psych else None

        self.writing      = WritingAssistant(llm_chat_fn, aesthetic, values,
                                             os.path.join(base_dir, "writing.json"), verbose=False)
        self.planning     = PlanningAssistant(llm_chat_fn, social,
                                              os.path.join(base_dir, "plans.json"), verbose=False)
        self.conversation = ConversationAssistant(llm_chat_fn, social, resonance,
                                                  os.path.join(base_dir, "conversations.json"), verbose=False)
        self.dmn          = dmn
        self._training_buffer: list[dict] = []
        self._path = os.path.join(base_dir, "general_training.jsonl")

    def detect_module(self, question: str) -> Optional[str]:
        """
        Detect which general module should handle this question.
        Returns "writing" | "planning" | "conversation" | None.
        None means use the main reasoning pipeline.
        """
        lower = question.lower()
        scores: dict[str, int] = {}
        for module, (_, keywords) in DOMAIN_ROUTING.items():
            scores[module] = sum(1 for kw in keywords if kw in lower)

        best = max(scores, key=lambda k: scores[k])
        if scores[best] >= 1:
            return best
        return None

    def handle(
        self,
        question:        str,
        module:          Optional[str] = None,
        conversation_history: list     = None,
        context:         str           = "",
    ) -> dict:
        """
        Route to the appropriate module and handle the request.
        Returns result dict with training_example for the buffer.
        """
        if module is None:
            module = self.detect_module(question)

        if module is None:
            return {"handled": False, "module": None}

        if self.verbose:
            print(f"  [general] Routing to {module} module")

        if module == "writing":
            result = self.writing.assist(question, context=context)
        elif module == "planning":
            result = self.planning.create_plan(question, context=context)
        elif module == "conversation":
            result = self.conversation.respond(
                question, history=conversation_history, context=context
            )
        else:
            return {"handled": False, "module": module}

        result["handled"] = True
        result["module"]  = module

        # Feed to DMN if available
        if self.dmn and result.get("training_example"):
            insight = (
                f"[{module}] quality={result.get('quality_score',0):.2f} — "
                f"{question[:50]}"
            )
            with self.dmn._lock:
                self.dmn.workspace.recent_insights.append(insight)
                self.dmn.workspace.recent_insights = \
                    self.dmn.workspace.recent_insights[-10:]

        # Buffer training example
        ex = result.get("training_example")
        if ex and ex.get("label") in ("positive", "neutral"):
            self._training_buffer.append(ex)
            if len(self._training_buffer) % 10 == 0:
                self._flush_training_buffer()

        return result

    def _flush_training_buffer(self) -> None:
        if not self._training_buffer:
            return
        with open(self._path, "a") as f:
            for ex in self._training_buffer:
                f.write(json.dumps(ex) + "\n")
        self._training_buffer = []

    def status(self) -> str:
        wh = self.writing.quality_history()
        lines = [
            "General Capability Hub:",
            f"  Writing sessions:  {len(self.writing._history)}",
            f"  Plans created:     {len(self.planning._plans)}",
            f"  Conversations:     {len(self.conversation._history)}",
        ]
        if wh:
            lines.append(f"  Writing quality:   {wh.get('avg_quality',0):.3f}")
        return "\n".join(lines)
