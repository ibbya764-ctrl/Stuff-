"""
brain_pipeline.py
=================

Formal separation of reasoning and communication — two specialised
systems coordinated through a clean interface, analogous to the
separation between the prefrontal cortex (reasoning) and Broca's
area (language production) in the brain.

Three components:

  Reasoner         — produces structured epistemic output.
                     Optimised for correctness, not readability.
                     Every step labelled [STRUCTURAL] / [DERIVED] /
                     [VERIFIABLE]. Outputs a ReasonerOutput dataclass,
                     not natural language. Can run a different model
                     than the Communicator.

  Communicator     — translates ReasonerOutput into natural language.
                     Never reasons. Never modifies epistemic content.
                     Adapts register (formal / pedagogical / intuitive)
                     to the question. Can run a different model than
                     the Reasoner.

  BrainPipeline    — coordinates both. Integrates memory, introspection,
                     novelty detection, training. Drop-in replacement
                     for EnhancedPipeline with the same ask() interface.

The interface between Reasoner and Communicator is a typed dataclass
(ReasonerOutput) — not a string, not a dict. This means:

  - Each side can be swapped independently
  - Failures in one don't corrupt the other
  - Training data for each is naturally separated
  - The Reasoner model can be replaced without touching the Communicator
  - Eventually different models can run each side

Today both use the same underlying llm_chat function. The architecture
is ready for different models immediately.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# ReasonerOutput — the interface between the two systems
# ============================================================

@dataclass
class ReasonerStep:
    """One step in a reasoning chain."""
    label:   str    # "STRUCTURAL" | "DERIVED" | "VERIFIABLE"
    content: str
    source:  str = "derived"   # "technique:{id}" | "derived" | "verified"


@dataclass
class ReasonerOutput:
    """
    The structured output of the Reasoner.
    This is what the Communicator receives — not natural language,
    but a typed epistemic structure.
    """
    run_id:                 str
    domain:                 str
    question:               str
    method:                 str
    steps:                  list[ReasonerStep]
    assumptions:            list[str]
    result:                 str
    verified:               bool
    confidence:             str       # "HIGH" | "MODERATE" | "LOW" | "UNCERTAIN"
    structural_fraction:    float
    obligations_raised:     list[str]
    obligations_discharged: list[str]
    timestamp:              float = field(default_factory=time.time)
    raw_json:               str   = ""   # the raw JSON from the Reasoner

    def to_pipeline_format(self) -> dict:
        """
        Convert to the existing pipeline's branch dict format.
        Allows BrainPipeline to feed into any module that expects
        the existing format.
        """
        return {
            "name":               f"reasoner::{self.method}",
            "method":             self.method,
            "assumptions":        self.assumptions,
            "steps":              [
                f"[{s.label}] {s.content}"
                for s in self.steps
            ],
            "candidate_result":   self.result,
            "structural_fraction": self.structural_fraction,
            "provenance": {
                "source":     "reasoner",
                "confidence": self.confidence,
                "verified":   self.verified,
            },
            "verification_report": {
                "verdict": {"verified": self.verified}
            },
        }

    def to_reasoner_training_text(self) -> str:
        """
        Format as training example for the Reasoner model.
        Epistemically precise, not engaging — that's correct for
        the Reasoner's training data.
        """
        lines = [f"Method: {self.method}", ""]
        if self.assumptions:
            lines.append("Assumptions:")
            for a in self.assumptions:
                lines.append(f"  - {a}")
            lines.append("")
        lines.append("Reasoning:")
        for step in self.steps:
            lines.append(f"  [{step.label}] {step.content}")
        lines.append(f"\nResult: {self.result}")
        lines.append(
            f"Verification: {'VERIFIED' if self.verified else 'UNVERIFIED'} "
            f"(confidence: {self.confidence})"
        )
        return "\n".join(lines)

    def to_communicator_input_text(self) -> str:
        """
        Format as input for the Communicator — structured enough
        for translation, clear about verification status.
        """
        lines = [
            f"Method: {self.method}",
            f"Verification: {'VERIFIED' if self.verified else 'UNVERIFIED'} "
            f"(confidence: {self.confidence})",
            "",
        ]
        if self.assumptions:
            lines.append("Assumptions:")
            for a in self.assumptions:
                lines.append(f"  - {a}")
            lines.append("")
        lines.append("Reasoning chain:")
        for step in self.steps:
            lines.append(f"  [{step.label}] {step.content}")
        lines.append(f"\nResult: {self.result}")
        if self.obligations_raised:
            lines.append(f"\nOpen questions raised:")
            for o in self.obligations_raised[:3]:
                lines.append(f"  - {o}")
        return "\n".join(lines)


# ============================================================
# Reasoner system prompt
# ============================================================

REASONER_SYSTEM_PROMPT = """You are a structured epistemic reasoning engine.

Your output is consumed by a separate communication system — not directly
by humans. Optimise for correctness and structure, not readability.

RULES:
- Every reasoning step must start with [STRUCTURAL], [DERIVED], or [VERIFIABLE]
- [STRUCTURAL]: follows from accumulated knowledge or a known technique template
- [DERIVED]:    requires inference — you are generating new epistemic content
- [VERIFIABLE]: can be checked symbolically or numerically
- State ALL assumptions explicitly before reasoning begins
- Be maximally concise — no filler words
- Do not explain your approach — execute it
- If you cannot resolve something, say so explicitly

CONFIDENCE:
- HIGH:      verified result with no open assumptions
- MODERATE:  verified but depends on unverified assumptions
- LOW:       not verified or depends on multiple assumptions
- UNCERTAIN: could not reach a conclusion

OUTPUT FORMAT — return ONLY valid JSON:
{
  "method": "name of the reasoning approach",
  "assumptions": ["assumption 1", "assumption 2"],
  "steps": [
    {"label": "STRUCTURAL", "content": "...", "source": "technique:id or derived"},
    {"label": "DERIVED",    "content": "...", "source": "derived"},
    {"label": "VERIFIABLE", "content": "...", "source": "verified"}
  ],
  "result": "concise statement of what was established",
  "confidence": "HIGH",
  "obligations_raised":     ["new gap 1"],
  "obligations_discharged": ["gap resolved 1"]
}"""


# ============================================================
# Communicator system prompts by register
# ============================================================

COMMUNICATOR_PROMPTS = {
    "formal": """You translate structured reasoning chains into precise formal language.

RULES:
- Do NOT reason — only translate
- Preserve ALL [STRUCTURAL] [DERIVED] [VERIFIABLE] labels exactly
- Make the logical flow between steps explicit
- Use precise technical language throughout
- Do not simplify — add logical connectives, not explanations
- Keep it tight: every sentence earns its place""",

    "pedagogical": """You translate structured reasoning chains into clear explanations
for someone learning this material.

RULES:
- Do NOT reason — only translate
- Preserve ALL [STRUCTURAL] [DERIVED] [VERIFIABLE] labels exactly
- Connect each step to the previous one explicitly
- Add one analogy per major concept where it genuinely illuminates
- Vary sentence length: short statements for structure, longer ones for derivations
- Make the "why" clear at each derived step without changing the what""",

    "intuitive": """You translate structured reasoning chains by foregrounding
the physical or conceptual intuition behind each step.

RULES:
- Do NOT reason — only translate
- Preserve ALL [STRUCTURAL] [DERIVED] [VERIFIABLE] labels exactly
- After each derived step, add a one-sentence intuitive gloss
- Use concrete analogies where a concept would otherwise seem arbitrary
- Do not dumb down — add insight, not simplification""",

    "conversational": """You translate structured reasoning chains into natural,
engaging language — as if explaining to an interested colleague.

RULES:
- Do NOT reason — only translate
- Preserve ALL [STRUCTURAL] [DERIVED] [VERIFIABLE] labels exactly
- Use natural transitions between steps
- Vary rhythm and sentence length
- Stay precise — engagement should not come at the cost of accuracy""",

    "balanced": """You translate structured reasoning chains into clear, engaging
language that balances precision and accessibility.

RULES:
- Do NOT reason — only translate
- Preserve ALL [STRUCTURAL] [DERIVED] [VERIFIABLE] labels exactly
- Make the prose less dry without sacrificing precision
- Add one analogy if it genuinely illuminates something
- Vary sentence length naturally""",
}


# ============================================================
# Reasoner
# ============================================================

class Reasoner:
    """
    Produces structured epistemic output.
    Optimised for correctness, not readability.
    Can use a different model than the Communicator.
    """

    def __init__(
        self,
        llm_chat_fn:    Callable,
        model_name:     str  = "",    # override model if supported by llm_chat_fn
        verbose:        bool = True,
    ):
        self.llm     = llm_chat_fn
        self.model   = model_name
        self.verbose = verbose

    def reason(
        self,
        question:          str,
        domain:            str     = "",
        memory_context:    str     = "",
        introspect_context: str    = "",
        run_id:            str     = "",
    ) -> ReasonerOutput:
        """
        Produce a structured ReasonerOutput for the given question.
        The output contains typed steps, not natural language.
        """
        run_id = run_id or f"run-{int(time.time())}"

        # Build the user prompt — memory and introspection injected here
        user_parts = []

        if introspect_context:
            user_parts.append(
                f"[SELF-KNOWLEDGE]\n{introspect_context}\n[/SELF-KNOWLEDGE]\n"
            )

        if memory_context:
            user_parts.append(
                f"[MEMORY CONTEXT]\n{memory_context}\n[/MEMORY CONTEXT]\n"
            )

        user_parts.append(f"Domain: {domain}")
        user_parts.append(f"Question: {question}")
        user_parts.append(
            "\nProduce a structured reasoning chain. Return ONLY valid JSON."
        )
        user = "\n".join(user_parts)

        if self.verbose:
            print(f"  [reasoner] Reasoning about: {question[:60]}...")

        try:
            raw = self.llm(REASONER_SYSTEM_PROMPT, user)
        except Exception as e:
            return self._fallback_output(run_id, domain, question, str(e))

        return self._parse_output(raw, run_id, domain, question)

    def _parse_output(
        self,
        raw: str,
        run_id: str,
        domain: str,
        question: str,
    ) -> ReasonerOutput:
        """Parse the JSON output from the Reasoner."""
        # Strip markdown fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(
                l for l in lines[1:] if not l.startswith("```")
            ).strip()

        # Find JSON object
        match = re.search(r"\{[\s\S]*\}", clean)
        if not match:
            return self._fallback_output(
                run_id, domain, question, "No JSON found in response"
            )

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            return self._fallback_output(run_id, domain, question, str(e))

        steps = [
            ReasonerStep(
                label=s.get("label", "DERIVED").upper(),
                content=s.get("content", ""),
                source=s.get("source", "derived"),
            )
            for s in data.get("steps", [])
        ]

        # Estimate structural fraction
        n_structural = sum(
            1 for s in steps
            if s.label in ("STRUCTURAL", "VERIFIABLE")
        )
        frac = n_structural / max(1, len(steps))

        # Determine verification status from step labels
        has_verifiable = any(s.label == "VERIFIABLE" for s in steps)
        confidence     = data.get("confidence", "MODERATE").upper()
        verified       = confidence in ("HIGH", "MODERATE") and has_verifiable

        if self.verbose:
            print(f"  [reasoner] {len(steps)} steps, "
                  f"structural={frac:.0%}, "
                  f"confidence={confidence}")

        return ReasonerOutput(
            run_id=run_id,
            domain=domain,
            question=question,
            method=data.get("method", "unknown"),
            steps=steps,
            assumptions=data.get("assumptions", []),
            result=data.get("result", ""),
            verified=verified,
            confidence=confidence,
            structural_fraction=frac,
            obligations_raised=data.get("obligations_raised", []),
            obligations_discharged=data.get("obligations_discharged", []),
            raw_json=match.group(0),
        )

    def _fallback_output(
        self,
        run_id: str,
        domain: str,
        question: str,
        error: str,
    ) -> ReasonerOutput:
        if self.verbose:
            print(f"  [reasoner] Parse failed: {error[:60]}")
        return ReasonerOutput(
            run_id=run_id, domain=domain, question=question,
            method="parse_failed", steps=[],
            assumptions=[], result=f"[reasoning failed: {error[:80]}]",
            verified=False, confidence="UNCERTAIN",
            structural_fraction=0.0,
            obligations_raised=[f"Reasoning failed for: {question[:80]}"],
            obligations_discharged=[],
        )


# ============================================================
# Communicator
# ============================================================

class Communicator:
    """
    Translates ReasonerOutput into natural language.
    Never reasons. Only translates.
    Can use a different model than the Reasoner.
    """

    # Simple register detection
    _FORMAL_KWS    = {"derive", "prove", "show", "calculate", "compute", "verify"}
    _PEDAGOGY_KWS  = {"why", "how does", "walk me", "teach", "help me", "understand"}
    _INTUITIVE_KWS = {"intuitively", "explain", "describe", "what is", "give me"}
    _CONV_KWS      = {"what do you think", "do you think", "in your view"}

    def __init__(
        self,
        llm_chat_fn:  Callable,
        model_name:   str  = "",
        style_adapter = None,
        verbose:      bool = True,
    ):
        self.llm     = llm_chat_fn
        self.model   = model_name
        self.adapter = style_adapter
        self.verbose = verbose

    def communicate(
        self,
        reasoning: ReasonerOutput,
        question:  str,
        register:  str = "auto",
    ) -> str:
        """
        Translate a ReasonerOutput into natural language.
        Returns the translated text.
        """
        if not reasoning.steps and not reasoning.result:
            return (
                f"I attempted to reason about '{question}' but could not "
                f"produce a structured reasoning chain. "
                f"The question may need a different approach."
            )

        # Detect register
        if register == "auto":
            register = self._detect_register(question)

        system = COMMUNICATOR_PROMPTS.get(register, COMMUNICATOR_PROMPTS["balanced"])

        structured_input = reasoning.to_communicator_input_text()

        user = (
            f"Question asked: {question}\n\n"
            f"Structured reasoning to translate:\n\n"
            f"{structured_input}\n\n"
            f"Translate this into natural language following your instructions. "
            f"Keep all [STRUCTURAL] [DERIVED] [VERIFIABLE] labels."
        )

        if self.verbose:
            print(f"  [communicator] Register: {register}")

        try:
            output = self.llm(system, user).strip()
        except Exception as e:
            # Fallback: structured representation
            return structured_input

        # Verify markers preserved — fallback if dropped
        original_markers = set(re.findall(
            r"\[STRUCTURAL\]|\[DERIVED\]|\[VERIFIABLE\]",
            structured_input
        ))
        output_markers = set(re.findall(
            r"\[STRUCTURAL\]|\[DERIVED\]|\[VERIFIABLE\]",
            output
        ))
        if original_markers and not original_markers.issubset(output_markers):
            if self.verbose:
                print(f"  [communicator] Markers dropped — using structured fallback")
            return structured_input

        return output

    @classmethod
    def _detect_register(cls, question: str) -> str:
        lower = question.lower()
        if any(k in lower for k in cls._CONV_KWS):
            return "conversational"
        if any(k in lower for k in cls._PEDAGOGY_KWS):
            return "pedagogical"
        if any(k in lower for k in cls._INTUITIVE_KWS):
            return "intuitive"
        if any(k in lower for k in cls._FORMAL_KWS):
            return "formal"
        return "balanced"


# ============================================================
# BrainPipeline — coordinator
# ============================================================

class BrainPipeline:
    """
    Coordinates Reasoner and Communicator.
    Drop-in replacement for EnhancedPipeline with the same ask() interface.

    Usage:
        from brain_pipeline import BrainPipeline
        from enhanced_pipeline import EnhancedConfig

        pipeline = BrainPipeline(
            reasoner_llm   = llm_chat,        # or a different model
            communicator_llm = llm_chat,      # can differ from Reasoner
            config         = EnhancedConfig(...),
        )
        result = pipeline.ask("Why do flat rotation curves exist?")
    """

    def __init__(
        self,
        reasoner_llm:     Callable,
        communicator_llm: Optional[Callable] = None,
        config            = None,
        search_fn:        Optional[Callable] = None,
        verbose:          bool = True,
    ):
        self.verbose = verbose

        # If only one LLM provided, both use it
        comm_llm = communicator_llm or reasoner_llm

        self.reasoner     = Reasoner(reasoner_llm,  verbose=verbose)
        self.communicator = Communicator(comm_llm,  verbose=verbose)

        # Initialise all scaffold modules through EnhancedPipeline
        self._enhanced: Optional[object] = None
        if config is not None:
            try:
                from enhanced_pipeline import EnhancedPipeline
                self._enhanced = EnhancedPipeline(
                    llm_chat_fn=reasoner_llm,
                    config=config,
                    search_fn=search_fn,
                )
                if verbose:
                    print("[brain_pipeline] Scaffold modules initialised via EnhancedPipeline")
            except Exception as e:
                if verbose:
                    print(f"[brain_pipeline] EnhancedPipeline init failed: {e}")

        # Training buffers: separate for Reasoner and Communicator
        self._reasoner_examples:    list[dict] = []
        self._communicator_examples: list[dict] = []

    # ---- Public interface ----

    def ask(
        self,
        question: str,
        domain:   str = "",
        register: str = "auto",
    ) -> dict:
        """
        Full pipeline run: reason → communicate → record.
        Returns a result dict with both the reasoning and the response.
        """
        if not domain and self._enhanced:
            domain = getattr(self._enhanced.cfg, "domain_name", "")

        t0 = time.time()

        # 1. Pre-processing: memory + introspection via EnhancedPipeline
        memory_ctx    = self._get_memory_context(question, domain)
        introspect_ctx = self._get_introspect_context(question, domain)

        # 2. Reason
        reasoning = self.reasoner.reason(
            question, domain,
            memory_context=memory_ctx,
            introspect_context=introspect_ctx,
        )

        # 3. Communicate
        response = self.communicator.communicate(reasoning, question, register)

        # 4. Post-processing via EnhancedPipeline scaffold
        pipeline_result = self._wrap_as_pipeline_result(reasoning, question)
        self._post_process(pipeline_result, reasoning, question, response)

        # 5. Store training examples
        self._store_training_example(reasoning, question, response)

        result = {
            "run_id":        reasoning.run_id,
            "question":      question,
            "domain":        domain,
            "reasoning":     reasoning,
            "response":      response,
            "verified":      reasoning.verified,
            "confidence":    reasoning.confidence,
            "elapsed":       round(time.time() - t0, 2),
            "selected_branch": reasoning.to_pipeline_format(),
        }

        if self.verbose:
            print(f"\n  Confidence: {reasoning.confidence}  "
                  f"Verified: {reasoning.verified}  "
                  f"({reasoning.structural_fraction:.0%} structural)")

        return result

    def consolidate(self, quiet: bool = False) -> dict:
        """Periodic maintenance — delegates to EnhancedPipeline."""
        if self._enhanced:
            return self._enhanced.consolidate(quiet=quiet)
        return {}

    def explore(self, n: int = 1) -> list:
        """Curiosity-driven autonomous exploration."""
        if self._enhanced:
            return self._enhanced.explore(n=n)
        return []

    def export_training_data(
        self,
        reasoner_path:     str = "./scaffold_data/reasoner_training.jsonl",
        communicator_path: str = "./scaffold_data/communicator_training.jsonl",
    ) -> tuple[int, int]:
        """
        Export separate training datasets for Reasoner and Communicator.
        These feed separate fine-tuning pipelines.
        """
        import os
        n_r = n_c = 0

        if self._reasoner_examples:
            os.makedirs(os.path.dirname(reasoner_path) or ".", exist_ok=True)
            with open(reasoner_path, "w") as f:
                for ex in self._reasoner_examples:
                    f.write(json.dumps(ex) + "\n")
            n_r = len(self._reasoner_examples)

        if self._communicator_examples:
            os.makedirs(os.path.dirname(communicator_path) or ".", exist_ok=True)
            with open(communicator_path, "w") as f:
                for ex in self._communicator_examples:
                    f.write(json.dumps(ex) + "\n")
            n_c = len(self._communicator_examples)

        if self.verbose:
            print(f"[brain_pipeline] Exported {n_r} Reasoner + "
                  f"{n_c} Communicator training examples")
        return n_r, n_c

    # ---- Helpers ----

    def _get_memory_context(self, question: str, domain: str) -> str:
        if not self._enhanced or not self._enhanced._memory:
            return ""
        try:
            mem = self._enhanced._memory.query(question, domain, top_k=3)
            if not mem.has_prior_experience():
                return ""
            parts = []
            if mem.relevant_techniques:
                lib   = self._enhanced._composer.library
                names = [
                    lib._data["techniques"].get(tid, {}).get("name", tid)
                    for tid, _ in mem.relevant_techniques[:3]
                ]
                parts.append(f"Relevant techniques: {', '.join(names)}")
            if mem.similar_past_runs:
                parts.append(
                    f"{len(mem.similar_past_runs)} similar past runs found "
                    f"(freshness {mem.freshness_score:.2f})"
                )
            return "\n".join(parts)
        except Exception:
            return ""

    def _get_introspect_context(self, question: str, domain: str) -> str:
        if not self._enhanced or not self._enhanced._introspect:
            return ""
        try:
            return self._enhanced._introspect.format_for_branch_generation(
                question, domain
            )
        except Exception:
            return ""

    def _wrap_as_pipeline_result(
        self, reasoning: ReasonerOutput, question: str
    ) -> dict:
        return {
            "run_id":          reasoning.run_id,
            "question":        question,
            "domain":          reasoning.domain,
            "selected_branch": reasoning.to_pipeline_format(),
            "all_branches":    [reasoning.to_pipeline_format()],
            "errors":          [],
            "resolution": {
                "obligations": [
                    {"status": "discharged", "text": o}
                    for o in reasoning.obligations_discharged
                ] + [
                    {"status": "open", "text": o}
                    for o in reasoning.obligations_raised
                ]
            },
        }

    def _post_process(
        self,
        pipeline_result: dict,
        reasoning:       ReasonerOutput,
        question:        str,
        response:        str,
    ) -> None:
        """Run post-processing via EnhancedPipeline scaffold modules."""
        if not self._enhanced:
            return
        try:
            self._enhanced._post_novelty(pipeline_result, reasoning.domain)
            self._enhanced._post_episodic(pipeline_result)
            self._enhanced._post_learning(pipeline_result)
            self._enhanced._post_adjuster()
        except Exception:
            pass

    def _store_training_example(
        self,
        reasoning: ReasonerOutput,
        question:  str,
        response:  str,
    ) -> None:
        """Store separate training examples for Reasoner and Communicator."""
        if not reasoning.verified:
            return

        from scaffold_trainer import SYSTEM_PROMPT

        # Reasoner training example
        self._reasoner_examples.append({
            "messages": [
                {"role": "system",    "content": REASONER_SYSTEM_PROMPT},
                {"role": "user",      "content": f"Domain: {reasoning.domain}\nQuestion: {question}"},
                {"role": "assistant", "content": reasoning.raw_json or reasoning.to_reasoner_training_text()},
            ],
            "metadata": {
                "run_id":    reasoning.run_id,
                "verified":  reasoning.verified,
                "confidence": reasoning.confidence,
            },
        })

        # Communicator training example
        self._communicator_examples.append({
            "messages": [
                {"role": "system",    "content": COMMUNICATOR_PROMPTS["balanced"]},
                {"role": "user",      "content": (
                    f"Question asked: {question}\n\n"
                    f"Structured reasoning to translate:\n\n"
                    f"{reasoning.to_communicator_input_text()}"
                )},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "run_id":   reasoning.run_id,
                "register": Communicator._detect_register(question),
                "verified": reasoning.verified,
            },
        })

    # ---- Inspection ----

    def status(self) -> str:
        n_r = len(self._reasoner_examples)
        n_c = len(self._communicator_examples)
        lines = [
            "BrainPipeline status:",
            f"  Reasoner training examples:     {n_r}",
            f"  Communicator training examples: {n_c}",
            f"  Scaffold modules active: "
            f"{len(self._enhanced._active_modules()) if self._enhanced else 0}",
        ]
        return "\n".join(lines)
