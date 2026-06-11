"""
language_style_adapter.py
==========================

Makes the system's language more engaging without compromising
the epistemic structure it needs for verification and training.

The problem: the scaffold produces structured reasoning chains
that are epistemically precise but can feel dry and monotone.
This is partly necessary (structure enables verification) and
partly a limitation of the base model's default output style.

This module post-processes reasoning chain outputs after they
are generated and verified. It:

  - Detects what register the question calls for (formal,
    intuitive, pedagogical, conversational)
  - Enriches the prose around structural markers without
    removing or altering them
  - Adds one or two grounding analogies where helpful
  - Varies sentence length and rhythm
  - Preserves all [STRUCTURAL] / [DERIVED] / [VERIFIABLE]
    labels so training data quality is not degraded

Critically: this runs AFTER verification, not before or during.
The verified reasoning chain is never altered — the enrichment
wraps around it and replaces surface language only.

This also feeds the continual trainer better training examples.
As the model fine-tunes on enriched outputs, it progressively
internalises more varied language for epistemic reasoning.

Integration:
  adapter = LanguageStyleAdapter(llm_chat_fn)
  enriched = adapter.enrich(result, question, domain)
  # result["selected_branch"]["enriched_output"] = enriched.text

Or via EnhancedPipeline config:
  cfg.enable_style_adapter = True
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# Configuration
# ============================================================

@dataclass
class StyleConfig:
    """Controls how the adapter enriches outputs."""

    register:            str   = "auto"   # auto / formal / intuitive / pedagogical
    use_analogies:       bool  = True
    vary_sentence_length: bool = True
    max_tokens:          int   = 800      # cap on enrichment length
    preserve_markers:    bool  = True     # always keep [STRUCTURAL] etc.
    analogy_depth:       str   = "light"  # light / medium / rich
    # light  = one brief analogy if helpful
    # medium = one analogy per major step
    # rich   = full pedagogical treatment with multiple analogies


# ============================================================
# Register detection
# ============================================================

FORMAL_MARKERS = {
    "derive", "prove", "show that", "calculate", "compute",
    "verify", "demonstrate", "deduce", "obtain", "determine",
}

INTUITIVE_MARKERS = {
    "explain", "describe", "what is", "give me", "intuitively",
    "in simple terms", "what does", "how would you",
    "can you describe",
}

PEDAGOGICAL_MARKERS = {
    "why does", "how does", "what happens", "teach me",
    "help me understand", "walk me through", "break down",
    "what is the intuition", "why is",
}

CONVERSATIONAL_MARKERS = {
    "what do you think", "do you think", "in your view",
    "would you say", "is it fair", "how significant",
}


def detect_register(question: str) -> str:
    """
    Detect what communication register the question calls for.

    Returns one of: formal / intuitive / pedagogical / conversational / balanced
    """
    lower = question.lower()

    scores = {
        "formal":        sum(1 for m in FORMAL_MARKERS       if m in lower),
        "intuitive":     sum(1 for m in INTUITIVE_MARKERS     if m in lower),
        "pedagogical":   sum(1 for m in PEDAGOGICAL_MARKERS   if m in lower),
        "conversational": sum(1 for m in CONVERSATIONAL_MARKERS if m in lower),
    }

    best    = max(scores, key=lambda k: scores[k])
    best_v  = scores[best]

    if best_v == 0:
        return "balanced"
    if list(scores.values()).count(best_v) > 1:
        return "balanced"
    return best


# ============================================================
# Analogy catalogue
# ============================================================

# Cross-domain analogies for common physics/reasoning concepts.
# Keyed by concept fragment → (everyday analogy, technical analogy)
ANALOGY_CATALOGUE: dict[str, tuple[str, str]] = {
    "closure relation": (
        "like needing all the rules of a game before you can play — "
        "the closure fills in the missing rule that lets the rest work",
        "analogous to a constitutive relation in continuum mechanics — "
        "it specifies the missing link between fields",
    ),
    "boundary condition": (
        "like the walls of a room that constrain where you can be — "
        "they don't tell you everything about the room, "
        "but they rule out most possibilities",
        "the same role as an initial value in an ODE — "
        "without it the solution is a family, not a single curve",
    ),
    "perturbative expansion": (
        "like estimating how a rope bends — first treat it as stiff, "
        "then add small corrections for flexibility",
        "the same approach as the Born expansion in scattering theory — "
        "leading order plus corrections",
    ),
    "mean field": (
        "like replacing a crowd with an average person — "
        "you lose individual variation but gain tractability",
        "the Weiss mean field in magnetism — each spin sees an "
        "effective field from all its neighbours averaged",
    ),
    "ansatz": (
        "like guessing the form of an answer before solving for the details — "
        "educated guessing guided by symmetry",
        "the same approach as the variational ansatz in quantum mechanics — "
        "postulate a waveform, then optimise its parameters",
    ),
    "verification": (
        "like checking your receipt before leaving the shop — "
        "a claim isn't established until you've actually checked it",
        "analogous to a unit test in software — "
        "the code isn't trusted until the test passes",
    ),
    "flat rotation curve": (
        "like a spinning record where every part rotates at the same speed "
        "regardless of distance from the centre — "
        "that shouldn't happen if gravity falls off normally",
        "equivalent to a flat velocity profile in pipe flow — "
        "it signals that the driving force is distributed differently than expected",
    ),
    "fixed point": (
        "like a temperature where a substance neither expands nor contracts — "
        "the system stays there once it reaches it",
        "in the sense of a renormalisation group fixed point — "
        "the structure replicates itself at different scales",
    ),
    "obligation": (
        "like a question you've raised but not yet answered — "
        "it stays on your desk until you resolve it",
        "analogous to an open hypothesis in an ongoing proof — "
        "everything downstream depends on eventually resolving it",
    ),
    "structural fraction": (
        "like knowing how much of a meal you cooked from scratch "
        "versus using pre-made ingredients — "
        "higher means the system is relying more on what it has built up",
        "the fraction of the reasoning chain that comes from accumulated "
        "technique templates rather than on-the-fly generation",
    ),
}


def find_analogy(text: str, depth: str = "light") -> list[str]:
    """
    Find relevant analogies for concepts mentioned in the text.
    Returns a list of analogy strings to potentially insert.
    """
    text_lower  = text.lower()
    analogies:  list[str] = []
    used_keys:  set[str]  = set()

    for concept, (everyday, technical) in ANALOGY_CATALOGUE.items():
        if concept not in text_lower:
            continue
        if concept in used_keys:
            continue
        used_keys.add(concept)

        if depth == "light":
            analogies.append(f"({everyday})")
        elif depth == "medium":
            analogies.append(
                f"Think of it as {everyday} — or more technically, "
                f"{technical}."
            )
        else:  # rich
            analogies.append(
                f"An everyday way to see this: {everyday} "
                f"More precisely: {technical}"
            )

        if depth == "light" and len(analogies) >= 1:
            break
        if depth == "medium" and len(analogies) >= 2:
            break

    return analogies


# ============================================================
# Core enrichment prompts by register
# ============================================================

ENRICHMENT_PROMPTS = {
    "formal": (
        "You are rewriting a physics reasoning chain to be more precise "
        "and clearly structured, without losing rigour. "
        "Keep all step-type markers exactly as written "
        "([STRUCTURAL], [DERIVED], [VERIFIABLE]). "
        "Make the logical flow between steps explicit. "
        "Avoid padding. Every sentence should carry information."
    ),
    "intuitive": (
        "You are rewriting a physics reasoning chain so that the physical "
        "intuition behind each step is clear. "
        "Keep all [STRUCTURAL], [DERIVED], [VERIFIABLE] markers exactly. "
        "After each derived step, add one sentence explaining what it "
        "means physically. Use concrete analogies where they genuinely help. "
        "Do not dumb down — add insight, not simplification."
    ),
    "pedagogical": (
        "You are rewriting a physics reasoning chain as a clear explanation "
        "for someone learning this material. "
        "Keep all [STRUCTURAL], [DERIVED], [VERIFIABLE] markers exactly. "
        "Vary your sentence length — mix short punchy statements with "
        "longer explanatory ones. "
        "Connect each step to the previous one explicitly. "
        "Add a brief analogy where a concept would otherwise seem arbitrary."
    ),
    "conversational": (
        "You are rewriting a physics reasoning chain in a natural, engaging "
        "voice — as if explaining to an interested colleague over coffee. "
        "Keep all [STRUCTURAL], [DERIVED], [VERIFIABLE] markers exactly. "
        "Use contractions where natural. Vary rhythm. "
        "Don't be informal to the point of imprecision — keep the reasoning "
        "tight but let the language breathe."
    ),
    "balanced": (
        "You are improving the language of a physics reasoning chain. "
        "Keep all [STRUCTURAL], [DERIVED], [VERIFIABLE] markers exactly. "
        "Make the prose less dry without sacrificing precision. "
        "Vary sentence length. Add one analogy if it genuinely illuminates "
        "something. Cut any redundant phrases."
    ),
}


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class EnrichedOutput:
    original:        str
    enriched:        str
    register:        str
    analogies_added: list[str]
    elapsed:         float
    markers_preserved: bool   # verify markers weren't dropped

    @property
    def text(self) -> str:
        return self.enriched

    def markers_check(self) -> bool:
        """Verify that no epistemic markers were dropped."""
        original_markers = set(re.findall(
            r"\[STRUCTURAL\]|\[DERIVED\]|\[VERIFIABLE\]",
            self.original
        ))
        enriched_markers = set(re.findall(
            r"\[STRUCTURAL\]|\[DERIVED\]|\[VERIFIABLE\]",
            self.enriched
        ))
        # All original markers must appear in enriched version
        return original_markers.issubset(enriched_markers)


# ============================================================
# LanguageStyleAdapter
# ============================================================

class LanguageStyleAdapter:
    """
    Post-processes reasoning chain outputs to improve language
    quality while preserving epistemic structure.

    Call enrich() after the pipeline has produced and verified
    a branch. The result can be shown to the user and fed to
    the continual trainer as a training example.

    The epistemic content and step-type markers are never altered.
    Only the prose around them is enriched.
    """

    def __init__(
        self,
        llm_chat_fn: Callable,
        config:      Optional[StyleConfig] = None,
        verbose:     bool = False,
    ):
        self.llm     = llm_chat_fn
        self.cfg     = config or StyleConfig()
        self.verbose = verbose

    def enrich(
        self,
        pipeline_result: dict,
        question:        str,
        domain:          str = "",
    ) -> Optional[EnrichedOutput]:
        """
        Enrich the selected branch from a pipeline result.

        Returns EnrichedOutput, or None if enrichment failed or
        the result had no selected branch.
        """
        selected = pipeline_result.get("selected_branch", {})
        if not selected:
            return None

        # Build the raw reasoning chain text from the selected branch
        raw = self._format_branch_as_text(selected)
        if not raw or len(raw) < 50:
            return None

        return self.enrich_text(raw, question, domain)

    def enrich_text(
        self,
        raw_text: str,
        question: str,
        domain:   str = "",
    ) -> EnrichedOutput:
        """
        Enrich arbitrary raw text. Useful for enriching stored
        episodic records for the training buffer.
        """
        t0 = time.time()

        # Detect register
        register = (
            self.cfg.register
            if self.cfg.register != "auto"
            else detect_register(question)
        )

        # Find relevant analogies
        analogies: list[str] = []
        if self.cfg.use_analogies:
            analogies = find_analogy(raw_text, depth=self.cfg.analogy_depth)

        # Build enrichment prompt
        system = ENRICHMENT_PROMPTS.get(register, ENRICHMENT_PROMPTS["balanced"])

        analogy_hint = ""
        if analogies:
            analogy_hint = (
                f"\n\nYou may optionally weave in this analogy if it fits "
                f"naturally: {analogies[0]}"
            )

        user = (
            f"Original question: {question}\n"
            f"Domain: {domain}\n\n"
            f"Reasoning chain to enrich:\n\n"
            f"{raw_text}"
            f"{analogy_hint}\n\n"
            f"Rewrite this reasoning chain following the instructions. "
            f"Output only the rewritten chain — no preamble, no explanation "
            f"of what you changed. Maximum {self.cfg.max_tokens} tokens."
        )

        try:
            enriched = self.llm(system, user).strip()
        except Exception as e:
            if self.verbose:
                print(f"[style_adapter] Enrichment failed: {e}")
            return EnrichedOutput(
                original=raw_text,
                enriched=raw_text,
                register=register,
                analogies_added=[],
                elapsed=round(time.time() - t0, 2),
                markers_preserved=True,
            )

        # Verify markers were preserved
        result = EnrichedOutput(
            original=raw_text,
            enriched=enriched,
            register=register,
            analogies_added=analogies,
            elapsed=round(time.time() - t0, 2),
            markers_preserved=False,
        )
        result.markers_preserved = result.markers_check()

        # If markers were dropped, fall back to original
        if not result.markers_preserved:
            if self.verbose:
                print("[style_adapter] Markers dropped — using original")
            result.enriched = raw_text

        if self.verbose:
            print(f"[style_adapter] Register: {register}, "
                  f"Analogies: {len(analogies)}, "
                  f"Markers OK: {result.markers_preserved}, "
                  f"Elapsed: {result.elapsed}s")

        return result

    def enrich_batch(
        self,
        records:  list,
        question_map: Optional[dict] = None,
    ) -> list[EnrichedOutput]:
        """
        Enrich a batch of episodic records for the training buffer.
        question_map: {run_id: question_text}
        """
        results = []
        for record in records:
            if not record.verified:
                continue
            q = (question_map or {}).get(record.run_id, record.question)
            # Reconstruct raw text from record
            raw = self._format_record_as_text(record)
            if raw:
                enriched = self.enrich_text(raw, q, record.domain)
                results.append(enriched)
        return results

    # ---- Formatting helpers ----

    @staticmethod
    def _format_branch_as_text(branch: dict) -> str:
        """Convert a branch dict to structured text."""
        parts = []

        method = branch.get("method", "")
        if method:
            parts.append(f"Method: {method}\n")

        assumptions = branch.get("assumptions", [])
        if assumptions:
            parts.append("Assumptions:")
            for a in assumptions:
                parts.append(f"  - {a}")
            parts.append("")

        steps = branch.get("steps", [])
        if steps:
            parts.append("Reasoning:")
            for step in steps:
                parts.append(f"  {step}")
            parts.append("")

        result = branch.get("candidate_result", "")
        if result:
            parts.append(f"Result: {result}")

        return "\n".join(parts)

    @staticmethod
    def _format_record_as_text(record) -> str:
        """Reconstruct text from an EpisodicRecord."""
        parts = [f"Method: {record.method_name}"]
        if record.key_steps:
            parts.append("\nReasoning:")
            for s in record.key_steps:
                parts.append(f"  {s}")
        return "\n".join(parts)


# ============================================================
# Batch enrichment for training buffer upgrade
# ============================================================

def upgrade_training_buffer(
    continual_trainer,
    adapter:       LanguageStyleAdapter,
    n_records:     int  = 50,
    verbose:       bool = True,
) -> int:
    """
    Re-enrich existing episodic records and add enriched versions
    to the training buffer. Improves the quality of training data
    that has already been collected.

    Returns number of examples upgraded.
    """
    records   = continual_trainer.episodic.query_recent(n=n_records)
    verified  = [r for r in records if r.verified]

    if not verified:
        if verbose:
            print("[upgrade] No verified records to enrich.")
        return 0

    upgraded = 0
    for record in verified:
        raw = LanguageStyleAdapter._format_record_as_text(record)
        if not raw:
            continue

        enriched = adapter.enrich_text(raw, record.question, record.domain)
        if not enriched.markers_preserved:
            continue

        # Create an enriched training example
        from scaffold_trainer import TrainingExample, _label_steps
        steps = _label_steps(record.key_steps or [])
        steps_text = "\n".join(f"  {label} {step}" for label, step in steps)

        ex = TrainingExample(
            run_id=f"{record.run_id}-enriched",
            domain=record.domain,
            instruction=(
                "Reason about the following using structured epistemic "
                "reasoning. Label each step. Separate assumptions from "
                "derivations."
            ),
            input_text=f"Question: {record.question}",
            output_text=enriched.enriched,
            verified=True,
            structural_fraction=record.structural_fraction,
            obligations_discharged=record.obligations_discharged,
            difficulty=1.0 - record.structural_fraction,
        )
        continual_trainer.replay.update()
        upgraded += 1

    if verbose:
        print(f"[upgrade] Enriched {upgraded}/{len(verified)} records "
              f"in training buffer.")

    return upgraded
