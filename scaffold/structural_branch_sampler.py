"""
structural_branch_sampler.py
============================

Generates reasoning branch skeletons from accumulated structures
rather than from open-ended LLM generation.

This is the inversion point: instead of asking an LLM to generate
everything and then verifying the output, the structures generate
the skeleton and the LLM fills specific bounded gaps.

The output is identical to _generate_branches() in pipeline.py —
a list of dicts with: name, method, assumptions, steps,
candidate_result, notes. Drop-in compatible.

Three step types in each branch:

  STRUCTURAL  — derived from the technique's template and the
                 problem's known structure. Free. No LLM call.

  DERIVED     — requires reasoning that the structure can't yet
                 supply. The LLM is asked one bounded question
                 ("given X and Y, what is Z?") not an open one.

  VERIFIABLE  — can be checked by sympy or Z3 directly.
                 No LLM call needed.

The structural_fraction field tracks what proportion of each branch
came from structures. As the technique library grows, this rises.
When it consistently exceeds 0.7, the LLM is genuinely a gap-filler
rather than the primary generator — that's the threshold where
the native architecture is doing most of the work.
"""

import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# Step classification vocabulary
# ============================================================

STRUCTURAL_VERBS = {
    "identify", "find", "locate", "extract", "recognise", "recognize",
    "define", "formulate", "set up", "state", "express", "write",
    "postulate", "assume", "introduce", "denote", "let", "note",
    "observe", "record", "list",
}

DERIVED_VERBS = {
    "compute", "calculate", "derive", "apply", "substitute", "expand",
    "integrate", "differentiate", "project", "obtain", "solve",
    "simplify", "evaluate", "determine", "deduce", "infer",
    "combine", "use", "substitute", "insert",
}

VERIFIABLE_VERBS = {
    "verify", "check", "test", "confirm", "validate", "sample",
    "assert", "ensure", "compare", "agree", "match",
}


def _classify_sentence(sentence: str) -> str:
    """
    Classify a sentence as STRUCTURAL, DERIVED, or VERIFIABLE
    by its leading verb.
    """
    tokens = re.findall(r"[a-z]+", sentence.lower())
    for token in tokens[:3]:
        if token in VERIFIABLE_VERBS:
            return "verifiable"
        if token in STRUCTURAL_VERBS:
            return "structural"
        if token in DERIVED_VERBS:
            return "derived"
    return "derived"   # safe default — LLM fills unknown steps


# ============================================================
# Data structures
# ============================================================

@dataclass
class BranchStep:
    """One step in a branch skeleton."""

    description:  str
    step_type:    str           # "structural" | "derived" | "verifiable"
    source:       str           # "technique:{id}" | "derived" | "verified"
    gap_prompt:   str  = ""     # what to ask the LLM if step_type=="derived"
    result:       str  = ""     # filled by fill_skeleton()


@dataclass
class BranchSkeleton:
    """
    A branch before LLM gap-filling.
    Structural steps are already populated from technique templates.
    Derived steps are empty, with gap_prompt set.
    """

    name:                 str
    method_technique_id:  str
    method_name:          str
    domain:               str
    steps:                list[BranchStep]
    assumptions:          list[str]
    candidate_result:     str            = ""
    structural_fraction:  float          = 0.0
    provenance:           dict           = field(default_factory=dict)

    def n_structural(self) -> int:
        return sum(1 for s in self.steps if s.step_type == "structural")

    def n_derived(self) -> int:
        return sum(1 for s in self.steps if s.step_type == "derived")

    def n_verifiable(self) -> int:
        return sum(1 for s in self.steps if s.step_type == "verifiable")


# ============================================================
# Step extraction from technique descriptions
# ============================================================

def _extract_steps_from_technique(
    technique_dict: dict,
    problem_context: str = "",
) -> list[BranchStep]:
    """
    Parse a technique's description, when_to_use, and example_text
    into a structured list of BranchStep objects.

    Templates come from the example_text. Classification comes from
    the verb at the start of each sentence.
    """
    tid    = technique_dict.get("technique_id", "")
    name   = technique_dict.get("name", "")
    desc   = technique_dict.get("description", "")
    when   = technique_dict.get("when_to_use", "")
    ex     = technique_dict.get("example_text", "")

    steps: list[BranchStep] = []

    # Step 0 — always structural: identify what this technique addresses
    if when:
        steps.append(BranchStep(
            description=f"Confirm precondition: {when.rstrip('.')}.",
            step_type="structural",
            source=f"technique:{tid}",
        ))

    # Steps from example_text — the most template-rich field
    if ex:
        # Split on periods, semicolons, "then", numbered markers
        raw_steps = re.split(r"[.;]|\bthen\b|\n", ex)
        for raw in raw_steps:
            raw = raw.strip()
            if len(raw) < 8:
                continue
            stype = _classify_sentence(raw)
            gap = ""
            if stype == "derived":
                gap = (
                    f"Technique being applied: '{name}'. "
                    f"Prior context: {desc}. "
                    f"Step to complete: '{raw}'. "
                    f"Given the current problem, complete this specific step "
                    f"concisely in one or two sentences. Do not re-explain "
                    f"the technique — just execute this step."
                )
            steps.append(BranchStep(
                description=raw,
                step_type=stype,
                source=f"technique:{tid}",
                gap_prompt=gap,
            ))

    # If example_text gave us nothing, fall back to description sentences
    if len(steps) <= 1:
        for raw in re.split(r"[.;]", desc):
            raw = raw.strip()
            if len(raw) < 8:
                continue
            stype = _classify_sentence(raw)
            gap = ""
            if stype == "derived":
                gap = (
                    f"Technique: '{name}'. "
                    f"Step: '{raw}'. "
                    f"Execute this step for the current problem concisely."
                )
            steps.append(BranchStep(
                description=raw,
                step_type=stype,
                source=f"technique:{tid}",
                gap_prompt=gap,
            ))

    # Final step — always verifiable: check the result is consistent
    steps.append(BranchStep(
        description=(
            f"Verify the result of '{name}' is consistent with the "
            f"known constraints and the problem statement."
        ),
        step_type="verifiable",
        source=f"technique:{tid}",
    ))

    return steps


def _extract_assumptions_from_technique(technique_dict: dict) -> list[str]:
    """
    Extract assumptions from a technique description by finding
    sentences containing assumption markers.
    """
    MARKERS = {
        "assume", "assumption", "postulate", "ansatz", "given that",
        "treat as", "take as", "by hypothesis", "stipulate",
    }
    text = " ".join([
        technique_dict.get("description",   ""),
        technique_dict.get("when_to_use",   ""),
        technique_dict.get("example_text",  ""),
    ])
    assumptions = []
    for sentence in re.split(r"[.;]", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        lower = sentence.lower()
        if any(m in lower for m in MARKERS):
            assumptions.append(sentence)
    return assumptions[:4]   # cap at 4 to keep branches focused


# ============================================================
# StructuralBranchSampler
# ============================================================

class StructuralBranchSampler:
    """
    Generates branch skeletons from accumulated structures.

    Usage:
        from technique_embedder import EmbeddedTechniqueComposer
        from directed_graph import DirectedCoOccurrenceGraph
        from structural_branch_sampler import StructuralBranchSampler

        sampler = StructuralBranchSampler(
            embedded_composer=composer,
            directed_graph=dgraph,
        )

        # Drop-in replacement for _generate_branches() in pipeline.py
        branches = sampler.generate_branches(
            question="derive the MOND coefficient from CP2 closure",
            domain="physics_mond",
            n=3,
            llm_chat_fn=llm_chat,
        )
        # branches is a list of dicts identical to pipeline's branch format
    """

    def __init__(
        self,
        embedded_composer,
        directed_graph=None,
        verbose: bool = True,
    ):
        self.composer       = embedded_composer
        self.directed_graph = directed_graph
        self.verbose        = verbose

    # ---- Public API — drop-in for pipeline ----

    def generate_branches(
        self,
        question:    str,
        domain:      str,
        n:           int  = 3,
        llm_chat_fn: Optional[Callable] = None,
        obligations: Optional[list]     = None,
    ) -> list[dict]:
        """
        Drop-in replacement for _generate_branches() in pipeline.py.

        Returns list of branch dicts:
          {name, method, assumptions, steps, candidate_result, notes,
           structural_fraction, provenance}

        If the embedder has no trained techniques for this domain, falls
        back gracefully with a single diagnostic branch skeleton that
        tells the pipeline it needs more runs to build structure.
        """
        skeletons = self.sample_skeletons(question, domain, n, obligations)

        if not skeletons:
            return self._fallback_branch(question, domain)

        branches = []
        for sk in skeletons:
            filled = self.fill_skeleton(sk, question, llm_chat_fn)
            branches.append(filled)

        if self.verbose:
            fracs = [b.get("structural_fraction", 0) for b in branches]
            avg   = sum(fracs) / len(fracs) if fracs else 0
            print(f"  [sampler] {len(branches)} branches generated. "
                  f"Avg structural fraction: {avg:.0%}")

        return branches

    # ---- Skeleton generation ----

    def sample_skeletons(
        self,
        question:    str,
        domain:      str,
        n:           int  = 3,
        obligations: Optional[list] = None,
    ) -> list[BranchSkeleton]:
        """
        Generate n BranchSkeleton objects using the technique embedder
        for selection and the directed graph for sequencing.
        """
        if not self.composer.embedder.is_initialised:
            return []

        # Augment query with obligation context if available
        query = question
        if obligations:
            gap_texts = " ".join(
                o.get("text", "") for o in obligations[:3]
            )
            query = f"{question} [known gaps: {gap_texts}]"

        # Find candidate techniques
        candidates = self.composer.find_similar_techniques(
            query, domain=domain, top_n=n * 2,
        )
        if not candidates:
            return []

        skeletons: list[BranchSkeleton] = []
        used_names: set[str] = set()
        techs = self.composer.library._data.get("techniques", {})

        for tid, score in candidates:
            if len(skeletons) >= n:
                break
            t = techs.get(tid)
            if t is None:
                continue
            name = t.get("name", "")
            if name in used_names:
                continue
            used_names.add(name)

            sk = self._build_skeleton(tid, t, domain, score, query)
            skeletons.append(sk)

        return skeletons

    def _build_skeleton(
        self,
        tid:    str,
        t:      dict,
        domain: str,
        score:  float,
        query:  str,
    ) -> BranchSkeleton:
        """Build one BranchSkeleton from a technique dict."""
        steps       = _extract_steps_from_technique(t, query)
        assumptions = _extract_assumptions_from_technique(t)

        n_total      = max(1, len(steps))
        n_structural = sum(1 for s in steps if s.step_type == "structural")
        n_verifiable = sum(1 for s in steps if s.step_type == "verifiable")
        frac         = (n_structural + n_verifiable) / n_total

        # Add sequencing note from directed graph if available
        sequencing_note = ""
        if self.directed_graph is not None:
            companions = self.directed_graph.companions(tid, top_n=2)
            if companions:
                techs = self.composer.library._data.get("techniques", {})
                comp_names = [
                    techs[c_tid]["name"]
                    for c_tid, _ in companions
                    if c_tid in techs
                ]
                if comp_names:
                    sequencing_note = (
                        f"Historically co-occurs with: "
                        + ", ".join(comp_names)
                    )

        return BranchSkeleton(
            name=f"struct::{t.get('name', tid)}",
            method_technique_id=tid,
            method_name=t.get("name", ""),
            domain=domain,
            steps=steps,
            assumptions=assumptions,
            structural_fraction=frac,
            provenance={
                "technique_id":    tid,
                "similarity_score": round(score, 4),
                "source":          "structural_branch_sampler",
                "sequencing":      sequencing_note,
                "n_structural":    n_structural,
                "n_derived":       sum(1 for s in steps
                                       if s.step_type == "derived"),
                "n_verifiable":    n_verifiable,
            },
        )

    # ---- Gap filling ----

    def fill_skeleton(
        self,
        skeleton:    BranchSkeleton,
        question:    str,
        llm_chat_fn: Optional[Callable],
    ) -> dict:
        """
        Fill derived gaps in a skeleton with bounded LLM calls.
        Returns a pipeline-compatible branch dict.
        """
        filled_descriptions: list[str] = []

        for i, step in enumerate(skeleton.steps):
            if step.step_type in ("structural", "verifiable"):
                # No LLM needed
                filled_descriptions.append(step.description)
                step.result = step.description

            elif step.step_type == "derived":
                if llm_chat_fn and step.gap_prompt:
                    # Bounded LLM call — one specific question
                    prior = "\n".join(
                        f"  Step {j+1}: {s.result or s.description}"
                        for j, s in enumerate(skeleton.steps[:i])
                        if s.result or s.description
                    )
                    system = (
                        "You are completing one specific step in a structured "
                        "reasoning chain. The step is bounded and specific. "
                        "Answer in 1-3 sentences. Do not re-explain the method. "
                        "Do not add new assumptions. Just complete the step."
                    )
                    user = (
                        f"Problem: {question}\n\n"
                        f"Prior completed steps:\n{prior}\n\n"
                        f"Step to complete: {step.gap_prompt}"
                    )
                    try:
                        result = llm_chat_fn(system, user)
                        step.result = result.strip()[:400]
                    except Exception as e:
                        step.result = (
                            f"[gap: {step.description[:80]} "
                            f"— could not fill: {e}]"
                        )
                else:
                    step.result = f"[gap: {step.description}]"

                filled_descriptions.append(
                    step.result or f"[gap: {step.description}]"
                )

        # Derive the candidate result
        candidate_result = self._derive_candidate_result(
            skeleton, question, filled_descriptions, llm_chat_fn,
        )

        return {
            "name":                skeleton.name,
            "method":              skeleton.method_name,
            "assumptions":         skeleton.assumptions,
            "steps":               filled_descriptions,
            "candidate_result":    candidate_result,
            "notes": [
                f"structural_fraction={skeleton.structural_fraction:.0%}",
                f"technique: {skeleton.method_name}",
            ] + (
                [skeleton.provenance["sequencing"]]
                if skeleton.provenance.get("sequencing") else []
            ),
            "structural_fraction": skeleton.structural_fraction,
            "provenance":          skeleton.provenance,
        }

    def _derive_candidate_result(
        self,
        skeleton:             BranchSkeleton,
        question:             str,
        filled_descriptions:  list[str],
        llm_chat_fn:          Optional[Callable],
    ) -> str:
        """
        Derive the candidate result from the filled steps.
        If sympy can compute it, do that. Otherwise bounded LLM call.
        """
        # For now: bounded LLM call using only the filled step descriptions
        if not llm_chat_fn:
            return "[result not computed — no llm_chat_fn provided]"

        system = (
            "You are deriving the result of a structured reasoning chain. "
            "Based ONLY on the completed steps below, state what was "
            "established. One sentence. If the steps do not reach a "
            "conclusion, state what remains open."
        )
        steps_text = "\n".join(
            f"  {i+1}. {s}" for i, s in enumerate(filled_descriptions)
        )
        user = (
            f"Question: {question}\n\n"
            f"Completed steps for '{skeleton.method_name}':\n{steps_text}\n\n"
            f"What does this chain establish? One sentence."
        )
        try:
            return llm_chat_fn(system, user).strip()[:300]
        except Exception:
            return "[result derivation failed]"

    # ---- Fallback ----

    def _fallback_branch(self, question: str, domain: str) -> list[dict]:
        """
        When no trained techniques match, return a diagnostic branch
        that tells the pipeline the embedder needs more runs.
        """
        return [{
            "name":             "diagnostic::insufficient_structure",
            "method":           "diagnostic",
            "assumptions":      [],
            "steps":            [
                "The technique library has no trained embeddings "
                f"for domain '{domain}' yet.",
                "Run this question through the standard LLM-based pipeline "
                "to generate techniques for training.",
                "After 5+ verified runs, the structural sampler will have "
                "enough data to generate structured branches.",
            ],
            "candidate_result": (
                "Structural branch generation not yet available for this "
                "domain. Run LLM-based pipeline first to build technique library."
            ),
            "notes":            ["structural_fraction=0%", "fallback branch"],
            "structural_fraction": 0.0,
            "provenance":       {"source": "fallback"},
        }]

    # ---- Introspection ----

    def readiness(self, domain: str) -> dict:
        """
        How ready is the sampler for this domain?
        Returns a dict describing current state and what's needed.
        """
        techs = self.composer.library._data.get("techniques", {})
        domain_techs = [t for t in techs.values() if t.get("domain") == domain]
        n_embedded   = sum(
            1 for tid in techs
            if tid in self.composer.embedder._cache
        )
        trained = self.composer.embedder.history.steps > 0

        return {
            "domain":           domain,
            "n_techniques":     len(domain_techs),
            "n_embedded":       n_embedded,
            "embedder_trained": trained,
            "training_steps":   self.composer.embedder.history.steps,
            "ready":            trained and len(domain_techs) >= 3,
            "recommendation": (
                "Ready for structural branch generation."
                if trained and len(domain_techs) >= 3
                else f"Need ≥ 3 techniques in '{domain}' and trained embedder. "
                     f"Currently {len(domain_techs)} technique(s). "
                     "Run LLM-based pipeline to build library, then call "
                     "composer.initialise_and_train()."
            ),
        }
