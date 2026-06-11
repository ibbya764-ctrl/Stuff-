"""
domains.py
==========

Domain abstraction for the reasoning assistant.

A Domain bundles together everything that makes a problem-type specific:
  - what keywords signal targets, derivations, obligations
  - what tools are appropriate
  - how to extract proof obligations from a branch
  - how to format prompts for that domain

Built-in domains:
  - PHYSICS_MOND : the original physics-derivation setup (CP1/CP2/MOND coefficient work)
  - GENERAL_MATH : pure mathematical reasoning (proofs, identities, solving)
  - CODE         : code correctness reasoning (preconditions, invariants, tests)
  - ARGUMENT     : general argumentation (claims, evidence, logical structure)

Each domain is just a configuration object. New domains can be added by
instantiating Domain(...) without touching the rest of the pipeline.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Domain:
    """A reasoning domain: keywords, tools, and extraction strategy."""

    name: str
    description: str

    # Lexical signals used by the existing scoring layer
    target_keywords: list[str] = field(default_factory=list)
    derivation_keywords: list[str] = field(default_factory=list)
    assumption_markers: list[str] = field(default_factory=list)
    gap_markers: list[str] = field(default_factory=list)

    # Tool names (must exist in reasoning_tools.TOOL_REGISTRY)
    available_tools: list[str] = field(default_factory=list)

    # Domain-specific guidance for the branch-generation prompt
    branch_generation_guidance: str = ""

    # Domain-specific guidance for the verification prompt
    verification_guidance: str = ""

    # Optional callable: branch_dict -> list[str] of obligations
    custom_obligation_extractor: Optional[Callable] = None


# ============================================================
# PHYSICS_MOND: the existing setup
# ============================================================

PHYSICS_MOND = Domain(
    name="physics_mond",
    description=(
        "Physics derivations involving CP1/CP2 field theory, MOND-like "
        "modifications, halo coefficients, and stability analysis."
    ),
    target_keywords=[
        "sqrt", "square-root", "square root", "mond", "halo",
        "g_halo", "gbar", "g_bar", "a0", "acceleration",
        "coefficient", "closure", "cp2", "cp1",
    ],
    derivation_keywords=[
        "derive", "derived", "derivation", "start from", "insert",
        "compute", "integrate", "project", "obtain", "show",
        "from", "equation", "stress-energy", "poisson", "weak-field",
        "kernel", "source", "boundary", "field equation",
    ],
    assumption_markers=[
        "assume", "assumed", "there exists", "supplies", "maps", "insert",
        "postulate", "closure", "ansatz", "phenomenological", "treated as",
        "taken as", "by hand", "requires",
    ],
    gap_markers=[
        "missing", "gap", "not derived", "unproven", "unjustified",
        "overclaim", "circular", "matched", "imposed",
    ],
    available_tools=[
        "sym_simplify", "sym_check_equality", "sym_solve",
        "sym_substitute", "sym_hessian", "sym_limit",
        "sym_dimensional_check", "numerical_sample",
        "counterexample_search",
    ],
    branch_generation_guidance=(
        "Each branch should attempt a distinct derivation route. "
        "Distinguish constructive branches (genuine derivation attempts) "
        "from diagnostic branches (identifying obstacles) and "
        "phenomenological branches (admitting matching, not derivation)."
    ),
    verification_guidance=(
        "Use sym_dimensional_check on every coefficient relation, "
        "and counterexample_search on every positivity/stability claim. "
        "Treat algebraic matching to a target as NOT a derivation."
    ),
)


# ============================================================
# GENERAL_MATH: pure mathematical reasoning
# ============================================================

GENERAL_MATH = Domain(
    name="general_math",
    description=(
        "General mathematical reasoning: proving identities, solving "
        "equations, evaluating limits, checking inequalities."
    ),
    target_keywords=[
        "prove", "show", "identity", "equation", "inequality", "limit",
        "converges", "diverges", "exists", "unique", "solution",
    ],
    derivation_keywords=[
        "compute", "expand", "factor", "simplify", "substitute",
        "by induction", "by contradiction", "directly", "implies",
        "therefore", "hence", "follows", "WLOG", "without loss",
    ],
    assumption_markers=[
        "assume", "let", "suppose", "given", "if", "consider",
        "WLOG", "by hypothesis",
    ],
    gap_markers=[
        "should be", "obviously", "clearly", "trivially", "left to reader",
        "by inspection", "it is known", "standard result",
    ],
    available_tools=[
        "sym_simplify", "sym_check_equality", "sym_solve",
        "sym_substitute", "sym_limit", "numerical_sample",
        "counterexample_search",
    ],
    branch_generation_guidance=(
        "Each branch should be a distinct proof strategy: direct, "
        "by induction, by contradiction, by construction, etc. "
        "Each step must be a checkable mathematical claim."
    ),
    verification_guidance=(
        "Use sym_check_equality on every claimed equality. "
        "Use counterexample_search on every claimed inequality. "
        "Phrases like 'clearly' or 'obviously' are gap markers — "
        "demand explicit verification."
    ),
)


# ============================================================
# CODE: code correctness reasoning
# ============================================================

CODE = Domain(
    name="code",
    description=(
        "Reasoning about code correctness: preconditions, postconditions, "
        "loop invariants, edge cases, and test-based validation."
    ),
    target_keywords=[
        "correct", "correctness", "returns", "produces", "computes",
        "invariant", "precondition", "postcondition", "input", "output",
        "edge case", "boundary",
    ],
    derivation_keywords=[
        "iterate", "recurse", "by induction on", "trace", "step",
        "execute", "evaluate", "consider input", "for all",
    ],
    assumption_markers=[
        "assume input", "assume valid", "given that", "if input",
        "assuming no errors", "assuming",
    ],
    gap_markers=[
        "should work", "probably", "I think", "untested",
        "no edge cases considered", "happy path only",
    ],
    available_tools=[
        "code_execute", "code_test_cases", "sym_check_equality",
        "counterexample_search",
    ],
    branch_generation_guidance=(
        "Each branch should test a distinct execution path or input "
        "regime. Cover: typical input, empty/null input, boundary values, "
        "type-mismatched input, large input."
    ),
    verification_guidance=(
        "Run the actual code with code_execute on each input regime. "
        "Do not approve any branch that has not exercised edge cases."
    ),
)


# ============================================================
# ARGUMENT: general argumentation
# ============================================================

ARGUMENT = Domain(
    name="argument",
    description=(
        "General argumentation and claim-evidence reasoning. Useful for "
        "evaluating non-mathematical positions, weighing evidence, and "
        "identifying logical structure in prose."
    ),
    target_keywords=[
        "claim", "thesis", "conclusion", "position", "argues", "supports",
        "evidence", "implies", "follows from",
    ],
    derivation_keywords=[
        "because", "since", "given", "evidence shows", "data indicate",
        "study found", "analysis shows", "follows", "implies",
    ],
    assumption_markers=[
        "assume", "presuppose", "take for granted", "given that",
        "we accept", "common knowledge",
    ],
    gap_markers=[
        "everyone knows", "obviously", "common sense", "no evidence",
        "unsupported", "anecdotal", "non sequitur", "circular",
    ],
    available_tools=[
        "consistency_check", "claim_extract",
    ],
    branch_generation_guidance=(
        "Each branch should be a distinct interpretation or a distinct "
        "supporting line of argument. Steel-man branches and weak branches "
        "should both be represented."
    ),
    verification_guidance=(
        "Identify each load-bearing claim. Mark which are supported by "
        "evidence in the text vs which rely on unstated assumptions. "
        "Flag any rhetorical moves that disguise gaps."
    ),
)


# ============================================================
# Registry and lookup
# ============================================================

DOMAINS: dict[str, Domain] = {
    "physics_mond": PHYSICS_MOND,
    "general_math": GENERAL_MATH,
    "code": CODE,
    "argument": ARGUMENT,
}


def get_domain(name: str) -> Domain:
    if name not in DOMAINS:
        raise KeyError(
            f"Unknown domain {name!r}. Available: {list(DOMAINS.keys())}"
        )
    return DOMAINS[name]


def register_domain(domain: Domain) -> None:
    """Register a new domain. Useful for project-specific configurations."""
    DOMAINS[domain.name] = domain
