"""
router.py
=========

Auto-routing: reads any question and selects the right domain,
tools, and parameter ranges automatically.

This is what makes the system genuinely multipurpose. Instead of
the user needing to know which domain to configure, the router
detects the question type and sets everything up accordingly.

For other users who don't know the internals, this is the layer
that makes it feel like it just works.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RouteDecision:
    domain_name: str
    confidence: float           # 0-1, how sure the router is
    parameter_ranges: dict
    reasoning: str              # why this route was chosen
    suggested_n_branches: int = 3
    web_search_likely_useful: bool = False


# ── Keyword signals per domain ─────────────────────────────────────────────

ROUTE_SIGNALS = {
    "physics_mond": {
        "strong": [
            "mond", "halo", "dark matter", "gravitational", "field theory",
            "lagrangian", "cp1", "cp2", "rho_eff", "g_halo", "a0",
            "coefficient", "closure", "winding", "angular source",
            "effective potential", "mass distribution", "rotation curve",
        ],
        "weak": [
            "derive", "derivation", "field equation", "coupling",
            "sector", "symmetry breaking", "eigenvalue", "spectral",
        ],
    },
    "general_math": {
        "strong": [
            "prove", "proof", "theorem", "lemma", "identity", "converge",
            "diverge", "integral", "derivative", "series", "inequality",
            "equation", "solve for", "simplify", "evaluate",
        ],
        "weak": [
            "show that", "find", "compute", "calculate", "determine",
            "matrix", "vector", "function", "polynomial",
        ],
    },
    "code": {
        "strong": [
            "code", "function", "bug", "debug", "implement", "algorithm",
            "python", "javascript", "class", "method", "returns", "error",
            "exception", "test", "assert", "runtime",
        ],
        "weak": [
            "loop", "if", "else", "variable", "output", "input",
            "compile", "script", "module", "import",
        ],
    },
    "argument": {
        "strong": [
            "argue", "argument", "claim", "evidence", "reason",
            "position", "thesis", "disagree", "critique", "analyse",
            "evaluate", "assess", "compare", "contrast", "policy",
            "should", "ought", "better", "worse", "ethical",
        ],
        "weak": [
            "think", "believe", "opinion", "view", "consider",
            "explain why", "what do you", "is it true",
        ],
    },
}

# ── Default parameter ranges per domain ───────────────────────────────────

DEFAULT_RANGES = {
    "physics_mond": {
        "mT":    [-2.0, 2.0],
        "lam12": [ 0.0, 1.0],
        "x1":    [-1.0, 1.0],
        "y1":    [-1.0, 1.0],
    },
    "general_math": {
        "x":  [-10.0, 10.0],
        "y":  [-10.0, 10.0],
        "n":  [  1.0, 20.0],
        "a":  [ -5.0,  5.0],
    },
    "code": {},          # code domain uses execution, not numerical sampling
    "argument": {},      # argument domain uses claim extraction, not numerics
}


def route_question(question: str,
                   llm_chat_fn=None,
                   verbose: bool = False) -> RouteDecision:
    """
    Determine the right domain and configuration for a question.

    Uses keyword scoring as the primary signal, with an optional
    LLM confirmation pass for borderline cases.
    """
    q_lower = question.lower()

    # Score each domain
    scores = {}
    for domain, signals in ROUTE_SIGNALS.items():
        strong_hits = sum(1 for w in signals["strong"] if w in q_lower)
        weak_hits   = sum(1 for w in signals["weak"]   if w in q_lower)
        scores[domain] = strong_hits * 2 + weak_hits

    best_domain = max(scores, key=scores.get)
    best_score  = scores[best_domain]
    total_score = sum(scores.values())

    # Confidence: how dominant is the top domain?
    if total_score == 0:
        confidence = 0.3          # no signals — low confidence
        best_domain = "argument"  # default to general reasoning
    else:
        confidence = min(0.95, best_score / max(1, total_score))

    # For borderline cases, ask the LLM
    if confidence < 0.5 and llm_chat_fn is not None:
        best_domain, confidence = _llm_route(question, scores,
                                             best_domain, llm_chat_fn)

    # Decide whether web search would help
    web_useful = _needs_web_search(question)

    # Scale branches: complex questions benefit from more diversity
    word_count = len(question.split())
    n_branches = 4 if word_count > 30 else 3

    reasoning = (
        f"Domain signals: {dict(sorted(scores.items(), key=lambda x: -x[1]))}. "
        f"Selected '{best_domain}' with confidence {confidence:.0%}."
    )

    if verbose:
        print(f"[router] {reasoning}")
        if web_useful:
            print("[router] Web search likely useful for this question.")

    return RouteDecision(
        domain_name=best_domain,
        confidence=confidence,
        parameter_ranges=DEFAULT_RANGES.get(best_domain, {}),
        reasoning=reasoning,
        suggested_n_branches=n_branches,
        web_search_likely_useful=web_useful,
    )


def _llm_route(question: str, scores: dict,
               fallback: str, llm_chat_fn) -> tuple:
    """Use the LLM to confirm routing for borderline questions."""
    system = (
        "You classify questions into one of four domains. "
        "Reply with only the domain name, nothing else."
    )
    prompt = (
        f"Question: {question}\n\n"
        "Which domain best fits?\n"
        "- physics_mond  (physics derivations, field theory, MOND)\n"
        "- general_math  (proofs, equations, mathematical reasoning)\n"
        "- code          (programming, debugging, algorithms)\n"
        "- argument      (claims, evidence, ethical reasoning, opinions)\n\n"
        "Reply with exactly one domain name."
    )
    try:
        response = llm_chat_fn(system, prompt).strip().lower()
        for domain in ["physics_mond", "general_math", "code", "argument"]:
            if domain in response:
                return domain, 0.65
    except Exception:
        pass
    return fallback, 0.4


def _needs_web_search(question: str) -> bool:
    """Simple heuristic: does this question likely need current information?"""
    web_signals = [
        "recent", "latest", "current", "news", "published", "paper",
        "study", "research shows", "according to", "literature",
        "who is", "what is the status", "has anyone", "does anyone",
    ]
    q = question.lower()
    return any(w in q for w in web_signals)
