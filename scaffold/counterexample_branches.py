"""
counterexample_branches.py
==========================

Counterexample-driven branch generation.

Standard branches are LLM-generated reasoning attempts. Counterexample-driven
branches are different: they're generated *in response to* specific failures
detected by counterexample_search and other verification tools.

Two kinds of counterexample-driven branches:

  1. CHALLENGER branches — when a verification tool finds a counterexample
     to a claim in an existing branch, a challenger is generated that
     explicitly uses the counterexample to falsify the claim. Pushes the
     dialectic forward by forcing the system to either repair the claim or
     concede.

  2. DEFENDER branches — when a counterexample is found, an alternative
     defender branch may be generated that constrains the parameter space
     to where the claim does hold, identifies the missing precondition,
     and proposes a corrected version of the claim.

Together these implement adversarial branch generation: the system actively
tries to break its own claims and then either repairs or accepts the break.
"""

import json
import re
from typing import Callable, Optional

from reasoning_tools import execute_tool_call, parse_tool_calls
from domains import Domain


# ============================================================
# Identifying weak claims in branches
# ============================================================

def identify_weak_claims(branch: dict) -> list[dict]:
    """
    Heuristically extract claims from a branch that are candidates for
    counterexample testing. Looks for steps containing inequalities,
    equalities, or quantitative assertions.

    Returns a list of dicts: {step_index, claim_text, claim_type,
    extracted_expression}
    """
    weak_claims = []
    steps = branch.get("steps", []) or []

    # An expression we consider testable is mostly mathematical: variables,
    # operators, parens, numbers. We reject strings containing English verbs
    # or articles that signal it's prose, not a formula.
    MATH_TOKEN = r"[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*|[\+\-\*/\^\(\)]"
    MATH_EXPR = rf"(?:{MATH_TOKEN})(?:\s*(?:{MATH_TOKEN}))*"

    # Words that indicate the surrounding text is prose, not a pure expression.
    PROSE_WORDS = {
        "ensures", "implies", "means", "shows", "the", "this", "that", "is",
        "are", "compute", "let", "set", "define", "consider", "therefore",
        "hence", "thus", "so", "where", "with", "for", "given", "such",
        "ensure", "follows", "result", "of",
    }

    def is_pure_math(s: str) -> bool:
        """Reject expressions that contain prose-indicator words."""
        s_clean = re.sub(r"[\d\.\+\-\*/\^\(\)\s]", " ", s.lower())
        tokens = s_clean.split()
        # Allow any number of pure math tokens; reject if any prose word.
        return not any(t in PROSE_WORDS for t in tokens)

    # Tighter inequality pattern: math, comparator, math.
    inequality_pattern = re.compile(
        r"([A-Za-z_0-9\*\+\-/\(\)\^\s\.]{1,80}?)\s*(>=|<=|>|<)\s*"
        r"([A-Za-z_0-9\*\+\-/\(\)\^\s\.]{1,40}?)(?=\s|[,.;]|$)"
    )
    equality_pattern = re.compile(
        r"\b([A-Za-z_][A-Za-z_0-9]*(?:\([A-Za-z_0-9,\s]*\))?)"
        r"\s*=\s*"
        r"([A-Za-z_0-9\*\+\-/\(\)\^\s\.]{2,80}?)(?=\s*$|\s*[,;]|\s+\b(?:where|with|for|when|and|but)\b)"
    )

    positivity_pattern = re.compile(
        r"\b(positive|stable|bounded below|non-negative|grows|"
        r"converges|finite)\b",
        re.IGNORECASE,
    )

    for i, step in enumerate(steps):
        if not step:
            continue

        # Inequalities
        for m in inequality_pattern.finditer(step):
            lhs = m.group(1).strip()
            op = m.group(2)
            rhs = m.group(3).strip()
            if not is_pure_math(lhs) or not is_pure_math(rhs):
                continue
            if not lhs or not rhs:
                continue
            weak_claims.append({
                "step_index": i,
                "claim_text": step,
                "claim_type": "inequality",
                "lhs": lhs,
                "op": op,
                "rhs": rhs,
                "as_expression": f"({lhs}) - ({rhs})",
            })

        # Equalities — only the form "name = expression"
        for m in equality_pattern.finditer(step):
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()
            if not is_pure_math(rhs):
                continue
            # LHS must be a simple identifier or function, not a phrase.
            if not re.match(r"^[A-Za-z_][A-Za-z_0-9]*(\([A-Za-z_0-9,\s]*\))?$",
                            lhs):
                continue
            weak_claims.append({
                "step_index": i,
                "claim_text": step,
                "claim_type": "equality",
                "lhs": lhs,
                "rhs": rhs,
                "as_expression": f"({lhs}) - ({rhs})",
            })

        if positivity_pattern.search(step):
            weak_claims.append({
                "step_index": i,
                "claim_text": step,
                "claim_type": "positivity_assertion",
                "as_expression": None,
            })

    return weak_claims


# ============================================================
# Running tool-based falsification on a claim
# ============================================================

def attempt_falsification(
    claim: dict,
    parameter_ranges: dict,
    n_samples: int = 300,
) -> dict:
    """
    Try to falsify a claim using counterexample_search.
    Returns the tool result plus a verdict.
    """
    expr = claim.get("as_expression")
    if expr is None:
        return {
            "ok": False,
            "verdict": "not_testable",
            "reason": "Claim type does not support automated falsification",
            "claim": claim,
        }

    # For an inequality lhs OP rhs, the expression (lhs - rhs) should:
    #   if OP is >=  : be >= 0 always  -> search for negative
    #   if OP is <=  : be <= 0 always  -> negate and search
    #   if OP is =   : be 0 always     -> search for nonzero in either direction
    op = claim.get("op")
    if op == "<=":
        expr = f"-({expr})"  # flip so claim becomes "expr >= 0"

    if claim.get("claim_type") == "equality":
        # For equality, run two-sided counterexample searches.
        result_pos = execute_tool_call({
            "name": "counterexample_search",
            "arguments": {
                "expression": expr,
                "ranges": parameter_ranges,
                "n_samples": n_samples,
            },
        })
        result_neg = execute_tool_call({
            "name": "counterexample_search",
            "arguments": {
                "expression": f"-({expr})",
                "ranges": parameter_ranges,
                "n_samples": n_samples,
            },
        })
        # If either returns a counterexample, the equality is broken.
        broke_positive = (result_pos.get("ok")
                          and result_pos.get("claim_holds") is False)
        broke_negative = (result_neg.get("ok")
                          and result_neg.get("claim_holds") is False)
        broken = broke_positive or broke_negative
        return {
            "ok": True,
            "verdict": "falsified" if broken else "robust",
            "claim": claim,
            "counterexample": (
                result_pos.get("counterexample") if broke_positive
                else result_neg.get("counterexample") if broke_negative
                else None
            ),
            "details": {"positive_search": result_pos,
                        "negative_search": result_neg},
        }

    # Inequality case
    result = execute_tool_call({
        "name": "counterexample_search",
        "arguments": {
            "expression": expr,
            "ranges": parameter_ranges,
            "n_samples": n_samples,
        },
    })

    if not result.get("ok"):
        return {"ok": False, "verdict": "tool_error",
                "claim": claim, "details": result}

    if result.get("claim_holds") is False:
        return {
            "ok": True,
            "verdict": "falsified",
            "claim": claim,
            "counterexample": result.get("counterexample"),
            "value_at_counterexample": result.get("value_at_counterexample"),
        }
    return {
        "ok": True,
        "verdict": "robust",
        "claim": claim,
        "minimum_at": result.get("minimum_at"),
        "minimum_value": result.get("minimum_found"),
    }


# ============================================================
# LLM-driven challenger / defender generation
# ============================================================

CHALLENGER_PROMPT = (
    "You are generating a CHALLENGER branch — a reasoning attempt designed "
    "specifically to FALSIFY a claim from another branch using a known "
    "counterexample.\n\n"
    "Original branch: {original_branch}\n"
    "Specific claim being challenged: {claim_text}\n"
    "Counterexample found at parameters: {counterexample}\n"
    "Value at counterexample: {value}\n\n"
    "Generate a new branch (in JSON, same schema as original branches) "
    "that:\n"
    "  - Uses these specific parameter values\n"
    "  - Walks through what the original branch's reasoning produces at "
    "these values\n"
    "  - Concludes whether the original claim holds or fails there\n"
    "  - Suggests what assumption the original branch was implicitly "
    "making\n\n"
    'Return JSON exactly: {{"name": "...", "method": "Challenger via '
    'counterexample", "assumptions": [...], "steps": [...], '
    '"candidate_result": "..."}}'
)

DEFENDER_PROMPT = (
    "You are generating a DEFENDER branch — a reasoning attempt that "
    "RESPONDS to a counterexample by either restricting the claim's domain "
    "of validity or repairing the claim.\n\n"
    "Original branch: {original_branch}\n"
    "Claim under attack: {claim_text}\n"
    "Counterexample at: {counterexample}\n\n"
    "Generate a new branch (JSON same schema) that:\n"
    "  - Identifies the unstated precondition the original claim required\n"
    "  - States the corrected, restricted claim that DOES hold\n"
    "  - Verifies (in steps) why the corrected claim avoids the "
    "counterexample\n\n"
    'Return JSON exactly: {{"name": "...", "method": "Defender via domain '
    'restriction", "assumptions": [...], "steps": [...], '
    '"candidate_result": "..."}}'
)


def _parse_branch_json(response: str) -> Optional[dict]:
    """Extract a branch dict from LLM response, robust to surrounding prose."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_challenger_branch(
    original_branch: dict,
    falsification_result: dict,
    llm_chat_fn: Callable,
) -> Optional[dict]:
    """Generate a challenger branch from a falsification result."""
    if falsification_result.get("verdict") != "falsified":
        return None

    claim = falsification_result["claim"]
    counterexample = falsification_result.get("counterexample", {})
    value = falsification_result.get("value_at_counterexample", "?")

    prompt = CHALLENGER_PROMPT.format(
        original_branch=json.dumps({
            "name": original_branch.get("name"),
            "steps": original_branch.get("steps"),
            "result": original_branch.get("candidate_result"),
        }, indent=2),
        claim_text=claim.get("claim_text", ""),
        counterexample=counterexample,
        value=value,
    )

    response = llm_chat_fn(
        "You generate adversarial reasoning branches. Return only valid JSON.",
        prompt,
    )
    branch = _parse_branch_json(response)
    if branch is None:
        return None

    # Tag the branch
    branch["generation_kind"] = "challenger"
    branch["challenges_branch"] = original_branch.get("name")
    branch["counterexample_used"] = counterexample
    return branch


def generate_defender_branch(
    original_branch: dict,
    falsification_result: dict,
    llm_chat_fn: Callable,
) -> Optional[dict]:
    """Generate a defender branch from a falsification result."""
    if falsification_result.get("verdict") != "falsified":
        return None

    claim = falsification_result["claim"]
    counterexample = falsification_result.get("counterexample", {})

    prompt = DEFENDER_PROMPT.format(
        original_branch=json.dumps({
            "name": original_branch.get("name"),
            "steps": original_branch.get("steps"),
            "result": original_branch.get("candidate_result"),
        }, indent=2),
        claim_text=claim.get("claim_text", ""),
        counterexample=counterexample,
    )

    response = llm_chat_fn(
        "You generate corrected/restricted reasoning branches. Return only "
        "valid JSON.",
        prompt,
    )
    branch = _parse_branch_json(response)
    if branch is None:
        return None

    branch["generation_kind"] = "defender"
    branch["defends_branch"] = original_branch.get("name")
    branch["counterexample_addressed"] = counterexample
    return branch


# ============================================================
# Top-level orchestration
# ============================================================

def expand_branches_via_counterexamples(
    branches: list[dict],
    parameter_ranges: dict,
    llm_chat_fn: Callable,
    domain: Domain,
    generate_challengers: bool = True,
    generate_defenders: bool = True,
    max_new_branches_per_original: int = 2,
) -> dict:
    """
    Run the full counterexample-driven expansion:
      1. For each branch, identify weak claims
      2. Try to falsify each claim via counterexample_search
      3. For each falsification, optionally generate challenger and/or
         defender branches via the LLM
      4. Return augmented branch list and a falsification report

    Returns:
      {
        'original_branches': [...],
        'new_branches': [...],
        'falsification_reports': [...],
      }
    """
    new_branches = []
    reports = []

    for branch in branches:
        weak = identify_weak_claims(branch)
        if not weak:
            continue

        n_added = 0
        for claim in weak:
            if n_added >= max_new_branches_per_original:
                break

            falsification = attempt_falsification(
                claim, parameter_ranges, n_samples=200,
            )
            reports.append({
                "branch": branch.get("name"),
                "claim": claim.get("claim_text", ""),
                "verdict": falsification.get("verdict"),
                "counterexample": falsification.get("counterexample"),
            })

            if falsification.get("verdict") != "falsified":
                continue

            if generate_challengers:
                challenger = generate_challenger_branch(
                    branch, falsification, llm_chat_fn,
                )
                if challenger:
                    new_branches.append(challenger)
                    n_added += 1
                    if n_added >= max_new_branches_per_original:
                        break

            if generate_defenders:
                defender = generate_defender_branch(
                    branch, falsification, llm_chat_fn,
                )
                if defender:
                    new_branches.append(defender)
                    n_added += 1

    return {
        "original_branches": branches,
        "new_branches": new_branches,
        "falsification_reports": reports,
        "all_branches": branches + new_branches,
    }
