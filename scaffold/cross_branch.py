"""
cross_branch.py
===============

Cross-branch obligation passing.

The flow:
  1. After branches are generated, extract each branch's obligations
     (its assumptions, its gaps, its load-bearing unjustified steps).
  2. Pool them into a global obligation list.
  3. For each branch, re-prompt the LLM with the obligations raised by OTHER
     branches and ask which it can address, weaken, or which it concedes.
  4. Update each branch's score based on whether its existence helps
     discharge other branches' obligations.

This makes branches collaborative rather than independent, and surfaces the
real frontier of the problem: the obligations no branch can address.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

from domains import Domain


@dataclass
class Obligation:
    text: str
    source_branch: str
    kind: str  # "assumption" | "gap" | "claim_to_verify"
    priority: float = 1.0
    discharged_by: list[str] = field(default_factory=list)
    weakened_by: list[str] = field(default_factory=list)
    status: str = "open"  # "open" | "discharged" | "weakened" | "global_gap"

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Obligation extraction
# ============================================================

def extract_obligations(branch: dict, domain: Domain) -> list[Obligation]:
    """
    Extract obligations from a branch using domain-aware heuristics.
    Branches are dicts with keys: name, assumptions, steps, candidate_result, notes.
    """
    if domain.custom_obligation_extractor is not None:
        return domain.custom_obligation_extractor(branch)

    name = branch.get("name", "unknown")
    obligations: list[Obligation] = []

    # Assumptions are obligations by default.
    for a in branch.get("assumptions", []) or []:
        if not a or not a.strip():
            continue
        obligations.append(Obligation(
            text=a.strip(),
            source_branch=name,
            kind="assumption",
            priority=1.0,
        ))

    # Look for gap markers in steps and notes.
    text_blocks = list(branch.get("steps", []) or []) + \
                  list(branch.get("notes", []) or [])
    for block in text_blocks:
        if not block:
            continue
        for marker in domain.gap_markers:
            if re.search(rf"\b{re.escape(marker)}\b", block, re.IGNORECASE):
                obligations.append(Obligation(
                    text=block.strip(),
                    source_branch=name,
                    kind="gap",
                    priority=1.5,  # gaps weighted higher than assumptions
                ))
                break  # one gap-tag per block is enough

    # Look for assumption markers within steps (assumptions hidden inside steps).
    for step in branch.get("steps", []) or []:
        if not step:
            continue
        for marker in domain.assumption_markers:
            if re.search(rf"\b{re.escape(marker)}\b", step, re.IGNORECASE):
                obligations.append(Obligation(
                    text=step.strip(),
                    source_branch=name,
                    kind="assumption",
                    priority=0.8,  # in-step assumptions slightly lower
                ))
                break

    return _dedupe(obligations)


def _dedupe(obs: list[Obligation]) -> list[Obligation]:
    seen = set()
    out = []
    for o in obs:
        key = (o.text.lower(), o.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def pool_obligations(branches: list[dict], domain: Domain) -> list[Obligation]:
    """Extract and pool obligations from all branches."""
    pooled = []
    for b in branches:
        pooled.extend(extract_obligations(b, domain))
    return pooled


# ============================================================
# Cross-branch resolution prompt
# ============================================================

def _format_obligations_for_branch(target_branch: dict,
                                   all_obligations: list[Obligation]
                                   ) -> str:
    """Format obligations from OTHER branches for the target branch's review."""
    target_name = target_branch.get("name")
    others = [o for o in all_obligations if o.source_branch != target_name]
    if not others:
        return "(no obligations from other branches)"

    lines = []
    for i, o in enumerate(others):
        lines.append(
            f"  [{i}] (from '{o.source_branch}', kind={o.kind}, "
            f"priority={o.priority}): {o.text}"
        )
    return "\n".join(lines)


def _branch_summary(branch: dict) -> str:
    return (
        f"Branch: {branch.get('name')}\n"
        f"Method: {branch.get('method')}\n"
        f"Assumptions: {branch.get('assumptions', [])}\n"
        f"Steps:\n  - " + "\n  - ".join(branch.get('steps', []) or []) +
        f"\nResult: {branch.get('candidate_result')}"
    )


def cross_branch_resolve(
    branches: list[dict],
    domain: Domain,
    llm_chat_fn: Callable,
) -> dict:
    """
    Run the cross-branch resolution pass.

    Returns:
      {
        'obligations': [Obligation, ...],
        'branch_responses': {branch_name: response_text},
        'discharge_matrix': {obligation_index: [branch_names_that_claim_to_address_it]},
        'global_gaps': [Obligation, ...],
      }
    """
    pooled = pool_obligations(branches, domain)

    if not pooled:
        return {
            "obligations": [],
            "branch_responses": {},
            "discharge_matrix": {},
            "global_gaps": [],
        }

    system_prompt = (
        f"You are resolving cross-branch obligations in domain '{domain.name}'. "
        f"{domain.verification_guidance}\n\n"
        "You will be given one branch and a list of obligations from OTHER "
        "branches. For each obligation, decide whether the given branch:\n"
        "  - DISCHARGES it (provides what is needed to justify it),\n"
        "  - WEAKENS it (makes it less load-bearing or replaces it),\n"
        "  - CONCEDES it (cannot address; it remains a gap),\n"
        "  - IRRELEVANT (does not apply to this branch's approach).\n\n"
        "Output strict JSON in this exact shape:\n"
        "{\n"
        '  "responses": [\n'
        '    {"obligation_index": 0, "verdict": "discharges|weakens|concedes|irrelevant", "reason": "..."},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
    )

    branch_responses = {}
    discharge_matrix: dict[int, list[str]] = {}

    for branch in branches:
        target_name = branch.get("name")
        obligations_text = _format_obligations_for_branch(branch, pooled)
        # Also build a mapping from local index to global index.
        others_indices = [
            i for i, o in enumerate(pooled) if o.source_branch != target_name
        ]
        local_to_global = {
            local: global_idx
            for local, global_idx in enumerate(others_indices)
        }

        user_prompt = (
            f"Branch under review:\n{_branch_summary(branch)}\n\n"
            f"Obligations from other branches:\n{obligations_text}\n\n"
            "Return your JSON response now."
        )

        try:
            response_text = llm_chat_fn(system_prompt, user_prompt)
        except Exception as e:
            response_text = json.dumps({"error": str(e), "responses": []})

        branch_responses[target_name] = response_text

        # Parse responses
        parsed = _parse_resolution_response(response_text)
        for r in parsed.get("responses", []):
            local_idx = r.get("obligation_index")
            verdict = (r.get("verdict") or "").lower()
            if not isinstance(local_idx, int):
                continue
            global_idx = local_to_global.get(local_idx)
            if global_idx is None:
                continue
            obligation = pooled[global_idx]
            if verdict == "discharges":
                obligation.discharged_by.append(target_name)
                obligation.status = "discharged"
            elif verdict == "weakens":
                obligation.weakened_by.append(target_name)
                if obligation.status == "open":
                    obligation.status = "weakened"
            discharge_matrix.setdefault(global_idx, [])
            if verdict in ("discharges", "weakens"):
                discharge_matrix[global_idx].append(target_name)

    # Surface obligations no branch addressed.
    global_gaps = []
    for i, o in enumerate(pooled):
        if o.status == "open":
            o.status = "global_gap"
            global_gaps.append(o)

    return {
        "obligations": [o.to_dict() for o in pooled],
        "branch_responses": branch_responses,
        "discharge_matrix": discharge_matrix,
        "global_gaps": [o.to_dict() for o in global_gaps],
    }


def _parse_resolution_response(text: str) -> dict:
    """Try to extract JSON from an LLM response, robust to surrounding prose."""
    # First try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON block within the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"responses": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"responses": []}


# ============================================================
# Score adjustment
# ============================================================

def cross_branch_score_adjustments(resolution: dict
                                   ) -> dict[str, float]:
    """
    Compute a score adjustment per branch based on its cross-branch role.

    Branches that discharge other branches' obligations get a positive bonus.
    Branches whose obligations remain global gaps get a small penalty
    (if their core contribution depended on those obligations).
    """
    adjustments: dict[str, float] = {}

    discharge_matrix = resolution.get("discharge_matrix", {})
    obligations = resolution.get("obligations", [])

    # Reward branches that discharge or weaken obligations.
    for idx_str, branches in discharge_matrix.items():
        for b in branches:
            adjustments[b] = adjustments.get(b, 0.0) + 0.15

    # Penalise branches whose obligations remained global gaps.
    for o in obligations:
        if o["status"] == "global_gap":
            src = o["source_branch"]
            adjustments[src] = adjustments.get(src, 0.0) - 0.1 * o["priority"]

    # Clamp.
    return {k: max(-1.0, min(1.0, v)) for k, v in adjustments.items()}
