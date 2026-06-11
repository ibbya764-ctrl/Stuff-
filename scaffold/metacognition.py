"""
metacognition.py
================

Adversarial self-audit grounded in objective signals.

Replaces the prompt-based adaptive_self_audit. Three structural improvements:

  1. Challenger questions: questions about the winning branch are generated
     by the LOSING branches, not by the winner examining itself. This gives
     genuine adversarial pressure rather than self-reflection that shares
     the winner's blind spots.

  2. Objective confidence: confidence levels are derived from verification
     reports, counterexample search results, and obligation store state —
     not self-reported by the LLM via regex matching on "LOW confidence".

  3. Loop closure: questions that survive critique are fed back into the
     obligation store as new obligations and (optionally) into a follow-up
     cross-branch resolution pass, so the audit becomes part of the
     reasoning rather than a postscript.

Drop into the same directory as pipeline.py. Call after pipeline.run().
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# ============================================================
# Data shapes
# ============================================================

@dataclass
class AuditFinding:
    """One finding from the metacognitive audit."""

    question: str
    targets: str                 # which step/assumption it probes
    challenger: str              # which losing branch raised it
    answer: str                  # the winner's answer
    objective_confidence: str    # "LOW" | "MEDIUM" | "HIGH"
    confidence_reason: str       # why that confidence was assigned
    became_obligation: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "targets": self.targets,
            "challenger": self.challenger,
            "answer": self.answer,
            "objective_confidence": self.objective_confidence,
            "confidence_reason": self.confidence_reason,
            "became_obligation": self.became_obligation,
        }


# ============================================================
# Step 1: Challenger-based question generation
# ============================================================

CHALLENGER_SYSTEM = (
    "You are a competing reasoning branch that LOST to another approach. "
    "Your method was different and you have specific reasons to suspect "
    "the winning branch's reasoning may have weaknesses your approach would "
    "have avoided. Generate questions that expose those specific weaknesses.\n\n"
    "Constraints:\n"
    "  - Questions must target SPECIFIC steps or assumptions, not the conclusion.\n"
    "  - Questions must be impossible to deflect with confident restatement.\n"
    "  - Questions must come from YOUR method's perspective, not generic skepticism.\n"
    "  - If your method actually had no advantage on a particular point, say so.\n"
    "Return JSON only."
)


def _format_branch_for_challenger(branch: dict) -> str:
    return (
        f"Name: {branch.get('name', 'unknown')}\n"
        f"Method: {branch.get('method', '')}\n"
        f"Assumptions: {branch.get('assumptions', [])}\n"
        f"Steps:\n  - " + "\n  - ".join(branch.get('steps', []) or []) +
        f"\nResult: {branch.get('candidate_result', '')}"
    )


def generate_challenger_questions(
    winner: dict,
    losers: list[dict],
    llm_chat_fn: Callable,
    max_per_loser: int = 2,
    verbose: bool = True,
) -> list[dict]:
    """
    Each losing branch generates questions about the winner from its own
    methodological perspective.

    Returns a list of dicts:
      {"question": str, "targets": str, "challenger": str}
    """
    all_questions: list[dict] = []

    if not losers:
        if verbose:
            print("  [metacog] No losing branches available; skipping challenger phase.")
        return all_questions

    for loser in losers:
        challenger_name = loser.get("name", "unknown_challenger")
        user_prompt = f"""
You are this losing branch:
{_format_branch_for_challenger(loser)}

You lost to this winning branch:
{_format_branch_for_challenger(winner)}

Generate {max_per_loser} questions that expose specific weaknesses in the
winning branch from YOUR method's perspective. For each question, name the
step or assumption it targets.

Return JSON in this exact form:
{{
  "questions": [
    {{"question": "...", "targets": "..."}},
    ...
  ]
}}
""".strip()

        try:
            response = llm_chat_fn(CHALLENGER_SYSTEM, user_prompt)
            parsed = _extract_json(response)
            for q in parsed.get("questions", [])[:max_per_loser]:
                if "question" in q:
                    all_questions.append({
                        "question":   q["question"],
                        "targets":    q.get("targets", ""),
                        "challenger": challenger_name,
                    })
        except Exception as e:
            if verbose:
                print(f"  [metacog] Challenger '{challenger_name}' failed: {e}")
            continue

    if verbose:
        print(f"  [metacog] {len(all_questions)} challenger questions raised "
              f"by {len(losers)} losing branches.")
    return all_questions


# ============================================================
# Step 2: Question critique (still LLM but with explicit guard rails)
# ============================================================

CRITIQUE_SYSTEM = (
    "You evaluate whether each question is a strong probe. A strong question "
    "cannot be answered defensively without revealing real information. A weak "
    "question can be answered by restating the method confidently. A circular "
    "question assumes its own answer.\n\n"
    "For weak/circular questions, propose a stronger version if one exists. "
    "Return JSON only."
)


def critique_questions(
    questions: list[dict],
    winner: dict,
    llm_chat_fn: Callable,
    verbose: bool = True,
) -> list[dict]:
    """
    Filter and strengthen questions. Returns the surviving question dicts
    (with 'question', 'targets', 'challenger' preserved).
    """
    if not questions:
        return []

    numbered = "\n".join(
        f"{i+1}. {q['question']} [targets: {q.get('targets', '')}, "
        f"from: {q.get('challenger', '?')}]"
        for i, q in enumerate(questions)
    )

    user_prompt = f"""
Winning branch under audit:
Method: {winner.get('method', '')}
Result: {winner.get('candidate_result', '')}

Questions to evaluate:
{numbered}

For each question, return JSON:
{{
  "evaluations": [
    {{
      "index": 1,
      "verdict": "strong" | "weak" | "circular",
      "reason": "...",
      "stronger_version": "..." or null
    }},
    ...
  ]
}}
""".strip()

    try:
        response = llm_chat_fn(CRITIQUE_SYSTEM, user_prompt)
        parsed = _extract_json(response)
        evaluations = parsed.get("evaluations", [])
    except Exception as e:
        if verbose:
            print(f"  [metacog] Critique failed: {e}; keeping all questions.")
        return questions

    survivors: list[dict] = []
    for evaluation in evaluations:
        idx = evaluation.get("index")
        if not isinstance(idx, int) or not (1 <= idx <= len(questions)):
            continue
        original = questions[idx - 1]
        verdict = (evaluation.get("verdict") or "").lower()
        if verdict == "strong":
            survivors.append(original)
        elif evaluation.get("stronger_version"):
            survivors.append({
                "question":   evaluation["stronger_version"],
                "targets":    original.get("targets", ""),
                "challenger": original.get("challenger", ""),
            })

    if verbose:
        print(f"  [metacog] {len(survivors)}/{len(questions)} questions "
              f"survived critique.")
    return survivors


# ============================================================
# Step 3: Objective confidence — derived from real signals
# ============================================================

def derive_objective_confidence(
    branch: dict,
    obligation_store=None,
) -> tuple[str, str]:
    """
    Derive a confidence label from objective signals about the branch.

    Returns (label, reason) where label in {"LOW", "MEDIUM", "HIGH"}.

    Signals used (in priority order):
      1. Verification verdict (if branch has a verification_report)
      2. Counterexample failures recorded on the branch
      3. Total score relative to typical thresholds
      4. Whether branch's assumptions match persistent unresolved gaps
    """
    reasons: list[str] = []
    label = "MEDIUM"  # default if no strong signal

    # ---- 1. Verification verdict ----
    ver = branch.get("verification_report", {})
    verdict = ver.get("verdict", {}) if isinstance(ver, dict) else {}
    if verdict.get("verified") is False:
        return ("LOW",
                f"verification verdict was 'verified=False' "
                f"(note: {verdict.get('note', 'no detail')})")
    if verdict.get("verified") is True:
        label = "HIGH"
        reasons.append("verification verdict confirmed")

    # ---- 2. Counterexample failures ----
    falsifications = branch.get("falsifications", []) or []
    n_falsified = len(falsifications)
    if n_falsified > 0:
        # If anything was falsified, we override to LOW regardless.
        return ("LOW",
                f"{n_falsified} counterexample failure(s) on this branch's claims")

    # ---- 3. Score-based check ----
    score = branch.get("total_score", 0.0)
    if score < 0:
        return ("LOW", f"negative total_score ({score:.2f})")
    if score < 0.5:
        label = "MEDIUM" if label != "HIGH" else "MEDIUM"
        reasons.append(f"modest total_score ({score:.2f})")

    # ---- 4. Assumptions match persistent gaps ----
    if obligation_store is not None:
        try:
            persistent = obligation_store.query_persistent_gaps(top_n=20)
            persistent_texts = [_norm(g.text) for g in persistent]
            for assumption in branch.get("assumptions", []) or []:
                if _matches_any(_norm(assumption), persistent_texts):
                    return ("LOW",
                            f"branch rests on assumption matching a persistent "
                            f"unresolved gap: '{assumption[:80]}'")
        except Exception:
            pass

    if not reasons:
        reasons.append("no failing signals but no strong confirming signals either")
    return (label, "; ".join(reasons))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _matches_any(needle: str, haystack: list[str], min_overlap: int = 4) -> bool:
    """Loose match: at least min_overlap shared content words."""
    needle_words = set(w for w in needle.split() if len(w) > 3)
    for h in haystack:
        h_words = set(w for w in h.split() if len(w) > 3)
        if len(needle_words & h_words) >= min_overlap:
            return True
    return False


# ============================================================
# Step 4: Answer with grounded confidence
# ============================================================

def answer_with_grounded_confidence(
    questions: list[dict],
    winner: dict,
    objective_confidence: tuple[str, str],
    llm_chat_fn: Callable,
) -> list[dict]:
    """
    Have the winner answer each question. The objective_confidence is shown
    to the LLM so it cannot self-report a higher confidence than the
    evidence supports.

    Returns list of dicts: {"question", "answer", "challenger", "targets"}.
    """
    if not questions:
        return []

    obj_label, obj_reason = objective_confidence

    system = (
        "You are answering hard questions about a piece of reasoning. "
        "These questions were generated by competing approaches and survived "
        "critique. They cannot be deflected by restating the method.\n\n"
        f"OBJECTIVE CONFIDENCE CEILING for this branch: {obj_label}\n"
        f"  Reason: {obj_reason}\n\n"
        "You may not assert higher confidence than this ceiling. "
        "If the answer is genuinely 'I don't know' or 'this step is "
        "unjustified', say so. Fluent restatement is not evidence."
    )

    numbered = "\n".join(f"{i+1}. {q['question']}" for i, q in enumerate(questions))
    user = f"""
Branch under audit:
Method: {winner.get('method', '')}
Steps:  {winner.get('steps', [])}
Result: {winner.get('candidate_result', '')}

Questions:
{numbered}

For each question return JSON:
{{
  "answers": [
    {{"index": 1, "answer": "...", "honest_confidence": "LOW|MEDIUM|HIGH"}},
    ...
  ]
}}
""".strip()

    try:
        response = llm_chat_fn(system, user)
        parsed = _extract_json(response)
    except Exception as e:
        return [{**q, "answer": f"[answering failed: {e}]"} for q in questions]

    answers_by_index = {a.get("index"): a for a in parsed.get("answers", [])
                        if isinstance(a.get("index"), int)}
    out = []
    for i, q in enumerate(questions):
        a = answers_by_index.get(i + 1, {})
        out.append({
            **q,
            "answer": a.get("answer", "[no answer returned]"),
            "honest_confidence": a.get("honest_confidence", "UNKNOWN"),
        })
    return out


# ============================================================
# Step 5: Full audit + loop closure into obligation store
# ============================================================

def metacognitive_audit(
    pipeline_result: dict,
    llm_chat_fn: Callable,
    obligation_store=None,
    record_low_confidence_as_obligations: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Full audit pass on a completed pipeline result.

    Parameters
    ----------
    pipeline_result : dict returned by ReasoningPipeline.run()
    llm_chat_fn     : same llm_chat used by the pipeline
    obligation_store: optional ObligationStore — if given, low-confidence
                      findings are recorded back as new obligations
    record_low_confidence_as_obligations : feed audit findings back into
                                            persistent memory

    Returns a dict with full audit details and a summary.
    """
    selected = pipeline_result.get("selected_branch")
    all_branches = pipeline_result.get("all_branches", [])
    domain = pipeline_result.get("domain", "")
    run_id = pipeline_result.get("run_id", "unknown")

    if not selected:
        if verbose:
            print("[metacog] No selected branch — skipping audit.")
        return {"skipped": True, "reason": "no selected branch"}

    losers = [b for b in all_branches
              if b.get("name") != selected.get("name")]

    if verbose:
        print(f"\n{'='*60}")
        print(f"[metacog] Auditing branch: {selected.get('name')}")
        print(f"{'='*60}")

    # --- 1. Challenger questions
    raw = generate_challenger_questions(
        selected, losers, llm_chat_fn, verbose=verbose,
    )
    if not raw:
        return {"skipped": True, "reason": "no challenger questions raised"}

    # --- 2. Critique
    survivors = critique_questions(raw, selected, llm_chat_fn, verbose=verbose)
    if not survivors:
        if verbose:
            print("  [metacog] All challenger questions deflectable. "
                  "Recording this as a finding.")
        survivors = [{
            "question": "What is the most fundamental assumption this entire "
                        "approach rests on, and where does it come from?",
            "targets": "foundational assumptions",
            "challenger": "[fallback]",
        }]

    # --- 3. Objective confidence
    obj_conf = derive_objective_confidence(selected, obligation_store)
    if verbose:
        print(f"  [metacog] Objective confidence: {obj_conf[0]} — {obj_conf[1]}")

    # --- 4. Answer with grounded confidence ceiling
    answered = answer_with_grounded_confidence(
        survivors, selected, obj_conf, llm_chat_fn,
    )

    # --- 5. Build findings
    findings: list[AuditFinding] = []
    for a in answered:
        finding = AuditFinding(
            question=a["question"],
            targets=a.get("targets", ""),
            challenger=a.get("challenger", ""),
            answer=a.get("answer", ""),
            objective_confidence=obj_conf[0],
            confidence_reason=obj_conf[1],
        )
        findings.append(finding)

    # --- 6. Loop closure: feed low-confidence findings back as obligations
    n_recorded = 0
    if (record_low_confidence_as_obligations
            and obligation_store is not None
            and obj_conf[0] == "LOW"):
        new_obligations = []
        for f in findings:
            new_obligations.append({
                "text": f"[audit::{f.challenger}] {f.question} — "
                        f"answer was: {f.answer[:200]}",
                "kind": "audit_finding",
                "source_branch": selected.get("name", "selected"),
                "status": "global_gap",
            })
            f.became_obligation = True
        try:
            obligation_store.record_run(
                run_id=run_id + "_metacog_audit",
                domain=domain,
                question=f"[metacognitive_audit of {selected.get('name')}]",
                resolution={
                    "obligations": new_obligations,
                    "global_gaps": new_obligations,
                },
            )
            n_recorded = len(new_obligations)
            if verbose:
                print(f"  [metacog] Recorded {n_recorded} audit findings "
                      f"as obligations.")
        except Exception as e:
            if verbose:
                print(f"  [metacog] Failed to record obligations: {e}")

    summary = {
        "branch_audited": selected.get("name"),
        "n_challenger_questions": len(raw),
        "n_survived_critique": len(survivors),
        "objective_confidence": obj_conf[0],
        "confidence_reason": obj_conf[1],
        "n_findings": len(findings),
        "n_recorded_as_obligations": n_recorded,
        "findings": [f.to_dict() for f in findings],
    }

    if verbose:
        _print_audit_summary(summary)

    return summary


def _print_audit_summary(summary: dict) -> None:
    print(f"\n[metacog] AUDIT SUMMARY")
    print(f"  Branch: {summary['branch_audited']}")
    print(f"  Confidence: {summary['objective_confidence']} "
          f"({summary['confidence_reason']})")
    print(f"  Questions raised: {summary['n_challenger_questions']} "
          f"(challenger) → {summary['n_survived_critique']} (after critique)")
    if summary["n_recorded_as_obligations"]:
        print(f"  Recorded {summary['n_recorded_as_obligations']} new obligations.")
    print()
    for i, f in enumerate(summary["findings"], 1):
        print(f"  [{i}] (from {f['challenger']}, targets: {f['targets']})")
        print(f"      Q: {f['question']}")
        print(f"      A: {f['answer'][:200]}")
        print()


# ============================================================
# JSON extraction helper
# ============================================================

def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}
