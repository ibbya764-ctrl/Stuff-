"""
reasoning_tools_extended.py
===========================

Extended tools for non-physics domains: code execution and claim extraction.

These register into the same TOOL_REGISTRY as reasoning_tools.py, so importing
this module after reasoning_tools.py adds these tools to the same registry.
"""

import re
import io
import sys
import contextlib
import signal
import json
from typing import Any

from reasoning_tools import register_tool


# ============================================================
# Code execution tools
# ============================================================

class _ExecTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _ExecTimeout("Code execution timed out")


@register_tool(
    name="code_execute",
    description=(
        "Execute a Python code snippet and return its stdout and the value "
        "of any expression on the last line. Use this to test code claims "
        "empirically rather than reasoning about them in prose. Times out "
        "after a few seconds. The execution environment is fresh each call."
    ),
    parameters={
        "code": {"type": "string", "description": "Python code to execute"},
        "timeout_seconds": {"type": "integer", "default": 5},
    },
)
def code_execute(code: str, timeout_seconds: int = 5) -> dict:
    namespace: dict[str, Any] = {}
    stdout_buffer = io.StringIO()
    last_value = None

    # Try to extract the last expression as something we can evaluate.
    lines = code.strip().split("\n")
    last_line = lines[-1].strip() if lines else ""
    is_expression = (
        last_line
        and not last_line.startswith(("def ", "class ", "import ", "from "))
        and "=" not in last_line.split("#")[0]
        and ":" not in last_line
    )

    if is_expression and len(lines) > 1:
        body = "\n".join(lines[:-1])
        tail = last_line
    else:
        body = code
        tail = None

    has_signal = hasattr(signal, "SIGALRM")
    if has_signal:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(body, namespace)
            if tail is not None:
                last_value = eval(tail, namespace)
    except _ExecTimeout:
        return {
            "ok": False,
            "error": f"Execution timed out after {timeout_seconds}s",
            "stdout": stdout_buffer.getvalue(),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "stdout": stdout_buffer.getvalue(),
        }
    finally:
        if has_signal:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return {
        "ok": True,
        "stdout": stdout_buffer.getvalue(),
        "last_value": repr(last_value) if last_value is not None else None,
    }


@register_tool(
    name="code_test_cases",
    description=(
        "Run a function definition against a list of test cases. Each case "
        "is (input_args, expected_output). Useful for empirically validating "
        "a code branch's correctness claims across multiple inputs."
    ),
    parameters={
        "function_code": {
            "type": "string",
            "description": "Python source defining the function",
        },
        "function_name": {"type": "string"},
        "test_cases": {
            "type": "array",
            "description": (
                "List of {'args': [...], 'expected': value} or "
                "{'kwargs': {...}, 'expected': value}"
            ),
        },
    },
)
def code_test_cases(function_code: str, function_name: str,
                    test_cases: list) -> dict:
    namespace: dict[str, Any] = {}
    try:
        exec(function_code, namespace)
    except Exception as e:
        return {"ok": False, "error": f"Function definition failed: {e}"}

    if function_name not in namespace:
        return {"ok": False,
                "error": f"Function {function_name!r} not found after exec"}

    fn = namespace[function_name]
    results = []
    n_pass = 0
    for i, case in enumerate(test_cases):
        args = case.get("args", [])
        kwargs = case.get("kwargs", {})
        expected = case.get("expected")
        try:
            actual = fn(*args, **kwargs)
            passed = actual == expected
            results.append({
                "case": i,
                "args": args,
                "kwargs": kwargs,
                "expected": repr(expected),
                "actual": repr(actual),
                "passed": passed,
            })
            if passed:
                n_pass += 1
        except Exception as e:
            results.append({
                "case": i,
                "args": args,
                "kwargs": kwargs,
                "error": f"{type(e).__name__}: {e}",
                "passed": False,
            })

    return {
        "ok": True,
        "n_total": len(test_cases),
        "n_passed": n_pass,
        "all_passed": n_pass == len(test_cases),
        "results": results,
    }


# ============================================================
# Claim extraction (used in argument domain)
# ============================================================

# Heuristic claim-extraction. A more robust version would use an LLM, but a
# regex pass gives a useful first cut without extra model calls.

CLAIM_INDICATORS = [
    r"\bI argue\b", r"\bI claim\b", r"\bThe thesis is\b",
    r"\bIt follows that\b", r"\bTherefore\b", r"\bHence\b",
    r"\bThus\b", r"\bThis shows\b", r"\bThis means\b",
    r"\bImplies that\b", r"\bConcludes that\b",
]

EVIDENCE_INDICATORS = [
    r"\baccording to\b", r"\bstudy found\b", r"\bdata show\b",
    r"\bevidence indicates\b", r"\bdocumented that\b",
    r"\bresearch suggests\b",
]

UNSUPPORTED_INDICATORS = [
    r"\beveryone knows\b", r"\bobviously\b", r"\bclearly\b",
    r"\bcommon sense\b", r"\bit is well known\b",
    r"\bgoes without saying\b", r"\bof course\b",
]


@register_tool(
    name="claim_extract",
    description=(
        "Extract claims, evidence-citations, and unsupported assertions "
        "from a piece of prose. Useful for argument analysis and "
        "identifying which pieces of a position need verification."
    ),
    parameters={
        "text": {"type": "string"},
    },
)
def claim_extract(text: str) -> dict:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    evidence_citations = []
    unsupported = []

    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        for pat in CLAIM_INDICATORS:
            if re.search(pat, s, re.IGNORECASE):
                claims.append(s)
                break
        for pat in EVIDENCE_INDICATORS:
            if re.search(pat, s, re.IGNORECASE):
                evidence_citations.append(s)
                break
        for pat in UNSUPPORTED_INDICATORS:
            if re.search(pat, s, re.IGNORECASE):
                unsupported.append(s)
                break

    return {
        "ok": True,
        "n_sentences": len(sentences),
        "claims": claims,
        "evidence_citations": evidence_citations,
        "unsupported_assertions": unsupported,
        "support_ratio": (
            len(evidence_citations) / max(1, len(claims))
            if claims else None
        ),
    }


@register_tool(
    name="consistency_check",
    description=(
        "Check a list of claims for pairwise contradictions using simple "
        "string-based negation matching. Returns suspicious pairs that "
        "may contradict. This is heuristic — for hard cases use an LLM."
    ),
    parameters={
        "claims": {"type": "array", "items": {"type": "string"}},
    },
)
def consistency_check(claims: list) -> dict:
    suspicious = []
    negation_patterns = [
        (r"\bis\b", r"\bis not\b"),
        (r"\bare\b", r"\bare not\b"),
        (r"\bcan\b", r"\bcannot\b"),
        (r"\bdoes\b", r"\bdoes not\b"),
        (r"\bwill\b", r"\bwill not\b"),
        (r"\bshould\b", r"\bshould not\b"),
        (r"\bmust\b", r"\bmust not\b"),
    ]

    def normalise(s):
        return re.sub(r"\s+", " ", s.lower().strip())

    norm_claims = [normalise(c) for c in claims]

    for i in range(len(norm_claims)):
        for j in range(i + 1, len(norm_claims)):
            for pos, neg in negation_patterns:
                if re.search(pos, norm_claims[i]) and re.search(neg, norm_claims[j]):
                    a_core = re.sub(pos, "", norm_claims[i])
                    b_core = re.sub(neg, "", norm_claims[j])
                    a_words = set(a_core.split())
                    b_words = set(b_core.split())
                    if len(a_words & b_words) >= 3:
                        suspicious.append({
                            "claim_a": claims[i],
                            "claim_b": claims[j],
                            "shared_terms": sorted(a_words & b_words),
                        })

    return {
        "ok": True,
        "n_claims": len(claims),
        "n_suspicious_pairs": len(suspicious),
        "suspicious_pairs": suspicious,
    }
