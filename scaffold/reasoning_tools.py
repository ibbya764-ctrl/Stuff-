"""
reasoning_tools.py
==================

Tool-use layer for the Quantum Reasoning Assistant.

Provides:
  - A registry of symbolic and numerical verification tools (built on sympy/numpy)
  - A parser for LLM-emitted tool calls
  - An execution layer with structured results
  - A tool-augmented branch verification function
  - Integration helpers for the existing scoring pipeline

Designed to be dropped into the existing notebook and called from the branch
generation / scoring code without breaking the current architecture.
"""

import re
import json
import random
from typing import Any, Callable

import sympy as sp
import numpy as np


# ============================================================
# Tool Registry
# ============================================================

TOOL_REGISTRY: dict[str, dict] = {}


def register_tool(name: str, description: str, parameters: dict):
    """Decorator to register a tool with a structured schema."""
    def wrapper(fn: Callable):
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "fn": fn,
        }
        return fn
    return wrapper


# ============================================================
# Symbolic Tools (sympy-based)
# ============================================================

@register_tool(
    name="sym_simplify",
    description=(
        "Simplify a symbolic expression. Use to check whether complex "
        "algebraic forms reduce to something simpler, or to test if an "
        "expression is identically zero."
    ),
    parameters={
        "expression": {"type": "string",
                       "description": "Sympy-parseable expression"}
    },
)
def sym_simplify(expression: str) -> dict:
    try:
        expr = sp.sympify(expression)
        simplified = sp.simplify(expr)
        return {
            "ok": True,
            "input": str(expr),
            "simplified": str(simplified),
            "is_zero": bool(simplified == 0),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@register_tool(
    name="sym_check_equality",
    description=(
        "Check whether two expressions are mathematically equal by simplifying "
        "their difference. Use this to verify any claimed equality in a derivation."
    ),
    parameters={
        "lhs": {"type": "string"},
        "rhs": {"type": "string"},
    },
)
def sym_check_equality(lhs: str, rhs: str) -> dict:
    try:
        a = sp.sympify(lhs)
        b = sp.sympify(rhs)
        diff = sp.simplify(a - b)
        return {
            "ok": True,
            "lhs": str(a),
            "rhs": str(b),
            "difference": str(diff),
            "equal": bool(diff == 0),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@register_tool(
    name="sym_solve",
    description=(
        "Solve an equation 'expression = 0' for a variable. Returns all "
        "symbolic solutions found by sympy."
    ),
    parameters={
        "equation": {"type": "string", "description": "Expression set to zero"},
        "variable": {"type": "string"},
    },
)
def sym_solve(equation: str, variable: str) -> dict:
    try:
        expr = sp.sympify(equation)
        var = sp.Symbol(variable)
        solutions = sp.solve(expr, var)
        return {
            "ok": True,
            "equation": f"{equation} = 0",
            "variable": variable,
            "solutions": [str(s) for s in solutions],
            "n_solutions": len(solutions),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@register_tool(
    name="sym_substitute",
    description=(
        "Substitute values into an expression and simplify. Useful for "
        "checking what an expression reduces to under specific assumptions."
    ),
    parameters={
        "expression": {"type": "string"},
        "substitutions": {
            "type": "object",
            "description": "Mapping of variable name to substitution value",
        },
    },
)
def sym_substitute(expression: str, substitutions: dict) -> dict:
    try:
        expr = sp.sympify(expression)
        subs = {sp.Symbol(k): sp.sympify(v) for k, v in substitutions.items()}
        result = sp.simplify(expr.subs(subs))
        return {
            "ok": True,
            "original": str(expr),
            "substitutions": {k: str(v) for k, v in substitutions.items()},
            "result": str(result),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@register_tool(
    name="sym_hessian",
    description=(
        "Compute the Hessian matrix of an expression with respect to given "
        "variables, and return its eigenvalues. Used for local stability "
        "analysis (positive eigenvalues => local minimum)."
    ),
    parameters={
        "expression": {"type": "string"},
        "variables": {"type": "array", "items": {"type": "string"}},
    },
)
def sym_hessian(expression: str, variables: list) -> dict:
    try:
        expr = sp.sympify(expression)
        vars_sym = [sp.Symbol(v) for v in variables]
        H = sp.hessian(expr, vars_sym)
        eigenvalues = list(H.eigenvals().items())
        return {
            "ok": True,
            "hessian": str(H),
            "eigenvalues": [(str(ev), int(mult)) for ev, mult in eigenvalues],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@register_tool(
    name="sym_limit",
    description=(
        "Compute the limit of an expression as a variable approaches a value. "
        "Use 'oo' for positive infinity, '-oo' for negative infinity. "
        "Excellent for counterexample search and edge-case analysis."
    ),
    parameters={
        "expression": {"type": "string"},
        "variable": {"type": "string"},
        "value": {"type": "string"},
    },
)
def sym_limit(expression: str, variable: str, value: str) -> dict:
    try:
        expr = sp.sympify(expression)
        var = sp.Symbol(variable)
        val = sp.sympify(value)
        result = sp.limit(expr, var, val)
        return {
            "ok": True,
            "expression": str(expr),
            "limit": f"{variable} -> {value}",
            "result": str(result),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@register_tool(
    name="sym_dimensional_check",
    description=(
        "Check that a symbolic expression has expected powers of fundamental "
        "scales (M, L, T). Variables given dimensions via the 'dim_map'. "
        "Returns the inferred dimension and whether it matches expected."
    ),
    parameters={
        "expression": {"type": "string"},
        "dim_map": {
            "type": "object",
            "description": "Mapping variable -> [M_power, L_power, T_power]",
        },
        "expected": {
            "type": "array",
            "description": "Expected [M_power, L_power, T_power]",
        },
    },
)
def sym_dimensional_check(expression: str, dim_map: dict, expected: list) -> dict:
    try:
        # Replace each variable with a tagged symbol for each dimension.
        M_sym, L_sym, T_sym = sp.symbols("__M __L __T", positive=True)
        expr = sp.sympify(expression)
        subs = {}
        for var, dims in dim_map.items():
            m, l, t = dims
            subs[sp.Symbol(var)] = M_sym**m * L_sym**l * T_sym**t
        replaced = expr.subs(subs)
        replaced = sp.simplify(replaced)

        # Extract powers of M, L, T.
        def power_of(symbol):
            return sp.Poly(sp.expand_log(sp.log(replaced)),
                           sp.log(symbol)).nth(1) if False else None

        m_pow = sp.simplify(sp.log(replaced).diff(M_sym) * M_sym / replaced *
                            replaced)  # crude — instead use degree:
        # Cleaner: extract via Poly on each axis.
        try:
            poly = sp.Poly(replaced, M_sym, L_sym, T_sym)
            monoms = poly.monoms()
            if len(monoms) != 1:
                return {
                    "ok": True,
                    "matches": False,
                    "warning": "Expression is not a single monomial in (M,L,T)",
                    "expression": str(replaced),
                }
            m_p, l_p, t_p = monoms[0]
        except sp.PolynomialError:
            # Fallback: use degree counting
            m_p = sp.degree(replaced, M_sym)
            l_p = sp.degree(replaced, L_sym)
            t_p = sp.degree(replaced, T_sym)

        actual = [int(m_p), int(l_p), int(t_p)]
        return {
            "ok": True,
            "actual": actual,
            "expected": list(expected),
            "matches": actual == list(expected),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ============================================================
# Numerical Tools (counterexample search)
# ============================================================

@register_tool(
    name="numerical_sample",
    description=(
        "Numerically sample an expression at random parameter values to detect "
        "regimes where it behaves unexpectedly (e.g. negative when claimed "
        "positive, divergent, etc). Strong tool for falsifying claims."
    ),
    parameters={
        "expression": {"type": "string"},
        "ranges": {
            "type": "object",
            "description": "Mapping variable -> [low, high]",
        },
        "n_samples": {"type": "integer", "default": 100},
    },
)
def numerical_sample(expression: str, ranges: dict, n_samples: int = 100) -> dict:
    try:
        expr = sp.sympify(expression)
        var_names = list(ranges.keys())
        var_syms = [sp.Symbol(v) for v in var_names]
        f = sp.lambdify(var_syms, expr, modules="numpy")

        rng = np.random.default_rng(42)
        samples = []
        for _ in range(n_samples):
            args = [rng.uniform(ranges[v][0], ranges[v][1]) for v in var_names]
            try:
                val = float(f(*args))
                if np.isfinite(val):
                    samples.append((args, val))
            except Exception:
                continue

        if not samples:
            return {"ok": True, "expression": str(expr),
                    "n_samples": n_samples, "all_failed": True}

        values = np.array([s[1] for s in samples])
        return {
            "ok": True,
            "expression": str(expr),
            "n_evaluated": len(samples),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "n_negative": int(np.sum(values < 0)),
            "n_near_zero": int(np.sum(np.abs(values) < 1e-10)),
            "n_diverging": n_samples - len(samples),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@register_tool(
    name="counterexample_search",
    description=(
        "Search for parameter values where a claimed inequality fails. "
        "Pass an expression that should be >= 0; returns any counterexample found. "
        "Use this aggressively to falsify stability/positivity claims."
    ),
    parameters={
        "expression": {"type": "string",
                       "description": "Should be >= 0 if claim holds"},
        "ranges": {"type": "object"},
        "n_samples": {"type": "integer", "default": 500},
    },
)
def counterexample_search(expression: str, ranges: dict,
                          n_samples: int = 500) -> dict:
    try:
        expr = sp.sympify(expression)
        var_names = list(ranges.keys())
        var_syms = [sp.Symbol(v) for v in var_names]
        f = sp.lambdify(var_syms, expr, modules="numpy")

        rng = np.random.default_rng(7)
        worst_args = None
        worst_val = float("inf")

        for _ in range(n_samples):
            args = [rng.uniform(ranges[v][0], ranges[v][1]) for v in var_names]
            try:
                val = float(f(*args))
                if np.isfinite(val) and val < worst_val:
                    worst_val = val
                    worst_args = args
            except Exception:
                continue

        if worst_args is None:
            return {"ok": True, "claim_holds": None,
                    "note": "Could not evaluate expression numerically."}

        if worst_val < 0:
            return {
                "ok": True,
                "claim_holds": False,
                "counterexample": dict(zip(var_names, worst_args)),
                "value_at_counterexample": worst_val,
                "note": "Expression evaluated negative — claim falsified.",
            }
        return {
            "ok": True,
            "claim_holds": True,
            "minimum_found": worst_val,
            "minimum_at": dict(zip(var_names, worst_args)),
            "note": "No counterexample found; minimum >= 0 across samples.",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ============================================================
# Tool execution
# ============================================================

def execute_tool_call(call: dict) -> dict:
    """Execute a parsed tool call and return its structured result."""
    name = call.get("name")
    args = call.get("arguments", {})

    if name not in TOOL_REGISTRY:
        return {"ok": False, "error": f"Unknown tool: {name!r}",
                "available": list(TOOL_REGISTRY.keys())}

    fn = TOOL_REGISTRY[name]["fn"]
    try:
        return fn(**args)
    except TypeError as e:
        return {"ok": False, "error": f"Argument error in {name}: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Tool {name} raised: {e}"}


def get_tools_prompt() -> str:
    """Generate a system-prompt fragment describing all tools."""
    lines = [
        "You have access to symbolic and numerical verification tools.",
        "To call a tool, emit a block exactly like this:",
        "",
        "<tool_call>",
        '{"name": "TOOL_NAME", "arguments": {"arg": "value"}}',
        "</tool_call>",
        "",
        "Available tools:",
        "",
    ]
    for tool in TOOL_REGISTRY.values():
        lines.append(f"- {tool['name']}: {tool['description']}")
        for param, spec in tool["parameters"].items():
            desc = spec.get("description", "")
            ptype = spec.get("type", "any")
            lines.append(f"    {param} ({ptype}): {desc}")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# Tool call parsing
# ============================================================

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract all tool calls from an LLM response."""
    calls = []
    for match in TOOL_CALL_PATTERN.findall(text):
        try:
            calls.append(json.loads(match))
        except json.JSONDecodeError as e:
            calls.append({"name": "_parse_error", "error": str(e),
                          "raw": match})
    return calls


def format_tool_results(calls: list[dict], results: list[dict]) -> str:
    """Format tool results for re-injection into the LLM context."""
    out = []
    for call, result in zip(calls, results):
        out.append(f"<tool_result name=\"{call.get('name', '?')}\">")
        out.append(json.dumps(result, indent=2))
        out.append("</tool_result>")
    return "\n".join(out)


# ============================================================
# Branch verification using tools
# ============================================================

def verify_branch_with_tools(branch_dict: dict, llm_chat_fn,
                             max_rounds: int = 3) -> dict:
    """
    Run a tool-driven verification of a single branch.

    Asks the LLM to verify the branch's candidate result using tools.
    Returns a verification report containing all tool calls and their results,
    plus a final verdict.

    Parameters
    ----------
    branch_dict: dict from the branches_json structure (name, steps,
                 candidate_result, etc.)
    llm_chat_fn: function (system_prompt, user_prompt) -> str (your existing
                 llm_chat).
    max_rounds: how many rounds of tool use to allow.
    """
    system_prompt = (
        "You are a rigorous proof-checker. You will be given a reasoning "
        "branch with steps and a candidate result. Your job is to verify "
        "every concrete algebraic or numerical claim using the tools below. "
        "Do not approve any step you have not verified with a tool. After "
        "your investigation, output a final JSON verdict in this exact form:\n"
        "<verdict>\n"
        '{"verified": true|false, "verified_steps": [...], '
        '"failed_steps": [...], "notes": "..."}'
        "\n</verdict>\n\n"
        + get_tools_prompt()
    )

    branch_text = (
        f"Branch name: {branch_dict.get('name')}\n"
        f"Method: {branch_dict.get('method')}\n"
        f"Assumptions: {branch_dict.get('assumptions', [])}\n"
        f"Steps:\n"
        + "\n".join(f"  {i+1}. {s}" for i, s
                    in enumerate(branch_dict.get('steps', [])))
        + f"\nCandidate result: {branch_dict.get('candidate_result')}"
    )

    transcript = [branch_text]
    all_calls = []
    all_results = []

    for round_idx in range(max_rounds):
        user_prompt = "\n\n".join(transcript)
        response = llm_chat_fn(system_prompt, user_prompt)
        transcript.append(f"[ASSISTANT ROUND {round_idx}]\n{response}")

        calls = parse_tool_calls(response)
        if not calls:
            # No more tool calls; expect a verdict.
            break

        results = [execute_tool_call(c) for c in calls]
        all_calls.extend(calls)
        all_results.extend(results)
        transcript.append(format_tool_results(calls, results))

    # The verdict should be in the most recent assistant turn.
    assistant_turns = [t for t in transcript if t.startswith("[ASSISTANT ROUND")]
    final_response = assistant_turns[-1] if assistant_turns else ""
    verdict = _extract_verdict(final_response)

    return {
        "branch_name": branch_dict.get("name"),
        "tool_calls": all_calls,
        "tool_results": all_results,
        "n_calls": len(all_calls),
        "verdict": verdict,
        "transcript": transcript,
    }


def _extract_verdict(text: str) -> dict:
    match = re.search(r"<verdict>\s*(\{.*?\})\s*</verdict>", text, re.DOTALL)
    if not match:
        return {"verified": None, "note": "No verdict block found"}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        return {"verified": None, "note": f"Verdict parse failed: {e}"}


# ============================================================
# Scoring integration
# ============================================================

def tool_verification_score(verification_report: dict) -> float:
    """
    Convert a verification report into a [-1, +1] score that can be added to
    the existing branch.total_score in your pipeline.
    """
    verdict = verification_report.get("verdict", {})
    n_calls = verification_report.get("n_calls", 0)

    if verdict.get("verified") is True:
        base = 0.5
    elif verdict.get("verified") is False:
        base = -0.7
    else:
        base = 0.0

    # Reward effort: branches that triggered actual verification work get a
    # small bonus over branches that produced no checkable claims.
    effort_bonus = min(0.2, 0.05 * n_calls)

    # Penalise branches whose verification produced tool errors.
    failures = sum(1 for r in verification_report.get("tool_results", [])
                   if not r.get("ok", True))
    error_penalty = min(0.3, 0.05 * failures)

    return float(base + effort_bonus - error_penalty)
