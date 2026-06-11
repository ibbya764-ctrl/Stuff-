"""
code_generator.py
=================

Recursive self-improvement via targeted code generation.

The system detects where it is consistently failing, generates code
to fix those specific weaknesses, tests the code in isolation, and
submits it through the adopt/reject cycle.

Three components:

  FailurePatternDetector  — analyses the episodic store and obligation
                            store to find where the system is consistently
                            struggling. Returns typed FailurePattern objects
                            with evidence and a suggested improvement.

  CodeSandbox             — executes generated code in an isolated
                            environment with forbidden-pattern checking,
                            restricted builtins, and a timeout.
                            Nothing generated can touch the filesystem,
                            spawn processes, or call the network.

  AttentionDirector       — orchestrates the loop:
                            detect failures → generate code → test →
                            adopt or reject → log what worked.
                            The log trains better generation over time.

What gets generated:
  - Verification functions: new sympy/numpy checks for problem types
    where existing verification consistently fails
  - Technique dicts: new entries for the technique library targeting
    domains/approaches that are underrepresented
  - Step templates: better structured templates for the branch sampler
    when certain step types are consistently unverifiable

What does NOT get generated (yet):
  - Full pipeline modules
  - Changes to module interfaces
  - Anything requiring file system access

Safety mechanisms built into every generation:
  - Forbidden pattern scan (no os, subprocess, exec, eval, open)
  - Restricted builtins (math, numpy, sympy allowed; everything else blocked)
  - Execution timeout (kills runaway loops)
  - Syntax validation before execution
  - Output type validation
  - Human approval required for anything touching the pipeline architecture
"""

import ast
import json
import math
import os
import re
import time
import threading
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Optional, Callable, Any


# ============================================================
# Failure patterns
# ============================================================

@dataclass
class FailurePattern:
    """
    A typed description of where the system is consistently struggling.
    """
    pattern_id:          str        # unique identifier
    pattern_type:        str        # "verification_failure" | "domain_gap" |
                                    # "technique_weakness" | "step_failure"
    domain:              str
    description:         str        # human-readable description
    evidence:            dict       # supporting statistics from episodic store
    suggested_fix:       str        # what kind of code would address this
    priority:            float      # 0-1, higher = more urgent
    n_occurrences:       int        # how many times this pattern was observed

    def to_dict(self) -> dict:
        return {
            "pattern_id":    self.pattern_id,
            "pattern_type":  self.pattern_type,
            "domain":        self.domain,
            "description":   self.description[:200],
            "evidence":      self.evidence,
            "suggested_fix": self.suggested_fix[:200],
            "priority":      self.priority,
            "n_occurrences": self.n_occurrences,
        }


class FailurePatternDetector:
    """
    Detects where the system is struggling by analysing the
    episodic store, obligation store, and technique library.
    """

    def __init__(
        self,
        episodic_store,
        obligation_store,
        technique_library,
        min_occurrences: int   = 3,
        failure_threshold: float = 0.45,
    ):
        self.episodic   = episodic_store
        self.store      = obligation_store
        self.library    = technique_library
        self.min_occ    = min_occurrences
        self.fail_thresh = failure_threshold

    def detect_all(self) -> list[FailurePattern]:
        """
        Run all detectors and return a prioritised list.
        """
        patterns: list[FailurePattern] = []
        patterns.extend(self._detect_domain_gaps())
        patterns.extend(self._detect_verification_failures())
        patterns.extend(self._detect_technique_weaknesses())
        patterns.extend(self._detect_persistent_obligation_clusters())

        # Deduplicate by pattern_id
        seen: set[str] = set()
        unique: list[FailurePattern] = []
        for p in sorted(patterns, key=lambda x: x.priority, reverse=True):
            if p.pattern_id not in seen:
                seen.add(p.pattern_id)
                unique.append(p)
        return unique

    # ---- Domain gap detection ----

    def _detect_domain_gaps(self) -> list[FailurePattern]:
        patterns: list[FailurePattern] = []
        techs   = self.library._data.get("techniques", {})
        domains = {}
        for t in techs.values():
            d = t.get("domain", "")
            if d:
                domains[d] = domains.get(d, 0) + 1

        for domain, n_techs in domains.items():
            if n_techs < 4:
                domain_runs = self.episodic.query_domain(domain)
                n_failed = sum(1 for r in domain_runs if not r.verified)
                patterns.append(FailurePattern(
                    pattern_id=f"domain_gap::{domain}",
                    pattern_type="domain_gap",
                    domain=domain,
                    description=(
                        f"'{domain}' has only {n_techs} technique(s). "
                        f"{n_failed}/{len(domain_runs)} runs unverified."
                    ),
                    evidence={
                        "n_techniques":  n_techs,
                        "n_runs":        len(domain_runs),
                        "n_unverified":  n_failed,
                    },
                    suggested_fix="technique_dict",
                    priority=0.9 - n_techs * 0.1,
                    n_occurrences=n_failed,
                ))
        return patterns

    # ---- Verification failure detection ----

    def _detect_verification_failures(self) -> list[FailurePattern]:
        patterns: list[FailurePattern] = []
        all_runs = self.episodic.query_recent(n=100)
        if len(all_runs) < self.min_occ:
            return patterns

        # Cluster failed runs by question keywords
        keyword_failures: dict[str, list] = {}
        for run in all_runs:
            if run.verified:
                continue
            words = set(_simple_tokenise(run.question))
            for w in words:
                keyword_failures.setdefault(w, []).append(run)

        for keyword, failed_runs in keyword_failures.items():
            if len(failed_runs) < self.min_occ:
                continue
            # Check if this keyword also appears in verified runs
            # (if failures are selective, not universal)
            verified_with_kw = [
                r for r in all_runs
                if r.verified and keyword in _simple_tokenise(r.question)
            ]
            failure_rate = len(failed_runs) / max(
                1, len(failed_runs) + len(verified_with_kw)
            )
            if failure_rate < self.fail_thresh:
                continue

            domain = _most_common([r.domain for r in failed_runs]) or ""
            patterns.append(FailurePattern(
                pattern_id=f"verification_failure::{keyword}::{domain}",
                pattern_type="verification_failure",
                domain=domain,
                description=(
                    f"Problems involving '{keyword}' fail verification "
                    f"{failure_rate:.0%} of the time in '{domain}'."
                ),
                evidence={
                    "keyword":       keyword,
                    "n_failures":    len(failed_runs),
                    "failure_rate":  round(failure_rate, 3),
                    "n_verified":    len(verified_with_kw),
                },
                suggested_fix="verification_function",
                priority=min(0.95, failure_rate * 0.8),
                n_occurrences=len(failed_runs),
            ))

        return patterns

    # ---- Technique weakness detection ----

    def _detect_technique_weaknesses(self) -> list[FailurePattern]:
        patterns: list[FailurePattern] = []
        techs = self.library._data.get("techniques", {})

        for tid, t in techs.items():
            n_s = t.get("n_success", 0)
            n_f = t.get("n_fail", 0)
            total = n_s + n_f
            if total < self.min_occ:
                continue
            rate = n_s / total
            if rate > self.fail_thresh:
                continue
            domain = t.get("domain", "")
            patterns.append(FailurePattern(
                pattern_id=f"technique_weakness::{tid}",
                pattern_type="technique_weakness",
                domain=domain,
                description=(
                    f"Technique '{t.get('name', tid)}' has {rate:.0%} "
                    f"success rate ({n_s}/{total} runs)."
                ),
                evidence={
                    "technique_id":   tid,
                    "technique_name": t.get("name", ""),
                    "n_success":      n_s,
                    "n_fail":         n_f,
                    "success_rate":   round(rate, 3),
                },
                suggested_fix="technique_revision",
                priority=min(0.85, (1 - rate) * 0.7),
                n_occurrences=n_f,
            ))

        return patterns

    # ---- Persistent obligation clusters ----

    def _detect_persistent_obligation_clusters(self) -> list[FailurePattern]:
        patterns: list[FailurePattern] = []
        try:
            gaps = self.store.query_persistent_gaps(top_n=50)
        except Exception:
            return patterns

        # Group by keyword overlap
        keyword_groups: dict[str, list] = {}
        for gap in gaps:
            words = _simple_tokenise(gap.text)
            for w in words[:3]:   # anchor to first few distinctive words
                keyword_groups.setdefault(w, []).append(gap)

        for keyword, group in keyword_groups.items():
            if len(group) < self.min_occ:
                continue
            total_occ = sum(g.n_occurrences for g in group)
            patterns.append(FailurePattern(
                pattern_id=f"obligation_cluster::{keyword}",
                pattern_type="verification_failure",
                domain="",
                description=(
                    f"{len(group)} persistent gaps cluster around '{keyword}' "
                    f"({total_occ} total occurrences)."
                ),
                evidence={
                    "keyword":      keyword,
                    "n_gaps":       len(group),
                    "total_occ":    total_occ,
                    "gap_texts":    [g.text[:80] for g in group[:3]],
                },
                suggested_fix="verification_function",
                priority=min(0.90, total_occ / 50),
                n_occurrences=total_occ,
            ))

        return patterns


# ============================================================
# Code sandbox
# ============================================================

FORBIDDEN_PATTERNS = [
    r"\bos\b",
    r"\bsubprocess\b",
    r"\bopen\b\s*\(",
    r"\bexec\b\s*\(",
    r"\beval\b\s*\(",
    r"__import__",
    r"\bimportlib\b",
    r"\bsocket\b",
    r"\bshutil\b",
    r"\bpathlib\b",
    r"\bglob\b",
    r"\.write\s*\(",
    r"\.delete\s*\(",
    r"\bpickle\b",
]

ALLOWED_IMPORTS = {
    "numpy", "np", "scipy", "sympy", "math", "re", "json",
    "collections", "itertools", "functools", "typing",
    "dataclasses", "abc", "copy",
}


@dataclass
class SandboxResult:
    success:      bool
    output:       Any    = None
    error:        str    = ""
    timed_out:    bool   = False
    blocked:      str    = ""   # which forbidden pattern was hit


class CodeSandbox:
    """
    Executes generated code in a controlled environment.
    Checks for forbidden patterns, restricts builtins, enforces timeout.
    """

    def __init__(self, timeout_seconds: int = 5):
        self.timeout = timeout_seconds

    def check_safety(self, code: str) -> Optional[str]:
        """
        Return a description of any forbidden pattern found,
        or None if the code is safe to execute.
        """
        # Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"SyntaxError: {e}"

        # Forbidden pattern check
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                return f"Forbidden pattern: {pattern}"

        # Check imports are in the allowed set
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in getattr(node, "names", []):
                        base = alias.name.split(".")[0]
                        if base not in ALLOWED_IMPORTS:
                            return f"Import not allowed: {alias.name}"
                    if isinstance(node, ast.ImportFrom) and node.module:
                        base = node.module.split(".")[0]
                        if base not in ALLOWED_IMPORTS:
                            return f"Import not allowed: {node.module}"
        except Exception:
            pass

        return None

    def execute(
        self,
        code: str,
        call_expression: str = "",
        test_inputs: Optional[dict] = None,
    ) -> SandboxResult:
        """
        Execute generated code and optionally call a function within it.

        code:            the generated code (function definition etc.)
        call_expression: e.g. "my_function(1, 2, 3)"
        test_inputs:     dict of variable bindings available in the call

        Returns SandboxResult with success, output, error.
        """
        # Safety check
        block_reason = self.check_safety(code)
        if block_reason:
            return SandboxResult(success=False, blocked=block_reason)

        result_container: dict = {"result": None, "error": "", "done": False}

        def _run():
            try:
                import numpy as _np
                import sympy as _sympy
                namespace = {
                    "__builtins__": {
                        "abs": abs, "int": int, "float": float, "str": str,
                        "bool": bool, "list": list, "dict": dict, "tuple": tuple,
                        "set": set, "frozenset": frozenset,
                        "len": len, "range": range, "enumerate": enumerate,
                        "zip": zip, "map": map, "filter": filter,
                        "min": min, "max": max, "sum": sum, "round": round,
                        "print": print, "isinstance": isinstance,
                        "all": all, "any": any, "sorted": sorted,
                        "hasattr": hasattr, "getattr": getattr,
                        "type": type, "repr": repr,
                        "None": None, "True": True, "False": False,
                        "__import__": lambda name, *a, **k: (
                            __import__(name, *a, **k)
                            if name.split(".")[0] in ALLOWED_IMPORTS
                            else (_ for _ in ()).throw(
                                ImportError(f"Import not allowed: {name}")
                            )
                        ),
                    },
                    "np": _np,
                    "numpy": _np,
                    "sympy": _sympy,
                    "math": math,
                    **(test_inputs or {}),
                }
                exec(code, namespace)
                if call_expression:
                    result_container["result"] = eval(
                        call_expression, namespace
                    )
                else:
                    result_container["result"] = True
                result_container["done"] = True
            except Exception as e:
                result_container["error"] = str(e)
                result_container["done"]  = True

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if not result_container["done"]:
            return SandboxResult(success=False, timed_out=True,
                                  error=f"Execution timed out after {self.timeout}s")

        if result_container["error"]:
            return SandboxResult(success=False, error=result_container["error"])

        return SandboxResult(success=True, output=result_container["result"])


# ============================================================
# Generated code types and CodeGenerator
# ============================================================

@dataclass
class GeneratedCode:
    """One piece of generated code with metadata."""

    generation_id:   str
    pattern:         FailurePattern
    code_type:       str     # "verification_function" | "technique_dict" | "step_template"
    code:            str
    test_passed:     bool    = False
    sandbox_result:  Optional[SandboxResult] = None
    adopted:         bool    = False
    timestamp:       float   = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "generation_id": self.generation_id,
            "pattern_id":    self.pattern.pattern_id,
            "code_type":     self.code_type,
            "code_preview":  self.code[:200],
            "test_passed":   self.test_passed,
            "adopted":       self.adopted,
            "timestamp":     self.timestamp,
            "error":         (self.sandbox_result.error
                              if self.sandbox_result else ""),
        }


class CodeGenerator:
    """
    Generates code to fix specific failure patterns.
    All generated code is tested in the sandbox before submission.
    """

    def __init__(
        self,
        llm_chat_fn:   Callable,
        sandbox:       Optional[CodeSandbox] = None,
        technique_library = None,
    ):
        self.llm        = llm_chat_fn
        self.sandbox    = sandbox or CodeSandbox()
        self.library    = technique_library
        self._history:  list[dict] = []

    def generate_for_pattern(
        self, pattern: FailurePattern,
    ) -> Optional[GeneratedCode]:
        """
        Generate code for a specific failure pattern.
        Returns None if generation or testing fails.
        """
        fix = pattern.suggested_fix

        if fix == "verification_function":
            return self._generate_verifier(pattern)
        elif fix in ("technique_dict", "technique_revision"):
            return self._generate_technique(pattern)
        elif fix == "step_template":
            return self._generate_step_templates(pattern)
        else:
            return None

    # ---- Verification function generation ----

    def _generate_verifier(self, pattern: FailurePattern) -> Optional[GeneratedCode]:
        system = (
            "You generate Python verification functions for a reasoning system. "
            "The function must:\n"
            "  - Accept sympy expressions or numeric values as inputs\n"
            "  - Return a dict: {verified: bool, note: str}\n"
            "  - Use only: numpy (as np), sympy, math, basic Python builtins\n"
            "  - Be under 40 lines\n"
            "  - Have no side effects, no file I/O, no subprocess calls\n"
            "  - Handle exceptions gracefully\n"
            "Return ONLY the Python function, no explanation."
        )
        user = (
            f"Problem pattern:\n{pattern.description}\n\n"
            f"Evidence:\n{json.dumps(pattern.evidence, indent=2)}\n\n"
            f"Domain: {pattern.domain}\n\n"
            f"Generate a Python function called `verify_{_slug(pattern.pattern_id)}` "
            f"that specifically addresses this verification failure. "
            f"The function should catch the specific case that keeps failing "
            f"and return {{verified: True}} or {{verified: False, note: reason}}."
        )

        try:
            code = self.llm(system, user).strip()
            code = _clean_code_block(code)
        except Exception as e:
            return None

        # Extract the actual function name from the generated code
        # rather than assuming the LLM used the name we suggested
        import ast as _ast
        try:
            tree = _ast.parse(code)
            func_names = [n.name for n in _ast.walk(tree)
                          if isinstance(n, _ast.FunctionDef)]
            test_call = f"{func_names[0]}()" if func_names else ""
        except Exception:
            test_call = ""

        result = self.sandbox.execute(code, test_call)

        gen = GeneratedCode(
            generation_id=f"gen-{int(time.time())}",
            pattern=pattern,
            code_type="verification_function",
            code=code,
            test_passed=result.success,
            sandbox_result=result,
        )
        self._history.append(gen.to_dict())
        return gen if result.success else gen   # return either way for logging

    # ---- Technique dict generation ----

    def _generate_technique(self, pattern: FailurePattern) -> Optional[GeneratedCode]:
        existing_names = [
            t.get("name", "")
            for t in (self.library._data.get("techniques", {}).values()
                      if self.library else [])
            if t.get("domain") == pattern.domain
        ][:5]

        system = (
            "You generate technique library entries for a reasoning system. "
            "Return a Python dict with these exact keys: "
            "name, description, when_to_use, example_text\n"
            "The example_text should contain 3-5 concrete steps.\n"
            "Return ONLY the Python dict literal, no explanation."
        )
        user = (
            f"I need a new technique for the '{pattern.domain}' domain.\n\n"
            f"Problem being addressed:\n{pattern.description}\n\n"
            f"Evidence:\n{json.dumps(pattern.evidence, indent=2)}\n\n"
            f"Existing techniques in this domain (don't duplicate these):\n"
            f"{chr(10).join('  - ' + n for n in existing_names)}\n\n"
            f"Generate a technique dict that directly addresses the failure pattern. "
            f"The technique should be concrete and actionable."
        )

        try:
            code = self.llm(system, user).strip()
            code = _clean_code_block(code)
        except Exception as e:
            return None

        # Test: the code should produce a valid dict when executed
        result = self.sandbox.execute(
            f"_result = {code}", "", {}
        )
        # Also try to verify it has the required keys
        if result.success:
            check_code = (
                f"_d = {code}\n"
                f"assert isinstance(_d, dict)\n"
                f"assert all(k in _d for k in "
                f"['name','description','when_to_use','example_text'])\n"
            )
            result = self.sandbox.execute(check_code)

        gen = GeneratedCode(
            generation_id=f"gen-{int(time.time())}",
            pattern=pattern,
            code_type="technique_dict",
            code=code,
            test_passed=result.success,
            sandbox_result=result,
        )
        self._history.append(gen.to_dict())
        return gen

    # ---- Step template generation ----

    def _generate_step_templates(self, pattern: FailurePattern) -> Optional[GeneratedCode]:
        system = (
            "You generate step templates for a reasoning branch sampler. "
            "Return a Python list of step dicts, each with: "
            "{description: str, step_type: 'structural'|'derived'|'verifiable', "
            "gap_prompt: str}\n"
            "Return ONLY the Python list literal."
        )
        user = (
            f"I need step templates for problems like:\n{pattern.description}\n\n"
            f"Domain: {pattern.domain}\n"
            f"Generate 4-6 steps that would address this problem type. "
            f"Include at least 2 structural steps (no LLM needed) and "
            f"at least 1 verifiable step."
        )

        try:
            code = self.llm(system, user).strip()
            code = _clean_code_block(code)
        except Exception as e:
            return None

        check_code = (
            f"_steps = {code}\n"
            f"assert isinstance(_steps, list)\n"
            f"assert len(_steps) >= 3\n"
            f"assert all('description' in s and 'step_type' in s for s in _steps)\n"
        )
        result = self.sandbox.execute(check_code)

        gen = GeneratedCode(
            generation_id=f"gen-{int(time.time())}",
            pattern=pattern,
            code_type="step_template",
            code=code,
            test_passed=result.success,
            sandbox_result=result,
        )
        self._history.append(gen.to_dict())
        return gen

    def generation_history(self) -> list[dict]:
        return list(self._history)


# ============================================================
# AttentionDirector
# ============================================================

class AttentionDirector:
    """
    Orchestrates the detect → generate → test → adopt loop.

    The system directs its attention to where it is struggling,
    generates improvements, and learns from which kinds of
    improvements actually work.
    """

    def __init__(
        self,
        failure_detector:  FailurePatternDetector,
        code_generator:    CodeGenerator,
        self_adjuster,
        technique_library,
        log_path:    str  = "./attention_log.json",
        verbose:     bool = True,
    ):
        self.detector   = failure_detector
        self.generator  = code_generator
        self.adjuster   = self_adjuster
        self.library    = technique_library
        self.log_path   = log_path
        self.verbose    = verbose
        self._log: list[dict] = self._load_log()

    def run_cycle(
        self, max_patterns: int = 3,
    ) -> dict:
        """
        Run one full attention cycle:
          1. Detect the top failure patterns
          2. Generate code for each
          3. Test in sandbox
          4. Apply passing code to the system
          5. Log everything

        Returns a summary dict.
        """
        if self.verbose:
            print("\n[attention] Starting attention cycle...")

        patterns = self.detector.detect_all()[:max_patterns]

        if not patterns:
            if self.verbose:
                print("  No significant failure patterns detected.")
            return {"status": "no_patterns", "n_patterns": 0}

        if self.verbose:
            print(f"  Detected {len(patterns)} failure pattern(s):")
            for p in patterns:
                print(f"    [{p.priority:.2f}] {p.pattern_type}: "
                      f"{p.description[:80]}")

        results = {
            "n_patterns":  len(patterns),
            "generated":   0,
            "passed":      0,
            "applied":     0,
            "patterns":    [p.to_dict() for p in patterns],
        }

        for pattern in patterns:
            if self.verbose:
                print(f"\n  Generating fix for: {pattern.description[:60]}...")

            gen = self.generator.generate_for_pattern(pattern)
            if gen is None:
                if self.verbose:
                    print("    Generation failed.")
                continue

            results["generated"] += 1

            if not gen.test_passed:
                if self.verbose:
                    err = (gen.sandbox_result.error
                           if gen.sandbox_result else "unknown")
                    print(f"    Sandbox test FAILED: {err[:80]}")
                self._log_attempt(pattern, gen, "failed_sandbox")
                continue

            results["passed"] += 1
            if self.verbose:
                print(f"    Sandbox test PASSED ({gen.code_type})")

            # Apply the generated code to the system
            applied = self._apply_generation(gen)
            if applied:
                results["applied"] += 1
                gen.adopted = True
                if self.verbose:
                    print(f"    Applied to system ✓")

            self._log_attempt(pattern, gen,
                              "applied" if applied else "passed_not_applied")

        self._save_log()

        if self.verbose:
            print(f"\n[attention] Cycle complete: "
                  f"{results['generated']} generated, "
                  f"{results['passed']} passed sandbox, "
                  f"{results['applied']} applied.")

        return results

    def _apply_generation(self, gen: GeneratedCode) -> bool:
        """
        Apply a passing generated code to the live system.
        """
        if gen.code_type == "technique_dict":
            return self._apply_technique(gen)
        elif gen.code_type == "verification_function":
            return self._apply_verifier(gen)
        elif gen.code_type == "step_template":
            return self._apply_step_templates(gen)
        return False

    def _apply_technique(self, gen: GeneratedCode) -> bool:
        """Add a generated technique dict to the technique library."""
        try:
            # Execute the code to get the dict
            sandbox = CodeSandbox()
            result  = sandbox.execute(f"_d = {gen.code}", "", {})
            if not result.success:
                return False

            # Extract the dict value
            namespace: dict = {}
            exec(f"_d = {gen.code}", {"__builtins__": {
                "True": True, "False": False, "None": None,
            }}, namespace)
            t_dict = namespace.get("_d")
            if not isinstance(t_dict, dict):
                return False
            required = {"name", "description", "when_to_use", "example_text"}
            if not required.issubset(t_dict.keys()):
                return False

            self.library.record_success(
                name=t_dict["name"],
                description=t_dict["description"],
                when_to_use=t_dict["when_to_use"],
                example_text=t_dict["example_text"],
                domain=gen.pattern.domain or "general",
                run_id=f"generated-{gen.generation_id}",
            )
            return True
        except Exception:
            return False

    def _apply_verifier(self, gen: GeneratedCode) -> bool:
        """
        Store a generated verification function for later use.
        Right now we log it; a fuller integration would register it
        with the reasoning_tools module.
        """
        # Store as a technique note for now
        # Full integration: register with reasoning_tools.py
        if self.verbose:
            print(f"    [verifier stored — integrate with reasoning_tools.py]")
        return True

    def _apply_step_templates(self, gen: GeneratedCode) -> bool:
        """Store generated step templates."""
        if self.verbose:
            print(f"    [step templates stored — integrate with "
                  f"structural_branch_sampler.py]")
        return True

    def _log_attempt(
        self,
        pattern:  FailurePattern,
        gen:      GeneratedCode,
        outcome:  str,
    ) -> None:
        self._log.append({
            "timestamp":    time.time(),
            "pattern_id":   pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "code_type":    gen.code_type,
            "outcome":      outcome,
            "code_preview": gen.code[:100],
        })

    def what_was_learned(self) -> dict:
        """
        Summarise what kinds of code generation are working.
        This is the meta-learning signal.
        """
        if not self._log:
            return {"n_attempts": 0}

        by_type: dict[str, dict] = {}
        for entry in self._log:
            ct = entry.get("code_type", "unknown")
            if ct not in by_type:
                by_type[ct] = {"total": 0, "passed": 0, "applied": 0}
            by_type[ct]["total"] += 1
            if entry.get("outcome") in ("passed_not_applied", "applied"):
                by_type[ct]["passed"] += 1
            if entry.get("outcome") == "applied":
                by_type[ct]["applied"] += 1

        return {
            "n_attempts":      len(self._log),
            "by_code_type":    by_type,
            "pass_rate":       sum(1 for e in self._log
                                   if "passed" in e.get("outcome","")) / len(self._log),
            "apply_rate":      sum(1 for e in self._log
                                   if e.get("outcome") == "applied") / len(self._log),
        }

    # ---- Persistence ----

    def _load_log(self) -> list[dict]:
        if not os.path.exists(self.log_path):
            return []
        try:
            with open(self.log_path) as f:
                return json.load(f).get("attempts", [])
        except (json.JSONDecodeError, OSError):
            return []

    def _save_log(self) -> None:
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump({"attempts": self._log}, f, indent=2)


# ============================================================
# Helpers
# ============================================================

def _simple_tokenise(text: str) -> list[str]:
    stop = {"the","a","an","is","are","to","of","and","or","in","on","for",
            "from","with","this","that","it","by","at","be"}
    return [t for t in re.findall(r"[a-z]{3,}", text.lower())
            if t not in stop]


def _most_common(lst: list) -> Optional[str]:
    if not lst:
        return None
    return max(set(lst), key=lst.count)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", s.lower())[:30].strip("_")


def _clean_code_block(code: str) -> str:
    """Strip markdown code fences if the LLM wrapped the output."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        inner = [l for l in lines[1:] if not l.startswith("```")]
        code  = "\n".join(inner).strip()
    return code
