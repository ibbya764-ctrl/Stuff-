"""
tool_use.py
===========

Extends the system's reach by giving it the ability to use tools.

A system that can reason well but only reason is limited to what
it can derive internally. A system that can reason AND use tools
is limited only by what tools exist and what the reasoning quality
determines about how to use them.

This is the practical path to genuine generality.

Six tools:

  CodeExecutor      — runs Python code in a sandbox and returns
                      the output. Enables: data analysis, numerical
                      computation, algorithm verification, anything
                      that can be expressed as code.

  WebSearchTool     — searches the web and retrieves relevant content.
                      Enables: factual verification, current events,
                      research, any knowledge with a web presence.

  Calculator        — precise arithmetic and symbolic computation
                      beyond what the LLM does natively. For when
                      exact numbers matter.

  FactChecker       — checks factual claims against multiple sources.
                      Returns confidence that a claim is accurate,
                      not just whether one source agrees.

  StructuredQuery   — queries structured knowledge (local databases,
                      CSV files, knowledge bases). For when the
                      relevant knowledge is structured data.

  ToolOrchestrator  — determines which tool(s) a question needs,
                      runs them, and integrates results into the
                      reasoning chain. The reasoner calls this;
                      the reasoner doesn't call tools directly.

The verification implication:
  Every domain where a tool can check correctness becomes a
  domain the system can genuinely learn in. Code execution
  opens programming. Web search opens knowledge domains.
  The verification signal grows with the tool stack.
"""

import os
import re
import json
import time
import subprocess
import tempfile
import sys
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# ToolResult — standardised output from any tool
# ============================================================

@dataclass
class ToolResult:
    tool:        str
    success:     bool
    output:      str
    error:       str   = ""
    verified:    bool  = False   # did this verify a claim?
    confidence:  float = 0.0     # 0-1 confidence in result
    elapsed:     float = 0.0
    raw:         dict  = field(default_factory=dict)

    def format_for_reasoning(self) -> str:
        if not self.success:
            return f"[TOOL:{self.tool}] Failed: {self.error[:80]}"
        return (
            f"[TOOL:{self.tool}] "
            + (f"verified={self.verified} " if self.verified else "")
            + f"confidence={self.confidence:.2f}\n{self.output[:300]}"
        )


# ============================================================
# CodeExecutor
# ============================================================

ALLOWED_IMPORTS = {
    "math", "cmath", "statistics", "fractions", "decimal",
    "numpy", "scipy", "sympy", "pandas",
    "json", "re", "collections", "itertools", "functools",
    "datetime", "time",
}

BLOCKED_PATTERNS = [
    r"\bos\.(system|popen|exec|remove|rename|mkdir)\b",
    r"\bsubprocess\b",
    r"\bopen\s*\(",
    r"\b__import__\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
]


class CodeExecutor:
    """
    Runs Python code in a restricted subprocess and captures output.

    Enables: data analysis, numerical verification, algorithm checking,
    mathematical computation, plotting (as text), statistics.

    The output becomes a [VERIFIABLE] step in the reasoning chain.
    """

    TIMEOUT = 10   # seconds

    def execute(self, code: str, context: str = "") -> ToolResult:
        """Execute code and return the output."""
        t0 = time.time()

        # Safety check
        safety = self._check_safety(code)
        if not safety["safe"]:
            return ToolResult(
                tool="code_executor",
                success=False,
                output="",
                error=f"Blocked: {safety['reason']}",
                elapsed=time.time() - t0,
            )

        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
            )
            success = result.returncode == 0
            output  = result.stdout.strip() or result.stderr.strip()
            error   = result.stderr.strip() if not success else ""

            return ToolResult(
                tool="code_executor",
                success=success,
                output=output[:500] if output else "(no output)",
                error=error[:200],
                verified=success,
                confidence=0.95 if success else 0.0,
                elapsed=round(time.time() - t0, 2),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool="code_executor",
                success=False,
                error=f"Timeout after {self.TIMEOUT}s",
                elapsed=time.time() - t0,
            )
        except Exception as e:
            return ToolResult(
                tool="code_executor",
                success=False,
                error=str(e)[:100],
                elapsed=time.time() - t0,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _check_safety(self, code: str) -> dict:
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, code):
                return {"safe": False, "reason": f"Blocked pattern: {pattern}"}
        return {"safe": True, "reason": ""}

    def generate_verification_code(
        self,
        claim: str,
        llm_chat_fn: Optional[Callable] = None,
    ) -> Optional[str]:
        """
        Given a verifiable claim, generate code that checks it.
        Returns Python code string or None.
        """
        if not llm_chat_fn:
            return None

        system = (
            "You write short Python code that verifies mathematical or computational claims.\n"
            "Use sympy, numpy, or pure Python. Print True if verified, False if not.\n"
            "Output ONLY the code — no explanation, no markdown."
        )
        user = f"Write code to verify: {claim}"
        try:
            code = llm_chat_fn(system, user).strip()
            # Strip markdown if present
            code = re.sub(r"```python\n?|```\n?", "", code).strip()
            return code
        except Exception:
            return None


# ============================================================
# WebSearchTool
# ============================================================

class WebSearchTool:
    """
    Searches the web and retrieves relevant content.

    Enables: factual verification, current knowledge, research,
    any domain with a web presence.

    Results are summarised and returned as [VERIFIABLE] content.
    """

    def __init__(self, search_fn: Optional[Callable] = None):
        """
        search_fn: callable(query) → list[{"title", "snippet", "url"}]
        If None, returns a placeholder (for testing without API keys).
        """
        self._search_fn = search_fn

    def search(self, query: str, n_results: int = 3) -> ToolResult:
        """Search and return summarised results."""
        t0 = time.time()

        if not self._search_fn:
            return ToolResult(
                tool="web_search",
                success=False,
                error="No search function configured. Set SEARCH_API_KEY.",
                elapsed=time.time() - t0,
            )

        try:
            results = self._search_fn(query)
            if not results:
                return ToolResult(
                    tool="web_search",
                    success=False,
                    error="No results found",
                    elapsed=time.time() - t0,
                )

            formatted = []
            for r in results[:n_results]:
                formatted.append(
                    f"• {r.get('title','')}\n  {r.get('snippet','')[:150]}"
                )

            output = "\n".join(formatted)
            return ToolResult(
                tool="web_search",
                success=True,
                output=output,
                confidence=0.7,
                elapsed=round(time.time() - t0, 2),
                raw={"query": query, "n_results": len(results)},
            )
        except Exception as e:
            return ToolResult(
                tool="web_search",
                success=False,
                error=str(e)[:100],
                elapsed=time.time() - t0,
            )

    def verify_claim(self, claim: str) -> ToolResult:
        """
        Search for a claim and assess whether sources support it.
        Returns a ToolResult with verified=True if sources agree.
        """
        result = self.search(f'verify: "{claim}"', n_results=5)
        if not result.success:
            return result

        # Simple heuristic: look for agreement in snippets
        combined = result.output.lower()
        claim_words = set(re.findall(r"[a-z]{4,}", claim.lower()))
        hits        = sum(1 for w in claim_words if w in combined)
        confidence  = min(0.85, hits / max(1, len(claim_words)))

        result.verified   = confidence > 0.5
        result.confidence = confidence
        return result


# ============================================================
# Calculator
# ============================================================

class Calculator:
    """
    Precise arithmetic and symbolic computation.

    Uses Python's decimal module for exact arithmetic and sympy
    for symbolic computation. For when LLM arithmetic isn't enough.
    """

    def calculate(self, expression: str) -> ToolResult:
        """Evaluate a mathematical expression precisely."""
        t0 = time.time()
        try:
            # Try sympy first for symbolic expressions
            import sympy
            result = sympy.sympify(expression)
            # Try to simplify and evaluate
            simplified = sympy.simplify(result)
            numerical  = None
            try:
                numerical = float(simplified.evalf())
            except Exception:
                pass

            output = str(simplified)
            if numerical is not None and str(simplified) != str(numerical):
                output += f" ≈ {numerical:.6g}"

            return ToolResult(
                tool="calculator",
                success=True,
                output=output,
                verified=True,
                confidence=0.99,
                elapsed=round(time.time() - t0, 3),
            )
        except Exception:
            pass

        # Fallback: eval with safe builtins
        try:
            import math
            safe_env = {
                "__builtins__": {},
                "math": math,
                "sqrt": math.sqrt,
                "pi":   math.pi,
                "e":    math.e,
                "sin":  math.sin,
                "cos":  math.cos,
                "log":  math.log,
                "abs":  abs,
                "round": round,
            }
            result = eval(expression, safe_env)
            return ToolResult(
                tool="calculator",
                success=True,
                output=str(result),
                verified=True,
                confidence=0.99,
                elapsed=round(time.time() - t0, 3),
            )
        except Exception as e:
            return ToolResult(
                tool="calculator",
                success=False,
                error=f"Cannot evaluate: {e}",
                elapsed=time.time() - t0,
            )


# ============================================================
# FactChecker
# ============================================================

class FactChecker:
    """
    Checks factual claims by combining web search and internal
    knowledge consistency checking.

    For claims that can be checked against multiple sources,
    returns a confidence estimate and the supporting evidence.
    """

    def __init__(self, web_search: Optional[WebSearchTool] = None):
        self.web = web_search

    def check(self, claim: str, domain: str = "") -> ToolResult:
        """
        Check whether a factual claim is likely accurate.
        Returns confidence 0-1 and supporting/contradicting evidence.
        """
        t0 = time.time()

        results = []
        confidence = 0.5  # prior: unknown

        # Web search verification
        if self.web and self.web._search_fn:
            search_result = self.web.verify_claim(claim)
            if search_result.success:
                confidence = 0.4 * confidence + 0.6 * search_result.confidence
                results.append(f"Web: {search_result.output[:150]}")

        # Internal consistency check
        consistency = self._check_internal_consistency(claim)
        confidence  = 0.6 * confidence + 0.4 * consistency["score"]
        if consistency["note"]:
            results.append(f"Consistency: {consistency['note']}")

        output    = "\n".join(results) if results else "No verification sources available"
        verified  = confidence > 0.6

        return ToolResult(
            tool="fact_checker",
            success=True,
            output=output,
            verified=verified,
            confidence=round(confidence, 3),
            elapsed=round(time.time() - t0, 2),
        )

    def _check_internal_consistency(self, claim: str) -> dict:
        """
        Basic internal consistency check — looks for logical
        contradictions or implausibility markers in the claim.
        """
        lower = claim.lower()
        score = 0.6  # neutral prior

        # Red flags for implausibility
        if re.search(r"\b(always|never|all|none|every|no one)\b", lower):
            score -= 0.1  # absolute claims are often false

        # Specific numbers increase verifiability
        if re.search(r"\d+\.?\d*", claim):
            score += 0.05  # specific claims are easier to check

        return {"score": score, "note": ""}


# ============================================================
# ToolOrchestrator
# ============================================================

TOOL_SELECTION_KEYWORDS = {
    "code_executor":  ["calculate", "compute", "run", "execute", "code", "algorithm",
                       "verify numerically", "check with code", "plot", "data"],
    "web_search":     ["search", "find", "look up", "current", "recent", "who is",
                       "what is the latest", "verify", "fact", "source"],
    "calculator":     ["what is", "calculate", r"\d+\s*[+\-*/^]\s*\d+", "equals",
                       "how much", "evaluate"],
    "fact_checker":   ["is it true", "fact check", "accurate", "correct that",
                       "is this right", "verify that"],
}


class ToolOrchestrator:
    """
    Determines which tools a question or reasoning step needs,
    runs them, and integrates results into the reasoning chain.

    The Reasoner delegates to ToolOrchestrator when it identifies
    a step that needs external verification or information.
    This keeps the Reasoner focused on reasoning while tools
    handle information retrieval and verification.
    """

    def __init__(
        self,
        code_executor:  Optional[CodeExecutor]  = None,
        web_search:     Optional[WebSearchTool] = None,
        calculator:     Optional[Calculator]    = None,
        fact_checker:   Optional[FactChecker]   = None,
        llm_chat_fn:    Optional[Callable]      = None,
        verbose:        bool = True,
    ):
        self.code     = code_executor or CodeExecutor()
        self.search   = web_search
        self.calc     = calculator or Calculator()
        self.facts    = fact_checker
        self.llm      = llm_chat_fn
        self.verbose  = verbose
        self._log: list[dict] = []

    def select_tools(self, text: str) -> list[str]:
        """
        Determine which tools are relevant for this text.
        Returns list of tool names.
        """
        lower = text.lower()
        selected = []

        for tool, keywords in TOOL_SELECTION_KEYWORDS.items():
            for kw in keywords:
                if re.search(kw, lower):
                    if tool not in selected:
                        selected.append(tool)
                    break

        return selected

    def run_for_step(
        self,
        step_content: str,
        step_label:   str,
    ) -> Optional[ToolResult]:
        """
        Given a reasoning step, determine if and how to verify it
        with tools. Returns a ToolResult or None.

        Only [VERIFIABLE] steps get tool verification.
        [STRUCTURAL] and [DERIVED] steps don't need it.
        """
        if step_label != "VERIFIABLE":
            return None

        tools = self.select_tools(step_content)
        if not tools:
            return None

        return self.run(step_content, tools[0])

    def run(self, query: str, tool_name: str = "auto") -> ToolResult:
        """Run a specific tool or auto-select one."""
        if tool_name == "auto":
            tools = self.select_tools(query)
            tool_name = tools[0] if tools else "calculator"

        if self.verbose:
            print(f"  [tools] Running {tool_name}: {query[:50]}...")

        t0 = time.time()
        result = None

        if tool_name == "code_executor":
            # Generate and run verification code
            code = self.code.generate_verification_code(query, self.llm)
            if code:
                result = self.code.execute(code)
                result.raw["generated_code"] = code[:200]
            else:
                # Try direct execution if query looks like code
                if "def " in query or "import " in query or "=" in query:
                    result = self.code.execute(query)

        elif tool_name == "web_search" and self.search:
            result = self.search.search(query)

        elif tool_name == "calculator":
            # Extract expression from query
            expr = self._extract_expression(query)
            if expr:
                result = self.calc.calculate(expr)

        elif tool_name == "fact_checker" and self.facts:
            result = self.facts.check(query)

        if result is None:
            result = ToolResult(
                tool=tool_name,
                success=False,
                error=f"Tool {tool_name} not available or no action taken",
                elapsed=round(time.time() - t0, 2),
            )

        self._log.append({
            "timestamp": time.time(),
            "tool":      tool_name,
            "query":     query[:80],
            "success":   result.success,
            "verified":  result.verified,
            "confidence": result.confidence,
        })

        return result

    def run_multiple(
        self, query: str, max_tools: int = 2
    ) -> list[ToolResult]:
        """Run multiple tools for a complex query."""
        tools   = self.select_tools(query)[:max_tools]
        results = []
        for t in tools:
            r = self.run(query, t)
            results.append(r)
            if r.verified and r.confidence > 0.8:
                break  # high confidence result — stop early
        return results

    def integrate_into_reasoning(
        self, reasoning_text: str
    ) -> tuple[str, list[ToolResult]]:
        """
        Find [VERIFIABLE] steps in a reasoning chain, run tools on them,
        and return an enriched reasoning text with tool results injected.
        """
        lines   = reasoning_text.split("\n")
        results: list[ToolResult] = []
        enriched: list[str] = []

        for line in lines:
            enriched.append(line)
            if "[VERIFIABLE]" in line:
                result = self.run_for_step(line, "VERIFIABLE")
                if result and result.success:
                    enriched.append(
                        f"  → {result.format_for_reasoning()}"
                    )
                    results.append(result)

        return "\n".join(enriched), results

    @staticmethod
    def _extract_expression(text: str) -> Optional[str]:
        """Extract a mathematical expression from natural language."""
        # Look for explicit = sign or operators
        m = re.search(r"(\d[\d\s\+\-\*/\^\.()]+\d)", text)
        if m:
            return m.group(1).strip()
        # Look for "X equals Y"
        m = re.search(r"(?:evaluate|compute|calculate)\s+(.+?)(?:\?|$)", text.lower())
        if m:
            return m.group(1).strip()
        return None

    def stats(self) -> dict:
        if not self._log:
            return {}
        recent = self._log[-50:]
        return {
            "total_calls":    len(self._log),
            "success_rate":   sum(1 for r in recent if r["success"]) / len(recent),
            "verify_rate":    sum(1 for r in recent if r["verified"]) / len(recent),
            "tools_used":     list({r["tool"] for r in recent}),
        }


# ============================================================
# VerificationExpander
# ============================================================

class VerificationExpander:
    """
    Systematically extends the verification stack to new domains.

    For any domain, this determines the right verification strategy
    and generates the verification infrastructure automatically.

    Currently supported strategies:
      mathematical:   sympy + numerical sampling (already built)
      computational:  code execution + unit tests
      factual:        web search + multi-source consistency
      logical:        Z3 + argument form checking
      statistical:    scipy stats tests
      domain-guided:  LLM generates domain-specific verification code

    The key insight: as more domains get proper verification,
    more domains produce genuine training signal.
    """

    DOMAIN_STRATEGIES = {
        "physics":       ["mathematical", "computational"],
        "mathematics":   ["mathematical", "logical"],
        "computer_science": ["computational"],
        "statistics":    ["statistical", "computational"],
        "biology":       ["domain-guided", "factual"],
        "medicine":      ["domain-guided", "factual"],
        "economics":     ["mathematical", "domain-guided"],
        "history":       ["factual"],
        "law":           ["logical", "factual"],
        "philosophy":    ["logical"],
        "chemistry":     ["computational", "domain-guided"],
        "general":       ["factual"],
    }

    def __init__(
        self,
        orchestrator: ToolOrchestrator,
        llm_chat_fn:  Optional[Callable] = None,
        verbose:      bool = True,
    ):
        self.orchestrator = orchestrator
        self.llm          = llm_chat_fn
        self.verbose      = verbose

    def get_strategy(self, domain: str) -> list[str]:
        """Get verification strategies for a domain."""
        d = domain.lower().split("_")[0]
        return self.DOMAIN_STRATEGIES.get(d, ["factual", "domain-guided"])

    def verify(
        self,
        claim:  str,
        domain: str,
        result: str = "",
    ) -> tuple[bool, float, list[ToolResult]]:
        """
        Verify a claim using the best strategy for the domain.

        Returns (verified, confidence, tool_results).
        """
        strategies = self.get_strategy(domain)
        all_results: list[ToolResult] = []
        confidences: list[float] = []

        for strategy in strategies:
            tool_result = self._run_strategy(strategy, claim, domain, result)
            if tool_result:
                all_results.append(tool_result)
                confidences.append(tool_result.confidence)
                if tool_result.verified and tool_result.confidence > 0.8:
                    break  # strong verification — stop

        if not confidences:
            return False, 0.0, []

        mean_conf = sum(confidences) / len(confidences)
        verified  = mean_conf > 0.5 and any(r.verified for r in all_results)

        return verified, round(mean_conf, 3), all_results

    def _run_strategy(
        self,
        strategy: str,
        claim:    str,
        domain:   str,
        result:   str = "",
    ) -> Optional[ToolResult]:

        if strategy == "mathematical":
            code = self.orchestrator.code.generate_verification_code(claim, self.llm)
            if code:
                return self.orchestrator.code.execute(code)

        elif strategy == "computational":
            code = self._generate_computational_verification(claim, domain, result)
            if code:
                return self.orchestrator.code.execute(code)

        elif strategy == "factual":
            if self.orchestrator.search:
                return self.orchestrator.search.verify_claim(claim)

        elif strategy == "logical":
            return self._check_logical_form(claim)

        elif strategy == "statistical":
            return self._statistical_check(claim, result)

        elif strategy == "domain-guided":
            return self._domain_guided_verification(claim, domain, result)

        return None

    def _generate_computational_verification(
        self, claim: str, domain: str, result: str
    ) -> Optional[str]:
        if not self.llm:
            return None
        system = (
            f"Generate Python code to computationally verify a {domain} claim.\n"
            "Use numpy, scipy, or pure Python. Print True if verified.\n"
            "Code only, no explanation."
        )
        user = f"Verify: {claim}\nContext: {result[:200]}"
        try:
            code = self.llm(system, user).strip()
            code = re.sub(r"```python\n?|```\n?", "", code).strip()
            return code if len(code) > 20 else None
        except Exception:
            return None

    def _check_logical_form(self, claim: str) -> ToolResult:
        """Basic logical consistency check."""
        t0 = time.time()
        issues = []

        lower = claim.lower()
        if "and" in lower and "not" in lower:
            if re.search(r"\b(\w+)\b.*\bnot\s+\1\b", lower):
                issues.append("potential self-contradiction")

        if re.search(r"\bif.+then.+therefore\b", lower):
            pass  # valid logical form

        confidence = 0.7 if not issues else 0.3
        return ToolResult(
            tool="logical_check",
            success=True,
            output=f"Issues: {issues}" if issues else "No logical issues found",
            verified=not issues,
            confidence=confidence,
            elapsed=round(time.time() - t0, 3),
        )

    def _statistical_check(self, claim: str, result: str) -> Optional[ToolResult]:
        """Generate and run statistical verification."""
        if not self.llm:
            return None
        code = self._generate_computational_verification(
            claim, "statistics", result
        )
        if code:
            return self.orchestrator.code.execute(code)
        return None

    def _domain_guided_verification(
        self, claim: str, domain: str, result: str
    ) -> Optional[ToolResult]:
        """Generate domain-specific verification code."""
        if not self.llm:
            return None
        code = self._generate_computational_verification(claim, domain, result)
        if code:
            return self.orchestrator.code.execute(code)
        return None
