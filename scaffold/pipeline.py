"""
pipeline.py
===========

Complete orchestration of every module built in this project.

How to use in your Colab notebook:
------------------------------------
1. Upload all .py files to /content/ via the Colab file browser.
2. Add this at the top of a new cell (after your existing model/llm_chat setup):

    import sys
    sys.path.append('/content')
    from pipeline import ReasoningPipeline, PipelineConfig

3. Create a pipeline instance once:

    config = PipelineConfig(
        domain_name="physics_mond",
        parameter_ranges={
            "mT":    [-2.0, 2.0],
            "lam12": [0.0,  1.0],
            "x1":    [-1.0, 1.0],
            "y1":    [-1.0, 1.0],
        },
        # Optional: point at Drive for persistence across sessions
        obligation_store_path="/content/drive/MyDrive/obligation_store.json",
        technique_library_path="/content/drive/MyDrive/technique_library.json",
    )
    pipeline = ReasoningPipeline(config, llm_chat_fn=llm_chat)

4. Run on any question:

    result = pipeline.run(test_question)

5. The result dict contains everything; print_report(result) prints a
   human-readable summary.

The pipeline calls your existing llm_chat and, if provided, your existing
auto_score_physics_branch_v3. Everything else it handles internally.
"""

import sys
import json
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

# ============================================================
# Lazy-import all project modules — fail gracefully per module
# ============================================================

def _try_import(module_name):
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError as e:
        print(f"  [pipeline] Warning: could not import {module_name}: {e}")
        return None


# ============================================================
# Configuration
# ============================================================

@dataclass
class PipelineConfig:
    """All tunable settings for a pipeline run."""

    domain_name: str = "physics_mond"

    # Parameter ranges for counterexample search.
    # Keys must match variable names in branch steps.
    parameter_ranges: dict = field(default_factory=lambda: {
        "mT":    [-2.0, 2.0],
        "lam12": [ 0.0, 1.0],
        "x1":    [-1.0, 1.0],
        "y1":    [-1.0, 1.0],
    })

    # Persistent storage paths.
    # Set to a Google Drive path (/content/drive/MyDrive/...) to survive
    # Colab runtime resets.
    obligation_store_path: str  = "./obligation_store.json"
    technique_library_path: str = "./technique_library.json"

    # How many branches to generate before counterexample expansion.
    n_seed_branches: int = 3

    # How many new branches counterexample expansion may add per original.
    max_new_branches_per_original: int = 2

    # How many persistent gaps / techniques to inject into the prompt.
    n_context_items: int = 3

    # Whether to run tool-based verification on the top branch.
    verify_top_branch: bool = True
    run_self_audit: bool = True

    # Whether to run counterexample expansion.
    run_counterexample_expansion: bool = True

    # Whether to extract new techniques at the end of the run.
    extract_techniques: bool = True

    # Verbose printing during the run.
    verbose: bool = True


# ============================================================
# Branch generation (uses existing llm_chat)
# ============================================================

BRANCH_GEN_SYSTEM = (
    "You are a physics reasoning assistant operating in domain: {domain}.\n"
    "{persistent_gaps}\n"
    "{techniques}\n\n"
    "Generate {n} distinct reasoning branches for the question below.\n"
    "Each branch must be a JSON object with exactly these keys:\n"
    "  name, method, assumptions (list), steps (list), "
    "candidate_result, notes (list)\n\n"
    "Return a JSON array of {n} branch objects and nothing else.\n"
    "Branches should be genuinely distinct — different methods, not the "
    "same approach with different wording.\n"
    "At least one branch should be a DIAGNOSTIC branch that honestly "
    "identifies the hardest obstacles without trying to derive past them.\n"
    "{domain_guidance}"
)


def _generate_branches(
    question: str,
    domain,
    llm_chat_fn: Callable,
    n: int,
    persistent_gaps_fragment: str,
    techniques_fragment: str,
    verbose: bool = True,
) -> list[dict]:
    """Ask the LLM to generate `n` reasoning branches."""
    system = BRANCH_GEN_SYSTEM.format(
        domain=domain.name if domain else "general",
        n=n,
        persistent_gaps=persistent_gaps_fragment or "",
        techniques=techniques_fragment or "",
        domain_guidance=domain.branch_generation_guidance if domain else "",
    )
    user = f"Question: {question}"
    if verbose:
        print(f"  [pipeline] Generating {n} seed branches...")

    try:
        response = llm_chat_fn(system, user)
    except Exception as e:
        print(f"  [pipeline] Branch generation failed: {e}")
        return []

    return _parse_branches(response, verbose=verbose)


def _parse_branches(response: str, verbose: bool = False) -> list[dict]:
    """Robustly extract a list of branch dicts from LLM output."""
    # Try direct JSON parse
    try:
        obj = json.loads(response.strip())
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array inside the text
    match = re.search(r"\[[\s\S]*\]", response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to find individual JSON objects
    branches = []
    for match in re.finditer(r"\{[\s\S]*?\}", response):
        try:
            b = json.loads(match.group(0))
            if "name" in b and "steps" in b:
                branches.append(b)
        except json.JSONDecodeError:
            continue

    if verbose and not branches:
        print("  [pipeline] Warning: could not parse any branches from LLM output.")
    return branches


# ============================================================
# Scoring integration
# ============================================================


class _BranchNamespace:
    """Wraps a branch dict as an object with dot-notation access.
    Needed for compatibility with existing scorers that use branch.assumptions
    rather than branch['assumptions']."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)
        self._dict = d
    def get(self, key, default=None):
        return self._dict.get(key, default)
    def __getitem__(self, key):
        return self._dict[key]
    def __contains__(self, key):
        return key in self._dict
    def __repr__(self):
        return f"BranchNamespace({self._dict.get('name', '?')})"

def _apply_existing_scorer(
    branch: dict,
    existing_score_fn: Optional[Callable],
    verbose: bool = False,
) -> float:
    """Apply the existing auto_score_physics_branch_v3 if available."""
    if existing_score_fn is None:
        return 0.0
    try:
        result = existing_score_fn(_BranchNamespace(branch))
        if isinstance(result, dict):
            return float(result.get("total_score", 0.0))
        return float(result)
    except Exception as e:
        if verbose:
            print(f"  [pipeline] Existing scorer failed on {branch.get('name')}: {e}")
        return 0.0


# ============================================================
# Main Pipeline class
# ============================================================

class ReasoningPipeline:
    """
    Orchestrates all modules into a single run call.

    Pass your existing `llm_chat` function at construction time.
    Optionally pass `existing_score_fn` (your auto_score_physics_branch_v3)
    and it will be called alongside the new tool-based verification.
    """

    def __init__(
        self,
        config: PipelineConfig,
        llm_chat_fn: Callable,
        existing_score_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.llm_chat = llm_chat_fn
        self.existing_score_fn = existing_score_fn
        self._load_modules()

    def _load_modules(self):
        cfg = self.config
        if self.config.verbose:
            print("[pipeline] Loading modules...")

        # Core tools
        self.rt = _try_import("reasoning_tools")
        rte  = _try_import("reasoning_tools_extended")  # registers extra tools
        _try_import("web_search_tool")                   # registers web_search tool
        self.domain_module = _try_import("domains")
        self.cross_branch_module = _try_import("cross_branch")
        self.obligation_module = _try_import("obligation_store")
        self.counterexample_module = _try_import("counterexample_branches")
        self.technique_module   = _try_import("technique_library")
        self.router_module      = _try_import("router")
        self.audit_module       = None   # audit functions defined inline below
        self.verification_module = self.rt

        # Domain object
        self.domain = None
        if self.domain_module:
            try:
                self.domain = self.domain_module.get_domain(cfg.domain_name)
            except KeyError:
                print(f"  [pipeline] Warning: domain '{cfg.domain_name}' not found; "
                      "using physics_mond as default.")
                self.domain = self.domain_module.PHYSICS_MOND

        # Persistent stores
        self.obligation_store = None
        if self.obligation_module:
            self.obligation_store = self.obligation_module.ObligationStore(
                cfg.obligation_store_path
            )

        self.technique_library = None
        if self.technique_module:
            self.technique_library = self.technique_module.TechniqueLibrary(
                cfg.technique_library_path
            )

        loaded = [
            name for name, mod in {
                "reasoning_tools": self.rt,
                "domains": self.domain_module,
                "cross_branch": self.cross_branch_module,
                "obligation_store": self.obligation_module,
                "counterexample_branches": self.counterexample_module,
                "technique_library": self.technique_module,
                "router":            self.router_module,
            }.items() if mod is not None
        ]
        if self.config.verbose:
            print(f"  [pipeline] Loaded: {', '.join(loaded)}")

    # ----------------------------------------------------------
    # Main run
    # ----------------------------------------------------------

    def run(self, question: str, context: str = "") -> dict:
        """
        Run the full pipeline on a question.

        Parameters
        ----------
        question : the reasoning question to address
        context  : optional extra context string (passed to branch gen)

        Returns
        -------
        A result dict. Call print_report(result) to see a readable summary.
        """
        cfg = self.config
        run_id = f"run-{int(time.time() * 1000)}"
        t0 = time.time()

        # Auto-route: detect domain from question if router available
        if self.router_module and self.domain is not None:
            route = self.router_module.route_question(
                question, llm_chat_fn=self.llm_chat,
                verbose=cfg.verbose
            )
            # Only override domain if router is confident and domain differs
            if route.confidence > 0.6 and route.domain_name != cfg.domain_name:
                try:
                    self.domain = self.domain_module.get_domain(route.domain_name)
                    if cfg.verbose:
                        print(f"[router] Auto-routed to domain: {route.domain_name} "
                              f"({route.confidence:.0%} confidence)")
                except Exception:
                    pass
            # Update branch count suggestion
            cfg.n_seed_branches = max(
                cfg.n_seed_branches,
                route.suggested_n_branches
            )

        if cfg.verbose:
            print(f"\n{'='*60}")
            print(f"[pipeline] Run: {run_id}")
            print(f"[pipeline] Question: {question[:100]}...")
            print(f"{'='*60}")

        result = {
            "run_id":    run_id,
            "question":  question,
            "domain":    cfg.domain_name,
            "stages":    {},
            "errors":    [],
        }

        # ---- Stage 1: Pull context from persistent memory ----
        result["stages"]["context"] = self._stage_context(question)

        # ---- Stage 2: Generate seed branches ----
        ctx = result["stages"]["context"]
        branches = self._stage_generate(
            question, context,
            ctx["persistent_gaps_fragment"],
            ctx["techniques_fragment"],
        )
        result["stages"]["branch_generation"] = {
            "n_branches": len(branches),
            "branch_names": [b.get("name") for b in branches],
        }
        if not branches:
            result["errors"].append("No branches generated — check LLM output.")
            return result

        # ---- Stage 3: Score initial branches with existing scorer ----
        branches = self._stage_score_existing(branches)

        # ---- Stage 4: Counterexample expansion ----
        if cfg.run_counterexample_expansion and self.counterexample_module:
            branches, falsification_reports = self._stage_counterexample(branches)
            # Ensure all new branches have a total_score
            for b in branches:
                if b.get("total_score") is None:
                    b["total_score"] = _apply_existing_scorer(
                        b, self.existing_score_fn, cfg.verbose
                    )
            result["stages"]["counterexample_expansion"] = {
                "n_new_branches": len(branches) - result["stages"]["branch_generation"]["n_branches"],
                "reports": falsification_reports,
            }
        else:
            result["stages"]["counterexample_expansion"] = {"skipped": True}

        # ---- Stage 5: Tool-based verification on top branches ----
        if cfg.verify_top_branch and self.rt:
            verification_results = self._stage_verify(branches)
            result["stages"]["verification"] = verification_results
        else:
            result["stages"]["verification"] = {"skipped": True}

        # ---- Stage 6: Cross-branch obligation resolution ----
        resolution = {}
        if self.cross_branch_module and self.domain:
            resolution = self._stage_cross_branch(branches)
            result["stages"]["cross_branch"] = {
                "n_obligations_pooled": len(resolution.get("obligations", [])),
                "n_global_gaps":        len(resolution.get("global_gaps", [])),
            }
            # Apply cross-branch score adjustments
            adjustments = self.cross_branch_module.cross_branch_score_adjustments(resolution)
            for b in branches:
                name = b.get("name", "")
                if name in adjustments:
                    b["total_score"] = b.get("total_score", 0.0) + adjustments[name]
        else:
            result["stages"]["cross_branch"] = {"skipped": True}

        # ---- Stage 7: Select best branch ----
        branches_sorted = sorted(
            branches,
            key=lambda b: b.get("total_score", 0.0),
            reverse=True,
        )
        selected = branches_sorted[0] if branches_sorted else None

        # ---- Stage 8: Record obligations to persistent store ----
        if self.obligation_store and resolution:
            self.obligation_store.record_run(
                run_id=run_id,
                domain=cfg.domain_name,
                question=question,
                resolution=resolution,
            )
            result["stages"]["obligation_store"] = {
                "recorded": True,
                "store_summary": self.obligation_store.summary(),
            }

        # ---- Stage 9: Extract and store new techniques ----
        new_technique_ids = []
        if cfg.extract_techniques and self.technique_library and resolution:
            new_technique_ids = self._stage_extract_techniques(
                run_id, question, branches, resolution
            )
            result["stages"]["technique_library"] = {
                "new_techniques": len(new_technique_ids),
                "library_summary": self.technique_library.summary(),
            }

        # ---- Stage 10: Self-audit (adaptive) ----
        audit_findings = []
        if cfg.get("run_self_audit", True) if hasattr(cfg, "get") else getattr(cfg, "run_self_audit", True):
            if selected and self.obligation_store:
                try:
                    audit_findings = self._stage_self_audit(
                        selected, branches, resolution, run_id
                    )
                    result["stages"]["self_audit"] = {
                        "n_findings": len(audit_findings),
                        "findings": audit_findings[:5],
                    }
                except Exception as e:
                    if cfg.verbose:
                        print(f"  [pipeline] Self-audit error: {e}")

        # ---- Assemble final result ----
        elapsed = round(time.time() - t0, 1)
        result.update({
            "all_branches":          branches,
            "selected_branch":       selected,
            "resolution":            resolution,
            "global_gaps":           resolution.get("global_gaps", []),
            "persistent_gaps":       ctx.get("persistent_gaps", []),
            "new_technique_ids":     new_technique_ids,
            "elapsed_seconds":       elapsed,
        })

        if cfg.verbose:
            self._print_run_summary(result)

        return result

    # ----------------------------------------------------------
    # Stage implementations
    # ----------------------------------------------------------

    def _stage_context(self, question: str) -> dict:
        """Pull persistent gaps and relevant techniques for prompt seeding."""
        gaps_fragment = ""
        techniques_fragment = ""
        persistent_gaps = []
        relevant_techniques = []

        if self.obligation_store:
            persistent_gaps = self.obligation_store.query_persistent_gaps(
                top_n=self.config.n_context_items
            )
            gaps_fragment = self.obligation_store.get_persistent_gaps_prompt_fragment(
                top_n=self.config.n_context_items
            )
            if gaps_fragment and self.config.verbose:
                print(f"  [pipeline] Injecting {len(persistent_gaps)} persistent gaps into prompt.")

        if self.technique_library:
            relevant_techniques = self.technique_library.find_relevant_techniques(
                question, self.config.domain_name,
                top_n=self.config.n_context_items,
            )
            techniques_fragment = self.technique_library.format_for_prompt(
                relevant_techniques
            )
            if techniques_fragment and self.config.verbose:
                print(f"  [pipeline] Injecting {len(relevant_techniques)} relevant techniques.")

        return {
            "persistent_gaps":        [g.to_dict() if hasattr(g, 'to_dict') else g
                                       for g in persistent_gaps],
            "relevant_techniques":    [t.to_dict() if hasattr(t, 'to_dict') else t
                                       for t in relevant_techniques],
            "persistent_gaps_fragment":   gaps_fragment,
            "techniques_fragment":        techniques_fragment,
        }

    def _stage_generate(
        self, question: str, context: str,
        gaps_fragment: str, techniques_fragment: str,
    ) -> list[dict]:
        full_question = f"{question}\n\nContext: {context}" if context else question
        return _generate_branches(
            question=full_question,
            domain=self.domain,
            llm_chat_fn=self.llm_chat,
            n=self.config.n_seed_branches,
            persistent_gaps_fragment=gaps_fragment,
            techniques_fragment=techniques_fragment,
            verbose=self.config.verbose,
        )

    def _stage_score_existing(self, branches: list[dict]) -> list[dict]:
        if not self.existing_score_fn:
            for b in branches:
                b.setdefault("total_score", 0.0)
            return branches
        if self.config.verbose:
            print(f"  [pipeline] Scoring {len(branches)} branches with existing scorer...")
        for b in branches:
            score = _apply_existing_scorer(b, self.existing_score_fn,
                                           self.config.verbose)
            b["total_score"] = score
        return branches

    def _stage_counterexample(
        self, branches: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        if self.config.verbose:
            print(f"  [pipeline] Running counterexample expansion on "
                  f"{len(branches)} branches...")
        try:
            expansion = self.counterexample_module.expand_branches_via_counterexamples(
                branches=branches,
                parameter_ranges=self.config.parameter_ranges,
                llm_chat_fn=self.llm_chat,
                domain=self.domain,
                max_new_branches_per_original=self.config.max_new_branches_per_original,
            )
            new_count = len(expansion.get("new_branches", []))
            if self.config.verbose and new_count:
                print(f"  [pipeline] Added {new_count} counterexample-driven branches.")
            return expansion["all_branches"], expansion.get("falsification_reports", [])
        except Exception as e:
            self.config.verbose and print(f"  [pipeline] Counterexample expansion error: {e}")
            return branches, []

    def _stage_verify(self, branches: list[dict]) -> dict:
        """Verify the top 2 branches with the tool layer."""
        branches_sorted = sorted(
            branches, key=lambda b: b.get("total_score", 0.0), reverse=True
        )
        top = branches_sorted[:2]
        results = {}
        for b in top:
            if self.config.verbose:
                print(f"  [pipeline] Verifying branch: {b.get('name')}...")
            try:
                report = self.rt.verify_branch_with_tools(b, self.llm_chat)
                score_adj = self.rt.tool_verification_score(report)
                b["total_score"] = b.get("total_score", 0.0) + score_adj
                b["verification_report"] = report
                results[b.get("name")] = {
                    "verified": report.get("verdict", {}).get("verified"),
                    "score_adjustment": score_adj,
                    "n_tool_calls": report.get("n_calls", 0),
                }
            except Exception as e:
                results[b.get("name")] = {"error": str(e)}
        return results

    def _stage_cross_branch(self, branches: list[dict]) -> dict:
        if self.config.verbose:
            print(f"  [pipeline] Running cross-branch obligation resolution "
                  f"on {len(branches)} branches...")
        try:
            return self.cross_branch_module.cross_branch_resolve(
                branches, self.domain, self.llm_chat
            )
        except Exception as e:
            self.config.verbose and print(f"  [pipeline] Cross-branch error: {e}")
            return {"obligations": [], "global_gaps": [], "branch_responses": {},
                    "discharge_matrix": {}}

    def _stage_extract_techniques(
        self, run_id: str, question: str,
        branches: list[dict], resolution: dict,
    ) -> list[str]:
        if self.config.verbose:
            print("  [pipeline] Extracting techniques from this run...")
        try:
            return self.technique_library.extract_techniques_from_run(
                run_id=run_id,
                domain=self.config.domain_name,
                question=question,
                branches=branches,
                resolution=resolution,
                llm_chat_fn=self.llm_chat,
            )
        except Exception as e:
            self.config.verbose and print(f"  [pipeline] Technique extraction error: {e}")
            return []

    # ----------------------------------------------------------
    # Report printing
    # ----------------------------------------------------------

    def _stage_self_audit(self, selected, branches, resolution, run_id):
        """Quick inline self-audit — checks low-confidence steps."""
        import re as _re
        question_prompt = f"""
Branch selected: {selected.get('name')}
Method: {selected.get('method')}
Steps: {selected.get('steps')}
Result: {selected.get('candidate_result')}

Other branches: {[b.get('name') for b in branches if b.get('name') != selected.get('name')]}

In one sentence each, answer:
1. What is the single step most likely to be pattern-matching rather than genuine reasoning?
2. What assumption, if false, would collapse this entire branch?
3. Which rejected branch raised the point this one handles least well?
"""
        system = (
            "You audit reasoning briefly and honestly. "
            "Three sentences maximum. Prioritise identifying genuine gaps "
            "over defending the selected branch."
        )
        try:
            response = self.llm_chat(system, question_prompt)
            if self.config.verbose:
                print(f"  [audit] {response[:200]}...")
            uncertainty = [
                s.strip() for s in _re.split(r"[.!?]\n?", response)
                if any(w in s.lower() for w in [
                    "assume", "unclear", "might", "could",
                    "pattern", "weakest", "gap", "unjustified"
                ])
            ]
            if uncertainty and self.obligation_store:
                from obligation_store import ObligationStore
                self.obligation_store.record_run(
                    run_id=run_id + "_audit",
                    domain=self.config.domain_name,
                    question="self_audit",
                    resolution={
                        "obligations": [
                            {"text": f"[Audit] {s[:200]}", "kind": "self_doubt",
                             "source_branch": "self_audit", "status": "global_gap"}
                            for s in uncertainty
                        ],
                        "global_gaps": []
                    }
                )
            return uncertainty
        except Exception:
            return []

    def _print_run_summary(self, result: dict) -> None:
        print(f"\n{'='*60}")
        print(f"[pipeline] Run complete in {result.get('elapsed_seconds')}s")
        print(f"{'='*60}")

        selected = result.get("selected_branch")
        if selected:
            print(f"\n>> Selected branch: {selected.get('name')}")
            print(f"   Method: {selected.get('method')}")
            print(f"   Score: {selected.get('total_score', 0.0):.3f}")
            print(f"   Result: {selected.get('candidate_result', '')[:120]}")

        gaps = result.get("global_gaps", [])
        if gaps:
            print(f"\n>> Global gaps (obligations no branch addressed):")
            for g in gaps[:5]:
                print(f"   [{g['kind']}] {g['text'][:80]}")
            if len(gaps) > 5:
                print(f"   ... and {len(gaps)-5} more")

        p_gaps = result.get("persistent_gaps", [])
        if p_gaps:
            print(f"\n>> Persistent gaps from prior runs:")
            for g in p_gaps[:3]:
                occ = g.get('n_occurrences', '?')
                print(f"   [seen {occ}x] {g.get('text', '')[:80]}")

        stage_ver = result["stages"].get("verification", {})
        if stage_ver and not stage_ver.get("skipped"):
            print(f"\n>> Tool verification:")
            for name, r in stage_ver.items():
                if isinstance(r, dict):
                    v = r.get("verified")
                    adj = r.get("score_adjustment")
                    calls = r.get("n_tool_calls", 0)
                    adj_str = f"{adj:+.3f}" if adj is not None else "n/a"
                    print(f"   {name}: verified={v}, score_adj={adj_str}, "
                          f"tool_calls={calls}")

        stage_tl = result["stages"].get("technique_library", {})
        if stage_tl and not stage_tl.get("skipped"):
            n_new = stage_tl.get("new_techniques", 0)
            n_total = stage_tl.get("library_summary", {}).get("n_techniques", 0)
            if n_new:
                print(f"\n>> Techniques: {n_new} new extracted; "
                      f"{n_total} in library total.")

        audit = result["stages"].get("self_audit", {})
        if audit and audit.get("n_findings", 0):
            print(f"\n>> Self-audit flagged {audit['n_findings']} uncertain steps")
            for f in audit.get("findings", [])[:3]:
                print(f"   - {f[:80]}")

        if result.get("errors"):
            print(f"\n>> Errors encountered: {result['errors']}")

        print()


# ============================================================
# Convenience wrapper for compatibility with existing ask_reasoning_assistant
# ============================================================

_PIPELINE_INSTANCE: Optional[ReasoningPipeline] = None


def setup_pipeline(
    llm_chat_fn: Callable,
    domain_name: str = "physics_mond",
    parameter_ranges: Optional[dict] = None,
    existing_score_fn: Optional[Callable] = None,
    obligation_store_path: str = "./obligation_store.json",
    technique_library_path: str = "./technique_library.json",
    verbose: bool = True,
) -> ReasoningPipeline:
    """
    Convenience setup function. Creates and stores a global pipeline instance
    so you don't need to pass it around. Call this once after loading the model.

    Example:
        from pipeline import setup_pipeline, ask
        setup_pipeline(llm_chat, existing_score_fn=auto_score_physics_branch_v3)
        result = ask(test_question)
    """
    global _PIPELINE_INSTANCE

    ranges = parameter_ranges or {
        "mT":    [-2.0, 2.0],
        "lam12": [ 0.0, 1.0],
        "x1":    [-1.0, 1.0],
        "y1":    [-1.0, 1.0],
    }

    config = PipelineConfig(
        domain_name=domain_name,
        parameter_ranges=ranges,
        obligation_store_path=obligation_store_path,
        technique_library_path=technique_library_path,
        verbose=verbose,
    )
    _PIPELINE_INSTANCE = ReasoningPipeline(
        config, llm_chat_fn=llm_chat_fn,
        existing_score_fn=existing_score_fn,
    )
    print(f"[pipeline] Ready. Call ask(question) to run.")
    return _PIPELINE_INSTANCE


def ask(question: str, context: str = "") -> dict:
    """Run the pipeline on a question using the globally configured instance."""
    if _PIPELINE_INSTANCE is None:
        raise RuntimeError(
            "Pipeline not set up. Call setup_pipeline(llm_chat) first."
        )
    return _PIPELINE_INSTANCE.run(question, context)


def print_report(result: dict) -> None:
    """Print a readable report from a pipeline result dict."""
    if _PIPELINE_INSTANCE:
        _PIPELINE_INSTANCE._print_run_summary(result)
    else:
        print(json.dumps(result, indent=2, default=str))
