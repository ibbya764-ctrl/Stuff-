"""
enhanced_pipeline.py
====================

Drop-in replacement for ReasoningPipeline that automatically engages
all new capabilities on every run.

Every call to ask() now does:

  PRE-RUN
  ├─ Domain detection → bootstrap if new domain encountered
  │   └─ Searches for domain methodology, generates verification tools
  ├─ Unified memory query → what does the system know about this question?
  │   └─ Relevant techniques, similar past runs, related obligations
  ├─ Introspective context → what does the system know about itself?
  │   └─ Domain maturity, confidence modifiers, known gaps
  └─ Structural branches → generate branch skeletons from accumulated knowledge

  CORE RUN (existing ReasoningPipeline logic, unchanged)
  └─ Branch generation, verification, scoring, selection

  POST-RUN
  ├─ Novelty check → is this result genuinely new?
  │   └─ Internal check + optional web search
  ├─ Episodic record → store run compactly in episodic memory
  ├─ Learning updates → directed graph, technique embedder, GUE monitor
  ├─ Metacognitive audit → challenger questions on the selected branch
  └─ Adjuster notification → track runs toward adjustment evaluation

  PERIODIC (call consolidate() between sessions)
  ├─ Memory consolidation → merge duplicates, prune noise
  ├─ Self-adjustment cycle → calibrate thresholds and scoring weights
  ├─ Attention cycle → generate code for failure patterns
  └─ Curiosity exploration → pursue unresolved gaps autonomously

Usage:
    from enhanced_pipeline import EnhancedPipeline, EnhancedConfig

    pipeline = EnhancedPipeline(
        llm_chat_fn=llm_chat,
        base_dir="/content/drive/MyDrive/scaffold",
        verbose=True,
    )

    result = pipeline.ask("derive the MOND coefficient from CP2 closure")

    # Between sessions
    pipeline.consolidate()

    # Autonomous exploration
    pipeline.explore(n=3)
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# Configuration
# ============================================================

@dataclass
class EnhancedConfig:
    """
    Configuration for the enhanced pipeline.
    All capability flags default to True — set False to disable.
    """
    # Paths
    base_dir:              str   = "./scaffold_data"

    # Core pipeline settings (passed through to ReasoningPipeline)
    domain_name:           str   = "physics_mond"
    n_seed_branches:       int   = 3
    verbose:               bool  = True

    # Capability flags — set False to disable individual modules
    enable_bootstrap:      bool  = True   # auto-bootstrap new domains
    enable_memory:         bool  = True   # episodic store + unified retrieval
    enable_introspection:  bool  = True   # inject self-knowledge into prompts
    enable_structural:     bool  = True   # structural branches from library
    enable_novelty:        bool  = True   # check finding novelty
    enable_metacognition:  bool  = True   # challenger audit post-run
    enable_gue:            bool  = True   # GUE convergence monitoring
    enable_adjuster:       bool  = True   # self-adjustment tracking
    enable_web_search:     bool  = False  # web search for novelty/bootstrap
                                          # (off by default — costs API credits)

    # Thresholds
    novelty_search_threshold: float = 0.5   # only search if internal novelty < this
    consolidate_every_n:      int   = 20    # auto-consolidate after N runs

    def paths(self) -> dict:
        b = self.base_dir
        return {
            "obligation_store":   f"{b}/obligation_store.json",
            "technique_library":  f"{b}/technique_library.json",
            "directed_graph":     f"{b}/directed_graph.json",
            "embedder":           f"{b}/embedder",
            "episodic":           f"{b}/episodic_store.json",
            "memory_metrics":     f"{b}/memory_metrics.json",
            "self_model_cache":   f"{b}/self_model_cache.json",
            "adj_log":            f"{b}/self_adjustment_log.json",
            "attention_log":      f"{b}/attention_log.json",
            "bootstrap_log":      f"{b}/bootstrap_log.json",
            "gue_log":            f"{b}/gue_log.json",
        }


# ============================================================
# EnhancedPipeline
# ============================================================

class EnhancedPipeline:
    """
    Wraps ReasoningPipeline with all new capabilities firing
    automatically on every run.

    Graceful degradation: every new module is optional. If a module
    fails to initialise or raises during a run, the core pipeline
    continues without it. Errors are logged but never fatal.
    """

    def __init__(
        self,
        llm_chat_fn:    Callable,
        config:         Optional[EnhancedConfig] = None,
        search_fn:      Optional[Callable]       = None,
        base_pipeline_kwargs: dict               = None,
    ):
        self.cfg        = config or EnhancedConfig()
        self._llm_raw   = llm_chat_fn
        self._search_fn = search_fn
        self._run_count  = 0

        paths = self.cfg.paths()
        os.makedirs(self.cfg.base_dir, exist_ok=True)

        if self.cfg.verbose:
            print(f"[enhanced_pipeline] Initialising...")

        # ---- Core pipeline ----
        self._core = self._init_core(llm_chat_fn, paths, base_pipeline_kwargs or {})

        # ---- New modules ----
        self._directed    = self._init_directed(paths)
        self._composer    = self._init_composer(paths)
        self._episodic    = self._init_episodic(paths)
        self._memory      = self._init_memory(paths)
        self._self_model  = self._init_self_model(paths)
        self._introspect  = self._init_introspection()
        self._sampler     = self._init_sampler()
        self._novelty     = self._init_novelty()
        self._gue         = self._init_gue(paths)
        self._adjuster    = self._init_adjuster(paths)
        self._attention   = self._init_attention(paths)
        self._curiosity   = self._init_curiosity()
        self._bootstrapper = self._init_bootstrapper(paths)

        if self.cfg.verbose:
            active = self._active_modules()
            print(f"[enhanced_pipeline] Ready. "
                  f"{len(active)} modules active: {', '.join(active)}")

    # ============================================================
    # Public interface
    # ============================================================

    def ask(self, question: str, domain: str = "", context: str = "") -> dict:
        """
        Run the full enhanced pipeline on a question.
        Returns the same dict format as ReasoningPipeline.run()
        plus an 'enhancements' key with module outputs.
        """
        if not domain:
            domain = self.cfg.domain_name

        enhancements = {}
        t0 = time.time()

        # ---- PRE-RUN ----
        enhancements["bootstrap"]    = self._pre_bootstrap(question, domain)
        enhancements["memory"]       = self._pre_memory(question, domain)
        enhancements["introspection"] = self._pre_introspection(question, domain)

        # Wrap llm_chat to inject introspective context
        llm_for_this_run = self._make_introspective_llm(
            question, domain, enhancements.get("introspection", {})
        )

        # Update core pipeline's llm to the wrapped version
        if self._core:
            self._core.llm_chat = llm_for_this_run

        # ---- CORE RUN ----
        result = self._run_core(question, context)

        # ---- POST-RUN ----
        enhancements["novelty"]       = self._post_novelty(result, domain)
        enhancements["episodic"]      = self._post_episodic(result)
        enhancements["learning"]      = self._post_learning(result)
        enhancements["gue"]           = self._post_gue(result)
        enhancements["metacognition"] = self._post_metacognition(result)
        enhancements["adjuster"]      = self._post_adjuster()

        result["enhancements"] = enhancements
        result["enhanced_elapsed"] = round(time.time() - t0, 2)

        # Auto-consolidate every N runs
        self._run_count += 1
        if (self.cfg.enable_adjuster
                and self._run_count % self.cfg.consolidate_every_n == 0):
            if self.cfg.verbose:
                print(f"[enhanced_pipeline] Auto-consolidating "
                      f"(every {self.cfg.consolidate_every_n} runs)...")
            self.consolidate(quiet=True)

        return result

    def consolidate(self, quiet: bool = False) -> dict:
        """
        Periodic maintenance: consolidate memory, run self-adjustment,
        run attention cycle, optionally run curiosity.
        Call between sessions or overnight.
        """
        report = {}
        if not quiet and self.cfg.verbose:
            print("\n[enhanced_pipeline] Running consolidation cycle...")

        if self._memory and self.cfg.enable_memory:
            try:
                report["consolidation"] = self._memory.consolidate()
            except Exception as e:
                report["consolidation_error"] = str(e)

        if self._adjuster and self.cfg.enable_adjuster:
            try:
                report["adjustment"] = self._adjuster.run_cycle()
                evaluated = self._adjuster.evaluate_sandboxed()
                report["evaluated"] = [r.to_dict() for r in evaluated]
            except Exception as e:
                report["adjustment_error"] = str(e)

        if self._attention and self.cfg.enable_adjuster:
            try:
                report["attention"] = self._attention.run_cycle(max_patterns=3)
            except Exception as e:
                report["attention_error"] = str(e)

        return report

    def explore(self, n: int = 3) -> list:
        """
        Run n curiosity-driven autonomous explorations.
        The system pursues its own highest-priority unresolved gaps.
        """
        if not self._curiosity:
            return []
        try:
            return self._curiosity.pursue(n_iterations=n)
        except Exception as e:
            if self.cfg.verbose:
                print(f"[enhanced_pipeline] Curiosity error: {e}")
            return []

    # ============================================================
    # Pre-run steps
    # ============================================================

    def _pre_bootstrap(self, question: str, domain: str) -> dict:
        if not self._bootstrapper or not self.cfg.enable_bootstrap:
            return {}
        try:
            result = self._bootstrapper.check_and_bootstrap(
                domain, self._self_model
            )
            if result and self.cfg.verbose:
                print(f"  [bootstrap] New domain '{domain}': "
                      f"{result.get('added_techniques', 0)} techniques, "
                      f"{result.get('added_verifiers', 0)} verifiers added")
            return result or {}
        except Exception as e:
            return {"error": str(e)}

    def _pre_memory(self, question: str, domain: str) -> dict:
        if not self._memory or not self.cfg.enable_memory:
            return {}
        try:
            mem = self._memory.query(question, domain)
            if self.cfg.verbose and mem.has_prior_experience():
                print(f"  [memory] {len(mem.similar_past_runs)} similar past runs, "
                      f"{len(mem.relevant_techniques)} relevant techniques, "
                      f"freshness={mem.freshness_score:.2f}")
            return {
                "n_techniques":   len(mem.relevant_techniques),
                "n_past_runs":    len(mem.similar_past_runs),
                "n_obligations":  len(mem.related_obligations),
                "freshness":      mem.freshness_score,
                "suggested_seq":  mem.suggested_sequence[:3],
            }
        except Exception as e:
            return {"error": str(e)}

    def _pre_introspection(self, question: str, domain: str) -> dict:
        if not self._introspect or not self.cfg.enable_introspection:
            return {}
        try:
            if self._self_model:
                self._self_model.compute_domain_profiles()
            freshness = None
            mem_data  = None
            if self._memory:
                try:
                    mem = self._memory.query(question, domain, top_k=3)
                    freshness = mem.freshness_score
                except Exception:
                    pass
            context_str = self._introspect.format_for_branch_generation(
                question, domain, freshness
            )
            modifier = (self._self_model.confidence_modifier(domain)
                        if self._self_model else 1.0)
            return {
                "context": context_str,
                "confidence_modifier": modifier,
                "domain_maturity": (
                    self._self_model._profiles.get(domain, {})
                    .overall_maturity if self._self_model else "unknown"
                ) if hasattr(
                    self._self_model._profiles.get(domain, object()),
                    "overall_maturity"
                ) else "unknown",
            }
        except Exception as e:
            return {"error": str(e)}

    # ============================================================
    # LLM wrapping — injects introspective context into prompts
    # ============================================================

    def _make_introspective_llm(
        self,
        question: str,
        domain:   str,
        introspection_data: dict,
    ) -> Callable:
        """
        Returns a wrapped llm_chat that prepends introspective context
        to branch generation prompts.
        """
        raw_llm   = self._llm_raw
        ctx_str   = introspection_data.get("context", "")
        enabled   = self.cfg.enable_introspection and bool(ctx_str)

        def _wrapped(system: str, user: str) -> str:
            if enabled:
                lower_sys = system.lower()
                # Detect branch generation calls by keywords in system prompt
                is_branch_gen = any(k in lower_sys for k in [
                    "branch", "method", "approach", "hypothesis",
                    "reasoning chain", "candidate", "derive", "generate",
                ])
                if is_branch_gen:
                    system = ctx_str + "\n\n" + system
            return raw_llm(system, user)

        return _wrapped

    # ============================================================
    # Core run
    # ============================================================

    def _run_core(self, question: str, context: str) -> dict:
        if self._core is None:
            return {
                "run_id":   f"run-{int(time.time())}",
                "question": question,
                "domain":   self.cfg.domain_name,
                "stages":   {},
                "errors":   ["Core pipeline not initialised"],
            }
        try:
            return self._core.run(question, context)
        except Exception as e:
            return {
                "run_id":   f"run-{int(time.time())}",
                "question": question,
                "domain":   self.cfg.domain_name,
                "stages":   {},
                "errors":   [f"Core pipeline error: {e}"],
            }

    # ============================================================
    # Post-run steps
    # ============================================================

    def _post_novelty(self, result: dict, domain: str) -> dict:
        if not self._novelty or not self.cfg.enable_novelty:
            return {}
        selected = result.get("selected_branch", {})
        if not selected:
            return {}
        claim = selected.get("candidate_result", "")
        if not claim or len(claim) < 20:
            return {}
        try:
            do_search = (
                self.cfg.enable_web_search
                and self._search_fn is not None
            )
            novelty = self._novelty.check(claim, domain, search=do_search)
            if self.cfg.verbose:
                print(f"  [novelty] {novelty.status.upper()} "
                      f"(confidence {novelty.confidence:.0%})")
            return {
                "status":     novelty.status,
                "confidence": novelty.confidence,
                "recommendation": novelty.recommendation,
                "n_internal": len(novelty.internal_matches),
                "n_external": len(novelty.external_matches),
                "n_contra":   len(novelty.contradictions),
            }
        except Exception as e:
            return {"error": str(e)}

    def _post_episodic(self, result: dict) -> dict:
        if not self._episodic or not self.cfg.enable_memory:
            return {}
        try:
            gue_score = None
            if self._gue:
                try:
                    gue_score = self._gue._branch_scorer.measure(
                        [b.get("total_score", 0)
                         for b in result.get("all_branches", [])]
                    ).gue_score
                except Exception:
                    pass
            record = self._episodic.record(result, gue_score=gue_score)
            return {"recorded": record is not None,
                    "run_id":   result.get("run_id")}
        except Exception as e:
            return {"error": str(e)}

    def _post_learning(self, result: dict) -> dict:
        if not self._composer or not self.cfg.enable_memory:
            return {}
        selected = result.get("selected_branch", {})
        if not selected:
            return {}
        tid = selected.get("provenance", {}).get("technique_id", "")
        if not tid:
            return {}
        try:
            self._composer.update_from_run([tid], success=True)
            return {"updated_technique": tid}
        except Exception as e:
            return {"error": str(e)}

    def _post_gue(self, result: dict) -> dict:
        if not self._gue or not self.cfg.enable_gue:
            return {}
        try:
            scores = [b.get("total_score", 0)
                      for b in result.get("all_branches", [])]
            report = self._gue.report(
                run_id=result.get("run_id", "unknown"),
                branch_scores=scores,
                library=(self._core.technique_library
                         if self._core else None),
                obligation_store=(self._core.obligation_store
                                  if self._core else None),
            )
            return {
                "overall_gue": report.overall_gue_score,
                "trend":       report.trend,
            }
        except Exception as e:
            return {"error": str(e)}

    def _post_metacognition(self, result: dict) -> dict:
        if not self.cfg.enable_metacognition:
            return {}
        try:
            from metacognition import metacognitive_audit
            audit = metacognitive_audit(
                result,
                llm_chat_fn=self._llm_raw,
                obligation_store=(self._core.obligation_store
                                  if self._core else None),
                verbose=False,
            )
            if audit.get("skipped"):
                return {"skipped": True}
            if self.cfg.verbose and audit.get("findings"):
                conf = audit.get("objective_confidence", "?")
                n_f  = audit.get("n_findings", 0)
                print(f"  [metacog] Confidence: {conf}, "
                      f"{n_f} finding(s)")
            return {
                "confidence":   audit.get("objective_confidence"),
                "n_findings":   audit.get("n_findings", 0),
                "n_obligations": audit.get("n_recorded_as_obligations", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def _post_adjuster(self) -> dict:
        if not self._adjuster or not self.cfg.enable_adjuster:
            return {}
        try:
            self._adjuster.notify_run_completed()
            return {"notified": True,
                    "runs_since_sandbox": self._adjuster._runs_since_sandbox}
        except Exception as e:
            return {"error": str(e)}

    # ============================================================
    # Module initialisation (all wrapped in try/except)
    # ============================================================

    def _init_core(self, llm_fn, paths, kwargs):
        try:
            import importlib
            pipeline_mod = importlib.import_module("pipeline")
            cfg_cls      = getattr(pipeline_mod, "PipelineConfig", None)
            pipe_cls     = getattr(pipeline_mod, "ReasoningPipeline", None)
            if cfg_cls is None or pipe_cls is None:
                return None
            cfg = cfg_cls(
                domain_name=self.cfg.domain_name,
                n_seed_branches=self.cfg.n_seed_branches,
                verbose=self.cfg.verbose,
                obligation_store_path=paths["obligation_store"],
                technique_library_path=paths["technique_library"],
                **{k: v for k, v in kwargs.items()
                   if k in cfg_cls.__dataclass_fields__},
            )
            return pipe_cls(cfg, llm_chat_fn=llm_fn)
        except Exception as e:
            if self.cfg.verbose:
                print(f"  [init] Core pipeline: {e}")
            return None

    def _init_directed(self, paths):
        try:
            from directed_graph import DirectedCoOccurrenceGraph
            return DirectedCoOccurrenceGraph(paths["directed_graph"])
        except Exception:
            return None

    def _init_composer(self, paths):
        if not self._directed:
            return None
        try:
            lib = self._core.technique_library if self._core else None
            if lib is None:
                from technique_library import TechniqueLibrary
                lib = TechniqueLibrary(paths["technique_library"])
            from technique_embedder import EmbeddedTechniqueComposer
            return EmbeddedTechniqueComposer(
                lib, self._directed,
                embedder_dir=paths["embedder"],
                verbose=False,
            )
        except Exception as e:
            if self.cfg.verbose:
                print(f"  [init] Composer: {e}")
            return None

    def _init_episodic(self, paths):
        try:
            from memory_system import EpisodicStore
            return EpisodicStore(paths["episodic"])
        except Exception:
            return None

    def _init_memory(self, paths):
        if not self._episodic or not self._composer:
            return None
        try:
            from memory_system import MemorySystem
            obl = self._core.obligation_store if self._core else None
            if obl is None:
                return None
            mem = MemorySystem(
                episodic_path=paths["episodic"],
                obligation_store=obl,
                embedded_composer=self._composer,
                directed_graph=self._directed,
                metrics_log_path=paths["memory_metrics"],
                verbose=False,
            )
            # Share the same episodic store instance
            mem.episodic = self._episodic
            return mem
        except Exception as e:
            if self.cfg.verbose:
                print(f"  [init] Memory: {e}")
            return None

    def _init_self_model(self, paths):
        if not self._episodic or not self._composer:
            return None
        try:
            from introspection import SelfModel
            lib = self._core.technique_library if self._core else None
            obl = self._core.obligation_store if self._core else None
            if lib is None or obl is None:
                return None
            return SelfModel(
                lib, obl, self._episodic,
                self._directed, self._composer,
                cache_path=paths["self_model_cache"],
            )
        except Exception as e:
            if self.cfg.verbose:
                print(f"  [init] SelfModel: {e}")
            return None

    def _init_introspection(self):
        if not self._self_model:
            return None
        try:
            from introspection import IntrospectiveContext
            return IntrospectiveContext(self._self_model)
        except Exception:
            return None

    def _init_sampler(self):
        if not self._composer:
            return None
        try:
            from structural_branch_sampler import StructuralBranchSampler
            return StructuralBranchSampler(
                self._composer, self._directed, verbose=False
            )
        except Exception:
            return None

    def _init_novelty(self):
        if not self._episodic:
            return None
        try:
            from knowledge_seeker import NoveltyDetector
            lib = self._core.technique_library if self._core else None
            obl = self._core.obligation_store if self._core else None
            return NoveltyDetector(
                self._episodic, obl, lib,
                search_fn=self._search_fn if self.cfg.enable_web_search else None,
            )
        except Exception:
            return None

    def _init_gue(self, paths):
        try:
            from gue_engine import SystemGUEMonitor
            return SystemGUEMonitor(
                log_path=paths["gue_log"],
                verbose=False,
            )
        except Exception:
            return None

    def _init_adjuster(self, paths):
        if not self._episodic:
            return None
        try:
            from memory_system import MemoryConsolidator
            from self_adjuster import SelfAdjuster
            consolidator = MemoryConsolidator(verbose=False)
            return SelfAdjuster(
                consolidator, self._episodic,
                self._composer or object(),
                log_path=paths["adj_log"],
                verbose=self.cfg.verbose,
                human_approval=True,
            )
        except Exception as e:
            if self.cfg.verbose:
                print(f"  [init] Adjuster: {e}")
            return None

    def _init_attention(self, paths):
        if not self._episodic:
            return None
        try:
            from code_generator import (
                FailurePatternDetector, CodeGenerator,
                CodeSandbox, AttentionDirector,
            )
            from memory_system import MemoryConsolidator
            from self_adjuster import SelfAdjuster
            lib = self._core.technique_library if self._core else None
            obl = self._core.obligation_store if self._core else None
            if lib is None:
                return None
            detector  = FailurePatternDetector(self._episodic, obl, lib)
            sandbox   = CodeSandbox(timeout_seconds=5)
            generator = CodeGenerator(self._llm_raw, sandbox, lib)
            consolidator = MemoryConsolidator(verbose=False)
            adjuster  = self._adjuster or SelfAdjuster(
                consolidator, self._episodic,
                self._composer or object(),
                log_path=paths["adj_log"],
                verbose=False,
            )
            return AttentionDirector(
                detector, generator, adjuster, lib,
                log_path=paths["attention_log"],
                verbose=self.cfg.verbose,
            )
        except Exception as e:
            if self.cfg.verbose:
                print(f"  [init] Attention: {e}")
            return None

    def _init_curiosity(self):
        if not self._core:
            return None
        try:
            from curiosity_engine import CuriosityEngine
            return CuriosityEngine(
                pipeline=self._core,
                verbose=self.cfg.verbose,
            )
        except Exception:
            return None

    def _init_bootstrapper(self, paths):
        if not self._episodic:
            return None
        try:
            from knowledge_seeker import (
                NoveltyDetector, VerificationBootstrapper,
                DomainBootstrapManager,
            )
            from code_generator import CodeGenerator, CodeSandbox
            lib = self._core.technique_library if self._core else None
            obl = self._core.obligation_store if self._core else None
            if lib is None:
                return None
            sandbox   = CodeSandbox(timeout_seconds=5)
            generator = CodeGenerator(self._llm_raw, sandbox, lib)
            verif_boot = VerificationBootstrapper(
                search_fn=self._search_fn or (lambda q: []),
                code_generator=generator,
                technique_library=lib,
                llm_chat_fn=self._llm_raw,
                verbose=self.cfg.verbose,
            )
            novelty_det = NoveltyDetector(
                self._episodic, obl, lib,
                search_fn=self._search_fn if self.cfg.enable_web_search else None,
            )
            return DomainBootstrapManager(
                verif_boot, novelty_det,
                log_path=paths["bootstrap_log"],
                verbose=self.cfg.verbose,
            )
        except Exception as e:
            if self.cfg.verbose:
                print(f"  [init] Bootstrapper: {e}")
            return None

    # ============================================================
    # Helpers
    # ============================================================

    def _active_modules(self) -> list[str]:
        return [
            name for name, obj in [
                ("core",         self._core),
                ("directed",     self._directed),
                ("composer",     self._composer),
                ("memory",       self._memory),
                ("introspection", self._introspect),
                ("novelty",      self._novelty),
                ("gue",          self._gue),
                ("adjuster",     self._adjuster),
                ("attention",    self._attention),
                ("curiosity",    self._curiosity),
                ("bootstrap",    self._bootstrapper),
            ] if obj is not None
        ]

    def health(self) -> dict:
        """Quick health report across all modules."""
        report = {"active_modules": self._active_modules()}
        if self._memory:
            try:
                m = self._memory.health_report()
                report["memory"] = m.to_dict()
            except Exception:
                pass
        if self._gue:
            report["gue_trend"] = self._gue.convergence_trend()[-5:]
        return report
