"""
curiosity_engine.py
===================

Self-directed exploration of the system's own knowledge frontier.

The pipeline currently waits to be asked questions. The curiosity engine
inverts that: it looks at the obligation store's persistent unresolved
gaps, ranks them, picks the highest-value gap, and runs a question
designed to probe that gap through the full pipeline. The system stops
being purely reactive and starts driving toward its own unknowns.

Key concepts:

  - Blocking score: a gap that recurs across many runs and many distinct
    source branches is more valuable to resolve than an isolated one,
    because resolving it likely unblocks downstream reasoning.

  - Exploratory question: a question crafted to probe a specific gap
    rather than to solve a user-supplied problem. The exploratory
    question is what the system "wants to know."

  - Curiosity run: a normal pipeline run, but the question came from
    the system itself and the run is tagged so curiosity-driven results
    don't get confused with user-driven ones.

Drop into the same directory as pipeline.py. Use after a pipeline is
already set up:

    from curiosity_engine import CuriosityEngine
    engine = CuriosityEngine(pipeline)
    engine.pursue(n_iterations=3)
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Data shapes
# ============================================================

@dataclass
class CuriosityTarget:
    """A gap the system has decided to investigate."""

    obligation_id: str
    text: str
    kind: str
    n_occurrences: int
    n_distinct_sources: int
    blocking_score: float
    selected_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "obligation_id": self.obligation_id,
            "text": self.text,
            "kind": self.kind,
            "n_occurrences": self.n_occurrences,
            "n_distinct_sources": self.n_distinct_sources,
            "blocking_score": self.blocking_score,
            "selected_at": self.selected_at,
        }


@dataclass
class CuriosityRun:
    """The result of one curiosity-driven investigation."""

    target: CuriosityTarget
    generated_question: str
    pipeline_result: dict
    discharged_target: bool
    new_techniques: int
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "target": self.target.to_dict(),
            "generated_question": self.generated_question,
            "discharged_target": self.discharged_target,
            "new_techniques": self.new_techniques,
            "notes": self.notes,
            # Don't dump the whole pipeline result; just the summary
            "selected_branch": self.pipeline_result.get(
                "selected_branch", {}
            ).get("name"),
            "elapsed_seconds": self.pipeline_result.get("elapsed_seconds"),
        }


# ============================================================
# Gap ranking
# ============================================================

def rank_gaps_by_blocking_score(
    obligation_store,
    top_n: int = 10,
    min_occurrences: int = 2,
) -> list[CuriosityTarget]:
    """
    Rank persistent gaps by how much they likely block downstream reasoning.

    Score combines:
      - n_occurrences (how often the gap recurs)
      - n_distinct_sources (gaps shared across many branches matter more)
      - kind weighting (gap > assumption — gaps are bigger blockers)
    """
    gaps = obligation_store.query_persistent_gaps(top_n=50)

    targets: list[CuriosityTarget] = []
    for g in gaps:
        if g.n_occurrences < min_occurrences:
            continue

        n_sources = len(set(g.source_branches))

        # Kind weighting: declared gaps weighted higher than mere assumptions
        kind_weight = {
            "gap": 1.5,
            "audit_finding": 1.4,   # surfaced by the metacognition module
            "claim_to_verify": 1.2,
            "assumption": 1.0,
        }.get(g.kind, 1.0)

        # Blocking score: occurrences scaled by source diversity, then
        # weighted by kind. A gap appearing in 5 runs across 3 distinct
        # branches is more "blocking" than one appearing in 5 runs of the
        # same branch.
        score = (g.n_occurrences * (1 + 0.5 * (n_sources - 1))) * kind_weight

        targets.append(CuriosityTarget(
            obligation_id=g.obligation_id,
            text=g.text,
            kind=g.kind,
            n_occurrences=g.n_occurrences,
            n_distinct_sources=n_sources,
            blocking_score=score,
        ))

    targets.sort(key=lambda t: t.blocking_score, reverse=True)
    return targets[:top_n]


# ============================================================
# Question generation
# ============================================================

QUESTION_GEN_SYSTEM = (
    "You translate an unresolved gap from a research log into a precise "
    "research question. The question should:\n"
    "  - Be answerable in principle by reasoning, derivation, or computation\n"
    "  - Probe the gap directly rather than working around it\n"
    "  - Be specific enough that progress on it can be measured\n"
    "  - Not assume the gap is resolvable — 'why is this still open' is fine\n\n"
    "Return JSON only."
)


def generate_exploratory_question(
    target: CuriosityTarget,
    domain_name: str,
    llm_chat_fn,
    related_techniques: Optional[list] = None,
) -> str:
    """
    Turn a CuriosityTarget into a question to feed into the pipeline.
    """
    techniques_text = ""
    if related_techniques:
        techniques_text = "\n\nRelated techniques that have worked before:\n"
        for t in related_techniques[:3]:
            tdict = t.to_dict() if hasattr(t, "to_dict") else t
            techniques_text += (
                f"  - {tdict.get('name', '?')}: "
                f"{tdict.get('description', '')[:120]}\n"
            )

    user = f"""
Domain: {domain_name}

Unresolved gap (seen {target.n_occurrences}x across {target.n_distinct_sources}
distinct branch(es), kind={target.kind}):

  {target.text}
{techniques_text}

Generate one research question that probes this gap directly.

Return JSON:
{{
  "question": "...",
  "rationale": "why this question targets this specific gap"
}}
""".strip()

    try:
        response = llm_chat_fn(QUESTION_GEN_SYSTEM, user)
        parsed = _extract_json(response)
        return parsed.get("question", "").strip() or _fallback_question(target)
    except Exception:
        return _fallback_question(target)


def _fallback_question(target: CuriosityTarget) -> str:
    """If LLM question generation fails, use a templated one."""
    return (f"Investigate the following persistent gap in the reasoning record "
            f"and either resolve it, weaken it, or characterise precisely why "
            f"it remains open: '{target.text[:300]}'")


# ============================================================
# Discharge detection: did the curiosity run actually resolve the target?
# ============================================================

def _did_curiosity_run_discharge_target(
    target: CuriosityTarget,
    pipeline_result: dict,
) -> bool:
    """
    Heuristic: a curiosity run discharges its target if the resolution's
    discharged obligations include text closely matching the target.
    """
    resolution = pipeline_result.get("resolution", {})
    discharged_texts = []
    for o in resolution.get("obligations", []) or []:
        if o.get("status") == "discharged":
            discharged_texts.append(_norm(o.get("text", "")))

    target_norm = _norm(target.text)
    target_words = set(w for w in target_norm.split() if len(w) > 3)

    for d in discharged_texts:
        d_words = set(w for w in d.split() if len(w) > 3)
        if not target_words:
            continue
        overlap = len(target_words & d_words)
        if overlap / max(1, len(target_words)) >= 0.5:
            return True
    return False


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


# ============================================================
# CuriosityEngine
# ============================================================

class CuriosityEngine:
    """
    Drives self-directed exploration of the system's knowledge frontier.

    Holds a reference to a configured ReasoningPipeline and uses its
    obligation_store and technique_library to pick targets and seed
    exploratory questions.
    """

    def __init__(
        self,
        pipeline,
        log_path: str = "./curiosity_log.json",
        verbose: bool = True,
    ):
        self.pipeline = pipeline
        self.log_path = log_path
        self.verbose = verbose

        if pipeline.obligation_store is None:
            raise ValueError(
                "CuriosityEngine requires a pipeline with an obligation_store. "
                "Set obligation_store_path in PipelineConfig."
            )

        self._history: list[dict] = []
        self._load_log()

    # -------------------- I/O --------------------

    def _load_log(self) -> None:
        import os
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, "r") as f:
                self._history = json.load(f).get("runs", [])
        except (json.JSONDecodeError, OSError):
            self._history = []

    def _save_log(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump({"runs": self._history}, f, indent=2, default=str)

    # -------------------- Target selection --------------------

    def _already_pursued(self, obligation_id: str) -> bool:
        """Have we already tried to investigate this gap?"""
        for h in self._history:
            if h.get("target", {}).get("obligation_id") == obligation_id:
                return True
        return False

    def select_target(self) -> Optional[CuriosityTarget]:
        """Pick the highest blocking-score gap not yet pursued."""
        candidates = rank_gaps_by_blocking_score(
            self.pipeline.obligation_store, top_n=20,
        )
        for c in candidates:
            if not self._already_pursued(c.obligation_id):
                return c
        if self.verbose:
            print("  [curiosity] All ranked gaps already pursued.")
        return None

    # -------------------- Single iteration --------------------

    def pursue_one(self) -> Optional[CuriosityRun]:
        """Pick the best target, generate a question, run the pipeline."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[curiosity] Selecting target...")
            print(f"{'='*60}")

        target = self.select_target()
        if target is None:
            if self.verbose:
                print("[curiosity] No target available; nothing to pursue.")
            return None

        if self.verbose:
            print(f"  Target: [{target.kind}, {target.n_occurrences}x, "
                  f"{target.n_distinct_sources} sources, "
                  f"score={target.blocking_score:.2f}]")
            print(f"  Text:   {target.text[:200]}")

        # Pull related techniques to seed the question
        related = []
        if self.pipeline.technique_library is not None:
            try:
                related = self.pipeline.technique_library.find_relevant_techniques(
                    target.text, self.pipeline.config.domain_name, top_n=3,
                )
            except Exception:
                related = []

        question = generate_exploratory_question(
            target,
            self.pipeline.config.domain_name,
            self.pipeline.llm_chat,
            related_techniques=related,
        )

        if self.verbose:
            print(f"\n  Generated question:\n    {question[:300]}")
            print(f"\n[curiosity] Running pipeline on this question...\n")

        # Run the full pipeline on the curiosity question
        try:
            result = self.pipeline.run(
                question,
                context=f"[curiosity-driven; targeting obligation "
                        f"{target.obligation_id}]",
            )
        except Exception as e:
            if self.verbose:
                print(f"  [curiosity] Pipeline run failed: {e}")
            return None

        discharged = _did_curiosity_run_discharge_target(target, result)
        n_techniques = len(result.get("new_technique_ids", []))

        run = CuriosityRun(
            target=target,
            generated_question=question,
            pipeline_result=result,
            discharged_target=discharged,
            new_techniques=n_techniques,
            notes=("Discharged target." if discharged
                   else "Target still open after run."),
        )

        # Log
        self._history.append({
            "target": target.to_dict(),
            "generated_question": question,
            "discharged_target": discharged,
            "new_techniques": n_techniques,
            "selected_branch": (result.get("selected_branch") or {}).get("name"),
            "run_id": result.get("run_id"),
            "timestamp": time.time(),
        })
        self._save_log()

        if self.verbose:
            verdict = ("✓ TARGET DISCHARGED" if discharged
                       else "✗ target still open")
            print(f"\n[curiosity] {verdict}")
            print(f"  New techniques extracted: {n_techniques}")

        return run

    # -------------------- Multiple iterations --------------------

    def pursue(self, n_iterations: int = 3) -> list[CuriosityRun]:
        """Run multiple iterations of self-directed exploration."""
        runs: list[CuriosityRun] = []
        for i in range(n_iterations):
            if self.verbose:
                print(f"\n[curiosity] === ITERATION {i+1}/{n_iterations} ===")
            run = self.pursue_one()
            if run is None:
                break
            runs.append(run)

        if self.verbose:
            self._print_session_summary(runs)
        return runs

    def _print_session_summary(self, runs: list[CuriosityRun]) -> None:
        print(f"\n{'='*60}")
        print(f"[curiosity] Session complete: {len(runs)} target(s) pursued")
        n_discharged = sum(1 for r in runs if r.discharged_target)
        n_techniques = sum(r.new_techniques for r in runs)
        print(f"  Discharged: {n_discharged}/{len(runs)}")
        print(f"  New techniques learned: {n_techniques}")
        print(f"{'='*60}\n")

    # -------------------- Inspection --------------------

    def history(self) -> list[dict]:
        return list(self._history)

    def summary(self) -> dict:
        n_total = len(self._history)
        n_discharged = sum(1 for h in self._history if h.get("discharged_target"))
        n_techniques = sum(h.get("new_techniques", 0) for h in self._history)
        return {
            "n_pursuits": n_total,
            "n_discharged": n_discharged,
            "discharge_rate": (n_discharged / n_total) if n_total else 0.0,
            "n_techniques_from_curiosity": n_techniques,
        }


# ============================================================
# JSON extraction helper
# ============================================================

def _extract_json(text: str) -> dict:
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
