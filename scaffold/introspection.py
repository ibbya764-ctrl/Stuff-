"""
introspection.py
================

The system's model of itself.

Gives the reasoning scaffold explicit, structured awareness of its own
internal state — knowledge coverage, organisation quality, failure
patterns, and where genuine uncertainty lies.

This is different from the metacognition module, which audits reasoning
chains AFTER the fact. Introspection happens BEFORE reasoning — it
shapes what gets attempted, how confidently, and what the system
explicitly flags as known unknowns.

Three components:

  DomainProfile     — per-domain snapshot: technique coverage, GUE
                      score, verification success rate, graph density.
                      Answers: "how well do I know this domain?"

  SelfModel         — full system self-description. Identifies weaknesses,
                      generates self-obligations (gaps in the system's
                      own coverage), computes confidence modifiers.
                      Answers: "what should I be uncertain about?"

  IntrospectiveContext — formats the self-model for injection into the
                      LLM's prompt. The LLM sees a structured description
                      of the system's own state and can reason about its
                      limitations when proposing methods.
                      Answers: "how do I tell myself what I know about myself?"

The key idea: the system's self-knowledge is represented in the same
data structures as all other knowledge. Self-obligations go into the
obligation store. Domain profiles get embedded into the same space as
techniques. The curiosity engine can target "improve my economics
coverage" the same way it targets "resolve this physics gap."

This makes self-improvement a special case of the general reasoning
loop, not a separate hard-coded process.
"""

import os
import re
import json
import time
import math
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Domain Profile
# ============================================================

@dataclass
class DomainProfile:
    """
    Everything the system knows about its own knowledge in one domain.
    """
    domain:               str
    n_techniques:         int
    gue_score:            Optional[float]   # eigenvalue organisation quality
    avg_success_rate:     Optional[float]   # fraction of runs that verified
    n_verified_runs:      int
    n_persistent_gaps:    int
    graph_density:        float             # 0 = no connections, 1 = fully connected
    imaginary_fraction:   float             # Hermitian directed structure
    freshness_of_recent:  float             # how novel are recent questions?

    # Derived assessments
    coverage_level:       str = ""   # "sparse" | "moderate" | "dense"
    organisation_level:   str = ""   # "disorganised" | "partial" | "organised"
    reliability_level:    str = ""   # "unreliable" | "partial" | "reliable"
    overall_maturity:     str = ""   # "early" | "developing" | "mature"

    def __post_init__(self):
        self.coverage_level = (
            "sparse"   if self.n_techniques < 5  else
            "moderate" if self.n_techniques < 20 else
            "dense"
        )
        self.organisation_level = (
            "disorganised" if (self.gue_score or 0) < 0.5  else
            "partial"      if (self.gue_score or 0) < 0.75 else
            "organised"
        )
        self.reliability_level = (
            "unreliable" if (self.avg_success_rate or 0) < 0.4 else
            "partial"    if (self.avg_success_rate or 0) < 0.7 else
            "reliable"
        )
        maturity_score = (
            (1 if self.coverage_level == "dense"       else
             0.5 if self.coverage_level == "moderate"  else 0)
            + (1 if self.organisation_level == "organised" else
               0.5 if self.organisation_level == "partial" else 0)
            + (1 if self.reliability_level == "reliable" else
               0.5 if self.reliability_level == "partial" else 0)
        )
        self.overall_maturity = (
            "mature"     if maturity_score >= 2.5 else
            "developing" if maturity_score >= 1.5 else
            "early"
        )

    def confidence_modifier(self) -> float:
        """
        How much to adjust confidence for claims in this domain.
        Returns a multiplier: 1.0 = no change, <1.0 = reduce confidence.
        """
        score = {
            "mature":     1.0,
            "developing": 0.75,
            "early":      0.5,
        }[self.overall_maturity]

        # Additional penalty if very fresh (novel problem types)
        if self.freshness_of_recent < 0.2:
            score *= 0.85

        return round(score, 3)

    def to_dict(self) -> dict:
        return {
            "domain":            self.domain,
            "n_techniques":      self.n_techniques,
            "gue_score":         self.gue_score,
            "avg_success_rate":  self.avg_success_rate,
            "n_verified_runs":   self.n_verified_runs,
            "n_persistent_gaps": self.n_persistent_gaps,
            "graph_density":     round(self.graph_density, 3),
            "imaginary_fraction": round(self.imaginary_fraction, 3),
            "coverage_level":    self.coverage_level,
            "organisation_level": self.organisation_level,
            "reliability_level": self.reliability_level,
            "overall_maturity":  self.overall_maturity,
            "confidence_modifier": self.confidence_modifier(),
        }

    def brief(self) -> str:
        gue_str = f", GUE {self.gue_score:.2f}" if self.gue_score is not None else ""
        return (
            f"{self.domain}: {self.n_techniques} techniques, "
            f"{self.overall_maturity} ({self.coverage_level} coverage, "
            f"{self.organisation_level}, {self.reliability_level}{gue_str}), "
            f"confidence modifier {self.confidence_modifier():.2f}"
        )


# ============================================================
# SelfModel
# ============================================================

@dataclass
class SelfObligation:
    """
    An obligation the system generates about improving itself.
    Goes into the obligation store as a 'self_improvement' kind.
    """
    text:     str
    domain:   str
    priority: float    # 0-1, higher = more urgent
    kind:     str      # "coverage_gap" | "organisation_gap" | "reliability_gap"

    def to_obligation_dict(self) -> dict:
        return {
            "text":          f"[self] {self.text}",
            "kind":          "self_improvement",
            "source_branch": "introspection",
            "status":        "open",
        }


class SelfModel:
    """
    The system's model of its own internal state.

    Reads from all available stores and computes:
      - Per-domain profiles
      - System-wide weaknesses
      - Self-obligations (what the system should improve)
      - Confidence modifiers by domain
    """

    def __init__(
        self,
        technique_library,
        obligation_store,
        episodic_store,
        directed_graph,
        embedded_composer,
        cache_path: str = "./self_model_cache.json",
    ):
        self.library    = technique_library
        self.store      = obligation_store
        self.episodic   = episodic_store
        self.graph      = directed_graph
        self.composer   = embedded_composer
        self.cache_path = cache_path
        self._profiles: dict[str, DomainProfile] = {}
        self._computed_at: float = 0.0

    # ---- Domain profiling ----

    def compute_domain_profiles(self) -> dict[str, DomainProfile]:
        """
        Build a profile for every domain in the technique library.
        """
        techs   = self.library._data.get("techniques", {})
        domains = {t.get("domain", "") for t in techs.values() if t.get("domain")}

        profiles: dict[str, DomainProfile] = {}

        for domain in domains:
            domain_techs = {
                tid: t for tid, t in techs.items()
                if t.get("domain") == domain
            }
            n_techniques = len(domain_techs)

            # GUE score from embedding space if available
            gue_score = self._domain_gue_score(domain, domain_techs)

            # Verification success rate from episodic memory
            domain_runs = self.episodic.query_domain(domain)
            n_verified  = sum(1 for r in domain_runs if r.verified)
            avg_success = (n_verified / len(domain_runs)) if domain_runs else None

            # Persistent gaps for this domain
            try:
                all_gaps = self.store.query_persistent_gaps(top_n=200)
                n_gaps   = sum(
                    1 for g in all_gaps
                    if domain.lower() in g.text.lower()
                    or any(domain.lower() in (sb or "").lower()
                           for sb in g.source_branches)
                )
            except Exception:
                n_gaps = 0

            # Graph density
            graph_density    = self._graph_density(list(domain_techs.keys()))
            imaginary_frac   = self.graph.imaginary_fraction(
                list(domain_techs.keys())
            ) if domain_techs else 0.0

            # Freshness: how novel are recent questions in this domain?
            freshness = self._domain_freshness(domain, domain_runs)

            profile = DomainProfile(
                domain=domain,
                n_techniques=n_techniques,
                gue_score=gue_score,
                avg_success_rate=avg_success,
                n_verified_runs=len(domain_runs),
                n_persistent_gaps=n_gaps,
                graph_density=graph_density,
                imaginary_fraction=imaginary_frac,
                freshness_of_recent=freshness,
            )
            profiles[domain] = profile

        self._profiles      = profiles
        self._computed_at   = time.time()
        return profiles

    def _domain_gue_score(self, domain: str, domain_techs: dict) -> Optional[float]:
        """Estimate GUE score for a domain via embedding space."""
        if not self.composer.embedder.is_initialised or not domain_techs:
            return None
        try:
            from gue_engine import gue_score as compute_gue
            embeddings = []
            for tid, t in domain_techs.items():
                e = self.composer.embedder.embed(tid, t)
                embeddings.append(e)
            if len(embeddings) < 4:
                return None
            import numpy as np
            E  = np.stack(embeddings)
            _, sv, _ = np.linalg.svd(E, full_matrices=False)
            return compute_gue(sv)
        except Exception:
            return None

    def _graph_density(self, technique_ids: list) -> float:
        """
        What fraction of possible pairs have co-occurrence data?
        0 = no connections, 1 = fully connected.
        """
        n = len(technique_ids)
        if n < 2:
            return 0.0
        possible = n * (n - 1) / 2
        actual   = sum(
            1 for i, a in enumerate(technique_ids)
            for b in technique_ids[i+1:]
            if self.graph.co_count(a, b) > 0
        )
        return actual / possible

    def _domain_freshness(self, domain: str, domain_runs: list) -> float:
        """
        How familiar is the recent question landscape for this domain?
        Low freshness = highly familiar territory.
        High freshness = novel questions being asked.
        """
        if not domain_runs:
            return 1.0   # completely novel
        recent = sorted(domain_runs, key=lambda r: r.timestamp, reverse=True)[:5]
        fracs  = [r.structural_fraction for r in recent]
        # High structural fraction = familiar (structure handled it)
        # Low structural fraction = LLM had to generate more (novel)
        avg_structural = sum(fracs) / len(fracs) if fracs else 0
        return 1.0 - avg_structural

    # ---- Self-obligations ----

    def generate_self_obligations(self) -> list[SelfObligation]:
        """
        Identify areas where the system should improve itself.
        These become obligations that the curiosity engine can pursue.
        """
        if not self._profiles:
            self.compute_domain_profiles()

        obligations: list[SelfObligation] = []

        for domain, profile in self._profiles.items():
            # Coverage gap
            if profile.coverage_level == "sparse":
                priority = 0.9 - profile.n_techniques * 0.1
                obligations.append(SelfObligation(
                    text=(
                        f"The '{domain}' technique library has only "
                        f"{profile.n_techniques} technique(s) — coverage is sparse. "
                        f"Identify what core reasoning moves are missing "
                        f"and generate questions that would produce them."
                    ),
                    domain=domain,
                    priority=max(0.3, priority),
                    kind="coverage_gap",
                ))

            # Organisation gap
            if profile.organisation_level == "disorganised" and profile.n_techniques >= 4:
                obligations.append(SelfObligation(
                    text=(
                        f"The '{domain}' technique library has "
                        f"{profile.n_techniques} techniques but is poorly organised "
                        f"(GUE score {profile.gue_score:.2f}). "
                        f"The techniques may be redundant or missing connecting structure. "
                        f"Run consolidation and consider what compositional links are absent."
                    ),
                    domain=domain,
                    priority=0.6,
                    kind="organisation_gap",
                ))

            # Reliability gap
            if (profile.reliability_level == "unreliable"
                    and profile.n_verified_runs >= 5):
                obligations.append(SelfObligation(
                    text=(
                        f"Verification success rate in '{domain}' is "
                        f"{(profile.avg_success_rate or 0):.0%} across "
                        f"{profile.n_verified_runs} runs — consistently unreliable. "
                        f"Identify which technique types are failing and why. "
                        f"The verifier may be misconfigured or the domain may need "
                        f"different reasoning approaches."
                    ),
                    domain=domain,
                    priority=0.75,
                    kind="reliability_gap",
                ))

        # Sort by priority
        obligations.sort(key=lambda o: o.priority, reverse=True)
        return obligations

    def record_self_obligations(self, obligation_store) -> int:
        """
        Record self-obligations into the main obligation store.
        Returns number of new obligations recorded.
        """
        self_obls = self.generate_self_obligations()
        if not self_obls:
            return 0

        obligation_store.record_run(
            run_id=f"introspection-{int(time.time())}",
            domain="__self__",
            question="[system self-assessment]",
            resolution={
                "obligations": [o.to_obligation_dict() for o in self_obls],
                "global_gaps": [],
            },
        )
        return len(self_obls)

    # ---- Confidence modifiers ----

    def confidence_modifier(self, domain: str) -> float:
        """
        How much to scale confidence for claims in this domain.
        Based on the domain's coverage, organisation, and reliability.
        """
        if domain not in self._profiles:
            self.compute_domain_profiles()
        profile = self._profiles.get(domain)
        if profile is None:
            return 0.5   # unknown domain — be cautious
        return profile.confidence_modifier()

    # ---- Overall system state ----

    def system_summary(self) -> dict:
        """
        A structured summary of the system's overall self-knowledge.
        """
        if not self._profiles:
            self.compute_domain_profiles()

        mature_domains     = [d for d, p in self._profiles.items()
                               if p.overall_maturity == "mature"]
        developing_domains = [d for d, p in self._profiles.items()
                               if p.overall_maturity == "developing"]
        early_domains      = [d for d, p in self._profiles.items()
                               if p.overall_maturity == "early"]

        episodic_stats = self.episodic.statistics()

        try:
            n_gaps = self.store.summary().get("n_persistent_unresolved", 0)
        except Exception:
            n_gaps = 0

        diag = self.composer.gue_diagnostic()
        global_gue = diag.get("gue") if isinstance(diag, dict) else None

        return {
            "n_domains":            len(self._profiles),
            "mature_domains":       mature_domains,
            "developing_domains":   developing_domains,
            "early_domains":        early_domains,
            "n_verified_runs":      episodic_stats.get("n_records", 0),
            "overall_verified_rate": episodic_stats.get("verified_fraction", 0),
            "n_persistent_gaps":    n_gaps,
            "global_gue_score":     global_gue,
            "avg_structural_fraction": episodic_stats.get("avg_struct_fraction", 0),
            "computed_at":          self._computed_at,
        }

    # ---- Persistence ----

    def save(self) -> None:
        if not self._profiles:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump({
                "computed_at": self._computed_at,
                "profiles":    {d: p.to_dict()
                                for d, p in self._profiles.items()},
            }, f, indent=2)


# ============================================================
# IntrospectiveContext
# ============================================================

class IntrospectiveContext:
    """
    Formats the self-model for injection into the LLM's reasoning context.

    The LLM sees a structured description of what the system knows about
    itself — domain coverage, confidence levels, known unknowns — and
    can reason about its own limitations when proposing methods.

    This is the strange-loop: the reasoning system reasoning about
    itself using its own reasoning apparatus.
    """

    def __init__(self, self_model: SelfModel):
        self.model = self_model

    def format_for_branch_generation(
        self,
        question:    str,
        domain:      str,
        freshness:   Optional[float] = None,
    ) -> str:
        """
        Format self-knowledge as a context block to prepend to the
        branch generation prompt. The LLM sees this before reasoning.
        """
        if not self.model._profiles:
            self.model.compute_domain_profiles()

        profile = self.model._profiles.get(domain)
        summary = self.model.system_summary()

        lines = ["[SELF-KNOWLEDGE — reason about these when proposing methods]"]

        # Domain-specific awareness
        if profile:
            lines.append(f"\nAbout my knowledge in '{domain}':")
            lines.append(f"  Coverage: {profile.n_techniques} techniques "
                         f"({profile.coverage_level})")
            if profile.gue_score is not None:
                lines.append(f"  Organisation: GUE score {profile.gue_score:.2f} "
                             f"({profile.organisation_level})")
            if profile.avg_success_rate is not None:
                lines.append(f"  Reliability: {profile.avg_success_rate:.0%} "
                             f"verification rate ({profile.reliability_level})")
            lines.append(f"  Overall maturity: {profile.overall_maturity}")
            conf = profile.confidence_modifier()
            if conf < 1.0:
                lines.append(
                    f"  → Confidence modifier: {conf:.2f} "
                    f"(I am systematically less reliable in this domain — "
                    f"flag uncertainty explicitly)"
                )
            if profile.n_persistent_gaps > 0:
                lines.append(
                    f"  Known persistent gaps: {profile.n_persistent_gaps} "
                    f"unresolved obligations in this domain"
                )

        # System-wide awareness
        lines.append(f"\nAbout my overall state:")
        lines.append(f"  Verified runs: {summary['n_verified_runs']}")
        lines.append(f"  Avg structural fraction: "
                     f"{summary['avg_structural_fraction']:.0%} of recent "
                     f"branches came from structure (not raw LLM generation)")
        if summary.get("global_gue_score"):
            lines.append(f"  Global GUE score: {summary['global_gue_score']:.3f}")

        # Freshness signal
        if freshness is not None:
            if freshness < 0.2:
                lines.append(
                    f"\nThis question type is familiar "
                    f"(freshness {freshness:.2f}) — "
                    f"I have relevant prior experience to draw on."
                )
            elif freshness > 0.7:
                lines.append(
                    f"\nThis question type is novel "
                    f"(freshness {freshness:.2f}) — "
                    f"I have little prior experience here. "
                    f"Prefer conservative methods and explicit uncertainty."
                )

        # Domain comparison — cross-domain recommendation
        if profile and profile.coverage_level == "sparse":
            better_domains = [
                d for d, p in self.model._profiles.items()
                if p.overall_maturity in ("mature", "developing")
                and d != domain
            ]
            if better_domains:
                lines.append(
                    f"\nMy '{domain}' coverage is sparse. "
                    f"Consider drawing on techniques from better-covered domains "
                    f"via structural analogy: {', '.join(better_domains[:3])}"
                )

        lines.append("\n[END SELF-KNOWLEDGE]")
        return "\n".join(lines)

    def format_domain_profiles_brief(self) -> str:
        """Brief summary of all domains for system-level decisions."""
        if not self.model._profiles:
            self.model.compute_domain_profiles()
        lines = ["Domain profiles:"]
        for p in sorted(self.model._profiles.values(),
                        key=lambda x: x.n_techniques, reverse=True):
            lines.append(f"  {p.brief()}")
        return "\n".join(lines)

    def generate_introspective_branch(
        self,
        question:    str,
        domain:      str,
        llm_chat_fn,
    ) -> Optional[dict]:
        """
        Ask the system to reason explicitly about its own limitations
        relative to this question. Produces a DIAGNOSTIC branch that
        maps what is known, what is uncertain, and what would need to
        be true for a confident answer to exist.

        This branch never claims to solve the problem — it maps the
        epistemic landscape instead. Feeds into the obligation store.
        """
        context = self.format_for_branch_generation(question, domain)

        system = (
            "You are generating a DIAGNOSTIC branch — not a solution. "
            "Your task is to map the epistemic landscape: "
            "what does the system know, what is uncertain, and what would "
            "need to be true for a confident answer to be possible? "
            "Be specific about where knowledge is thin. "
            "Do not try to solve the problem — describe the structure of "
            "what solving it would require."
        )
        user   = (
            f"{context}\n\n"
            f"Question: {question}\n\n"
            f"Generate a diagnostic branch that:\n"
            f"1. States what the system knows that's relevant (be specific)\n"
            f"2. States what genuine gaps exist in its knowledge\n"
            f"3. States what assumptions would need to be verified\n"
            f"4. States what techniques are missing from the library\n"
            f"Return as JSON: {{name, method, assumptions, steps, "
            f"candidate_result, notes}}"
        )

        try:
            response = llm_chat_fn(system, user)
            import re as _re
            match = _re.search(r"\{[\s\S]*\}", response)
            if match:
                branch = json.loads(match.group(0))
                branch["provenance"] = {
                    "source": "introspective_branch",
                    "domain": domain,
                }
                branch["structural_fraction"] = 0.0
                return branch
        except Exception:
            pass
        return None
