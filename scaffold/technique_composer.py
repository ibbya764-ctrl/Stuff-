"""
technique_composer.py
=====================

Compositional retrieval for the technique library.

Where `technique_library.find_relevant_techniques` does flat keyword
matching, this module:

  1. Extracts a structural fingerprint from each technique — what
     method-type it is (derivation, transformation, ansatz, verification,
     reduction, decomposition, approximation), what kinds of obligations
     it typically addresses, what domain markers it uses.

  2. Learns a co-occurrence graph from completed runs — which techniques
     actually appear together in branches that succeeded. Two techniques
     that consistently co-occur in successful resolutions form a
     compositional unit, even if neither alone matches a new problem.

  3. Generates compositions for new problems — pick seed techniques by
     structural + text similarity, then expand via the co-occurrence
     graph to suggest combinations the system has reason to believe will
     work together. Score each composition by predicted coverage of the
     problem's inferred obligation structure.

This is the step that lets the system handle problems no single
recorded technique solves alone, by composing techniques it has
recorded individually.

Drop into the same directory as `technique_library.py`. No changes to
`technique_library.py` required — this module reads from it.
"""

import os
import re
import json
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Iterable

# ============================================================
# Method-type vocabulary
#
# Keywords overlap across types intentionally; a technique can be both
# a derivation and a reduction, for instance. The fingerprint is a
# vector, not a single label.
# ============================================================

METHOD_VOCABULARY: dict[str, set[str]] = {
    "derivation": {
        "derive", "derived", "derivation", "integrate", "integration",
        "project", "projection", "compute", "obtain", "show", "follows",
        "consequence", "starting", "first principles", "field equation",
    },
    "transformation": {
        "substitute", "substitution", "replace", "rewrite", "transform",
        "transformation", "map", "mapping", "change variable", "promote",
        "demote", "normalise", "rescale",
    },
    "ansatz": {
        "assume", "assumption", "postulate", "ansatz", "treat as",
        "take as", "stipulate", "by hypothesis", "guess", "trial form",
    },
    "verification": {
        "check", "verify", "verification", "test", "validate", "sample",
        "falsify", "counterexample", "consistency", "sanity",
    },
    "reduction": {
        "reduce", "reduction", "special case", "limit", "limiting",
        "weak field", "weak-field", "low energy", "small parameter",
        "expansion", "leading order",
    },
    "decomposition": {
        "decompose", "decomposition", "split", "separate", "partial",
        "factor", "factorise", "factorize", "isolate",
    },
    "approximation": {
        "approximate", "approximation", "perturbation", "perturbative",
        "saddle point", "stationary phase", "mean field", "linearise",
        "linearize",
    },
}

# Markers from domains.py that hint at which obligation kind a technique
# addresses (we don't import domains.py to avoid circular deps).
OBLIGATION_HINT_MARKERS: dict[str, set[str]] = {
    "assumption": {
        "assume", "assumed", "assumption", "postulate", "ansatz",
        "treat as", "take as", "by hand", "phenomenological",
    },
    "gap": {
        "missing", "gap", "not derived", "unproven", "unjustified",
        "closure", "ad hoc", "fill", "supply",
    },
    "claim_to_verify": {
        "verify", "check", "test", "validate", "confirm", "sanity",
        "consistency",
    },
}


# ============================================================
# Structural fingerprint
# ============================================================

@dataclass
class StructuralFingerprint:
    """A structural representation of a technique or a problem."""

    method_type_scores: dict[str, float]      # method_type -> [0, 1]
    obligation_addressed: dict[str, float]     # kind -> [0, 1]
    domain: str
    raw_tokens: set[str] = field(default_factory=set)

    def dominant_methods(self, threshold: float = 0.3) -> list[str]:
        return [m for m, s in self.method_type_scores.items()
                if s >= threshold]

    def to_dict(self) -> dict:
        return {
            "method_type_scores":   self.method_type_scores,
            "obligation_addressed": self.obligation_addressed,
            "domain":               self.domain,
            "n_tokens":             len(self.raw_tokens),
        }


def _tokenise(text: str) -> set[str]:
    STOPWORDS = {
        "the", "a", "an", "is", "are", "be", "to", "of", "and", "or",
        "for", "in", "on", "at", "by", "with", "from", "as", "this",
        "that", "it", "any", "all", "some",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def _score_against_vocabulary(
    text: str, vocabulary: dict[str, set[str]],
) -> dict[str, float]:
    """For each category, count keyword/phrase hits and normalise to [0, 1]."""
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for category, terms in vocabulary.items():
        hits = 0
        for term in terms:
            if " " in term:
                hits += text_lower.count(term)
            else:
                hits += len(re.findall(rf"\b{re.escape(term)}\b", text_lower))
        # Saturating function so a technique with 8 derivation hits doesn't
        # dominate one with 3 — both clearly belong, more isn't more.
        scores[category] = 1.0 - math.exp(-0.6 * hits) if hits > 0 else 0.0
    return scores


def fingerprint_technique(technique_dict: dict) -> StructuralFingerprint:
    """Build a structural fingerprint from a Technique dict (as stored)."""
    text = " ".join([
        technique_dict.get("name", ""),
        technique_dict.get("description", ""),
        technique_dict.get("when_to_use", ""),
        technique_dict.get("example_text", ""),
    ])
    return StructuralFingerprint(
        method_type_scores=_score_against_vocabulary(text, METHOD_VOCABULARY),
        obligation_addressed=_score_against_vocabulary(
            text, OBLIGATION_HINT_MARKERS
        ),
        domain=technique_dict.get("domain", ""),
        raw_tokens=_tokenise(text),
    )


def fingerprint_problem(
    question: str,
    domain: str,
    obligations: Optional[list[dict]] = None,
) -> StructuralFingerprint:
    """
    Build a structural fingerprint for a problem (a question, optionally
    with already-extracted obligations).
    """
    parts = [question]
    obligation_kinds: dict[str, float] = {}
    if obligations:
        for o in obligations:
            parts.append(o.get("text", ""))
            kind = o.get("kind", "")
            if kind:
                obligation_kinds[kind] = obligation_kinds.get(kind, 0.0) + 1.0
        # Normalise: which obligation kinds dominate this problem
        max_k = max(obligation_kinds.values()) if obligation_kinds else 1.0
        obligation_kinds = {k: v / max_k for k, v in obligation_kinds.items()}

    text = " ".join(parts)
    return StructuralFingerprint(
        method_type_scores=_score_against_vocabulary(text, METHOD_VOCABULARY),
        # For a problem, obligation_addressed represents what KIND of help
        # the problem needs — we infer it from the obligations raised.
        obligation_addressed=(obligation_kinds
                              or _score_against_vocabulary(
                                  text, OBLIGATION_HINT_MARKERS)),
        domain=domain,
        raw_tokens=_tokenise(text),
    )


# ============================================================
# Similarity functions
# ============================================================

def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def structural_similarity(
    fp_a: StructuralFingerprint,
    fp_b: StructuralFingerprint,
) -> float:
    """
    Combined similarity between two fingerprints. Three components:
      - method-type cosine (40%)
      - obligation-kind cosine (30%)
      - token jaccard (30%)
    Domain mismatch zeros it out unless one side has no domain set.
    """
    if fp_a.domain and fp_b.domain and fp_a.domain != fp_b.domain:
        return 0.0
    method_sim = _cosine(fp_a.method_type_scores, fp_b.method_type_scores)
    obl_sim    = _cosine(fp_a.obligation_addressed, fp_b.obligation_addressed)
    tok_sim    = _jaccard(fp_a.raw_tokens, fp_b.raw_tokens)
    return 0.4 * method_sim + 0.3 * obl_sim + 0.3 * tok_sim


# ============================================================
# Co-occurrence graph
# ============================================================

class CoOccurrenceGraph:
    """
    Tracks which techniques actually co-occur in successful runs.

    Stores both raw co-occurrence counts and a lift score (how much
    more often a pair co-occurs than would be expected by chance).
    """

    def __init__(self, path: str = "./technique_cooccurrence.json"):
        self.path = path
        self._data: dict = {
            "version":   1,
            "n_runs":    0,
            "n_uses":    {},   # technique_id -> int
            "co_counts": {},   # f"{a}|{b}" (sorted) -> int
        }
        self._load()

    # -------------------- I/O --------------------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not load co-occurrence graph: {e}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    # -------------------- Updates --------------------

    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        return "|".join(sorted([a, b]))

    def update_from_run(self, technique_ids: Iterable[str]) -> None:
        """
        Record a successful run as a co-occurrence event among the given
        technique ids. Self-pairs are skipped.
        """
        ids = list(set(technique_ids))
        if not ids:
            return
        self._data["n_runs"] += 1
        for tid in ids:
            self._data["n_uses"][tid] = self._data["n_uses"].get(tid, 0) + 1
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                key = self._pair_key(a, b)
                self._data["co_counts"][key] = (
                    self._data["co_counts"].get(key, 0) + 1
                )
        self.save()

    # -------------------- Queries --------------------

    def co_count(self, a: str, b: str) -> int:
        return self._data["co_counts"].get(self._pair_key(a, b), 0)

    def lift(self, a: str, b: str) -> float:
        """
        Pointwise mutual information style measure:
          P(a, b) / (P(a) * P(b))
        Lift > 1 means a and b co-occur more than expected by chance.
        """
        n_runs = self._data["n_runs"]
        if n_runs == 0:
            return 0.0
        n_a = self._data["n_uses"].get(a, 0)
        n_b = self._data["n_uses"].get(b, 0)
        n_ab = self.co_count(a, b)
        if n_a == 0 or n_b == 0 or n_ab == 0:
            return 0.0
        p_ab = n_ab / n_runs
        p_a  = n_a / n_runs
        p_b  = n_b / n_runs
        return p_ab / (p_a * p_b)

    def companions(self, tid: str, top_n: int = 5,
                   min_lift: float = 1.2) -> list[tuple[str, float]]:
        """
        Return the top-N techniques that co-occur with `tid` more than chance,
        ranked by lift. Excludes pairs with insufficient evidence.
        """
        scored: list[tuple[str, float]] = []
        for key, count in self._data["co_counts"].items():
            a, b = key.split("|", 1)
            if tid not in (a, b):
                continue
            other = b if a == tid else a
            if count < 2:  # require evidence beyond a single coincidence
                continue
            lift = self.lift(tid, other)
            if lift >= min_lift:
                scored.append((other, lift))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def summary(self) -> dict:
        return {
            "n_runs":      self._data["n_runs"],
            "n_techniques": len(self._data["n_uses"]),
            "n_pairs":      len(self._data["co_counts"]),
        }


# ============================================================
# Compositions
# ============================================================

@dataclass
class TechniqueComposition:
    """A suggested combination of techniques for a problem."""

    technique_ids: list[str]
    technique_names: list[str]
    score: float                  # overall composition score
    seed_score: float             # similarity-based score for the seed
    coverage_score: float         # predicted obligation coverage [0, 1]
    cohesion_score: float         # how often these techniques co-occur (lift)
    rationale: str

    def to_dict(self) -> dict:
        return {
            "technique_ids":   self.technique_ids,
            "technique_names": self.technique_names,
            "score":           self.score,
            "seed_score":      self.seed_score,
            "coverage_score":  self.coverage_score,
            "cohesion_score":  self.cohesion_score,
            "rationale":       self.rationale,
        }


# ============================================================
# Coverage prediction
# ============================================================

def _predicted_coverage(
    problem_fp: StructuralFingerprint,
    techniques_fps: list[StructuralFingerprint],
) -> tuple[float, dict[str, float]]:
    """
    Predict how well a set of techniques covers a problem's structural
    needs.

    Returns (overall_coverage, per_dimension_coverage).
    """
    if not techniques_fps:
        return 0.0, {}

    problem_needs = {
        **problem_fp.method_type_scores,
        **{f"obl::{k}": v for k, v in problem_fp.obligation_addressed.items()},
    }

    # For each need, find the technique that best covers it.
    coverage: dict[str, float] = {}
    for need_key, need_strength in problem_needs.items():
        if need_strength <= 0.0:
            continue
        best = 0.0
        for fp in techniques_fps:
            if need_key.startswith("obl::"):
                k = need_key[5:]
                contrib = fp.obligation_addressed.get(k, 0.0)
            else:
                contrib = fp.method_type_scores.get(need_key, 0.0)
            if contrib > best:
                best = contrib
        coverage[need_key] = best * need_strength

    # Overall: weighted average where weights are the problem's need
    # strengths. This rewards techniques that cover what the problem
    # actually needs, not what it doesn't.
    total_need = sum(v for v in problem_needs.values() if v > 0.0)
    if total_need == 0.0:
        return 0.0, coverage
    overall = sum(coverage.values()) / total_need
    return min(1.0, overall), coverage


# ============================================================
# TechniqueComposer
# ============================================================

class TechniqueComposer:
    """
    Compositional retrieval over an existing TechniqueLibrary.

    Usage:
        from technique_library import TechniqueLibrary
        from technique_composer import TechniqueComposer

        lib = TechniqueLibrary("/path/to/techniques.json")
        composer = TechniqueComposer(lib, "/path/to/cooccurrence.json")

        # Per problem:
        compositions = composer.compose_for_problem(
            question, domain="physics_mond", top_k=3,
        )
        prompt_fragment = composer.format_compositions_for_prompt(compositions)

        # After a run:
        composer.update_from_run(used_technique_ids=[...], success=True)
    """

    def __init__(
        self,
        technique_library,
        cooccurrence_path: str = "./technique_cooccurrence.json",
    ):
        self.library = technique_library
        self.graph = CoOccurrenceGraph(cooccurrence_path)
        self._fp_cache: dict[str, StructuralFingerprint] = {}

    # -------------------- Fingerprinting --------------------

    def _technique_fingerprint(self, tid: str) -> Optional[StructuralFingerprint]:
        if tid in self._fp_cache:
            return self._fp_cache[tid]
        t_dict = self.library._data["techniques"].get(tid)
        if t_dict is None:
            return None
        fp = fingerprint_technique(t_dict)
        self._fp_cache[tid] = fp
        return fp

    def _all_techniques_in_domain(self, domain: str) -> list[str]:
        return [
            tid for tid, t in self.library._data["techniques"].items()
            if t.get("domain") == domain
        ]

    # -------------------- Composition --------------------

    def compose_for_problem(
        self,
        question: str,
        domain: str,
        top_k: int = 3,
        max_per_composition: int = 3,
        obligations: Optional[list[dict]] = None,
    ) -> list[TechniqueComposition]:
        """
        Return up to `top_k` ranked compositions of techniques.

        Each composition contains 1 to `max_per_composition` techniques.
        """
        problem_fp = fingerprint_problem(question, domain, obligations)
        domain_tids = self._all_techniques_in_domain(domain)
        if not domain_tids:
            return []

        # ---- Step 1: rank seeds by structural + text similarity ----
        seed_scores: list[tuple[str, float]] = []
        for tid in domain_tids:
            fp = self._technique_fingerprint(tid)
            if fp is None:
                continue
            sim = structural_similarity(problem_fp, fp)
            if sim > 0:
                seed_scores.append((tid, sim))
        seed_scores.sort(key=lambda x: x[1], reverse=True)

        if not seed_scores:
            return []

        # Take more seeds than needed; we'll prune compositions later.
        seed_pool = seed_scores[: max(top_k * 2, 5)]

        # ---- Step 2: for each seed, expand with co-occurring partners ----
        candidates: list[TechniqueComposition] = []
        for seed_tid, seed_sim in seed_pool:
            companions = self.graph.companions(seed_tid, top_n=5)

            # Build the candidate composition: seed + best companions whose
            # fingerprints add coverage we don't yet have.
            members = [seed_tid]
            members_fps = [self._technique_fingerprint(seed_tid)]

            for comp_tid, _lift in companions:
                if comp_tid in members:
                    continue
                fp = self._technique_fingerprint(comp_tid)
                if fp is None or fp.domain != domain:
                    continue
                # Only add if it improves predicted coverage
                cur_cov, _ = _predicted_coverage(problem_fp, members_fps)
                new_cov, _ = _predicted_coverage(
                    problem_fp, members_fps + [fp]
                )
                if new_cov > cur_cov + 0.02:
                    members.append(comp_tid)
                    members_fps.append(fp)
                if len(members) >= max_per_composition:
                    break

            coverage, _ = _predicted_coverage(problem_fp, members_fps)
            cohesion = self._composition_cohesion(members)
            score = (
                0.45 * seed_sim
                + 0.35 * coverage
                + 0.20 * min(1.0, cohesion / 3.0)  # lift normalised
            )

            names = [
                self.library._data["techniques"][tid]["name"]
                for tid in members
            ]
            rationale = self._compose_rationale(
                problem_fp, members, members_fps, coverage, cohesion,
            )
            candidates.append(TechniqueComposition(
                technique_ids=members,
                technique_names=names,
                score=score,
                seed_score=seed_sim,
                coverage_score=coverage,
                cohesion_score=cohesion,
                rationale=rationale,
            ))

        # ---- Step 3: dedupe and return top_k ----
        candidates.sort(key=lambda c: c.score, reverse=True)
        deduped: list[TechniqueComposition] = []
        seen_signatures: set[tuple] = set()
        for c in candidates:
            sig = tuple(sorted(c.technique_ids))
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            deduped.append(c)
            if len(deduped) >= top_k:
                break
        return deduped

    # -------------------- Cohesion / rationale helpers --------------------

    def _composition_cohesion(self, technique_ids: list[str]) -> float:
        """Mean lift across all pairs in the composition. 0 if singleton."""
        if len(technique_ids) < 2:
            return 0.0
        lifts: list[float] = []
        for i, a in enumerate(technique_ids):
            for b in technique_ids[i + 1:]:
                lifts.append(self.graph.lift(a, b))
        return sum(lifts) / len(lifts) if lifts else 0.0

    def _compose_rationale(
        self,
        problem_fp: StructuralFingerprint,
        members: list[str],
        member_fps: list[StructuralFingerprint],
        coverage: float,
        cohesion: float,
    ) -> str:
        """Human-readable explanation of why these techniques are grouped."""
        if len(members) == 1:
            fp = member_fps[0]
            methods = fp.dominant_methods()
            method_str = (
                ", ".join(methods) if methods else "general approach"
            )
            return (
                f"Single best match by structural similarity. "
                f"Method type: {method_str}. "
                f"Predicted obligation coverage: {coverage:.0%}."
            )

        # For multi-technique compositions, explain what each contributes
        problem_methods = problem_fp.dominant_methods()
        contributions = []
        for tid, fp in zip(members, member_fps):
            t_name = self.library._data["techniques"][tid]["name"]
            methods = fp.dominant_methods()
            if methods:
                contributions.append(
                    f"'{t_name}' contributes: {', '.join(methods)}"
                )
            else:
                contributions.append(f"'{t_name}' (general)")

        cohesion_note = ""
        if cohesion >= 1.5:
            cohesion_note = (
                f" These techniques co-occur in past successful runs "
                f"(lift {cohesion:.1f}), suggesting they work together."
            )
        elif cohesion > 0:
            cohesion_note = (
                f" Modest historical co-occurrence (lift {cohesion:.1f})."
            )
        else:
            cohesion_note = (
                " No prior co-occurrence — this composition is new "
                "and exploratory."
            )

        problem_method_str = (
            ", ".join(problem_methods) if problem_methods else "general"
        )
        return (
            f"Problem signature: {problem_method_str}. "
            + ". ".join(contributions)
            + f". Predicted coverage: {coverage:.0%}."
            + cohesion_note
        )

    # -------------------- Updates from completed runs --------------------

    def update_from_run(
        self,
        used_technique_ids: Iterable[str],
        success: bool = True,
    ) -> None:
        """
        Update the co-occurrence graph from a completed run.

        Only successful runs update the graph — co-occurrence in failed
        runs is anti-evidence and we'd rather have no signal than a
        misleading one.
        """
        if not success:
            return
        self.graph.update_from_run(used_technique_ids)

    # -------------------- Prompt formatting --------------------

    def format_compositions_for_prompt(
        self,
        compositions: list[TechniqueComposition],
    ) -> str:
        """
        Drop-in replacement for technique_library.format_for_prompt that
        suggests COMPOSITIONS rather than a flat list of techniques.
        """
        if not compositions:
            return ""

        lines = [
            "Suggested technique compositions for this problem "
            "(consider whether any apply, and prefer composing techniques "
            "rather than picking one in isolation):"
        ]
        for i, c in enumerate(compositions, 1):
            lines.append(f"\n  Composition {i} "
                         f"(score: {c.score:.2f}, "
                         f"coverage: {c.coverage_score:.0%}):")
            for tid, name in zip(c.technique_ids, c.technique_names):
                t = self.library._data["techniques"].get(tid, {})
                desc = t.get("description", "")
                when = t.get("when_to_use", "")
                lines.append(f"    - **{name}**")
                if desc:
                    lines.append(f"        what: {desc}")
                if when:
                    lines.append(f"        when: {when}")
            lines.append(f"    rationale: {c.rationale}")
        return "\n".join(lines)

    # -------------------- Inspection --------------------

    def summary(self) -> dict:
        return {
            "library":      self.library.summary(),
            "cooccurrence": self.graph.summary(),
            "fp_cache_size": len(self._fp_cache),
        }
