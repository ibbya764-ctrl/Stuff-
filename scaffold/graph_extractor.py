"""
graph_extractor.py
==================

Extract a RelationalGraph from a natural-language description of a
technique or problem, using an LLM with a constrained vocabulary.

The extraction is the bottleneck for SMT-style analogy. The LLM
handles the linguistic variability; the rest of the system operates
on the structured output.

Two extraction modes:

  - extract_technique_graph(technique_dict, llm_chat_fn) — for
    techniques in the library. Result cached to disk by technique id.

  - extract_problem_graph(question, domain, llm_chat_fn) — for the
    current problem. Not cached (problems are usually one-shot).

Both produce a RelationalGraph in the canonical vocabulary defined
in relational_graph.py.
"""

import os
import re
import json
import hashlib
from typing import Callable, Optional

from relational_graph import (
    RelationalGraph, Entity, Relation,
    ENTITY_TYPES, FIRST_ORDER_RELATIONS, HIGHER_ORDER_RELATIONS,
)


# ============================================================
# Extraction prompt
# ============================================================

EXTRACTION_SYSTEM = """You extract relational structure from descriptions of
reasoning techniques or problems, for a system that does cross-domain analogy.

The output is a graph: typed entities (nodes) and typed relations (edges).
You MUST use only the canonical vocabulary below. If something doesn't fit,
pick the closest match — do NOT invent new types.

ENTITY TYPES (pick one for each entity):
  VARIABLE    — something that can change (state, observable, varying parameter)
  PARAMETER   — something fixed but determining (constant, coefficient)
  STRUCTURE   — a configuration of relations (system, model)
  PROCESS     — a temporal unfolding (dynamics, evolution)
  STATE       — a configuration at a moment (equilibrium, fixed point)
  CONSTRAINT  — a relation that must hold (closure, conservation)
  AGENT       — something that acts (decision-maker, particle, individual)
  QUANTITY    — a magnitude or measurement
  DOMAIN      — a region or scope (boundary, regime, limit, scale)
  OPERATOR    — something that transforms (function, mapping, action)
  ASSUMPTION  — an unjustified premise the reasoning rests on
  GOAL        — what is to be derived/proven/computed

FIRST-ORDER RELATIONS (entity → entity):
  DEPENDS_ON, CAUSES, CONSTRAINS, AGGREGATES_FROM, DECOMPOSES_INTO,
  BALANCES, PERTURBS, DRIVES, COUPLES_TO, APPROXIMATES, TRANSFORMS_TO,
  EQUILIBRATES_WITH, GENERATES, BOUNDS, PARAMETERISES, OPERATES_ON,
  BELONGS_TO, MEASURES

HIGHER-ORDER RELATIONS (relation → relation):
  IF_THEN          — IF (condition_relation) THEN (consequence_relation)
  BECAUSE          — P holds BECAUSE Q holds
  IN_LIMIT_OF      — relation holds in the limit of some condition
  IMPLIES_VIA      — relation P implies Q via mechanism
  INVARIANT_UNDER  — relation invariant under transformation
  BREAKS_DOWN_AT   — relation breaks down at boundary/condition

EXTRACTION GUIDELINES:
  1. Extract the ABSTRACT structure, not the surface vocabulary.
     "spin neighbours" and "consumer agents" are both AGENT.
     "temperature" and "interest rate" are both VARIABLE.

  2. Prefer fewer, deeper relations over many shallow ones.
     If two things both relate via DEPENDS_ON, ask whether one
     CAUSES the other instead — that's deeper.

  3. Use higher-order relations where they apply. "X holds in the
     weak-field limit" → IN_LIMIT_OF(relation_about_X, weak_field_domain).

  4. Entity ids should be short, descriptive, snake_case
     (e.g. "fast_var", "slow_var", "coupling_constant").

  5. Don't extract more than ~8 entities or ~10 relations. Sparse,
     deep structure beats dense, shallow structure.

OUTPUT FORMAT — strict JSON, no prose:
{
  "entities": [
    {"id": "<id>", "type": "<TYPE>", "label": "<short human description>"}
  ],
  "relations": [
    {"type": "<TYPE>", "args": ["entity_id", "entity_id"], "label": ""},
    {"type": "IF_THEN", "args": [
        {"type": "DEPENDS_ON", "args": ["a", "b"]},
        {"type": "CAUSES", "args": ["a", "c"]}
    ], "label": ""}
  ]
}
"""


def build_extraction_user_prompt(text: str, kind: str) -> str:
    return f"""
{kind} description:
\"\"\"
{text}
\"\"\"

Return only the JSON object. No prose before or after.
""".strip()


# ============================================================
# Extraction core
# ============================================================

def _parse_extraction(response: str) -> Optional[dict]:
    """Pull JSON object out of an LLM response."""
    response = response.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # Strip code fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # Find largest balanced JSON object
    match = re.search(r"\{[\s\S]*\}", response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _build_graph_from_extraction(
    parsed: dict,
    domain: str,
    source_label: str,
) -> RelationalGraph:
    """Convert parsed JSON into a RelationalGraph, with validation."""
    g = RelationalGraph(domain=domain, source_label=source_label)

    for e in parsed.get("entities", []):
        if "id" not in e or "type" not in e:
            continue
        g.add_entity(Entity(
            id_=e["id"],
            type=e["type"],
            label=e.get("label", ""),
        ))

    for r in parsed.get("relations", []):
        rel = _build_relation(r)
        if rel is not None:
            g.add_relation(rel)

    return g


def _build_relation(r: dict) -> Optional[Relation]:
    """Recursively build a Relation, supporting nested higher-order."""
    if not isinstance(r, dict) or "type" not in r:
        return None
    args_raw = r.get("args", [])
    args: list = []
    for a in args_raw:
        if isinstance(a, dict) and "type" in a:
            sub = _build_relation(a)
            if sub is not None:
                args.append(sub)
        elif isinstance(a, str):
            args.append(a)
        # ignore malformed args
    if not args:
        return None
    return Relation(
        type=r["type"],
        args=tuple(args),
        label=r.get("label", ""),
    )


# ============================================================
# Public API: extract from technique dict
# ============================================================

class GraphExtractor:
    """
    Extracts and caches relational graphs for techniques and problems.
    Cache is keyed by content hash so it survives technique edits
    (re-extraction triggered if content changes).
    """

    def __init__(
        self,
        cache_path: str = "./graph_cache.json",
        llm_chat_fn: Optional[Callable] = None,
    ):
        self.cache_path = cache_path
        self.llm_chat = llm_chat_fn
        self._cache: dict[str, dict] = {"version": 1, "graphs": {}}
        self._load()

    # -------- I/O --------

    def _load(self) -> None:
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r") as f:
                self._cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    # -------- Cache key --------

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    # -------- Public extraction methods --------

    def extract_technique_graph(
        self,
        technique_dict: dict,
        force_refresh: bool = False,
    ) -> Optional[RelationalGraph]:
        """Extract a relational graph for a technique. Cached by content hash."""
        text = " ".join([
            technique_dict.get("name", ""),
            technique_dict.get("description", ""),
            technique_dict.get("when_to_use", ""),
            technique_dict.get("example_text", ""),
        ])
        key = (technique_dict.get("technique_id", "")
               + "::"
               + self._content_hash(text))
        if not force_refresh and key in self._cache["graphs"]:
            cached = self._cache["graphs"][key]
            return RelationalGraph.from_dict(cached)

        if self.llm_chat is None:
            return None

        graph = self._extract_via_llm(
            text=text,
            domain=technique_dict.get("domain", ""),
            source_label=technique_dict.get("name", ""),
            kind="Technique",
        )
        if graph is not None:
            self._cache["graphs"][key] = graph.to_dict()
            self.save()
        return graph

    def extract_problem_graph(
        self,
        question: str,
        domain: str,
        obligations: Optional[list[dict]] = None,
    ) -> Optional[RelationalGraph]:
        """Extract a graph for the current problem. NOT cached by default
        (problems vary too much to make caching worth the storage)."""
        text = question
        if obligations:
            text += "\n\nKnown obligations:\n" + "\n".join(
                f"  - {o.get('text', '')}" for o in obligations
            )
        if self.llm_chat is None:
            return None
        return self._extract_via_llm(
            text=text,
            domain=domain,
            source_label=question[:60],
            kind="Problem",
        )

    # -------- Internal --------

    def _extract_via_llm(
        self, text: str, domain: str, source_label: str, kind: str,
    ) -> Optional[RelationalGraph]:
        user = build_extraction_user_prompt(text, kind=kind)
        try:
            response = self.llm_chat(EXTRACTION_SYSTEM, user)
        except Exception as e:
            print(f"[graph_extractor] LLM call failed: {e}")
            return None

        parsed = _parse_extraction(response)
        if parsed is None:
            print(f"[graph_extractor] Could not parse LLM output for "
                  f"'{source_label}'")
            return None

        return _build_graph_from_extraction(
            parsed, domain=domain, source_label=source_label,
        )

    # -------- Inspection --------

    def cache_summary(self) -> dict:
        return {
            "n_cached_graphs": len(self._cache["graphs"]),
            "cache_path":      self.cache_path,
        }
