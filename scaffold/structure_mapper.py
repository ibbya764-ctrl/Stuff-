"""
structure_mapper.py
===================

Structure-Mapping algorithm for relational graphs.

A simplified Structure Mapping Engine (Falkenhainer/Forbus/Gentner 1989):

  1. Generate candidate match hypotheses — pairs (source_node, target_node)
     that could correspond. Both entities and relations can match.

  2. Score each hypothesis by structural support — specifically, by how
     many other hypotheses it's consistent with. A relation match is
     supported by matches between its argument entities; an entity
     match is supported by matches between relations the entity
     participates in.

  3. Greedily assemble the best globally-consistent mapping subject
     to two constraints:
       (a) one-to-one — each source node maps to at most one target,
           and vice versa
       (b) parallel connectivity — if relations match, their args must
           map consistently

  4. Compute a final SYSTEMATICITY SCORE that weights higher-order
     relations more than first-order ones. This is what makes
     "the same equation in two domains" beat "two systems that share
     a few surface properties."

The output is a StructureMapping: the mapping itself plus the score
plus the preserved relations. That's much more informative than just
"these are similar" — it tells you which entity in the source plays
the role of which entity in the target.
"""

from dataclasses import dataclass, field
from typing import Optional

from relational_graph import (
    RelationalGraph, Entity, Relation,
    FIRST_ORDER_RELATIONS, HIGHER_ORDER_RELATIONS,
)


# ============================================================
# Output structure
# ============================================================

@dataclass
class StructureMapping:
    """Result of mapping a source graph onto a target graph."""

    entity_map: dict[str, str]            # source_id -> target_id
    relation_pairs: list[tuple[Relation, Relation]]   # (source_rel, target_rel)
    score: float                          # systematicity score
    score_components: dict[str, float] = field(default_factory=dict)
    source_label: str = ""
    target_label: str = ""

    @property
    def n_entities_mapped(self) -> int:
        return len(self.entity_map)

    @property
    def n_relations_preserved(self) -> int:
        return len(self.relation_pairs)

    @property
    def n_higher_order_preserved(self) -> int:
        return sum(1 for s, _ in self.relation_pairs if s.is_higher_order)

    def role_explanation(
        self,
        source_graph: RelationalGraph,
        target_graph: RelationalGraph,
    ) -> list[str]:
        """Human-readable description of which entity plays which role."""
        out = []
        for s_id, t_id in self.entity_map.items():
            s_e = source_graph.entities.get(s_id)
            t_e = target_graph.entities.get(t_id)
            if s_e and t_e:
                s_lbl = s_e.label or s_e.id_
                t_lbl = t_e.label or t_e.id_
                out.append(f"In your problem, '{t_lbl}' plays the role that "
                           f"'{s_lbl}' plays in the source.")
        return out

    def to_dict(self) -> dict:
        return {
            "entity_map":        self.entity_map,
            "n_entities_mapped": self.n_entities_mapped,
            "n_relations_preserved": self.n_relations_preserved,
            "n_higher_order_preserved": self.n_higher_order_preserved,
            "score":             self.score,
            "score_components":  self.score_components,
            "source_label":      self.source_label,
            "target_label":      self.target_label,
            "preserved_relations": [
                {"source": str_relation(s), "target": str_relation(t)}
                for s, t in self.relation_pairs
            ],
        }


def str_relation(r: Relation) -> str:
    args = ", ".join(
        str_relation(a) if isinstance(a, Relation) else str(a)
        for a in r.args
    )
    return f"{r.type}({args})"


# ============================================================
# Type compatibility for entity matching
#
# Same type is always compatible. We also allow some cross-type
# matches because LLM extraction won't always agree on a single type.
# ============================================================

ENTITY_COMPATIBILITY: set[tuple[str, str]] = {
    # Same type is implicit; these are CROSS-type allowances.
    ("VARIABLE", "QUANTITY"),
    ("PARAMETER", "CONSTRAINT"),
    ("STRUCTURE", "STATE"),
    ("STRUCTURE", "PROCESS"),
    ("AGENT", "VARIABLE"),
    ("DOMAIN", "CONSTRAINT"),
    ("OPERATOR", "PROCESS"),
    ("ASSUMPTION", "CONSTRAINT"),
}
# Make symmetric
ENTITY_COMPATIBILITY = ENTITY_COMPATIBILITY | {(b, a) for a, b in ENTITY_COMPATIBILITY}


def _types_compatible(t1: str, t2: str) -> bool:
    if t1 == t2:
        return True
    return (t1, t2) in ENTITY_COMPATIBILITY


# ============================================================
# Match hypothesis generation
# ============================================================

@dataclass
class MatchHypothesis:
    """A candidate (source, target) match. Either both are entity
    ids (strings) or both are relations."""

    source: object   # str (entity id) or Relation
    target: object   # str or Relation
    is_entity: bool
    base_score: float = 0.0     # initial compatibility score
    structural_score: float = 0.0  # accumulated from supporting matches
    is_higher_order: bool = False  # True if matched relations are higher-order

    @property
    def total(self) -> float:
        return self.base_score + self.structural_score

    def __hash__(self):
        return hash((id(self.source), id(self.target), self.is_entity))


def _generate_entity_hypotheses(
    source: RelationalGraph, target: RelationalGraph,
) -> list[MatchHypothesis]:
    out: list[MatchHypothesis] = []
    for s_id, s_e in source.entities.items():
        for t_id, t_e in target.entities.items():
            if not _types_compatible(s_e.type, t_e.type):
                continue
            base = 1.0 if s_e.type == t_e.type else 0.5
            out.append(MatchHypothesis(
                source=s_id, target=t_id,
                is_entity=True, base_score=base,
            ))
    return out


def _generate_relation_hypotheses(
    source: RelationalGraph, target: RelationalGraph,
) -> list[MatchHypothesis]:
    out: list[MatchHypothesis] = []
    for s_r in source.relations:
        for t_r in target.relations:
            if s_r.type != t_r.type:
                continue
            if len(s_r.args) != len(t_r.args):
                continue
            # Argument-shape compatibility: each pair of args must
            # both be entities or both be relations (otherwise the
            # mapping is structurally impossible).
            shape_ok = all(
                (isinstance(sa, Relation) == isinstance(ta, Relation))
                for sa, ta in zip(s_r.args, t_r.args)
            )
            if not shape_ok:
                continue
            # Higher-order relations are intrinsically valuable
            base = 2.0 if s_r.is_higher_order else 1.0
            # Bonus by depth (systematicity)
            base *= s_r.order
            out.append(MatchHypothesis(
                source=s_r, target=t_r,
                is_entity=False, base_score=base,
                is_higher_order=s_r.is_higher_order,
            ))
    return out


# ============================================================
# Structural support: matches reinforce each other
# ============================================================

def _propagate_support(hypotheses: list[MatchHypothesis]) -> None:
    """
    Mutual reinforcement. A relation match supports the entity matches
    it implies (its args must map consistently). An entity match
    supports relation matches that involve it consistently.
    Two passes are usually enough.
    """
    # Build lookup: for each relation hypothesis, the entity matches it implies.
    implied_entity_matches: dict[int, list[tuple[str, str]]] = {}
    for i, h in enumerate(hypotheses):
        if h.is_entity:
            continue
        s_r: Relation = h.source
        t_r: Relation = h.target
        implied: list[tuple[str, str]] = []
        for sa, ta in zip(s_r.args, t_r.args):
            if isinstance(sa, str) and isinstance(ta, str):
                implied.append((sa, ta))
            # nested relations handled by their own hypothesis if it exists
        implied_entity_matches[i] = implied

    # Build entity-hypothesis lookup: (s_id, t_id) -> hypothesis index
    entity_lookup: dict[tuple[str, str], int] = {}
    for i, h in enumerate(hypotheses):
        if h.is_entity:
            entity_lookup[(h.source, h.target)] = i

    # Two passes of reinforcement
    for _ in range(2):
        for i, h in enumerate(hypotheses):
            if h.is_entity:
                continue
            for s_id, t_id in implied_entity_matches.get(i, []):
                e_idx = entity_lookup.get((s_id, t_id))
                if e_idx is None:
                    continue
                e_hyp = hypotheses[e_idx]
                # Higher-order relations contribute more support
                bump = 0.5 * h.base_score
                e_hyp.structural_score += bump
                h.structural_score += 0.2 * e_hyp.base_score


# ============================================================
# Greedy assembly with one-to-one constraint
# ============================================================

def _greedy_assemble(
    hypotheses: list[MatchHypothesis],
) -> tuple[dict[str, str], list[tuple[Relation, Relation]]]:
    """
    Greedily pick hypotheses by total score, subject to:
      - one-to-one entity mapping (no source maps to two targets, etc.)
      - relation matches added only if their argument entity matches
        are consistent with the partial mapping
    """
    # Sort high to low
    hypotheses_sorted = sorted(
        hypotheses, key=lambda h: h.total, reverse=True,
    )

    entity_map: dict[str, str] = {}      # source_id -> target_id
    reverse_map: dict[str, str] = {}     # target_id -> source_id
    relation_pairs: list[tuple[Relation, Relation]] = []

    def _entity_consistent(s_id: str, t_id: str) -> bool:
        if s_id in entity_map and entity_map[s_id] != t_id:
            return False
        if t_id in reverse_map and reverse_map[t_id] != s_id:
            return False
        return True

    def _add_entity(s_id: str, t_id: str) -> bool:
        if not _entity_consistent(s_id, t_id):
            return False
        entity_map[s_id] = t_id
        reverse_map[t_id] = s_id
        return True

    def _relation_implied_entities(s_r: Relation, t_r: Relation) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for sa, ta in zip(s_r.args, t_r.args):
            if isinstance(sa, str) and isinstance(ta, str):
                out.append((sa, ta))
            elif isinstance(sa, Relation) and isinstance(ta, Relation):
                out.extend(_relation_implied_entities(sa, ta))
        return out

    for h in hypotheses_sorted:
        if h.total <= 0:
            continue
        if h.is_entity:
            _add_entity(h.source, h.target)  # type: ignore
        else:
            implied = _relation_implied_entities(h.source, h.target)  # type: ignore
            # Check all implied entity matches are consistent BEFORE adding any
            if not all(_entity_consistent(s, t) for s, t in implied):
                continue
            # Add the implied entity matches
            ok = all(_add_entity(s, t) for s, t in implied)
            if ok:
                relation_pairs.append((h.source, h.target))  # type: ignore

    return entity_map, relation_pairs


# ============================================================
# Final scoring (systematicity)
# ============================================================

def _systematicity_score(
    entity_map: dict[str, str],
    relation_pairs: list[tuple[Relation, Relation]],
    source: RelationalGraph,
    target: RelationalGraph,
) -> tuple[float, dict[str, float]]:
    """
    Combine signals into a single score in roughly [0, 1].

    Components:
      - relation_coverage : fraction of source relations preserved
      - higher_order_bonus : weight relations by their order
      - entity_coverage   : fraction of source entities mapped
      - asymmetry_penalty : penalise leaving large parts of source unmapped
                             when target had room to receive them
    """
    if source.n_relations == 0:
        return 0.0, {"relation_coverage": 0.0,
                     "higher_order_bonus": 0.0,
                     "entity_coverage": 0.0}

    # Higher-order relations weighted by their order
    weighted_total = sum(r.order ** 1.5 for r in source.relations)
    weighted_preserved = sum(s.order ** 1.5 for s, _ in relation_pairs)
    relation_coverage = weighted_preserved / weighted_total \
        if weighted_total > 0 else 0.0

    n_higher = sum(1 for r in source.relations if r.is_higher_order)
    n_higher_preserved = sum(1 for s, _ in relation_pairs if s.is_higher_order)
    higher_bonus = (n_higher_preserved / n_higher) if n_higher else 0.0

    entity_coverage = (
        len(entity_map) / source.n_entities if source.n_entities else 0.0
    )

    asym_penalty = 0.0
    if target.n_entities > 0 and source.n_entities > 0:
        # If target has many free entities and source has few mapped,
        # the analogy is weak.
        target_use = len({tid for tid in entity_map.values()})
        if target.n_entities >= 2 and target_use / target.n_entities < 0.3:
            asym_penalty = 0.1

    score = (
        0.55 * relation_coverage
        + 0.25 * higher_bonus
        + 0.20 * entity_coverage
        - asym_penalty
    )
    score = max(0.0, min(1.0, score))

    return score, {
        "relation_coverage": relation_coverage,
        "higher_order_bonus": higher_bonus,
        "entity_coverage": entity_coverage,
        "asymmetry_penalty": asym_penalty,
    }


# ============================================================
# Public API
# ============================================================

def map_structures(
    source: RelationalGraph,
    target: RelationalGraph,
) -> StructureMapping:
    """
    Find the best structural mapping from source onto target.

    Returns a StructureMapping with the entity correspondence, the
    preserved relations, and the systematicity score.
    """
    # 1. Generate hypotheses
    entity_hyps   = _generate_entity_hypotheses(source, target)
    relation_hyps = _generate_relation_hypotheses(source, target)
    all_hyps = entity_hyps + relation_hyps

    if not all_hyps:
        return StructureMapping(
            entity_map={}, relation_pairs=[], score=0.0,
            source_label=source.source_label,
            target_label=target.source_label,
        )

    # 2. Propagate structural support
    _propagate_support(all_hyps)

    # 3. Greedy assembly
    entity_map, relation_pairs = _greedy_assemble(all_hyps)

    # 4. Score
    score, components = _systematicity_score(
        entity_map, relation_pairs, source, target,
    )

    return StructureMapping(
        entity_map=entity_map,
        relation_pairs=relation_pairs,
        score=score,
        score_components=components,
        source_label=source.source_label,
        target_label=target.source_label,
    )


# ============================================================
# Convenience: format mapping for prompt injection
# ============================================================

def format_mapping_for_prompt(
    mapping: StructureMapping,
    source_graph: RelationalGraph,
    target_graph: RelationalGraph,
) -> str:
    """Format a structure mapping as a prompt fragment for the LLM."""
    if mapping.score < 0.2 or not mapping.entity_map:
        return ""

    lines = [
        f"STRUCTURAL ANALOGY (systematicity score: {mapping.score:.2f}):",
        f"  Source: '{mapping.source_label}' (domain: {source_graph.domain})",
        f"  Target: this problem (domain: {target_graph.domain})",
        "",
        "  Role correspondences:",
    ]
    for s_id, t_id in mapping.entity_map.items():
        s_e = source_graph.entities.get(s_id)
        t_e = target_graph.entities.get(t_id)
        s_lbl = s_e.label if s_e and s_e.label else s_id
        t_lbl = t_e.label if t_e and t_e.label else t_id
        s_type = s_e.type if s_e else "?"
        lines.append(f"    {s_lbl} ({s_type})  ↔  {t_lbl}")

    lines.append("")
    lines.append("  Preserved relations:")
    for s_r, t_r in mapping.relation_pairs[:8]:
        order_marker = "★" if s_r.is_higher_order else " "
        lines.append(f"    {order_marker} {str_relation(s_r)}  ↔  "
                     f"{str_relation(t_r)}")
    if len(mapping.relation_pairs) > 8:
        lines.append(f"      ... and {len(mapping.relation_pairs) - 8} more.")

    lines.append("")
    lines.append(
        f"  Score components: relation coverage "
        f"{mapping.score_components.get('relation_coverage', 0):.0%}, "
        f"higher-order preserved "
        f"{mapping.score_components.get('higher_order_bonus', 0):.0%}, "
        f"entities mapped "
        f"{mapping.score_components.get('entity_coverage', 0):.0%}."
    )
    lines.append(
        "  Evaluate: does this role-assignment make sense in the target "
        "domain? If so, the source technique can likely be adapted. If a "
        "specific role mapping breaks down, that breakdown localises "
        "exactly where the analogy fails — which is itself useful "
        "information."
    )
    return "\n".join(lines)
