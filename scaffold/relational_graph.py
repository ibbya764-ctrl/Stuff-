"""
relational_graph.py
===================

Typed relational graphs for representing problems and techniques in a
form suitable for Structure Mapping Theory (SMT) style analogy.

The representation follows Gentner's principles:

  - Entities have types and identifiers but their specific identity
    doesn't matter for analogy — only their role in the relational
    structure.

  - Relations are typed and have arity. They take entities or other
    relations as arguments. Higher-order relations (relations whose
    arguments are themselves relations) capture the "deep" structure
    that makes analogies systematic.

  - The same graph format is used for both techniques and problems.
    This is critical: matching only works if both sides are
    represented in the same vocabulary.

The canonical vocabulary in this module is intentionally small —
about a dozen entity types and twenty relation types. Small enough
that an LLM can use it consistently; rich enough to capture the
structures that recur in scientific reasoning.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import json


# ============================================================
# Canonical vocabulary
# ============================================================

# Entity types: what KIND of thing each node represents.
# Kept deliberately small. Generic enough to span domains.

ENTITY_TYPES: set[str] = {
    "VARIABLE",      # something that can change (state, observable, parameter that varies)
    "PARAMETER",     # something fixed but determining (constants, coefficients)
    "STRUCTURE",     # a configuration of relations (a system, a model)
    "PROCESS",       # a temporal unfolding (dynamics, evolution)
    "STATE",         # a configuration at a moment (equilibrium, fixed point)
    "CONSTRAINT",    # a relation that must hold (closure, conservation)
    "AGENT",         # something that acts (decision-maker, particle)
    "QUANTITY",      # a magnitude or measurement
    "DOMAIN",        # a region or scope (boundary, regime, limit)
    "OPERATOR",      # something that transforms (function, mapping, action)
    "ASSUMPTION",    # an unjustified premise the reasoning rests on
    "GOAL",          # what is to be derived/proven/computed
}

# Relation types: how nodes RELATE.
# First-order relations connect entities. Higher-order relations
# connect relations.

FIRST_ORDER_RELATIONS: set[str] = {
    "DEPENDS_ON",         # A depends on B
    "CAUSES",             # A causes B
    "CONSTRAINS",         # A constrains B
    "AGGREGATES_FROM",    # B aggregates / averages / sums up from A
    "DECOMPOSES_INTO",    # A decomposes into B
    "BALANCES",           # A balances / is offset by B
    "PERTURBS",           # A perturbs B (small effect)
    "DRIVES",             # A drives B (large effect)
    "COUPLES_TO",         # A couples to B (mutual effect)
    "APPROXIMATES",       # A approximates B
    "TRANSFORMS_TO",      # A transforms / maps to B
    "EQUILIBRATES_WITH",  # A is in equilibrium with B
    "GENERATES",          # A generates B (B follows from A)
    "BOUNDS",             # A bounds B (limit, ceiling, floor)
    "PARAMETERISES",      # A parameterises B
    "OPERATES_ON",        # operator A operates on B
    "BELONGS_TO",         # A belongs to / is in B (domain, scale, regime)
    "MEASURES",           # A measures B
}

# Higher-order relations have relations as arguments.
HIGHER_ORDER_RELATIONS: set[str] = {
    "IF_THEN",            # IF (relation_or_condition) THEN (relation_or_consequence)
    "BECAUSE",            # P holds BECAUSE Q holds
    "IN_LIMIT_OF",        # P holds IN THE LIMIT OF condition
    "IMPLIES_VIA",        # P implies Q via mechanism/technique
    "INVARIANT_UNDER",    # P is invariant under transformation
    "BREAKS_DOWN_AT",     # relation breaks down at boundary/condition
}

ALL_RELATIONS: set[str] = FIRST_ORDER_RELATIONS | HIGHER_ORDER_RELATIONS


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class Entity:
    """A typed node in the graph. The id_ is a local label, type is
    drawn from ENTITY_TYPES."""
    id_: str
    type: str
    # Optional human-readable hint for the LLM and for debugging.
    # Not used for matching.
    label: str = ""

    def __post_init__(self):
        if self.type not in ENTITY_TYPES:
            object.__setattr__(self, "type",
                               self._coerce_type(self.type))

    @staticmethod
    def _coerce_type(t: str) -> str:
        # Allow lowercase / synonyms; coerce to canonical form.
        t_up = t.upper().replace(" ", "_").replace("-", "_")
        if t_up in ENTITY_TYPES:
            return t_up
        synonyms = {
            "FIELD": "VARIABLE",
            "OBSERVABLE": "VARIABLE",
            "COEFFICIENT": "PARAMETER",
            "CONSTANT": "PARAMETER",
            "SYSTEM": "STRUCTURE",
            "MODEL": "STRUCTURE",
            "DYNAMICS": "PROCESS",
            "EVOLUTION": "PROCESS",
            "EQUILIBRIUM": "STATE",
            "FIXED_POINT": "STATE",
            "CONDITION": "CONSTRAINT",
            "CLOSURE": "CONSTRAINT",
            "PARTICLE": "AGENT",
            "INDIVIDUAL": "AGENT",
            "SCALE": "DOMAIN",
            "REGIME": "DOMAIN",
            "LIMIT": "DOMAIN",
            "FUNCTION": "OPERATOR",
            "MAPPING": "OPERATOR",
            "PREMISE": "ASSUMPTION",
            "ANSATZ": "ASSUMPTION",
            "TARGET": "GOAL",
            "OBJECTIVE": "GOAL",
        }
        return synonyms.get(t_up, "STRUCTURE")  # safe default

    def to_dict(self) -> dict:
        return {"id": self.id_, "type": self.type, "label": self.label}


@dataclass(frozen=True)
class Relation:
    """
    A typed edge in the graph. Arguments may be Entity ids OR other
    Relations (for higher-order relations).
    """
    type: str
    args: tuple                    # tuple of (str | Relation) — entity ids or nested relations
    label: str = ""                # optional human-readable

    def __post_init__(self):
        if self.type not in ALL_RELATIONS:
            object.__setattr__(self, "type",
                               self._coerce_type(self.type))

    @staticmethod
    def _coerce_type(t: str) -> str:
        t_up = t.upper().replace(" ", "_").replace("-", "_")
        if t_up in ALL_RELATIONS:
            return t_up
        synonyms = {
            "REQUIRES": "DEPENDS_ON",
            "NEEDS": "DEPENDS_ON",
            "PRODUCES": "CAUSES",
            "RESULTS_IN": "CAUSES",
            "LIMITS": "CONSTRAINS",
            "RESTRICTS": "CONSTRAINS",
            "AVERAGES_OVER": "AGGREGATES_FROM",
            "SUMS_OVER": "AGGREGATES_FROM",
            "SPLITS_INTO": "DECOMPOSES_INTO",
            "PARTITIONS_INTO": "DECOMPOSES_INTO",
            "OFFSETS": "BALANCES",
            "CANCELS": "BALANCES",
            "INTERACTS_WITH": "COUPLES_TO",
            "MAPS_TO": "TRANSFORMS_TO",
            "REDUCES_TO": "TRANSFORMS_TO",
            "EXTENSION_OF": "GENERATES",
            "FOLLOWS_FROM": "GENERATES",
            "LIMITED_BY": "BOUNDS",
        }
        return synonyms.get(t_up, "DEPENDS_ON")  # safe default

    @property
    def is_higher_order(self) -> bool:
        return self.type in HIGHER_ORDER_RELATIONS

    @property
    def order(self) -> int:
        """Depth of nested relations. First-order = 1; relation of relations = 2; etc."""
        if not self.args:
            return 1
        max_arg_order = 0
        for a in self.args:
            if isinstance(a, Relation):
                max_arg_order = max(max_arg_order, a.order)
        return 1 + max_arg_order

    def all_entity_ids(self) -> set[str]:
        """Recursively collect entity ids referenced anywhere in this relation."""
        ids: set[str] = set()
        for a in self.args:
            if isinstance(a, Relation):
                ids.update(a.all_entity_ids())
            elif isinstance(a, str):
                ids.add(a)
        return ids

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "args": [a.to_dict() if isinstance(a, Relation) else a
                     for a in self.args],
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        args = tuple(
            cls.from_dict(a) if isinstance(a, dict) and "type" in a and "args" in a
            else a
            for a in d.get("args", [])
        )
        return cls(
            type=d["type"],
            args=args,
            label=d.get("label", ""),
        )


@dataclass
class RelationalGraph:
    """A graph: typed entities + first/higher-order relations."""

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    domain: str = ""
    source_label: str = ""        # name of technique or problem this came from

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id_] = entity

    def add_relation(self, relation: Relation) -> None:
        # Validate that all entity arguments exist
        for eid in relation.all_entity_ids():
            if eid not in self.entities:
                # Auto-add unknown entities as STRUCTURE — better than failing
                self.entities[eid] = Entity(id_=eid, type="STRUCTURE",
                                             label=eid)
        self.relations.append(relation)

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    @property
    def n_relations(self) -> int:
        return len(self.relations)

    @property
    def n_higher_order_relations(self) -> int:
        return sum(1 for r in self.relations if r.is_higher_order)

    def relations_by_order(self) -> dict[int, list[Relation]]:
        out: dict[int, list[Relation]] = {}
        for r in self.relations:
            out.setdefault(r.order, []).append(r)
        return out

    def to_dict(self) -> dict:
        return {
            "domain":       self.domain,
            "source_label": self.source_label,
            "entities":     [e.to_dict() for e in self.entities.values()],
            "relations":    [r.to_dict() for r in self.relations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelationalGraph":
        g = cls(domain=d.get("domain", ""),
                source_label=d.get("source_label", ""))
        for e_dict in d.get("entities", []):
            g.entities[e_dict["id"]] = Entity(
                id_=e_dict["id"],
                type=e_dict["type"],
                label=e_dict.get("label", ""),
            )
        for r_dict in d.get("relations", []):
            g.relations.append(Relation.from_dict(r_dict))
        return g

    def summary(self) -> str:
        lines = [f"Graph '{self.source_label}' (domain={self.domain})"]
        lines.append(f"  {self.n_entities} entities, "
                     f"{self.n_relations} relations "
                     f"({self.n_higher_order_relations} higher-order)")
        lines.append("  Entities:")
        for e in self.entities.values():
            lbl = f" — {e.label}" if e.label else ""
            lines.append(f"    {e.id_} : {e.type}{lbl}")
        lines.append("  Relations:")
        for r in self.relations:
            lines.append(f"    {_pretty_relation(r)}")
        return "\n".join(lines)


def _pretty_relation(r: Relation) -> str:
    args_str = ", ".join(
        _pretty_relation(a) if isinstance(a, Relation) else str(a)
        for a in r.args
    )
    return f"{r.type}({args_str})"
