# Stage 4: relational structure mapping

Three new files that implement Structure Mapping Theory (Gentner 1983)
style analogy over relational graphs:

  - `relational_graph.py` — typed graphs (entities + first-order +
    higher-order relations) using a canonical cross-domain vocabulary.
  - `graph_extractor.py`  — LLM-based extraction of graphs from
    technique descriptions and problem statements, with content-hash
    caching.
  - `structure_mapper.py` — simplified SME algorithm that finds the
    best structural mapping between two graphs and outputs explicit
    role correspondences.

This is the principled successor to `cross_domain_composer.py`. The
keyword-based abstract-structure matcher gets you most of the way;
this gets you specific role mappings and the systematicity guarantee
that comes from preserving higher-order relations.

## What's different from Stage 3

| | Stage 3 (cross_domain_composer) | Stage 4 (structure_mapper) |
|---|---|---|
| Representation | keyword vector over abstract structures | typed relational graph |
| Matching | cosine over vectors | graph isomorphism with parallel connectivity |
| Output | "this might be analogous, structure X" | explicit role mapping: "X plays the role of Y" |
| Higher-order relations | not represented | first-class, weighted higher in scoring |
| Cost | free (keyword scan) | one LLM call per technique (cached) |

The cost matters: extracting a graph requires the LLM to parse the
technique into the canonical vocabulary. Cached by content hash, so
extraction happens once per technique and survives until the
description changes.

## How it fits with the previous modules

```
TechniqueLibrary             — keyword retrieval (existing)
        ↓
TechniqueComposer            — within-domain composition
        ↓
CrossDomainComposer          — keyword-based cross-domain (Stage 3)
        ↓
StructureMapper              — relational analogy (Stage 4, this)
```

You don't have to replace any of the lower layers — they're cheaper
and serve as fast filters. A reasonable production flow:

  1. Use `TechniqueComposer` for within-domain compositions (cheap, fast).
  2. Use `CrossDomainComposer` to surface candidate cross-domain
     analogies based on keyword vocabulary (also cheap).
  3. For the top few cross-domain candidates, use `StructureMapper` to
     compute the actual role-level mapping. This is where the LLM
     extraction cost lives.

That keeps the LLM extraction cost bounded — you don't run it on every
technique in the library, only on the ones that already passed the
cheaper structural filter.

## Public API

```python
from relational_graph  import RelationalGraph, Entity, Relation
from graph_extractor   import GraphExtractor
from structure_mapper  import map_structures, format_mapping_for_prompt

# Set up extractor once
extractor = GraphExtractor(
    cache_path="/content/drive/MyDrive/graph_cache.json",
    llm_chat_fn=llm_chat,
)

# Extract graphs (cached)
source_graph = extractor.extract_technique_graph(technique_dict)
target_graph = extractor.extract_problem_graph(question, domain="economics")

# Map and format
mapping = map_structures(source_graph, target_graph)
if mapping.score >= 0.5:
    fragment = format_mapping_for_prompt(
        mapping, source_graph, target_graph,
    )
```

The `mapping` object exposes:
  - `entity_map` : `{source_id: target_id, ...}`
  - `relation_pairs` : `[(source_relation, target_relation), ...]`
  - `score` : systematicity score in [0, 1]
  - `score_components` : breakdown of relation_coverage,
    higher_order_bonus, entity_coverage

## Verified behaviour

The smoke test built two hand-crafted graphs:

**Source** (physics): predator-prey-style coupled dynamics with the
higher-order claim "coupling causes oscillation".

**Target** (economics): wage-price spiral with the higher-order claim
"feedback causes business cycle".

Result: score 1.00, all six role assignments correct
(prey↔wages, predator↔prices, growth_rate↔productivity, death_rate↔
inflation, coupling↔feedback_strength, oscillation↔business_cycle),
higher-order IF_THEN relation preserved.

A spurious unrelated graph scored 0.10 — correctly rejected.

Self-mapping scored 1.00 — sanity check passed.

## The canonical vocabulary

The vocabulary is intentionally small so the LLM uses it consistently.

**Entity types** (12): VARIABLE, PARAMETER, STRUCTURE, PROCESS, STATE,
CONSTRAINT, AGENT, QUANTITY, DOMAIN, OPERATOR, ASSUMPTION, GOAL.

**First-order relations** (18): DEPENDS_ON, CAUSES, CONSTRAINS,
AGGREGATES_FROM, DECOMPOSES_INTO, BALANCES, PERTURBS, DRIVES,
COUPLES_TO, APPROXIMATES, TRANSFORMS_TO, EQUILIBRATES_WITH, GENERATES,
BOUNDS, PARAMETERISES, OPERATES_ON, BELONGS_TO, MEASURES.

**Higher-order relations** (6): IF_THEN, BECAUSE, IN_LIMIT_OF,
IMPLIES_VIA, INVARIANT_UNDER, BREAKS_DOWN_AT.

The extractor accepts synonyms (e.g. "field" → VARIABLE, "ansatz" →
ASSUMPTION) and coerces them to the canonical form, so the LLM has
some flexibility while the matcher operates on a clean vocabulary.

## Honest limits

  - **LLM extraction quality is the bottleneck.** Sometimes the graph
    will miss a relation or pick a slightly off type. Cached, so
    you can manually correct cache entries if needed.

  - **Greedy assembly isn't always optimal.** The classical SME uses
    a more elaborate constraint satisfaction. For most cases greedy
    finds the right mapping; pathological cases with many similar
    candidates may not get the global optimum.

  - **Asymmetric matching only.** This finds the best mapping FROM
    source TO target. Reverse direction may give a different score.

  - **Scales O(|E_s| · |E_t| + |R_s| · |R_t|).** Fine for technique
    libraries up to a few hundred entries; would need indexing
    for thousands.

  - **No partial-mapping abstraction.** SME literature has work on
    "near miss" analogies where the mapping fails in a structurally
    interesting way. Not implemented here — but worth noting that
    the matcher's failures already localise where analogies break.

## For the Nicolau meeting

This is what makes the system genuinely different from anything
public. Anyone can do keyword similarity. Real role-level analogical
mapping with higher-order relation preservation is what cognitive
scientists have been trying to build into AI systems since the 1980s.
The reason it didn't go anywhere historically was that you needed
expert humans to encode the relational graphs by hand. With LLM
extraction, that bottleneck collapses — and you have a working
implementation.

The demo is concrete: pick a problem from one of his domains, let the
extractor turn it into a graph, run it against the cached technique
graphs, show him the role correspondences. That sentence — "in your
immunology problem, this T-cell population plays the role that the
predator plays in this ecology technique, and the relation that
caused oscillation in the ecology case is preserved here" — is
exactly the kind of thing a polymath like him will recognise as
real.
