# cross_domain_composer.py — usage and integration

The piece that lets the system see structural isomorphisms across
domains. A "mean-field approximation" in spin physics, a
"representative agent" model in macroeconomics, and a "well-mixed
population" assumption in epidemiology are the same abstract move.
This module recognises that.

## What it does

Adds a layer over `TechniqueComposer` that:

1. Maintains a vocabulary of **abstract structures** —
   domain-independent reasoning patterns (perturbative expansion,
   mean-field aggregation, fast-slow separation, symmetry-to-
   conservation, scaling argument, variational principle, feedback
   loop analysis, and others). Each pattern is defined by trigger
   keywords drawn from multiple domains plus example instantiations.

2. Computes an **abstract fingerprint** for techniques and problems
   alongside the existing concrete (within-domain) fingerprint.

3. Surfaces **cross-domain analogies** with explicit framing — these
   are candidate mappings the LLM has to evaluate, not prescriptions
   to apply blindly.

## Why it matters

The base composer answers "what techniques have worked on problems
like this in your domain." This answers a different and harder
question: "what techniques from *other* domains share this problem's
abstract structure?"

That's the kind of pattern matching that drives major scientific
unifications. Boltzmann's statistical mechanics is information theory.
Wiener's cybernetics is biology meets control theory. Modern ML scaling
laws look exactly like phase transitions. None of these connections
came from staying in domain — they came from someone recognising the
same structure across very different surface vocabularies.

## Standalone usage

```python
from technique_library import TechniqueLibrary
from technique_composer import TechniqueComposer
from cross_domain_composer import CrossDomainComposer

lib = TechniqueLibrary("/path/to/techniques.json")
base = TechniqueComposer(lib, "/path/to/cooccurrence.json")
cross = CrossDomainComposer(base)

# Within-domain compositions (delegates to base)
comps = cross.compose_for_problem(question, "physics_mond", top_k=3)

# Cross-domain analogies (new)
analogies = cross.find_cross_domain_analogies(
    question, problem_domain="physics_mond", top_n=3,
)

# Combined prompt fragment
fragment = cross.format_for_prompt(compositions=comps, analogies=analogies)
```

## Integration with the pipeline

In `pipeline.py`'s `_load_modules`, after the technique library is set up:

```python
from technique_composer import TechniqueComposer
from cross_domain_composer import CrossDomainComposer

self.technique_composer = TechniqueComposer(
    self.technique_library,
    cooccurrence_path="/content/drive/MyDrive/technique_cooccurrence.json",
)
self.cross_domain_composer = CrossDomainComposer(self.technique_composer)
```

Then in `_stage_context`, replace the existing technique retrieval with:

```python
compositions = self.technique_composer.compose_for_problem(
    question, self.config.domain_name,
    top_k=self.config.n_context_items,
)
analogies = self.cross_domain_composer.find_cross_domain_analogies(
    question, self.config.domain_name, top_n=2,
)
techniques_fragment = self.cross_domain_composer.format_for_prompt(
    compositions=compositions, analogies=analogies,
)
```

That's it. Everything else in the pipeline stays the same.

## How the LLM sees it

The prompt fragment for cross-domain analogies looks like this:

> CROSS-DOMAIN ANALOGIES — techniques from other domains whose
> ABSTRACT STRUCTURE matches this problem.
>
> These are CANDIDATE mappings, not direct prescriptions. For each,
> consider whether the structural correspondence actually holds in
> this problem's context. A strong structural fingerprint is evidence
> the analogy is worth examining — not proof it applies.
>
>   Analogy 1: 'mean-field theory for spin systems' (from physics)
>     Shared abstract structure: 'mean_field_aggregation' (strength 63%)
>     What this structure does: Replace pairwise or local interactions
>     with the effect of an averaged or representative interaction...
>     Examples across domains:
>       - Physics: mean-field for Ising; replace neighbour spins...
>       - Economics: representative-agent macro models
>       - Epidemiology: well-mixed SIR — no spatial structure
>       - Game theory: assume opponents play the population mix
>     Evaluate: does the move that worked in 'physics' map cleanly
>     onto your 'economics' problem? If the structural mapping is
>     valid, adapt the move; if not, note why the analogy breaks down
>     (that itself is informative).

The framing is deliberate: the LLM has to do the analogy-validation
work. We surface candidates with structural evidence; the LLM either
adapts the move or explains why the mapping fails. Both outcomes are
valuable.

## Honest limits

This is structural fingerprinting via keyword vocabularies, not
relational structure mapping in the Hofstadter / Gentner sense. A full
Structure Mapping Theory implementation would represent problems as
graphs of relations between objects and find isomorphisms between
those graphs. That's a much harder build (and a research project in
its own right).

What this module gets you is most of the practical value at a
fraction of the complexity. The 13 abstract structures cover the
majority of cross-domain reasoning patterns that appear in scientific
work. Within those patterns, it correctly identifies genuine
analogies and rejects spurious matches (verified in the smoke test).

If you ever want to go deeper, the natural next step is representing
techniques and problems as small graphs of typed relations and using
a proper graph-matching algorithm. But that's Stage 4 work — this is
the cleanest Stage 3 you can do without a research project.

## Verified behaviour

The smoke test demonstrates:

- Physics problem about many-body interactions correctly surfaces
  the economics representative-agent technique and the biology
  well-mixed population assumption — both via `mean_field_aggregation`.

- Economics problem about heterogeneous agents correctly surfaces
  the physics mean-field theory and the biology well-mixed assumption.

- ML problem about second-order optimization correctly surfaces
  the physics weak-field expansion via `perturbative_expansion`.

- Vague question with no clear abstract structure returns zero
  analogies (correctly — no spurious matches).

## For the Nicolau meeting

This is probably the most striking thing to demo. Someone whose career
spans mathematics, engineering, immunology, and biocomputing will
immediately understand the value of structural pattern matching across
domains. The "molecular agents solving NP problems" work he's done is
itself a cross-domain analogy — biological substrate, computational
structure. Show him a question from one of his domains and let the
system surface candidate analogies from another. That's a more
concrete demonstration of what the system does than any abstract
description of the architecture.
