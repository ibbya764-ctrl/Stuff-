# technique_composer.py — usage and integration

A compositional layer over your existing `TechniqueLibrary`. Doesn't
modify `technique_library.py` — reads from it and learns alongside it.

## What it does

Three improvements over flat keyword retrieval:

1. **Structural fingerprints.** Each technique gets a vector representing
   what kind of move it is (derivation, transformation, ansatz,
   verification, reduction, decomposition, approximation) and what kinds
   of obligations it tends to address (assumption, gap, claim_to_verify).
   Problems get fingerprinted the same way, so matching is structural,
   not just lexical.

2. **Co-occurrence graph.** Learns from completed runs which techniques
   actually appear together in successful resolutions. Stored
   persistently as JSON. Uses lift (PMI-style) so spurious pairings
   from a single coincidence don't get treated as evidence.

3. **Compositions.** For a new problem, picks seed techniques by
   structural similarity, then expands each seed with co-occurring
   companions whose fingerprints add coverage the seed lacks. Returns
   ranked combinations rather than a flat list.

## Files involved

- `technique_composer.py` — the new module.
- A new JSON file (path you choose) for the co-occurrence graph.
- No changes to `technique_library.py`.

## Standalone usage

```python
from technique_library import TechniqueLibrary
from technique_composer import TechniqueComposer

lib = TechniqueLibrary("/content/drive/MyDrive/technique_library.json")
composer = TechniqueComposer(
    lib,
    cooccurrence_path="/content/drive/MyDrive/technique_cooccurrence.json",
)

# For a new problem
compositions = composer.compose_for_problem(
    question="derive MOND coefficient from CP2 closure",
    domain="physics_mond",
    top_k=3,
)
print(composer.format_compositions_for_prompt(compositions))

# After a run completes successfully
composer.update_from_run(
    used_technique_ids=[tid1, tid2, tid3],
    success=True,
)
```

## Integration with the existing pipeline

The pipeline currently does this in `_stage_context`:

```python
relevant_techniques = self.technique_library.find_relevant_techniques(
    question, self.config.domain_name, top_n=self.config.n_context_items,
)
techniques_fragment = self.technique_library.format_for_prompt(
    relevant_techniques
)
```

To use compositions instead, replace those two calls with:

```python
compositions = self.technique_composer.compose_for_problem(
    question, self.config.domain_name,
    top_k=self.config.n_context_items,
)
techniques_fragment = self.technique_composer.format_compositions_for_prompt(
    compositions
)
```

And in `setup_pipeline` or `__init__`, instantiate the composer:

```python
from technique_composer import TechniqueComposer
self.technique_composer = TechniqueComposer(
    self.technique_library,
    cooccurrence_path="/content/drive/MyDrive/technique_cooccurrence.json",
)
```

Finally, the pipeline needs to update the composer after each run.
After `_stage_extract_techniques` returns the new technique ids, call:

```python
# Use techniques that were actually applied in successful branches
successful_branches = [
    b for b in branches
    if b.get("name") in branches_that_discharged_anything
]
# (You'd track which technique ids each branch invoked; for now,
#  pass the techniques that were extracted as new from this run plus
#  any retrieved-and-used techniques.)
self.technique_composer.update_from_run(
    used_technique_ids=new_technique_ids,
    success=True,
)
```

## Why this matters

The flat library answers "what worked on questions like this." The
composer answers "what combination of moves has the right shape to
cover what this problem actually needs." Two different questions, and
the second is closer to how researchers actually pull together known
techniques to handle new problems.

Concretely:

- A problem nobody has seen before, but whose obligation structure
  resembles past problems, will surface compositions that match the
  structure even if no single technique matches the surface.
- Two techniques that quietly always co-occur in successful runs but
  are never mentioned together in any technique's `when_to_use` text
  will start being suggested as a pair.
- Over time, as the co-occurrence graph fills in, the system's
  retrieved context gets richer and more specific without anyone
  having to write more techniques.

## Smoke test

The module includes a verified smoke test (run under `__main__` if
you'd like to add one) that demonstrates:

- Before learning co-occurrence: returns single-technique matches.
- After observing two techniques co-occur in 3+ successful runs: those
  two start being suggested as a composition with explicit rationale
  (`"These techniques co-occur in past successful runs (lift 1.5)"`).
- Coverage scores rise correspondingly when compositions actually
  match the problem's structural needs.

## Long-term

This is Stage 2 of the path we discussed for turning the scaffold
into a substrate. Stage 3 is the obligation store becoming a
structural model of "what's unresolved" — that's a bigger module and
a longer-term project. But the composer is the piece that takes you
from "remembered moves" to "learned combinations," which is the first
real step where the system's retained knowledge starts compounding
beyond what was explicitly taught to it.
