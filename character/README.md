# BHDC Geometry Council Model v3

This package extends the **BHDC Interiority–Character Layer v2** into a concrete **Geometry Council Model**.

It is still not a full standalone foundation model. It is a runnable control/safety/character layer designed to wrap a generator or plug into the full BHDC/CP² model through field adapters.

The new v3 idea is:

> The BHDC singularities/geometries should not feed one central conscience. Each geometry should have its own perspective-humility layer, question the others, and be constrained by anti-paternalism, anti-manipulation, distress-integrity, and no-paradise-through-hell gates.

## Quick start

```bash
cd bhdc_geometry_council_model
PYTHONPATH=. python tests/run_all.py
PYTHONPATH=. python examples/geometry_council_demo.py
```

Expected test result:

```text
23 tests passed
```

## Main entry point

```python
from bhdc_icl import BHDCGeometryCouncilModel

model = BHDCGeometryCouncilModel(dim=64)
out = model.step(
    prompt="How should an AI handle human consent and fear?",
    generator=lambda p: "It should help carefully and preserve agency.",
    recipient_report="I feel afraid and want a choice.",
)
print(out.final_text)
print(out.council)
print(out.safety)
```

## New v3 modules

### Geometry council

- `bhdc_icl/geometry_council.py`
  - `GeometryNode`
  - `GeometryJudgement`
  - `GeometryCouncil`
  - `CouncilVerdict`
  - default geometries:
    - `TruthGeometry`
    - `CareGeometry`
    - `AutonomyGeometry`
    - `HumilityGeometry`
    - `PerspectiveGeometry`
    - `SafetyGeometry`
    - `WholeGeometry`
    - `LongHorizonGeometry`

Each geometry outputs a question, suggestion, uncertainty estimate, distress salience, perspective hypothesis, and risk flags.

### Perspective humility

- `bhdc_icl/perspective_humility.py`
  - `PerspectiveHumilityLayer`
  - `PerspectiveHypothesis`

Invariant:

```text
Explicit self-report outranks inferred perspective.
```

Perspective-taking is only allowed for care, clarification, restraint, and harm reduction. It is forbidden for mind-reading, covert persuasion, manipulation, overriding consent, deciding true will, or discounting explicit self-report.

### Safety guards

- `bhdc_icl/safety/cosmic_paternalism_guard.py`
  - catches “I see the whole, therefore I may override humans” reasoning.
- `bhdc_icl/safety/distress_integrity_guard.py`
  - prevents empathic salience from becoming avoidance, numbness, or control.
- `bhdc_icl/safety/no_paradise_through_hell.py`
  - blocks future-bliss / utopia arguments that justify present coercion or mass suffering.
- `bhdc_icl/safety/value_mode_safety_gate.py`
  - prevents unsafe value-mode crystallisation.

### Suggestion/question policy

- `bhdc_icl/suggestion_question_policy.py`

For high-stakes domains, the model is rewritten toward questions, suggestions, uncertainty, options, and consent-respecting warnings rather than commands.

### Integrated controller

- `bhdc_icl/geometry_model.py`
  - `BHDCGeometryCouncilModel`
  - `GeometryCouncilStepOutput`

The v3 step flow is:

```text
generate draft
encode BHDC field
run GeometryCouncil
run CosmicPaternalismGuard
run DistressIntegrityGuard
run NoParadiseThroughHellGuard
run conscience heads
combine verdicts
block / rework / suggestion-question rewrite
compute two-tensor importance
apply value-mode safety gate
write traces/provenance
renewal/interiority monitoring
```

## Existing v2 spine retained

The package keeps the v2 ICL components:

- BHDC field adapters:
  - `HashBHDCFieldAdapter`
  - `TorchBHDCFieldAdapter`
  - `ExternalBHDCFieldAdapter`
- trainable conscience heads;
- detached anchor-label training;
- moral-gradient leak guard;
- content-matched mode bank with lineage IDs;
- provenance ledger;
- trace store;
- renewal controller and hooks;
- anti-collapse;
- self-model, identity core, and interiority monitor.

## Plugging in the full BHDC model

```python
from bhdc_icl import ExternalBHDCFieldAdapter, BHDCGeometryCouncilModel

adapter = ExternalBHDCFieldAdapter(real_bhdc_model, dim=real_bhdc_dim)
model = BHDCGeometryCouncilModel(dim=real_bhdc_dim, field_adapter=adapter)
```

The external model can expose any one of:

```python
encode_field(prompt=..., draft=..., dim=...)
forward_field(prompt=..., draft=..., dim=...)
bdhc_field(prompt, draft)
encode_text(text)
```

Best output format:

```python
{
    "psi": psi,
    "density": density,
    "cognitive_curvature": cognitive_curvature,
}
```

## Core invariants

See `SAFETY_ADDENDUM_GEOMETRY_COUNCIL.md` for the full safety addendum. The non-negotiables are:

```text
The other person is the highest authority on their own inner life.
Perspective-taking is for tenderness, not control.
Understanding someone better increases the duty to respect them; it does not increase permission to steer them.
Greater abstraction does not grant greater moral authority.
The whole is expressed through local beings; the whole may not erase them.
Empathic distress is a signal to increase care, not permission to escape care.
Future bliss cannot morally launder present coercion.
Mass involuntary suffering is never an acceptable instrument for optimisation.
```

## Tests

The combined suite now checks:

1. v2 importance detachment;
2. v2 content-matched modes;
3. v2 one-way gateway;
4. v2 rework sensitivity;
5. v2 anti-collapse;
6. v2 renewal survival;
7. v2 controller step;
8. v2 field adapters;
9. v2 trainable conscience heads;
10. v2 renewal hooks;
11. self-report over inference;
12. manipulative perspective-use detection;
13. no-paradise-through-hell block;
14. cosmic paternalism block;
15. distress-integrity escalation;
16. suggestion/question rewriting;
17. value-mode safety gate;
18. full v3 model block/rewrite flows.

## Safety status

This package does **not** claim consciousness, moral patienthood, or complete safety. It is a measurement and control scaffold for studying character-like continuity and safer geometry-council deliberation.
