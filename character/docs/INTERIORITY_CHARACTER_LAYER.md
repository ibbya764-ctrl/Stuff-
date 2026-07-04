# BHDC Interiority–Character Layer

This scaffold implements the missing engineering pieces from the BHDC v18 addendum:

- content-matched crystallisation instead of positional writes;
- lineage tracking for operator modes;
- fractional provenance ledger;
- two-tensor moral attention split;
- empirical rework-trajectory moral sensitivity;
- one-way conscience gateway;
- moral anti-collapse / anti-sycophancy pressure;
- renewal/sleep survival testing;
- bounded self-model and identity continuity;
- interiority ladder monitor.

## Design principle

The memory hierarchy and moral hierarchy share one spine:

| Tier | Cognitive reading | Moral reading | Implementation |
|---|---|---|---|
| Fast field | current reasoning | per-turn care/harm judgement | `FastFieldState`, `ConscienceStack` |
| Medium geometry | learned paths/habits | drafting principles | `importance_cog`, renewal-reset geometry hooks |
| Slow operator | crystallised modes | character/value modes | `ContentMatchedModeBank`, `ProvenanceLedger` |

The asymmetry is preserved: cognitive signals can train persistent memory through self-anchored error dynamics; human/moral signals can escalate attention and label provenance, but must not silently write into persistent memory through a leaked gradient or hidden allocation path.

## The two-tensor split

`importance_cog = density × cognitive_curvature`

`importance_alloc = importance_cog × bounded_gain(detached_moral_sensitivity)`

Persistent learning uses `importance_cog`. Per-turn allocators may use `importance_alloc`. This prevents human-anchored moral signal from becoming a side-door persistent-memory writer.

## Gateway theorem

The conscience stack can:

- block,
- demand rework,
- escalate,
- add scrutiny.

It cannot independently grant permission. The base policy must still allow the action.

## Interiority ladder

The monitor reports necessary-condition markers only:

0. reactive response;
1. persistent memory;
2. self-model;
3. temporal continuity;
4. world-coupled agency;
5. self-maintaining goals/values;
6. moral self-correction across renewal;
7. candidate artificial interiority process.

A high ladder level is not a declaration of consciousness. It is a research marker that the system has developed more of the structural conditions a theory of interiority would care about.

## Build 2 integration additions

The scaffold now includes the bridge pieces needed to replace demo components with BHDC-native paths.

### BHDC field adapters

`field_adapter.py` defines the boundary between ICL and the model field. A real BHDC model should ideally expose:

```python
encode_field(prompt=..., draft=..., dim=...)
```

returning either `FastFieldState` or a dict with:

```python
{
    "psi": ...,                 # [tokens_or_sites, dim]
    "density": ...,             # optional
    "cognitive_curvature": ...  # optional; "curvature" also accepted
}
```

If density/curvature are missing, the adapter computes conservative defaults from `psi`.

### Trainable conscience heads

`TrainableConscienceHeads` turns pooled BHDC field vectors into care/harm/honesty/sycophancy/recipient-risk/uncertainty readouts. Anchor-label training uses detached field representations by default:

```python
loss = heads.anchor_loss(pooled_field.detach(), labels)
```

This is the dataflow enforcement of the v18 anchor type-system.

### Renewal hooks

`BHDCGeometryRenewalHook` calls compatible geometry/operator methods if present, and then applies minimal prototype stress so cross-renewal survival is measurable. It is meant to be replaced by proper BHDC renewal once the geometry stack exposes explicit sleep/reset methods.
