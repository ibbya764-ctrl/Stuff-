# `continuum/model/` — the full continuous-hypothesis-field architecture

**[STRUCTURE COMPLETE — NOT TRAINED, NOT VALIDATED.]**

Every mechanism in `bhdc_continuous_branches_note` (§0–§5.10), implemented and
**composed into one `CP2Model`** that runs an end-to-end forward pass on CPU
(numpy-only). This is the "build the full model with all architecture aspects"
deliverable — built at full strength, with the honesty labels kept.

> Read this scope statement before reading the result as more than it is. The
> note itself says: *"the most coupled and least tested thing the program has
> produced… the picture is elegant, and elegance is a warning sign here, not
> evidence."* This package is the **structure**. A green forward pass means the
> pieces compose and each behaves as specified — **not** that the architecture
> works.

## Module map (note § → file)

| § | file | label | mechanism |
|---|---|---|---|
| 0,1,3b.1 | `operator.py` | structure [IMPLEMENTED] / science [PROPOSED] | diagonal SSM spine; eigenvalues `λ=−½+iν`; **−½ pin** enforced; GUE on frequencies; dilation flow; consolidation hook |
| 1 | `field.py` | [SPECULATIVE] | continuous field `ψ(s,x)` as the operator's orbit; mode occupations |
| 2 | `density.py` | [SPECULATIVE] | density `ρ(s,x)` landscape, attraction gradient, liquidity |
| — | `branches.py` | [ENGINEERING] / [SUBSTRATE-GATED win] | low-rank branch identities; gated **adaptive rank** (§5.5 knob 1); diversity |
| 3 | `observation.py` | **[SPECULATIVE — SUBSTRATE-GATED]** | thermofield-double coherence-preserving observation; unitary-vs-projective **mechanism demo** |
| 5 | `consolidation.py` | [SPECULATIVE] | gradual consolidation; crystallize dense/coupled regions into the spine, **−½ pinned** |
| 5.5 | `resolution.py` | [SPECULATIVE] / [ENGINEERING] | stuck-signal; two knobs (adaptive rank + adaptive sampling); hard cap + commit fallback |
| 5.6 | `scale_dynamics.py` *(parent)* | [SPECULATIVE] | attraction + GUE-targeted repulsion + noise *(validated separately, see `../PREREGISTRATION.md`)* |
| 5.7 | `anti_collapse.py` *(parent)* | [ENGINEERING] | the one anti-collapse primitive |
| 5.8 | `geometry.py` | [SPECULATIVE] | learned scale-space metric; **cosmological-constant** expansion; original-coordinate collapse diagnostic |
| 5.9 | `curvature.py` | [SPECULATIVE] | information curvature; `importance = density × curvature` (A/B vs density-only) |
| 5.10 | `memory.py` | [SPECULATIVE] | three timescales; gated renewal that keeps the spine |
| — | `collapse.py` | [SPECULATIVE] | assembly `Ψ(x)=∫c(s,x)ψ(s,x)ds` from many local regions |
| 7 | `model.py` | — | `CP2Model` composing the whole pipeline |

### Run it

```
python3 -m continuum.model.run_all     # all 14 module self-tests + forward pass
python3 -m continuum.model.model       # one CP2Model.forward + telemetry
```

Forward pass exercises: GUE operator spine (`<r~>≈0.58`), §5.6 sample motion,
density/liquidity, §5.9 importance, the §3 coherence distinction (unitary
purity 1.0 vs projective ≈0.2), §5.5 stuck-signal→adaptive rank, §5
crystallization into the spine (−½ preserved), §5.8/§5.10 metric + renewal, and
the §-assembly.

## What is real vs. stand-in

**Real CPU mechanism** (runs, behaves as specified): the −½ pin and its
enforcement; GUE frequency statistics; the anti-collapse primitive; the Dyson
log-gas force law (validated in `../eval_force_law.py`); density landscape and
attraction; adaptive-rank gating + hard cap; consolidation/spine growth;
learned-metric contraction bounded by the cosmological-constant term; importance
= density × curvature; gated renewal keeping the spine; local assembly.

**Substrate-gated stand-ins** (structure present, content needs the GPU/Groq/
Scaffold + canonical `cp2_plssm`): the trained operator spectrum, the
input encoder `B` and readout `C` (random fixed here), all *semantic* content,
and — most importantly — **§3's coherence-preserving-observation claim**. The
toy in `observation.py` shows the unitary/projective *distinction* is real; it
does **not** test whether that gives "the collapse signal without the collapse
cost" for real reasoning. That remains the gating experiment, unrun.

## Discipline (carried in from CLAUDE.md and the note)

- **−½ pinned** through all consolidation — enforced by `assert` in
  `consolidation.py`, not just documented.
- **Two spectral targets**: GUE/level-repulsion read on the operator
  frequencies only; never on weights.
- **Two-problem firewall**: no φ/q* constants; no claim this evidences the
  physics program. The GUE/log-gas machinery is an engineering target with a
  measurable signature.
- **Honest labels kept** even though the build ignored the "one arrow at a
  time" pacing: `[SPECULATIVE]`/`[ENGINEERING]`/`[SUBSTRATE-GATED]` mark status,
  not permission to build.

## The one thing this does not do

It does not make any of the [SPECULATIVE] claims true. Bringing this structure
to life — training the spectrum, running §3's coherence experiment in isolation
before anything depends on it, the §5.6 downstream-performance A/B, the matched
multi-seed onset-signal verification — all need the substrate and the
pre-registered experiments. This package makes that work *possible to wire up*;
it does not substitute for it.
