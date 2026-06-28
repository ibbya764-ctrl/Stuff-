# `continuum/` — first arrows of the continuous-hypothesis-field arm

Built from `bhdc_continuous_branches_note` (**[SPECULATIVE]** throughout). The
note is explicit that it is *"a theory note, not a build order to execute
wholesale... built one arrow at a time, each validated against the last."* This
package is the **first two arrows only** — the pieces the note flags as
buildable-first **and** that can be fully validated on CPU (numpy-only) without
the GPU / Groq / Scaffold substrate a remote session lacks.

## What is here (built + validated on CPU)

| file | note § | label | what |
|---|---|---|---|
| `anti_collapse.py` | 5.7 | **[ENGINEERING]** | `AntiCollapseController` — the one anti-collapse primitive (diversity loss / load-balancing / sample repulsion are the same mechanism) made first-class, with `strength`+`mode` per call and the spread/variance collapse diagnostic implemented **once**. Modes: `soft_spread` (RBF/SVGD), `hard_balance` (load-balance), `merge_or_repel` (operator-mode redundancy). |
| `scale_dynamics.py` | 5.6 | [SPECULATIVE] | Force-driven scale-sample motion `ds/dt = α·attraction − β·repulsion + noise`. GUE-targeted **Dyson log-gas** force law vs generic **RBF kernel** vs **no-repulsion** control. |
| `eval_force_law.py` | 5.6 | — | Pre-registered A/B harness; reports SUPPORT/FALSIFY against `PREREGISTRATION.md`. |
| `PREREGISTRATION.md` | 5.6 | — | Prediction + kill condition, committed before the harness was run. |

### Run it

```
python3 -m continuum.anti_collapse        # [ENGINEERING] self-test
python3 -m continuum.scale_dynamics       # dynamics self-test
python3 -m continuum.eval_force_law       # pre-registered multi-seed A/B
python3 -m continuum.eval_force_law --quick
```

### Result on this machine (full run, 4 seeds, N=48)

| arm | `<r~>` | Var(s) | reads as |
|---|---|---|---|
| `gue` | **0.574 ± 0.004** | 23.4 | **GUE** (target 0.5996; semicircle Var ≈ 24) |
| `rbf` | 0.391 ± 0.003 | 4.0 | Poisson — distinguishable from GUE |
| `none` | 0.380 ± 0.001 | 0.53 | Poisson; contracts toward the attractor |

**VERDICT: SUPPORT** (pre-registered). The GUE-targeted force law produces GUE
level statistics in the sample positions; the generic kernel does not. This is
the note's §5.6 claim that *"use the forces"* and *"target GUE spacing"* are one
proposal — here as a directly measured mechanism (the Dyson log-gas at β=2 has
the GUE ensemble as its stationary law) rather than an analogy.

## What this does **not** do (honest scope)

- **Only the spacing half** of §5.6's falsification block. The *downstream
  task performance* half — does GUE spacing help the architecture? — needs the
  built model and the GPU/Groq/Scaffold substrate. Green here is **necessary,
  not sufficient**.
- **§3's coherence-preserving observation** (the note's load-bearing claim, the
  experiment that gates the whole branch) is **not** here: it needs the built
  complex-amplitude BHDC state; a CPU toy would *assume* the claim, not test it.
- ψ(s) sampling on the real shared operator, the low-rank branch module (canon
  on the workstation), §5.8 learned metric, §5.9 curvature, §5.10
  memory/reset — all need the substrate and are **not** started.

## Discipline carried in

- **Two-problem firewall.** The log-gas is an engineering target with a
  measurable signature, A/B'd against a generic baseline. No claim it evidences
  the physics program or vice-versa. No φ/q* constants.
- **−½ pin untouched.** These are sample *positions* on the scale axis, never
  the dynamics operator's real part.
- **Pre-registration.** Prediction + kill condition committed before running
  (`PREREGISTRATION.md`), mirroring the harness runner's refusal to run
  un-registered configs.
- **Negatives first-class.** The harness emits `FALSIFY-GUE-CLAIM` as readily
  as `SUPPORT`; a clean falsification would be recorded, not softened.

## Next arrow (when the substrate is available)

Per the note's build order: ψ(s) sampling on the shared operator (close to what
the low-rank branch module already gives), then the gated adaptive-rank
controller, then adaptive local sampling, then §3's coherence test **in
isolation before anything depends on it**, then the learned metric last. None of
those are startable from a substrate-less remote session.
