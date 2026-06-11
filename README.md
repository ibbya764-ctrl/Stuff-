# CP² Program — AI Architecture Track

Repository for the AI-engineering side of the CP² Projective-Lorentzian research
program. Companion to the physics/mathematics program-state log (kept separately;
latest: 11 June 2026).

## Contents

| File | What it is |
|---|---|
| [`CP2_AI_Architecture_v3.md`](CP2_AI_Architecture_v3.md) | The current consolidated record: supersedes the v2 consolidated document. Contains the scaling-inefficiency diagnosis, the re-scoped negative result, the two-spectral-targets doctrine, the v3 architecture spec, the ranked "where the mathematics can earn its way in" section, and the 9-rung ablation ladder. |
| [`spectral_telemetry.py`](spectral_telemetry.py) | Numpy-only per-checkpoint diagnostics: spacing-ratio ⟨r̃⟩, spectral form factor with ramp readout, Hill/power-law α for weight ESDs, equivariance-error gate. Importable into `cp2_plssm`. |

## Quick start

```bash
python3 spectral_telemetry.py   # runs the self-test (needs numpy only)
```

Expected: GUE ⟨r̃⟩ ≈ 0.600, Poisson ≈ 0.386, SFF ramp slope ≈ 1, Hill α recovered,
equivariance error at machine precision.

## The two rules that govern everything here

1. **Two-problem firewall.** The CP² physics program and the modular/arithmetic
   program share vocabulary, not spaces — and neither is evidenced by, nor
   evidence for, any ML experiment in this repository.
2. **Two spectral targets.** GUE/level-repulsion applies to the **dynamics
   operator** only (diagnostic: SFF ramp). Trained **weight** matrices are read
   for heavy-tailed α ≈ 2 (HT-SR/SETOL) and must never be GUE-regularized.

Epistemic labels ([ESTABLISHED] / [IMPLEMENTED] / [TESTED-NEGATIVE] /
[PROPOSED] / [SPECULATIVE — gate hard]) are load-bearing; preserve them in all
derived documents.

## Highest-priority open measurement

Ablation rung 6: **Scaffold orchestration vs. the same engine called plainly**,
matched tokens. Nothing else in the program outranks it.
