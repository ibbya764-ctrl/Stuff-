# CP² Program — AI Architecture Track

Repository for the AI-engineering side of the CP² Projective-Lorentzian research
program. Companion to the physics/mathematics program-state log (kept separately;
latest: 11 June 2026).

## Contents

| File | What it is |
|---|---|
| [`CP2_AI_Architecture_v3.md`](CP2_AI_Architecture_v3.md) | The current consolidated record: supersedes the v2 consolidated document. Contains the scaling-inefficiency diagnosis, the re-scoped negative result, the two-spectral-targets doctrine, the v3 architecture spec, the ranked "where the mathematics can earn its way in" section, and the 9-rung ablation ladder. |
| [`spectral_telemetry.py`](spectral_telemetry.py) | Numpy-only per-checkpoint diagnostics: spacing-ratio ⟨r̃⟩, spectral form factor with ramp readout, Hill/power-law α for weight ESDs, equivariance-error gate. Importable into `cp2_plssm`. |
| [`harness/`](harness/) | The ladder's experiment harness: spectrum-parametrized pure-SSM science arm (`ssm.py`), long-range tasks with extrapolation evals (`tasks.py`), matched-budget runner with enforced pre-registration and per-checkpoint telemetry (`runner.py`), rung-1 optimizer three-way (`optimizers.py`), rung-6 Scaffold-vs-plain eval with calibration scoring (`eval_rung6.py`). |
| [`configs/`](configs/) | Pre-registered experiment configs (prediction + kill condition written before the run) for rungs 2–4, plus a smoke config. |

## Quick start

```bash
python3 spectral_telemetry.py                       # telemetry self-test (numpy only)
python3 -m harness.runner configs/smoke.json --quick # harness smoke test (needs torch)
python3 -m harness.optimizers --quick                # rung-1 preview (Adam/Muon/natgrad)
python3 -m harness.eval_rung6 --engine mock          # rung-6 scoring self-test (offline)
```

Real runs (workstation / Modal):

```bash
# Rung 6 — the highest-value measurement. Same suite, same token budget, both arms:
GROQ_API_KEY=... python3 -m harness.eval_rung6 --engine plain --model <groq-model>
python3 -m harness.eval_rung6 --engine http --url http://localhost:PORT/ask  # Scaffold

# Rungs 2–4 — A/B pairs at matched budget (3 seeds each via "seed" in the config):
python3 -m harness.runner configs/rung3_critical_line.json
python3 -m harness.runner configs/rung3_free_widths.json
python3 -c "from harness.runner import compare_runs; compare_runs(
    'runs/rung3_critical_line/summary.json', 'runs/rung3_free_widths/summary.json')"
```

The runner **refuses to start** unless the config carries a non-empty
`preregistration.prediction` and `.kill_condition` — write the prediction
before the experiment (the periodogram discipline).

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
