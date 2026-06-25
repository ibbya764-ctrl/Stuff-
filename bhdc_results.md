# BHDC v1.1 — run log (CPU, web session)

Status tags per the architecture paper. Empirical column was blank before this;
these are the first runs of the BHDC line. **CPU-only environment** — full-scale
runs (§9 leave-one-out, ~100M) are reserved for the workstation GPU.

## 1. Validation gate (paper §9, step 1) — PASSED

`python3 test_bhdc.py` — all checks green:

- **Parallel scan == sequential** (§7.2) to machine tolerance: max|err| ≤ 5e-7
  across T ∈ {1, 2, 8, 17, 64}. The Hillis–Steele associative scan is validated
  against the O(T) reference loop before use, as required. [CERTIFIED — numerics]
- **Geometry bank is unitary** (§3.4): Identity / Torus / SU(2)-block / Mesh all
  norm-preserving to ≤5e-7 — the coherent sum is well defined.
- **Encoder** emits unit-norm complex states.
- **Branch 0 pinned** to the identity scenario (scale = 1) — the phase reference.
- **Smoke test**: all 8 flag combinations
  {single, incoherent, coherent, coherent-complex} × {abs2, gelu} forward and
  backprop with finite gradients at toy scale.
- A 20-step Adam run strictly decreases a toy loss.

This clears the paper's pre-large-run bug-gate. [ENGINEERING]

## 2. Grokking read (paper §9, "cheap decisive read")

Modular addition `(a + b) mod p`. Pre-registered prediction + kill condition are
in `grok_bhdc.py`. Primary arm: `collapse=coherent` (real amplitudes), |·|².

**p = 97 (the classic grok size):** each full-batch step ≈ 0.5 s on this CPU, so
a 150 s budget buys only ~300 steps — far short of the memorize→generalize
transition. Verdict at that budget: *inconclusive* (train not yet saturated).
Not a negative result — the read is simply step-bound and belongs on the GPU.
[CANDID]

**p = 31 (CPU-tractable demonstration), 50% split, AdamW lr 5e-3 wd 0.5:**

| step | loss | train_acc | val_acc | collapse_entropy |
|---:|---:|---:|---:|---:|
| 200 | 2.09 | 0.246 | 0.000 | 1.02 |
| 600 | 0.85 | 0.648 | 0.004 | 1.32 |
| 1000 | 0.81 | 0.629 | 0.002 | 1.38 |
| 1400 | 0.63 | 0.848 | 0.048 | 1.38 |
| 1800 | 0.51 | 0.921 | 0.077 | 1.38 |

(stopped on 150 s budget; train not yet saturated)

The architecture **trains stably and fits the train set** (→0.92), and val
accuracy **lifts off zero to ~0.077 (>2× the 1/p = 0.032 chance line)** — the
onset of generalization on the toroidal modular task, with the classic grokking
shape (memorize first, generalization beginning to follow). The harness cleared
its pre-registered "signal present" gate. **[PROPOSED → early signal]** — this is
onset, *not* a completed grok; whether it generalizes fully is unresolved at this
budget and is the GPU run's job.

## 3. Honest status / next actions

- The coherent core, collapse arms, parallel scan, telemetry, and the grokking
  harness are implemented, validated, and runnable.
- **Not yet done (needs GPU / matched budget):** the baseline A/B that makes any
  win *attributable* — `--collapse single` vs `coherent` vs `incoherent` vs
  `coherent-complex` at identical p / split / steps / width — and the full-step
  p=97 grok. The arms are wired; only compute is missing.
- The GR geometry coupling (§4) and the entangled two-sector state (§5) are
  **not yet built** — `DilationSpine.modular_energy` and the per-token
  density-coupled `scale` hook are stubbed in for them, but the coupling and the
  TMSV/Schmidt block remain to do. Default-off until the core proves itself,
  per the paper.
