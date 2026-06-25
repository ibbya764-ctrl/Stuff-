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

## 3. Matched-budget A/B attribution (paper §3.5 / §8) — `ab_arms.py`

The decisive isolation: the four collapse arms at **identical** p / split / steps
/ width / seed (params are identical across arms — the flag only changes the
forward contraction). p=31, 1500 steps, AdamW lr 5e-3 wd 0.5, seed 0.

| arm | best_val_acc | train_saturated |
|---|---:|:--:|
| single (baseline) | 0.0000 | False |
| incoherent mixture | 0.0000 | False |
| **coherent (real, primary)** | **0.0644** | False |
| coherent-complex | 0.0541 | False |

`coherent − single = +0.064` (chance = 0.032).

**Reading [PROPOSED → on-thesis early signal]:** at matched budget, the coherent
arm is the *only* one that generalizes above chance; single-scenario and the
incoherent mixture both sit at val 0.0. That isolates the architecture's central
claim — it is the **inter-branch coherence** (the interference terms a softmax/
mixture cannot reproduce, §3.3), not multi-scenario averaging, that carries the
signal. coherent > coherent-complex is consistent with the paper's note that
complex amplitudes train twitchier (§3.3).

**Honest limits:** n = 1 seed; train not saturated at 1500 steps, so this is
*onset*, not a completed grok — single/incoherent could still generalize with a
longer budget. Decisive attribution wants ≥3 seeds and saturation (the
workstation/GPU run). The mechanism for cheap attribution is in place; only
compute is missing.

## 4. Entangled two-sector block (paper §5) — `bhdc_entangled.py`

The standalone TMSV/Schmidt block (§7.1). `python3 test_entangled.py` — all green:

- **Bounded by construction** (§5.3): `‖M‖_F = 1` for the Schmidt state
  `M = U_A diag(s(r)) U_B†` at every r — no SVD, no norm blow-up.
- **r = 0 → product state** (S < 0.1% of max), **r large → entangled** (S → log K);
  entanglement entropy monotone in r. The entangling knob behaves as the paper's
  squeezing strength.
- **Operator-in-the-wave-function** (§5.4): in the `eigen` basis the Schmidt
  ladder spacing IS the dilation operator's spectrum ω_n, so the operator's own
  eigenvalues set the Schmidt weights; the `generic` (uniform-ladder) basis is
  kept as the ablation contrast.
- `U_A, U_B` unitary (matrix-exp of an anti-Hermitian generator, computed once
  per forward); reduced-state eigenvalues == Schmidt occupations; finite grads
  in both bases. Entanglement entropy is emitted as telemetry (product-state
  floor = the §8 warning gate).

Default-off / standalone, per the paper — not yet wired into the combined model.

## 5. GR geometry coupling (paper §4) — `bhdc_v1_1.py::GeometryCoupling`

Default-off behind `geometry_coupling`. Relocates GR equations as the *functional
form* of an inductive bias (not a claim the net is spacetime), all knobs signed/
learnable so the data can switch them off:

- **density → dilation scale** (§4.2): `s_b → s_b·exp(−κ·ρ)`, ρ = collision
  concentration of the state — finer zoom where information is dense.
- **enriched density** (§4.4): ρ + γ·branch-disagreement (variance across
  branches) — where scenarios diverge is where the collapse carries information.
- **curvature → collapse temperature** (§4.3): τ = 1 + η·ρ, β = 1/τ, with the
  Boltzmann weighting `c_b ∝ exp(−β·⟨h_b|K|h_b⟩)` in the spine's own modular
  energy (the boost generator's expectation) — the GR and QM stages meeting at D.
- **horizon gate** (§4.4): `w = 1 + g·(σ(k·(ρ−ρ_h)) − ½)` on the residual.

Validated: with coupling on, all collapse modes forward and backprop with finite
grads; density telemetry is live; the default-off path is byte-for-byte the
prior behaviour (core tests unchanged).

## 6. Combined model (paper §7.1, third file) — `bhdc_combined.py`

The canonical training artifact: per layer `h ← CoherentBlock(h)` (collapse arms +
optional §4 coupling) then optional `h ← h + EntangledBlock(h)` (§5 pathway),
behind per-component flags. The entangled block's eigen ladder is tied **live**
to that layer's spine spectrum — the operator literally in the wave function
(§5.4). Emits the §8 telemetry and assertion gates (NaN loss/grad → hard stop;
entanglement→product and router collapse → warnings).

`python3 test_combined.py` — all green: every flag combo trains; flags toggle the
right telemetry channels; the eigen ladder follows the live spine; gates fire on
NaN / product-state / router collapse.

### Combined all-on training (p=31, 600 steps, geometry + entanglement[eigen] on)

| step | loss | train_acc | val_acc | collapse_ent | density | entangle_S | branch_load |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 150 | 2.75 | 0.069 | 0.000 | 1.33 | 0.259 | 1.384 | 0.216 |
| 300 | 0.69 | 0.723 | 0.004 | 1.21 | 0.421 | 0.428 | 0.165 |
| 450 | 0.43 | 0.927 | 0.035 | 1.31 | 0.415 | 0.438 | 0.204 |
| 600 | 0.19 | 0.923 | 0.062 | 1.38 | 0.385 | 0.409 | 0.220 |

**[PROPOSED → on-thesis]** The combined model trains stably end-to-end with every
component on. **Entanglement entropy settles at a stable nonzero ≈ 0.41** (well
above the 0.05 product-state floor) — it does **not** collapse to a product state
under gradient pressure, exactly the paper's "middle" prediction (§11, "the
entanglement entropy settles at a stable nonzero value correlated with task
structure"). Density and collapse-entropy channels stay live; branch load stays
balanced (no router collapse); **no assertion gate fired.** One seed, onset not
full grok — same honest caveats as §3.

## 7. Honest status / next actions

- **Built, validated, runnable on CPU:** coherent core + collapse arms + parallel
  scan + GR geometry coupling + telemetry (`bhdc_v1_1.py`); the entangled §5 block
  (`bhdc_entangled.py`); the combined model + §8 gates (`bhdc_combined.py`); the
  grokking harness (`grok_bhdc.py`); the matched-budget A/B (`ab_arms.py`); the
  combined demo (`combined_demo.py`). **Three validation gates green**
  (`test_bhdc.py`, `test_entangled.py`, `test_combined.py`).
- **Needs GPU / matched budget:** ≥3-seed runs to saturation; the full-step p=97
  grok; trained leave-one-out attribution across core / +geometry / +entanglement
  / all-on (the harness is ready — `bhdc_combined.py` flags + telemetry); the §5
  `eigen` vs `generic` trained contrast; the ~100M run (§9), sized d_model≈720 /
  ~14 layers.
- **Deferred by the paper:** hyperbolic/squeeze geometries (v2, firewall
  question); exp-of-squeezing entanglement (v2, needs a Gaussian-optics sim).
