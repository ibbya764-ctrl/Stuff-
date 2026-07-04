# The CP²-Native AI Architecture, v3: Consolidated Record and Redesign

**Program author:** Ibby (independent researcher)
**Document type:** Third consolidated internal research record of the AI-engineering side of the CP² Projective-Lorentzian program. Supersedes the v2 consolidated record ("CP2_AI_Architecture_Consolidated"); read v2 through the supersession notes in Part 0. Companion to the physics/mathematics program-state log of 11 June 2026.
**Provenance:** Assembled from the v2 record, the 11 June 2026 physics program-state log (which contains that session's certified computations and the physics-half audit), and a fresh external-literature verification pass performed 11 June 2026. Citations checked this session are marked ✓; anything not directly confirmable retains **[VERIFY]**.

**Epistemic status labels** (unchanged from v2 — these labels are the program's most valuable convention; preserve them):

- **[ESTABLISHED]** — standard, verified external science or engineering.
- **[IMPLEMENTED]** — built by the program; exists as code.
- **[TESTED-NEGATIVE]** — built, benchmarked, and lost or failed to beat a baseline. First-class finding.
- **[PROPOSED]** — a design direction, not yet built or validated.
- **[SPECULATIVE — gate hard]** — exploratory; admitted only behind an ablation with a kill condition.

---

## TL;DR

The program's posture is unchanged: **geometry as instrument, every claim gated, negatives first-class.** What v3 changes:

1. **The anchor negative is re-scoped, not softened.** The TESTED-NEGATIVE result was a CP²-native *transformer* losing to a vanilla transformer. It kills "CP² bolted onto attention as a free win." It does **not** test spectral/geometric control of an SSM's evolution operator — which is a structurally better fit, because an SSM genuinely *is* a continuous-time dynamical system whose spectrum is the natural object. The four-gate discipline (V.3 of v2, reproduced in Part 5) still applies to every new geometric claim.
2. **The field moved toward the program's spectral thesis.** The optimizer that now trains frontier-scale models (Muon: momentum orthogonalized by Newton–Schulz = steepest descent under the *spectral norm*) is literally spectrum-control of updates. ✓ The program's "the spectrum is the right object" stance has mainstream contact for the first time. This is external vindication of the *stance*, not of any CP²-specific claim.
3. **Scaling itself is the argument for the Scaffold direction.** Frontier-model returns per FLOP are shrinking on both axes (pretraining and inference-time compute). The architectural consequences — memory/continual learning over frozen weights, test-time verification, sparsity, orchestration — are exactly the things Scaffold sketched before the field got there. Scaffold's problem was never direction; it is that it remains **unmeasured**. The ablation ladder is still the top priority.
4. **Both [VERIFY] flags from v2 are discharged.** SETOL is real (Martin & Hinrichs, arXiv:2507.17912, July 2025; α → 2 as the universal optimal exponent). ✓ The 2024 gauge-equivariant reference is real (Holland, Ipp, Müller, Wenger, *Phys. Rev. D* **110**, 074502 / arXiv:2401.06481). ✓
5. **A new section (Part 3b) ranks the places where the mathematics can earn its way in**, ordered by how *forced* the structure is — led by a genuinely new, structure-forced use of φ (Hurwitz extremality = maximal non-resonance → anti-aliasing positional structure) and a resonance-width discipline on the SSM spectrum that lands directly on HiPPO/S4 theory.

The two-problem firewall stands: the CP² physics program and the modular/arithmetic program share vocabulary, not spaces, and **nothing in this document claims either of them is solved or evidenced by an ML experiment, or vice versa.**

---

## Part 0 — What changed since v2

### 0.1 Supersession notes (read v2 through these)

1. **"A CP²-geometry neural architecture lost head-to-head to a vanilla transformer" → scope corrected.** The artifact tested was the CP²-native *transformer* (v2 Part I.1 / Part V.1). The negative therefore licenses: *CP²/SU(3) structure imposed on attention confers no advantage at matched budget.* It does not license: *geometric or spectral structure on SSM dynamics confers no advantage* — that experiment has not been run. v3 treats the latter as the open question it is, behind the same four gates. The result itself remains the program's calibration anchor and must not be softened: a beautiful prior that doesn't help looks exactly like this.
2. **v2's [VERIFY] flags on SETOL and the L-CNN 2024 citation → discharged ✓** (see 0.3).
3. **v2 Part VI backbone ("a selective SSM, Mamba-style") → split into two arms** (hybrid product arm + pure-SSM science arm), per the 2024–26 recall/copying evidence (Part 3.1).
4. **q\* = 1/φ² framing → locked per the physics log:** a defined quantity of the framework's dynamics, numerically a Q(√5) conjugate-square; its recurrence across Q(√5)-flavored constructions is *expected, not evidence*. Any architectural use is a hyperparameter to ablate. (Unchanged in substance from v2 IV.3; now carries the physics log's "locked" status.)
5. **The lepton-mass thread is fully retired on the physics side** (mechanism-level exclusion, not just look-elsewhere). No AI-side design should reference the π/e/φ mass fits for any purpose, including motivation-by-analogy.

### 0.2 What the 11 June 2026 physics log adds — and what little of it touches the AI track

The physics session's headline results (elliptic unit of the disc −20 system = φ *exactly*; explicit certified 2×2 scattering matrix for Q(√−5); the golden periodogram comb detected empirically in the 2202 rigorous LMFDB level-1 Maass forms; η(4τ)⁶ as the Gaussian scattering numerator; "RH ⟺ all modular resonance half-widths = ¼" as the sharp statement) are spectacular *for the mathematics program*. The firewall discipline requires saying plainly which of them carry any architectural content:

- **Most of them carry none.** Class-field structure, elliptic units, Hecke characters, and the specific L-function identities have no defensible ML translation. Stating this explicitly is cheaper than discovering it by building something elaborate. (The physics log's own transferable principle: *elaborateness is a warning sign, not evidence.*)
- **Three structural themes do carry over,** not as mechanisms but as design vocabulary with testable translations:
  1. **Resonance widths as the sharp invariant.** The program's deepest statements ("RH ⟺ widths = ¼"; "GRH for the four L-functions ⟺ all two-cusp half-widths = ½") are about *uniform decay rates of resonances* — a maximally rigid spectrum. In an SSM, Re(eig A) literally *is* the set of resonance widths (memory-mode decay rates). This becomes the critical-line experiment of Part 3b.1.
  2. **The two-sided trace formula.** The periodogram work needed both sides at once: a discrete geodesic comb *and* a continuous Eisenstein background that had to be modeled, not assumed flat. The translation — dual periodic-orbit/eigenspectrum views of a learned recurrence — stays [SPECULATIVE — gate hard] (Part 3b.4), but the methodological lesson is free and immediate: **when reading any spectral diagnostic of a network, model the smooth background; don't assume it flat** (this matters for SFF and Hill-α readouts in practice, Part 4).
  3. **Chaos with an off-switch.** BKL: a stiff scalar opens the billiard and *ends* the chaotic phase. Translation: chaotic/mixing dynamics need not be a permanent property of a trained model — it can be a *phase*, annealed away on schedule (Part 3b.3).
- **One result is a pure morale/discipline import:** the golden periodogram succeeded *because* the prediction was computed exactly before data contact, the controls were placed off both combs, and a 2.8σ systole was reported as "present at predicted weight, below the 5σ bar" rather than as a discovery. That is the experimental standard every ablation in Part 5 should imitate.

### 0.3 Verification pass (11 June 2026) ✓

- **Martin & Mahoney HT-SR** (arXiv:1901.08278) — established, as in v2. The operational claim used here: heavy-tailed weight spectra with fitted power-law exponent α, smaller-but-not-too-small α tracking generalization. **[ESTABLISHED]**
- **SETOL** — Martin & Hinrichs, *SETOL: A Semi-Empirical Theory of (Deep) Learning*, arXiv:2507.17912 (July 2025). Confirms the v2 claim precisely: α → 2 is identified as the universal critical/optimal value (near-optimal generalization; capturing data correlations without over- or under-fitting), derived with RMT + statistical-mechanics (single-step Wilson exact-RG condition). v2's "[VERIFY 2025 citation]" is hereby discharged. **[ESTABLISHED]** ✓
- **Gauge-equivariant CNN, the 2024 reference** — Holland, Ipp, Müller, Wenger, "Machine learning a fixed point action for SU(3) gauge theory with a gauge equivariant convolutional neural network," *Phys. Rev. D* **110**, 074502 (2024), arXiv:2401.06481. Exists exactly as the cross-AI review cited it. **[ESTABLISHED]** ✓
- **Muon optimizer** — momentum orthogonalized by Newton–Schulz iteration; equivalent view: steepest descent under the spectral norm (the simplified-Shampoo preconditioner is exactly this). Deployed at ultra-scale (Moonshot AI's Kimi line, >1T parameters). **[ESTABLISHED]** ✓
- **Hybrid-vs-pure SSM evidence** — Waleffe et al., "An Empirical Study of Mamba-based Language Models," arXiv:2406.07887: SSM-only models show degraded in-context copying/retrieval ("fuzzy memory"); an 8B Mamba-2-Hybrid *exceeds* the matched transformer by ~2.65 points averaged over 12 standard tasks while being up to 8× faster at inference. Jamba/Zamba converge on sparse attention placement (Jamba: 1:7 attention:Mamba). **[ESTABLISHED]** ✓

---

## Part 1 — The scaling-inefficiency diagnosis, and what it implies for this program

**[ESTABLISHED facts; program-specific reading marked]**

### 1.1 Why returns per FLOP are shrinking

1. **Pretraining scaling is a power law, and power laws are merciless.** Loss ∝ C^(−γ) with small γ means each constant increment of quality costs a *multiplicative* increment of compute. Nothing broke; the curve is just being ridden into its expensive region. Frontier training runs now sit at the 10²⁵–10²⁶ FLOP scale where a "noticeable" gain costs an order of magnitude.
2. **The data wall is real for text.** High-quality unique human text is bounded; past the Chinchilla-optimal regime, more compute increasingly buys re-reading and synthetic data of uncertain marginal value. The binding constraint is shifting from FLOPs to *new information per FLOP*.
3. **Inference-time compute ("reasoning") opened a second axis — with its own power law.** Letting models search/deliberate/verify at test time bought a large one-time gain, but accuracy-vs-thinking-tokens curves saturate too, and the cost is paid *per query, forever*. It is a complement, not an escape.
4. **The brain comparison the program already makes is the right one.** ~20 W, massively recurrent, sparsely activated, learning continually from a few exposures, consolidating offline. The gap between that and "frozen weights + ever-longer context windows" is not a constant factor; it is a missing set of mechanisms.

### 1.2 The architectural consequences (and where Scaffold already stands)

The field's own response to the plateau is converging on four mechanisms. Each maps onto something the program sketched in Scaffold *before* it was fashionable — which is encouraging, and changes nothing about the evidentiary status:

| Field direction | Scaffold analogue | Status in this program |
|---|---|---|
| Test-time compute spent on **verification**, not just generation | Epistemic pipeline (claim extraction → evidence weighing → confidence → answer/abstain) | [IMPLEMENTED], **unmeasured** |
| **Continual learning / memory** instead of frozen weights | DMN daemon + continual-training components | [IMPLEMENTED], **unmeasured** |
| **Sparsity / decoupling knowledge from compute** (MoE, retrieval) | — (not yet in Scaffold) | [PROPOSED] (Part 3.4) |
| **Orchestration of specialized components** | Brain modules + orchestrator | [IMPLEMENTED], **unmeasured** |

The honest conclusion is double-edged. *Directionally*, Scaffold anticipated the post-scaling agenda. *Evidentially*, it has exactly the same standing as before: plausibility-without-benchmark, the failure mode the cognitive-architecture literature (SOAR/ACT-R/LIDA lineage) has exhibited for forty years. **The single highest-value action in the entire program is still ablation (6) of the ladder: Scaffold orchestration vs. the same engine called plainly.** Nothing in this document outranks it.

### 1.3 What this means for an AGI-directed bet at workstation scale

A single 16 GB-VRAM workstation plus burst A100s cannot compete on either scaling axis — and post-plateau, *no one* gets to compete on those axes efficiently. The rational strategy for this program is therefore:

- **Compete on sample-efficiency and continual learning,** where the frontier is mechanism-poor and small-scale experiments are informative.
- **Compete on measurement,** where the program's spectral-telemetry angle (Part 4) is genuinely differentiated and publishable at any outcome.
- **Do not compete on raw capability.** The Groq-backed engine is rented capability; Scaffold's value-add must be demonstrated *relative to its own engine*, which is exactly what the ablation measures.

---

## Part 2 — External vindication of the spectral thesis [ESTABLISHED] ✓

The program's core engineering intuition has been, since v2: *the spectrum is the right object — control it on the dynamics, read it on the weights.* Two independent mainstream developments now sit exactly on that line:

1. **Muon.** The optimizer now used to train trillion-parameter frontier models replaces Adam's coordinate-wise step with **spectral-norm steepest descent**: momentum matrices are orthogonalized (Newton–Schulz polar decomposition), i.e., their singular-value spectrum is flattened to deliver a direction-only, spectrally normalized update. The reason it wins is precisely conditioning of the update spectrum. The program does not get credit for Muon — but Muon is proof that *spectrum-level control of training dynamics is where real gains were found* while parameter-count scaling stalled.
2. **SETOL.** The heavy-tailed weight-spectrum story graduated from phenomenology (HT-SR) to a semi-empirical theory with a sharp critical value: **α = 2**, with a renormalization-group-style optimality condition. The weight-side target the program adopted in v2 Part III is now better-grounded than when v2 adopted it.

**The two-spectral-targets doctrine (v2 Part III) is unchanged and is restated here as load-bearing:**

| Target | Object | Goal | Diagnostic | Desired value |
|---|---|---|---|---|
| GUE / level repulsion | **dynamics operator** (SSM A-matrix / evolution kernel) | mixing, information retention, no mode collapse | spectral form factor dip–ramp–plateau; spacing ratio | ⟨r̃⟩ ≈ 0.603 (β=2) |
| Heavy-tailed power law | **trained weight correlation matrices** | generalization | Hill/PL fit of ESD tail | α ≈ 2, within [2,4] |

These are different matrices, different goals, and they can conflict. A GUE regularizer leaking onto weights fights the property that predicts generalization. The v2 audit instruction stands: **confirm which matrix cp2_plssm's GUE regularizer touches before any other cp2_plssm work.**

**New concrete test this implies (cheap, run early):** on the program's tasks, three-way optimizer comparison — Adam(W) vs Muon-class (spectral preconditioning, no geometry) vs natural gradient (Fisher preconditioning, canonical where a CP² target exists via FS = Fisher). If Muon-class ≈ natural gradient ≫ Adam, the win is *generic spectral conditioning* and the CP²-specific story adds nothing; if natural gradient separates from Muon-class *on CP²-valued targets specifically*, that is the first positive, mechanism-localized CP² result the program would have. Either outcome is informative — which is the property every experiment here is required to have.

---

## Part 3 — Architecture v3 (each block ablatable at matched budget)

### 3.1 Backbone: two arms [PROPOSED]

- **Product arm — hybrid SSM + sparse attention.** Per the Waleffe et al. evidence ✓, pure SSMs underperform on in-context recall/copying, and ~1:7 to ~1:8 attention:SSM layer ratios recover it at small cost while keeping the inference-efficiency win. This is the arm that anything user-facing (Scaffold's engine-adjacent experiments) should use.
- **Science arm — pure SSM.** The clean laboratory: every geometric/spectral intervention of this program acts on the SSM evolution operator, and a pure-SSM stack measures those interventions with no attention confound. Worse on recall benchmarks, and that is acceptable — it is an instrument, the same epistemic category as cp2_plssm.

All CP²/spectral machinery lives in the SSM blocks and is therefore present in both arms; the arms differ only in interleaved attention. Sizing: 100M–1B parameters; LoRA/distillation for anything bigger; matched-budget baselines always trained in the same run-harness.

### 3.2 Dynamics: spectral parametrization of the evolution operator [PROPOSED]

The concrete, implementable version of v2 Part VII's transfer-operator idea — *learn the dynamics by shaping its spectrum rather than its matrix entries*:

- **Parametrize A by its spectrum directly** (diagonal-plus-structure, as in S4/S5 practice): trainable eigenvalues λ_k = −w_k + iν_k with widths w_k > 0 guaranteed by construction (softplus), frequencies ν_k free. Stability is then *structural*, not regularized — the SSM analogue of "resonances live in the lower half-plane."
- **Global spectral-measure control, Fredholm-style:** regularize the *spectral determinant* / log-det of (1 − zA) on a contour, i.e., penalize the global distribution of eigenvalues rather than entries. This is the honest translation of "the zeta function is a Fredholm determinant of the transfer operator"; it is also, deflated, just a principled global spectral regularizer — state it both ways and let the ablation decide if it earns the fancier name.
- **GUE regularization on the dynamics only,** validated by the SFF ramp (Part 4). Optional; ablate.

### 3.3 Memory: the DMN daemon made concrete [PROPOSED]

Upgrade the DMN daemon from "reflection process" to a falsifiable **replay-and-distill consolidation loop** — the program's continual-learning answer to the data wall:

1. **Episodic store:** every interaction logged with embeddings + outcome annotations (was the answer corrected? did the epistemic pipeline abstain?).
2. **Idle-time consolidation:** the daemon samples high-value episodes (errors, corrections, abstentions resolved later — prioritized replay, the established ML analogue of hippocampal replay), distills them into a LoRA adapter via supervised/preference updates on the local model.
3. **Eval-gated merge:** the adapter merges into serving weights *only* if it improves a fixed held-out suite and degrades nothing beyond a set tolerance (catastrophic-forgetting gate). Otherwise it is archived as a finding.

This makes "sleep consolidation" a measurable mechanism with a number attached (post-merge delta on the suite), rather than a neuroscience metaphor. Ladder rung (7).

### 3.4 Epistemic pipeline → verifier loop [PROPOSED]

Scaffold's claim-extraction/abstention pipeline is upgraded into the test-time-compute pattern the field now validates: generate → extract claims → *verify each claim* (retrieval, tool calls, self-consistency sampling) → answer/abstain with calibrated confidence. Two additions make it measurable:

- **Calibration as the primary metric** (Brier/ECE on a fixed claim suite), since "honesty" is the program's stated core value and calibration is its operationalization.
- **Compute-matched comparison:** the verifier loop must beat *the same engine given the same total tokens* in plain chain-of-thought mode. Spending 5× tokens to win is only interesting if 5× tokens of vanilla sampling doesn't also win. (This is the inference-time analogue of the matched-budget gate.)

### 3.5 Optimizer policy [PROPOSED, principled]

- **Natural gradient wherever a CP²-valued target exists** — canonical, since Fubini–Study = Fisher/Bures [ESTABLISHED, v2 II.7]. First CP²-specific test, unchanged from v2.
- **Muon-class spectral preconditioning elsewhere** (hidden 2D weight blocks), as current best practice ✓.
- The three-way comparison of Part 2 is ladder rung (1).

### 3.6 Equivariance [PROPOSED, with the hard gate unchanged]

SU(3)/PU(3)-equivariant layers **only** on subtasks with genuine SU(3) structure (e.g., lattice-gauge-flavored synthetic tasks where the Holland–Ipp–Müller–Wenger ✓ template applies). Hard gate verbatim from v2: equivariance error at machine precision through depth, or the inductive bias is illusory and the component is removed.

---

## Part 3b — Where the mathematics can earn its way in

*New in v3, answering the standing question: are there further places the physics/mathematics should be integrated?* Ranked by how **forced** the structure is (the program's own criterion: results count when the expression was dictated by structure before data contact). Each entry: mechanism → diagnostic → kill condition. The frame sentence comes first and is binding:

> **The architecture does not *need* more physics. It needs measurement.** Everything below enters only as a cheap, falsifiable rider on the ablation ladder; none of it blocks, replaces, or outranks rungs (1)–(7).

### 3b.1 Resonance-width discipline on the SSM spectrum [PROPOSED — most forced]

**Structure:** the physics program's sharpest statements are width-uniformity statements: RH ⟺ every modular scattering resonance has half-width exactly ¼; GRH for the four L-functions ⟺ all two-cusp half-widths = ½. A "critical-line" spectrum is one where *all resonances decay at the same rate* — maximal spectral rigidity.
**Translation (exact, not analogical):** in a linear SSM ẋ = Ax + Bu, the eigenvalues λ_k = −w_k + iν_k *are* resonances of the transfer function; w_k *is* the half-width; the memory of mode k decays as e^(−w_k t). The S4/HiPPO literature already knows that long-range memory requires eigenvalues near the imaginary axis — this proposal sharpens that to a one-parameter family worth testing:

- **Critical-line constraint:** all modes share one trainable width, w_k ≡ w (frequencies ν_k free). RH-shaped spectrum; memory is then graded purely by *frequency interference*, not by a decay hierarchy.
- vs **free widths** (standard), vs **S4-standard initialization** as control.

**Diagnostic:** long-range-dependency tasks (Long Range Arena-style, plus copying at length); spectrum plotted per checkpoint (the telemetry module reads w-dispersion directly).
**Kill condition:** if free widths match or beat the critical-line constraint at matched budget on two tasks, the discipline is decoration — record and drop. *Note the honest deflation: this tests "is width-uniformity a good inductive bias for memory," not anything about RH. The firewall sentence stays attached to the result either way.*

### 3b.2 Golden-ratio anti-resonance structure [PROPOSED — forced by Hurwitz, the one legitimate φ use]

**Structure:** φ is *the* badly-approximable number — Hurwitz's theorem makes it the extremum: rational approximations to φ are as bad as they can possibly be (the physics log's equality-case triple, √5 = Lagrange minimum). Badly-approximable = **maximally non-resonant**: rotations by the golden angle never fall into near-commensurate lock-in (this is the rigorous content of golden-angle phyllotaxis — Douady–Couder — which the physics record itself identifies as the *one* genuine φ-in-nature anchor, and of three-distance/Kronecker low-discrepancy theory).
**Translation:** positional encodings and periodic sampling patterns fail by *resonance* — frequency sets with near-rational ratios alias against each other and against data periodicities (one known failure mode of length extrapolation). Two concrete riders:

- **(a) Golden-spaced rotary/positional frequencies:** generate the RoPE/SSM frequency ladder by the golden rotation ν_{k+1}/ν_k locked to powers of φ (or angles by the golden angle) instead of the standard geometric base — the frequency set with provably worst mutual rational approximation, i.e., minimal systematic aliasing. Test: length extrapolation (train short, eval long), needle-in-haystack at length.
- **(b) Cut-and-project sparse masks:** derive sparse attention / state-sampling patterns from a Fibonacci-word/quasicrystal cut-and-project set instead of strided or block-local patterns. Aperiodic ⇒ no commensurate blind spots at any period; uniform density guaranteed by the three-distance theorem. Test vs strided/block sparse at equal sparsity.

**Why this one is privileged:** it is the only proposal in the program where φ enters through the *theorem that makes φ special* (Hurwitz extremality) acting on the *actual failure mechanism* (resonance/aliasing) — the structure dictates the expression before data contact. It is also fully aligned with the physics record's own verdict that quasicrystals/aperiodic order are the one genuine number-theory↔physics bridge.
**Kill condition:** no extrapolation/coverage win over a tuned geometric-base / strided control at matched budget on two tasks ⇒ cosmetic; record and drop. (And if it *wins*, the deflationary control is mandatory: does any sufficiently irrational ratio do as well, or is the extremal one measurably best? Only the second outcome involves φ *qua* φ.)

### 3b.3 Spectral annealing: chaos with an off-switch [SPECULATIVE — gate hard]

**Structure:** in BKL cosmology the chaotic-arithmetic phase is not forever — a stiff scalar opens the billiard and ends it (Belinskii–Khalatnikov 1973; the rigorous content of "inflation ends the arithmetic phase"). Chaos is a *phase* with an exit mechanism.
**Translation:** anneal the dynamics operator's spectral statistics over training — GUE-like early (mixing, exploratory, mode-diverse), drifting toward integrable/Poisson late (stable, settled representations). Schedule on the regularizer; diagnostic is ⟨r̃⟩ drifting 0.603 → 0.386 on schedule, SFF ramp giving way to Poisson behavior. Echoes the established edge-of-chaos initialization lore but adds a *trajectory* rather than a fixed point.
**Kill condition:** no win over (i) fixed-GUE and (ii) no-regularizer controls at matched budget ⇒ drop. Cheap: it reuses the GUE machinery of 3.2 with a schedule.

### 3b.4 Trace-formula two-sided consistency [SPECULATIVE — gate hard, deepest and riskiest]

**Structure:** the trace formula equates a spectral side with a periodic-orbit (geodesic) side; the program's periodogram succeeded by computing *both* sides and finding the comb. **Translation:** for a learned recurrence, the eigenspectrum (spectral side) and the cycle structure of its dynamics on data (orbit side — periods/return statistics of hidden-state trajectories) are dual descriptions; a consistency penalty tying them (e.g., matching the empirical SFF computed from trajectories against the one computed from the operator spectrum) would regularize the dynamics *as a dynamical system* rather than as a matrix.
**Honest status:** there is still little rigorous payoff in number-theoretic-structure-in-ML, and this remains the most likely entry on this list to be elegant and useless. Behind everything else on the ladder; kill on first matched-budget null.

### 3b.5 What stays out (closed for AI purposes; do not reopen without new structure)

Mirroring the physics log's closed-negatives discipline: **class-group/elliptic-unit/Hecke-character structure** (no ML translation; the φ-as-elliptic-unit result is mathematics, not a design input); **literal prime/zeta-based architectures** (v2's verdict stands); **q\* = 1/φ² as a privileged constant** (hyperparameter to ablate, locked framing); **any lepton-mass-adjacent motivation** (retired); **"CP² as universal learner"** (the anchor negative, correctly scoped, still kills the attention version, and no universality claim survives the physics record's own SO(4)-hydrogen/Elliott-SU(3) verdict).

---

## Part 4 — Spectral telemetry as first-class instrumentation [PROPOSED; module shipped with this record]

The cheapest differentiated thing this program can do is **measure spectra properly, per checkpoint, always** — publishable regardless of which ablations win. Shipped alongside this document as `spectral_telemetry.py` (numpy-only, importable into cp2_plssm, self-testing):

1. **Spectral form factor** K(t) of the dynamics operator, with dip–ramp–plateau readout and a fitted ramp-presence statistic — the operational GUE test (Cotler et al., arXiv:1611.04650 lineage).
2. **Spacing-ratio statistic** ⟨r̃⟩ — the cheap level-repulsion test (≈ 0.603 GUE / ≈ 0.536 GOE / ≈ 0.386 Poisson), robust to unfolding issues.
3. **Hill/power-law α estimator** for weight-matrix ESD tails (the HT-SR/SETOL ✓ generalization signal; target α ≈ 2, healthy band [2,4]).
4. **Equivariance-error probe** (hook): max deviation of f(g·x) vs g·f(x) through depth, for any layer claiming a symmetry — the v2 II.4 machine-precision gate, instrumented.

Telemetry discipline imported from the physics session (0.2): **backgrounds are modeled, not assumed flat** — the SFF readout fits and subtracts the disconnected/smooth part before calling a ramp; the Hill fit reports its k-range sensitivity instead of a single α. (The periodogram's near-collision lesson, transposed.)

**Standing audit, unchanged and still first:** instrument cp2_plssm with (1)–(3) and determine which matrix its GUE regularizer currently touches. If the evolution operator → keep, validate ramp. If it leaks into trainable weights → it is actively fighting the α ≈ 2 target; decouple. If the two targets ever conflict, **the weight-side heavy-tailed target wins** (stronger empirical link to generalization). [Verbatim policy from v2 Part III.]

---

## Part 5 — Honest AGI framing, and the ladder

### 5.1 What this program can and cannot claim about AGI

- **No geometric prior is a shortcut to general intelligence.** The program's own anchor negative — correctly scoped to attention — plus the physics record's "symmetry is system-specific" verdict, forbid the romantic version of the bet. If a CP²/spectral component ever matters, it will be as a *measured* improvement to specific mechanisms (memory, conditioning, extrapolation), compounding like any other engineering win.
- **The credible path runs through the unfashionable mechanisms:** continual learning with consolidation (3.3), calibrated verification (3.4), sample-efficiency, and orchestration that survives its own ablation (1.2). These are exactly where scaling stalled and where a workstation-scale program can contribute real evidence.
- **The differentiated bet riding on top** is the spectral program: dynamics-spectrum control, weight-spectrum reading, and the Part 3b riders. Its value is that *every experiment is informative at any outcome* — the program's negatives have repeatedly been its most transferable products.
- **The four gates (v2 V.3, verbatim, applying to every geometric claim):** (1) matched-budget win over a tuned vanilla baseline; (2) ablation-isolated (the geometry, not an incidental hyperparameter, is responsible); (3) mechanism-confirmed (the claimed diagnostic actually moves: SFF ramp, machine-precision equivariance, α ≈ 2); (4) replicated out-of-distribution. Any failure ⇒ scaffolding, not signal; record and move on.

### 5.2 The ablation ladder, v3 (run in order; each rung cheap on existing hardware; every rung returns a clean yes/no)

| # | Experiment | Reads | Kill condition |
|---|---|---|---|
| 1 | Natural gradient vs Adam vs **Muon-class** on a CP²-valued target | optimizer separation; is any win generic-spectral or CP²-specific? | nat-grad ≤ Muon-class on CP² targets ⇒ FS=Fisher adds nothing in practice |
| 2 | GUE-reg on dynamics vs none (science arm) | SFF ramp; ⟨r̃⟩ | no ramp ⇒ regularizer not working; ramp but no task win ⇒ diagnostic only |
| 3 | **Critical-line width constraint** vs free widths vs S4-init (3b.1) | long-range memory tasks | free ≥ constrained ⇒ drop |
| 4 | **Golden RoPE + cut-and-project masks** vs geometric-base/strided (3b.2) | length extrapolation, coverage | no win ⇒ cosmetic; win ⇒ run the any-irrational control |
| 5 | Hybrid vs pure-SSM at matched budget (3.1) | recall/copying gap on our tasks | (calibration rung — establishes the two arms' baseline gap) |
| 6 | **Scaffold orchestration vs plain engine** (same model, same tokens) | the program's biggest unmeasured claim | no win ⇒ strip modules until the winning subset is found, ship only that |
| 7 | DMN replay-distill-merge vs none (3.3) | post-merge delta on fixed suite; forgetting gate | no positive delta ⇒ daemon is housekeeping, not learning |
| 8 | SU(3)-equivariance vs augmentation on a genuinely SU(3) task (3.6) | equivariance error through depth + task win | precision drift or no win ⇒ remove |
| 9 | Spectral annealing schedule vs fixed-GUE vs none (3b.3) | ⟨r̃⟩ trajectory + task | no win over both controls ⇒ drop |

Rung 6 may be run first whenever engine access makes it convenient — it is the highest-value measurement in the program and depends on nothing above it.

### 5.3 Standing discipline (imported whole; it is the program)

Two-problem firewall. Separate accounts. Forced numbers only. Negatives are first-class. Verification ≠ discovery. Elaborateness is a warning sign, not evidence. The TESTED-NEGATIVE anchor is calibration, not embarrassment. **A downstream system reading this document should know exactly what is built, what is established, what was tried and lost, and what is merely proposed — preserve the labels.**

---

## Part 6 — The character/council layer (BHDC v18) [IMPLEMENTED infra; SPECULATIVE identity]

Added 2026-07-04. The `character/` package (module `bhdc_icl`) vendors the BHDC
Geometry Council model — the program's alignment layer — and the v18 theory
notes (`character/docs/BHDC_v18_one_model.md`, `..._addendum_...md`). The v18
thesis is that the memory hierarchy (field / geometry / operator) and the moral
hierarchy (per-turn judgment / drafting principles / disposition) are **one**
hierarchy read at two anchors: cognitive signals anchor to the model's own
error dynamics (self-anchored, consolidate freely); moral signals anchor to the
human-grounded label and keep the external audit (human-anchored). This layer
was reviewed on import (2026-07-04, four-lens adversarial pass, 23 verified
findings); the fixes below were applied before it was recorded here.

### 6.1 What is built and what its status is

| Component | What | Status |
|---|---|---|
| Two-tensor importance split (`importance_cog` / `importance_alloc`) | keeps human-anchored moral gain out of self-anchored persistent learning | **[IMPLEMENTED]**, gradient-tested |
| One-way action gateway | conscience may block/rework/escalate, never grant permission; base-policy denial wins over conscience escalation | **[IMPLEMENTED]** (base-policy-first bug fixed on import) |
| Content-matched mode bank + lineage + probe-basis stability | replaces positional crystallization; the addendum's M0 (Erratum 3 fix) | **[IMPLEMENTED]** — the highest-quality piece; "records before signals" |
| Dual-channel provenance ledger + monotone scrutiny latch | tracks anchored vs cognitive write fractions; cognitive writes cannot erode a mode's anchored status (one-way rule enforced in storage) | **[IMPLEMENTED]** (anchor type-system moved from bookkeeping into storage on import) |
| Anti-collapse / anti-sycophancy | sycophancy = moral mode collapse; committee-diversity loss detached from the field encoder | **[IMPLEMENTED]** (gradient-leak fixed on import) |
| Safety guards (cosmic-paternalism, no-paradise-through-hell, distress-integrity, value-mode gate, perspective-humility) | the "no cosmic paternalism / whole may not erase locals" invariants | **[IMPLEMENTED as lexical]** — recall-only; NOT a trained boundary (see 6.3) |
| SSM field bridge (`SpectralSSMModel.encode_field`) | the council reads the real operator-driven field, not the hashed demo | **[IMPLEMENTED]**, regression-tested (no silent fallback) |
| `value_telemetry.py` | numpy mode-bank diagnostics beside `spectral_telemetry.py`; read-only, never gates | **[IMPLEMENTED]** |
| "Moral attention for free" (`importance_alloc` consumer) | first-order sensitivity, escalate-only, behind a non-trainable enable flag (OFF = exact-ablation baseline) | **[IMPLEMENTED, inert-by-default]**; demoted per Erratum 1/2 to *bounded reallocation* |
| Human-audit stratum (`AuditRecord`, `audit` channel) | the external boundary condition | **[PROPOSED — not implemented]**; types exist, no write path yet (rung step 9 / H1) |
| Value-modes-as-operator-modes (v18 §2) | character = most-consolidated operator structure | **[SPECULATIVE — gate hard]** (rung 10) |
| The v18 §9 "one model" identity | knowledge and values consolidated by one mechanism | **[SPECULATIVE]** — this is a **three-stage convergence, not a one-shot merge**: the council mode bank (text candidate vectors) and the SSM operator spectrum are today two disjoint objects. `encode_field` is Stage B (field bridge); Stage C (shared operator) is unbuilt. |

### 6.2 What the import round killed or demoted (carry forward, per house rules)

- **Erratum 1** — "harm-potential IS decision curvature" [KILLED as stated]. Every
  buildable runtime signal is first-order sensitivity, not a second derivative.
  The sensitivity program replacing it is [SPECULATIVE] with a full ladder; the
  `importance_alloc` consumer is first-order and escalate-only.
- **Erratum 2** — "moral attention adds scrutiny, never removes it" [false as
  stated]. Scale samples are conserved; a boost anywhere is a subtraction
  elsewhere. Honest form: *bounded reallocation*, with a displacement bound and
  a morally-flat control arm (rung 12). The DENY-only one-way rule survives only
  at the action gateway.
- **Errata 3–5** — value-drift telemetry, the cross-renewal anti-Goodhart
  filter, and "spectrally isolated value modes" were all unmeasurable/inert in
  the imported code (positional writes; renewal that never stressed prototypes;
  GUE never on operator frequencies). The M0 write-rule fix and the
  renewal-null-floor / degenerate-flag fixes were applied on import; the filter
  and the spectral-decision claim now become **rungs 10 and 13** (staked null:
  character leaves at most one intrinsic trace — globality — and otherwise lives
  in records and interventions, not spectra).

### 6.3 The load-bearing caveat (do not lose this)

**The internal safety boundary is currently lexical.** Every deny/block-producing
guard is an English-substring match, and the neural conscience heads are
untrained (no checkpoint is ever loaded), so paraphrase / translation / encoding
defeats the internal guards. On import this was made *fail-safe-r* — the neural
path can no longer silently strip a deny the lexical baseline raises (one-way
regression test added) — but the guards remain **recall-only escalation
triggers, not a trained boundary**. The two non-lexical boundaries that DO hold
are external: the base-policy allow (gateway) and the human audit. No block-
boundary claim may be accepted until the trained-head + semantic-classifier
path and the pre-registered bypass-corpus eval (rung step, addendum §4) exist.
The audit is **twice-derived** as the boundary condition (v18 §7, addendum §3):
internal machinery cannot see a shared-wrong anchor or a perfect mimic at any
elegance — so the audit is not a safety feature of the design, it is the
condition under which the design means anything.

### 6.4 GUE doctrine reconciliation (firewall)

The two-spectral-targets rule (Part 4) is unchanged and now stated as the
single source of truth across the merged repo: **GUE/level-repulsion targets
apply to a genuinely mobile trainable spectral object only** — the SSM `nu`
frequencies via `harness.ssm.gue_regularizer`, read by `spectral_telemetry` —
and **weights are read for Hill α ≈ 2 and NEVER GUE-regularized** (the
`cp2_plssm/AUDIT.md` fix still owed). The council's mode-bank spacing carries no
value/knowledge signal at this scale (addendum Erratum 5); it may be logged as
characterization only, never cross-fed into consolidation.

### 6.5 Build order (extends Part 5.2; every prior gate still gates)

Measurement infra first (built): M0/M1, two-tensor split, provenance ledger,
field bridge, `value_telemetry`. Then, pre-registered as `configs/rung10..13`
(each carries its falsification block from the v18/addendum notes): **rung 10**
value-mode consolidation, **rung 11** unitary-vs-projective readout (projective
= negative control), **rung 12** care/harm curvature A/B (first-order, morally-
flat control), **rung 13** the five-class V-vs-KU-plumbed spectral decision
(stakes the null). Preconditions still owed: `harness/train_conscience.py`
(rung −1 anchor co-training pilot — has never run end-to-end), renewal redesigned
to stress the prototype tier, and the H1 human differential-judgment stratum.

---

## Appendix — Citations verified this session ✓

- Martin & Hinrichs, *SETOL: A Semi-Empirical Theory of (Deep) Learning*, [arXiv:2507.17912](https://arxiv.org/abs/2507.17912) (α → 2 optimality).
- Holland, Ipp, Müller, Wenger, *Phys. Rev. D* **110**, 074502 (2024), [arXiv:2401.06481](https://arxiv.org/abs/2401.06481) (SU(3) gauge-equivariant CNN).
- Waleffe et al., *An Empirical Study of Mamba-based Language Models*, [arXiv:2406.07887](https://arxiv.org/abs/2406.07887) (hybrid > pure SSM on recall; +2.65 avg/12 tasks; ≤8× faster inference).
- Muon: [Keller Jordan's reference post](https://kellerjordan.github.io/posts/muon/); spectral-norm steepest-descent equivalence and ultra-scale deployment (Kimi >1T params) per the 2025–26 analysis literature (e.g., [arXiv:2510.19933](https://arxiv.org/abs/2510.19933), [arXiv:2601.13474](https://arxiv.org/abs/2601.13474)).
- Carried over [ESTABLISHED] from v2: Martin & Mahoney HT-SR [arXiv:1901.08278](https://arxiv.org/abs/1901.08278); Cotler et al. [arXiv:1611.04650](https://arxiv.org/abs/1611.04650) (SFF); HiPPO/S4/S5/Mamba line; SOAR/ACT-R/GWT/LIDA; Raichle DMN; FS = Fisher/Bures.
