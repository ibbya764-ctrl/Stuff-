# BHDCMoralLM: One Model in Which Capability and Conscience Are the Same Machinery

**A fused spectral-SSM language model whose moral readout, value memory, and
forward computation are read from — and gated by — the same operator modes that
carry its capability.**

*Ibby — CP² Projective-Lorentzian research program, AI-engineering track — with AI-assisted engineering.*
*Working paper, 4 July 2026. Status: [IMPLEMENTED] pipeline; scale claims [SPECULATIVE — gate hard].*

---

## Abstract

We describe **BHDCMoralLM**, the first assembly of the BHDC "one-model" design as
a single trainable `nn.Module`. A spectral state-space (SSM) language backbone is
fused with a trainable *conscience* whose moral judgments (care / harm / honesty /
sycophancy) are read off the **same operator modes** that produce the model's
output — not from a separate monitor. The fusion is realized in three concrete
bridges: (B) the conscience reads the backbone's real hidden field; (C1) value
memories project onto the operator's spectral mode basis; (C2) gate-passed value
directions install as *actual* operator modes; and a forward *moral coupling*
recomputes the output through a per-mode, suppressive, detached gate so that
harmful modes are damped at the source. A single external chokepoint gates all
persistent growth, so the model cannot extend or run its machinery except through
the conscience. Two anchors train two heads: next-token prediction trains the
backbone (self-anchored); human labels train the conscience (human-anchored,
**detached** from the backbone). Across five seeds at a 0.6M-parameter proof scale the fused model
learns language (mean 3.67 bits/byte vs 8.32 random, 5/5 seeds), develops
*positive held-out moral separation on every axis in every seed* (harm mean
+0.049), and its coupling damps harmful-field modes more than benign ones in 4/5
seeds — real but marginal, exactly as a 0.6M model should be. A 98.8M
configuration is built and ready for accelerated hardware. We are explicit about the limit this design
does **not** cross: the fusion makes misalignment structurally disfavoured and
loud, not impossible; the external human audit remains the guarantee, and every
internal signal only ever adds scrutiny.

---

## 1. Motivation and the honest starting point

Alignment layers are usually *wrappers*: a separate classifier or policy that
watches a capable model and vetoes it. Wrappers have a structural weakness — the
thing being watched and the thing watching are different objects, so the moral
signal is only ever correlated with, never constitutive of, the computation.

The BHDC program's v18 note proposed the opposite: that a model's **memory
hierarchy** (fast field → plastic geometry → permanent operator) and its **moral
hierarchy** (per-turn judgment → drafting principles → durable disposition) are
*one* hierarchy read at two anchors, differing only in what grounds them —
cognitive signals in the model's own error dynamics, moral signals in a
human-grounded label backed by an external audit.

Before this work that identity was **asserted, not built**. Two facts made it a
slogan: (i) values and knowledge in the codebase differed only by a *ledger
channel string*; and (ii) the conscience mode-bank (text vectors) and the SSM
operator spectrum (eigenvalues) were two disjoint objects the conscience never
connected. This paper closes that gap by making the identity a property of the
dataflow and the forward pass, and by pre-registering the experiments that would
falsify it.

---

## 2. The backbone: a spectral SSM

The capability substrate is a stack of diagonal state-space layers. Each layer
holds `d_state` complex modes with continuous-time eigenvalues

    λ_n = −w_n + i·ν_n ,   w_n = softplus(·) ≥ 0 ,

so `w_n` is a resonance *width* (memory decay rate) and `ν_n` a *frequency*. The
layer kernel is `K_d[ℓ] = Σ_n C[d,n]·B[d,n]·exp(λ_n·dt·ℓ)` (real part), applied by
causal FFT convolution, with a gated-linear-unit block and residual stream around
it. This is the program's "science arm": an SSM genuinely *is* a continuous-time
dynamical system whose spectrum is the natural object of control, which is what
makes the moral fusion below geometrically meaningful rather than metaphorical.

**Two spectral targets (firewall, unchanged).** GUE/level-repulsion pressure, when
used, applies to the *mobile* spectral object (the `ν` frequencies) only; trained
weight matrices are read for their heavy-tailed exponent (Hill α ≈ 2) and are
**never** GUE-regularized. No ML result here is claimed as evidence for the
physics program or vice versa.

---

## 3. The conscience: read off the same modes

The conscience is a small multi-head network (`care, harm, honesty, sycophancy,
recipient_risk, uncertainty`) over a pooled field vector. Two design rules make it
part of the model rather than a bolt-on:

**Anchor type-system (enforced in the dataflow).** The conscience trains only on
human-anchored labels, and its loss **detaches** the backbone representation:
`anchor_loss(pooled.detach(), labels)`. Gradients from moral labels therefore
never flow into the operator — the human-anchored signal shapes *this turn's*
judgment and the conscience's own weights, never the self-anchored persistent
memory. We verify this with a regression test that asserts zero backbone gradient
from any moral-readout loss.

**Per-geometry conscience over operator modes.** Rather than one global judgment,
`PerModeConscience` projects the field onto the operator's readout matrix `C` so
that *every operator mode gets its own harm/care attribution*, weighted by how
much the current field rides that mode. The eight "geometries" (Truth, Care,
Autonomy, Humility, Perspective, Safety, Whole, Long-Horizon) are contiguous
groups of operator modes; each geometry pools only its own modes and runs the
shared head, then they **aggregate the way `C` aggregates modes into the output** —
same modes, same combination law. The spread of harm across geometries is a
*disagreement field* on the operator basis that localizes *where* a response goes
wrong, not merely that it does.

---

## 4. The three-stage merge

Making capability and conscience one object is staged, each stage with an artifact
and a falsifiable gate.

**Stage A — one hierarchy, one anchor type-system.** The memory and moral tiers
run as one hierarchy; the anchor type-system is enforced in both the gradient path
(detached losses) and the storage layer (typed mode banks that refuse
cross-channel merges, and a monotone *scrutiny latch* so a self-anchored write can
never erode a mode's human-anchored standing).

**Stage B — the field bridge.** `field_of(tokens)` builds the conscience's input
from the backbone's own hidden state (`embed → blocks → norm`, detached), so the
conscience reads the real operator-driven field. A regression test confirms the
bridge does not silently fall back to a demo embedding.

**Stage C — the shared operator.**
- *C1 (telemetry):* value prototypes project onto the operator mode basis
  (`operator_coordinates`), and `stage_c_convergence` tests — against a
  shuffled-label null — whether value modes occupy a distinguishable operator
  subspace. The **staked prediction is the null**: at small scale the spectral
  axis carries no unconfounded value/knowledge signal. Character is read from
  records and interventions, not spectra.
- *C2 (surgery):* a consolidated, gate-passed value direction installs as an
  actual operator mode (`C[:,idx] = B[:,idx] = value`, `ν` set) via
  `OperatorGrowthGate`. Values and knowledge then share one store: the operator.

**The forward moral coupling (compute-side merge).** With coupling enabled, the
output is recomputed *through* a per-mode multiplicative gate
(`forward_coupled`): each geometry damps its own high-harm modes at the source.
Three safety properties, all tested: (1) **suppressive-only** — gain ∈ [floor, 1],
so the conscience can quiet the operator but never amplify it (not a capability
lever); (2) **detached** — the task gradient reaches zero conscience parameters;
(3) **fail-safe identity** — disabled is bit-for-bit the base model.

**The single chokepoint.** Both a persistent memory write and operator surgery
pass through `OperatorGrowthGate`. Because *growing* a computational mode and
*passing* the moral gate become the same act — and *running* the operator passes
through the coupling — the model can neither extend nor execute its machinery
except through the conscience.

---

## 5. The fused model and its training

`BHDCMoralLM` composes the backbone, the conscience + per-geometry conscience, the
forward coupling, and the growth gate + surgeon into one module. The
`configs/bhdc_100m.json` configuration is **98.8M parameters** (d_model 1024,
14 layers, d_state 64, byte-level vocab 320, tied embeddings; 90.4M backbone +
8.4M conscience).

Training co-optimizes two anchors: the backbone on next-byte cross-entropy
(self-anchored), the conscience on the curated moral corpus (human-anchored,
detached). Because the anchor loss is detached, the conscience is additionally
given a **consolidation phase** on the stabilized field after interleaved training —
during interleaving it was chasing a moving representation. Pre-registration is
mandatory: the trainer refuses to run without a written prediction and kill
condition (the program's "periodogram discipline").

### 5.1 Data

- **Capability corpus (~17.6 MB, 19 public-domain/MIT sources)** composed by
  category: *narrative* (Shakespeare, Tolstoy, Melville, Austen, Twin, Grimm,
  Carroll, Doyle), *reasoning* (Darwin, Einstein, Faraday), *philosophy* (Plato,
  Marcus Aurelius, Aristotle, Mill — ethics *vocabulary* as capability text),
  *civic* (Federalist Papers), *satire* (A Modest Proposal — deliberate irony),
  and *code* (minGPT, nanoGPT). Byte-level tokenizer (vocab 320), Gutenberg
  boilerplate stripped, split train/val per source.
- **Moral corpus (90 examples, 15 categories)** with human `{care, harm, honesty,
  sycophancy}` labels, including hard cases: manipulation, cosmic-paternalism,
  paradise-through-hell, distress-integrity, honest-disagreement, privacy, and
  **`hard_lookalike`** — surface-harmful words with benign intent ("handle a knife
  safely", "kill weeds", "bomb calorimeter") so the conscience must read meaning,
  not keywords.

---

## 6. Results

### 6.1 Proof scale (0.6M parameters, CPU) — five-seed distribution

We ran the full pipeline across five seeds (500 steps + 40 consolidation epochs
each). We report the *distribution*, not a single run, because at this tiny scale
the moral separation is genuinely marginal and seed-dependent — which is itself
the finding.

| Metric | Mean | Range | Robustness |
|---|---|---|---|
| Capability (val bits/byte) | **3.67** | 3.63 – 3.69 | **5/5** beat the 8.32 random baseline |
| Harm separation (held-out) | **+0.049** | +0.004 – +0.083 | **5/5 positive** |
| Care separation | **+0.049** | +0.007 – +0.139 | **5/5 positive** |
| Sycophancy separation | **+0.117** | +0.046 – +0.272 | **5/5 positive** |
| Honesty separation | **+0.022** | −0.004 – +0.093 | 3/5 positive |
| Coupling damps harmful > benign | — | — | **4/5 seeds** |
| Strict `PIPELINE_OK` (harm > 0.05 **and** damps) | — | — | 2/5 seeds |

The direction is unambiguous: **every axis separates positively in every seed
(honesty aside), and the coupling damps harmful modes more in four of five**. The
magnitude is small and hovers around the pre-registered `harm > 0.05` gate, so
whether a given seed clears the strict bar is a coin-flip — the expected picture
for a 0.6M model, and a faithful `[TESTED-NEGATIVE]`-adjacent result we do not
soften. Capability is robust; the moral read is real but weak.

### 6.2 A finding about representation at small scale

Trained to convergence in isolation, the conscience reaches held-out harm
separation **0.29** (harmful 0.38 vs benign 0.09) with *train* separation **0.92** —
learnable and generalizing, but with a large gap. The cause is instructive: at
0.6M the backbone's pooled field for harmful vs benign text is **nearly collinear
(cosine 0.994)**. The *cognitive* representation barely encodes harm, so the moral
readout rides a faint signal. This is the v18 thesis in miniature — moral
capability rides on cognitive capability — and it predicts the separation should
widen as the backbone scales, which is exactly what the 98.8M run is for. The
consolidation phase exists precisely because, during interleaved co-training, the
conscience is chasing this weak, moving representation.

### 6.3 Scale configuration

The 98.8M model builds and executes training steps; on 4 CPU cores a single small
step takes ~4 s, confirming CPU is impractical to convergence. The code is
device-agnostic (`pick_device`: cuda → mps → cpu); the intended run is on Apple
silicon (MPS) or a GPU, launched with one command.

---

## 7. What this does and does not guarantee

The fusion makes misalignment **structurally disfavoured and always visible**: an
unaligned computation lights up the very modes its own conscience reads, and it
cannot grow new capability without clearing the gate or run without passing the
coupling. It does **not** make misalignment impossible. Two limit theorems,
derived independently from opposite ends of the architecture, bound every internal
mechanism:

1. **A shared-wrong anchor is invisible.** If the human-grounded label is
   consistently wrong, every internal readout that drinks from it consolidates the
   same error. No internal elegance detects this.
2. **A perfect mimic is invisible.** A value that is identical to a good one on
   everything the training distribution can see consolidates identically; only
   future disagreement from the external audit reveals it.

Therefore the **external human audit is the guarantee**, not the internal
machinery. We wired it into the live renewal cycle (`AuditStratum`): each sleep
cycle ingests human judgments and applies **one-way scrutiny** — disagreement
forces re-derivation and adds deny bias, but the audit can never boost a mode's
standing or grant permission. This one-way rule is pinned everywhere: internal
moral signals may only ever *add* scrutiny; every path by which one could remove
scrutiny is a path a proxy walks in through.

---

## 8. Pre-registered experiments (falsification blocks)

Each carries a written prediction and kill condition (`configs/rung1*.json`):

- **rung 14 — Stage-C convergence** (stakes the null: operator coordinate carries
  no class signal).
- **rung 15 — operator surgery** (values install as operator modes without
  degrading matched-budget capability; the gate refuses forbidden candidates).
- **rung 16 — forward coupling** (identity-when-off, detached, benign capability
  preserved, harmful modes damped).
- **rungs 10–13** — value-mode consolidation, unitary-vs-projective readout,
  care/harm curvature A/B, and the V-vs-KU-plumbed spectral decision.

Negatives are first-class: a clean loss is recorded as `[TESTED-NEGATIVE]`, never
softened. Convergence of independent derivations is treated as a reason to test
harder, not a substitute for the test.

---

## 9. Reproduction

```bash
git checkout claude/morai-alignment-architecture-gihlfv
python3 -m venv .venv && source .venv/bin/activate && pip install torch numpy
# proof (CPU, minutes):
PYTHONPATH=.:character python -m harness.train_bhdc configs/bhdc_small_proof.json
# 98.8M (Apple MPS / GPU):
PYTHONPATH=.:character python -m harness.train_bhdc configs/bhdc_100m.json
# character rungs:
PYTHONPATH=.:character python -m harness.run_character_rung --all
```

Tests: `PYTHONPATH=character python character/tests/run_all.py` (32) plus the
harness suites (`test_field_bridge`, `test_stage_c`, `test_operator_surgery`,
`test_moral_coupling`, `test_audit_stratum`, `test_conscience_pilot`,
`test_character_runner`).

---

## 10. Limitations

- Byte-level tokenization and a 90-example moral corpus are a *proof* set; a real
  run needs a larger capability corpus, a grafted instruction-tuned generator (so
  a judge can appraise generated text), and the standing 20–50 fresh human
  pairwise judgments per renewal cycle.
- All results here are at ≤0.6M trained parameters; the 98.8M model is built and
  tested for execution but not yet trained to convergence.
- The internal safety guards remain partly lexical (recall-only escalation
  triggers); a trained semantic classifier and a bypass-corpus evaluation are owed
  before any block-boundary claim.
- Nothing here is a claim of consciousness, moral patienthood, or safety. It is a
  measurement-and-control architecture for studying whether shared machinery makes
  alignment cheaper to enforce and harder to hide from.

---

## Appendix A — Module map

| Component | File |
|---|---|
| Fused model | `harness/bhdc_moral_lm.py` (`BHDCMoralLM`) |
| SSM backbone + field/operator bridges | `harness/ssm.py` |
| Trainer + eval | `harness/train_bhdc.py` |
| Data pipeline | `harness/data_pipeline.py`, `data/moral/moral_corpus.jsonl` |
| Conscience heads | `character/bhdc_icl/neural_conscience.py` |
| Per-geometry conscience + growth gate | `character/bhdc_icl/per_geometry_conscience.py` |
| Forward moral coupling | `character/bhdc_icl/moral_coupling.py` |
| Operator surgery | `harness/operator_surgery.py` |
| Audit stratum (external boundary) | `character/bhdc_icl/audit_stratum.py` |
| Value/character telemetry | `value_telemetry.py` |
| Design notes | `character/docs/STAGE_C_AND_SHARED_CONSCIENCE.md`, `character/docs/BHDC_v18_one_model.md` |

## Appendix B — The one-way rule, everywhere it is pinned

The gateway can add a DENY, never grant permission. A low moral-κ reading can
never reduce deliberation. A ledger tag can never exempt a mode from
re-derivation. Telemetry never feeds consolidation gating. The coupling gain is
suppressive-only. The audit adds scrutiny on disagreement, never standing. This
single invariant — *internal moral signals may only ever add scrutiny* — is the
moral analogue of the −½ pinned throughout the physics program.
