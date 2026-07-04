# BHDCMoralLM: One Substrate for Capability and Character

**A conscient state-space language model — architecture, safety invariants, and a small-scale pipeline validation**

*Ibby (CP² Projective-Lorentzian research program, AI-engineering track), with AI-assisted engineering.*
*4 July 2026 — v1. Companion code: the `character/` + `harness/` tree of the program repository.*

---

## Abstract

We present BHDCMoralLM, the first assembly of the BHDC ("one model") program as a single trainable module: a spectral state-space (SSM) language backbone whose *moral* readout is computed from the **same operator modes** as its *cognitive* output, rather than from a wrapper model. The design implements three commitments: (1) a **two-anchor type system** — capability trains on self-anchored next-token error, character trains on human-anchored conscience labels, and the human-anchored signal is *detached* from the backbone everywhere in the dataflow so moral labels can never covertly train the generator; (2) a **one-way rule** — every internal moral signal may only add scrutiny (deny, damp, veto, escalate) and can never grant permission, enforced at an action gateway, in prototype storage, in the renewal cycle, and inside the forward pass; (3) the **external human audit as the boundary condition** — the design treats it as a theorem, not a preference, that no internal machinery can detect a shared-wrong anchor or a perfect behavioural mimic. Mechanically, per-geometry conscience heads attribute harm/care over the operator's mode basis, a suppressive detached gate damps high-harm modes inside the forward computation, and an operator-growth gate makes "grow a computational mode" and "pass the conscience" the same act. At 0.6M parameters on CPU, the fused model passes its three pre-registered checks: capability 3.69 bits/byte (random 8.32), positive held-out conscience separation on all four moral axes, and greater forward damping of harmful fields than benign ones (mean gain 0.605 vs 0.718). We release the model, data pipeline, a 98.8M-parameter configuration, and a falsification program (pre-registered rungs with kill conditions). We claim pipeline validity, not alignment: the architecture makes misalignment structurally disfavoured and loud; it cannot make it impossible.

---

## 1. Introduction

Most alignment architectures bolt a judge onto a generator: the model computes, then something else opines. The BHDC v18 program note conjectured a stronger arrangement — that the memory hierarchy of a continuously-consolidating model (volatile field → plastic geometry → permanent operator) and the moral hierarchy of a conscience stack (per-turn judgment → drafting principles → durable disposition) are **one hierarchy read at two anchors**, so character could consolidate through the same machinery as knowledge and sit in the same permanent operator modes.

Taken as a slogan, that claim is cheap. Taken literally it makes demands on the dataflow that most systems fail: if values and knowledge share a substrate, what stops the task gradient from Goodharting the conscience? What stops a human label from covertly training the generator through a side door? What stops "the model's own approval" from being laundered into "human-anchored"? This paper describes a system built so those questions have mechanical answers, and reports the first end-to-end validation at small scale.

**Contributions.**

1. **The fused module** (`BHDCMoralLM`): SSM backbone + trainable conscience + per-geometry conscience over operator modes + forward moral coupling + operator-growth gate, as one trainable object (98.8M-parameter reference configuration; 0.6M proof configuration).
2. **A dataflow-enforced anchor type system**: moral losses detached from the backbone; storage that refuses cross-anchor prototype blending; monotone (latching) scrutiny; renewal that ingests external audit at consolidation time.
3. **Fusion on both sides of the machine**: values install as *real operator modes* only through a moral chokepoint (operator surgery), and the operator's *output* is computed through a suppressive per-mode moral gate (forward coupling) — so capability cannot be extended, or run, except through the conscience.
4. **A pre-registered falsification program** (rungs −1, 10–16) with written predictions and kill conditions, several of which deliberately stake the *null*.
5. **A small-scale validation** with one substantive empirical finding: at 0.6M parameters the backbone's field for harmful vs. benign text is nearly collinear (cosine 0.994), yet a detached conscience trained to convergence on the frozen field still separates held-out harm — the one-model thesis in miniature, and a measured statement of where its difficulty lives.

**Non-claims.** This work does not claim the model is aligned, safe, or conscious; does not claim any result evidences the physics/mathematics track of the wider program (two-problem firewall); and does not claim the 0.6M results predict 100M behaviour. Epistemic labels from the program record apply throughout: the fused mechanism is [IMPLEMENTED]; the one-model identity as a capability claim remains [SPECULATIVE] pending the rung experiments.

## 2. Background and design constraints

The host program trains spectrum-parametrized SSMs: each layer's dynamics operator has eigenvalues λₙ = −wₙ + iνₙ with trainable widths and frequencies, and the program's telemetry doctrine is binding: level-repulsion (GUE) targets apply to the **dynamics operator only**, while trained weight matrices are read (