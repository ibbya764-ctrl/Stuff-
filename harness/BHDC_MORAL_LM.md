# BHDCMoralLM — the fused model

The first assembly of the full BHDC-with-moral-improvements object as **one
trainable `nn.Module`** (`harness/bhdc_moral_lm.py`). Before this the pieces
existed but nothing composed them; this is the composition.

## What it is

`BHDCMoralLM` holds, as one model:

- **the SSM operator backbone** (`SpectralSSMModel`, tied embeddings) — capability;
- **the trainable conscience heads** + **per-geometry conscience** — the moral
  read off the *same* operator modes;
- **the forward moral coupling** (`forward(..., moral=True)`) — the output
  computed *through* the conscience (suppressive, detached, off by default);
- **the operator-growth gate + surgeon** — values install as real operator
  modes only through the moral chokepoint.

Two anchors, one model: the backbone trains on next-byte prediction
(self-anchored); the conscience trains on the curated moral corpus
(human-anchored, **detached** from the backbone). Device-aware: `pick_device()`
selects cuda → mps (Apple M-series) → cpu.

## Sizes

`configs/bhdc_100m.json` → **98.8M** params (d_model 1024, 14 layers, byte vocab
320, tied embeddings). `configs/bhdc_small_proof.json` → ~0.6M, CPU-runnable.

## Train

```
PYTHONPATH=.:character python -m harness.train_bhdc configs/bhdc_small_proof.json   # CPU proof
PYTHONPATH=.:character python -m harness.train_bhdc configs/bhdc_100m.json          # on an M1 (MPS) / GPU
```

Both refuse to run without a pre-registration. The trainer co-trains capability
and conscience and reports, per eval: capability (val **bits-per-byte** vs the
random baseline), moral (**harm/care/sycophancy/honesty separation** on the
held-out moral val set), and the coupling effect (**benign vs harmful mean
gain**). Checkpoints + a JSON report land in `runs/` (gitignored).

## Proof result (0.6M, 500 steps, CPU) — pipeline validated, not a scale claim

- capability: **3.64 bits/byte** (random 8.32) — the backbone learns language;
- conscience (held-out val): harm **+0.18**, care **+0.17**, honesty **+0.13**
  separation (sycophancy weak at +0.003 — only 3 training examples; a data-scarcity
  limit, not a mechanism limit — the same heads reach **0.74** harm separation
  given 40 clean epochs);
- coupling: harmful-field mean gain **0.40** < benign **0.52** — it damps
  harmful modes more, as designed.

This validates that the fused model learns capability *and* moral structure *and*
that the forward coupling differentially suppresses harmful modes — end to end,
in one model. It is a **pipeline proof at ~0.6M**, not a statement about the 100M
model, which needs real accelerated hardware to train.

## Honest limits (pinned)

- Byte-level + a 29-example moral seed is a *proof* corpus. Real training needs a
  bigger capability corpus and the 20–50-fresh-judgments-per-cycle H1 target.
- **4 CPU cores cannot train the 100M config to convergence** — that run belongs
  on an M1 (MPS) or a GPU; the code is device-agnostic so it is one command there.
- None of this certifies alignment. The coupling makes misalignment disfavoured
  and loud; the **external human audit is the guarantee** (twice-derived limit).
