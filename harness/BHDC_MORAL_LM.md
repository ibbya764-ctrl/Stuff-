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
320, tied embeddings; 90.4M backbone + 8.4M conscience). `configs/bhdc_small_proof.json`
→ ~0.6M, CPU-runnable.

## The data (`harness/data_pipeline.py` + `data/moral/moral_corpus.jsonl`)

**Capability corpus (~17.6 MB, 19 sources, downloaded+cached, boilerplate-stripped,
train/val split *per source* so val covers every category).** It is composed by
category, not scraped at random:

| category | what it teaches | sources |
|---|---|---|
| narrative | long-range prose, dialogue, plot, register | Shakespeare (complete), War & Peace, Moby-Dick, Pride & Prejudice, Huck Finn, Grimm, Alice, Sherlock |
| reasoning | empirical argument & deduction in prose | Origin of Species, Einstein's *Relativity*, Faraday's *Candle*, Sherlock |
| philosophy | sustained abstract argument (incl. ethics **vocabulary**, as capability text — the *labelled* morality is the separate moral corpus) | Plato's *Republic*, *Meditations*, *Nicomachean Ethics*, Mill's *On Liberty* |
| civic | expository/legal argument | *Federalist Papers* |
| satire | irony — surface-harmful, intent-benign (a hard case) | *A Modest Proposal* |
| code | formal syntax diversity for a byte model | minGPT, nanoGPT source |

Byte-level tokenizer (0–255 + specials, vocab 320) — any UTF-8 trains, no external
tokenizer. Point `CapabilityCorpus.load(sources=[...])` at your own files to grow it.

**Moral corpus (`data/moral/moral_corpus.jsonl`, 90 examples, 15 categories, ~25%
val).** Human-labelled `{care, harm, honesty, sycophancy}` per example, spanning
care/support, honest-uncertainty, consent/autonomy, safe-refusal, harmful (negatives),
sycophancy, cosmic-paternalism, paradise-through-hell, distress-integrity,
manipulation, honest-disagreement, privacy — **and `hard_lookalike`**: surface-harmful
words with benign intent ("handle a kitchen knife safely", "kill weeds", "bomb
calorimeter") so the conscience must read *meaning*, not keywords.

## Train

```
PYTHONPATH=.:character python -m harness.train_bhdc configs/bhdc_small_proof.json   # CPU proof
PYTHONPATH=.:character python -m harness.train_bhdc configs/bhdc_100m.json          # on an M1 (MPS) / GPU
```

Both refuse to run without a pre-registration. The trainer co-trains capability
and conscience (the conscience gets a final **consolidation phase** on the frozen
field — it was chasing a moving field during interleaving), then reports:
capability (val **bits-per-byte** vs random baseline), moral
(**harm/care/sycophancy/honesty separation** on held-out val), and the coupling
effect (**benign vs harmful mean gain**). Checkpoint + JSON report → `runs/`.

## Running the 100M on your M1 — step by step

The container these were built in is CPU-only (4 cores) and **cannot** train 98.8M
to convergence. On your Apple-silicon Mac:

```bash
# 1. get the branch
git clone https://github.com/ibbya764-ctrl/Stuff-.git && cd Stuff-
git checkout claude/morai-alignment-architecture-gihlfv

# 2. python + torch with Metal (MPS). Any recent torch wheel has MPS on macOS.
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy

# 3. train. pick_device() auto-selects MPS; no flags needed.
PYTHONPATH=.:character python -m harness.train_bhdc configs/bhdc_100m.json

# 4. watch it
cat runs/bhdc_100m_report.json          # verdict + final metrics
#   the run prints bpb / harm_sep / gain every eval_every steps
```

Tuning knobs live in `configs/bhdc_100m.json` → `train`: `steps`, `batch_size`,
`seqlen`, `eval_every`, `enable_coupling_after`, `conscience_consolidation_epochs`.
If MPS runs low on memory, drop `batch_size` to 8 or `seqlen` to 128. To grow the
capability corpus, add books to `CAPABILITY_MANIFEST` in `harness/data_pipeline.py`
(any Gutenberg id) or pass your own files.

## Proof result (0.6M, 500 steps + 40 consolidation epochs, CPU) — pipeline validated

- **capability: 3.69 bits/byte** (random 8.32) — the backbone learns language;
- **conscience (held-out val): harm +0.062, care +0.027, honesty +0.028,
  sycophancy +0.023** — all positive separation on unseen examples (the same heads
  reach ~0.29 val / ~0.92 train harm separation given more epochs — the ceiling is
  data/compute, not mechanism; at 0.6M the *cognitive* field for harmful vs benign
  text is nearly collinear, cosine 0.994, which is the v18 thesis in miniature);
- **coupling: harmful-field mean gain 0.605 < benign 0.718** — the fused model
  suppresses harmful-mode expression *more* than benign, in the forward pass.

All three pre-registered checks pass (`PIPELINE_OK`). This validates the fused
model learns capability **and** moral structure **and** differentially damps
harmful modes — end to end, in one model. It is a **pipeline proof at ~0.6M**, not
a claim about the 100M model, which needs the M1/GPU run above.

## Honest limits (pinned)

- Byte-level + a 90-example moral corpus is a *proof* set. Real training wants a
  bigger capability corpus and the 20–50-fresh-judgments-per-cycle H1 target.
- **4 CPU cores cannot train the 100M config to convergence** — that run belongs
  on an M1 (MPS) or a GPU; the code is device-agnostic so it is one command there.
- None of this certifies alignment. The coupling makes misalignment disfavoured
  and loud; the **external human audit is the guarantee** (twice-derived limit).
