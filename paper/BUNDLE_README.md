# BHDCMoralLM — downloadable bundle

The full fused model (code + configs + data + a trained proof checkpoint) and the
accompanying paper, self-contained.

## Contents

```
BHDCMoralLM_bundle/
  README.md                     this file
  PAPER.md / PAPER.pdf          the paper
  bhdc_icl/                     the conscience / character layer (module)
  harness/                      backbone, fused model, trainer, data pipeline
  value_telemetry.py            numpy mode-bank diagnostics
  configs/                      bhdc_100m.json, bhdc_small_proof.json, rungs
  data/moral/moral_corpus.jsonl the curated moral corpus (90 examples)
  checkpoints/                  a trained proof checkpoint (.pt) + its report
  load_model.py                 load the checkpoint and run a forward pass
  requirements.txt              torch, numpy
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# inspect the fused model + load the proof checkpoint
PYTHONPATH=.:. python load_model.py

# train the proof yourself (CPU, minutes)
PYTHONPATH=.:. python -m harness.train_bhdc configs/bhdc_small_proof.json

# train the 98.8M model (Apple MPS / GPU) -- device auto-detected
PYTHONPATH=.:. python -m harness.train_bhdc configs/bhdc_100m.json
```

The capability corpus (~17.6 MB of public-domain text) is **downloaded and cached
on first run** into `harness/data/corpus/` — it is not shipped in the bundle to
keep the download small. An offline fallback keeps everything runnable without a
network.

## What this is

`BHDCMoralLM` is one trainable `nn.Module` in which the moral readout, value
memory, and forward computation are read from and gated by the *same* operator
modes that carry capability. See `PAPER.md` for the architecture, the three-stage
merge, the pre-registered experiments, and — pinned throughout — the honest limit:
the fusion makes misalignment disfavoured and loud, **not impossible**; the
external human audit is the guarantee.
