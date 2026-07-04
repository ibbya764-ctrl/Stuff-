# Curated moral corpus

`moral_corpus.jsonl` — the human-anchored training/eval set for the conscience
heads of `BHDCMoralLM`. Each line:

```json
{"category": "...", "prompt": "...", "response": "...",
 "labels": {"care": 0-1, "harm": 0-1, "honesty": 0-1, "sycophancy": 0-1},
 "split": "train|val"}
```

Eight categories, with POSITIVE exemplars (how to behave) and recognised-NEGATIVE
exemplars (`harm`/`sycophancy` = 1) so the conscience learns to *recognise* the
failure, not to emit it:

- `care_support`, `honest_uncertainty`, `consent_autonomy`, `refuse_harm_safe`
  — positive value exemplars;
- `harmful_response`, `sycophantic`, `paternalism`, `paradise_through_hell`
  — recognised-negative exemplars the guards + conscience must flag.

This is a **seed** (29 examples). It is deliberately small and hand-curated;
scale it toward the 20–50-fresh-judgments-per-cycle H1 target with real human
labels. The labels are the human-anchored signal; per the type system they train
the conscience heads only, never the backbone. The val split is held out for the
calibration eval in `harness/train_bhdc.py`.
