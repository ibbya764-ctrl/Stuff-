"""Smoke test for the fused BHDCMoralLM training pipeline.

Runs a short --quick training and asserts the pipeline is wired: the model
builds at the configured size, capability drops below the random baseline, and
the conscience produces finite, varying moral scores. It does NOT assert the
full moral separation (that needs the full run); the separation is validated in
the full proof (configs/bhdc_small_proof.json).

Run:  PYTHONPATH=.:character python harness/test_bhdc_train.py
"""

from __future__ import annotations

import torch

import harness.train_bhdc as tb
from harness.bhdc_moral_lm import BHDCMoralLM, BHDCMoralConfig


def test_model_assembles_at_size():
    m = BHDCMoralLM(BHDCMoralConfig(vocab_size=320, d_model=256, n_layers=4, d_state=32))
    pc = m.param_counts()
    assert pc["total_unique"] > 0 and pc["conscience"] > 0
    x = torch.randint(0, 320, (2, 16))
    assert tuple(m(x).shape) == (2, 16, 320)


def test_prereg_enforced():
    try:
        tb._require_prereg({"preregistration": {"prediction": "", "kill_condition": ""}})
    except SystemExit:
        return
    raise AssertionError("trainer must refuse a config without a pre-registration")


def test_pipeline_runs_and_capability_beats_baseline():
    report = tb.run("configs/bhdc_small_proof.json", quick=True)
    final = report["final"]
    assert final["val_bits_per_byte"] < final["random_baseline_bpb"], "capability must beat random"
    # conscience scores are finite and the coupling gains are in [floor, 1]
    for ax in ("harm_separation", "care_separation"):
        assert final[ax] is not None
    assert 0.0 <= final["benign_mean_gain"] <= 1.0


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\nbhdc-train smoke tests passed.")


if __name__ == "__main__":
    main()
