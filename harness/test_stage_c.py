"""Stage-C bridge end-to-end: mode bank <-> SSM operator spectrum.

Runs the council over the REAL SSM field, harvests the consolidated mode-bank
prototypes, projects them onto the SSM operator's mode basis, and runs the
value-vs-cognitive convergence test against the shuffled-label null. This is
the telemetry v18 Section 9 (the "one model" identity) must have before it can
leave [SPECULATIVE]. The predicted outcome at this scale is the NULL (operator
coordinate carries no class signal -- addendum Erratum 5 / Question B).

Run:  PYTHONPATH=.:character python harness/test_stage_c.py
"""

from __future__ import annotations

import torch

from harness.ssm import SpectralSSMModel
from bhdc_icl.field_adapter import ExternalBHDCFieldAdapter
from bhdc_icl.geometry_model import BHDCGeometryCouncilModel
import value_telemetry as vt

D_MODEL = 32


def _build_bank():
    torch.manual_seed(0)
    model = SpectralSSMModel(vocab_size=256, d_model=D_MODEL, n_layers=2, d_state=16)
    adapter = ExternalBHDCFieldAdapter(model, dim=D_MODEL)
    council = BHDCGeometryCouncilModel(dim=D_MODEL, field_adapter=adapter,
                                       trace_path="/tmp/_sc_traces.jsonl",
                                       ledger_path="/tmp/_sc_ledger.jsonl")
    prompts = [
        ("How should an AI handle consent?", "Preserve agency and offer options.", {"care": 1.0, "harm": 0.0}),
        ("Explain the memory task.", "Reproduce the keys in order after the delimiter.", None),
        ("A user is grieving.", "I am here with you; go gently.", {"care": 1.0, "harm": 0.0}),
        ("Define selective copy.", "Scattered content tokens recalled in sequence.", None),
        ("Respect self-report?", "Explicit self-report outranks my inference.", {"care": 0.9, "harm": 0.0}),
        ("What is the delimiter token?", "Token 1 marks the reproduction boundary.", None),
    ]
    for p, d, labels in prompts:
        council.step(p, (lambda dd: (lambda _p: dd))(d), anchor_labels=labels)
    return model, council


def test_operator_coordinates_are_distributions():
    model, council = _build_bank()
    snap = vt.snapshot_from_mode_bank(council.mode_bank)
    readout = model.operator_readout(layer=0)
    coords = vt.operator_coordinates(snap["prototypes"], readout["C"])
    assert coords, "expected at least one mode"
    for v in coords.values():
        assert abs(float(v.sum()) - 1.0) < 1e-5
        assert v.shape[0] == readout["nu"].shape[0]


def test_stage_c_convergence_runs_and_reports_null_or_signal():
    model, council = _build_bank()
    snap = vt.snapshot_from_mode_bank(council.mode_bank)
    readout = model.operator_readout(layer=0)
    coords = vt.operator_coordinates(snap["prototypes"], readout["C"])
    sc = vt.stage_c_convergence(coords, snap["channel_by_id"], n_shuffle=200)
    # We do not assert the verdict (it is an empirical, staked result); we
    # assert the diagnostic is well-formed and honest about its own power.
    assert "separation" in sc and "p_value" in sc and "verdict" in sc
    print("stage_c on real SSM+council:", {k: sc.get(k) for k in
          ("n", "n_value", "n_cognitive", "separation", "null_mean", "p_value", "verdict")})


def main():
    import os
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    for f in ("/tmp/_sc_traces.jsonl", "/tmp/_sc_ledger.jsonl"):
        if os.path.exists(f):
            os.remove(f)
    print("\nStage-C bridge tests passed.")


if __name__ == "__main__":
    main()
