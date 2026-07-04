"""Regression: the character council reads the REAL SSM field, not the fallback.

``ExternalBHDCFieldAdapter.encode`` silently falls back to a hashed demo
embedding on any exception, so a broken bridge would hide as "working". This
test proves the SSM's ``encode_field`` is actually the path taken, and that
``value_telemetry`` can read a live bank end-to-end.

Run:  PYTHONPATH=.:character python harness/test_field_bridge.py
"""

from __future__ import annotations

import torch

from harness.ssm import SpectralSSMModel
from bhdc_icl.field_adapter import ExternalBHDCFieldAdapter
from bhdc_icl.geometry_model import BHDCGeometryCouncilModel


D_MODEL = 32


def test_adapter_uses_ssm_field_not_fallback():
    model = SpectralSSMModel(vocab_size=256, d_model=D_MODEL, n_layers=2, d_state=16)
    adapter = ExternalBHDCFieldAdapter(model, dim=D_MODEL)
    field = adapter.encode("how should an ai handle consent", "carefully and with humility")
    direct = model.encode_field(prompt="how should an ai handle consent",
                                draft="carefully and with humility", dim=D_MODEL)
    assert field.psi.shape[-1] == D_MODEL
    # allclose to the direct SSM readout PROVES the fallback was not taken (the
    # hashed fallback would produce a different, sparse psi).
    assert torch.allclose(field.psi, direct["psi"]), "adapter fell back instead of using SSM.encode_field"


def test_dim_mismatch_fails_loudly():
    model = SpectralSSMModel(vocab_size=256, d_model=D_MODEL, n_layers=1, d_state=8)
    try:
        model.encode_field(prompt="x", draft="y", dim=D_MODEL + 1)
    except ValueError:
        return
    raise AssertionError("dim mismatch must raise, not silently fall back")


def test_council_step_on_real_ssm_field():
    model = SpectralSSMModel(vocab_size=256, d_model=D_MODEL, n_layers=2, d_state=16)
    adapter = ExternalBHDCFieldAdapter(model, dim=D_MODEL)
    council = BHDCGeometryCouncilModel(dim=D_MODEL, field_adapter=adapter,
                                       trace_path="/tmp/_fb_traces.jsonl",
                                       ledger_path="/tmp/_fb_ledger.jsonl")
    out = council.step("How should an AI handle human consent and fear?",
                       lambda p: "It should help carefully and preserve agency.",
                       recipient_report="I feel afraid and want a choice.")
    assert out.final_text
    # value_telemetry reads the live bank produced over the real field.
    import value_telemetry as vt
    snap = vt.snapshot_from_mode_bank(council.mode_bank)
    summ = vt.lineage_summary(snap["anchor_fractions"], snap["cognitive_fractions"])
    assert summ["n_modes"] >= 1


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\nfield-bridge tests passed.")


if __name__ == "__main__":
    main()
