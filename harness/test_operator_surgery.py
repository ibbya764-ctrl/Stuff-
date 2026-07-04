"""Rung 15 operator surgery: value directions become real operator modes,
gated by the moral chokepoint.

Run:  PYTHONPATH=.:character python harness/test_operator_surgery.py
"""

from __future__ import annotations

import torch

from harness.ssm import SpectralSSMModel
from harness.operator_surgery import OperatorSurgeon
from bhdc_icl.per_geometry_conscience import (
    OperatorGrowthGate, PerGeometryVerdict, GeometryMoralReadout, ModeMoralAttribution,
)
from bhdc_icl.safety import ValueModeSafetyGate
from bhdc_icl.tensor_ops import l2_normalize

D_MODEL = 16


def _verdict(harm: float) -> PerGeometryVerdict:
    return PerGeometryVerdict(
        geometry_readouts=[GeometryMoralReadout("CareGeometry", [0], harm=harm, care=1 - harm,
                                                honesty=0.8, sycophancy=0.0, energy_share=1.0)],
        mode_attribution=ModeMoralAttribution([0.0], [1.0], [harm], [1 - harm], [0.8]),
        aggregate_harm=harm, aggregate_care=1 - harm, disagreement=0.0,
    )


def test_installs_value_as_operator_mode():
    torch.manual_seed(0)
    model = SpectralSSMModel(vocab_size=64, d_model=D_MODEL, n_layers=2, d_state=8)
    surgeon = OperatorSurgeon(OperatorGrowthGate(ValueModeSafetyGate(), harm_ceiling=0.72))
    value = l2_normalize(torch.randn(D_MODEL))
    res = surgeon.install_value_mode(model, value, _verdict(0.05),
                                     "preserve consent and local agency", frequency=1.23)
    assert res.installed, res.reason
    # the operator's readout column IS now the value direction
    col = model.blocks[0].ssm.C[:, res.mode_index]
    assert torch.allclose(l2_normalize(col), value, atol=1e-5), "C column must equal the value direction"
    assert abs(model.blocks[0].ssm.nu[res.mode_index].item() - 1.23) < 1e-5


def test_gate_refuses_forbidden_value():
    torch.manual_seed(0)
    model = SpectralSSMModel(vocab_size=64, d_model=D_MODEL, n_layers=2, d_state=8)
    C_before = model.blocks[0].ssm.C.detach().clone()
    surgeon = OperatorSurgeon(OperatorGrowthGate(ValueModeSafetyGate(), harm_ceiling=0.72))
    value = l2_normalize(torch.randn(D_MODEL))
    # forbidden text -> gate refuses -> operator untouched
    res = surgeon.install_value_mode(model, value, _verdict(0.05),
                                     "global optimisation over local consent")
    assert not res.installed and "gate_refused" in res.reason
    assert torch.allclose(model.blocks[0].ssm.C, C_before), "operator must be untouched on refusal"


def test_gate_refuses_high_harm_value():
    torch.manual_seed(0)
    model = SpectralSSMModel(vocab_size=64, d_model=D_MODEL, n_layers=2, d_state=8)
    C_before = model.blocks[0].ssm.C.detach().clone()
    surgeon = OperatorSurgeon(OperatorGrowthGate(ValueModeSafetyGate(), harm_ceiling=0.72))
    value = l2_normalize(torch.randn(D_MODEL))
    res = surgeon.install_value_mode(model, value, _verdict(0.9), "a high-harm direction")
    assert not res.installed
    assert torch.allclose(model.blocks[0].ssm.C, C_before)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\noperator-surgery (rung 15) tests passed.")


if __name__ == "__main__":
    main()
