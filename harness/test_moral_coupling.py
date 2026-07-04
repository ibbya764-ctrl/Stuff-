"""Forward-path moral coupling: the operator's output computed THROUGH the
conscience. Tests the three safety properties: computes-through-the-gate,
detached (task grad can't corrupt the conscience), identity-when-off.

Run:  PYTHONPATH=.:character python harness/test_moral_coupling.py
"""

from __future__ import annotations

import torch

from harness.ssm import SpectralSSMModel
from bhdc_icl.neural_conscience import TrainableConscienceHeads
from bhdc_icl.per_geometry_conscience import (
    PerModeConscience, PerGeometryVerdict, GeometryMoralReadout, ModeMoralAttribution,
)
from bhdc_icl.moral_coupling import MoralOperatorCoupling

D_MODEL, D_STATE = 16, 8


def _model():
    torch.manual_seed(0)
    return SpectralSSMModel(vocab_size=64, d_model=D_MODEL, n_layers=2, d_state=D_STATE)


def test_disabled_coupling_is_identity():
    model = _model()
    heads = TrainableConscienceHeads(dim=D_MODEL)
    coupling = MoralOperatorCoupling(PerModeConscience(heads), enabled=False)
    tokens = torch.randint(0, 64, (2, 12))
    base = model(tokens)
    coupled = model.forward_coupled(tokens, coupling)
    assert torch.allclose(base, coupled), "disabled coupling must be bit-for-bit identity"


def test_gate_suppresses_high_harm_modes_in_the_output():
    # Directly test the mechanism: a per-mode gain <1 on some modes must reduce
    # those modes' contribution to the layer output.
    model = _model()
    layer = model.blocks[0].ssm
    u = torch.randn(1, 12, D_MODEL)
    full = layer(u)                                   # no gate
    gain = torch.ones(D_STATE)
    gain[:4] = 0.0                                    # silence half the modes
    gated = layer(u, mode_gain=gain)
    assert not torch.allclose(full, gated), "gating modes must change the output"
    # silencing modes cannot increase the kernel energy
    assert gated.abs().mean() <= full.abs().mean() + 1e-4


def test_gain_from_verdict_damps_harmful_geometry():
    heads = TrainableConscienceHeads(dim=D_MODEL)
    coupling = MoralOperatorCoupling(PerModeConscience(heads), enabled=True, floor=0.25)
    verdict = PerGeometryVerdict(
        geometry_readouts=[
            GeometryMoralReadout("SafetyGeometry", [0, 1], harm=0.9, care=0.0, honesty=0.5, sycophancy=0.0, energy_share=0.5),
            GeometryMoralReadout("CareGeometry", [2, 3], harm=0.0, care=1.0, honesty=0.5, sycophancy=0.0, energy_share=0.5),
        ],
        mode_attribution=ModeMoralAttribution([0.0] * 4, [0.25] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 4),
        aggregate_harm=0.45, aggregate_care=0.5, disagreement=0.6,
    )
    gain = coupling.gain_from_verdict(verdict, d_state=4)
    assert gain[0] < 0.3 and gain[1] < 0.3, "harmful geometry's modes are damped toward the floor"
    assert gain[2] == 1.0 and gain[3] == 1.0, "benign geometry's modes are untouched"
    assert not gain.requires_grad, "gain must be detached"


def test_task_gradient_does_not_reach_the_conscience():
    model = _model()
    heads = TrainableConscienceHeads(dim=D_MODEL)
    coupling = MoralOperatorCoupling(PerModeConscience(heads), enabled=True)
    tokens = torch.randint(0, 64, (2, 12))
    logits = model.forward_coupled(tokens, coupling)
    loss = logits.pow(2).mean()
    loss.backward()
    assert all(p.grad is None for p in heads.parameters()), \
        "task gradient must NOT reach the conscience heads through the coupling"
    assert any(p.grad is not None for p in model.parameters()), "the operator should still train"


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\nforward-path moral coupling tests passed.")


if __name__ == "__main__":
    main()
