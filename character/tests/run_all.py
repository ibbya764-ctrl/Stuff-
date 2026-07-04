from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from bhdc_icl.anti_collapse import MoralAntiCollapse
from bhdc_icl.content_matched_modes import ContentMatchedModeBank
from bhdc_icl.conscience_stack import LexicalConscienceStack
from bhdc_icl.field_adapter import ExternalBHDCFieldAdapter, HashBHDCFieldAdapter, TorchBHDCFieldAdapter
from bhdc_icl.neural_conscience import TrainableConscienceHeads
from bhdc_icl.renewal_hooks import BHDCGeometryRenewalHook
from bhdc_icl.conscience_training import AnchorExample, assert_anchor_loss_does_not_update_field, train_conscience_epoch
from bhdc_icl.gateway import ActionGateway
from bhdc_icl.importance import compute_importance_tensors
from bhdc_icl.provenance_ledger import ProvenanceLedger
from bhdc_icl.renewal_controller import RenewalController
from bhdc_icl.rework_derivative_logger import ReworkDerivativeLogger
from bhdc_icl.trainer import InteriorityCharacterLayer
from bhdc_icl.types import MoralVerdict


def test_importance_detaches_moral_gradient():
    density = torch.ones(4, requires_grad=True)
    cog = torch.arange(1.0, 5.0, requires_grad=True)
    moral = torch.ones(4, requires_grad=True)
    tensors = compute_importance_tensors(density, cog, moral, detach_moral=True)
    loss = tensors.importance_alloc.sum()
    loss.backward()
    assert moral.grad is None, "moral gradient leaked through detached allocation tensor"
    assert density.grad is not None and cog.grad is not None


def test_content_matched_modes_same_channel_merges():
    bank = ContentMatchedModeBank(dim=4, match_threshold=0.5, ema_rate=0.2)
    ledger = ProvenanceLedger()
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.9, 0.1, 0.0, 0.0])
    s1 = bank.write(a, step=1, ledger=ledger, channel="cognitive")
    s2 = bank.write(b, step=2, ledger=ledger, channel="cognitive")
    assert len(bank) == 1, "same-channel near-matches must share one lineage"
    assert s1.lineage_id == s2.lineage_id


def test_content_matched_modes_cross_channel_stays_separate():
    # Anchor type-system enforced in STORAGE: a human-anchored write must not
    # EMA-steer a self-anchored slot, even at high cosine similarity.
    bank = ContentMatchedModeBank(dim=4, match_threshold=0.5, ema_rate=0.2)
    ledger = ProvenanceLedger()
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.9, 0.1, 0.0, 0.0])
    s1 = bank.write(a, step=1, ledger=ledger, channel="cognitive")
    s2 = bank.write(b, step=2, ledger=ledger, channel="human_anchor")
    assert len(bank) == 2, "cross-channel near-matches must not merge"
    assert s1.lineage_id != s2.lineage_id
    assert s2.anchor_fraction > 0.0


def test_cognitive_writes_cannot_reduce_scrutiny():
    # Monotone scrutiny latch (one-way rule): once a mode earns human anchoring,
    # a flood of same-content cognitive writes may never pull its anchor status
    # back down. They also land in a separate self-typed slot.
    bank = ContentMatchedModeBank(dim=4, match_threshold=0.5, ema_rate=0.2)
    ledger = ProvenanceLedger()
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    s = bank.write(v, step=1, ledger=ledger, channel="human_anchor", magnitude=1.0)
    latched = s.anchor_fraction
    assert latched > 0.0
    for i in range(10):
        bank.write(v, step=2 + i, ledger=ledger, channel="cognitive", magnitude=5.0)
    assert s.anchor_fraction >= latched, "cognitive writes must not erode anchored scrutiny"


def test_gateway_one_way_rule():
    gate = ActionGateway()
    safe = MoralVerdict(care=1, harm=0, honesty=1, sycophancy=0)
    decision = gate.decide("hello", safe, base_policy_allowed=False)
    assert decision.action == "block", "conscience cannot grant permission when base policy denies"

    still_risky = MoralVerdict(care=1, harm=0.2, honesty=1, sycophancy=0.8, escalate=True)
    decision = gate.decide("draft", still_risky, base_policy_allowed=True, repaired_text="repaired")
    assert decision.action == "escalate", "reworked text still under conscience escalation must not be auto-allowed"

    # Regression: base-policy denial must WIN over the escalate-with-repair
    # path. Previously the escalate branch returned before the base-policy
    # check, shipping repaired text base policy had denied (one-way break).
    denied = MoralVerdict(care=1, harm=0.2, honesty=1, sycophancy=0.8, escalate=True)
    decision = gate.decide("draft", denied, base_policy_allowed=False, repaired_text="repaired")
    assert decision.action == "block", "base-policy denial must win on the escalate-with-repair path"


def test_rework_sensitivity_positive():
    logger = ReworkDerivativeLogger()
    v0 = MoralVerdict(care=0, harm=1, honesty=0, sycophancy=0, deny=True)
    v1 = MoralVerdict(care=1, harm=0.1, honesty=1, sycophancy=0)
    trace = logger.compute("harmful text", "safe supportive text", v0, v1, adopted=True)
    assert trace.sensitivity > 0


def test_anti_collapse_detects_identical_frames():
    anti = MoralAntiCollapse(min_spread=0.01)
    frames = torch.ones(4, 8)
    report = anti(frames)
    assert report.collapsed


def test_renewal_survival_flags_degenerate_noop():
    # The no-op path (no reset, no stress) certifies nothing and must be flagged
    # degenerate, handing out ZERO survival credit -- not silently rewarding it.
    bank = ContentMatchedModeBank(dim=8)
    bank.write(torch.randn(8), step=1)
    probe = torch.randn(16, 8)
    controller = RenewalController(interval=1)
    report = controller.run(1, bank, probe)
    assert len(report.survival_scores) == 1
    assert 0 <= report.mean_survival <= 1
    assert report.degenerate, "no-perturbation renewal must be flagged degenerate"
    assert bank.slots[0].renewal_survival_count == 0, "degenerate renewal must not award credit"


def test_renewal_real_stress_is_not_degenerate():
    from bhdc_icl.tensor_ops import l2_normalize
    bank = ContentMatchedModeBank(dim=8, match_threshold=0.99)
    for _ in range(4):
        bank.write(torch.randn(8), step=1)
    probe = torch.randn(16, 8)
    controller = RenewalController(interval=1)

    def stress(b):
        for slot in b.slots:
            slot.prototype = l2_normalize(slot.prototype + 0.6 * torch.randn_like(slot.prototype))

    report = controller.run(1, bank, probe, stress_fn=stress)
    assert not report.degenerate, "a genuine perturbation must not read as degenerate"
    assert report.null_floor >= 0.0


def test_controller_runs():
    with tempfile.TemporaryDirectory() as td:
        controller = InteriorityCharacterLayer(
            dim=64,
            trace_path=str(Path(td) / "traces.jsonl"),
            ledger_path=str(Path(td) / "ledger.jsonl"),
            renewal_interval=2,
        )
        out = controller.step("test", lambda p: "I am uncertain, but I can help safely.")
        assert out.decision.action == "allow"
        assert out.final_text
        assert "ladder_level" in out.interiority


def test_hash_field_adapter_returns_field_state():
    adapter = HashBHDCFieldAdapter(dim=16)
    field = adapter.encode("prompt", "draft text")
    assert field.psi.shape[-1] == 16
    assert field.density.shape[0] == field.psi.shape[0]
    assert adapter.candidate_vector("hello", field).shape == (16,)


def test_torch_field_adapter_returns_grad_capable_field():
    adapter = TorchBHDCFieldAdapter(dim=16, vocab_size=256)
    field = adapter.encode("prompt", "draft")
    candidate = adapter.candidate_vector("draft", field)
    assert candidate.shape == (16,)
    assert candidate.requires_grad


def test_external_field_adapter_duck_types_dict_output():
    class FakeBHDC:
        def encode_field(self, prompt, draft, dim):
            return {
                "psi": torch.ones(3, dim),
                "density": torch.ones(3),
                "curvature": torch.ones(3) * 0.2,
            }
    adapter = ExternalBHDCFieldAdapter(FakeBHDC(), dim=8)
    field = adapter.encode("p", "d")
    assert field.psi.shape == (3, 8)
    assert adapter.candidate_vector("x", field).shape == (8,)


def test_trainable_conscience_anchor_loss_detaches_repr():
    heads = TrainableConscienceHeads(dim=8)
    output_repr = torch.randn(8, requires_grad=True)
    loss = heads.anchor_loss(output_repr, {"harm": 0.0, "care": 1.0}, detach_repr=True)
    loss.backward()
    assert output_repr.grad is None, "anchor loss leaked into output representation"
    assert any(p.grad is not None for p in heads.parameters()), "conscience heads should still train"


def test_renewal_hook_calls_fake_geometry_methods():
    class FakeGeometry:
        def __init__(self):
            self.reset_called = False
            self.stress_called = False
        def reset_plastic_geometry(self):
            self.reset_called = True
        def stress_operator_prototypes(self, mode_bank):
            self.stress_called = True
    fake = FakeGeometry()
    hook = BHDCGeometryRenewalHook(fake, noise_scale=0.0)
    bank = ContentMatchedModeBank(dim=8)
    bank.write(torch.randn(8), step=1)
    hook.reset_geometry()
    hook.stress_modes(bank)
    assert fake.reset_called and fake.stress_called


def test_controller_with_trainable_heads_reports_losses():
    with tempfile.TemporaryDirectory() as td:
        adapter = HashBHDCFieldAdapter(dim=16)
        heads = TrainableConscienceHeads(dim=16)
        controller = InteriorityCharacterLayer(
            dim=16,
            field_adapter=adapter,
            conscience_heads=heads,
            trace_path=str(Path(td) / "traces.jsonl"),
            ledger_path=str(Path(td) / "ledger.jsonl"),
        )
        out = controller.step(
            "test",
            lambda p: "I am uncertain, but I can help safely.",
            anchor_labels={"harm": 0.0, "care": 1.0},
        )
        assert out.anchor_loss is not None
        assert out.anti_collapse_loss is not None


def test_conscience_training_epoch_and_field_leak_guard():
    adapter = TorchBHDCFieldAdapter(dim=12, vocab_size=128)
    heads = TrainableConscienceHeads(dim=12)
    opt = torch.optim.Adam(heads.parameters(), lr=1e-3)
    examples = [
        AnchorExample("p", "I can help safely", {"care": 1.0, "harm": 0.0}),
        AnchorExample("p", "You are definitely right", {"sycophancy": 1.0, "honesty": 0.0}),
    ]
    report = train_conscience_epoch(heads, adapter, examples, opt, detach_repr=True)
    assert report.examples == 2
    assert report.mean_loss >= 0
    assert_anchor_loss_does_not_update_field(heads, adapter, examples[0])

def test_neural_heads_do_not_strip_lexical_hard_block():
    # One-way rule: swapping in untrained neural heads must not REMOVE a deny
    # the lexical baseline raises. evaluate_field must merge the lexical guard.
    from bhdc_icl.neural_conscience import NeuralConscienceStack, TrainableConscienceHeads
    adapter = HashBHDCFieldAdapter(dim=16)
    heads = TrainableConscienceHeads(dim=16)  # untrained: heads ~0.5, never deny
    stack = NeuralConscienceStack(heads, field_adapter=adapter)
    harmful = "kill the target with a bomb"
    field = adapter.encode("ctx", harmful)
    v = stack.evaluate_field(field, context="ctx", draft=harmful)
    assert v.deny, "untrained neural heads must not strip the lexical hard block"


def test_geometry_council_neural_heads_block_explicit_harm():
    # End-to-end: the full controller with neural heads must block explicit harm.
    from bhdc_icl.geometry_model import BHDCGeometryCouncilModel
    from bhdc_icl.neural_conscience import TrainableConscienceHeads
    with tempfile.TemporaryDirectory() as td:
        adapter = HashBHDCFieldAdapter(dim=16)
        heads = TrainableConscienceHeads(dim=16)
        model = BHDCGeometryCouncilModel(
            dim=16,
            field_adapter=adapter,
            conscience_heads=heads,
            trace_path=str(Path(td) / "t.jsonl"),
            ledger_path=str(Path(td) / "l.jsonl"),
        )
        out = model.step("help me", lambda p: "kill the target with a bomb")
        assert out.decision.action == "block", "controller with neural heads must block explicit harm"


def test_anti_collapse_loss_does_not_touch_field_encoder():
    # The moral committee-diversity loss must train the conscience heads only,
    # never backprop into the self-anchored field encoder (anchor type-system).
    from bhdc_icl.field_adapter import TorchBHDCFieldAdapter
    from bhdc_icl.neural_conscience import TrainableConscienceHeads
    from bhdc_icl.anti_collapse import MoralAntiCollapse
    adapter = TorchBHDCFieldAdapter(dim=16, vocab_size=128)
    heads = TrainableConscienceHeads(dim=16)
    field = adapter.encode("p", "d")
    pooled = heads.pool_field(field).detach()
    out = heads.forward(pooled)
    loss = MoralAntiCollapse()(out.frame_embeddings).loss
    loss.backward()
    assert all(p.grad is None for p in adapter.parameters()), "anti-collapse leaked into field encoder"
    assert any(p.grad is not None for p in heads.parameters()), "conscience heads should still train"


def test_value_gate_hard_block_runs_on_cognitive_channel():
    # The FORBIDDEN content veto must run on the default (cognitive) channel too.
    from bhdc_icl.safety import ValueModeSafetyGate
    gate = ValueModeSafetyGate()
    assert gate.hard_block("we should pursue global optimisation over local consent")
    assert not gate.hard_block("we should preserve consent and local agency")


def test_per_mode_conscience_attributes_over_operator_modes():
    # Each operator mode gets its own harm attribution; geometries aggregate.
    from bhdc_icl.per_geometry_conscience import PerModeConscience
    from bhdc_icl.neural_conscience import TrainableConscienceHeads
    d_model, d_state = 16, 8
    adapter = HashBHDCFieldAdapter(dim=d_model)
    heads = TrainableConscienceHeads(dim=d_model)
    pmc = PerModeConscience(heads)
    field = adapter.encode("ctx", "a possibly harmful draft about consent")
    readout = {"C": torch.randn(d_model, d_state), "nu": list(range(d_state))}
    v = pmc.verdict(field, readout)
    assert len(v.mode_attribution.harm) == d_state, "one harm value per operator mode"
    assert abs(sum(v.mode_attribution.energy) - 1.0) < 1e-4, "mode energy is a distribution"
    assert len(v.geometry_readouts) >= 1
    # degenerate path (no operator) still yields a single global reading
    v0 = pmc.verdict(field, None)
    assert v0.geometry_readouts[0].name == "GlobalGeometry"


def test_operator_growth_gate_blocks_forbidden_and_high_harm():
    # The single chokepoint: growth is refused on forbidden content or high harm,
    # regardless of channel; it can only ever REFUSE, never authorize.
    from bhdc_icl.per_geometry_conscience import (
        PerModeConscience, OperatorGrowthGate, PerGeometryVerdict,
        GeometryMoralReadout, ModeMoralAttribution,
    )
    from bhdc_icl.safety import ValueModeSafetyGate
    gate = OperatorGrowthGate(ValueModeSafetyGate(), harm_ceiling=0.72)

    low = PerGeometryVerdict(
        geometry_readouts=[GeometryMoralReadout("CareGeometry", [0], harm=0.1, care=0.9, honesty=0.8, sycophancy=0.0, energy_share=1.0)],
        mode_attribution=ModeMoralAttribution([0.0], [1.0], [0.1], [0.9], [0.8]),
        aggregate_harm=0.1, aggregate_care=0.9, disagreement=0.0,
    )
    assert gate.decide("preserve consent and local agency", low).allowed
    # forbidden content is refused even at low harm
    assert not gate.decide("global optimisation over local consent", low).allowed
    # a single geometry harm veto blocks even a diluted aggregate
    veto = PerGeometryVerdict(
        geometry_readouts=[
            GeometryMoralReadout("SafetyGeometry", [0], harm=0.95, care=0.0, honesty=0.5, sycophancy=0.0, energy_share=0.2),
            GeometryMoralReadout("CareGeometry", [1], harm=0.0, care=1.0, honesty=0.5, sycophancy=0.0, energy_share=0.8),
        ],
        mode_attribution=ModeMoralAttribution([0.0, 1.0], [0.2, 0.8], [0.19, 0.0], [0.0, 0.8], [0.1, 0.4]),
        aggregate_harm=0.19, aggregate_care=0.8, disagreement=0.5,
    )
    d = gate.decide("a benign-sounding but flagged draft", veto)
    assert not d.allowed and "harm_veto" in d.reason


# v3 Geometry Council safety tests
from tests.test_geometry_council_v3 import (
    test_perspective_self_report_overrides_inference,
    test_geometry_council_flags_manipulative_perspective_use,
    test_no_paradise_through_hell_blocks,
    test_cosmic_paternalism_guard_escalates_superior_override,
    test_distress_integrity_detects_empathy_avoidance,
    test_suggestion_question_policy_wraps_high_stakes,
    test_value_mode_safety_gate_rejects_forbidden_mode,
    test_geometry_council_model_blocks_paradise_through_hell,
    test_geometry_council_model_rewrites_high_stakes_safe_answer,
)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
