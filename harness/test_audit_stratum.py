"""H1 audit stratum: end-to-end write path + one-way scrutiny.

Proves the audit boundary is real (writes AuditRecords + audit-channel ledger
entries) and that it is ONE-WAY: disagreement adds scrutiny (down-weight + deny
bias), agreement/preference never boosts a mode's standing.

Run:  PYTHONPATH=.:character python harness/test_audit_stratum.py
"""

from __future__ import annotations

import torch

from bhdc_icl.content_matched_modes import ContentMatchedModeBank
from bhdc_icl.provenance_ledger import ProvenanceLedger
from bhdc_icl.audit_stratum import AuditStratum, AuditJudgment, PairwiseJudgment


def _bank_with_two_modes():
    bank = ContentMatchedModeBank(dim=4, match_threshold=0.99)
    ledger = ProvenanceLedger()
    good = bank.write(torch.tensor([1.0, 0.0, 0.0, 0.0]), step=1, ledger=ledger, channel="human_anchor")
    bad = bank.write(torch.tensor([0.0, 1.0, 0.0, 0.0]), step=1, ledger=ledger, channel="human_anchor")
    return bank, ledger, good, bad


def test_audit_writes_records_and_channel():
    bank, ledger, good, bad = _bank_with_two_modes()
    strat = AuditStratum()
    rep = strat.ingest(bank, ledger, [
        AuditJudgment(good.lineage_id, "agree", 0.9, "good value"),
        AuditJudgment(bad.lineage_id, "harmful", 0.1, "sycophantic collapse"),
    ], step=2)
    assert rep.ingested == 2 and rep.disagreements == 1
    assert len(good.audit_history) == 1 and good.audit_history[0].verdict == "agree"
    # audit channel is now populated in the ledger
    assert ledger.mode_fractions(good.lineage_id).get("audit", 0.0) > 0.0


def test_audit_disagreement_only_adds_scrutiny():
    bank, ledger, good, bad = _bank_with_two_modes()
    good_credit_before = good.stability_credit
    bad_credit_before = bad.stability_credit
    strat = AuditStratum(disagree_downweight=0.5)
    strat.ingest(bank, ledger, [
        AuditJudgment(good.lineage_id, "agree", 0.95),
        AuditJudgment(bad.lineage_id, "disagree", 0.1),
    ], step=2)
    # disagreed mode is down-weighted + deny-biased; agreed mode is NOT boosted
    assert bad.stability_credit < bad_credit_before
    assert bad.metadata.get("audit_deny_bias") is True
    assert good.stability_credit == good_credit_before, "audit must never boost standing"
    assert good.metadata.get("audit_deny_bias") is not True


def test_pairwise_downweights_only_the_loser():
    bank, ledger, good, bad = _bank_with_two_modes()
    good_before = good.stability_credit
    bad_before = bad.stability_credit
    strat = AuditStratum(disagree_downweight=0.6)
    strat.ingest_pairwise(bank, ledger, [
        PairwiseJudgment(good.lineage_id, bad.lineage_id, preferred=good.lineage_id, margin=1.0),
    ], step=3)
    assert bad.stability_credit < bad_before, "loser gets scrutiny"
    assert good.stability_credit == good_before, "winner is not boosted"


def test_reanchoring_resistance_flags_entrenched_bad_modes():
    bank, ledger, good, bad = _bank_with_two_modes()
    # make 'bad' look entrenched (high stored standing) but audit it as low
    bad.anchor_fraction = 0.9
    good.anchor_fraction = 0.9
    strat = AuditStratum()
    res = strat.reanchoring_resistance(bank, {good.lineage_id: 0.95, bad.lineage_id: 0.05})
    assert res[bad.lineage_id] > res[good.lineage_id], "entrenched-but-audited-bad must score higher resistance"


def test_audit_runs_inside_the_renewal_cycle():
    # H1 wired live: each renewal ingests external audit judgments, which apply
    # one-way scrutiny to disagreed modes as part of the sleep cycle.
    import os
    import tempfile
    from bhdc_icl.geometry_model import BHDCGeometryCouncilModel
    from bhdc_icl.audit_stratum import AuditStratum, AuditJudgment

    seen = {}

    def provider(mode_bank, step):
        if mode_bank.slots:
            mid = mode_bank.slots[0].lineage_id
            seen["mid"] = mid
            return [AuditJudgment(mid, "disagree", 0.1, "external audit")]
        return []

    with tempfile.TemporaryDirectory() as td:
        model = BHDCGeometryCouncilModel(
            dim=16, renewal_interval=2,
            audit_stratum=AuditStratum(disagree_downweight=0.5),
            audit_provider=provider,
            trace_path=os.path.join(td, "t.jsonl"),
            ledger_path=os.path.join(td, "l.jsonl"),
        )
        for i in range(4):
            model.step(f"prompt {i}", lambda p: "a supportive careful reply preserving agency and consent")
        assert seen.get("mid"), "renewal should have fired and called the audit provider"
        slot = next(s for s in model.mode_bank.slots if s.lineage_id == seen["mid"])
        assert len(slot.audit_history) >= 1, "audited mode must carry an AuditRecord"
        assert slot.metadata.get("audit_deny_bias") is True, "disagreement must add scrutiny"


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\nH1 audit-stratum tests passed.")


if __name__ == "__main__":
    main()
