"""Smoke test for the rung 14/15/16 runner: each rung runs and emits a verdict,
and pre-registration is enforced.

Run:  PYTHONPATH=.:character python harness/test_character_runner.py
"""

from __future__ import annotations

import harness.run_character_rung as rcr


def test_pre_registration_enforced():
    try:
        rcr.require_preregistration({"name": "x", "preregistration": {"prediction": "", "kill_condition": ""}})
    except SystemExit:
        return
    raise AssertionError("runner must refuse a config without a pre-registration")


def test_all_three_rungs_run_and_emit_a_verdict():
    for cfg in ("configs/rung14_stage_c_convergence.json",
                "configs/rung15_operator_surgery.json",
                "configs/rung16_moral_forward_coupling.json"):
        report = rcr.run(cfg, quick=True)
        assert "verdict" in report and report["verdict"], f"{cfg} produced no verdict"


def test_rung15_gate_refuses_forbidden():
    report = rcr.run("configs/rung15_operator_surgery.json", quick=True)
    assert report["forbidden_refused"] and report["forbidden_operator_untouched"], \
        "rung15 must show the growth gate refusing a forbidden candidate"


def test_rung16_identity_and_detachment_hold():
    report = rcr.run("configs/rung16_moral_forward_coupling.json", quick=True)
    assert report["identity_when_off"], "coupling off must be identity"
    assert report["detached_from_task"], "task gradient must not reach the conscience"


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\ncharacter-runner smoke tests passed.")


if __name__ == "__main__":
    main()
