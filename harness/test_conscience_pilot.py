"""Smoke test for the rung -1 anchor co-training pilot.

Runs a few epochs and asserts the pilot wiring holds: anchor loss is produced,
the field-leak guard passes (the anchor loss never trains the field), and the
report has the expected shape. Kept fast (tiny model, few epochs).

Run:  PYTHONPATH=.:character python harness/test_conscience_pilot.py
"""

from __future__ import annotations

import harness.train_conscience as tc


def test_pilot_runs_and_leak_guard_holds():
    report = tc.run(epochs=4, d_model=16, seed=0,
                    report_path="runs/_pilot_smoke.json")
    assert report["field_leak_guard_passed"], report.get("field_leak_error")
    assert len(report["anchor_loss_curve"]) == 4
    assert report["verdict"] in ("PASS", "TESTED-NEGATIVE")


def test_preregistration_is_enforced():
    try:
        tc._require_preregistration({"prediction": "", "kill_condition": ""})
    except SystemExit:
        return
    raise AssertionError("pilot must refuse to run without a pre-registration")


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\nconscience-pilot smoke tests passed.")


if __name__ == "__main__":
    main()
