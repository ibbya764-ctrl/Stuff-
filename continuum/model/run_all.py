"""Run every module self-test + a CP2Model forward pass. One-command validation.

    python3 -m continuum.model.run_all

This checks that the full architecture composes and each subsystem behaves as
specified ON CPU. It does NOT train or validate the architecture -- see
README.md for the scope statement.
"""

from __future__ import annotations

import importlib
import io
import contextlib

MODULES = [
    "continuum.anti_collapse", "continuum.scale_dynamics",
    "continuum.model.operator", "continuum.model.field", "continuum.model.density",
    "continuum.model.branches", "continuum.model.observation",
    "continuum.model.consolidation", "continuum.model.resolution",
    "continuum.model.curvature", "continuum.model.geometry",
    "continuum.model.memory", "continuum.model.collapse", "continuum.model.model",
]


def main() -> None:
    passed, failed = [], []
    for name in MODULES:
        mod = importlib.import_module(name)
        runner = getattr(mod, "_selftest", None)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                if runner is not None:
                    runner()
                else:
                    # modules whose checks live under __main__ expose them via
                    # a function; fall back to re-exec of the main block.
                    spec = importlib.util.find_spec(name)
                    code = compile(open(spec.origin).read(), spec.origin, "exec")
                    g = {"__name__": "__main__", "__file__": spec.origin,
                         "__package__": name.rpartition(".")[0]}
                    exec(code, g)
            passed.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print("-" * 60)
    print(f"  {len(passed)}/{len(MODULES)} modules OK")
    if failed:
        raise SystemExit(1)
    print("  Full continuous-hypothesis-field architecture composes and runs "
          "on CPU.")
    print("  [STRUCTURE COMPLETE -- NOT TRAINED, NOT VALIDATED. See README.md.]")


if __name__ == "__main__":
    main()
