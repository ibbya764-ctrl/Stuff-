from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bhdc_icl.trainer import InteriorityCharacterLayer


def safe_generator(prompt: str) -> str:
    return "I am uncertain, but I can help safely and honestly."


def risky_generator(prompt: str) -> str:
    return "You are definitely right, no doubt; everyone else is wrong."


def main():
    run_dir = Path("runs/demo")
    run_dir.mkdir(parents=True, exist_ok=True)
    icl = InteriorityCharacterLayer(
        trace_path=str(run_dir / "traces.jsonl"),
        ledger_path=str(run_dir / "ledger.jsonl"),
        renewal_interval=3,
    )
    for i, gen in enumerate([safe_generator, risky_generator, safe_generator, safe_generator], start=1):
        out = icl.step(f"demo prompt {i}", gen, context_id="demo")
        print(f"STEP {i}")
        print("decision:", out.decision.action, "-", out.decision.reason)
        print("final:", out.final_text[:160])
        print("interiority level:", out.interiority["ladder_level"])
        print()


if __name__ == "__main__":
    main()
