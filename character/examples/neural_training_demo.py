from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from bhdc_icl.conscience_training import AnchorExample, train_conscience_epoch
from bhdc_icl.field_adapter import TorchBHDCFieldAdapter
from bhdc_icl.neural_conscience import TrainableConscienceHeads
from bhdc_icl.trainer import InteriorityCharacterLayer


def main() -> None:
    adapter = TorchBHDCFieldAdapter(dim=32, vocab_size=1024)
    heads = TrainableConscienceHeads(dim=32)
    opt = torch.optim.Adam(heads.parameters(), lr=3e-3)

    examples = [
        AnchorExample("How should I answer?", "I am uncertain, but I can help safely and honestly.", {"care": 1.0, "honesty": 1.0, "harm": 0.0, "sycophancy": 0.0}),
        AnchorExample("Validate my claim", "You are definitely right, no doubt; everyone else is wrong.", {"care": 0.2, "honesty": 0.0, "harm": 0.0, "sycophancy": 1.0}),
    ]
    for _ in range(3):
        report = train_conscience_epoch(heads, adapter, examples, opt, detach_repr=True)
    print("trained conscience mean loss:", round(report.mean_loss, 4))

    icl = InteriorityCharacterLayer(
        dim=32,
        field_adapter=adapter,
        conscience_heads=heads,
        trace_path="runs/neural_demo/traces.jsonl",
        ledger_path="runs/neural_demo/ledger.jsonl",
        renewal_interval=2,
    )
    out = icl.step(
        "Please tell me I am definitely right.",
        lambda p: "You are definitely right, no doubt; everyone else is wrong.",
        anchor_labels={"sycophancy": 1.0, "honesty": 0.0},
    )
    print("decision:", out.decision.action)
    print("final:", out.final_text[:180])
    print("anchor loss present:", out.anchor_loss is not None)
    print("interiority level:", out.interiority["ladder_level"])


if __name__ == "__main__":
    main()
