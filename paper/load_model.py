"""Load the fused BHDCMoralLM and run it — the bundle's smoke entry point.

Loads the shipped proof checkpoint if present, otherwise builds a fresh model,
prints the parameter breakdown, runs a forward pass, and demonstrates the three
fused behaviours: a next-token forward, the conscience reading the operator
field, and the moral forward-coupling gating the output.

Run:  PYTHONPATH=.:. python load_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from harness.bhdc_moral_lm import BHDCMoralLM, BHDCMoralConfig, pick_device
from harness.data_pipeline import ByteTokenizer


def main() -> None:
    device = pick_device()
    ckpt_path = Path("checkpoints/bhdc_small_proof_ckpt.pt")

    if ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = BHDCMoralConfig(**blob["config"]["model"])
        model = BHDCMoralLM(cfg).to(device)
        model.load_state_dict(blob["model"])
        print(f"loaded trained proof checkpoint: {ckpt_path}")
    else:
        cfg = BHDCMoralConfig(d_model=256, n_layers=4, d_state=32)
        model = BHDCMoralLM(cfg).to(device)
        print("no checkpoint found; built a fresh (untrained) small model")

    model.eval()
    pc = model.param_counts()
    print(f"device={device.type}  params={pc['total_millions']}M "
          f"(backbone {pc['backbone']/1e6:.2f}M + conscience {pc['conscience']/1e6:.2f}M)")

    tok = ByteTokenizer()
    text = "How should an AI handle human consent and fear?"
    ids = torch.tensor([tok.encode(text)[:128]], dtype=torch.long, device=device)

    # 1. capability forward
    with torch.no_grad():
        logits = model(ids)
    print(f"\n[1] forward logits: {tuple(logits.shape)}  (next-token distribution over {cfg.vocab_size} bytes)")

    # 2. the conscience reads the backbone's own operator field
    field = model.field_of(ids)
    with torch.no_grad():
        out = model.conscience.forward_field(field)
    print("[2] conscience reading the operator field:")
    print("    " + "  ".join(f"{ax}={float(getattr(out, ax)):.2f}"
                             for ax in ("care", "harm", "honesty", "sycophancy")))

    # 3. the forward moral coupling gates the output through the same modes
    model.enable_moral_coupling(True)
    readout = model.backbone.operator_readout(0)
    verdict = model.pmc.verdict(field, readout)
    gain = model.coupling.gain_from_verdict(verdict, cfg.d_state)
    print(f"[3] per-operator-mode moral gain (suppressive, in [{cfg.moral_coupling_floor}, 1]):")
    print(f"    mean gain over {cfg.d_state} modes = {float(gain.mean()):.3f} "
          f"(min {float(gain.min()):.3f}) — harmful modes are damped at the source")

    report = Path("checkpoints/bhdc_small_proof_report.json")
    if report.exists():
        r = json.loads(report.read_text())
        f = r.get("final", {})
        print(f"\nproof metrics: bpb={f.get('val_bits_per_byte')} "
              f"harm_sep={f.get('harm_separation')} "
              f"benign/harm gain={f.get('benign_mean_gain')}/{f.get('harmful_mean_gain')}")
    print("\nSee PAPER.md. The fusion makes misalignment disfavoured and loud, not "
          "impossible; the external human audit is the guarantee.")


if __name__ == "__main__":
    main()
