"""Matched-budget experiment runner (ladder rungs 2-4 and beyond).

Usage:
    python -m harness.runner configs/rung3_critical_line.json
    python -m harness.runner configs/smoke.json --quick

Guarantees enforced here, per the v3 record:

  * PRE-REGISTRATION: the config must contain a non-empty
    ``preregistration.prediction`` and ``preregistration.kill_condition``;
    the runner refuses to start otherwise, and logs them (with a config
    hash and timestamp) before the first step.
  * MATCHED BUDGET: parameter count and total training tokens are logged;
    ``compare_runs`` flags any A/B pair whose budgets differ by >2%.
  * TELEMETRY PER CHECKPOINT: dynamics spectrum (spacing ratio, width
    dispersion) and weight Hill-alpha, via the repo-root
    spectral_telemetry module.
  * EXTRAPOLATION EVAL: accuracy at train length AND at longer eval
    lengths every checkpoint (the Part 3b riders are extrapolation bets).

Outputs: runs/<name>/metrics.jsonl + summary.json (+ registration.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

import spectral_telemetry as st
from harness.ssm import SpectralSSMModel
from harness.tasks import evaluate, get_task

DEFAULTS = {
    "seed": 0,
    "vocab": 16,
    "model": {"d_model": 128, "n_layers": 4, "d_state": 64,
              "width_mode": "free", "freq_init": "s4", "dt": 1e-2},
    "task": {"name": "copy", "train_length": 64,
             "eval_lengths": [64, 128, 256], "task_kw": {}},
    "train": {"steps": 4000, "batch_size": 32, "lr": 3e-3,
              "weight_decay": 0.01, "gue_reg_weight": 0.0,
              "checkpoint_every": 500},
}


def _merged(cfg: dict) -> dict:
    out = json.loads(json.dumps(DEFAULTS))
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _require_preregistration(cfg: dict) -> dict:
    pre = cfg.get("preregistration", {})
    if not (pre.get("prediction", "").strip() and
            pre.get("kill_condition", "").strip()):
        raise SystemExit(
            "REFUSING TO RUN: config lacks preregistration.prediction / "
            ".kill_condition. Write the prediction before the experiment "
            "(the periodogram discipline)."
        )
    return pre


def telemetry_snapshot(model: SpectralSSMModel) -> dict:
    lam = model.dynamics_spectrum()
    widths = -lam.real
    # spacing ratio per layer, then averaged: layers initialize identically,
    # so pooling their spectra creates spurious near-zero spacings.
    per_layer = [st.spacing_ratio(
        b.ssm.eigenvalues().detach().cpu().numpy().imag)
        for b in model.blocks]
    snap = {
        "dyn_spacing_ratio": float(np.mean(per_layer)),
        "dyn_width_mean": float(widths.mean()),
        "dyn_width_dispersion": float(widths.std() / (widths.mean() + 1e-12)),
        "weight_alpha": {},
    }
    for name, w in model.weight_matrices().items():
        if min(w.shape) >= 32:                       # Hill needs a real tail
            snap["weight_alpha"][name] = round(st.hill_alpha(w)["alpha"], 3)
    return snap


def run(config_path: str, quick: bool = False) -> dict:
    with open(config_path) as f:
        raw = json.load(f)
    cfg = _merged(raw)
    pre = _require_preregistration(raw)
    if quick:
        cfg["train"]["steps"] = min(cfg["train"]["steps"], 300)
        cfg["train"]["checkpoint_every"] = 100

    name = cfg.get("name") or os.path.splitext(os.path.basename(config_path))[0]
    run_dir = os.path.join("runs", name)
    os.makedirs(run_dir, exist_ok=True)

    cfg_hash = hashlib.sha256(
        json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12]
    with open(os.path.join(run_dir, "registration.json"), "w") as f:
        json.dump({"preregistration": pre, "config_hash": cfg_hash,
                   "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "config": cfg}, f, indent=2)

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpectralSSMModel(vocab_size=cfg["vocab"], **cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tcfg, kcfg = cfg["train"], cfg["task"]
    task = get_task(kcfg["name"])
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    total_tokens = tcfg["steps"] * tcfg["batch_size"] * (
        2 * kcfg["train_length"] + 1)
    print(f"[{name}] params={n_params:,} budget_tokens={total_tokens:,} "
          f"device={device} hash={cfg_hash}")

    log_path = os.path.join(run_dir, "metrics.jsonl")
    log = open(log_path, "a")
    model.train()
    for step in range(tcfg["steps"]):
        x, y = task(tcfg["batch_size"], kcfg["train_length"], cfg["vocab"],
                    device=device, **kcfg["task_kw"])
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if tcfg["gue_reg_weight"] > 0:
            loss = loss + tcfg["gue_reg_weight"] * model.gue_regularizer()
        opt.zero_grad()
        loss.backward()
        opt.step()

        last = step == tcfg["steps"] - 1
        if step % tcfg["checkpoint_every"] == 0 or last:
            accs = {str(L): round(evaluate(
                model, kcfg["name"], L, cfg["vocab"], device=device,
                **kcfg["task_kw"]), 4) for L in kcfg["eval_lengths"]}
            rec = {"step": step, "loss": round(float(loss.item()), 5),
                   "acc": accs, "telemetry": telemetry_snapshot(model)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"  step {step:6d} loss {rec['loss']:.4f} acc {accs}")
    log.close()

    summary = {"name": name, "config_hash": cfg_hash, "n_params": n_params,
               "budget_tokens": total_tokens, "final": rec,
               "preregistration": pre}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    return summary


def compare_runs(*summary_paths: str) -> None:
    """Matched-budget check + side-by-side finals for an A/B(/C) set."""
    sums = []
    for p in summary_paths:
        with open(p) as f:
            sums.append(json.load(f))
    base = sums[0]
    for s in sums[1:]:
        for key in ("n_params", "budget_tokens"):
            a, b = base[key], s[key]
            if abs(a - b) / max(a, b) > 0.02:
                print(f"!! BUDGET MISMATCH {key}: {base['name']}={a:,} "
                      f"vs {s['name']}={b:,} (>2%) — comparison is invalid")
    for s in sums:
        print(f"{s['name']:32s} acc={s['final']['acc']} "
              f"r~={s['final']['telemetry']['dyn_spacing_ratio']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--quick", action="store_true",
                    help="300-step smoke run (not a real experiment)")
    args = ap.parse_args()
    run(args.config, quick=args.quick)
