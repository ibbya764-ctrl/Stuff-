"""Pre-registered, matched-budget runner for the BHDC GPU experiments.

Discipline enforced (program non-negotiables):
  * REFUSES any config without a written preregistration.prediction AND
    .kill_condition (pre-registration is the point -- do not work around it).
  * Matched budget: refuses an A/B whose arms differ in parameter count by >2%
    unless the config sets "allow_budget_delta": true with a reason.
  * Multi-seed: runs every arm over all `seeds`, reports mean +/- std.
  * Telemetry + assertion gates (section 8) every eval; hard-stops on NaN.
  * Echoes the pre-registered prediction + kill condition next to the numbers so
    the verdict is made against what was written, not post-hoc.

Runs on GPU automatically when available (CombinedBHDC is device-agnostic; the
scan, geometry, and K x K matrix-exp all follow the input device).

Usage:
  python3 bhdc_runner.py configs/bhdc_collapse_ab.json
  python3 bhdc_runner.py configs/bhdc_grok_p97.json --device cuda
"""
import argparse, copy, json, os, time
import torch, torch.nn.functional as F

from bhdc_combined import CombinedConfig, CombinedBHDC, aggregate_telemetry, assertion_gates
from grok_bhdc import make_data, accuracy

MODEL_KEYS = set(CombinedConfig.__dataclass_fields__) - {"vocab_size", "max_len"}


def build_model(p, model_kw):
    bad = set(model_kw) - MODEL_KEYS
    if bad:
        raise ValueError(f"unknown model keys: {bad}")
    return CombinedBHDC(CombinedConfig(vocab_size=p, max_len=2, **model_kw))


def arms_of(cfg):
    """Yield (label, model_kw) for each arm (single-arm config -> one entry)."""
    base = cfg.get("model", {})
    if "arms" in cfg:
        for arm in cfg["arms"]:
            kw = copy.deepcopy(base)
            kw.update(arm.get("model", {}))
            yield arm["label"], kw
    else:
        yield cfg.get("name", "arm"), copy.deepcopy(base)


def check_matched_budget(cfg, p):
    counts = {label: build_model(p, kw).n_params() for label, kw in arms_of(cfg)}
    lo, hi = min(counts.values()), max(counts.values())
    spread = (hi - lo) / lo if lo else 0.0
    print("# parameter budget per arm:")
    for label, c in counts.items():
        print(f"#   {label:22s} {c:>9,}")
    print(f"# spread = {100*spread:.2f}%")
    if spread > 0.02 and not cfg.get("allow_budget_delta", False):
        raise SystemExit(
            f"REFUSED: arm param spread {100*spread:.2f}% > 2% matched-budget limit. "
            f"Add a capacity-matched control (capacity_pad), or set "
            f'"allow_budget_delta": true with a documented reason.')
    return counts


def train_arm(p, train_frac, model_kw, train_kw, seed, device):
    torch.manual_seed(seed)
    (xtr, ytr), (xva, yva) = make_data(p, train_frac, seed)
    xtr, ytr, xva, yva = (t.to(device) for t in (xtr, ytr, xva, yva))
    model = build_model(p, model_kw).to(device)
    steps = train_kw.get("steps", 30000)
    bs = train_kw.get("batch_size", 0)
    lr = train_kw.get("lr", 1e-3)
    wd = train_kw.get("wd", 1.0)
    eval_every = train_kw.get("eval_every", 200)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))

    best_val, train_sat, stop_reason = 0.0, False, None
    warns_seen, last_agg = set(), {}
    t0 = time.time()
    for step in range(1, steps + 1):
        if bs and bs < xtr.shape[0]:
            idx = torch.randint(0, xtr.shape[0], (bs,), device=device)
            xb, yb = xtr[idx], ytr[idx]
        else:
            xb, yb = xtr, ytr
        opt.zero_grad()
        logits, teles = model(xb, return_telemetry=True)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        stops, warns = assertion_gates(loss.item(), gnorm.item(), teles)
        if stops:
            stop_reason = f"hard stop at step {step}: {stops}"
            break
        warns_seen.update(warns)
        opt.step()
        if step % eval_every == 0 or step == steps:
            last_agg = aggregate_telemetry(teles)
            tr, va = accuracy(model, xtr, ytr), accuracy(model, xva, yva)
            best_val = max(best_val, va)
            train_sat = train_sat or tr > 0.99
    return {
        "seed": seed, "best_val": best_val, "train_saturated": train_sat,
        "params": model.n_params(), "seconds": round(time.time() - t0, 1),
        "telemetry": last_agg, "warnings": sorted(warns_seen), "stop_reason": stop_reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="results JSON (default results/<name>.json)")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = json.load(f)

    prereg = cfg.get("preregistration", {})
    if not prereg.get("prediction", "").strip() or not prereg.get("kill_condition", "").strip():
        raise SystemExit("REFUSED: config has no written prediction + kill_condition "
                         "(pre-registration is mandatory).")

    p = cfg["p"]
    train_frac = cfg.get("train_frac", 0.5)
    train_kw = cfg.get("train", {})
    seeds = cfg.get("seeds", [0])
    chance = 1.0 / p

    print(f"# ===== {cfg['name']} | device={a.device} | p={p} | seeds={seeds} =====")
    print(f"# PREDICTION: {prereg['prediction']}")
    print(f"# KILL:       {prereg['kill_condition']}")
    print(f"# chance={chance:.4f}  2x chance={2*chance:.4f}")
    check_matched_budget(cfg, p)

    results = {}
    for label, model_kw in arms_of(cfg):
        runs = [train_arm(p, train_frac, model_kw, train_kw, s, a.device) for s in seeds]
        bv = torch.tensor([r["best_val"] for r in runs])
        results[label] = {"runs": runs, "mean_best_val": bv.mean().item(),
                          "std_best_val": bv.std(unbiased=False).item(),
                          "all_train_saturated": all(r["train_saturated"] for r in runs)}
        print(f"\n## arm: {label}")
        for r in runs:
            extra = f"  STOP[{r['stop_reason']}]" if r["stop_reason"] else ""
            warn = f"  WARN{r['warnings']}" if r["warnings"] else ""
            print(f"   seed {r['seed']}: best_val={r['best_val']:.4f} "
                  f"train_sat={r['train_saturated']} {r['seconds']}s "
                  f"tele={ {k: round(v,3) for k,v in r['telemetry'].items()} }{warn}{extra}")
        print(f"   => mean best_val = {results[label]['mean_best_val']:.4f} "
              f"+/- {results[label]['std_best_val']:.4f}")

    print(f"\n# ===== SUMMARY ({cfg['name']}) =====")
    print(f"# {'arm':22s} {'mean_val':>9} {'std':>7} {'train_sat':>10} {'>2x?':>6}")
    for label, r in results.items():
        print(f"# {label:22s} {r['mean_best_val']:>9.4f} {r['std_best_val']:>7.4f} "
              f"{str(r['all_train_saturated']):>10} {str(r['mean_best_val'] > 2*chance):>6}")
    base = cfg.get("baseline_arm")
    if base and base in results:
        bvb = results[base]["mean_best_val"]
        print(f"#\n# deltas vs baseline '{base}' ({bvb:.4f}):")
        for label, r in results.items():
            if label != base:
                print(f"#   {label:22s} {r['mean_best_val']-bvb:+.4f}")
    print("# Verdict is made by reading these numbers against the PREDICTION/KILL above.")

    out = a.out or os.path.join("results", f"{cfg['name']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": cfg, "device": a.device, "results": results}, f, indent=2)
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
