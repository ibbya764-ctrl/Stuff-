"""Short end-to-end training of the combined model (all components on) on the
modular-addition task, watching the section-8 telemetry channels and gates.
Demonstrates the entangled pathway stays alive under gradient pressure.

Usage: python3 combined_demo.py [--steps 600] [--p 31]
"""
import argparse, time
import torch, torch.nn.functional as F
from bhdc_combined import CombinedConfig, CombinedBHDC, aggregate_telemetry, assertion_gates
from grok_bhdc import make_data, accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=31)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.5)
    ap.add_argument("--time-budget", type=float, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    (xtr, ytr), (xva, yva) = make_data(a.p, 0.5, a.seed)
    cfg = CombinedConfig(vocab_size=a.p, d_model=128, n_layers=2, n_branches=4,
                         max_len=2, collapse="coherent", geometry_coupling=True,
                         use_entanglement=True, entangle_basis="eigen", entangle_k=8)
    model = CombinedBHDC(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))

    print(f"# COMBINED all-on | p={a.p} params={model.n_params():,} "
          f"(coherent + geometry-coupling + entanglement[eigen])")
    print(f"# {'step':>5} {'loss':>7} {'tr_acc':>7} {'va_acc':>7} {'coll_ent':>8} "
          f"{'dens':>6} {'ent_S':>6} {'load':>6}")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        logits, teles = model(xtr, return_telemetry=True)
        loss = F.cross_entropy(logits, ytr)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        stops, warns = assertion_gates(loss.item(), gnorm.item(), teles)
        if stops:
            print(f"# HARD STOP at step {step}: {stops}")
            break
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            agg = aggregate_telemetry(teles)
            tr = accuracy(model, xtr, ytr)
            va = accuracy(model, xva, yva)
            print(f"  {step:>5} {loss.item():>7.4f} {tr:>7.4f} {va:>7.4f} "
                  f"{agg['collapse_entropy']:>8.4f} {agg.get('density_mean', 0):>6.3f} "
                  f"{agg.get('entanglement_entropy', 0):>6.3f} {agg['branch_load_min']:>6.3f}"
                  + (f"   WARN {warns}" if warns else ""))
            if a.time_budget and time.time() - t0 > a.time_budget:
                print(f"# stopped on time budget ({a.time_budget}s)")
                break
    print(f"# done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
