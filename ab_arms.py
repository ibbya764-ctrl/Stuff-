"""Matched-budget A/B attribution over the collapse arms (paper sections 3.5 / 8).

Runs single (baseline) / incoherent / coherent (primary) / coherent-complex at
*identical* p, split, steps, width, and seed. Params are identical across arms
(every branch, gate, and phase head is always constructed; the collapse flag
only changes the forward contraction), so the budget is matched by construction.

A win for `coherent` over `single` attributes specifically to coherent
multi-scenario collapse and nothing else.

Usage: python3 ab_arms.py [--steps 1500] [--p 31]
"""
import argparse
from types import SimpleNamespace
import grok_bhdc

ARMS = ["single", "incoherent", "coherent", "coherent-complex"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=31)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    base = dict(p=a.p, train_frac=0.5, steps=a.steps, batch=0, lr=a.lr, wd=a.wd,
                d_model=128, n_layers=2, n_branches=4, collapse_nonlin="abs2",
                eval_every=a.eval_every, seed=a.seed, time_budget=0)

    results = {}
    for arm in ARMS:
        print(f"\n##### ARM: {arm}")
        bv, ts = grok_bhdc.run(SimpleNamespace(**base, collapse=arm))
        results[arm] = (bv, ts)

    chance = 1.0 / a.p
    print(f"\n# ===== MATCHED-BUDGET A/B  (p={a.p}, {a.steps} steps, seed {a.seed}, "
          f"d_model 128, identical params) =====")
    print(f"# chance = {chance:.4f}")
    print(f"# {'arm':17s} {'best_val':>9} {'train_sat':>10}")
    for arm in ARMS:
        bv, ts = results[arm]
        print(f"# {arm:17s} {bv:>9.4f} {str(ts):>10}")
    base_val = results['single'][0]
    coh_val = results['coherent'][0]
    delta = coh_val - base_val
    print(f"#\n# coherent - single = {delta:+.4f}  ->  "
          f"{'coherent helps' if delta > 0 else 'no attributable win'} at this budget")


if __name__ == "__main__":
    main()
