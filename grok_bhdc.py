"""BHDC grokking read -- the cheap decisive experiment (paper section 9).

Task: modular addition (a + b) mod p. It lives on a circle, which is the
toroidal geometry in the bank; the terminal |.|^2 interference cross-term is
exactly the nonlinear interaction these tasks need; and grokking is a regime
where inductive bias dramatically changes the outcome.

PRE-REGISTRATION (program discipline -- written before contact, able to fail):
  PREDICTION: with the geometry/phase machinery on (coherent collapse, torus in
    the bank), the model reaches a clean late val-accuracy jump well above the
    1/p chance line on a held-out split -- i.e. it groks, not just memorizes.
  PRIMARY ARM: collapse=coherent (real amplitudes), abs2 collapse.
  KILL CONDITION: if val accuracy never exceeds 2/p (twice chance) within the
    step budget while train accuracy saturates at ~1.0, the bias did not help
    on this task -> record [TESTED-NEGATIVE], do not soften.
  MATCHED BUDGET: compare arms only at identical p / split / steps / width.

This is a CPU-runnable toy. Defaults are sized for a few minutes on CPU; full
grokking delay typically wants more steps (raise --steps on the workstation).

Usage:
  python3 grok_bhdc.py --steps 3000 --collapse coherent
  python3 grok_bhdc.py --steps 3000 --collapse single   # baseline arm
"""
import argparse
import time
import torch
import torch.nn.functional as F
from bhdc_v1_1 import BHDCConfig, BHDCModel


def make_data(p, train_frac, seed):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    x = torch.stack([a, b], dim=1)          # tokens [a, b], vocab = p
    y = (a + b) % p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_tr = int(train_frac * x.shape[0])
    tr, va = perm[:n_tr], perm[n_tr:]
    return (x[tr], y[tr]), (x[va], y[va])


@torch.no_grad()
def accuracy(model, x, y, bs=4096):
    model.eval()
    correct = 0
    for i in range(0, x.shape[0], bs):
        logits = model(x[i:i + bs])
        correct += (logits.argmax(-1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / x.shape[0]


def run(args):
    torch.manual_seed(args.seed)
    (xtr, ytr), (xva, yva) = make_data(args.p, args.train_frac, args.seed)
    cfg = BHDCConfig(vocab_size=args.p, d_model=args.d_model, n_layers=args.n_layers,
                     n_branches=args.n_branches, max_len=2, collapse=args.collapse,
                     collapse_nonlin=args.collapse_nonlin)
    model = BHDCModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))

    chance = 1.0 / args.p
    print(f"# BHDC grokking | p={args.p} split={args.train_frac} collapse={args.collapse}"
          f" d_model={args.d_model} L={args.n_layers} branches={args.n_branches}")
    print(f"# params={n_params:,}  train={xtr.shape[0]}  val={xva.shape[0]}  chance={chance:.4f}")
    print(f"# {'step':>6} {'loss':>8} {'train_acc':>10} {'val_acc':>9} {'coll_ent':>9}")

    best_val, train_sat = 0.0, False
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.batch and args.batch < xtr.shape[0]:
            idx = torch.randint(0, xtr.shape[0], (args.batch,))
            xb, yb = xtr[idx], ytr[idx]
        else:
            xb, yb = xtr, ytr
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            _, teles = model(xtr[:512], return_telemetry=True)
            ent = sum(t["collapse_entropy"] for t in teles) / len(teles)
            tr_acc = accuracy(model, xtr, ytr)
            va_acc = accuracy(model, xva, yva)
            best_val = max(best_val, va_acc)
            train_sat = train_sat or tr_acc > 0.99
            print(f"  {step:>6} {loss.item():>8.4f} {tr_acc:>10.4f} {va_acc:>9.4f} {ent:>9.4f}")
            if args.time_budget and (time.time() - t0) > args.time_budget:
                print(f"# stopped on time budget ({args.time_budget}s)")
                break

    dt = time.time() - t0
    print(f"# done in {dt:.1f}s | best_val={best_val:.4f} | train_saturated={train_sat}")
    # pre-registered verdict
    if best_val > 2 * chance:
        print(f"# VERDICT: val {best_val:.4f} > 2x chance ({2*chance:.4f}) -- signal present "
              f"(grokking gate cleared at this budget).")
    elif train_sat:
        print(f"# VERDICT: [TESTED-NEGATIVE] train saturated but val {best_val:.4f} <= 2x chance "
              f"({2*chance:.4f}) -- memorized, did not generalize at this budget.")
    else:
        print(f"# VERDICT: inconclusive -- train not yet saturated at this budget; raise --steps.")
    return best_val, train_sat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=0, help="0 = full batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0, help="weight decay (grokking wants high wd)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-branches", type=int, default=4)
    ap.add_argument("--collapse", default="coherent")
    ap.add_argument("--collapse-nonlin", default="abs2")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=0, help="seconds, 0 = no limit")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
