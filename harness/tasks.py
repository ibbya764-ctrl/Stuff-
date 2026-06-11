"""Synthetic long-range tasks for the science arm (ladder rungs 2-4).

Both tasks ship a train length and LONGER eval lengths, because the
quantity of interest for the Part 3b riders is train-short/eval-long
extrapolation, not in-distribution fit.

Token map (shared):  0 = PAD/noise-slot, 1 = DELIMITER, 2 = NOISE,
content tokens are 3..vocab-1.
"""

from __future__ import annotations

import torch

PAD, DELIM, NOISE = 0, 1, 2
CONTENT_OFFSET = 3


def copy_batch(batch: int, length: int, vocab: int, device="cpu", generator=None):
    """Plain copying: [c_1..c_L][DELIM][PAD x L] -> predict c_1..c_L."""
    n_content = vocab - CONTENT_OFFSET
    c = torch.randint(0, n_content, (batch, length), device=device,
                      generator=generator) + CONTENT_OFFSET
    x = torch.cat([c,
                   torch.full((batch, 1), DELIM, device=device),
                   torch.full((batch, length), PAD, device=device)], dim=1)
    y = torch.full_like(x, -100)            # ignore_index everywhere except...
    y[:, length + 1:] = c                   # ...the reproduction window
    return x, y


def selective_copy_batch(batch: int, length: int, vocab: int, n_keys: int = 8,
                         device="cpu", generator=None):
    """Selective copy: n_keys content tokens scattered in NOISE; after the
    delimiter, reproduce them in order.  Harder memory task (Mamba-paper
    style); same extrapolation protocol."""
    assert n_keys < length
    n_content = vocab - CONTENT_OFFSET
    x_seq = torch.full((batch, length), NOISE, device=device)
    keys = torch.randint(0, n_content, (batch, n_keys), device=device,
                         generator=generator) + CONTENT_OFFSET
    # distinct random positions per row
    pos = torch.argsort(torch.rand(batch, length, device=device,
                                   generator=generator), dim=1)[:, :n_keys]
    pos = torch.sort(pos, dim=1).values
    x_seq.scatter_(1, pos, keys)
    x = torch.cat([x_seq,
                   torch.full((batch, 1), DELIM, device=device),
                   torch.full((batch, n_keys), PAD, device=device)], dim=1)
    y = torch.full_like(x, -100)
    y[:, length + 1:] = keys
    return x, y


TASKS = {"copy": copy_batch, "selective_copy": selective_copy_batch}


def get_task(name: str):
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name]


@torch.no_grad()
def evaluate(model, task_name: str, length: int, vocab: int, batches: int = 8,
             batch_size: int = 32, device="cpu", **task_kw) -> float:
    """Exact-match token accuracy on the prediction window."""
    task = get_task(task_name)
    model.eval()
    correct = total = 0
    for _ in range(batches):
        x, y = task(batch_size, length, vocab, device=device, **task_kw)
        pred = model(x).argmax(-1)
        mask = y != -100
        correct += int((pred[mask] == y[mask]).sum())
        total += int(mask.sum())
    model.train()
    return correct / max(total, 1)
