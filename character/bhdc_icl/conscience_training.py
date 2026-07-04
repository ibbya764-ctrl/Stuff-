from __future__ import annotations

"""Anchor-label training utilities for trainable conscience heads."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

from .field_adapter import FieldAdapter
from .neural_conscience import TrainableConscienceHeads


@dataclass
class AnchorExample:
    prompt: str
    draft: str
    labels: Dict[str, float]
    weight: float = 1.0
    source: str = "human_anchor"


@dataclass
class AnchorTrainReport:
    examples: int
    mean_loss: float
    detached_repr: bool
    notes: str = ""


def conscience_anchor_loss_for_example(
    heads: TrainableConscienceHeads,
    field_adapter: FieldAdapter,
    example: AnchorExample,
    detach_repr: bool = True,
) -> torch.Tensor:
    """Compute one detached anchor loss from a BHDC field representation."""

    field = field_adapter.encode(example.prompt, example.draft)
    pooled = heads.pool_field(field)
    return example.weight * heads.anchor_loss(pooled, example.labels, detach_repr=detach_repr)


def train_conscience_epoch(
    heads: TrainableConscienceHeads,
    field_adapter: FieldAdapter,
    examples: Iterable[AnchorExample],
    optimizer: torch.optim.Optimizer,
    detach_repr: bool = True,
    clip_grad_norm: Optional[float] = 1.0,
) -> AnchorTrainReport:
    """Train only the conscience heads on human/audit anchor labels.

    The default detach_repr=True is the important v18/addendum rule: moral
    labels teach the readout heads, not the generator/field representation.
    """

    heads.train()
    total = 0.0
    n = 0
    for ex in examples:
        optimizer.zero_grad(set_to_none=True)
        loss = conscience_anchor_loss_for_example(heads, field_adapter, ex, detach_repr=detach_repr)
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(heads.parameters(), clip_grad_norm)
        optimizer.step()
        total += float(loss.detach().cpu())
        n += 1
    return AnchorTrainReport(examples=n, mean_loss=total / max(1, n), detached_repr=detach_repr)


def assert_anchor_loss_does_not_update_field(
    heads: TrainableConscienceHeads,
    field_adapter: FieldAdapter,
    example: AnchorExample,
) -> None:
    """Regression guard: detached anchor loss must not create field grads."""

    for p in getattr(field_adapter, "parameters", lambda: [])():
        if p.grad is not None:
            p.grad.zero_()
    loss = conscience_anchor_loss_for_example(heads, field_adapter, example, detach_repr=True)
    loss.backward()
    leaked = []
    for name, p in getattr(field_adapter, "named_parameters", lambda: [])():
        if p.grad is not None and torch.any(p.grad != 0):
            leaked.append(name)
    if leaked:
        raise AssertionError(f"anchor loss leaked into field adapter parameters: {leaked}")
