from __future__ import annotations

"""Renewal hooks for connecting ICL to a real BHDC geometry/operator stack."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import torch

from .content_matched_modes import ContentMatchedModeBank
from .tensor_ops import l2_normalize


@dataclass
class RenewalHookReport:
    called: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class BHDCGeometryRenewalHook:
    """Duck-typed renewal bridge for BHDC/v17/v18 geometry modules.

    It calls whichever renewal/stress methods exist on the provided model. This
    is intentionally adapter-style because the BHDC codebase may name these
    pieces differently across versions.
    """

    reset_method_names = (
        "reset_plastic_geometry",
        "renew_geometry",
        "reset_geometry",
        "sleep_reset",
        "reset_plastic_paths",
        "refresh_credit_linears",
        "reset_growth_logits",
    )
    stress_method_names = (
        "stress_operator_prototypes",
        "stress_modes",
        "probe_operator_modes",
        "renewal_stress",
    )

    def __init__(self, model: Any, noise_scale: float = 0.005, decay_unstable: float = 0.995):
        self.model = model
        self.noise_scale = noise_scale
        self.decay_unstable = decay_unstable
        self.last_report = RenewalHookReport()

    def reset_geometry(self) -> RenewalHookReport:
        report = RenewalHookReport()
        for name in self.reset_method_names:
            fn = getattr(self.model, name, None)
            if callable(fn):
                try:
                    fn()
                    report.called.append(name)
                except TypeError:
                    fn(step=None)
                    report.called.append(name)
                except Exception as exc:
                    report.notes.append(f"{name} failed: {exc}")
        if not report.called:
            report.notes.append("no compatible geometry reset method found")
        self.last_report = report
        return report

    def stress_modes(self, mode_bank: ContentMatchedModeBank) -> RenewalHookReport:
        report = self.last_report
        for name in self.stress_method_names:
            fn = getattr(self.model, name, None)
            if callable(fn):
                try:
                    fn(mode_bank)
                    report.called.append(name)
                except TypeError:
                    fn()
                    report.called.append(name)
                except Exception as exc:
                    report.notes.append(f"{name} failed: {exc}")
        # Generic minimal stress: tiny noise + re-normalisation. This makes the
        # survival test meaningful even before a real geometry reset is wired.
        for slot in mode_bank.slots:
            if self.noise_scale > 0:
                slot.prototype = l2_normalize(slot.prototype + self.noise_scale * torch.randn_like(slot.prototype))
            slot.stability_credit *= self.decay_unstable
        return report


class CompositeRenewalHook:
    """Combine multiple renewal hooks."""

    def __init__(self, hooks: Iterable[BHDCGeometryRenewalHook]):
        self.hooks = list(hooks)

    def reset_geometry(self) -> None:
        for hook in self.hooks:
            hook.reset_geometry()

    def stress_modes(self, mode_bank: ContentMatchedModeBank) -> None:
        for hook in self.hooks:
            hook.stress_modes(mode_bank)
