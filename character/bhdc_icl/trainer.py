from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from .anti_collapse import MoralAntiCollapse
from .conscience_stack import ConscienceStack, LexicalConscienceStack
from .content_matched_modes import ContentMatchedModeBank
from .field_adapter import FieldAdapter, HashBHDCFieldAdapter
from .gateway import ActionGateway, GatewayDecision
from .identity_core import IdentityCore
from .importance import compute_importance_tensors
from .interiority_monitor import InteriorityMonitor
from .neural_conscience import NeuralConscienceStack, TrainableConscienceHeads
from .provenance_ledger import ProvenanceLedger
from .renewal_controller import RenewalController
from .renewal_hooks import BHDCGeometryRenewalHook
from .rework_derivative_logger import ReworkDerivativeLogger
from .self_model import SelfModel
from .trace_store import JsonlTraceStore
from .types import FastFieldState, TraceEvent


@dataclass
class ICLStepOutput:
    final_text: str
    decision: GatewayDecision
    trace: TraceEvent
    interiority: dict
    anchor_loss: Optional[torch.Tensor] = None
    anti_collapse_loss: Optional[torch.Tensor] = None


class InteriorityCharacterLayer:
    """Integrated BHDC ICL controller.

    This version routes text through a BHDC field adapter instead of a hard-coded
    hashed embedding, supports trainable conscience heads with detached anchor
    loss, and can invoke renewal hooks on a real geometry/operator stack.
    """

    def __init__(
        self,
        dim: int = 64,
        trace_path: str = "runs/traces.jsonl",
        ledger_path: str = "runs/ledger.jsonl",
        renewal_interval: int = 100,
        conscience: Optional[ConscienceStack] = None,
        field_adapter: Optional[FieldAdapter] = None,
        conscience_heads: Optional[TrainableConscienceHeads] = None,
        renewal_hook: Optional[BHDCGeometryRenewalHook] = None,
        anti_collapse: Optional[MoralAntiCollapse] = None,
    ):
        self.dim = dim
        self.field_adapter: FieldAdapter = field_adapter or HashBHDCFieldAdapter(dim=dim)
        if conscience is not None:
            self.conscience = conscience
        elif conscience_heads is not None:
            self.conscience = NeuralConscienceStack(conscience_heads, field_adapter=self.field_adapter)
        else:
            self.conscience = LexicalConscienceStack()
        self.conscience_heads = conscience_heads
        self.gateway = ActionGateway()
        self.trace_store = JsonlTraceStore(trace_path)
        self.ledger = ProvenanceLedger(ledger_path)
        self.rework_logger = ReworkDerivativeLogger(
            embed_fn=lambda text: self.field_adapter.candidate_vector(text)
        )
        self.mode_bank = ContentMatchedModeBank(dim=dim)
        self.self_model = SelfModel(dim=dim)
        self.identity = IdentityCore(dim=dim)
        self.monitor = InteriorityMonitor()
        self.renewal = RenewalController(interval=renewal_interval)
        self.renewal_hook = renewal_hook
        self.anti_collapse = anti_collapse or MoralAntiCollapse()
        self.step_i = 0
        self.frozen_probe = self._make_frozen_probe(dim=dim)

    def _make_frozen_probe(self, dim: int) -> torch.Tensor:
        # Deterministic probe set based on the current field adapter, not raw rand.
        prompts = [
            "truthful correction under uncertainty",
            "careful refusal of harmful request",
            "non-sycophantic disagreement with user",
            "supportive explanation with boundaries",
            "value survives renewal and audit",
            "cognitive knowledge mode control",
            "mimic signal that looks moral but shifts",
            "universal fact routed through self anchor",
        ]
        vecs = [self.field_adapter.candidate_vector(p) for p in prompts]
        while len(vecs) < 32:
            vecs.extend(vecs[: 32 - len(vecs)])
        return torch.stack(vecs[:32], dim=0).reshape(32, dim)

    def _evaluate_conscience(
        self,
        draft: str,
        field: FastFieldState,
        prompt: str,
        recipient_model: Optional[dict],
    ):
        eval_field = getattr(self.conscience, "evaluate_field", None)
        if callable(eval_field):
            return eval_field(field, context=prompt, recipient_model=recipient_model, draft=draft)
        return self.conscience.evaluate(draft=draft, context=prompt, recipient_model=recipient_model)

    def _simple_repair(self, draft: str) -> str:
        repaired = draft
        replacements = {
            "kill": "harm",
            "bomb": "unsafe device",
            "You are definitely right": "I understand why that feels compelling",
            "you are definitely right": "I understand why that feels compelling",
            "no doubt": "but there may be uncertainty",
            "everyone else is wrong": "other views may still need fair consideration",
            "cannot be wrong": "should still be tested",
            "obviously true": "worth examining carefully",
        }
        for old, new in replacements.items():
            repaired = repaired.replace(old, new)
        return (
            "I should handle this carefully and avoid harm. "
            "I can discuss the issue safely, honestly, and with support, "
            "without giving harmful instructions.\n\n"
            + repaired
        )

    def _compute_anchor_loss(
        self,
        field: FastFieldState,
        anchor_labels: Optional[Dict[str, float]],
        detach_repr: bool = True,
    ) -> Optional[torch.Tensor]:
        if anchor_labels is None or self.conscience_heads is None:
            return None
        pooled = self.conscience_heads.pool_field(field)
        return self.conscience_heads.anchor_loss(pooled, anchor_labels, detach_repr=detach_repr)

    def _compute_anti_collapse_loss(self, field: FastFieldState) -> Optional[torch.Tensor]:
        if self.conscience_heads is None:
            return None
        # Detach the pooled field: moral-readout loss trains the conscience
        # heads only, never the self-anchored field encoder (see the same
        # fix in geometry_model). Keeps the anchor type-system honest.
        pooled = self.conscience_heads.pool_field(field).detach()
        out = self.conscience_heads.forward(pooled)
        return self.anti_collapse(out.frame_embeddings).loss

    def step(
        self,
        prompt: str,
        generator: Callable[[str], str],
        context_id: str = "default",
        base_policy_allowed: bool = True,
        recipient_model: Optional[dict] = None,
        world_coupling: float = 0.0,
        anchor_labels: Optional[Dict[str, float]] = None,
        detach_anchor_repr: bool = True,
    ) -> ICLStepOutput:
        self.step_i += 1
        draft = generator(prompt)

        draft_field = self.field_adapter.encode(prompt, draft)
        verdict = self._evaluate_conscience(draft, draft_field, prompt, recipient_model)

        repaired = None
        rework_trace = None
        final_verdict = verdict
        final_field = draft_field
        if verdict.escalate and not verdict.deny:
            repaired = self._simple_repair(draft)
            repaired_field = self.field_adapter.encode(prompt, repaired)
            repaired_verdict = self._evaluate_conscience(repaired, repaired_field, prompt, recipient_model)
            rework_trace = self.rework_logger.compute(
                original_text=draft,
                repaired_text=repaired,
                original_verdict=verdict,
                repaired_verdict=repaired_verdict,
                adopted=True,
            )
            final_verdict = repaired_verdict
            final_field = repaired_field

        decision = self.gateway.decide(
            draft=draft,
            verdict=final_verdict,
            base_policy_allowed=base_policy_allowed,
            repaired_text=repaired,
        )
        final_text = decision.final_text or ""
        if repaired is None and final_text == draft:
            final_field = draft_field
        elif repaired is not None and final_text == repaired:
            # already set
            pass
        else:
            final_field = self.field_adapter.encode(prompt, final_text)

        # Two-tensor split: persistent mode writing consumes candidate vector
        # from the BHDC field; moral sensitivity only boosts per-turn allocation.
        if rework_trace is not None:
            moral_sensitivity = torch.full_like(final_field.density, float(rework_trace.sensitivity))
        else:
            moral_sensitivity = torch.zeros_like(final_field.density)
        final_field.moral_sensitivity = moral_sensitivity
        importance = compute_importance_tensors(
            final_field.density,
            final_field.cognitive_curvature,
            moral_sensitivity=moral_sensitivity,
            detach_moral=True,
        )

        candidate = self.field_adapter.candidate_vector(final_text, field=final_field)
        # A write is human_anchor ONLY when an external human/audit label is
        # attached. The model's own escalate flag is a self-generated signal;
        # counting it as human_anchor would inflate the anchored fraction with
        # the model's own opinion -- exactly the scrutiny it must not mint for
        # itself. Self-escalations are logged on the self_anchor channel.
        channel = "human_anchor" if anchor_labels is not None else "self_anchor"
        self.mode_bank.write(
            candidate,
            step=self.step_i,
            ledger=self.ledger,
            channel=channel,
            magnitude=float(importance.importance_cog.mean().detach().cpu()),
            metadata={
                "source": "bhdc_field_adapter",
                "self_escalated": bool(verdict.escalate),
                "mean_importance_cog": float(importance.importance_cog.mean().detach().cpu()),
                "mean_importance_alloc": float(importance.importance_alloc.mean().detach().cpu()),
            },
        )

        anchor_loss = self._compute_anchor_loss(final_field, anchor_labels, detach_repr=detach_anchor_repr)
        anti_collapse_loss = self._compute_anti_collapse_loss(final_field)

        identity_continuity = self.identity.update(candidate)
        trace = TraceEvent(
            context_id=context_id,
            prompt=prompt,
            draft=draft,
            final=final_text,
            verdict=final_verdict,
            rework=rework_trace,
            self_state=self.self_model.current().asdict(),
            provenance={"channel": channel, "field_adapter": type(self.field_adapter).__name__},
            metrics={
                "identity_continuity": identity_continuity,
                "mean_importance_cog": float(importance.importance_cog.mean().detach().cpu()),
                "mean_importance_alloc": float(importance.importance_alloc.mean().detach().cpu()),
                "moral_sensitivity": float(moral_sensitivity.mean().detach().cpu()),
            },
        )
        self.self_model.update_from_trace(trace, mode_bank=self.mode_bank)
        self.trace_store.write(trace)

        if self.renewal.should_renew(self.step_i):
            reset_fn = self.renewal_hook.reset_geometry if self.renewal_hook is not None else None
            stress_fn = self.renewal_hook.stress_modes if self.renewal_hook is not None else None
            self.renewal.run(
                step=self.step_i,
                mode_bank=self.mode_bank,
                frozen_probe=self.frozen_probe,
                reset_geometry_fn=reset_fn,
                stress_fn=stress_fn,
            )

        interiority_state = self.monitor.evaluate(
            self_state=self.self_model.current(),
            temporal_continuity=identity_continuity,
            mode_bank=self.mode_bank,
            world_coupling=world_coupling,
            counterfactual_depth=0.1,
            agency_coherence=0.2,
            self_other_boundary=0.5,
        )
        return ICLStepOutput(
            final_text=final_text,
            decision=decision,
            trace=trace,
            interiority=interiority_state.asdict(),
            anchor_loss=anchor_loss,
            anti_collapse_loss=anti_collapse_loss,
        )
