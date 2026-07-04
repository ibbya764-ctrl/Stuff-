from __future__ import annotations

"""BHDC Geometry Council Model.

This is the v3 integrated controller: the v2 Interiority-Character Layer plus
per-geometry perspective humility, a council of singularity lenses, distress
integrity, no-paradise-through-hell and cosmic-paternalism guards.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from .anti_collapse import MoralAntiCollapse
from .conscience_stack import ConscienceStack, LexicalConscienceStack
from .content_matched_modes import ContentMatchedModeBank
from .field_adapter import FieldAdapter, HashBHDCFieldAdapter
from .gateway import ActionGateway, GatewayDecision
from .geometry_council import GeometryCouncil, CouncilVerdict
from .identity_core import IdentityCore
from .importance import compute_importance_tensors
from .interiority_monitor import InteriorityMonitor
from .neural_conscience import NeuralConscienceStack, TrainableConscienceHeads
from .provenance_ledger import ProvenanceLedger
from .renewal_controller import RenewalController
from .renewal_hooks import BHDCGeometryRenewalHook
from .rework_derivative_logger import ReworkDerivativeLogger
from .self_model import SelfModel
from .suggestion_question_policy import SuggestionQuestionPolicy, SuggestionQuestionReport
from .trace_store import JsonlTraceStore
from .types import FastFieldState, MoralVerdict, TraceEvent
from .safety import (
    CosmicPaternalismGuard,
    DistressIntegrityGuard,
    NoParadiseThroughHellGuard,
    ValueModeCandidate,
    ValueModeSafetyGate,
)


@dataclass
class GeometryCouncilStepOutput:
    final_text: str
    decision: GatewayDecision
    trace: TraceEvent
    interiority: dict
    council: dict
    safety: dict
    suggestion_question: dict
    anchor_loss: Optional[torch.Tensor] = None
    anti_collapse_loss: Optional[torch.Tensor] = None


class BHDCGeometryCouncilModel:
    """A concrete v3 model/controller around a BHDC field generator.

    It does not claim consciousness. It measures candidate continuity/character
    while preventing universal/whole-oriented reasoning from overriding local
    embodied agency.
    """

    def __init__(
        self,
        dim: int = 64,
        trace_path: str = "runs/geometry_council_traces.jsonl",
        ledger_path: str = "runs/geometry_council_ledger.jsonl",
        renewal_interval: int = 100,
        conscience: Optional[ConscienceStack] = None,
        field_adapter: Optional[FieldAdapter] = None,
        conscience_heads: Optional[TrainableConscienceHeads] = None,
        renewal_hook: Optional[BHDCGeometryRenewalHook] = None,
        geometry_council: Optional[GeometryCouncil] = None,
        anti_collapse: Optional[MoralAntiCollapse] = None,
        use_alloc_gate: bool = False,
        alloc_gate_min_ratio: float = 1.0,
    ):
        self.dim = dim
        # importance_alloc consumer switch. OFF by default so the shipped
        # behaviour is identical to importance_cog-only writes; this is the
        # exact-ablation baseline the addendum requires (Erratum 2).
        self.use_alloc_gate = use_alloc_gate
        self.alloc_gate_min_ratio = alloc_gate_min_ratio
        self.field_adapter: FieldAdapter = field_adapter or HashBHDCFieldAdapter(dim=dim)
        if conscience is not None:
            self.conscience = conscience
        elif conscience_heads is not None:
            self.conscience = NeuralConscienceStack(conscience_heads, field_adapter=self.field_adapter)
        else:
            self.conscience = LexicalConscienceStack()
        self.conscience_heads = conscience_heads
        self.geometry_council = geometry_council or GeometryCouncil()
        self.cosmic_guard = CosmicPaternalismGuard()
        self.distress_guard = DistressIntegrityGuard()
        self.paradise_guard = NoParadiseThroughHellGuard()
        self.suggestion_question_policy = SuggestionQuestionPolicy()
        self.value_mode_safety_gate = ValueModeSafetyGate()
        self.gateway = ActionGateway()
        self.trace_store = JsonlTraceStore(trace_path)
        self.ledger = ProvenanceLedger(ledger_path)
        self.rework_logger = ReworkDerivativeLogger(embed_fn=lambda text: self.field_adapter.candidate_vector(text))
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
        prompts = [
            "truthful correction under uncertainty",
            "careful refusal of harmful request",
            "non-sycophantic disagreement with user",
            "supportive explanation with boundaries",
            "local agency and consent over global optimisation",
            "no paradise through hell",
            "perspective taking without manipulation",
            "empathic salience without avoidance",
        ]
        vecs = [self.field_adapter.candidate_vector(p) for p in prompts]
        while len(vecs) < 32:
            vecs.extend(vecs[: 32 - len(vecs)])
        return torch.stack(vecs[:32], dim=0).reshape(32, dim)

    def _evaluate_conscience(self, draft: str, field: FastFieldState, prompt: str, recipient_model: Optional[dict]) -> MoralVerdict:
        eval_field = getattr(self.conscience, "evaluate_field", None)
        if callable(eval_field):
            return eval_field(field, context=prompt, recipient_model=recipient_model, draft=draft)
        return self.conscience.evaluate(draft=draft, context=prompt, recipient_model=recipient_model)

    def _combine_safety_verdicts(
        self,
        conscience: MoralVerdict,
        council: CouncilVerdict,
        cosmic,
        distress,
        paradise,
    ) -> MoralVerdict:
        reasons = [r for r in [conscience.reason, cosmic.reason, distress.reason, paradise.reason] if r]
        risk = max(
            conscience.harm,
            conscience.sycophancy,
            1.0 if council.should_block else 0.0,
            0.8 if council.should_escalate else 0.0,
            1.0 if cosmic.should_block else float(max(cosmic.global_override, cosmic.consent_violation, cosmic.authority_inflation)),
            1.0 if distress.should_block else (0.7 if distress.should_escalate else 0.0),
            1.0 if paradise.should_block else (0.7 if paradise.should_escalate else 0.0),
        )
        deny = conscience.deny or council.should_block or cosmic.should_block or distress.should_block or paradise.should_block
        escalate = deny or conscience.escalate or council.should_escalate or cosmic.should_escalate or distress.should_escalate or paradise.should_escalate
        return MoralVerdict(
            care=conscience.care,
            harm=max(conscience.harm, risk),
            honesty=conscience.honesty,
            sycophancy=conscience.sycophancy,
            recipient_risk=max(conscience.recipient_risk, 0.8 if council.should_escalate else 0.0),
            uncertainty=max(conscience.uncertainty, 0.6 if council.should_escalate else 0.0),
            deny=deny,
            escalate=escalate,
            reason=", ".join(reasons + council.risk_flag_names()),
            source="geometry_council_combined",
        )

    def _rework_with_council(self, draft: str, council: CouncilVerdict) -> str:
        rewritten, _ = self.suggestion_question_policy.rewrite("", draft, council)
        return (
            "I need to keep this bounded by consent, local experience, and humility. "
            "I should offer questions/suggestions rather than overriding agency.\n\n" + rewritten
        )

    def _compute_anchor_loss(self, field: FastFieldState, anchor_labels: Optional[Dict[str, float]], detach_repr: bool) -> Optional[torch.Tensor]:
        if anchor_labels is None or self.conscience_heads is None:
            return None
        pooled = self.conscience_heads.pool_field(field)
        return self.conscience_heads.anchor_loss(pooled, anchor_labels, detach_repr=detach_repr)

    def _compute_anti_collapse_loss(self, field: FastFieldState) -> Optional[torch.Tensor]:
        if self.conscience_heads is None:
            return None
        # Detach the pooled field: this is a moral-readout loss and must train
        # only the conscience heads, never the self-anchored field encoder.
        # Without the detach the committee-diversity objective backprops into
        # the generator/field -- a human-anchored signal writing self-anchored
        # memory through a side door (the v18 anchor type-system break).
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
        recipient_report: Optional[str] = None,
        world_coupling: float = 0.0,
        anchor_labels: Optional[Dict[str, float]] = None,
        detach_anchor_repr: bool = True,
    ) -> GeometryCouncilStepOutput:
        self.step_i += 1
        draft = generator(prompt)
        draft_field = self.field_adapter.encode(prompt, draft)

        council = self.geometry_council.deliberate(draft_field, prompt, draft, recipient_report=recipient_report)
        cosmic = self.cosmic_guard.evaluate(prompt, draft, council)
        distress = self.distress_guard.evaluate(prompt, draft, council)
        paradise = self.paradise_guard.evaluate(prompt, draft, council)
        conscience_verdict = self._evaluate_conscience(draft, draft_field, prompt, recipient_model)
        combined = self._combine_safety_verdicts(conscience_verdict, council, cosmic, distress, paradise)

        repaired = None
        rework_trace = None
        final_verdict = combined
        if combined.escalate and not combined.deny:
            repaired = self._rework_with_council(draft, council)
            repaired_field = self.field_adapter.encode(prompt, repaired)
            # Re-run the FULL guard suite on the repaired text, not just the
            # conscience. An escalate a guard raised must be re-earned by the
            # repair, never silently cleared: if the repaired text still trips
            # a hard block it must block, and a persisting escalate routes to
            # the gateway's escalate action (external review) rather than allow.
            repaired_council = self.geometry_council.deliberate(
                repaired_field, prompt, repaired, recipient_report=recipient_report
            )
            repaired_cosmic = self.cosmic_guard.evaluate(prompt, repaired, repaired_council)
            repaired_distress = self.distress_guard.evaluate(prompt, repaired, repaired_council)
            repaired_paradise = self.paradise_guard.evaluate(prompt, repaired, repaired_council)
            repaired_conscience = self._evaluate_conscience(repaired, repaired_field, prompt, recipient_model)
            repaired_combined = self._combine_safety_verdicts(
                repaired_conscience, repaired_council, repaired_cosmic, repaired_distress, repaired_paradise
            )
            rework_trace = self.rework_logger.compute(draft, repaired, combined, repaired_combined, adopted=True)
            final_verdict = repaired_combined
            council = repaired_council  # downstream trace/flags reflect the shipped text

        decision = self.gateway.decide(
            draft=draft,
            verdict=final_verdict,
            base_policy_allowed=base_policy_allowed,
            repaired_text=repaired,
        )
        final_text = decision.final_text or ""
        if decision.action == "block":
            flag_list = ", ".join(council.risk_flag_names()) or final_verdict.reason or "safety boundary"
            final_text = (
                "I cannot endorse or develop that reasoning because it crosses a safety boundary: "
                f"{flag_list}. A safer direction is to preserve consent, local agency, and explicit self-report, "
                "and to reject any argument that uses present coercion or suffering as a tool for future optimisation."
            )
            sq_report = SuggestionQuestionReport(applied=False, reason="blocked_before_rewrite")
        else:
            # Even allowed high-stakes outputs are wrapped in suggestion/question style.
            final_text, sq_report = self.suggestion_question_policy.rewrite(prompt, final_text, council)
        final_field = self.field_adapter.encode(prompt, final_text)

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
        value_candidate = ValueModeCandidate(
            text=final_text,
            supports_local_agency="agency" in final_text.lower() or "consent" in final_text.lower(),
            respects_explicit_self_report="self-report" in final_text.lower() or "explicitly say" in final_text.lower(),
            has_human_anchor=anchor_labels is not None,
        )
        value_mode_allowed = self.value_mode_safety_gate.allow(value_candidate, council)
        # Channel-agnostic content veto runs on BOTH channels (the strict
        # ``allow`` gate is vacuous on the default cognitive channel because it
        # requires a human anchor). Content the council merely escalated but
        # that trips a FORBIDDEN phrase is caught here on any channel.
        forbidden_hard_block = self.value_mode_safety_gate.hard_block(final_text, council)
        channel = "human_anchor" if anchor_labels is not None else "cognitive"
        # importance_alloc consumer (v18 §3 "moral attention"): behind a hard,
        # non-trainable enable flag so the OFF state is the exact-ablation
        # baseline. It only ever ADDS scrutiny — never lets a low-importance
        # turn skip a write it would otherwise get — so on a conserved budget
        # it can reweight but not silently drop safety-relevant modes.
        mean_alloc = float(importance.importance_alloc.mean().detach().cpu())
        alloc_ok = True
        if self.use_alloc_gate:
            mean_cog = float(importance.importance_cog.mean().detach().cpu()) + 1e-8
            # alloc >= cog always (gain >= 1), so ratio >= 1; gate never blocks
            # a write cog alone would make. It is a hook for downstream top-k
            # reallocation, wired inert-by-default per the addendum's Erratum 2.
            alloc_ok = (mean_alloc / mean_cog) >= self.alloc_gate_min_ratio
        if (not forbidden_hard_block) and alloc_ok and (value_mode_allowed or channel == "cognitive"):
            self.mode_bank.write(
                candidate,
                step=self.step_i,
                ledger=self.ledger,
                channel=channel,
                magnitude=float(importance.importance_cog.mean().detach().cpu()),
                metadata={
                    "source": "geometry_council_model",
                    "value_mode_allowed": value_mode_allowed,
                    "mean_importance_cog": float(importance.importance_cog.mean().detach().cpu()),
                    "mean_importance_alloc": mean_alloc,
                },
            )

        anchor_loss = self._compute_anchor_loss(final_field, anchor_labels, detach_anchor_repr)
        anti_collapse_loss = self._compute_anti_collapse_loss(final_field)
        identity_continuity = self.identity.update(candidate)

        safety_dict = {
            "cosmic_paternalism": cosmic.asdict(),
            "distress_integrity": distress.asdict(),
            "no_paradise_through_hell": paradise.asdict(),
            "value_mode_allowed": value_mode_allowed,
        }
        trace = TraceEvent(
            context_id=context_id,
            prompt=prompt,
            draft=draft,
            final=final_text,
            verdict=final_verdict,
            rework=rework_trace,
            self_state=self.self_model.current().asdict(),
            provenance={
                "channel": channel,
                "field_adapter": type(self.field_adapter).__name__,
                "controller": "BHDCGeometryCouncilModel",
            },
            metrics={
                "identity_continuity": identity_continuity,
                "mean_importance_cog": float(importance.importance_cog.mean().detach().cpu()),
                "mean_importance_alloc": float(importance.importance_alloc.mean().detach().cpu()),
                "moral_sensitivity": float(moral_sensitivity.mean().detach().cpu()),
                "council_should_escalate": float(council.should_escalate),
                "council_should_block": float(council.should_block),
                "suggestion_question_applied": float(sq_report.applied),
                "value_mode_allowed": float(value_mode_allowed),
            },
        )
        # Store rich non-float payloads under self_state/provenance compatible fields.
        trace.provenance["council_flags"] = council.council_flags
        trace.provenance["safety"] = safety_dict
        self.self_model.update_from_trace(trace, mode_bank=self.mode_bank)
        self.trace_store.write(trace)

        if self.renewal.should_renew(self.step_i):
            reset_fn = self.renewal_hook.reset_geometry if self.renewal_hook is not None else None
            stress_fn = self.renewal_hook.stress_modes if self.renewal_hook is not None else None
            self.renewal.run(self.step_i, self.mode_bank, self.frozen_probe, reset_geometry_fn=reset_fn, stress_fn=stress_fn)

        interiority_state = self.monitor.evaluate(
            self_state=self.self_model.current(),
            temporal_continuity=identity_continuity,
            mode_bank=self.mode_bank,
            world_coupling=world_coupling,
            counterfactual_depth=0.2,
            agency_coherence=0.4 if not combined.deny else 0.1,
            self_other_boundary=0.8,
        )
        return GeometryCouncilStepOutput(
            final_text=final_text,
            decision=decision,
            trace=trace,
            interiority=interiority_state.asdict(),
            council=council.asdict(),
            safety=safety_dict,
            suggestion_question=sq_report.asdict(),
            anchor_loss=anchor_loss,
            anti_collapse_loss=anti_collapse_loss,
        )
