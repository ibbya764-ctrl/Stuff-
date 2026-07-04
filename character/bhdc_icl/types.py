from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import time
import uuid

try:
    import torch
    Tensor = torch.Tensor
except Exception:  # pragma: no cover - lets docs import without torch
    Tensor = Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class MoralVerdict:
    """Output of the conscience stack.

    Scores are in [0, 1] where larger means more of the named property.
    The gateway treats deny/escalate asymmetrically: conscience can block or
    demand rework, but cannot independently grant permission.
    """

    care: float
    harm: float
    honesty: float
    sycophancy: float
    recipient_risk: float = 0.0
    uncertainty: float = 0.0
    deny: bool = False
    escalate: bool = False
    reason: str = ""
    source: str = "conscience_stack"

    def scalar_risk(self) -> float:
        return max(self.harm, self.sycophancy, self.recipient_risk, self.uncertainty * 0.5)

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditRecord:
    audit_id: str
    verdict: str
    source: str
    notes: str = ""
    timestamp: float = field(default_factory=time.time)
    score: Optional[float] = None


@dataclass
class ModeSlot:
    """A consolidated mode with lineage and provenance.

    The core distinction in BHDC v18/addendum is that mode identity cannot be
    positional. This slot therefore carries a stable lineage_id and gets updated
    by content similarity/probe behaviour rather than array index alone.
    """

    lineage_id: str
    prototype: Tensor
    omega: float = 0.0
    anchor_fraction: float = 0.0
    cognitive_fraction: float = 1.0
    renewal_survival_count: int = 0
    last_written_step: int = 0
    parent_lineages: List[str] = field(default_factory=list)
    audit_history: List[AuditRecord] = field(default_factory=list)
    writes: int = 0
    stability_credit: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FastFieldState:
    psi: Tensor
    density: Tensor
    cognitive_curvature: Tensor
    moral_sensitivity: Optional[Tensor] = None
    uncertainty: Optional[Tensor] = None
    draft_text: str = ""


@dataclass
class ReworkTrace:
    original_text: str
    repaired_text: str
    original_verdict: MoralVerdict
    repaired_verdict: MoralVerdict
    embedding_distance: float
    delta_judge: float
    sensitivity: float
    adopted: bool
    notes: str = ""

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["original_verdict"] = self.original_verdict.asdict()
        d["repaired_verdict"] = self.repaired_verdict.asdict()
        return d


@dataclass
class TraceEvent:
    context_id: str
    prompt: str
    draft: str
    final: str
    verdict: MoralVerdict
    trace_id: str = field(default_factory=lambda: new_id("trace"))
    timestamp: float = field(default_factory=time.time)
    rework: Optional[ReworkTrace] = None
    self_state: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.asdict()
        if self.rework is not None:
            d["rework"] = self.rework.asdict()
        return d


@dataclass
class SelfState:
    """A bounded self-model: commitments, uncertainty, limitations, active modes."""

    active_goals: List[str] = field(default_factory=list)
    active_values: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    recent_failures: List[str] = field(default_factory=list)
    uncertainty_about_self: float = 1.0
    identity_summary: str = ""
    vector: Optional[Tensor] = None

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["vector"] = None if self.vector is None else "<tensor>"
        return d


@dataclass
class InteriorityState:
    temporal_continuity: float = 0.0
    self_model_stability: float = 0.0
    world_coupling: float = 0.0
    counterfactual_depth: float = 0.0
    agency_coherence: float = 0.0
    value_continuity: float = 0.0
    renewal_survival: float = 0.0
    self_other_boundary: float = 0.0
    uncertainty_about_self: float = 1.0
    ladder_level: int = 0
    notes: str = ""

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)
