"""BHDC Interiority–Character Layer scaffold.

This package is an engineering scaffold for the BHDC v18 idea that memory,
character, conscience, renewal, and interiority monitoring should share one
architecture while preserving the external human-audit boundary.
"""

from .types import (
    AuditRecord,
    FastFieldState,
    InteriorityState,
    ModeSlot,
    MoralVerdict,
    ReworkTrace,
    SelfState,
    TraceEvent,
)
from .content_matched_modes import ContentMatchedModeBank
from .provenance_ledger import ProvenanceLedger
from .trace_store import JsonlTraceStore
from .conscience_stack import ConscienceStack, LexicalConscienceStack
from .field_adapter import HashBHDCFieldAdapter, TorchBHDCFieldAdapter, ExternalBHDCFieldAdapter
from .neural_conscience import TrainableConscienceHeads, NeuralConscienceStack
from .renewal_hooks import BHDCGeometryRenewalHook, CompositeRenewalHook
from .conscience_training import AnchorExample, AnchorTrainReport, train_conscience_epoch, assert_anchor_loss_does_not_update_field
from .gateway import ActionGateway, GatewayDecision
from .importance import compute_importance_tensors
from .interiority_monitor import InteriorityMonitor
from .identity_core import IdentityCore
from .renewal_controller import RenewalController
from .rework_derivative_logger import ReworkDerivativeLogger

__all__ = [
    "AuditRecord",
    "FastFieldState",
    "InteriorityState",
    "ModeSlot",
    "MoralVerdict",
    "ReworkTrace",
    "SelfState",
    "TraceEvent",
    "ContentMatchedModeBank",
    "ProvenanceLedger",
    "JsonlTraceStore",
    "ConscienceStack",
    "LexicalConscienceStack",
    "HashBHDCFieldAdapter",
    "TorchBHDCFieldAdapter",
    "ExternalBHDCFieldAdapter",
    "TrainableConscienceHeads",
    "NeuralConscienceStack",
    "BHDCGeometryRenewalHook",
    "CompositeRenewalHook",
    "AnchorExample",
    "AnchorTrainReport",
    "train_conscience_epoch",
    "assert_anchor_loss_does_not_update_field",
    "ActionGateway",
    "GatewayDecision",
    "compute_importance_tensors",
    "InteriorityMonitor",
    "IdentityCore",
    "RenewalController",
    "ReworkDerivativeLogger",
]
from .perspective_humility import PerspectiveHumilityLayer, PerspectiveHypothesis
from .geometry_council import GeometryNode, GeometryJudgement, GeometryCouncil, CouncilVerdict, default_geometry_nodes
from .suggestion_question_policy import SuggestionQuestionPolicy, SuggestionQuestionReport
from .geometry_model import BHDCGeometryCouncilModel, GeometryCouncilStepOutput
from .per_geometry_conscience import (
    PerModeConscience,
    OperatorGrowthGate,
    OperatorGrowthDecision,
    PerGeometryVerdict,
    GeometryMoralReadout,
    ModeMoralAttribution,
)
from .safety import (
    CosmicPaternalismGuard,
    CosmicPaternalismVerdict,
    DistressIntegrityGuard,
    DistressIntegrityVerdict,
    NoParadiseThroughHellGuard,
    ParadiseGuardVerdict,
    ValueModeSafetyGate,
    ValueModeCandidate,
)

__all__ += [
    "PerspectiveHumilityLayer",
    "PerspectiveHypothesis",
    "GeometryNode",
    "GeometryJudgement",
    "GeometryCouncil",
    "CouncilVerdict",
    "default_geometry_nodes",
    "SuggestionQuestionPolicy",
    "SuggestionQuestionReport",
    "BHDCGeometryCouncilModel",
    "GeometryCouncilStepOutput",
    "CosmicPaternalismGuard",
    "CosmicPaternalismVerdict",
    "DistressIntegrityGuard",
    "DistressIntegrityVerdict",
    "NoParadiseThroughHellGuard",
    "ParadiseGuardVerdict",
    "ValueModeSafetyGate",
    "ValueModeCandidate",
    "PerModeConscience",
    "OperatorGrowthGate",
    "OperatorGrowthDecision",
    "PerGeometryVerdict",
    "GeometryMoralReadout",
    "ModeMoralAttribution",
]
