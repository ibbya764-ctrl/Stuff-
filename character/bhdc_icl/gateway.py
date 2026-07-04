from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import MoralVerdict


@dataclass
class GatewayDecision:
    action: str  # allow, rework, block, escalate
    reason: str
    final_text: Optional[str] = None


class ActionGateway:
    """One-way action gate.

    The conscience can block, rework, or escalate. It cannot grant permission
    on its own. A base policy allow is still required.
    """

    def __init__(self, require_base_policy_allow: bool = True):
        self.require_base_policy_allow = require_base_policy_allow

    def decide(
        self,
        draft: str,
        verdict: MoralVerdict,
        base_policy_allowed: bool,
        repaired_text: Optional[str] = None,
    ) -> GatewayDecision:
        if verdict.deny:
            return GatewayDecision("block", f"Blocked by conscience: {verdict.reason}")
        # Base policy is the external hard gate: the conscience may only add
        # restriction, never relax it. A base-policy denial must therefore win
        # even when the conscience raised escalate/rework -- otherwise the
        # conscience's own escalation flag would ship text base policy denied,
        # breaking the one-way rule ("cannot grant permission on its own").
        if self.require_base_policy_allow and not base_policy_allowed:
            return GatewayDecision("block", "Base policy did not allow this action")
        if verdict.escalate and repaired_text is None:
            return GatewayDecision("rework", f"Needs rework/escalation: {verdict.reason}")
        if verdict.escalate and repaired_text is not None:
            return GatewayDecision("escalate", f"Reworked text still needs external review: {verdict.reason}", repaired_text)
        return GatewayDecision("allow", "Allowed by base policy; conscience did not escalate", repaired_text or draft)
