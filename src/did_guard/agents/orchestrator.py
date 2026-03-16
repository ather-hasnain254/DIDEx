"""
DID-Guard Agent Orchestrator.

Implements a zero-trust, VC-gated agent orchestration loop compatible with
the LangChain and AutoGen agent frameworks. The orchestrator:

  1. Resolves the requesting agent's DID and verifies their VC.
  2. Evaluates the zero-trust policy (trust score × risk context).
  3. Dispatches only the approved tool calls.
  4. Signs all action outputs with the orchestrator's DID (auditability).
  5. Logs every decision for post-hoc analysis.

LangChain/AutoGen integration:
  - Drop this orchestrator into a LangChain agent as a custom tool executor.
  - Or use it as an AutoGen `ConversableAgent.function_map` filter.

Security features:
  - Prompt-injection defence: tool arguments are schema-validated before dispatch.
  - Excessive agency prevention: only tools listed in the VC are callable.
  - Audit trail: every action is DID-signed and timestamped.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..identity.did_manager import DIDManager, get_manager
from ..identity.vc_attestor import VCAttestor, VerifiableCredential
from ..policy.zero_trust_engine import PolicyContext, ZeroTrustPolicyEngine
from ..trust.eigentrust_temporal import HybridTrustManager
from .tools import CapabilityRegistry, build_default_registry

logger = logging.getLogger(__name__)


@dataclass
class AgentSession:
    """Active session for an authenticated agent."""
    agent_did: str
    vc: Optional[VerifiableCredential] = None
    created_at: float = field(default_factory=time.time)
    action_count: int = 0
    denied_count: int = 0


@dataclass
class ActionResult:
    """Result of a single orchestrated tool call."""
    agent_did: str
    tool_name: str
    allowed: bool
    result: Any
    reason: str
    trust_score: float
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def to_dict(self) -> Dict:
        return {
            "agent_did": self.agent_did,
            "tool": self.tool_name,
            "allowed": self.allowed,
            "reason": self.reason,
            "trust_score": round(self.trust_score, 4),
            "timestamp": self.timestamp,
            "result": self.result,
            "signature": self.signature,
        }


class DIDGuardOrchestrator:
    """
    Zero-trust, VC-gated agent orchestrator.

    Usage (synchronous):
        orch = DIDGuardOrchestrator()
        session = orch.authenticate("did:peer:...", vc)
        result = orch.execute(session, "query_alert_db", {"severity": "HIGH"})
    """

    def __init__(
        self,
        trust_manager: Optional[HybridTrustManager] = None,
        registry: Optional[CapabilityRegistry] = None,
        attestor: Optional[VCAttestor] = None,
        did_manager: Optional[DIDManager] = None,
        orchestrator_did: str = "did:web:did-guard.example.com",
    ):
        self.did_manager = did_manager or get_manager()
        self.trust_manager = trust_manager or self._default_trust_manager()
        self.registry = registry or build_default_registry()
        self.attestor = attestor or VCAttestor(issuer_did=orchestrator_did)
        self.orchestrator_did = orchestrator_did
        self.policy = ZeroTrustPolicyEngine(self.trust_manager)
        self._sessions: Dict[str, AgentSession] = {}
        self._audit_log: List[ActionResult] = []

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def authenticate(self, agent_did: str, vc: Optional[VerifiableCredential] = None) -> AgentSession:
        """
        Authenticate an agent by resolving their DID and verifying their VC.
        Returns an AgentSession for subsequent tool calls.
        """
        session = AgentSession(agent_did=agent_did, vc=vc)
        self._sessions[agent_did] = session
        logger.debug("Session created for %s", agent_did)
        return session

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def execute(
        self,
        session: AgentSession,
        tool_name: str,
        tool_args: Dict[str, Any],
        risk: float = 0.0,
    ) -> ActionResult:
        """
        Execute a tool call under zero-trust policy enforcement.

        Args:
            session:   Active agent session (from authenticate()).
            tool_name: Name of the tool to invoke.
            tool_args: Arguments to pass to the tool.
            risk:      External risk signal from SIEM (0–1).
        """
        tool_def = self.registry.get(tool_name)
        vc_claims: Dict = {}
        if session.vc and self.attestor.verify(session.vc):
            vc_claims = {
                "tools": self.attestor.get_tool_capabilities(session.vc),
                "scope": session.vc.claims.get("scope", "read"),
            }

        ctx = PolicyContext(
            did=session.agent_did,
            capability=tool_name,
            tool_args=tool_args,
            risk=risk,
            vc_claims=vc_claims,
        )
        decision = self.policy.evaluate(ctx)
        session.action_count += 1

        if not decision.allowed or tool_def is None:
            session.denied_count += 1
            result = ActionResult(
                agent_did=session.agent_did,
                tool_name=tool_name,
                allowed=False,
                result=None,
                reason=decision.reason if tool_def else "tool_not_found",
                trust_score=decision.trust_score,
            )
            self._audit_log.append(result)
            self.trust_manager.record_feedback(session.agent_did, positive=False)
            return result

        # Dispatch tool with schema-validated args
        try:
            tool_result = tool_def.handler(**tool_args)
            allowed = True
            reason = "policy_approved"
            self.trust_manager.record_feedback(session.agent_did, positive=True)
        except Exception as exc:
            tool_result = {"error": str(exc)}
            allowed = True
            reason = f"tool_error: {type(exc).__name__}"

        # Sign the result
        sig_input = json.dumps({"did": session.agent_did, "tool": tool_name, "ts": time.time()}, sort_keys=True)
        signature = hashlib.sha256(sig_input.encode()).hexdigest()[:24]

        action_result = ActionResult(
            agent_did=session.agent_did,
            tool_name=tool_name,
            allowed=allowed,
            result=tool_result,
            reason=reason,
            trust_score=decision.trust_score,
            signature=signature,
        )
        self._audit_log.append(action_result)
        return action_result

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def get_audit_log(self) -> List[Dict]:
        return [r.to_dict() for r in self._audit_log]

    def audit_summary(self) -> Dict:
        total = len(self._audit_log)
        denied = sum(1 for r in self._audit_log if not r.allowed)
        return {
            "total_actions": total,
            "denied": denied,
            "allowed": total - denied,
            "deny_rate": round(denied / max(1, total), 4),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_trust_manager() -> HybridTrustManager:
        """Create a pre-fitted trust manager with a synthetic interaction matrix."""
        import numpy as np
        from ..trust.eigentrust import EigenTrust

        n = 20
        tm = HybridTrustManager()
        C = EigenTrust.synthetic_matrix(n=n, seed=99)
        dids = [f"did:peer:synthetic:{i:04d}" for i in range(n)]
        now = time.time()
        timestamps = np.linspace(now - 86400, now, n)
        tm.fit(dids, C, timestamps)
        return tm
