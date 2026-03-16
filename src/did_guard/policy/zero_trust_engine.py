"""
Zero-Trust Policy Engine implementing NIST SP 800-207.

Core principles (NIST SP 800-207):
  1. All data sources and computing services are considered resources.
  2. All communication is secured regardless of network location.
  3. Access to individual enterprise resources is granted on a per-session basis.
  4. Access to resources is determined by dynamic policy.
  5. The enterprise monitors and measures the integrity and security posture
     of all owned and associated assets.
  6. All resource authentication and authorization is dynamic and strictly
     enforced before access is allowed.
  7. The enterprise collects as much information as possible about the current
     state of assets, network infrastructure and communications.

Reference: NIST SP 800-207 (2020)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class TrustSource(Protocol):
    """Protocol for any trust scoring backend."""

    def get_score(self, did: str) -> float:
        """Return a trust score in [0, 1]."""
        ...


@dataclass
class PolicyContext:
    """
    Runtime context for a single access decision.

    Attributes:
        did:         Requesting agent DID
        capability:  Requested operation / tool name
        tool_args:   Arguments to the requested tool call
        timestamp:   Unix timestamp of the request
        risk:        External risk signal in [0, 1] (e.g., from SIEM)
        ip:          Source IP (optional, for logging)
        vc_claims:   Claims extracted from the presented VC
    """
    did: str
    capability: str
    tool_args: Dict[str, Any]
    timestamp: float = 0.0
    risk: float = 0.0
    ip: Optional[str] = None
    vc_claims: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.vc_claims is None:
            self.vc_claims = {}


@dataclass
class PolicyDecision:
    """Result of a policy evaluation."""
    allowed: bool
    reason: str
    trust_score: float
    risk_score: float
    latency_ms: float = 0.0

    def __bool__(self) -> bool:
        return self.allowed


PolicyRule = Callable[[PolicyContext, float], Optional[bool]]
"""
A PolicyRule receives (context, trust_score) and returns:
  True  – explicitly allow
  False – explicitly deny
  None  – abstain (next rule evaluated)
"""


def _rule_vc_required(ctx: PolicyContext, trust: float) -> Optional[bool]:
    """Deny if no VC claims are present (identity not attested)."""
    if not ctx.vc_claims:
        return False
    return None


def _rule_capability_check(ctx: PolicyContext, trust: float) -> Optional[bool]:
    """Deny if requested capability is not in the VC's tool list."""
    allowed_tools = ctx.vc_claims.get("tools", [])
    if allowed_tools and ctx.capability not in allowed_tools:
        return False
    return None


def _rule_trust_threshold(ctx: PolicyContext, trust: float) -> Optional[bool]:
    """Deny if trust score is below minimum threshold."""
    from ..config import POLICY_TRUST_THRESHOLD
    if trust < POLICY_TRUST_THRESHOLD:
        return False
    return None


def _rule_risk_adjusted(ctx: PolicyContext, trust: float) -> Optional[bool]:
    """Allow if adjusted score (trust × (1-risk)) passes threshold."""
    from ..config import POLICY_TRUST_THRESHOLD
    adjusted = trust * (1.0 - ctx.risk)
    return adjusted > POLICY_TRUST_THRESHOLD


DEFAULT_RULES: List[PolicyRule] = [
    _rule_vc_required,
    _rule_capability_check,
    _rule_trust_threshold,
    _rule_risk_adjusted,
]


class ZeroTrustPolicyEngine:
    """
    Evaluates access requests using a chain-of-responsibility rule set.

    Decision logic:
      1. Resolve trust score for the requesting DID.
      2. Apply each rule in order; first explicit True/False wins.
      3. If all rules abstain, default-deny.
    """

    def __init__(
        self,
        trust_source: TrustSource,
        rules: Optional[List[PolicyRule]] = None,
        audit_log: bool = True,
    ):
        self.trust_source = trust_source
        self.rules = rules or DEFAULT_RULES
        self.audit_log = audit_log
        self._decisions: List[Dict] = []

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        t0 = time.monotonic()
        trust = self.trust_source.get_score(ctx.did)

        allowed = False
        reason = "default-deny"
        for rule in self.rules:
            result = rule(ctx, trust)
            if result is True:
                allowed = True
                reason = rule.__name__
                break
            if result is False:
                allowed = False
                reason = rule.__name__
                break

        latency_ms = (time.monotonic() - t0) * 1000
        decision = PolicyDecision(
            allowed=allowed,
            reason=reason,
            trust_score=trust,
            risk_score=ctx.risk,
            latency_ms=latency_ms,
        )

        if self.audit_log:
            self._log(ctx, decision)

        return decision

    def _log(self, ctx: PolicyContext, decision: PolicyDecision):
        entry = {
            "ts": ctx.timestamp,
            "did": ctx.did,
            "capability": ctx.capability,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "trust": round(decision.trust_score, 4),
            "risk": round(decision.risk_score, 4),
            "latency_ms": round(decision.latency_ms, 3),
        }
        self._decisions.append(entry)
        logger.debug("Policy decision: %s", entry)

    def get_audit_log(self) -> List[Dict]:
        return list(self._decisions)

    def clear_audit_log(self):
        self._decisions.clear()
