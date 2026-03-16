"""
VC-gated tool capability registry.

Each tool is assigned a required capability string. Before an agent can invoke
a tool, the policy engine verifies:
  1. The agent presents a valid VC containing the required capability.
  2. The agent's trust score meets the minimum threshold.
  3. The runtime context (risk level, time, IP) passes policy rules.

Tools mirror realistic security operations:
  - Network forensics (read-only)
  - Alert management (read-write)
  - Asset quarantine (write, elevated trust required)
  - MITRE Caldera simulation (admin)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Metadata for a registered tool."""
    name: str
    description: str
    required_capability: str
    required_scope: str      # "read" | "write" | "admin"
    min_trust_score: float
    handler: Callable[..., Any]


class CapabilityRegistry:
    """Registry of tools with their required capabilities."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# Concrete tool implementations (realistic stubs for the scaffold)
# ---------------------------------------------------------------------------

def _tool_read_network_log(path: str, lines: int = 100) -> Dict:
    """Read the last N lines of a network log file."""
    logger.debug("read_network_log: path=%s lines=%d", path, lines)
    # Stub: return synthetic log lines
    entries = [
        {"ts": time.time() - i, "src": f"10.0.0.{i % 255}", "dst": "192.168.1.1",
         "proto": "TCP", "flag": "SYN", "bytes": 64 * i}
        for i in range(1, min(lines + 1, 11))
    ]
    return {"path": path, "entries": entries, "count": len(entries)}


def _tool_query_alert_db(severity: str = "HIGH", limit: int = 20) -> Dict:
    """Query the alert database for recent alerts."""
    logger.debug("query_alert_db: severity=%s limit=%d", severity, limit)
    import random
    alerts = [
        {"id": hashlib.sha256(str(i).encode()).hexdigest()[:8],
         "severity": severity,
         "type": random.choice(["port_scan", "brute_force", "exfil", "c2_beacon"]),
         "src_ip": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
         "ts": time.time() - random.randint(0, 3600)}
        for i in range(min(limit, 5))
    ]
    return {"alerts": alerts, "total": len(alerts)}


def _tool_quarantine_asset(asset_id: str, reason: str = "") -> Dict:
    """Quarantine a network asset by blocking its traffic."""
    logger.info("quarantine_asset: asset_id=%s reason=%s", asset_id, reason)
    return {
        "asset_id": asset_id,
        "action": "quarantined",
        "timestamp": time.time(),
        "reason": reason,
        "success": True,
    }


def _tool_run_caldera_sim(operation: str = "default") -> Dict:
    """Launch a MITRE Caldera red-team simulation operation."""
    logger.info("run_caldera_sim: operation=%s", operation)
    return {
        "operation": operation,
        "status": "launched",
        "job_id": hashlib.sha256(operation.encode()).hexdigest()[:12],
        "timestamp": time.time(),
    }


def _tool_get_did_reputation(did: str) -> Dict:
    """Query the decentralized trust layer for a DID's reputation."""
    logger.debug("get_did_reputation: did=%s", did)
    seed = int(hashlib.sha256(did.encode()).hexdigest(), 16) % 1000
    score = (seed % 800 + 100) / 1000.0
    return {"did": did, "trust_score": round(score, 4), "tier": "silver" if score > 0.5 else "bronze"}


def _tool_issue_vc(subject_did: str, tools: list, scope: str = "read") -> Dict:
    """Issue a capability VC for the specified DID."""
    logger.info("issue_vc: subject=%s tools=%s scope=%s", subject_did, tools, scope)
    return {
        "vc_id": f"urn:uuid:{hashlib.sha256(subject_did.encode()).hexdigest()[:32]}",
        "subject": subject_did,
        "tools": tools,
        "scope": scope,
        "issued_at": time.time(),
        "expires_at": time.time() + 3600,
    }


# ---------------------------------------------------------------------------
# Build default registry
# ---------------------------------------------------------------------------

def build_default_registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register(ToolDefinition(
        name="read_network_log",
        description="Read network flow log entries.",
        required_capability="read_network_log",
        required_scope="read",
        min_trust_score=0.3,
        handler=_tool_read_network_log,
    ))
    reg.register(ToolDefinition(
        name="query_alert_db",
        description="Query the SIEM alert database.",
        required_capability="query_alert_db",
        required_scope="read",
        min_trust_score=0.3,
        handler=_tool_query_alert_db,
    ))
    reg.register(ToolDefinition(
        name="quarantine_asset",
        description="Quarantine a network asset.",
        required_capability="quarantine_asset",
        required_scope="write",
        min_trust_score=0.6,
        handler=_tool_quarantine_asset,
    ))
    reg.register(ToolDefinition(
        name="run_caldera_sim",
        description="Launch a MITRE Caldera red-team simulation.",
        required_capability="run_caldera_sim",
        required_scope="admin",
        min_trust_score=0.8,
        handler=_tool_run_caldera_sim,
    ))
    reg.register(ToolDefinition(
        name="get_did_reputation",
        description="Query decentralized trust score for a DID.",
        required_capability="get_did_reputation",
        required_scope="read",
        min_trust_score=0.2,
        handler=_tool_get_did_reputation,
    ))
    reg.register(ToolDefinition(
        name="issue_vc",
        description="Issue a capability VC.",
        required_capability="issue_vc",
        required_scope="admin",
        min_trust_score=0.85,
        handler=_tool_issue_vc,
    ))
    return reg
