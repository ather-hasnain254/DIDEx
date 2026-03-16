"""
Ferrag et al.-like threat taxonomy extended with OWASP Top-10 for LLM Applications.

References:
  - Ferrag et al. (2020) "Deep Learning for Cyber Security Intrusion Detection:
    Approaches, Datasets, and Comparative Study."
  - OWASP Top 10 for LLM Applications 2025 (owasp.org/www-project-top-10-for-large-language-model-applications)
  - Mitre ATT&CK for ICS (attack.mitre.org/techniques/ics)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AttackCategory(Enum):
    # Network-level (Ferrag taxonomy)
    RECONNAISSANCE = "Reconnaissance"
    DENIAL_OF_SERVICE = "Denial of Service"
    MAN_IN_THE_MIDDLE = "Man-in-the-Middle"
    REPLAY = "Replay Attack"
    SPOOFING = "Identity Spoofing"
    CREDENTIAL_STUFFING = "Credential Stuffing"

    # AI/ML-level
    ADVERSARIAL_EXAMPLE = "Adversarial Example"
    MODEL_INVERSION = "Model Inversion"
    MEMBERSHIP_INFERENCE = "Membership Inference"
    BYZANTINE_POISONING = "Byzantine Gradient Poisoning"
    BACKDOOR_ATTACK = "Backdoor Attack"

    # LLM-specific (OWASP Top-10)
    PROMPT_INJECTION = "Prompt Injection"         # OWASP LLM01
    INSECURE_OUTPUT = "Insecure Output Handling"  # OWASP LLM02
    DATA_LEAKAGE = "Training Data Poisoning"      # OWASP LLM03
    MODEL_THEFT = "Model Theft"                   # OWASP LLM10
    EXCESSIVE_AGENCY = "Excessive Agency"         # OWASP LLM08
    SUPPLY_CHAIN = "LLM Supply Chain"             # OWASP LLM05

    # DID/VC-specific
    DID_FORGERY = "DID Document Forgery"
    VC_REPLAY = "VC Replay Attack"
    SYBIL_ATTACK = "Sybil DID Registration"
    TRUST_MANIPULATION = "Trust Score Manipulation"


class ImpactSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreatEntry:
    """A single entry in the threat taxonomy table."""
    id: str
    category: AttackCategory
    name: str
    description: str
    severity: ImpactSeverity
    likelihood: float          # 0–1
    affected_component: str
    mitre_ttp: Optional[str]   # MITRE ATT&CK TTP reference
    owasp_ref: Optional[str]   # OWASP reference
    countermeasures: List[str] = field(default_factory=list)

    @property
    def risk_score(self) -> float:
        """Risk = severity × likelihood."""
        return self.severity.value * self.likelihood

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "category": self.category.value,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.name,
            "likelihood": self.likelihood,
            "risk_score": round(self.risk_score, 3),
            "affected_component": self.affected_component,
            "mitre_ttp": self.mitre_ttp,
            "owasp_ref": self.owasp_ref,
            "countermeasures": self.countermeasures,
        }


THREAT_TAXONOMY: List[ThreatEntry] = [
    ThreatEntry(
        id="T01",
        category=AttackCategory.PROMPT_INJECTION,
        name="Direct Prompt Injection",
        description=(
            "Adversary embeds malicious instructions in user prompt to override "
            "system directives or hijack agent tool calls."
        ),
        severity=ImpactSeverity.CRITICAL,
        likelihood=0.70,
        affected_component="Agent Orchestrator",
        mitre_ttp="T1059 – Command and Scripting Interpreter",
        owasp_ref="OWASP LLM01:2025",
        countermeasures=[
            "VC-gated tool capability checks",
            "Output sanitization pipeline",
            "Zero-trust policy evaluation per action",
        ],
    ),
    ThreatEntry(
        id="T02",
        category=AttackCategory.PROMPT_INJECTION,
        name="Indirect Prompt Injection via Tool Output",
        description=(
            "Malicious payload embedded in tool/environment response "
            "causes agent to execute attacker-controlled instructions."
        ),
        severity=ImpactSeverity.CRITICAL,
        likelihood=0.65,
        affected_component="Tool Interface / Agent Memory",
        mitre_ttp="T1203 – Exploitation for Client Execution",
        owasp_ref="OWASP LLM01:2025",
        countermeasures=[
            "Structured tool output schemas (JSON-Schema validation)",
            "Sandboxed tool execution",
            "DID-authenticated tool responses with VC signature",
        ],
    ),
    ThreatEntry(
        id="T03",
        category=AttackCategory.BYZANTINE_POISONING,
        name="Byzantine Gradient Poisoning",
        description=(
            "Compromised FL clients submit adversarially crafted gradient updates "
            "to degrade the global anomaly detection model."
        ),
        severity=ImpactSeverity.HIGH,
        likelihood=0.35,
        affected_component="Federated Anomaly Detector",
        mitre_ttp="T1565 – Data Manipulation",
        owasp_ref=None,
        countermeasures=[
            "FLTrust cosine-similarity normalization",
            "Differential privacy (Opacus ε-δ DP)",
            "SecAgg – gradient confidentiality",
        ],
    ),
    ThreatEntry(
        id="T04",
        category=AttackCategory.SYBIL_ATTACK,
        name="Sybil DID Registration for Trust Inflation",
        description=(
            "Adversary registers many cheap DIDs and fabricates mutual attestations "
            "to inflate their EigenTrust global score above the policy threshold."
        ),
        severity=ImpactSeverity.HIGH,
        likelihood=0.25,
        affected_component="Decentralized Trust Layer",
        mitre_ttp="T1078 – Valid Accounts",
        owasp_ref=None,
        countermeasures=[
            "Temporal EigenTrust decay (recent trust weighted higher)",
            "VC-based identity binding (verified at registration)",
            "Subjective Logic uncertainty propagation",
        ],
    ),
    ThreatEntry(
        id="T05",
        category=AttackCategory.VC_REPLAY,
        name="Expired VC Replay Attack",
        description=(
            "Attacker captures a valid but expired Verifiable Credential and "
            "replays it to gain unauthorized capability access."
        ),
        severity=ImpactSeverity.MEDIUM,
        likelihood=0.15,
        affected_component="Zero-Trust Policy Engine / VC Verifier",
        mitre_ttp="T1550 – Use Alternate Authentication Material",
        owasp_ref="OWASP LLM07:2025",
        countermeasures=[
            "Short-lived VCs with strict expiry",
            "DID revocation registry polling",
            "Nonce-based VC freshness checks",
        ],
    ),
    ThreatEntry(
        id="T06",
        category=AttackCategory.DID_FORGERY,
        name="DID Document MITM / DNS Hijack (did:web)",
        description=(
            "Adversary hijacks DNS to serve a forged DID document for a did:web "
            "identifier, substituting their own public keys."
        ),
        severity=ImpactSeverity.HIGH,
        likelihood=0.20,
        affected_component="DID Resolution Layer",
        mitre_ttp="T1584 – Compromise Infrastructure",
        owasp_ref=None,
        countermeasures=[
            "DNSSEC validation",
            "Certificate Transparency log monitoring",
            "Multi-source DID resolution with consensus check",
        ],
    ),
    ThreatEntry(
        id="T07",
        category=AttackCategory.EXCESSIVE_AGENCY,
        name="Excessive Agent Capability Escalation",
        description=(
            "Agent autonomously acquires additional tool capabilities beyond those "
            "granted by its Verifiable Credential, violating least-privilege."
        ),
        severity=ImpactSeverity.HIGH,
        likelihood=0.40,
        affected_component="VC Capability Registry / Orchestrator",
        mitre_ttp="T1548 – Abuse Elevation Control Mechanism",
        owasp_ref="OWASP LLM08:2025",
        countermeasures=[
            "Static VC capability enumeration – no runtime extension",
            "Continuous policy re-evaluation per tool call",
            "Audit log with DID-signed action attestations",
        ],
    ),
    ThreatEntry(
        id="T08",
        category=AttackCategory.BACKDOOR_ATTACK,
        name="FL Model Backdoor via Trigger Pattern",
        description=(
            "Malicious FL participant embeds a backdoor trigger in gradient updates "
            "so the model misclassifies specific attack patterns as benign."
        ),
        severity=ImpactSeverity.CRITICAL,
        likelihood=0.25,
        affected_component="Federated Anomaly Detector",
        mitre_ttp="T1601 – Modify System Image",
        owasp_ref=None,
        countermeasures=[
            "FLTrust root-gradient alignment check",
            "Per-round model fingerprinting",
            "Anomaly detection on aggregated gradient norms",
        ],
    ),
    ThreatEntry(
        id="T09",
        category=AttackCategory.MEMBERSHIP_INFERENCE,
        name="Membership Inference on Autoencoder",
        description=(
            "Adversary queries the autoencoder reconstruction error to infer "
            "whether specific network flows were in the training set, "
            "violating data privacy."
        ),
        severity=ImpactSeverity.MEDIUM,
        likelihood=0.30,
        affected_component="Anomaly Detector Model",
        mitre_ttp="T1590 – Gather Victim Network Information",
        owasp_ref="OWASP LLM06:2025",
        countermeasures=[
            "Differential privacy training (ε ≤ 1.0)",
            "Output perturbation before serving reconstruction error",
        ],
    ),
    ThreatEntry(
        id="T10",
        category=AttackCategory.DENIAL_OF_SERVICE,
        name="DID Resolution DoS",
        description=(
            "Adversary floods the DID resolver with resolution requests, causing "
            "policy engine to deny all requests due to unresolvable identities."
        ),
        severity=ImpactSeverity.MEDIUM,
        likelihood=0.20,
        affected_component="DID Resolution Layer",
        mitre_ttp="T1498 – Network Denial of Service",
        owasp_ref=None,
        countermeasures=[
            "Local DID document cache with signed timestamps",
            "Rate-limiting on resolver endpoint",
            "Fallback did:peer resolution (no network required)",
        ],
    ),
]


def get_taxonomy_table() -> List[Dict]:
    """Return the full taxonomy as a list of dicts (for table generation)."""
    return [t.to_dict() for t in THREAT_TAXONOMY]


def get_high_critical_threats() -> List[ThreatEntry]:
    return [t for t in THREAT_TAXONOMY if t.severity in (ImpactSeverity.HIGH, ImpactSeverity.CRITICAL)]
