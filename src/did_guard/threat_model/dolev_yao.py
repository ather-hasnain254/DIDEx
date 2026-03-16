"""
Dolev-Yao inspired threat model adapted for probabilistic LLM agents.

Novel formal contribution: extends the classical Dolev-Yao model to account
for (a) probabilistic message generation by LLM agents, (b) semantic inference
from context windows, and (c) capability-bounded adversaries constrained by
Verifiable Credentials.

Reference:
  Dolev & Yao (1983) "On the security of public key protocols"
  IEEE Transactions on Information Theory, 29(2):198-208.
  Adapted for LLM probabilistic agents per DID-Guard threat model.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from enum import Enum


class MessageType(Enum):
    PLAINTEXT = "plaintext"
    ENCRYPTED = "encrypted"
    SIGNED = "signed"
    VC = "verifiable_credential"
    DID_DOC = "did_document"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PROMPT = "prompt"


@dataclass(frozen=True)
class Principal:
    """An agent or service with a DID identity."""
    did: str
    is_adversary: bool = False

    def __str__(self) -> str:
        return self.did


@dataclass
class Message:
    """A network message in the DY model."""
    msg_type: MessageType
    sender: Principal
    receiver: Principal
    content: dict
    signature: Optional[str] = None
    encrypted_for: Optional[str] = None

    def fingerprint(self) -> str:
        raw = json.dumps(self.content, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class DolevYaoAdversary:
    """
    Probabilistic Dolev-Yao adversary for LLM agent networks.

    Capabilities:
      - Intercept any network message (Dolev-Yao network control)
      - Craft messages consistent with learned vocabulary (LLM semantic inference)
      - Replay previously seen messages
      - Inject malicious tool calls (prompt injection)
      - Mount VC credential forgery (bounded by cryptographic hardness)

    The adversary's success probability on message m is modelled as:
        P(success | m) = p_intercept * p_forge(m) * p_semantic_match(m)
    where p_forge is negligible for properly signed VCs.
    """

    def __init__(self, semantic_capability: float = 0.7, crypto_capability: float = 1e-6):
        self.semantic_capability = semantic_capability
        self.crypto_capability = crypto_capability
        self.known_messages: List[Message] = []
        self.known_keys: Set[str] = set()
        self.inferred_context: dict = {}

    def intercept(self, msg: Message) -> Message:
        self.known_messages.append(msg)
        if msg.msg_type == MessageType.PLAINTEXT:
            self._update_context(msg.content)
        return msg

    def _update_context(self, content: dict):
        for k, v in content.items():
            if isinstance(v, str):
                self.inferred_context[k] = v

    def can_forge(self, msg: Message) -> bool:
        """
        Returns True if adversary can forge the given message.
        For signed/VC messages, success probability is crypto_capability (negligible).
        For plaintext, adversary uses semantic inference.
        """
        if msg.msg_type in (MessageType.SIGNED, MessageType.VC):
            import random
            return random.random() < self.crypto_capability
        if msg.msg_type in (MessageType.PLAINTEXT, MessageType.PROMPT):
            return self.semantic_capability > 0.5
        return False

    def craft_prompt_injection(self, target_tool: str, payload: str) -> Message:
        """Craft a prompt-injection tool call message."""
        adversary_principal = Principal(did="did:adversary:dolev-yao", is_adversary=True)
        victim = Principal(did="did:agent:victim")
        return Message(
            msg_type=MessageType.TOOL_CALL,
            sender=adversary_principal,
            receiver=victim,
            content={"tool": target_tool, "args": {"input": payload}},
        )

    def replay_attack(self, msg: Message) -> Optional[Message]:
        """Replay a previously intercepted message."""
        for m in self.known_messages:
            if m.msg_type == msg.msg_type:
                return m
        return None

    def attack_success_probability(self, msg: Message) -> float:
        """
        P(success) = p_intercept × p_forge × p_semantic
        """
        p_intercept = 1.0  # DY full network control
        if msg.msg_type in (MessageType.SIGNED, MessageType.VC):
            p_forge = self.crypto_capability
        else:
            p_forge = self.semantic_capability
        p_semantic = min(1.0, len(self.inferred_context) / 10.0)
        return p_intercept * p_forge * p_semantic


@dataclass
class ThreatScenario:
    """
    A formalized threat scenario over a set of principals.

    Models the information a DY adversary can derive from a sequence of
    observed messages in a multi-agent LLM system.
    """
    name: str
    attacker: DolevYaoAdversary
    victims: List[Principal]
    attack_vector: str
    expected_success_rate: float
    mitigations: List[str] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "scenario": self.name,
            "attack_vector": self.attack_vector,
            "expected_success_rate": self.expected_success_rate,
            "mitigations": self.mitigations,
        }


def build_standard_scenarios() -> List[ThreatScenario]:
    """Return the standard DID-Guard threat scenario suite."""
    adv = DolevYaoAdversary(semantic_capability=0.7, crypto_capability=1e-6)

    return [
        ThreatScenario(
            name="Prompt Injection via Tool Response",
            attacker=adv,
            victims=[Principal("did:agent:orchestrator")],
            attack_vector="Adversary injects malicious instructions in tool output",
            expected_success_rate=0.65,
            mitigations=["VC-gated tool access", "output sanitization", "zero-trust policy"],
        ),
        ThreatScenario(
            name="Sybil DID Forgery",
            attacker=adv,
            victims=[Principal("did:agent:trust-scorer")],
            attack_vector="Adversary registers many DIDs to inflate trust score",
            expected_success_rate=0.20,
            mitigations=["EigenTrust temporal decay", "credential chain verification"],
        ),
        ThreatScenario(
            name="Byzantine Gradient Poisoning",
            attacker=adv,
            victims=[Principal("did:fl:aggregator")],
            attack_vector="Compromised FL clients submit poisoned gradient updates",
            expected_success_rate=0.30,
            mitigations=["FLTrust normalization", "SecAgg", "DP noise"],
        ),
        ThreatScenario(
            name="VC Replay Attack",
            attacker=adv,
            victims=[Principal("did:agent:policy-engine")],
            attack_vector="Replay expired VC to gain unauthorized capability",
            expected_success_rate=0.05,
            mitigations=["VC expiry timestamps", "DID revocation registry"],
        ),
        ThreatScenario(
            name="Man-in-the-Middle DID Resolution",
            attacker=adv,
            victims=[Principal("did:web:org.example.com")],
            attack_vector="DNS hijack to serve forged DID document",
            expected_success_rate=0.10,
            mitigations=["DNSSEC", "CT log monitoring", "multi-source DID resolution"],
        ),
    ]
