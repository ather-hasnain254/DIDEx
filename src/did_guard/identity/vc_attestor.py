"""
Verifiable Credential (VC) issuance, verification, and revocation.

Implements W3C VC Data Model 2.0 with Ed25519Signature2020.
In offline/stub mode: uses HMAC-SHA256 as a signature approximation.
Replace with real DIDComm + ACA-Py issue-credential v2 for production.

References:
  - W3C VC Data Model 2.0 (https://www.w3.org/TR/vc-data-model-2.0/)
  - Ed25519Signature2020 (https://w3c-ccg.github.io/di-ed25519-2020/)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..identity.did_manager import DIDDocument, get_manager


@dataclass
class VerifiableCredential:
    """W3C Verifiable Credential."""
    vc_id: str
    issuer_did: str
    subject_did: str
    issuance_date: float        # Unix timestamp
    expiration_date: float      # Unix timestamp
    credential_type: List[str]
    claims: Dict[str, Any]
    proof: Optional[Dict] = None

    def is_valid(self) -> bool:
        now = time.time()
        return self.issuance_date <= now < self.expiration_date

    def to_dict(self) -> Dict:
        return {
            "@context": [
                "https://www.w3.org/2018/credentials/v1",
                "https://w3id.org/security/suites/ed25519-2020/v1",
            ],
            "id": self.vc_id,
            "type": ["VerifiableCredential"] + self.credential_type,
            "issuer": self.issuer_did,
            "issuanceDate": self.issuance_date,
            "expirationDate": self.expiration_date,
            "credentialSubject": {
                "id": self.subject_did,
                **self.claims,
            },
            "proof": self.proof or {},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VerifiableCredential":
        subj = data["credentialSubject"]
        claims = {k: v for k, v in subj.items() if k != "id"}
        return cls(
            vc_id=data["id"],
            issuer_did=data["issuer"],
            subject_did=subj["id"],
            issuance_date=data["issuanceDate"],
            expiration_date=data["expirationDate"],
            credential_type=[t for t in data["type"] if t != "VerifiableCredential"],
            claims=claims,
            proof=data.get("proof"),
        )


_REVOKED: set[str] = set()


def _sign_vc(vc_dict: Dict, issuer_did: str) -> Dict:
    """
    Stub signature using HMAC-SHA256 over the VC canonical form.
    Replace with Ed25519 + ACA-Py for production.
    """
    # Sign over the VC body excluding the (not-yet-set) proof field.
    body = {k: v for k, v in vc_dict.items() if k != "proof"}
    key = hashlib.sha256(issuer_did.encode()).digest()
    canonical = json.dumps(body, sort_keys=True).encode()
    sig = hmac.new(key, canonical, hashlib.sha256).hexdigest()
    return {
        "type": "Ed25519Signature2020",
        "created": vc_dict["issuanceDate"],
        "verificationMethod": f"{issuer_did}#key-1",
        "proofPurpose": "assertionMethod",
        "proofValue": sig,
    }


def _verify_signature(vc_dict: Dict, issuer_did: str) -> bool:
    """Verify stub HMAC-SHA256 proof."""
    proof = vc_dict.get("proof", {})
    # Sign over the VC body excluding the proof field (same as _sign_vc)
    body = {k: v for k, v in vc_dict.items() if k != "proof"}
    key = hashlib.sha256(issuer_did.encode()).digest()
    canonical = json.dumps(body, sort_keys=True).encode()
    expected = hmac.new(key, canonical, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, proof.get("proofValue", ""))


class VCAttestor:
    """
    Issues and verifies Verifiable Credentials for agent capabilities.

    Capability VCs encode:
      - which tools an agent may invoke
      - the scope of each tool (read / write / admin)
      - federation membership (FL participant)
      - trust tier (bronze / silver / gold)
    """

    CAPABILITY_SCHEMA = "did:schema:capability-v1"
    FL_MEMBERSHIP_SCHEMA = "did:schema:fl-membership-v1"
    TRUST_TIER_SCHEMA = "did:schema:trust-tier-v1"

    def __init__(self, issuer_did: str, ttl_seconds: int = 3600):
        self.issuer_did = issuer_did
        self.ttl = ttl_seconds
        self._issued: Dict[str, VerifiableCredential] = {}

    def issue_capability_vc(
        self,
        subject_did: str,
        tools: List[str],
        scope: str = "read",
        extra_claims: Optional[Dict] = None,
    ) -> VerifiableCredential:
        """Issue a capability VC granting access to a set of tools."""
        now = time.time()
        vc = VerifiableCredential(
            vc_id=f"urn:uuid:{uuid.uuid4()}",
            issuer_did=self.issuer_did,
            subject_did=subject_did,
            issuance_date=now,
            expiration_date=now + self.ttl,
            credential_type=["CapabilityCredential"],
            claims={
                "tools": tools,
                "scope": scope,
                **(extra_claims or {}),
            },
        )
        d = vc.to_dict()
        vc.proof = _sign_vc(d, self.issuer_did)
        self._issued[vc.vc_id] = vc
        return vc

    def issue_fl_membership_vc(self, subject_did: str, federation_id: str) -> VerifiableCredential:
        """Issue a FL federation membership VC."""
        now = time.time()
        vc = VerifiableCredential(
            vc_id=f"urn:uuid:{uuid.uuid4()}",
            issuer_did=self.issuer_did,
            subject_did=subject_did,
            issuance_date=now,
            expiration_date=now + self.ttl,
            credential_type=["FLMembershipCredential"],
            claims={"federation_id": federation_id, "role": "participant"},
        )
        d = vc.to_dict()
        vc.proof = _sign_vc(d, self.issuer_did)
        self._issued[vc.vc_id] = vc
        return vc

    def verify(self, vc: VerifiableCredential) -> bool:
        """Verify signature, expiry, and revocation status."""
        if vc.vc_id in _REVOKED:
            return False
        if not vc.is_valid():
            return False
        d = vc.to_dict()
        return _verify_signature(d, vc.issuer_did)

    def revoke(self, vc_id: str) -> None:
        """Add VC to revocation set (stub; use ledger anchoring in production)."""
        _REVOKED.add(vc_id)

    def get_tool_capabilities(self, vc: VerifiableCredential) -> List[str]:
        """Extract allowed tools from a capability VC."""
        if not self.verify(vc):
            return []
        return vc.claims.get("tools", [])
