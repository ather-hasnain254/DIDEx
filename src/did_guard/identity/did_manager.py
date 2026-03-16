"""
ACA-Py admin REST client with stub fallback for offline/synthetic operation.

Supports:
  - did:peer (pairwise, no ledger required)
  - did:web  (organizational, served from HTTPS)
  - did:key  (ephemeral, derivable from public key)

Reference: ACA-Py documentation https://aca-py.org
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

from ..config import ACAPY_ADMIN_URL


@dataclass
class DIDDocument:
    """Minimal DID Document representation (W3C DID Core 1.0)."""
    did: str
    verification_methods: List[Dict[str, Any]] = field(default_factory=list)
    authentication: List[str] = field(default_factory=list)
    services: List[Dict[str, Any]] = field(default_factory=list)
    created: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": self.did,
            "verificationMethod": self.verification_methods,
            "authentication": self.authentication,
            "service": self.services,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DIDDocument":
        return cls(
            did=data["id"],
            verification_methods=data.get("verificationMethod", []),
            authentication=data.get("authentication", []),
            services=data.get("service", []),
        )


def _stub_peer_did(label: str) -> str:
    """Generate a deterministic stub did:peer for offline use."""
    seed = hashlib.sha256(label.encode()).hexdigest()[:32]
    return f"did:peer:2.Ez{seed[:22]}.Vz{seed[22:44]}"


def _stub_web_did(domain: str, path: str = "") -> str:
    """Generate a did:web identifier."""
    encoded = domain.replace(":", "%3A")
    if path:
        encoded += ":" + path.replace("/", ":")
    return f"did:web:{encoded}"


def _stub_key_did(label: str) -> str:
    """Generate a deterministic stub did:key for offline use."""
    seed = hashlib.sha256(f"key:{label}".encode()).hexdigest()
    return f"did:key:z6Mk{seed[:43]}"


def _make_verification_method(did: str, key_id: str, public_key_hex: str) -> Dict:
    return {
        "id": f"{did}#{key_id}",
        "type": "Ed25519VerificationKey2020",
        "controller": did,
        "publicKeyMultibase": f"z{public_key_hex[:43]}",
    }


def _make_service(did: str, endpoint: str) -> Dict:
    return {
        "id": f"{did}#agent",
        "type": "DIDCommMessaging",
        "serviceEndpoint": endpoint,
    }


class DIDManager:
    """
    Manages DID lifecycle: create, resolve, update, deactivate.
    Falls back to deterministic stubs when ACA-Py is not reachable.
    """

    def __init__(self, admin_url: str = ACAPY_ADMIN_URL):
        self.admin_url = admin_url
        self._registry: Dict[str, DIDDocument] = {}

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_peer_did(self, label: str) -> DIDDocument:
        """Create a pairwise did:peer DID (no ledger registration required)."""
        try:
            return self._acapy_create_peer_did(label)
        except Exception:
            return self._stub_create_peer_did(label)

    def create_web_did(self, domain: str, path: str = "") -> DIDDocument:
        """Create an organizational did:web DID."""
        did = _stub_web_did(domain, path)
        seed = hashlib.sha256(did.encode()).hexdigest()
        vm = _make_verification_method(did, "key-1", seed)
        doc = DIDDocument(
            did=did,
            verification_methods=[vm],
            authentication=[f"{did}#key-1"],
            services=[_make_service(did, f"https://{domain}/didcomm")],
        )
        self._registry[did] = doc
        return doc

    def create_key_did(self, label: str) -> DIDDocument:
        """Create an ephemeral did:key DID."""
        did = _stub_key_did(label)
        seed = hashlib.sha256(did.encode()).hexdigest()
        vm = _make_verification_method(did, "1", seed)
        doc = DIDDocument(
            did=did,
            verification_methods=[vm],
            authentication=[f"{did}#1"],
        )
        self._registry[did] = doc
        return doc

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, did: str) -> Optional[DIDDocument]:
        if did in self._registry:
            return self._registry[did]
        if did.startswith("did:peer:"):
            return self._resolve_peer(did)
        if did.startswith("did:web:"):
            return self._resolve_web(did)
        return None

    def _resolve_peer(self, did: str) -> DIDDocument:
        seed = hashlib.sha256(did.encode()).hexdigest()
        vm = _make_verification_method(did, "key-1", seed)
        doc = DIDDocument(
            did=did,
            verification_methods=[vm],
            authentication=[f"{did}#key-1"],
        )
        self._registry[did] = doc
        return doc

    def _resolve_web(self, did: str) -> DIDDocument:
        domain = did.replace("did:web:", "").replace("%3A", ":").replace(":", "/")
        seed = hashlib.sha256(did.encode()).hexdigest()
        vm = _make_verification_method(did, "key-1", seed)
        doc = DIDDocument(
            did=did,
            verification_methods=[vm],
            authentication=[f"{did}#key-1"],
            services=[_make_service(did, f"https://{domain}/didcomm")],
        )
        self._registry[did] = doc
        return doc

    # ------------------------------------------------------------------
    # ACA-Py REST stubs
    # ------------------------------------------------------------------

    def _acapy_create_peer_did(self, label: str) -> DIDDocument:
        if not _HTTPX_AVAILABLE:
            raise RuntimeError("httpx not available")
        resp = httpx.post(
            f"{self.admin_url}/wallet/did/create",
            json={"method": "peer", "options": {"key_type": "ed25519"}},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()["result"]
        did = data["did"]
        vm = _make_verification_method(did, "1", data.get("verkey", "0" * 64))
        doc = DIDDocument(
            did=did,
            verification_methods=[vm],
            authentication=[f"{did}#1"],
        )
        self._registry[did] = doc
        return doc

    def _stub_create_peer_did(self, label: str) -> DIDDocument:
        uid = str(uuid.uuid4()).replace("-", "")[:16]
        did = _stub_peer_did(f"{label}-{uid}")
        seed = hashlib.sha256(did.encode()).hexdigest()
        vm = _make_verification_method(did, "key-1", seed)
        doc = DIDDocument(
            did=did,
            verification_methods=[vm],
            authentication=[f"{did}#key-1"],
        )
        self._registry[did] = doc
        return doc


# Module-level singleton
_manager: Optional[DIDManager] = None


def get_manager() -> DIDManager:
    global _manager
    if _manager is None:
        _manager = DIDManager()
    return _manager
