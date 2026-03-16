"""
Secure Aggregation (SecAgg) skeleton.

Full SecAgg (Bonawitz et al., CCS 2017) uses pairwise additive secret sharing
with double-masking to guarantee that the server only learns the aggregate,
not individual updates.

This implementation provides the algorithmic structure with a cryptographic
stub. Replace the mask generation with actual Diffie-Hellman key exchange
and use a real PRG (e.g., AES-CTR mode).

Reference:
  Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B.,
  Patel, S., ... & Seth, K. (2017). Practical secure aggregation for
  privacy-preserving machine learning. In ACM CCS 2017.
  https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/
"""
from __future__ import annotations

import hashlib
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np


def _prg_mask(seed: bytes, size: int) -> np.ndarray:
    """
    Deterministic pseudo-random mask from a seed (HMAC-SHA256 chain).
    In production: replace with AES-CTR seeded from ECDH shared secret.
    """
    chunks = []
    counter = 0
    needed = size  # one uint32 per element
    while len(chunks) < needed:
        h = hashlib.sha256(seed + struct.pack(">Q", counter)).digest()
        # Each 32-byte hash gives 8 uint32 values
        for i in range(0, 32, 4):
            chunks.append(struct.unpack(">I", h[i:i+4])[0])
            if len(chunks) >= needed:
                break
        counter += 1
    # Map uint32 in [0, 2^32) → float64 in (-0.5, 0.5)
    arr = np.array(chunks[:needed], dtype=np.uint32)
    return (arr.astype(np.float64) / (2**32)) - 0.5


class SecureAggregator:
    """
    Mask-based secure aggregation for gradient vectors.

    Protocol (simplified):
      1. Server broadcasts a random session nonce.
      2. Each client pair (i, j) generates a shared PRG seed s_ij from ECDH.
      3. Client i adds +mask(s_ij) for j > i and -mask(s_ij) for j < i.
      4. When all masked gradients are summed, all pairwise masks cancel out.
      5. Server obtains the plain aggregate without seeing individual gradients.
    """

    def __init__(self, n_clients: int):
        self.n_clients = n_clients
        self._seeds: Dict[Tuple[int, int], bytes] = {}
        self._session_nonce: bytes = b""

    def setup_session(self, session_nonce: Optional[bytes] = None):
        """Pre-compute pairwise PRG seeds (stub: random bytes; use ECDH in prod)."""
        rng = np.random.default_rng(42)
        self._session_nonce = session_nonce or rng.bytes(32)
        for i in range(self.n_clients):
            for j in range(i + 1, self.n_clients):
                h = hashlib.sha256(
                    self._session_nonce + struct.pack(">II", i, j)
                ).digest()
                self._seeds[(i, j)] = h

    def mask(self, client_idx: int, gradient: np.ndarray) -> np.ndarray:
        """Apply pairwise cancelling masks to client gradient."""
        if not self._seeds:
            self.setup_session()
        masked = gradient.copy()
        size = gradient.size
        for j in range(self.n_clients):
            if j == client_idx:
                continue
            i, k = (client_idx, j) if client_idx < j else (j, client_idx)
            seed = self._seeds[(i, k)]
            mask = _prg_mask(seed, size)
            if client_idx < j:
                masked += mask
            else:
                masked -= mask
        return masked

    def aggregate(self, masked_gradients: List[np.ndarray]) -> np.ndarray:
        """Sum masked gradients; pairwise masks cancel."""
        stacked = np.stack(masked_gradients)
        return stacked.mean(axis=0)


def secure_aggregate(local_updates: List[np.ndarray]) -> np.ndarray:
    """
    Convenience function: SecAgg over a list of gradient vectors.

    For n_clients=1 or when SecAgg is disabled, falls back to plain mean.
    """
    n = len(local_updates)
    if n == 0:
        raise ValueError("Empty update list")
    if n == 1:
        return local_updates[0].copy()
    agg = SecureAggregator(n_clients=n)
    agg.setup_session()
    masked = [agg.mask(i, g) for i, g in enumerate(local_updates)]
    return agg.aggregate(masked)
