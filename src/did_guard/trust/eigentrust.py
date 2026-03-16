"""
EigenTrust global reputation algorithm with power-iteration.

Reference:
  Kamvar, S. D., Schlosser, M. T., & Garcia-Molina, H. (2003).
  The eigentrust algorithm for reputation management in P2P networks.
  Proceedings of the 12th international conference on World Wide Web (pp. 640-651).
  https://dl.acm.org/doi/10.1145/775152.775242
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EigenTrust:
    """
    Classic EigenTrust power-iteration implementation.

    Given a trust matrix C where C[i,j] is the number of positive interactions
    agent i has had with agent j, computes a global trust vector t such that:

        t_k+1 = (1 - d) * C_norm^T t_k + d/n * 1

    where d is the damping factor (analogous to PageRank).

    The global trust vector is normalised to sum to 1.
    """

    def __init__(
        self,
        damping: float = 0.15,
        max_iter: int = 100,
        tol: float = 1e-8,
    ):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.global_trust: Optional[np.ndarray] = None
        self.n_agents: int = 0
        self._did_index: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def fit(self, C: np.ndarray) -> np.ndarray:
        """
        Compute global trust vector from a raw positive-interaction matrix C.

        Args:
            C: (n × n) non-negative interaction count matrix.
               C[i,j] = number of positive transactions i has with j.

        Returns:
            Normalised global trust vector of shape (n,).
        """
        n = C.shape[0]
        self.n_agents = n

        # Normalise rows to get stochastic transition matrix
        row_sums = C.sum(axis=1, keepdims=True) + 1e-10
        P = C / row_sums                                   # row-stochastic

        # Uniform prior (pre-trusted set = uniform for simplicity)
        p_prior = np.ones(n) / n

        # Power iteration
        t = p_prior.copy()
        for iteration in range(self.max_iter):
            t_new = (1.0 - self.damping) * (P.T @ t) + self.damping * p_prior
            delta = np.linalg.norm(t_new - t, ord=1)
            t = t_new
            if delta < self.tol:
                logger.debug("EigenTrust converged after %d iterations", iteration + 1)
                break

        # Normalise to [0, 1]
        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)

        self.global_trust = t
        return t

    def fit_from_dids(
        self,
        dids: List[str],
        interaction_matrix: np.ndarray,
    ) -> np.ndarray:
        """Fit EigenTrust and store DID→index mapping."""
        assert len(dids) == interaction_matrix.shape[0]
        self._did_index = {did: i for i, did in enumerate(dids)}
        return self.fit(interaction_matrix)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_score(self, did_or_idx) -> float:
        """Return the global trust score for a DID or integer index."""
        if self.global_trust is None:
            raise RuntimeError("Call fit() before get_score()")
        if isinstance(did_or_idx, str):
            idx = self._did_index.get(did_or_idx)
            if idx is None:
                return 0.0
        else:
            idx = int(did_or_idx)
        return float(np.clip(self.global_trust[idx], 0.0, 1.0))

    def top_k(self, k: int = 10) -> List[tuple]:
        """Return top-k (did, score) pairs sorted by trust score."""
        if self.global_trust is None:
            return []
        inv_index = {v: k for k, v in self._did_index.items()}
        ranked = sorted(
            [(inv_index.get(i, str(i)), float(s)) for i, s in enumerate(self.global_trust)],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]

    # ------------------------------------------------------------------
    # Utility: synthetic trust matrix
    # ------------------------------------------------------------------

    @staticmethod
    def synthetic_matrix(n: int = 20, seed: int = 42) -> np.ndarray:
        """Generate a synthetic positive-interaction matrix for testing."""
        rng = np.random.default_rng(seed)
        C = rng.exponential(scale=5.0, size=(n, n)).astype(float)
        np.fill_diagonal(C, 0.0)
        # Inject a few high-trust "authorities"
        C[:3, :] += rng.exponential(scale=20.0, size=(3, n))
        return C
