"""
Temporal EigenTrust: EigenTrust extended with time-decay weighting.

Interaction weights decay exponentially with age:
    w(t) = 0.5^((now - t) / half_life)

This prevents stale reputation from dominating current trust scores,
defending against slow-burn trust accumulation attacks.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .eigentrust import EigenTrust
from .subjective_logic import Opinion, fuse_many, opinion_from_trust_score


class TemporalEigenTrust(EigenTrust):
    """
    EigenTrust with per-interaction temporal decay weights.

    Extended model:
        C_temporal[i,j] = Σ_k w(t_k) * c_ijk

    where t_k is the timestamp of the k-th interaction, and
    w(t) = 0.5^((now - t) / half_life_seconds).
    """

    def __init__(
        self,
        half_life_hours: float = 24.0,
        damping: float = 0.15,
        max_iter: int = 100,
        tol: float = 1e-8,
    ):
        super().__init__(damping=damping, max_iter=max_iter, tol=tol)
        self.half_life_seconds = half_life_hours * 3600.0

    def decay_weight(self, timestamp: float, now: float) -> float:
        """Compute the temporal decay weight for a single interaction."""
        age = max(0.0, now - timestamp)
        return 0.5 ** (age / self.half_life_seconds)

    def decay_weights(self, timestamps: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        """Vectorised decay for an array of timestamps."""
        import time as _time
        now = now or _time.time()
        ages = np.maximum(0.0, now - timestamps)
        return np.power(0.5, ages / self.half_life_seconds)

    def fit_temporal(
        self,
        C: np.ndarray,
        timestamps: np.ndarray,
        now: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute temporal trust vector from a raw interaction matrix C.

        Args:
            C:          (n × n) raw interaction count matrix.
            timestamps: (n,) array of per-agent last-interaction timestamps.
                        Older agents get lower weight.
            now:        Reference time (default: current time).

        Returns:
            Normalised global trust vector (n,).
        """
        weights = self.decay_weights(timestamps, now)
        C_temporal = C * weights[:, np.newaxis]  # row-wise weighting
        return self.fit(C_temporal)


class HybridTrustManager:
    """
    Combines EigenTrust (network reputation) with Subjective Logic opinions
    to produce a unified trust score per DID.

    Algorithm:
      1. EigenTrust gives a global reputation score g_i ∈ [0, 1].
      2. Subjective Logic accumulates pairwise binary feedback into an opinion ω_i.
      3. Final trust = CBF(opinion_from_score(g_i), ω_i).projected_trust
    """

    def __init__(
        self,
        half_life_hours: float = 24.0,
        sl_uncertainty: float = 0.15,
    ):
        self.et = TemporalEigenTrust(half_life_hours=half_life_hours)
        self.sl_uncertainty = sl_uncertainty
        self._sl_opinions: dict = {}   # DID → Opinion
        self._fitted = False

    def fit(
        self,
        dids: List[str],
        C: np.ndarray,
        timestamps: np.ndarray,
    ):
        """Fit EigenTrust from interaction matrix and timestamps."""
        self.et.fit_from_dids(dids, C)
        self._did_list = dids
        self._fitted = True

    def record_feedback(self, did: str, positive: bool):
        """Record a binary trust/distrust feedback event for a DID."""
        prior = self._sl_opinions.get(did, Opinion.vacuous())
        if positive:
            feedback = Opinion(b=0.9, d=0.05, u=0.05)
        else:
            feedback = Opinion(b=0.05, d=0.9, u=0.05)
        from .subjective_logic import fuse
        self._sl_opinions[did] = fuse(prior, feedback)

    def get_score(self, did: str) -> float:
        """Return the fused trust score for a DID."""
        # EigenTrust component
        if self._fitted:
            et_score = self.et.get_score(did)
        else:
            et_score = 0.5
        et_opinion = opinion_from_trust_score(et_score, self.sl_uncertainty)

        # Subjective Logic component
        sl_opinion = self._sl_opinions.get(did, Opinion.vacuous(a=0.5))

        # Fuse
        from .subjective_logic import fuse
        combined = fuse(et_opinion, sl_opinion)
        return round(combined.projected_trust, 6)

    def get_opinion(self, did: str) -> Opinion:
        """Return the full fused opinion for a DID."""
        if self._fitted:
            et_score = self.et.get_score(did)
        else:
            et_score = 0.5
        et_opinion = opinion_from_trust_score(et_score, self.sl_uncertainty)
        sl_opinion = self._sl_opinions.get(did, Opinion.vacuous(a=0.5))
        from .subjective_logic import fuse
        return fuse(et_opinion, sl_opinion)
