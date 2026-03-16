"""
Jøsang's Subjective Logic operators for trust fusion.

Implements:
  - Consensus fusion (⊕_CB – cumulative belief fusion)
  - Trust discounting (⊗ – transitivity operator)
  - Opinion-to-probability projection (expectation value)
  - Uncertainty maximisation (vacuous opinion)

Reference:
  Jøsang, A. (2016). Subjective Logic: A Formalism for Reasoning Under
  Uncertainty. Springer. ISBN 978-3-319-42337-1.
  https://link.springer.com/book/10.1007/978-3-319-42337-1
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Opinion:
    """
    A subjective opinion ω_A^X = (b, d, u, a) where:
      b  – belief mass (agent A believes X is true)
      d  – disbelief mass (agent A believes X is false)
      u  – uncertainty mass (agent A is uncertain about X)
      a  – base rate (prior probability of X)

    Constraint: b + d + u = 1, all ∈ [0, 1].
    """
    b: float   # belief
    d: float   # disbelief
    u: float   # uncertainty
    a: float = 0.5  # base rate

    def __post_init__(self):
        total = self.b + self.d + self.u
        if abs(total - 1.0) > 1e-6:
            # Normalise if slightly off due to floating point
            self.b /= total
            self.d /= total
            self.u /= total
        # Clamp to [0, 1]
        self.b = max(0.0, min(1.0, self.b))
        self.d = max(0.0, min(1.0, self.d))
        self.u = max(0.0, min(1.0, self.u))

    @property
    def expectation(self) -> float:
        """Projected probability E[X] = b + a * u."""
        return self.b + self.a * self.u

    @property
    def projected_trust(self) -> float:
        """Alias for expectation used in trust scoring."""
        return self.expectation

    @classmethod
    def dogmatic_belief(cls, a: float = 0.5) -> "Opinion":
        """Dogmatic belief: u=0, certainty=1."""
        return cls(b=1.0, d=0.0, u=0.0, a=a)

    @classmethod
    def dogmatic_disbelief(cls, a: float = 0.5) -> "Opinion":
        """Dogmatic disbelief: u=0."""
        return cls(b=0.0, d=1.0, u=0.0, a=a)

    @classmethod
    def vacuous(cls, a: float = 0.5) -> "Opinion":
        """Vacuous opinion: maximum uncertainty."""
        return cls(b=0.0, d=0.0, u=1.0, a=a)

    @classmethod
    def from_binomial(cls, pos: int, neg: int, a: float = 0.5, W: float = 2.0) -> "Opinion":
        """
        Beta-distribution-derived opinion from positive/negative evidence counts.
        Uses the Binomial Opinion Likelihood framework.
        """
        r, s = float(pos), float(neg)
        b = r / (r + s + W)
        d = s / (r + s + W)
        u = W / (r + s + W)
        return cls(b=b, d=d, u=u, a=a)

    def __repr__(self) -> str:
        return f"Opinion(b={self.b:.3f}, d={self.d:.3f}, u={self.u:.3f}, a={self.a:.3f})"


def fuse(o1: Opinion, o2: Opinion) -> Opinion:
    """
    Cumulative Belief Fusion (CBF): ⊕_CB.

    Combines two independent opinions about the same proposition.
    When both opinions are dogmatic (u1=u2=0), the result is a weighted average.
    """
    eps = 1e-12
    denom = o1.u + o2.u - o1.u * o2.u + eps

    b = (o1.b * o2.u + o2.b * o1.u) / denom
    d = (o1.d * o2.u + o2.d * o1.u) / denom
    u = (o1.u * o2.u) / denom
    a = (o1.a * o2.u + o2.a * o1.u - (o1.a + o2.a) * o1.u * o2.u) / (
        (o1.u + o2.u - 2 * o1.u * o2.u) + eps
    )
    return Opinion(b=b, d=d, u=u, a=max(0.0, min(1.0, a)))


def fuse_many(opinions: List[Opinion]) -> Opinion:
    """Iteratively apply CBF to a list of opinions."""
    if not opinions:
        return Opinion.vacuous()
    result = opinions[0]
    for op in opinions[1:]:
        result = fuse(result, op)
    return result


def discount(truster: Opinion, trustee: Opinion) -> Opinion:
    """
    Transitivity / Trust Discounting (⊗).

    If A trusts B with opinion `truster`, and B holds opinion `trustee` about X,
    then A's derived opinion about X through B is computed via discounting.
    """
    return Opinion(
        b=truster.b * trustee.b,
        d=truster.b * trustee.d,
        u=truster.d + truster.u + truster.b * trustee.u,
        a=trustee.a,
    )


def average_fusion(o1: Opinion, o2: Opinion) -> Opinion:
    """
    Averaged Belief Fusion (ABF): used when sources are not independent.
    """
    eps = 1e-12
    denom = o1.u + o2.u + eps
    b = (o1.b * o2.u + o2.b * o1.u) / denom
    d = (o1.d * o2.u + o2.d * o1.u) / denom
    u = (2 * o1.u * o2.u) / denom
    a = (o1.a + o2.a) / 2
    return Opinion(b=b, d=d, u=u, a=a)


def opinion_from_trust_score(score: float, uncertainty: float = 0.1) -> Opinion:
    """Convert a scalar trust score ∈ [0, 1] to a subjective opinion."""
    score = max(0.0, min(1.0, score))
    u = max(0.0, min(1.0, uncertainty))
    b = score * (1.0 - u)
    d = (1.0 - score) * (1.0 - u)
    return Opinion(b=b, d=d, u=u, a=score)
