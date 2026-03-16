"""
FLTrust Byzantine-robust aggregation.

FLTrust defends against Byzantine gradient poisoning by comparing each
client's gradient to a trusted root gradient (computed from a small clean
server-side dataset). Client gradients whose cosine similarity with the root
is low are down-weighted or excluded.

Reference:
  Cao, X., Fang, M., Liu, J., & Gong, N. Z. (2021).
  FLTrust: Byzantine-robust federated learning via trust bootstrapping.
  In NDSS 2021. arXiv:2012.13995.
  https://arxiv.org/abs/2012.13995
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a) + 1e-10
    norm_b = np.linalg.norm(b) + 1e-10
    return float(np.dot(a, b) / (norm_a * norm_b))


def relu(x: float) -> float:
    return max(0.0, x)


def fltrust(
    root_gradient: np.ndarray,
    client_gradients: List[np.ndarray],
    clip_percentile: float = 80.0,
) -> np.ndarray:
    """
    FLTrust aggregation.

    Algorithm:
      1. Normalise each client gradient to have the same L2-norm as root.
      2. Compute trust score = ReLU(cosine_similarity(g_i, root)).
      3. Weight-average the normalised gradients by trust scores.

    Args:
        root_gradient:    Gradient from the trusted root dataset (server-side).
        client_gradients: List of client gradient vectors.
        clip_percentile:  Optional percentile clipping of trust scores
                          to further reduce influence of borderline gradients.

    Returns:
        Aggregated gradient vector.
    """
    n = len(client_gradients)
    if n == 0:
        raise ValueError("No client gradients provided")

    root_norm = np.linalg.norm(root_gradient) + 1e-10
    trust_scores = []
    normalised = []

    for g in client_gradients:
        sim = cosine_similarity(g, root_gradient)
        ts = relu(sim)
        trust_scores.append(ts)
        g_norm = np.linalg.norm(g) + 1e-10
        # Normalise client gradient to same L2-norm as root
        normalised.append(g / g_norm * root_norm)

    trust_arr = np.array(trust_scores)

    # Optional: clip top-percentile outliers
    if clip_percentile < 100.0 and trust_arr.max() > 0:
        threshold = np.percentile(trust_arr[trust_arr > 0], clip_percentile)
        trust_arr = np.minimum(trust_arr, threshold)

    total_trust = trust_arr.sum()
    if total_trust < 1e-10:
        # All clients untrusted – fall back to plain mean
        return np.stack(client_gradients).mean(axis=0)

    weights = trust_arr / total_trust
    aggregated = np.zeros_like(root_gradient, dtype=float)
    for w, g in zip(weights, normalised):
        aggregated += w * g

    return aggregated


def fltrust_with_diagnostics(
    root_gradient: np.ndarray,
    client_gradients: List[np.ndarray],
) -> Tuple[np.ndarray, dict]:
    """
    FLTrust with detailed diagnostics (trust scores, Byzantine detection flags).

    Returns:
        (aggregated_gradient, diagnostics_dict)
    """
    root_norm = np.linalg.norm(root_gradient) + 1e-10
    sims = []
    trust_scores = []
    normalised = []

    for g in client_gradients:
        sim = cosine_similarity(g, root_gradient)
        sims.append(sim)
        ts = relu(sim)
        trust_scores.append(ts)
        g_norm = np.linalg.norm(g) + 1e-10
        normalised.append(g / g_norm * root_norm)

    trust_arr = np.array(trust_scores)
    total_trust = trust_arr.sum()
    if total_trust < 1e-10:
        agg = np.stack(client_gradients).mean(axis=0)
    else:
        weights = trust_arr / total_trust
        agg = sum(w * g for w, g in zip(weights, normalised))

    byzantine_flags = [s < 0.0 for s in sims]
    diagnostics = {
        "cosine_similarities": sims,
        "trust_scores": trust_scores.copy(),
        "n_byzantine_detected": sum(byzantine_flags),
        "byzantine_flags": byzantine_flags,
        "mean_trust": float(np.mean(trust_scores)),
        "min_trust": float(np.min(trust_scores)),
    }
    return np.array(agg), diagnostics
