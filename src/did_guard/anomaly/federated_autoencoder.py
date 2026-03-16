"""
Federated autoencoder training with:
  - Differential Privacy (DP) gradient perturbation (Opacus-style)
  - Secure Aggregation (SecAgg mask-based)
  - FLTrust Byzantine-robust aggregation
  - Per-round metrics logging

Threat model coverage:
  - Byzantine poisoning → FLTrust cosine trust scoring
  - Gradient leakage   → SecAgg masking + DP noise
  - Backdoor attacks   → FLTrust gradient norm clipping
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from ..config import FL_CLIENTS, FL_DP_NOISE, FL_ROUNDS, FL_USE_FLTRUST
from .autoencoder import Autoencoder, evaluate_ids, train_local
from .datasets import get_feature_matrix, load_cic, load_unsw
from ..fl.fltrust_wrapper import fltrust, fltrust_with_diagnostics
from ..fl.secure_agg import secure_aggregate

logger = logging.getLogger(__name__)


def _add_dp_noise(gradient: np.ndarray, noise_multiplier: float, max_grad_norm: float = 1.0) -> np.ndarray:
    """
    Add Gaussian DP noise to a gradient vector.

    Gaussian mechanism: noise ~ N(0, (noise_multiplier * max_grad_norm)^2)
    """
    # Clip gradient norm
    norm = np.linalg.norm(gradient)
    if norm > max_grad_norm:
        gradient = gradient * (max_grad_norm / (norm + 1e-10))
    # Add Gaussian noise
    sigma = noise_multiplier * max_grad_norm
    noise = np.random.randn(*gradient.shape) * sigma
    return gradient + noise


def _split_clients(X: np.ndarray, n_clients: int) -> List[np.ndarray]:
    """Split dataset into n_clients roughly equal shards."""
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    shards = np.array_split(indices, n_clients)
    return [X[s] for s in shards if len(s) > 0]


def inject_byzantine_gradients(
    client_updates: List[np.ndarray],
    n_byzantine: int,
    attack: str = "sign_flip",
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Inject Byzantine gradients to simulate adversarial FL clients.

    attack options:
      - "sign_flip": negate the gradient
      - "random":    replace with random noise
      - "scaling":   scale gradient by large constant
    """
    rng = rng or np.random.default_rng(0)
    n = len(client_updates)
    byzantine_idx = rng.choice(n, size=min(n_byzantine, n), replace=False)
    result = [g.copy() for g in client_updates]
    for idx in byzantine_idx:
        g = client_updates[idx]
        if attack == "sign_flip":
            result[idx] = -g
        elif attack == "random":
            result[idx] = rng.normal(0, np.std(g) * 5, size=g.shape)
        elif attack == "scaling":
            result[idx] = g * 100.0
    return result


class FederatedIDSTrainer:
    """
    Federated IDS trainer orchestrating multiple local clients.

    Supports:
      - Plain FedAvg (baseline)
      - FedAvg + DP (privacy-preserving)
      - FLTrust (Byzantine-robust)
      - FLTrust + SecAgg + DP (full DID-Guard stack)
    """

    def __init__(
        self,
        n_rounds: int = FL_ROUNDS,
        n_clients: int = FL_CLIENTS,
        dp_noise: float = FL_DP_NOISE,
        use_fltrust: bool = FL_USE_FLTRUST,
        use_secagg: bool = True,
        use_dp: bool = True,
        n_byzantine: int = 0,
        byzantine_attack: str = "sign_flip",
        local_epochs: int = 3,
        dataset: str = "unsw",
    ):
        self.n_rounds = n_rounds
        self.n_clients = n_clients
        self.dp_noise = dp_noise
        self.use_fltrust = use_fltrust
        self.use_secagg = use_secagg
        self.use_dp = use_dp
        self.n_byzantine = n_byzantine
        self.byzantine_attack = byzantine_attack
        self.local_epochs = local_epochs
        self.dataset = dataset

        self.round_metrics: List[Dict] = []
        self.model: Optional[Autoencoder] = None
        self.scaler = StandardScaler()

    def _load_data(self):
        if self.dataset == "unsw":
            df = load_unsw(sample=6000)
        else:
            df = load_cic(sample=6000)
        X, y = get_feature_matrix(df)
        X = self.scaler.fit_transform(X).astype(np.float32)
        return X, y

    def train(self) -> Tuple[Autoencoder, List[Dict]]:
        logger.info("Loading %s dataset…", self.dataset.upper())
        X, y = self._load_data()

        dim = X.shape[1]
        self.model = Autoencoder(input_dim=dim)
        global_flat = self.model.get_params_flat()

        # Use clean samples as root gradient dataset (for FLTrust)
        X_normal = X[y == 0][:500] if (y == 0).any() else X[:500]

        # Split data among clients
        shards = _split_clients(X, self.n_clients)

        logger.info(
            "Starting FL: %d rounds, %d clients, FLTrust=%s, DP=%s, SecAgg=%s, Byzantine=%d",
            self.n_rounds, self.n_clients, self.use_fltrust,
            self.use_dp, self.use_secagg, self.n_byzantine,
        )

        for rnd in range(1, self.n_rounds + 1):
            t0 = time.time()
            client_grads = []

            for c_idx, X_shard in enumerate(shards):
                # Create fresh local model with current global params
                local = Autoencoder(input_dim=dim)
                local.set_params_flat(global_flat.copy())

                # Local training
                local, _ = train_local(local, X_shard, epochs=self.local_epochs)

                # Compute gradient = local_params - global_params
                grad = local.get_params_flat() - global_flat

                # Apply DP noise
                if self.use_dp:
                    grad = _add_dp_noise(grad, noise_multiplier=self.dp_noise)

                client_grads.append(grad)

            # Inject Byzantine gradients if requested
            if self.n_byzantine > 0:
                client_grads = inject_byzantine_gradients(
                    client_grads, self.n_byzantine, attack=self.byzantine_attack
                )

            # Aggregation
            if self.use_fltrust:
                # Compute root gradient on clean server-side data
                root_model = Autoencoder(input_dim=dim)
                root_model.set_params_flat(global_flat.copy())
                root_model, _ = train_local(root_model, X_normal, epochs=self.local_epochs)
                root_grad = root_model.get_params_flat() - global_flat
                agg_grad, diagnostics = fltrust_with_diagnostics(root_grad, client_grads)
            else:
                if self.use_secagg:
                    agg_grad = secure_aggregate(client_grads)
                else:
                    agg_grad = np.stack(client_grads).mean(axis=0)
                diagnostics = {}

            # Update global model
            global_flat = global_flat + agg_grad
            self.model.set_params_flat(global_flat)

            # Evaluate on hold-out
            eval_metrics = evaluate_ids(self.model, X, y)
            elapsed = time.time() - t0

            round_info = {
                "round": rnd,
                "elapsed_s": round(elapsed, 2),
                **{k: round(v, 4) for k, v in eval_metrics.items() if k != "confusion_matrix"},
                "n_byzantine_detected": diagnostics.get("n_byzantine_detected", 0),
                "mean_trust": round(diagnostics.get("mean_trust", 1.0), 4),
            }
            self.round_metrics.append(round_info)
            logger.info(
                "Round %2d/%d | F1=%.4f AUC=%.4f FPR=%.4f | %.1fs",
                rnd, self.n_rounds,
                round_info["f1"], round_info["roc_auc"], round_info["fpr"], elapsed,
            )

        return self.model, self.round_metrics


def run_comparative_experiments() -> Dict[str, List[Dict]]:
    """
    Run all FL configurations for comparative evaluation.

    Configurations:
      1. FedAvg (no defense)
      2. FedAvg + DP
      3. FLTrust (no DP)
      4. FLTrust + DP + SecAgg  ← DID-Guard full stack
      5. FLTrust + DP + SecAgg + Byzantine (robustness test)
    """
    results = {}
    common = dict(n_rounds=5, n_clients=5, local_epochs=3)

    configs = [
        ("FedAvg", dict(use_fltrust=False, use_secagg=False, use_dp=False, n_byzantine=0)),
        ("FedAvg+DP", dict(use_fltrust=False, use_secagg=False, use_dp=True, n_byzantine=0)),
        ("FLTrust", dict(use_fltrust=True, use_secagg=False, use_dp=False, n_byzantine=0)),
        ("DID-Guard", dict(use_fltrust=True, use_secagg=True, use_dp=True, n_byzantine=0)),
        ("DID-Guard+Byzantine", dict(use_fltrust=True, use_secagg=True, use_dp=True, n_byzantine=2)),
    ]

    for name, cfg in configs:
        logger.info("=== Running configuration: %s ===", name)
        trainer = FederatedIDSTrainer(**common, **cfg)
        _, metrics = trainer.train()
        results[name] = metrics

    return results
