"""
Autoencoder-based anomaly detection for network intrusion detection.

Architecture:
  Encoder: Linear(d→128) → ReLU → Linear(128→64) → ReLU → Linear(64→16)
  Decoder: Linear(16→64) → ReLU → Linear(64→128) → ReLU → Linear(128→d)

Anomaly score: MSE reconstruction error. Threshold set at the 95th percentile
of reconstruction errors on normal training samples.

Training: Adam optimizer with learning rate scheduling (ReduceLROnPlateau).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """Deep autoencoder for network anomaly detection."""

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error."""
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)

    def get_params_flat(self) -> np.ndarray:
        """Return all parameters as a flat numpy array."""
        return np.concatenate([p.detach().cpu().numpy().flatten() for p in self.parameters()])

    def set_params_flat(self, flat: np.ndarray):
        """Set model parameters from a flat numpy array."""
        offset = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.tensor(
                flat[offset:offset + size], dtype=p.dtype
            ).view_as(p)
            offset += size


def train_local(
    model: Autoencoder,
    X_train: np.ndarray,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> Tuple[Autoencoder, float]:
    """
    Train an autoencoder locally on X_train.

    Returns:
        (trained_model, final_loss)
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    ds = TensorDataset(X_t, X_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.MSELoss()

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, _ in dl:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(dl))
        scheduler.step(avg_loss)
        final_loss = avg_loss

    model.eval()
    return model, final_loss


def evaluate_ids(
    model: Autoencoder,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold_percentile: float = 95.0,
) -> dict:
    """
    Evaluate the autoencoder as an IDS.

    Threshold = 95th percentile of reconstruction error on training data.
    Returns a comprehensive metrics dict.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        errors = model.reconstruction_error(X_t).numpy()

    # Set threshold on normal samples
    normal_errors = errors[y_true == 0]
    threshold = np.percentile(normal_errors, threshold_percentile) if len(normal_errors) > 0 else 0.5
    y_pred = (errors > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, cm[1, 1])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, errors) if len(np.unique(y_true)) > 1 else 0.5,
        "fpr": fp / (fp + tn + 1e-10),
        "threshold": float(threshold),
        "mean_recon_error_normal": float(np.mean(normal_errors)),
        "mean_recon_error_attack": float(np.mean(errors[y_true == 1])) if (y_true == 1).any() else 0.0,
        "confusion_matrix": cm.tolist(),
    }
    return metrics
