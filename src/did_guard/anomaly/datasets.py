"""
Dataset loading with automatic synthetic fallback.

When USE_SYNTHETIC=1 (default), generates procedural datasets that mimic
the statistical properties of CIC-IDS2017 and UNSW-NB15:
  - Gaussian mixture model for normal traffic
  - Shifted distribution for attack traffic
  - Realistic feature dimensionality and label distribution

Drop real CSV files into data/cicids2017/ and data/unsw-nb15/ and set
DID_GUARD_SYNTH=0 to use the real datasets without code changes.

Datasets:
  - CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
  - UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..config import ROOT_DATASET_PATHS, USE_SYNTHETIC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------

def _synth_tabular(
    n: int = 10000,
    n_features: int = 40,
    anomaly_ratio: float = 0.12,
    seed: int = 42,
    attack_types: Optional[list] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic network flow dataset with multi-class attack labels.

    Normal traffic: Gaussian(0, 1) mixture.
    Attack traffic: per-class shifted Gaussian cluster.
    """
    rng = np.random.default_rng(seed)
    attack_types = attack_types or [
        "DDoS", "PortScan", "BruteForce", "Infiltration", "Botnet"
    ]
    n_normal = int(n * (1 - anomaly_ratio))
    n_attack = n - n_normal
    per_class = max(1, n_attack // len(attack_types))

    # Normal traffic
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    y_normal = np.zeros(n_normal, dtype=int)

    # Attack traffic (each class in a different region of feature space)
    X_attacks = []
    y_attacks = []
    for cls_idx, atype in enumerate(attack_types):
        shift = rng.uniform(2.0, 5.0, size=n_features)
        noise = rng.normal(loc=shift, scale=0.8, size=(per_class, n_features))
        X_attacks.append(noise)
        y_attacks.append(np.full(per_class, cls_idx + 1, dtype=int))

    X = np.vstack([X_normal] + X_attacks)
    y = np.hstack([y_normal] + y_attacks)

    cols = [f"f{i:03d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    df["is_attack"] = (y > 0).astype(int)
    attack_map = {0: "BENIGN"}
    for i, t in enumerate(attack_types):
        attack_map[i + 1] = t
    df["attack_type"] = df["label"].map(attack_map)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _ensure_synthetic(path: Path, **kwargs) -> pd.DataFrame:
    """Generate and cache synthetic data at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        logger.info("Generating synthetic dataset at %s", path)
        df = _synth_tabular(**kwargs)
        df.to_csv(path, index=False)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_unsw(sample: int = 5000) -> pd.DataFrame:
    """Load UNSW-NB15 training set (or generate synthetic stand-in)."""
    path = ROOT_DATASET_PATHS["unsw_nb15"] / "UNSW_NB15_training-set.csv"
    if USE_SYNTHETIC:
        df = _ensure_synthetic(path, n=12000, n_features=45, anomaly_ratio=0.13, seed=7)
    else:
        if not path.exists():
            raise FileNotFoundError(
                f"UNSW-NB15 dataset not found at {path}. "
                "Set DID_GUARD_SYNTH=1 to use synthetic data, or place the real "
                "CSV at the expected path."
            )
        df = pd.read_csv(path)
    return df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)


def load_cic(sample: int = 5000) -> pd.DataFrame:
    """Load CIC-IDS2017 (or generate synthetic stand-in)."""
    path = ROOT_DATASET_PATHS["cicids2017"] / "Monday-WorkingHours.pcap_ISCX.csv"
    if USE_SYNTHETIC:
        df = _ensure_synthetic(
            path, n=15000, n_features=35, anomaly_ratio=0.10, seed=13,
            attack_types=["DoS Hulk", "PortScan", "DDoS", "FTP-Patator", "SSH-Patator"],
        )
    else:
        if not path.exists():
            raise FileNotFoundError(
                f"CIC-IDS2017 dataset not found at {path}. "
                "Set DID_GUARD_SYNTH=1 to use synthetic data."
            )
        df = pd.read_csv(path)
    return df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)


def get_feature_matrix(
    df: pd.DataFrame,
    label_col: str = "is_attack",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (X, y) from a dataset DataFrame.

    Drops non-numeric and label columns before returning.
    """
    drop_cols = [c for c in ["label", "is_attack", "attack_type"] if c in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).values.astype(np.float32)
    y = df[label_col].values.astype(int) if label_col in df.columns else np.zeros(len(df), dtype=int)
    return X, y
