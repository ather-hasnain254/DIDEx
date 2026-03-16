"""
DID-Guard global configuration.

All paths and hyper-parameters are centralised here.
Override via environment variables for containerised deployments.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TAB_DIR = BASE_DIR / "outputs" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
ROOT_DATASET_PATHS = {
    "cicids2017": DATA_DIR / "cicids2017",
    "unsw_nb15": DATA_DIR / "unsw-nb15",
    "agentdojo": DATA_DIR / "agentdojo",
    "asb": DATA_DIR / "asb",
}

# ---------------------------------------------------------------------------
# Synthetic data flag
# ---------------------------------------------------------------------------
# Set DID_GUARD_SYNTH=0 to use real datasets (drop CSVs in data/).
USE_SYNTHETIC: bool = os.getenv("DID_GUARD_SYNTH", "1") == "1"

# ---------------------------------------------------------------------------
# ACA-Py / Hyperledger integration
# ---------------------------------------------------------------------------
ACAPY_ADMIN_URL: str = os.getenv("ACAPY_ADMIN_URL", "http://localhost:8031")
ACAPY_API_KEY: str = os.getenv("ACAPY_API_KEY", "")
FABRIC_PEER_URL: str = os.getenv("FABRIC_PEER_URL", "grpc://localhost:7051")

# ---------------------------------------------------------------------------
# Federated Learning
# ---------------------------------------------------------------------------
FL_ROUNDS: int = int(os.getenv("FL_ROUNDS", "5"))
FL_CLIENTS: int = int(os.getenv("FL_CLIENTS", "5"))
FL_DP_NOISE: float = float(os.getenv("FL_DP_NOISE", "0.001"))
FL_USE_FLTRUST: bool = os.getenv("FL_USE_FLTRUST", "1") == "1"

# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------
POLICY_TRUST_THRESHOLD: float = float(os.getenv("POLICY_TRUST_THRESHOLD", "0.4"))

# ---------------------------------------------------------------------------
# Trust scoring
# ---------------------------------------------------------------------------
EIGENTRUST_DAMPING: float = float(os.getenv("EIGENTRUST_DAMPING", "0.15"))
EIGENTRUST_HALF_LIFE_HOURS: float = float(os.getenv("EIGENTRUST_HALF_LIFE_HOURS", "24.0"))
TRUST_HALF_LIFE_HOURS: float = EIGENTRUST_HALF_LIFE_HOURS
