"""
Federated IDS Training Script.

Runs the full federated anomaly detection pipeline:
  1. Load synthetic (or real) network flow datasets.
  2. Train across multiple FL configurations (FedAvg, FedAvg+DP, FLTrust, DID-Guard full).
  3. Save per-round metrics to outputs/tables/.
  4. Compute and save final IDS comparison metrics.

Usage:
    python scripts/run_training.py [--dataset unsw|cic] [--rounds N] [--clients N]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.did_guard.anomaly.federated_autoencoder import run_comparative_experiments
from src.did_guard.config import FL_CLIENTS, FL_ROUNDS, TAB_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_training")


def parse_args():
    p = argparse.ArgumentParser(description="DID-Guard Federated IDS Training")
    p.add_argument("--dataset", choices=["unsw", "cic"], default="unsw")
    p.add_argument("--rounds", type=int, default=FL_ROUNDS)
    p.add_argument("--clients", type=int, default=FL_CLIENTS)
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=== DID-Guard Federated IDS Training ===")
    logger.info("Dataset: %s | Rounds: %d | Clients: %d", args.dataset, args.rounds, args.clients)

    results = run_comparative_experiments()

    # Save full per-round metrics
    per_round_path = TAB_DIR / "fl_per_round_metrics.json"
    per_round_path.write_text(json.dumps(results, indent=2))
    logger.info("Per-round metrics saved → %s", per_round_path)

    # Compute and save final summary (last-round metrics per config)
    summary = {}
    for config_name, rounds in results.items():
        last = rounds[-1]
        summary[config_name] = {
            "accuracy": last["accuracy"],
            "precision": last["precision"],
            "recall": last["recall"],
            "f1": last["f1"],
            "roc_auc": last["roc_auc"],
            "fpr": last["fpr"],
            "n_byzantine_detected": last.get("n_byzantine_detected", 0),
        }

    summary_path = TAB_DIR / "fl_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary metrics saved → %s", summary_path)

    # Print table to stdout
    logger.info("\n=== Final IDS Metrics (last round) ===")
    header = f"{'Config':<25} {'F1':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'FPR':>6}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, m in summary.items():
        logger.info(
            "%-25s %6.4f %6.4f %6.4f %6.4f %6.4f",
            name, m["f1"], m["roc_auc"], m["precision"], m["recall"], m["fpr"]
        )

    logger.info("=== Training complete ===")
    return results


if __name__ == "__main__":
    main()
