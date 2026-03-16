"""
DID-Guard end-to-end runner.

Runs the full pipeline in one command:
  1. generate_synthetic_data.py
  2. run_training.py
  3. evaluate_agent_security.py
  4. generate_figures_tables.py

Usage:
    python scripts/run_all.py [--skip-data] [--skip-training] [--skip-eval]
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_script(name: str, desc: str):
    logger.info("=== %s ===", desc)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / name)],
        check=False,
    )
    if result.returncode != 0:
        logger.error("%s FAILED (exit code %d)", name, result.returncode)
        sys.exit(result.returncode)
    logger.info("=== %s complete ===", desc)


def main():
    p = argparse.ArgumentParser(description="Run the full DID-Guard pipeline.")
    p.add_argument("--skip-data", action="store_true", help="Skip synthetic data generation")
    p.add_argument("--skip-training", action="store_true", help="Skip FL IDS training")
    p.add_argument("--skip-eval", action="store_true", help="Skip agent security eval")
    args = p.parse_args()

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║         DID-Guard Full Pipeline Runner        ║")
    logger.info("╚══════════════════════════════════════════════╝")

    if not args.skip_data:
        run_script("generate_synthetic_data.py", "Step 1/4 – Synthetic Data Generation")

    if not args.skip_training:
        run_script("run_training.py", "Step 2/4 – Federated IDS Training")

    if not args.skip_eval:
        run_script("evaluate_agent_security.py", "Step 3/4 – Agent Security Evaluation")

    run_script("generate_figures_tables.py", "Step 4/4 – Figure & Table Generation")

    logger.info("══════════════════════════════════════════════")
    logger.info("Pipeline complete. Results in outputs/")
    logger.info("  Figures: outputs/figures/")
    logger.info("  Tables:  outputs/tables/")
    logger.info("══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
