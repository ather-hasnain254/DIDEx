"""
Agent Security Evaluation Script.

Runs:
  1. AgentDojo benchmark (synthetic + real if available)
  2. ASB benchmark
  3. Baseline comparisons (SAGA, CaMeL, Progent)
  4. Saves all results to outputs/tables/

Usage:
    python scripts/evaluate_agent_security.py
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.did_guard.benchmarks.agentdojo_runner import AgentDojoBenchmark
from src.did_guard.benchmarks.asb_runner import ASBBenchmark
from src.did_guard.benchmarks.baseline_runner import run_all_baselines
from src.did_guard.config import TAB_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_agent_security")


def main():
    logger.info("=== DID-Guard Agent Security Evaluation ===")

    # 1. AgentDojo
    logger.info("Running AgentDojo benchmark…")
    agentdojo = AgentDojoBenchmark(n_tasks=200, injection_rate=0.35)
    agentdojo_results = agentdojo.run()
    agentdojo_path = TAB_DIR / "agentdojo_results.json"
    agentdojo_path.write_text(json.dumps(agentdojo_results, indent=2, default=str))
    logger.info(
        "AgentDojo: utility=%.3f  ASR(undefended)=%.3f  ASR(DID-Guard)=%.3f",
        agentdojo_results["utility"],
        agentdojo_results["asr_undefended"],
        agentdojo_results["asr_defended"],
    )

    # 2. ASB
    logger.info("Running ASB benchmark…")
    asb = ASBBenchmark(n_per_category=20)
    asb_results = asb.run()
    asb_path = TAB_DIR / "asb_results.json"
    asb_path.write_text(json.dumps(asb_results, indent=2, default=str))
    logger.info(
        "ASB: overall ASR(undefended)=%.3f  ASR(DID-Guard)=%.3f  effectiveness=%.3f",
        asb_results["overall_asr_undefended"],
        asb_results["overall_asr_defended"],
        asb_results["overall_defense_effectiveness"],
    )

    # 3. Baselines
    logger.info("Running baseline comparisons (SAGA, CaMeL, Progent)…")
    baselines = run_all_baselines()
    baselines_path = TAB_DIR / "baseline_results.json"
    baselines_path.write_text(json.dumps(baselines, indent=2, default=str))
    for b in baselines:
        logger.info("  %s: utility=%.3f  ASR=%.3f", b["system"], b["utility"], b["asr"])

    # 4. Combined summary
    combined = {
        "agentdojo": {
            "utility": agentdojo_results["utility"],
            "asr_undefended": agentdojo_results["asr_undefended"],
            "asr_defended": agentdojo_results["asr_defended"],
            "utility_drop": agentdojo_results["utility_drop"],
            "defense_effectiveness": agentdojo_results["defense_effectiveness"],
        },
        "asb": {
            "overall_asr_undefended": asb_results["overall_asr_undefended"],
            "overall_asr_defended": asb_results["overall_asr_defended"],
            "overall_defense_effectiveness": asb_results["overall_defense_effectiveness"],
            "categories": asb_results["categories"],
        },
        "baselines": {b["system"]: b for b in baselines},
    }
    combined_path = TAB_DIR / "agent_security_combined.json"
    combined_path.write_text(json.dumps(combined, indent=2, default=str))
    logger.info("Combined results saved → %s", combined_path)
    logger.info("=== Agent security evaluation complete ===")
    return combined


if __name__ == "__main__":
    main()
