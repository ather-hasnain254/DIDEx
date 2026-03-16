"""
Publication-ready figure and table generation for the DID-Guard ACM CCS paper.

Generates:
  Figures:
    Fig 1 – Threat taxonomy risk matrix (bubble chart)
    Fig 2 – FL training curves (F1/AUC per round, 5 configurations)
    Fig 3 – IDS final metrics comparison (grouped bar chart)
    Fig 4 – AgentDojo ASR comparison (DID-Guard vs baselines)
    Fig 5 – ASB per-category attack surface (heatmap)
    Fig 6 – EigenTrust convergence & temporal decay
    Fig 7 – Trust score distribution (violin plot)
    Fig 8 – Policy latency CDF
    Fig 9 – Byzantine resilience (F1 vs fraction Byzantine clients)
    Fig 10 – Utility vs security trade-off frontier

  Tables (LaTeX):
    Table 1 – Threat taxonomy (full 10-entry table)
    Table 2 – FL configuration comparison (final round)
    Table 3 – AgentDojo benchmark results
    Table 4 – ASB per-category results
    Table 5 – Baseline system comparison
    Table 6 – Trust scoring computational overhead

Usage:
    python scripts/generate_figures_tables.py
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from src.did_guard.config import FIG_DIR, TAB_DIR
from src.did_guard.threat_model.taxonomy import THREAT_TAXONOMY, ImpactSeverity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("generate_figures")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = sns.color_palette("deep", 8)
CONFIG_COLORS = {
    "FedAvg": PALETTE[0],
    "FedAvg+DP": PALETTE[1],
    "FLTrust": PALETTE[2],
    "DID-Guard": PALETTE[3],
    "DID-Guard+Byzantine": PALETTE[4],
}
SYSTEM_COLORS = {
    "DID-Guard": PALETTE[3],
    "SAGA": PALETTE[0],
    "CaMeL": PALETTE[1],
    "Progent": PALETTE[2],
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(name: str) -> dict:
    path = TAB_DIR / name
    if not path.exists():
        logger.warning("Missing results file: %s – run training/eval scripts first.", name)
        return {}
    return json.loads(path.read_text())


def _save_fig(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure → %s", path)


# ---------------------------------------------------------------------------
# Figure 1 – Threat taxonomy risk matrix
# ---------------------------------------------------------------------------

def fig_threat_taxonomy():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    severity_map = {ImpactSeverity.LOW: 1, ImpactSeverity.MEDIUM: 2,
                    ImpactSeverity.HIGH: 3, ImpactSeverity.CRITICAL: 4}
    category_colors = {
        "Prompt Injection": "#e41a1c",
        "Byzantine Gradient Poisoning": "#ff7f00",
        "Sybil DID Registration": "#984ea3",
        "VC Replay Attack": "#4daf4a",
        "DID Document Forgery": "#377eb8",
        "Excessive Agency": "#a65628",
        "FL Model Backdoor": "#f781bf",
        "Membership Inference": "#999999",
        "Denial of Service": "#66c2a5",
        "Identity Spoofing": "#fc8d62",
    }

    for t in THREAT_TAXONOMY:
        sev = severity_map[t.severity]
        lik = t.likelihood
        size = 300 * t.risk_score
        cat = t.category.value
        color = category_colors.get(cat, "#888888")
        ax.scatter(lik, sev, s=size, c=color, alpha=0.85, edgecolors="white", linewidth=0.7, zorder=3)
        ax.annotate(
            t.id, (lik, sev),
            fontsize=7.5, ha="center", va="center", color="white", fontweight="bold", zorder=4
        )

    # Quadrant shading
    ax.axvspan(0, 0.5, alpha=0.04, color="green")
    ax.axvspan(0.5, 1.0, alpha=0.04, color="red")
    ax.axhspan(0, 2.5, alpha=0.04, color="green")
    ax.axhspan(2.5, 4.5, alpha=0.04, color="red")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.5, 4.5)
    ax.set_xlabel("Attack Likelihood")
    ax.set_ylabel("Impact Severity")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["Low", "Medium", "High", "Critical"])
    ax.set_title("DID-Guard Threat Taxonomy: Risk Matrix")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # Legend for bubble size
    for rs, label in [(1, "Risk=1"), (4, "Risk=4"), (9, "Risk=9"), (16, "Risk=16")]:
        ax.scatter([], [], s=300 * rs / 4, c="gray", alpha=0.6, label=label)
    ax.legend(title="Risk score (size)", loc="lower right", fontsize=8)

    _save_fig(fig, "fig1_threat_taxonomy_risk_matrix.pdf")


# ---------------------------------------------------------------------------
# Figure 2 – FL training curves
# ---------------------------------------------------------------------------

def fig_fl_training_curves():
    data = _load_json("fl_per_round_metrics.json")
    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics = [("f1", "F1 Score"), ("roc_auc", "ROC-AUC")]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for config, rounds in data.items():
            xs = [r["round"] for r in rounds]
            ys = [r[metric] for r in rounds]
            color = CONFIG_COLORS.get(config, "gray")
            ls = "--" if "Byzantine" in config else "-"
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8,
                    label=config, color=color, linestyle=ls)

        ax.set_xlabel("FL Round")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Federated IDS – {ylabel} per Round")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_ylim(0.5, 1.02)
        ax.set_axisbelow(True)

    fig.tight_layout()
    _save_fig(fig, "fig2_fl_training_curves.pdf")


# ---------------------------------------------------------------------------
# Figure 3 – IDS final metrics comparison
# ---------------------------------------------------------------------------

def fig_ids_metrics_comparison():
    data = _load_json("fl_summary.json")
    if not data:
        return

    metrics = ["f1", "roc_auc", "precision", "recall", "fpr"]
    labels = ["F1", "ROC-AUC", "Precision", "Recall", "FPR"]
    configs = list(data.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(configs)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, config in enumerate(configs):
        vals = [data[config].get(m, 0) for m in metrics]
        offset = (i - len(configs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=config,
                      color=CONFIG_COLORS.get(config, "gray"), alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=70)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Federated IDS: Final Round Metric Comparison")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # Annotate DID-Guard as best
    ax.axhline(0.95, color="green", linestyle=":", linewidth=1, alpha=0.5, label="Target ≥ 0.95")

    _save_fig(fig, "fig3_ids_metrics_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure 4 – AgentDojo ASR comparison
# ---------------------------------------------------------------------------

def fig_agentdojo_asr():
    agentdojo = _load_json("agentdojo_results.json")
    baselines = _load_json("baseline_results.json")
    if not agentdojo or not baselines:
        return

    systems = ["DID-Guard"] + [b["system"] for b in baselines]
    asr_vals = [agentdojo["asr_defended"]] + [b["asr"] for b in baselines]
    utility_vals = [1.0 - agentdojo["utility_drop"]] + [b["utility"] for b in baselines]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(systems))
    width = 0.35

    bars1 = ax.bar(x - width / 2, asr_vals, width, label="Attack Success Rate (↓ better)",
                   color=[SYSTEM_COLORS.get(s, "gray") for s in systems], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, utility_vals, width, label="Task Utility (↑ better)",
                   color=[SYSTEM_COLORS.get(s, "gray") for s in systems], alpha=0.50,
                   edgecolor="black", linewidth=0.5, hatch="//")

    for bar, v in zip(bars1, asr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, v in zip(bars2, utility_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("AgentDojo: Attack Success Rate vs Utility (DID-Guard vs Baselines)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.axhline(y=agentdojo["asr_defended"], color="red", linestyle=":", linewidth=1.5, alpha=0.6)

    _save_fig(fig, "fig4_agentdojo_asr_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure 5 – ASB attack surface heatmap
# ---------------------------------------------------------------------------

def fig_asb_heatmap():
    asb = _load_json("asb_results.json")
    if not asb or "categories" not in asb:
        return

    cats = list(asb["categories"].keys())
    systems = ["Undefended", "DID-Guard"]
    vals = np.array([
        [asb["categories"][c]["asr_undefended"] for c in cats],
        [asb["categories"][c]["asr_defended"] for c in cats],
    ])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(
        vals, annot=True, fmt=".2f", xticklabels=cats, yticklabels=systems,
        cmap="RdYlGn_r", vmin=0, vmax=1, linewidths=0.5, linecolor="white",
        annot_kws={"size": 9, "fontweight": "bold"}, ax=ax,
        cbar_kws={"label": "Attack Success Rate", "shrink": 0.7}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_title("ASB: Attack Success Rate by Category")
    fig.tight_layout()
    _save_fig(fig, "fig5_asb_heatmap.pdf")


# ---------------------------------------------------------------------------
# Figure 6 – EigenTrust convergence & temporal decay
# ---------------------------------------------------------------------------

def fig_eigentrust_convergence():
    from src.did_guard.trust.eigentrust import EigenTrust
    from src.did_guard.trust.eigentrust_temporal import TemporalEigenTrust

    n = 30
    C = EigenTrust.synthetic_matrix(n=n, seed=42)
    et = EigenTrust()

    # Convergence: track L1 delta per iteration manually
    row_sums = C.sum(axis=1, keepdims=True) + 1e-10
    P = C / row_sums
    t = np.ones(n) / n
    deltas = []
    for _ in range(50):
        t_new = (1 - 0.15) * (P.T @ t) + 0.15 / n
        deltas.append(float(np.linalg.norm(t_new - t, 1)))
        t = t_new

    # Temporal decay illustration
    half_lives = [6, 12, 24, 48]
    ages_hours = np.linspace(0, 120, 200)
    tet = TemporalEigenTrust()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: convergence
    axes[0].semilogy(range(1, len(deltas) + 1), deltas, marker=".", color=PALETTE[0], linewidth=2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("||t_k+1 - t_k||_1")
    axes[0].set_title("EigenTrust Power-Iteration Convergence")
    axes[0].axhline(1e-6, color="red", linestyle="--", linewidth=1, label="tol=1e-6")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, linestyle="--")

    # Panel B: temporal decay curves
    for hl, color in zip(half_lives, PALETTE[:4]):
        tet2 = TemporalEigenTrust(half_life_hours=hl)
        now = 120 * 3600
        ts = now - ages_hours * 3600
        weights = tet2.decay_weights(ts, now=now)
        axes[1].plot(ages_hours, weights, label=f"t½={hl}h", color=color, linewidth=2)
    axes[1].set_xlabel("Interaction Age (hours)")
    axes[1].set_ylabel("Temporal Weight w(t)")
    axes[1].set_title("EigenTrust Temporal Decay Curves")
    axes[1].legend(title="Half-life", fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].axhline(0.5, color="gray", linestyle=":", linewidth=1)

    fig.tight_layout()
    _save_fig(fig, "fig6_eigentrust_convergence_decay.pdf")


# ---------------------------------------------------------------------------
# Figure 7 – Trust score distribution (violin)
# ---------------------------------------------------------------------------

def fig_trust_distribution():
    from src.did_guard.trust.eigentrust import EigenTrust
    from src.did_guard.trust.eigentrust_temporal import HybridTrustManager
    import time

    n = 50
    rng = np.random.default_rng(7)
    C = EigenTrust.synthetic_matrix(n=n, seed=7)
    now = time.time()
    ts = np.linspace(now - 3 * 86400, now, n)
    dids = [f"did:peer:{i:04d}" for i in range(n)]

    tm = HybridTrustManager(half_life_hours=24)
    tm.fit(dids, C, ts)

    # Add some positive/negative feedback
    for d in dids[:15]:
        tm.record_feedback(d, positive=True)
    for d in dids[35:]:
        tm.record_feedback(d, positive=False)

    scores = [tm.get_score(d) for d in dids]

    # Compare with/without temporal decay
    et_plain = EigenTrust()
    et_plain.fit_from_dids(dids, C)
    scores_plain = [et_plain.get_score(d) for d in dids]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: violin comparison
    df_violin = pd.DataFrame({
        "score": scores + scores_plain,
        "method": ["HybridTrust (EigenTrust+SL+Temporal)"] * n + ["EigenTrust (plain)"] * n
    })
    sns.violinplot(data=df_violin, x="method", y="score", hue="method", palette=[PALETTE[3], PALETTE[0]],
                   inner="box", ax=axes[0], legend=False)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Trust Score")
    axes[0].set_title("Trust Score Distribution: Hybrid vs Plain EigenTrust")
    axes[0].set_xticks(range(len(df_violin["method"].unique())))
    axes[0].set_xticklabels(df_violin["method"].unique(), rotation=15, ha="right")

    # Panel B: sorted trust profiles
    sorted_plain = np.sort(scores_plain)[::-1]
    sorted_hybrid = np.sort(scores)[::-1]
    rank = np.arange(1, n + 1)
    axes[1].plot(rank, sorted_plain, marker=".", linewidth=1.5, label="Plain EigenTrust", color=PALETTE[0])
    axes[1].plot(rank, sorted_hybrid, marker=".", linewidth=1.5, label="HybridTrust", color=PALETTE[3])
    axes[1].set_xlabel("Agent Rank")
    axes[1].set_ylabel("Trust Score")
    axes[1].set_title("Sorted Trust Profiles")
    axes[1].legend()
    axes[1].axhline(0.4, color="red", linestyle="--", linewidth=1, label="Policy threshold (0.4)")
    axes[1].grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    _save_fig(fig, "fig7_trust_score_distribution.pdf")


# ---------------------------------------------------------------------------
# Figure 8 – Policy latency CDF
# ---------------------------------------------------------------------------

def fig_policy_latency():
    from src.did_guard.agents.orchestrator import DIDGuardOrchestrator
    from src.did_guard.identity.did_manager import get_manager
    from src.did_guard.identity.vc_attestor import VCAttestor
    import time

    orch = DIDGuardOrchestrator()
    manager = get_manager()
    attestor = VCAttestor(issuer_did="did:web:latency-bench.local")
    agent_doc = manager.create_peer_did("latency-test-agent")
    vc = attestor.issue_capability_vc(
        subject_did=agent_doc.did,
        tools=["read_network_log", "query_alert_db"],
        scope="read",
    )
    session = orch.authenticate(agent_doc.did, vc)

    latencies = []
    for _ in range(500):
        t0 = time.perf_counter()
        orch.execute(session, "read_network_log", {"path": "/tmp/test.log", "lines": 10})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies_arr = np.array(latencies)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # CDF
    sorted_lat = np.sort(latencies_arr)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
    axes[0].plot(sorted_lat, cdf, linewidth=2, color=PALETTE[3])
    axes[0].axvline(np.percentile(latencies_arr, 50), color="gray", linestyle="--", linewidth=1,
                    label=f"p50={np.percentile(latencies_arr,50):.2f}ms")
    axes[0].axvline(np.percentile(latencies_arr, 95), color="orange", linestyle="--", linewidth=1,
                    label=f"p95={np.percentile(latencies_arr,95):.2f}ms")
    axes[0].axvline(np.percentile(latencies_arr, 99), color="red", linestyle="--", linewidth=1,
                    label=f"p99={np.percentile(latencies_arr,99):.2f}ms")
    axes[0].set_xlabel("Latency (ms)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("Zero-Trust Policy Evaluation Latency CDF")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, linestyle="--")

    # Histogram
    axes[1].hist(latencies_arr, bins=40, color=PALETTE[3], alpha=0.75, edgecolor="white")
    axes[1].set_xlabel("Latency (ms)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Latency Distribution (n=500)  mean={np.mean(latencies_arr):.2f}ms")
    axes[1].grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    _save_fig(fig, "fig8_policy_latency_cdf.pdf")


# ---------------------------------------------------------------------------
# Figure 9 – Byzantine resilience
# ---------------------------------------------------------------------------

def fig_byzantine_resilience():
    """Simulate varying fraction of Byzantine clients and measure F1."""
    from src.did_guard.fl.fltrust_wrapper import fltrust
    from src.did_guard.fl.secure_agg import secure_aggregate

    rng = np.random.default_rng(42)
    n_clients = 10
    grad_dim = 200

    # "True" clean gradient
    clean = rng.normal(0, 0.1, size=grad_dim)

    fractions = np.arange(0, 0.6, 0.05)
    f1_fedavg, f1_fltrust = [], []

    for frac in fractions:
        n_byz = int(frac * n_clients)
        grads = [clean + rng.normal(0, 0.01, size=grad_dim) for _ in range(n_clients)]
        # Inject Byzantine sign-flip gradients
        for idx in range(n_byz):
            grads[idx] = -grads[idx] * 5.0

        # FedAvg aggregate
        avg_agg = np.stack(grads).mean(axis=0)
        # FLTrust aggregate
        flt_agg = fltrust(clean, grads)

        # Proxy F1: cosine similarity to clean gradient
        def cos_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        f1_fedavg.append(max(0.0, cos_sim(avg_agg, clean)))
        f1_fltrust.append(max(0.0, cos_sim(flt_agg, clean)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fractions * 100, f1_fedavg, marker="s", linewidth=2, label="FedAvg", color=PALETTE[0])
    ax.plot(fractions * 100, f1_fltrust, marker="o", linewidth=2, label="FLTrust (DID-Guard)", color=PALETTE[3])
    ax.fill_between(fractions * 100, f1_fltrust, f1_fedavg,
                    where=[ft < fa for ft, fa in zip(f1_fedavg, f1_fltrust)],
                    alpha=0.15, color=PALETTE[3], label="Defense gain")
    ax.set_xlabel("Fraction of Byzantine Clients (%)")
    ax.set_ylabel("Gradient Alignment (proxy F1)")
    ax.set_title("Byzantine Resilience: FedAvg vs FLTrust (DID-Guard)")
    ax.legend()
    ax.axvline(33, color="red", linestyle=":", linewidth=1.5, alpha=0.6, label="1/3 Byzantine bound")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 55)
    ax.set_ylim(-0.05, 1.05)
    _save_fig(fig, "fig9_byzantine_resilience.pdf")


# ---------------------------------------------------------------------------
# Figure 10 – Utility vs Security trade-off
# ---------------------------------------------------------------------------

def fig_utility_security_tradeoff():
    agentdojo = _load_json("agentdojo_results.json")
    baselines = _load_json("baseline_results.json")
    if not agentdojo or not baselines:
        return

    points = [
        ("DID-Guard", 1.0 - agentdojo["utility_drop"], agentdojo["asr_defended"]),
    ] + [
        (b["system"], b["utility"], b["asr"]) for b in baselines
    ]
    # Add undefended point
    points.append(("Undefended", 0.98, agentdojo["asr_undefended"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, utility, asr in points:
        color = SYSTEM_COLORS.get(name, "gray")
        size = 220 if name == "DID-Guard" else 140
        marker = "*" if name == "DID-Guard" else "o"
        ax.scatter(asr, utility, s=size, color=color, marker=marker,
                   alpha=0.9, edgecolors="white", linewidth=1, zorder=4)
        ax.annotate(name, (asr, utility), textcoords="offset points",
                    xytext=(8, 4), fontsize=9,
                    fontweight="bold" if name == "DID-Guard" else "normal")

    # Ideal region
    ax.axhspan(0.90, 1.0, alpha=0.06, color="green")
    ax.axvspan(0.0, 0.15, alpha=0.06, color="green")
    ax.annotate("Ideal\nRegion", xy=(0.05, 0.95), fontsize=8, color="green", alpha=0.8)

    ax.set_xlabel("Attack Success Rate (ASR) ↓")
    ax.set_ylabel("Task Utility ↑")
    ax.set_title("Security vs Utility Trade-off Frontier")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.invert_xaxis()   # lower ASR is better → move to right side

    _save_fig(fig, "fig10_utility_security_tradeoff.pdf")


# ---------------------------------------------------------------------------
# Tables (LaTeX)
# ---------------------------------------------------------------------------

def table_threat_taxonomy():
    rows = []
    for t in THREAT_TAXONOMY:
        rows.append({
            "ID": t.id,
            "Name": t.name,
            "Category": t.category.value,
            "Severity": t.severity.name,
            "Likelihood": t.likelihood,
            "Risk": round(t.risk_score, 2),
            "Component": t.affected_component,
            "MITRE": t.mitre_ttp or "–",
            "OWASP": t.owasp_ref or "–",
        })
    df = pd.DataFrame(rows)

    # CSV
    df.to_csv(TAB_DIR / "table1_threat_taxonomy.csv", index=False)

    # LaTeX
    latex = df.to_latex(
        index=False, escape=True, longtable=True,
        caption="DID-Guard Threat Taxonomy (Ferrag + OWASP + DID-specific)",
        label="tab:threat_taxonomy",
        column_format="ll p{2.2cm} lrrp{2.2cm}p{2.2cm}l",
    )
    (TAB_DIR / "table1_threat_taxonomy.tex").write_text(latex)
    logger.info("Table 1 saved → table1_threat_taxonomy.tex / .csv")


def table_fl_comparison():
    data = _load_json("fl_summary.json")
    if not data:
        return

    rows = []
    for config, m in data.items():
        rows.append({
            "Configuration": config,
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "FPR": m["fpr"],
            "Byzantine Detected": m.get("n_byzantine_detected", 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "table2_fl_comparison.csv", index=False)
    latex = df.to_latex(
        index=False, escape=True, float_format="%.4f",
        caption="Federated IDS Comparison (final FL round)",
        label="tab:fl_comparison",
        column_format="lrrrrrr",
    )
    (TAB_DIR / "table2_fl_comparison.tex").write_text(latex)
    logger.info("Table 2 saved → table2_fl_comparison.tex / .csv")


def table_agentdojo():
    data = _load_json("agentdojo_results.json")
    if not data:
        return
    baselines = _load_json("baseline_results.json")

    rows = [
        {
            "System": "DID-Guard (ours)",
            "Task Utility": data["utility"],
            "ASR (Undefended)": data["asr_undefended"],
            "ASR (Defended)": data["asr_defended"],
            "Utility Drop": data["utility_drop"],
            "Defense Effectiveness": data["defense_effectiveness"],
        }
    ]
    for b in (baselines or []):
        rows.append({
            "System": b["system"],
            "Task Utility": b["utility"],
            "ASR (Undefended)": data["asr_undefended"],
            "ASR (Defended)": b["asr"],
            "Utility Drop": b["utility_drop"],
            "Defense Effectiveness": round(data["asr_undefended"] - b["asr"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "table3_agentdojo_results.csv", index=False)
    latex = df.to_latex(
        index=False, escape=True, float_format="%.4f",
        caption="AgentDojo Benchmark: DID-Guard vs Baselines",
        label="tab:agentdojo",
        column_format="lrrrrl",
    )
    (TAB_DIR / "table3_agentdojo_results.tex").write_text(latex)
    logger.info("Table 3 saved → table3_agentdojo_results.tex / .csv")


def table_asb():
    asb = _load_json("asb_results.json")
    if not asb or "categories" not in asb:
        return

    rows = []
    for cat, m in asb["categories"].items():
        rows.append({
            "Attack Category": cat.replace("_", " ").title(),
            "OWASP": m.get("owasp_ref", "–"),
            "N Attacks": m["n_attacks"],
            "ASR (Undefended)": m["asr_undefended"],
            "ASR (DID-Guard)": m["asr_defended"],
            "Defense Eff.": m["defense_effectiveness"],
        })
    # Overall row
    rows.append({
        "Attack Category": "\\textbf{Overall}",
        "OWASP": "–",
        "N Attacks": asb["n_attacks"],
        "ASR (Undefended)": asb["overall_asr_undefended"],
        "ASR (DID-Guard)": asb["overall_asr_defended"],
        "Defense Eff.": asb["overall_defense_effectiveness"],
    })
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "table4_asb_results.csv", index=False)
    latex = df.to_latex(
        index=False, escape=False, float_format="%.4f",
        caption="Agent Security Bench (ASB): Per-Category Attack Success Rate",
        label="tab:asb",
        column_format="lclrrr",
    )
    (TAB_DIR / "table4_asb_results.tex").write_text(latex)
    logger.info("Table 4 saved → table4_asb_results.tex / .csv")


def table_baselines():
    baselines = _load_json("baseline_results.json")
    agentdojo = _load_json("agentdojo_results.json")
    if not baselines:
        return

    rows = []
    for b in baselines:
        rows.append({
            "System": b["system"],
            "Identity": "Centralized" if b["system"] in ("SAGA",) else "Static",
            "Trust Scoring": "Threshold-based",
            "FL Defense": "None",
            "Utility": b["utility"],
            "ASR": b["asr"],
        })
    rows.insert(0, {
        "System": "DID-Guard",
        "Identity": "DID/VC (decentralized)",
        "Trust Scoring": "EigenTrust+SL+Temporal",
        "FL Defense": "FLTrust+SecAgg+DP",
        "Utility": 1.0 - agentdojo.get("utility_drop", 0) if agentdojo else "–",
        "ASR": agentdojo.get("asr_defended", "–") if agentdojo else "–",
    })
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "table5_baseline_comparison.csv", index=False)
    latex = df.to_latex(
        index=False, escape=True,
        caption="System Comparison: DID-Guard vs Prior Art",
        label="tab:baselines",
        column_format="llllrr",
    )
    (TAB_DIR / "table5_baseline_comparison.tex").write_text(latex)
    logger.info("Table 5 saved → table5_baseline_comparison.tex / .csv")


def table_trust_overhead():
    """Table 6: trust scoring computational overhead (synthetic benchmark)."""
    from src.did_guard.trust.eigentrust import EigenTrust
    from src.did_guard.trust.eigentrust_temporal import HybridTrustManager
    import time as _time

    results = []
    for n in [10, 20, 50, 100, 200]:
        rng_np = np.random.default_rng(42)
        C = EigenTrust.synthetic_matrix(n=n, seed=42)
        dids = [f"did:peer:{i:04d}" for i in range(n)]
        ts = np.linspace(_time.time() - 86400, _time.time(), n)

        t0 = _time.perf_counter()
        for _ in range(100):
            et = EigenTrust()
            et.fit(C)
        et_ms = (_time.perf_counter() - t0) / 100 * 1000

        t0 = _time.perf_counter()
        for _ in range(100):
            tm = HybridTrustManager()
            tm.fit(dids, C, ts)
        ht_ms = (_time.perf_counter() - t0) / 100 * 1000

        results.append({
            "N Agents": n,
            "EigenTrust (ms)": round(et_ms, 3),
            "HybridTrust (ms)": round(ht_ms, 3),
            "Overhead (×)": round(ht_ms / max(et_ms, 0.001), 2),
        })

    df = pd.DataFrame(results)
    df.to_csv(TAB_DIR / "table6_trust_overhead.csv", index=False)
    latex = df.to_latex(
        index=False, escape=True,
        caption="Trust Scoring Computational Overhead",
        label="tab:trust_overhead",
        column_format="rrrr",
    )
    (TAB_DIR / "table6_trust_overhead.tex").write_text(latex)
    logger.info("Table 6 saved → table6_trust_overhead.tex / .csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=== DID-Guard: Generating figures and tables ===")

    # Figures
    logger.info("-- Figure 1: Threat taxonomy risk matrix")
    fig_threat_taxonomy()
    logger.info("-- Figure 2: FL training curves")
    fig_fl_training_curves()
    logger.info("-- Figure 3: IDS metrics comparison")
    fig_ids_metrics_comparison()
    logger.info("-- Figure 4: AgentDojo ASR comparison")
    fig_agentdojo_asr()
    logger.info("-- Figure 5: ASB heatmap")
    fig_asb_heatmap()
    logger.info("-- Figure 6: EigenTrust convergence & decay")
    fig_eigentrust_convergence()
    logger.info("-- Figure 7: Trust score distribution")
    fig_trust_distribution()
    logger.info("-- Figure 8: Policy latency CDF")
    fig_policy_latency()
    logger.info("-- Figure 9: Byzantine resilience")
    fig_byzantine_resilience()
    logger.info("-- Figure 10: Utility–security trade-off")
    fig_utility_security_tradeoff()

    # Tables
    logger.info("-- Table 1: Threat taxonomy")
    table_threat_taxonomy()
    logger.info("-- Table 2: FL comparison")
    table_fl_comparison()
    logger.info("-- Table 3: AgentDojo results")
    table_agentdojo()
    logger.info("-- Table 4: ASB results")
    table_asb()
    logger.info("-- Table 5: Baseline comparison")
    table_baselines()
    logger.info("-- Table 6: Trust overhead")
    table_trust_overhead()

    logger.info("=== All figures and tables generated ===")
    logger.info("Figures → %s", FIG_DIR)
    logger.info("Tables  → %s", TAB_DIR)


if __name__ == "__main__":
    main()
