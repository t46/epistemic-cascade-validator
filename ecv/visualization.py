"""Visualization module.

Generates publication-quality figures for the epistemic cascade validator:
  1. Cascade graph with contamination highlighted
  2. Confidence distribution evolution across chain depth
  3. Side-by-side comparison charts
  4. Sensitivity analysis plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from ecv.cascade import CascadeSimulator, CascadeResult
from ecv.comparison import ComparisonResult


# Style configuration
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

OUTPUT_DIR = Path("output")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def plot_cascade_graph(
    sim: CascadeSimulator,
    result: CascadeResult,
    title: str = "Cascade Graph",
    filename: str = "cascade_graph.png",
) -> Path:
    """Visualize the cascade graph with contamination highlighted."""
    ensure_output_dir()

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    G = sim.graph
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot") if _has_graphviz() else \
        _hierarchical_layout(G)

    # Color nodes by state
    node_colors = []
    node_sizes = []
    for node_name in G.nodes():
        state = result.node_states.get(node_name, {})
        if state.get("gated", False):
            node_colors.append("#FFA500")  # orange: gated
            node_sizes.append(500)
        elif state.get("contaminated", False):
            node_colors.append("#FF4444")  # red: contaminated
            node_sizes.append(600)
        elif state.get("positive", False) and state.get("ground_truth", True):
            node_colors.append("#44BB44")  # green: true positive
            node_sizes.append(500)
        elif not state.get("positive", True):
            node_colors.append("#AAAAAA")  # gray: negative
            node_sizes.append(400)
        else:
            node_colors.append("#4488FF")  # blue: other
            node_sizes.append(500)

    nx.draw_networkx(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=8,
        font_weight="bold",
        edge_color="#888888",
        arrows=True,
        arrowsize=15,
        width=1.5,
    )

    # Add confidence annotations
    for node_name in G.nodes():
        state = result.node_states.get(node_name, {})
        conf = state.get("confidence")
        if conf is not None:
            x, y = pos[node_name]
            ax.annotate(
                f"{conf:.2f}",
                (x, y - 20),
                fontsize=7,
                ha="center",
                color="#333333",
            )

    # Legend
    legend_elements = [
        mpatches.Patch(color="#44BB44", label="True Positive"),
        mpatches.Patch(color="#FF4444", label="Contaminated (FP)"),
        mpatches.Patch(color="#FFA500", label="Gated"),
        mpatches.Patch(color="#AAAAAA", label="Negative"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_comparison_bars(
    comparison: ComparisonResult,
    filename: str = "comparison_bars.png",
) -> Path:
    """Side-by-side bar chart comparing scored vs unscored modes."""
    ensure_output_dir()

    summary = comparison.summary()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Contamination rate
    ax = axes[0]
    bars = ax.bar(
        ["Without\nScoring", "With\nScoring"],
        [summary["mean_unscored_contamination"], summary["mean_scored_contamination"]],
        color=["#FF6B6B", "#4ECDC4"],
        width=0.5,
        edgecolor="white",
        linewidth=2,
    )
    ax.set_ylabel("Contamination Rate")
    ax.set_title("Contamination Rate", fontweight="bold")
    ax.set_ylim(0, max(summary["mean_unscored_contamination"] * 1.3, 0.1))
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                f"{height:.3f}", ha="center", va="bottom", fontsize=10)

    # 2. False Discovery Rate
    ax = axes[1]
    bars = ax.bar(
        ["Without\nScoring", "With\nScoring"],
        [summary["mean_unscored_fdr"], summary["mean_scored_fdr"]],
        color=["#FF6B6B", "#4ECDC4"],
        width=0.5,
        edgecolor="white",
        linewidth=2,
    )
    ax.set_ylabel("False Discovery Rate")
    ax.set_title("False Discovery Rate", fontweight="bold")
    ax.set_ylim(0, max(summary["mean_unscored_fdr"] * 1.3, 0.1))
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                f"{height:.3f}", ha="center", va="bottom", fontsize=10)

    # 3. Reduction percentages
    ax = axes[2]
    bars = ax.bar(
        ["Contamination\nReduction", "FDR\nReduction"],
        [summary["contamination_reduction_pct"], summary["fdr_reduction_pct"]],
        color=["#45B7D1", "#96CEB4"],
        width=0.5,
        edgecolor="white",
        linewidth=2,
    )
    ax.set_ylabel("Reduction (%)")
    ax.set_title("Improvement from Scoring", fontweight="bold")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f"{height:.1f}%", ha="center", va="bottom", fontsize=10)

    # Add statistical significance annotation
    test = summary["statistical_test"]
    sig_text = f"p = {test['p_value']:.4f}" + (" *" if test["significant"] else " (n.s.)")
    fig.text(0.5, 0.01, f"Paired t-test: {sig_text}", ha="center", fontsize=10,
             style="italic")

    fig.suptitle(f"Scenario: {comparison.scenario_name}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_contamination_distribution(
    comparison: ComparisonResult,
    filename: str = "contamination_distribution.png",
) -> Path:
    """Histogram of contamination rates across seeds."""
    ensure_output_dir()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bins = np.linspace(0, 1, 25)

    ax.hist(
        comparison.unscored_contamination_rates,
        bins=bins, alpha=0.6, color="#FF6B6B", label="Without Scoring",
        edgecolor="white",
    )
    ax.hist(
        comparison.scored_contamination_rates,
        bins=bins, alpha=0.6, color="#4ECDC4", label="With Scoring",
        edgecolor="white",
    )

    ax.axvline(
        np.mean(comparison.unscored_contamination_rates),
        color="#FF0000", linestyle="--", linewidth=2, label="Mean (unscored)",
    )
    ax.axvline(
        np.mean(comparison.scored_contamination_rates),
        color="#008888", linestyle="--", linewidth=2, label="Mean (scored)",
    )

    ax.set_xlabel("Contamination Rate")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Contamination Rate Distribution ({comparison.n_seeds} seeds)",
        fontweight="bold",
    )
    ax.legend()

    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_sensitivity_analysis(
    sensitivity_results: list[dict],
    filename: str = "sensitivity_analysis.png",
) -> Path:
    """Plot contamination and FDR vs confidence threshold."""
    ensure_output_dir()

    thresholds = [r["threshold"] for r in sensitivity_results]
    contamination_unscored = [r["mean_unscored_contamination"] for r in sensitivity_results]
    contamination_scored = [r["mean_scored_contamination"] for r in sensitivity_results]
    fdr_unscored = [r["mean_unscored_fdr"] for r in sensitivity_results]
    fdr_scored = [r["mean_scored_fdr"] for r in sensitivity_results]
    gated = [r["mean_gated_nodes"] for r in sensitivity_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Contamination vs threshold
    ax = axes[0]
    ax.plot(thresholds, contamination_unscored, "o--", color="#FF6B6B",
            label="Without Scoring", linewidth=2, markersize=6)
    ax.plot(thresholds, contamination_scored, "s-", color="#4ECDC4",
            label="With Scoring", linewidth=2, markersize=6)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Mean Contamination Rate")
    ax.set_title("Contamination vs Threshold", fontweight="bold")
    ax.legend()

    # 2. FDR vs threshold
    ax = axes[1]
    ax.plot(thresholds, fdr_unscored, "o--", color="#FF6B6B",
            label="Without Scoring", linewidth=2, markersize=6)
    ax.plot(thresholds, fdr_scored, "s-", color="#4ECDC4",
            label="With Scoring", linewidth=2, markersize=6)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Mean False Discovery Rate")
    ax.set_title("FDR vs Threshold", fontweight="bold")
    ax.legend()

    # 3. Gated nodes vs threshold (cost of scoring)
    ax = axes[2]
    ax.plot(thresholds, gated, "D-", color="#45B7D1",
            linewidth=2, markersize=6)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Mean Gated Nodes")
    ax.set_title("Cost: Gated Nodes vs Threshold", fontweight="bold")

    fig.suptitle("Sensitivity Analysis: Confidence Threshold", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_chain_depth_confidence(
    sim: CascadeSimulator,
    result: CascadeResult,
    filename: str = "chain_depth_confidence.png",
) -> Path:
    """Plot confidence score evolution across chain depth."""
    ensure_output_dir()

    # Extract depth and confidence for each node
    depths = []
    confidences = []
    colors = []

    for node_name in nx.topological_sort(sim.graph):
        # Depth = longest path from root
        roots = [n for n in sim.graph.nodes() if sim.graph.in_degree(n) == 0]
        if roots:
            try:
                depth = nx.shortest_path_length(sim.graph, roots[0], node_name)
            except nx.NetworkXNoPath:
                depth = 0
        else:
            depth = 0

        state = result.node_states.get(node_name, {})
        conf = state.get("confidence")
        if conf is not None:
            depths.append(depth)
            confidences.append(conf)
            if state.get("contaminated"):
                colors.append("#FF4444")
            elif state.get("gated"):
                colors.append("#FFA500")
            else:
                colors.append("#4ECDC4")

    if not depths:
        # No confidence data (unscored mode)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, "No confidence data (unscored mode)",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        filepath = OUTPUT_DIR / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.scatter(depths, confidences, c=colors, s=80, alpha=0.7, edgecolors="white", linewidth=1)

    # Trend line
    if len(set(depths)) > 1:
        z = np.polyfit(depths, confidences, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(depths), max(depths), 100)
        ax.plot(x_trend, p(x_trend), "--", color="#333333", alpha=0.5, linewidth=2)

    ax.set_xlabel("Chain Depth")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Confidence Score vs Chain Depth", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)

    legend_elements = [
        mpatches.Patch(color="#4ECDC4", label="Accepted"),
        mpatches.Patch(color="#FF4444", label="Contaminated"),
        mpatches.Patch(color="#FFA500", label="Gated"),
    ]
    ax.legend(handles=legend_elements)

    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def _has_graphviz() -> bool:
    """Check if graphviz layout is available."""
    try:
        import pygraphviz  # noqa: F401
        return True
    except ImportError:
        return False


def _hierarchical_layout(G: nx.DiGraph) -> dict:
    """Simple hierarchical layout for DAGs without graphviz."""
    pos = {}
    # Use topological generations for y-coordinate
    for i, generation in enumerate(nx.topological_generations(G)):
        for j, node in enumerate(sorted(generation)):
            # Spread nodes horizontally within each generation
            x = (j - (len(generation) - 1) / 2) * 2
            y = -i * 2
            pos[node] = (x, y)
    return pos
