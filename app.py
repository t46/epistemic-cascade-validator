"""Epistemic Cascade Validator -- Interactive Web Demo.

Streamlit application for exploring ECV confidence scoring,
cascade contamination simulation, and autoresearch-lite validation.

Usage:
    uv run streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from ecv.cascade import CascadeSimulator, CascadeResult, ExperimentNode
from ecv.comparison import (
    run_linear_chain_comparison,
    run_branching_cascade_comparison,
    ComparisonResult,
)
from ecv.confidence import ConfidenceScorer, ConfidenceScore
from ecv.decision import DecisionEngine
from ecv.adapters.karpathy import (
    KarpathyAutoresearchAdapter,
    parse_results_tsv,
)
from ecv.adapters.base import AdaptedResult

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Epistemic Cascade Validator",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

# Green (high) -> Yellow (mid) -> Red (low)
_CONFIDENCE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "confidence", ["#d32f2f", "#fbc02d", "#388e3c"]
)


def confidence_color(score: float) -> str:
    """Return a hex color for a confidence score (0-1)."""
    rgba = _CONFIDENCE_CMAP(np.clip(score, 0.0, 1.0))
    return mcolors.to_hex(rgba)


def status_color(status: str) -> str:
    """Return a color string for autoresearch status."""
    return {"keep": "#388e3c", "discard": "#f57c00", "crash": "#d32f2f"}.get(
        status, "#757575"
    )


# ---------------------------------------------------------------------------
# Graph drawing helpers
# ---------------------------------------------------------------------------

def _hierarchical_layout(G: nx.DiGraph) -> dict:
    """Simple hierarchical layout for DAGs."""
    pos = {}
    for i, generation in enumerate(nx.topological_generations(G)):
        for j, node in enumerate(sorted(generation)):
            x = (j - (len(generation) - 1) / 2) * 2.5
            y = -i * 2
            pos[node] = (x, y)
    return pos


def draw_simulation_graph(
    sim: CascadeSimulator,
    result: CascadeResult,
    title: str,
) -> plt.Figure:
    """Draw a cascade graph colored by confidence / contamination state."""
    G = sim.graph
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    pos = _hierarchical_layout(G)

    fig, ax = plt.subplots(figsize=(max(12, G.number_of_nodes() * 0.8), 8))

    node_colors = []
    edge_colors_list = []
    node_edge_colors = []
    node_sizes = []

    for node_name in G.nodes():
        state = result.node_states.get(node_name, {})
        conf = state.get("confidence")
        gated = state.get("gated", False)
        contaminated = state.get("contaminated", False)

        if conf is not None:
            node_colors.append(confidence_color(conf))
        elif state.get("positive", False) and not contaminated:
            node_colors.append("#66bb6a")
        elif contaminated:
            node_colors.append("#ef5350")
        else:
            node_colors.append("#bdbdbd")

        # Red border for gated nodes
        if gated:
            node_edge_colors.append("#d32f2f")
            node_sizes.append(600)
        else:
            node_edge_colors.append("#424242")
            node_sizes.append(500)

    for u, v in G.edges():
        u_state = result.node_states.get(u, {})
        v_state = result.node_states.get(v, {})
        if u_state.get("contaminated") or v_state.get("contaminated"):
            edge_colors_list.append("#ef5350")
        elif u_state.get("gated") or v_state.get("gated"):
            edge_colors_list.append("#ffb74d")
        else:
            edge_colors_list.append("#9e9e9e")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=2.0,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7,
        font_weight="bold",
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors_list,
        arrows=True,
        arrowsize=15,
        width=1.5,
    )

    # Confidence annotations
    for node_name in G.nodes():
        state = result.node_states.get(node_name, {})
        conf = state.get("confidence")
        if conf is not None:
            x, y = pos[node_name]
            ax.annotate(
                f"{conf:.2f}",
                (x, y - 0.7),
                fontsize=7,
                ha="center",
                color="#333333",
            )

    legend_elements = [
        mpatches.Patch(facecolor="#66bb6a", edgecolor="#424242", label="True Positive"),
        mpatches.Patch(facecolor="#ef5350", edgecolor="#424242", label="Contaminated (FP)"),
        mpatches.Patch(facecolor="#bdbdbd", edgecolor="#424242", label="Negative"),
        mpatches.Patch(facecolor="#ffffff", edgecolor="#d32f2f", linewidth=2, label="Gated (red border)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.9)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_autoresearch_cascade(
    results: list[AdaptedResult],
    adapter: KarpathyAutoresearchAdapter,
    scorer: ConfidenceScorer,
    engine: DecisionEngine,
    threshold: float,
) -> plt.Figure:
    """Draw the autoresearch cascade chain graph."""
    edges = adapter.build_cascade_chain(results)
    id_to_result = {r.experiment_id: r for r in results}

    G = nx.DiGraph()
    for r in results:
        G.add_node(r.experiment_id)
    for upstream, downstream in edges:
        if upstream in G.nodes and downstream in G.nodes:
            G.add_edge(upstream, downstream)

    # Score nodes
    node_scores: dict[str, ConfidenceScore] = {}
    node_decisions: dict[str, bool] = {}
    for r in results:
        score = scorer.score(r.evidence)
        decision = engine.should_use_as_premise(score, threshold=threshold)
        node_scores[r.experiment_id] = score
        node_decisions[r.experiment_id] = decision.use

    pos = _hierarchical_layout(G)

    fig, ax = plt.subplots(figsize=(max(14, G.number_of_nodes() * 0.6), 10))

    node_colors = []
    node_edge_colors = []
    node_sizes = []

    for node_name in G.nodes():
        r = id_to_result.get(node_name)
        score = node_scores.get(node_name)
        status = r.raw_scores.get("status", "") if r else ""

        if score:
            node_colors.append(confidence_color(score.overall))
        else:
            node_colors.append("#bdbdbd")

        # Red border if gated
        if not node_decisions.get(node_name, True):
            node_edge_colors.append("#d32f2f")
            node_sizes.append(550)
        else:
            node_edge_colors.append("#424242")
            node_sizes.append(450)

    edge_colors_list = []
    for u, v in G.edges():
        u_status = id_to_result.get(u, AdaptedResult("", "", None)).raw_scores.get("status", "")
        if u_status == "keep":
            edge_colors_list.append("#388e3c")
        else:
            edge_colors_list.append("#bdbdbd")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=2.0,
    )

    # Labels: short commit + status
    labels = {}
    for n in G.nodes():
        r = id_to_result.get(n)
        if r:
            status = r.raw_scores.get("status", "?")
            labels[n] = f"{n[:7]}\n({status})"
        else:
            labels[n] = n[:7]

    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=6,
        font_weight="bold",
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors_list,
        arrows=True,
        arrowsize=12,
        width=1.5,
    )

    # Confidence annotations
    for node_name in G.nodes():
        score = node_scores.get(node_name)
        if score:
            x, y = pos[node_name]
            ax.annotate(
                f"{score.overall:.2f}",
                (x, y - 0.7),
                fontsize=6,
                ha="center",
                color="#333333",
            )

    legend_elements = [
        mpatches.Patch(facecolor="#388e3c", edgecolor="#424242", label="High confidence"),
        mpatches.Patch(facecolor="#fbc02d", edgecolor="#424242", label="Medium confidence"),
        mpatches.Patch(facecolor="#d32f2f", edgecolor="#424242", label="Low confidence"),
        mpatches.Patch(facecolor="#ffffff", edgecolor="#d32f2f", linewidth=2, label="Gated (red border)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("Autoresearch Cascade Chain", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

def render_simulation_mode(
    confidence_threshold: float,
    false_positive_rate: float,
    chain_length: int,
    num_seeds: int,
):
    """Render the simulation mode page."""
    st.header("Cascade Simulation")

    seed = st.sidebar.number_input("Random seed", value=42, min_value=0, max_value=9999)
    contamination_start = st.sidebar.slider(
        "Contamination start index",
        min_value=0, max_value=max(chain_length - 1, 1),
        value=min(3, chain_length - 1),
        help="Index in the chain where ground truth flips to False",
    )

    # --- Single run visualization ---
    st.subheader("Single Cascade Run")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Without Scoring**")
        sim_no = CascadeSimulator(seed=seed)
        sim_no.build_linear_chain(
            length=chain_length,
            false_positive_rate=false_positive_rate,
            contamination_start=contamination_start,
        )
        result_no = sim_no.run_without_scoring()
        fig_no = draw_simulation_graph(sim_no, result_no, "WITHOUT Confidence Scoring")
        st.pyplot(fig_no)
        plt.close(fig_no)

        c1, c2, c3 = st.columns(3)
        c1.metric("Contaminated", result_no.contaminated_nodes)
        c2.metric("Contamination Rate", f"{result_no.contamination_rate:.3f}")
        c3.metric("FDR", f"{result_no.false_discovery_rate:.3f}")

    with col2:
        st.markdown("**With Scoring**")
        sim_yes = CascadeSimulator(seed=seed)
        sim_yes.build_linear_chain(
            length=chain_length,
            false_positive_rate=false_positive_rate,
            contamination_start=contamination_start,
        )
        result_yes = sim_yes.run_with_scoring(confidence_threshold=confidence_threshold)
        fig_yes = draw_simulation_graph(sim_yes, result_yes, "WITH Confidence Scoring")
        st.pyplot(fig_yes)
        plt.close(fig_yes)

        c1, c2, c3 = st.columns(3)
        c1.metric("Contaminated", result_yes.contaminated_nodes)
        c2.metric("Contamination Rate", f"{result_yes.contamination_rate:.3f}")
        c3.metric("Gated", result_yes.gated_nodes)

    # --- Comparison ---
    st.subheader("Statistical Comparison (multi-seed)")
    with st.spinner(f"Running {num_seeds} seeds..."):
        comp = run_linear_chain_comparison(
            chain_length=chain_length,
            false_positive_rate=false_positive_rate,
            contamination_start=contamination_start,
            confidence_threshold=confidence_threshold,
            n_seeds=num_seeds,
        )
    summary = comp.summary()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        "Contamination (unscored)",
        f"{summary['mean_unscored_contamination']:.4f}",
    )
    col_b.metric(
        "Contamination (scored)",
        f"{summary['mean_scored_contamination']:.4f}",
    )
    col_c.metric(
        "Reduction",
        f"{summary['contamination_reduction_pct']:.1f}%",
    )

    col_d, col_e, col_f = st.columns(3)
    col_d.metric("FDR (unscored)", f"{summary['mean_unscored_fdr']:.4f}")
    col_e.metric("FDR (scored)", f"{summary['mean_scored_fdr']:.4f}")
    col_f.metric("FDR Reduction", f"{summary['fdr_reduction_pct']:.1f}%")

    test = summary["statistical_test"]
    sig = "significant" if test["significant"] else "not significant"
    st.info(f"Paired t-test: t={test['t_stat']:.3f}, p={test['p_value']:.4f} ({sig})")

    # Comparison bar chart
    fig_comp, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    bars = ax.bar(
        ["Without\nScoring", "With\nScoring"],
        [summary["mean_unscored_contamination"], summary["mean_scored_contamination"]],
        color=["#ef5350", "#4db6ac"], width=0.5, edgecolor="white", linewidth=2,
    )
    ax.set_ylabel("Contamination Rate")
    ax.set_title("Contamination Rate", fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    ax = axes[1]
    bars = ax.bar(
        ["Without\nScoring", "With\nScoring"],
        [summary["mean_unscored_fdr"], summary["mean_scored_fdr"]],
        color=["#ef5350", "#4db6ac"], width=0.5, edgecolor="white", linewidth=2,
    )
    ax.set_ylabel("False Discovery Rate")
    ax.set_title("False Discovery Rate", fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    fig_comp.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    # --- Sensitivity analysis ---
    st.subheader("Sensitivity Analysis")
    st.markdown("Move the threshold slider to see how contamination changes in real time.")

    thresholds = np.arange(0.05, 0.95, 0.05).tolist()
    sens_seeds = max(num_seeds // 2, 20)

    with st.spinner(f"Running sensitivity sweep ({len(thresholds)} thresholds x {sens_seeds} seeds)..."):
        sens_results = []
        for t in thresholds:
            c = run_linear_chain_comparison(
                chain_length=chain_length,
                false_positive_rate=false_positive_rate,
                contamination_start=contamination_start,
                confidence_threshold=t,
                n_seeds=sens_seeds,
            )
            s = c.summary()
            s["threshold"] = t
            sens_results.append(s)

    fig_sens, axes = plt.subplots(1, 3, figsize=(16, 5))

    ts = [r["threshold"] for r in sens_results]
    cont_u = [r["mean_unscored_contamination"] for r in sens_results]
    cont_s = [r["mean_scored_contamination"] for r in sens_results]
    fdr_u = [r["mean_unscored_fdr"] for r in sens_results]
    fdr_s = [r["mean_scored_fdr"] for r in sens_results]
    gated = [r["mean_gated_nodes"] for r in sens_results]

    ax = axes[0]
    ax.plot(ts, cont_u, "o--", color="#ef5350", label="Without Scoring", linewidth=2)
    ax.plot(ts, cont_s, "s-", color="#4db6ac", label="With Scoring", linewidth=2)
    ax.axvline(confidence_threshold, color="#1565c0", linestyle=":", alpha=0.7,
               label=f"Current ({confidence_threshold:.2f})")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Mean Contamination Rate")
    ax.set_title("Contamination vs Threshold", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ts, fdr_u, "o--", color="#ef5350", label="Without Scoring", linewidth=2)
    ax.plot(ts, fdr_s, "s-", color="#4db6ac", label="With Scoring", linewidth=2)
    ax.axvline(confidence_threshold, color="#1565c0", linestyle=":", alpha=0.7,
               label=f"Current ({confidence_threshold:.2f})")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Mean FDR")
    ax.set_title("FDR vs Threshold", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(ts, gated, "D-", color="#5c6bc0", linewidth=2)
    ax.axvline(confidence_threshold, color="#1565c0", linestyle=":", alpha=0.7)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Mean Gated Nodes")
    ax.set_title("Cost: Gated Nodes", fontweight="bold")

    fig_sens.suptitle("Sensitivity Analysis", fontsize=14, fontweight="bold")
    fig_sens.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig_sens)
    plt.close(fig_sens)


# ---------------------------------------------------------------------------
# Autoresearch mode
# ---------------------------------------------------------------------------

def render_autoresearch_mode(
    confidence_threshold: float,
    tsv_path: Path,
):
    """Render the autoresearch-lite data mode."""
    st.header("Autoresearch-Lite Data Validation")

    if not tsv_path.exists():
        st.error(f"results.tsv not found at `{tsv_path}`")
        st.info("Upload a results.tsv file or correct the path in the sidebar.")
        uploaded = st.file_uploader("Upload results.tsv", type=["tsv"])
        if uploaded is not None:
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
            tmp.write(uploaded.read())
            tmp.flush()
            tsv_path = Path(tmp.name)
        else:
            return

    adapter = KarpathyAutoresearchAdapter()
    results = adapter.load(tsv_path)
    if not results:
        st.error("No results loaded from TSV.")
        return

    scorer = ConfidenceScorer()
    engine = DecisionEngine(risk_tolerance="medium")

    # --- Scored table ---
    st.subheader("Experiment Scores")

    rows = []
    scored_list: list[tuple[AdaptedResult, ConfidenceScore]] = []
    for r in results:
        score = scorer.score(r.evidence)
        decision = engine.should_use_as_premise(score, threshold=confidence_threshold)
        scored_list.append((r, score))
        rows.append({
            "Commit": r.experiment_id[:7],
            "Status": r.raw_scores["status"],
            "Val Acc": r.raw_scores["val_accuracy"],
            "Delta": r.raw_scores["improvement"],
            "Confidence": round(score.overall, 4),
            "Repro": round(score.reproduction_component, 3),
            "Effect": round(score.effect_size_component, 3),
            "Stat": round(score.statistical_component, 3),
            "Code": round(score.code_quality_component, 3),
            "Uncertainty": round(score.uncertainty, 3),
            "Risk": decision.risk_level,
            "Gate": "PASS" if decision.use else "BLOCK",
        })

    df = pd.DataFrame(rows)

    # Color the status column
    def _color_status(val):
        c = status_color(val)
        return f"color: {c}; font-weight: bold"

    def _color_gate(val):
        if val == "BLOCK":
            return "color: #d32f2f; font-weight: bold"
        return "color: #388e3c"

    def _color_confidence(val):
        c = confidence_color(val)
        return f"background-color: {c}; color: white; font-weight: bold"

    styled = df.style.map(_color_status, subset=["Status"])
    styled = styled.map(_color_gate, subset=["Gate"])
    styled = styled.map(_color_confidence, subset=["Confidence"])
    styled = styled.format({
        "Val Acc": "{:.4f}",
        "Delta": "{:+.4f}",
        "Confidence": "{:.4f}",
        "Repro": "{:.3f}",
        "Effect": "{:.3f}",
        "Stat": "{:.3f}",
        "Code": "{:.3f}",
        "Uncertainty": "{:.3f}",
    })

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Summary metrics ---
    status_counts = {}
    for r in results:
        s = r.raw_scores["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    n_blocked = sum(1 for row in rows if row["Gate"] == "BLOCK")
    n_total = len(rows)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Experiments", n_total)
    col2.metric("Keep", status_counts.get("keep", 0))
    col3.metric("Discard", status_counts.get("discard", 0))
    col4.metric("Crash / Blocked", f"{status_counts.get('crash', 0)} / {n_blocked}")

    # --- Cascade graph ---
    st.subheader("Cascade Chain Visualization")
    fig_cascade = draw_autoresearch_cascade(
        results, adapter, scorer, engine, confidence_threshold,
    )
    st.pyplot(fig_cascade)
    plt.close(fig_cascade)

    # --- Keep chain analysis ---
    st.subheader("Keep-Chain Compounded Confidence")

    keep_chain_ids = [
        r.experiment_id for r in results
        if r.raw_scores.get("status") == "keep"
    ]
    node_scores = {r.experiment_id: scorer.score(r.evidence) for r in results}

    compounded = 1.0
    chain_rows = []
    for i, cid in enumerate(keep_chain_ids):
        local = node_scores[cid].overall
        compounded *= local
        chain_rows.append({
            "Depth": i,
            "Commit": cid[:7],
            "Local Confidence": round(local, 4),
            "Chain Confidence": round(compounded, 4),
        })

    if chain_rows:
        df_chain = pd.DataFrame(chain_rows)

        def _color_chain_conf(val):
            c = confidence_color(val)
            return f"background-color: {c}; color: white; font-weight: bold"

        styled_chain = df_chain.style.map(
            _color_chain_conf, subset=["Chain Confidence"]
        )
        styled_chain = styled_chain.format({
            "Local Confidence": "{:.4f}",
            "Chain Confidence": "{:.4f}",
        })
        st.dataframe(styled_chain, use_container_width=True, hide_index=True)

        # Plot chain confidence decay
        fig_chain, ax = plt.subplots(figsize=(10, 4))
        depths = [r["Depth"] for r in chain_rows]
        local_vals = [r["Local Confidence"] for r in chain_rows]
        chain_vals = [r["Chain Confidence"] for r in chain_rows]

        ax.plot(depths, local_vals, "o-", color="#5c6bc0", label="Local Confidence", linewidth=2)
        ax.plot(depths, chain_vals, "s-", color="#ef5350", label="Chain Confidence (compounded)", linewidth=2)
        ax.axhline(confidence_threshold, color="#ff9800", linestyle="--",
                    label=f"Threshold ({confidence_threshold:.2f})", alpha=0.7)
        ax.set_xlabel("Keep Chain Depth")
        ax.set_ylabel("Confidence")
        ax.set_title("Confidence Decay Along Keep Chain", fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        fig_chain.tight_layout()
        st.pyplot(fig_chain)
        plt.close(fig_chain)

        final_conf = chain_rows[-1]["Chain Confidence"]
        st.info(
            f"Final chain confidence: **{final_conf:.4f}** -- the model's accuracy claim "
            f"inherits **{(1 - final_conf) * 100:.1f}%** compounded uncertainty."
        )

    # --- Gating effectiveness ---
    st.subheader("Gating Effectiveness")

    tp = fp = tn = fn = 0
    for row in rows:
        is_keep = row["Status"] == "keep"
        is_blocked = row["Gate"] == "BLOCK"
        if is_keep and not is_blocked:
            tp += 1
        elif is_keep and is_blocked:
            fn += 1
        elif not is_keep and is_blocked:
            tn += 1
        else:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    gc1, gc2, gc3, gc4 = st.columns(4)
    gc1.metric("Precision", f"{precision:.3f}")
    gc2.metric("Recall", f"{recall:.3f}")
    gc3.metric("F1 Score", f"{f1:.3f}")
    gc4.metric("Accuracy", f"{(tp + tn) / max(n_total, 1):.3f}")

    # Confusion matrix as a small table
    cm_df = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["Actual Keep", "Actual Non-Keep"],
        columns=["Predicted Keep (PASS)", "Predicted Non-Keep (BLOCK)"],
    )
    st.table(cm_df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("Epistemic Cascade Validator")
    st.markdown(
        "Interactive demo for exploring confidence scoring and "
        "cascade contamination in autonomous research pipelines."
    )

    # --- Sidebar ---
    st.sidebar.title("Settings")

    mode = st.sidebar.radio(
        "Mode",
        ["Simulation", "Autoresearch-Lite Data"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameters")

    confidence_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.05, max_value=0.95, value=0.40, step=0.05,
        help="Minimum confidence to use a result as a premise",
    )

    if mode == "Simulation":
        false_positive_rate = st.sidebar.slider(
            "False positive rate",
            min_value=0.01, max_value=0.50, value=0.15, step=0.01,
        )
        chain_length = st.sidebar.slider(
            "Chain length",
            min_value=3, max_value=30, value=15, step=1,
        )
        num_seeds = st.sidebar.slider(
            "Number of seeds (comparison)",
            min_value=10, max_value=500, value=100, step=10,
        )

        render_simulation_mode(
            confidence_threshold=confidence_threshold,
            false_positive_rate=false_positive_rate,
            chain_length=chain_length,
            num_seeds=num_seeds,
        )

    else:
        default_tsv = Path.home() / "unktok/dev/autoresearch-lite/results.tsv"
        tsv_input = st.sidebar.text_input(
            "Path to results.tsv",
            value=str(default_tsv),
        )
        tsv_path = Path(tsv_input)

        render_autoresearch_mode(
            confidence_threshold=confidence_threshold,
            tsv_path=tsv_path,
        )


if __name__ == "__main__":
    main()
