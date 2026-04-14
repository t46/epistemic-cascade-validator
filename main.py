"""Epistemic Cascade Validator -- Demo & Comparison Runner.

Runs the full experiment comparing cascade contamination with and without
confidence scoring, generates visualizations, and prints summary statistics.

Usage:
    uv run main.py
    uv run main.py --seeds 200 --chain-length 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from ecv.cascade import CascadeSimulator
from ecv.comparison import (
    run_linear_chain_comparison,
    run_branching_cascade_comparison,
    run_sensitivity_analysis,
)
from ecv.visualization import (
    plot_cascade_graph,
    plot_comparison_bars,
    plot_contamination_distribution,
    plot_sensitivity_analysis,
    plot_chain_depth_confidence,
)


def print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_summary(summary: dict, indent: int = 2) -> None:
    prefix = " " * indent
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            for k2, v2 in value.items():
                print(f"{prefix}  {k2}: {v2}")
        elif isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        else:
            print(f"{prefix}{key}: {value}")


def run_demo(args: argparse.Namespace) -> None:
    """Run the full demo."""
    start_time = time.time()

    # ================================================================
    # 1. Single cascade visualization (with and without scoring)
    # ================================================================
    print_header("1. Single Cascade Visualization")

    # Without scoring
    print("\n--- Linear Chain WITHOUT Scoring ---")
    sim_no = CascadeSimulator(seed=42)
    sim_no.build_linear_chain(
        length=args.chain_length,
        false_positive_rate=args.fpr,
        contamination_start=3,
    )
    result_no = sim_no.run_without_scoring()
    print(f"  Total nodes: {result_no.total_nodes}")
    print(f"  Contaminated: {result_no.contaminated_nodes}")
    print(f"  Contamination rate: {result_no.contamination_rate:.3f}")
    print(f"  False discovery rate: {result_no.false_discovery_rate:.3f}")

    path = plot_cascade_graph(
        sim_no, result_no,
        title="Linear Chain -- WITHOUT Confidence Scoring",
        filename="cascade_no_scoring.png",
    )
    print(f"  Saved: {path}")

    # With scoring
    print("\n--- Linear Chain WITH Scoring ---")
    sim_yes = CascadeSimulator(seed=42)
    sim_yes.build_linear_chain(
        length=args.chain_length,
        false_positive_rate=args.fpr,
        contamination_start=3,
    )
    result_yes = sim_yes.run_with_scoring(confidence_threshold=args.threshold)
    print(f"  Total nodes: {result_yes.total_nodes}")
    print(f"  Contaminated: {result_yes.contaminated_nodes}")
    print(f"  Gated: {result_yes.gated_nodes}")
    print(f"  Contamination rate: {result_yes.contamination_rate:.3f}")
    print(f"  False discovery rate: {result_yes.false_discovery_rate:.3f}")

    path = plot_cascade_graph(
        sim_yes, result_yes,
        title="Linear Chain -- WITH Confidence Scoring",
        filename="cascade_with_scoring.png",
    )
    print(f"  Saved: {path}")

    # Confidence vs depth
    path = plot_chain_depth_confidence(
        sim_yes, result_yes,
        filename="chain_depth_confidence.png",
    )
    print(f"  Saved: {path}")

    # ================================================================
    # 2. Branching cascade
    # ================================================================
    print_header("2. Branching Cascade Visualization")

    sim_branch = CascadeSimulator(seed=42)
    sim_branch.build_branching_cascade(
        depth=3,
        branching_factor=3,
        false_positive_rate=args.fpr,
        contamination_at_root=True,
    )
    result_branch_no = sim_branch.run_without_scoring()
    path = plot_cascade_graph(
        sim_branch, result_branch_no,
        title="Branching Cascade -- WITHOUT Scoring (contaminated root)",
        filename="branching_no_scoring.png",
    )
    print(f"  Saved: {path}")

    sim_branch2 = CascadeSimulator(seed=42)
    sim_branch2.build_branching_cascade(
        depth=3,
        branching_factor=3,
        false_positive_rate=args.fpr,
        contamination_at_root=True,
    )
    result_branch_yes = sim_branch2.run_with_scoring(confidence_threshold=args.threshold)
    path = plot_cascade_graph(
        sim_branch2, result_branch_yes,
        title="Branching Cascade -- WITH Scoring (contaminated root)",
        filename="branching_with_scoring.png",
    )
    print(f"  Saved: {path}")

    # ================================================================
    # 3. Statistical comparison (linear chain)
    # ================================================================
    print_header("3. Statistical Comparison -- Linear Chain")

    comp_linear = run_linear_chain_comparison(
        chain_length=args.chain_length,
        false_positive_rate=args.fpr,
        contamination_start=3,
        confidence_threshold=args.threshold,
        n_seeds=args.seeds,
    )
    summary_linear = comp_linear.summary()
    print_summary(summary_linear)

    path = plot_comparison_bars(comp_linear, filename="comparison_linear.png")
    print(f"\n  Saved: {path}")
    path = plot_contamination_distribution(comp_linear, filename="distribution_linear.png")
    print(f"  Saved: {path}")

    # ================================================================
    # 4. Statistical comparison (branching)
    # ================================================================
    print_header("4. Statistical Comparison -- Branching Cascade")

    comp_branch = run_branching_cascade_comparison(
        depth=4,
        branching_factor=2,
        false_positive_rate=args.fpr,
        contamination_at_root=True,
        confidence_threshold=args.threshold,
        n_seeds=args.seeds,
    )
    summary_branch = comp_branch.summary()
    print_summary(summary_branch)

    path = plot_comparison_bars(comp_branch, filename="comparison_branching.png")
    print(f"\n  Saved: {path}")
    path = plot_contamination_distribution(comp_branch, filename="distribution_branching.png")
    print(f"  Saved: {path}")

    # ================================================================
    # 5. Sensitivity analysis
    # ================================================================
    print_header("5. Sensitivity Analysis -- Threshold Sweep")

    sens_results = run_sensitivity_analysis(
        chain_length=args.chain_length,
        false_positive_rate=args.fpr,
        contamination_start=3,
        n_seeds=max(args.seeds // 2, 30),
    )

    print("\n  Threshold | Contamination (unscored) | Contamination (scored) | Reduction")
    print("  " + "-" * 80)
    for r in sens_results:
        print(
            f"  {r['threshold']:.1f}      | "
            f"{r['mean_unscored_contamination']:.4f}                  | "
            f"{r['mean_scored_contamination']:.4f}                | "
            f"{r['contamination_reduction_pct']:.1f}%"
        )

    path = plot_sensitivity_analysis(sens_results, filename="sensitivity_analysis.png")
    print(f"\n  Saved: {path}")

    # ================================================================
    # Save all results as JSON
    # ================================================================
    all_results = {
        "linear_chain": summary_linear,
        "branching_cascade": summary_branch,
        "sensitivity_analysis": sens_results,
        "parameters": {
            "chain_length": args.chain_length,
            "false_positive_rate": args.fpr,
            "confidence_threshold": args.threshold,
            "n_seeds": args.seeds,
        },
    }

    results_path = "output/results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {results_path}")

    elapsed = time.time() - start_time
    print_header(f"Done in {elapsed:.1f}s")

    # ================================================================
    # Summary
    # ================================================================
    print("\nKey findings:")
    print(f"  - Linear chain contamination reduction: "
          f"{summary_linear['contamination_reduction_pct']:.1f}%")
    print(f"  - Branching cascade contamination reduction: "
          f"{summary_branch['contamination_reduction_pct']:.1f}%")
    print(f"  - Linear chain FDR reduction: "
          f"{summary_linear['fdr_reduction_pct']:.1f}%")
    sig = summary_linear['statistical_test']
    print(f"  - Statistical significance: p={sig['p_value']:.4f} "
          f"({'significant' if sig['significant'] else 'not significant'})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Epistemic Cascade Validator -- Demo & Comparison"
    )
    parser.add_argument(
        "--seeds", type=int, default=100,
        help="Number of random seeds for comparison (default: 100)"
    )
    parser.add_argument(
        "--chain-length", type=int, default=15,
        help="Length of the linear chain (default: 15)"
    )
    parser.add_argument(
        "--fpr", type=float, default=0.15,
        help="False positive rate (default: 0.15)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="Confidence threshold for gating (default: 0.4)"
    )
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
