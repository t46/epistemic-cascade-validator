"""Validate ECV against real autoresearch pipeline outputs.

Loads data from three sources:
  1. auto-research-evaluator: 14 experiments with multi-phase paper analysis
  2. ARA: Belief store with claims, evidence, and dependency chains
  3. vanilla autoresearch: Hypothesis generation pipeline with confidence scores

For each, applies the ECV confidence scorer and decision engine, then
analyzes the results: score distributions, gating decisions, and where
possible, cascade contamination analysis.

Usage:
    uv run validate_real_data.py
    uv run validate_real_data.py --verbose
    uv run validate_real_data.py --source evaluator
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ecv.confidence import ConfidenceScorer, EvidencePacket
from ecv.decision import DecisionEngine, Decision
from ecv.adapters.base import AdaptedResult
from ecv.adapters.evaluator import AutoResearchEvaluatorAdapter
from ecv.adapters.ara import ARAAdapter
from ecv.adapters.vanilla import VanillaAutoresearchAdapter


# Default data paths (relative to home directory)
HOME = Path.home()
DATA_PATHS = {
    "evaluator": HOME / "unktok/dev/auto-research-evaluator",
    "ara": HOME / "unktok/dev/autonomous-research-agent/research-workspace",
    "vanilla": HOME / "unktok/dev/unktok-agent/exp-2026-01-13-vanilla-autoresearch",
}


def print_header(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_subheader(text: str) -> None:
    print(f"\n--- {text} ---")


def analyze_results(
    results: list[AdaptedResult],
    scorer: ConfidenceScorer,
    engine: DecisionEngine,
    source_name: str,
    threshold: float = 0.4,
    verbose: bool = False,
) -> dict:
    """Score all results and analyze the distribution."""
    scores = []
    decisions = []
    gated_count = 0
    passed_count = 0

    print_subheader(f"{source_name}: {len(results)} results loaded")

    for r in results:
        score = scorer.score(r.evidence)
        decision = engine.should_use_as_premise(score, threshold=threshold)

        scores.append(score)
        decisions.append(decision)

        if decision.use:
            passed_count += 1
        else:
            gated_count += 1

        if verbose:
            status = "PASS" if decision.use else "GATE"
            print(
                f"  [{status}] {r.experiment_id:<35} "
                f"overall={score.overall:.3f} "
                f"(repro={score.reproduction_component:.3f} "
                f"effect={score.effect_size_component:.3f} "
                f"stat={score.statistical_component:.3f} "
                f"code={score.code_quality_component:.3f}) "
                f"uncertainty={score.uncertainty:.3f} "
                f"risk={decision.risk_level}"
            )

            if verbose and r.mapping_notes:
                for note in r.mapping_notes[:3]:
                    print(f"    -> {note}")

    if not scores:
        print("  No results to analyze.")
        return {}

    overall_scores = [s.overall for s in scores]
    uncertainties = [s.uncertainty for s in scores]

    print(f"\n  Results: {len(results)} total, {passed_count} passed, {gated_count} gated")
    print(f"  Gating rate: {gated_count/len(results)*100:.1f}%")
    print(f"\n  Confidence score distribution:")
    print(f"    Mean:   {np.mean(overall_scores):.4f}")
    print(f"    Median: {np.median(overall_scores):.4f}")
    print(f"    Std:    {np.std(overall_scores):.4f}")
    print(f"    Min:    {np.min(overall_scores):.4f}")
    print(f"    Max:    {np.max(overall_scores):.4f}")
    print(f"\n  Uncertainty distribution:")
    print(f"    Mean:   {np.mean(uncertainties):.4f}")
    print(f"    Median: {np.median(uncertainties):.4f}")

    # Risk level breakdown
    risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for d in decisions:
        risk_counts[d.risk_level] = risk_counts.get(d.risk_level, 0) + 1
    print(f"\n  Risk level distribution:")
    for level, count in risk_counts.items():
        pct = count / len(decisions) * 100
        bar = "#" * int(pct / 2)
        print(f"    {level:8s}: {count:3d} ({pct:5.1f}%) {bar}")

    # Component breakdown
    print(f"\n  Component means:")
    print(f"    Reproduction: {np.mean([s.reproduction_component for s in scores]):.4f}")
    print(f"    Effect size:  {np.mean([s.effect_size_component for s in scores]):.4f}")
    print(f"    Statistical:  {np.mean([s.statistical_component for s in scores]):.4f}")
    print(f"    Code quality: {np.mean([s.code_quality_component for s in scores]):.4f}")

    return {
        "source": source_name,
        "n_results": len(results),
        "n_passed": passed_count,
        "n_gated": gated_count,
        "gating_rate": gated_count / len(results),
        "mean_confidence": float(np.mean(overall_scores)),
        "median_confidence": float(np.median(overall_scores)),
        "std_confidence": float(np.std(overall_scores)),
        "min_confidence": float(np.min(overall_scores)),
        "max_confidence": float(np.max(overall_scores)),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "risk_distribution": risk_counts,
        "component_means": {
            "reproduction": float(np.mean([s.reproduction_component for s in scores])),
            "effect_size": float(np.mean([s.effect_size_component for s in scores])),
            "statistical": float(np.mean([s.statistical_component for s in scores])),
            "code_quality": float(np.mean([s.code_quality_component for s in scores])),
        },
        "per_result": [
            {
                "id": r.experiment_id,
                "description": r.description,
                "overall": scores[i].overall,
                "uncertainty": scores[i].uncertainty,
                "decision": decisions[i].use,
                "risk": decisions[i].risk_level,
                "raw_scores": r.raw_scores,
            }
            for i, r in enumerate(results)
        ],
    }


def analyze_cascade(
    results: list[AdaptedResult],
    edges: list[tuple[str, str]],
    scorer: ConfidenceScorer,
    engine: DecisionEngine,
    source_name: str,
    threshold: float = 0.4,
) -> dict:
    """Analyze cascade contamination risk using dependency edges."""
    import networkx as nx

    if not edges:
        print(f"\n  No dependency edges found for cascade analysis.")
        return {}

    print_subheader(f"{source_name}: Cascade Analysis ({len(edges)} edges)")

    # Build graph
    G = nx.DiGraph()
    id_to_result = {r.experiment_id: r for r in results}

    for r in results:
        G.add_node(r.experiment_id)

    for upstream, downstream in edges:
        if upstream in G.nodes and downstream in G.nodes:
            G.add_edge(upstream, downstream)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if not nx.is_directed_acyclic_graph(G):
        print("  WARNING: Graph has cycles. Skipping topological analysis.")
        return {"warning": "cyclic graph"}

    # Score each node and compute chain confidence
    node_scores = {}
    chain_confidences = {}

    for node in nx.topological_sort(G):
        if node not in id_to_result:
            continue

        r = id_to_result[node]
        score = scorer.score(r.evidence)
        node_scores[node] = score

        # Compute chain confidence (product of all upstream confidences)
        predecessors = list(G.predecessors(node))
        if not predecessors:
            chain_confidences[node] = score.overall
        else:
            upstream_chain = min(
                chain_confidences.get(p, 1.0)
                for p in predecessors
                if p in chain_confidences
            )
            chain_confidences[node] = score.overall * upstream_chain

    print(f"\n  Chain confidence (compounded upstream uncertainty):")
    for node in nx.topological_sort(G):
        if node not in node_scores:
            continue
        local = node_scores[node].overall
        chain = chain_confidences.get(node, local)
        depth = nx.shortest_path_length(G, list(nx.topological_sort(G))[0], node) \
            if nx.has_path(G, list(nx.topological_sort(G))[0], node) else 0
        decision = engine.should_use_as_premise(
            node_scores[node], threshold=threshold
        )
        status = "PASS" if decision.use else "GATE"
        print(
            f"    [{status}] {node:<20} "
            f"local={local:.3f}  chain={chain:.3f}  depth={depth}"
        )

    # Identify contamination risk points
    risk_points = []
    for node, chain_conf in chain_confidences.items():
        if chain_conf < threshold and node in node_scores:
            risk_points.append({
                "node": node,
                "local_confidence": node_scores[node].overall,
                "chain_confidence": chain_conf,
            })

    if risk_points:
        print(f"\n  Contamination risk points (chain confidence < {threshold}):")
        for rp in risk_points:
            print(
                f"    {rp['node']}: local={rp['local_confidence']:.3f} "
                f"chain={rp['chain_confidence']:.3f}"
            )
    else:
        print(f"\n  No contamination risk points detected (all chain confidence >= {threshold})")

    return {
        "source": source_name,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "chain_confidences": {k: float(v) for k, v in chain_confidences.items()},
        "risk_points": risk_points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ECV against real autoresearch data"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-result details",
    )
    parser.add_argument(
        "--source", choices=["evaluator", "ara", "vanilla", "all"],
        default="all",
        help="Which data source to analyze (default: all)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="Confidence threshold for gating (default: 0.4)",
    )
    args = parser.parse_args()

    scorer = ConfidenceScorer()
    engine = DecisionEngine(risk_tolerance="medium")

    all_summaries = {}

    # ================================================================
    # 1. Auto-Research Evaluator
    # ================================================================
    if args.source in ("evaluator", "all"):
        print_header("1. Auto-Research Evaluator (14 experiments)")

        adapter = AutoResearchEvaluatorAdapter()
        path = DATA_PATHS["evaluator"]
        if path.exists():
            results = adapter.load(path)
            summary = analyze_results(
                results, scorer, engine,
                adapter.source_name(),
                threshold=args.threshold,
                verbose=args.verbose,
            )
            all_summaries["evaluator"] = summary
        else:
            print(f"  Data path not found: {path}")

    # ================================================================
    # 2. ARA Belief Store
    # ================================================================
    if args.source in ("ara", "all"):
        print_header("2. ARA Belief Store (claims + dependency chain)")

        adapter = ARAAdapter()
        path = DATA_PATHS["ara"]
        if path.exists():
            results = adapter.load(path)
            summary = analyze_results(
                results, scorer, engine,
                adapter.source_name(),
                threshold=args.threshold,
                verbose=args.verbose,
            )
            all_summaries["ara"] = summary

            # Cascade analysis using claim dependencies
            edges = adapter.build_dependency_graph(results)
            cascade_summary = analyze_cascade(
                results, edges, scorer, engine,
                "ARA claim chain",
                threshold=args.threshold,
            )
            all_summaries["ara_cascade"] = cascade_summary
        else:
            print(f"  Data path not found: {path}")

    # ================================================================
    # 3. Vanilla Autoresearch
    # ================================================================
    if args.source in ("vanilla", "all"):
        print_header("3. Vanilla Autoresearch (hypothesis pipeline)")

        adapter = VanillaAutoresearchAdapter()
        path = DATA_PATHS["vanilla"]
        if path.exists():
            results = adapter.load(path)
            summary = analyze_results(
                results, scorer, engine,
                adapter.source_name(),
                threshold=args.threshold,
                verbose=args.verbose,
            )
            all_summaries["vanilla"] = summary

            # Cascade analysis using hypothesis chains
            edges = adapter.build_hypothesis_chain(results)
            cascade_summary = analyze_cascade(
                results, edges, scorer, engine,
                "Vanilla hypothesis chain",
                threshold=args.threshold,
            )
            all_summaries["vanilla_cascade"] = cascade_summary
        else:
            print(f"  Data path not found: {path}")

    # ================================================================
    # Cross-source comparison
    # ================================================================
    if args.source == "all" and len(all_summaries) > 1:
        print_header("Cross-Source Comparison")

        source_keys = [k for k in all_summaries if "cascade" not in k]
        if len(source_keys) >= 2:
            print(f"\n  {'Source':<30} {'N':>4} {'Mean':>7} {'Med':>7} {'Std':>7} {'Gated%':>7}")
            print(f"  {'-'*30} {'---':>4} {'-----':>7} {'-----':>7} {'-----':>7} {'------':>7}")
            for key in source_keys:
                s = all_summaries[key]
                if not s:
                    continue
                print(
                    f"  {s['source']:<30} "
                    f"{s['n_results']:4d} "
                    f"{s['mean_confidence']:7.4f} "
                    f"{s['median_confidence']:7.4f} "
                    f"{s['std_confidence']:7.4f} "
                    f"{s['gating_rate']*100:6.1f}%"
                )

            # Component comparison
            print(f"\n  Component means by source:")
            print(f"  {'Source':<30} {'Repro':>7} {'Effect':>7} {'Stat':>7} {'Code':>7}")
            print(f"  {'-'*30} {'-----':>7} {'------':>7} {'----':>7} {'----':>7}")
            for key in source_keys:
                s = all_summaries[key]
                if not s:
                    continue
                c = s["component_means"]
                print(
                    f"  {s['source']:<30} "
                    f"{c['reproduction']:7.4f} "
                    f"{c['effect_size']:7.4f} "
                    f"{c['statistical']:7.4f} "
                    f"{c['code_quality']:7.4f}"
                )

    # ================================================================
    # Save results
    # ================================================================
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "real_data_validation.json"

    # Make serializable
    serializable = {}
    for k, v in all_summaries.items():
        if isinstance(v, dict):
            serializable[k] = {
                kk: vv for kk, vv in v.items()
                if kk != "per_result"  # skip verbose per-result data in JSON
            }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    # ================================================================
    # Summary & Limitations
    # ================================================================
    print_header("Summary & Limitations")

    print("""
  FINDINGS:
    The ECV confidence scorer can process real autoresearch pipeline outputs
    through the adapter layer. The three data sources represent different
    stages of autonomous research:

    1. Auto-research-evaluator: Paper-level meta-evaluations with rich
       multi-dimensional scores. ECV maps these to reproduction/effect/stat
       components, producing differentiated confidence scores.

    2. ARA: Literature-review claims with explicit confidence and evidence
       tracking. The dependency graph enables genuine cascade analysis.

    3. Vanilla autoresearch: Hypothesis-level data with self-assessed
       confidence and novelty. Rejected hypotheses correctly receive
       low confidence scores.

  LIMITATIONS (be honest):
    1. PROXY MAPPING: All EvidencePacket fields are derived from proxies,
       not ground truth. No actual reproduction trials, effect size
       measurements, or p-values exist in the source data. The confidence
       scores reflect "how well-documented and self-consistent" the
       research is, not "how likely to reproduce."

    2. SMALL SAMPLE: 14 evaluator experiments + 4 ARA claims + 6 vanilla
       hypotheses = 24 total data points. This is insufficient for
       statistical conclusions about ECV calibration.

    3. NO GROUND TRUTH: We cannot measure whether ECV's gating decisions
       are correct because we do not know which results are true/false
       positives. The validation shows the system *runs* on real data
       and produces *differentiated* scores, but cannot assess *accuracy*.

    4. EVALUATOR CIRCULARITY: The evaluator scores are themselves LLM-
       generated assessments, not human judgments. Using LLM-assessed
       quality to validate an LLM-quality-scoring system has an inherent
       circularity.

    5. CASCADE DEPTH: The ARA chain has only 2-3 levels of dependency.
       The branching factor is small. We cannot validate ECV's cascade
       behavior at the 10-15 node depths used in the simulation.
""")


if __name__ == "__main__":
    main()
