"""Comparison Experiment.

Runs both modes (with/without confidence scoring) on the same cascade
scenarios across multiple random seeds, and quantifies the contamination
suppression effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats

from ecv.cascade import CascadeSimulator, CascadeResult


@dataclass
class ComparisonResult:
    """Aggregated comparison between scored and unscored modes."""

    n_seeds: int
    scenario_name: str

    # Per-seed results
    unscored_results: list[CascadeResult] = field(default_factory=list)
    scored_results: list[CascadeResult] = field(default_factory=list)

    @property
    def unscored_contamination_rates(self) -> list[float]:
        return [r.contamination_rate for r in self.unscored_results]

    @property
    def scored_contamination_rates(self) -> list[float]:
        return [r.contamination_rate for r in self.scored_results]

    @property
    def unscored_fdr(self) -> list[float]:
        return [r.false_discovery_rate for r in self.unscored_results]

    @property
    def scored_fdr(self) -> list[float]:
        return [r.false_discovery_rate for r in self.scored_results]

    @property
    def contamination_reduction(self) -> float:
        """Mean reduction in contamination rate."""
        u = np.mean(self.unscored_contamination_rates)
        s = np.mean(self.scored_contamination_rates)
        if u == 0:
            return 0.0
        return float((u - s) / u)

    @property
    def fdr_reduction(self) -> float:
        """Mean reduction in false discovery rate."""
        u = np.mean(self.unscored_fdr)
        s = np.mean(self.scored_fdr)
        if u == 0:
            return 0.0
        return float((u - s) / u)

    def statistical_test(self) -> dict:
        """Paired t-test on contamination rates."""
        u = self.unscored_contamination_rates
        s = self.scored_contamination_rates
        if len(u) < 2:
            return {"t_stat": 0.0, "p_value": 1.0, "significant": False}

        t_stat, p_value = sp_stats.ttest_rel(u, s)
        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    def summary(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "n_seeds": self.n_seeds,
            "mean_unscored_contamination": float(np.mean(self.unscored_contamination_rates)),
            "mean_scored_contamination": float(np.mean(self.scored_contamination_rates)),
            "contamination_reduction_pct": self.contamination_reduction * 100,
            "mean_unscored_fdr": float(np.mean(self.unscored_fdr)),
            "mean_scored_fdr": float(np.mean(self.scored_fdr)),
            "fdr_reduction_pct": self.fdr_reduction * 100,
            "mean_gated_nodes": float(np.mean([r.gated_nodes for r in self.scored_results])),
            "statistical_test": self.statistical_test(),
        }


def run_linear_chain_comparison(
    chain_length: int = 15,
    false_positive_rate: float = 0.15,
    contamination_start: int = 3,
    confidence_threshold: float = 0.4,
    n_seeds: int = 100,
) -> ComparisonResult:
    """Run comparison on a linear chain scenario."""
    comparison = ComparisonResult(
        n_seeds=n_seeds,
        scenario_name=f"linear_chain_L{chain_length}_FPR{false_positive_rate}_CS{contamination_start}",
    )

    for seed in range(n_seeds):
        # Without scoring
        sim = CascadeSimulator(seed=seed)
        sim.build_linear_chain(
            length=chain_length,
            false_positive_rate=false_positive_rate,
            contamination_start=contamination_start,
        )
        result_no_score = sim.run_without_scoring()
        comparison.unscored_results.append(result_no_score)

        # With scoring (same seed for fair comparison)
        sim2 = CascadeSimulator(seed=seed)
        sim2.build_linear_chain(
            length=chain_length,
            false_positive_rate=false_positive_rate,
            contamination_start=contamination_start,
        )
        result_with_score = sim2.run_with_scoring(
            confidence_threshold=confidence_threshold
        )
        comparison.scored_results.append(result_with_score)

    return comparison


def run_branching_cascade_comparison(
    depth: int = 4,
    branching_factor: int = 2,
    false_positive_rate: float = 0.15,
    contamination_at_root: bool = True,
    confidence_threshold: float = 0.4,
    n_seeds: int = 100,
) -> ComparisonResult:
    """Run comparison on a branching cascade scenario."""
    comparison = ComparisonResult(
        n_seeds=n_seeds,
        scenario_name=f"branching_D{depth}_B{branching_factor}_FPR{false_positive_rate}",
    )

    for seed in range(n_seeds):
        # Without scoring
        sim = CascadeSimulator(seed=seed)
        sim.build_branching_cascade(
            depth=depth,
            branching_factor=branching_factor,
            false_positive_rate=false_positive_rate,
            contamination_at_root=contamination_at_root,
        )
        result_no_score = sim.run_without_scoring()
        comparison.unscored_results.append(result_no_score)

        # With scoring
        sim2 = CascadeSimulator(seed=seed)
        sim2.build_branching_cascade(
            depth=depth,
            branching_factor=branching_factor,
            false_positive_rate=false_positive_rate,
            contamination_at_root=contamination_at_root,
        )
        result_with_score = sim2.run_with_scoring(
            confidence_threshold=confidence_threshold
        )
        comparison.scored_results.append(result_with_score)

    return comparison


def run_sensitivity_analysis(
    thresholds: list[float] | None = None,
    chain_length: int = 15,
    false_positive_rate: float = 0.15,
    contamination_start: int = 3,
    n_seeds: int = 50,
) -> list[dict]:
    """Run comparison across different confidence thresholds.

    Returns a list of summary dicts for each threshold.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = []
    for threshold in thresholds:
        comp = run_linear_chain_comparison(
            chain_length=chain_length,
            false_positive_rate=false_positive_rate,
            contamination_start=contamination_start,
            confidence_threshold=threshold,
            n_seeds=n_seeds,
        )
        summary = comp.summary()
        summary["threshold"] = threshold
        results.append(summary)

    return results
