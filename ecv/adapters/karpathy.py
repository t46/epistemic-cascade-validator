"""Adapter for Karpathy-style autoresearch-lite pipeline output (results.tsv).

The autoresearch-lite system runs iterative ML experiments on a codebase,
each time proposing a single change, training, evaluating, and recording
the outcome. The results are stored in a TSV file with columns:

    commit  val_accuracy  memory_gb  status  description

Where status is one of:
  - keep:    Improvement over previous best -> becomes new baseline
  - discard: No improvement -> reverted
  - crash:   Training failed with error -> reverted

Mapping strategy:
  - val_accuracy improvement over baseline -> effect_size proxy.
    Larger improvement = larger observed effect.
  - status (keep/discard/crash) -> reproduction proxy.
    keep: high confidence (the improvement is real and measured).
    discard: moderate (experiment ran but didn't improve).
    crash: low (no evidence at all).
  - memory_gb > 0 and description length -> code_quality proxy.
    Crash experiments with 0 memory = code didn't even run.
    Description quality is a rough indicator of design rigor.
  - Experiment chain (baseline -> keep -> keep -> ...) forms the cascade
    graph. Each "keep" becomes the new baseline for all subsequent
    experiments, creating an explicit dependency chain.

Key advantage over other adapters: This is REAL experimental data with
ground truth -- val_accuracy is an actual measured metric, and the
keep/discard/crash labels are objective outcomes. The cascade chain is
also genuine: each keep truly modifies the codebase that subsequent
experiments build upon.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

from ecv.confidence import EvidencePacket
from ecv.adapters.base import BaseAdapter, AdaptedResult


@dataclass
class KarpathyExperiment:
    """Parsed row from results.tsv."""

    commit: str = ""
    val_accuracy: float = 0.0
    memory_gb: float = 0.0
    status: str = ""  # keep, discard, crash
    description: str = ""
    # Computed fields
    improvement: float = 0.0  # accuracy improvement over current best
    baseline_accuracy: float = 0.0  # the best accuracy at time of this experiment
    chain_parent: str = ""  # commit of the keep this experiment depends on


def parse_results_tsv(tsv_path: Path) -> list[KarpathyExperiment]:
    """Parse a Karpathy-style results.tsv into experiment objects.

    Also computes the improvement over the current baseline for each
    experiment and identifies chain dependencies.
    """
    experiments: list[KarpathyExperiment] = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            exp = KarpathyExperiment(
                commit=row["commit"].strip(),
                val_accuracy=float(row["val_accuracy"]),
                memory_gb=float(row["memory_gb"]),
                status=row["status"].strip(),
                description=row["description"].strip(),
            )
            experiments.append(exp)

    if not experiments:
        return experiments

    # Walk through experiments to compute improvements and chain parents.
    # The first row is the baseline. Each "keep" becomes the new baseline.
    current_best_accuracy = 0.0
    current_best_commit = ""

    for exp in experiments:
        exp.baseline_accuracy = current_best_accuracy
        exp.chain_parent = current_best_commit

        if exp.status == "keep":
            exp.improvement = exp.val_accuracy - current_best_accuracy
            current_best_accuracy = exp.val_accuracy
            current_best_commit = exp.commit
        elif exp.status == "discard":
            exp.improvement = exp.val_accuracy - current_best_accuracy
        else:  # crash
            exp.improvement = 0.0

    return experiments


def _experiment_to_evidence(
    exp: KarpathyExperiment,
) -> tuple[EvidencePacket, list[str]]:
    """Convert a Karpathy experiment to an EvidencePacket.

    Returns (packet, mapping_notes).
    """
    notes: list[str] = []

    # --- Reproduction ---
    # Status is our strongest signal:
    # - keep: The improvement was measured and accepted. Model as 4/5 successful
    #   reproductions (high but not perfect -- could still be noise).
    # - discard: The experiment ran but failed to improve. Model as 2/5
    #   (the experiment itself is reproducible, but the hypothesis failed).
    # - crash: No usable result. Model as 0/5.
    # - baseline: Special case -- the starting point. Model as 3/5
    #   (it's a known architecture but we don't have replication data).
    reproduction_attempts = 5
    if exp.description == "baseline":
        reproduction_successes = 3
        notes.append("reproduction: baseline experiment -> 3/5 (assumed)")
    elif exp.status == "keep":
        reproduction_successes = 4
        notes.append(
            f"reproduction: status=keep (improvement={exp.improvement:+.4f}) -> 4/5"
        )
    elif exp.status == "discard":
        reproduction_successes = 2
        notes.append(
            f"reproduction: status=discard (delta={exp.improvement:+.4f}) -> 2/5"
        )
    else:  # crash
        reproduction_successes = 0
        notes.append("reproduction: status=crash -> 0/5")

    # --- Effect size ---
    # val_accuracy improvement is a direct effect size measurement.
    # Map the improvement to Cohen's d scale:
    #   +0.03 (3% accuracy gain) is a substantial effect in CIFAR-10 tuning.
    #   We use improvement / 0.03 as a scaling factor, capped at d=1.5.
    # For crashes: effect_size = 0.
    # For discards with negative improvement: use abs but note it.
    if exp.status == "crash":
        effect_size = 0.0
        notes.append("effect_size: crash -> d=0.0")
    elif exp.description == "baseline":
        # Baseline has no comparison point -- use a moderate default
        effect_size = 0.3
        notes.append(f"effect_size: baseline (acc={exp.val_accuracy:.4f}) -> d=0.3 (default)")
    else:
        # Scale: 1% improvement = d~0.33, 3% = d~1.0
        raw_d = abs(exp.improvement) / 0.03
        effect_size = min(1.5, max(0.0, raw_d))
        direction = "improvement" if exp.improvement >= 0 else "regression"
        notes.append(
            f"effect_size: {direction}={exp.improvement:+.4f} "
            f"-> d={effect_size:.3f} (scaled by 0.03 reference)"
        )

    # --- p-value ---
    # We don't have actual p-values, but we can approximate:
    # - keep results with large improvements -> low p (strong evidence)
    # - discard results near baseline -> high p (weak evidence)
    # - crash -> p=1.0 (no evidence)
    # Use the improvement magnitude and status to estimate.
    if exp.status == "crash":
        p_value = 1.0
        notes.append("p_value: crash -> p=1.0")
    elif exp.description == "baseline":
        p_value = 0.05
        notes.append("p_value: baseline -> p=0.05 (assumed)")
    else:
        # Map: larger improvement -> lower p-value
        # improvement of 0.03 -> p~0.001; improvement of 0 -> p~0.5
        abs_imp = abs(exp.improvement)
        if abs_imp > 0:
            # p = 10^(-abs_imp * 100): 0.01 improvement -> p=0.1, 0.03 -> p=0.001
            exponent = abs_imp * 100
            p_value = min(1.0, max(1e-10, 10 ** (-exponent)))
        else:
            p_value = 0.5
        # Discard results get a p-value penalty (result was not kept)
        if exp.status == "discard":
            p_value = min(1.0, p_value * 3.0)
        notes.append(
            f"p_value: improvement={exp.improvement:+.4f}, "
            f"status={exp.status} -> p={p_value:.4f}"
        )

    # --- Sample size ---
    # CIFAR-10 has 10,000 validation images. All experiments use the same
    # validation set, so sample_size is constant.
    sample_size = 10000
    notes.append(f"sample_size: CIFAR-10 validation set -> n={sample_size}")

    # --- Code quality ---
    # Real signals:
    # - memory_gb > 0: code at least ran
    # - crash: code failed entirely
    # - description length: longer, more specific descriptions suggest
    #   more deliberate experimentation
    code_passes_lint = exp.status != "crash"
    code_has_tests = True  # autoresearch-lite has built-in val_accuracy testing
    code_test_coverage = 0.0

    if exp.status == "crash":
        code_test_coverage = 0.0
        code_complexity = 0.9  # crashed code is "complex" (broken)
        notes.append("code_quality: crash -> coverage=0.0, complexity=0.9")
    else:
        # Memory usage as a sanity signal (all non-crash should be ~1.1 GB)
        code_test_coverage = 0.6  # all non-crash experiments have val_accuracy
        # Description quality: longer, more specific = better planning
        desc_len = len(exp.description)
        if desc_len > 100:
            code_test_coverage += 0.1
        if desc_len > 150:
            code_test_coverage += 0.1
        # Complexity: keep results suggest well-understood changes
        if exp.status == "keep":
            code_complexity = 0.3
        else:
            code_complexity = 0.5
        notes.append(
            f"code_quality: coverage={code_test_coverage:.2f}, "
            f"complexity={code_complexity:.2f} "
            f"(desc_len={desc_len}, status={exp.status})"
        )

    packet = EvidencePacket(
        reproduction_successes=reproduction_successes,
        reproduction_attempts=reproduction_attempts,
        effect_size=effect_size,
        p_value=p_value,
        sample_size=sample_size,
        code_passes_lint=code_passes_lint,
        code_has_tests=code_has_tests,
        code_test_coverage=min(1.0, code_test_coverage),
        code_complexity_score=code_complexity,
    )

    return packet, notes


class KarpathyAutoresearchAdapter(BaseAdapter):
    """Adapter for Karpathy-style autoresearch-lite results.tsv."""

    def source_name(self) -> str:
        return "karpathy-autoresearch-lite"

    def load(self, path: Path) -> list[AdaptedResult]:
        """Load experiments from a results.tsv file.

        Args:
            path: Path to the results.tsv file (or directory containing it).
        """
        path = Path(path)
        if path.is_dir():
            tsv_path = path / "results.tsv"
        else:
            tsv_path = path

        if not tsv_path.exists():
            return []

        experiments = parse_results_tsv(tsv_path)
        results = []

        for exp in experiments:
            packet, notes = _experiment_to_evidence(exp)

            result = AdaptedResult(
                experiment_id=exp.commit,
                source_path=str(tsv_path),
                evidence=packet,
                raw_scores={
                    "val_accuracy": exp.val_accuracy,
                    "memory_gb": exp.memory_gb,
                    "status": exp.status,
                    "improvement": exp.improvement,
                    "baseline_accuracy": exp.baseline_accuracy,
                    "chain_parent": exp.chain_parent,
                },
                description=(
                    f"[{exp.status}] acc={exp.val_accuracy:.4f} "
                    f"(delta={exp.improvement:+.4f}) {exp.description[:80]}"
                ),
                mapping_notes=notes,
            )
            results.append(result)

        return results

    def build_cascade_chain(
        self, results: list[AdaptedResult],
    ) -> list[tuple[str, str]]:
        """Build the experiment dependency graph.

        In Karpathy-style autoresearch, the cascade structure is:
          - The baseline is the root
          - Each "keep" result becomes the new baseline
          - ALL subsequent experiments (keep, discard, crash) depend on the
            most recent "keep" before them

        Returns list of (upstream_commit, downstream_commit) edges.
        """
        edges = []

        for r in results:
            parent = r.raw_scores.get("chain_parent", "")
            if parent and parent != r.experiment_id:
                edges.append((parent, r.experiment_id))

        return edges

    def get_keep_chain(
        self, results: list[AdaptedResult],
    ) -> list[AdaptedResult]:
        """Extract the linear chain of "keep" results (baseline -> keep1 -> keep2 -> ...).

        This is the critical path: the sequence of accepted improvements
        that the final model is built upon.
        """
        return [
            r for r in results
            if r.raw_scores.get("status") in ("keep",)
            and r.description != "baseline"  # baseline is status=keep but not a "decision"
        ]
