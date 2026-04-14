"""Cascade Simulator.

Models experiment chains as directed graphs where each node is an experiment
with a configurable false positive rate. Simulates contamination propagation
in two modes:

  1. WITHOUT confidence scoring: downstream experiments blindly trust upstream
     results, allowing false positives to cascade unchecked.
  2. WITH confidence scoring: downstream experiments weight upstream results
     by their confidence scores, suppressing contamination.

This models the core problem from the intel: "a keep that doesn't reproduce
is worse than a discard, because subsequent research builds on it."
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from ecv.confidence import ConfidenceScore, ConfidenceScorer, EvidencePacket
from ecv.decision import DecisionEngine


@dataclass
class ExperimentNode:
    """A single experiment in the cascade."""

    name: str
    # Probability that this experiment produces a false positive
    false_positive_rate: float = 0.15
    # Probability that this experiment produces a true positive (power)
    true_positive_rate: float = 0.80
    # Whether the underlying hypothesis is actually true
    ground_truth: bool = True
    # Simulation state
    observed_positive: bool = False
    confidence: Optional[ConfidenceScore] = None
    # Whether this node was contaminated (false positive that propagated)
    contaminated: bool = False
    # Reproduction trial parameters
    reproduction_attempts: int = 3
    # Whether the node was gated (rejected by decision engine)
    gated: bool = False

    def simulate_experiment(self, rng: np.random.Generator) -> bool:
        """Run the experiment and observe a result.

        Returns True if the experiment reports a positive result.
        """
        if self.ground_truth:
            self.observed_positive = rng.random() < self.true_positive_rate
        else:
            self.observed_positive = rng.random() < self.false_positive_rate
        return self.observed_positive

    def simulate_reproductions(
        self, rng: np.random.Generator
    ) -> EvidencePacket:
        """Simulate reproduction attempts and generate evidence."""
        successes = 0
        for _ in range(self.reproduction_attempts):
            if self.ground_truth:
                if rng.random() < self.true_positive_rate:
                    successes += 1
            else:
                if rng.random() < self.false_positive_rate:
                    successes += 1

        # Generate realistic evidence packet
        if self.observed_positive:
            effect_size = rng.normal(0.5 if self.ground_truth else 0.1, 0.15)
            p_value = rng.beta(1, 20) if self.ground_truth else rng.beta(1, 3)
        else:
            effect_size = rng.normal(0.05, 0.1)
            p_value = rng.uniform(0.1, 0.9)

        return EvidencePacket(
            reproduction_successes=successes,
            reproduction_attempts=self.reproduction_attempts,
            effect_size=max(0, effect_size),
            p_value=min(max(p_value, 1e-20), 1.0),
            sample_size=int(rng.integers(20, 200)),
            code_passes_lint=rng.random() > 0.1,
            code_has_tests=rng.random() > 0.5,
            code_test_coverage=rng.uniform(0.0, 0.8),
            code_complexity_score=rng.uniform(0.2, 0.8),
        )


@dataclass
class CascadeResult:
    """Result of a cascade simulation run."""

    total_nodes: int = 0
    contaminated_nodes: int = 0
    gated_nodes: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    node_states: dict = field(default_factory=dict)

    @property
    def contamination_rate(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return self.contaminated_nodes / self.total_nodes

    @property
    def false_discovery_rate(self) -> float:
        positives = self.true_positives + self.false_positives
        if positives == 0:
            return 0.0
        return self.false_positives / positives

    def to_dict(self) -> dict:
        return {
            "total_nodes": self.total_nodes,
            "contaminated_nodes": self.contaminated_nodes,
            "gated_nodes": self.gated_nodes,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "contamination_rate": self.contamination_rate,
            "false_discovery_rate": self.false_discovery_rate,
        }


class CascadeSimulator:
    """Simulates epistemic cascade contamination.

    Builds a directed graph of experiments where edges represent dependency
    (downstream experiment uses upstream result as a premise).
    """

    def __init__(self, seed: int = 42):
        self.graph = nx.DiGraph()
        self.rng = np.random.default_rng(seed)
        self.scorer = ConfidenceScorer()
        self.decision_engine = DecisionEngine()

    def build_linear_chain(
        self,
        length: int = 10,
        false_positive_rate: float = 0.15,
        initial_ground_truth: bool = True,
        contamination_start: int = 0,
    ) -> None:
        """Build a linear chain A -> B -> C -> ... of experiments.

        Args:
            length: Number of experiments in the chain.
            false_positive_rate: Base FPR for each node.
            initial_ground_truth: Whether the first experiment's hypothesis is true.
            contamination_start: Index at which ground truth flips to False
                (simulates a point where a false positive enters the chain).
        """
        self.graph.clear()
        nodes = []
        for i in range(length):
            gt = initial_ground_truth if i < contamination_start else (
                i < contamination_start if contamination_start > 0 else initial_ground_truth
            )
            # After contamination_start, ground truth is False but experiments
            # may still report positive because they're building on a false premise
            if contamination_start > 0 and i >= contamination_start:
                gt = False

            node = ExperimentNode(
                name=f"Exp_{i:02d}",
                false_positive_rate=false_positive_rate,
                ground_truth=gt,
            )
            self.graph.add_node(node.name, data=node)
            nodes.append(node.name)

        for i in range(len(nodes) - 1):
            self.graph.add_edge(nodes[i], nodes[i + 1])

    def build_branching_cascade(
        self,
        depth: int = 4,
        branching_factor: int = 2,
        false_positive_rate: float = 0.15,
        contamination_at_root: bool = False,
    ) -> None:
        """Build a tree-shaped cascade (branching experiments).

        A root experiment spawns `branching_factor` children at each level.
        If `contamination_at_root` is True, the root has ground_truth=False
        but may report a false positive, contaminating the entire tree.
        """
        self.graph.clear()
        counter = [0]

        def make_node(level: int, parent_contaminated: bool) -> str:
            idx = counter[0]
            counter[0] += 1
            gt = not parent_contaminated
            if level == 0 and contamination_at_root:
                gt = False

            node = ExperimentNode(
                name=f"Exp_{idx:02d}",
                false_positive_rate=false_positive_rate,
                ground_truth=gt,
            )
            self.graph.add_node(node.name, data=node)
            return node.name

        def build_tree(parent: str, level: int, parent_contaminated: bool):
            if level >= depth:
                return
            for _ in range(branching_factor):
                child = make_node(level + 1, parent_contaminated)
                self.graph.add_edge(parent, child)
                # If parent was contaminated and reported positive,
                # children build on a false premise
                build_tree(child, level + 1, parent_contaminated)

        root = make_node(0, False)
        root_data = self.graph.nodes[root]["data"]
        build_tree(
            root, 0, contamination_at_root and not root_data.ground_truth
        )

    def run_without_scoring(self) -> CascadeResult:
        """Simulate cascade WITHOUT confidence scoring.

        All positive results are blindly trusted and propagated downstream.
        """
        result = CascadeResult(total_nodes=self.graph.number_of_nodes())

        # Process nodes in topological order (parents before children)
        for node_name in nx.topological_sort(self.graph):
            node: ExperimentNode = self.graph.nodes[node_name]["data"]

            # Check if any parent was a false positive (contamination)
            parents = list(self.graph.predecessors(node_name))
            parent_contaminated = False
            for p in parents:
                p_data: ExperimentNode = self.graph.nodes[p]["data"]
                if p_data.observed_positive and not p_data.ground_truth:
                    parent_contaminated = True
                    break

            # If parent was contaminated, this node builds on a false premise
            # We model this as: the node's ground truth is corrupted
            if parent_contaminated:
                node.ground_truth = False

            # Run the experiment
            positive = node.simulate_experiment(self.rng)

            # Track contamination
            if positive and not node.ground_truth:
                node.contaminated = True
                result.contaminated_nodes += 1
                result.false_positives += 1
            elif positive and node.ground_truth:
                result.true_positives += 1
            elif not positive and not node.ground_truth:
                result.true_negatives += 1
            else:
                result.false_negatives += 1

            result.node_states[node_name] = {
                "positive": positive,
                "ground_truth": node.ground_truth,
                "contaminated": node.contaminated,
                "gated": False,
                "confidence": None,
            }

        return result

    def run_with_scoring(
        self, confidence_threshold: float = 0.4
    ) -> CascadeResult:
        """Simulate cascade WITH confidence scoring.

        Each experiment result gets a confidence score. Downstream experiments
        only use upstream results that pass the confidence threshold.
        If an upstream result is gated (below threshold), the downstream
        experiment treats it as if the upstream reported negative.
        """
        result = CascadeResult(total_nodes=self.graph.number_of_nodes())

        for node_name in nx.topological_sort(self.graph):
            node: ExperimentNode = self.graph.nodes[node_name]["data"]

            # Check parent confidence scores
            parents = list(self.graph.predecessors(node_name))
            parent_contaminated = False
            parent_gated = False

            for p in parents:
                p_data: ExperimentNode = self.graph.nodes[p]["data"]
                if p_data.gated:
                    # Parent was gated -- don't build on it
                    parent_gated = True
                    continue
                if p_data.observed_positive and not p_data.ground_truth:
                    # Parent is a false positive that passed the gate
                    parent_contaminated = True

            # If parent was gated, this node effectively starts fresh
            # (doesn't inherit the contaminated premise)
            if parent_gated:
                # Node may still be valid on its own merits; we don't
                # corrupt its ground truth
                pass
            elif parent_contaminated:
                node.ground_truth = False

            # Run the experiment
            positive = node.simulate_experiment(self.rng)

            # Generate confidence score
            evidence = node.simulate_reproductions(self.rng)
            score = self.scorer.score(evidence)
            node.confidence = score

            # Decision gate
            decision = self.decision_engine.should_use_as_premise(
                score, threshold=confidence_threshold
            )

            if positive and not decision.use:
                node.gated = True
                result.gated_nodes += 1

            # Track contamination
            if positive and not node.ground_truth and not node.gated:
                node.contaminated = True
                result.contaminated_nodes += 1
                result.false_positives += 1
            elif positive and node.ground_truth and not node.gated:
                result.true_positives += 1
            elif not positive or node.gated:
                if not node.ground_truth:
                    result.true_negatives += 1
                else:
                    result.false_negatives += 1

            result.node_states[node_name] = {
                "positive": positive,
                "ground_truth": node.ground_truth,
                "contaminated": node.contaminated,
                "gated": node.gated,
                "confidence": score.overall if score else None,
                "uncertainty": score.uncertainty if score else None,
            }

        return result
