"""Epistemic Cascade Validator (ECV).

A confidence scoring system that detects and prevents epistemic cascade
contamination in autonomous research pipelines.
"""

from ecv.confidence import ConfidenceScore, ConfidenceScorer
from ecv.cascade import CascadeSimulator, ExperimentNode
from ecv.decision import DecisionEngine

__all__ = [
    "ConfidenceScore",
    "ConfidenceScorer",
    "CascadeSimulator",
    "ExperimentNode",
    "DecisionEngine",
]
