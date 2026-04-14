"""Decision Engine.

Confidence-aware decision logic for autonomous research pipelines.
Answers the question: "Should I use this upstream result as a premise?"

Design:
  - Bayesian updating: downstream inherits and compounds uncertainty
  - Threshold-based gating with configurable risk tolerance
  - Provides audit trail for why a result was accepted or rejected
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ecv.confidence import ConfidenceScore


@dataclass
class Decision:
    """Decision on whether to use an upstream result."""

    use: bool
    reason: str
    adjusted_confidence: float  # after compounding upstream uncertainty
    risk_level: str  # "low", "medium", "high", "critical"

    def __repr__(self) -> str:
        return (
            f"Decision(use={self.use}, risk={self.risk_level}, "
            f"adj_conf={self.adjusted_confidence:.3f}, reason='{self.reason}')"
        )


class DecisionEngine:
    """Confidence-aware premise gating.

    Computes adjusted confidence by compounding upstream uncertainty,
    then applies threshold-based gating.
    """

    def __init__(
        self,
        default_threshold: float = 0.4,
        risk_tolerance: str = "medium",
    ):
        self.default_threshold = default_threshold
        self.risk_tolerance = risk_tolerance

        # Risk tolerance -> threshold modifier
        self._tolerance_modifiers = {
            "low": 0.15,       # more conservative: higher threshold
            "medium": 0.0,     # default
            "high": -0.10,     # more aggressive: lower threshold
            "exploratory": -0.20,  # very aggressive for early-stage research
        }

    def should_use_as_premise(
        self,
        score: ConfidenceScore,
        upstream_scores: Optional[list[ConfidenceScore]] = None,
        threshold: Optional[float] = None,
    ) -> Decision:
        """Decide whether to use a result as a premise for downstream work.

        Args:
            score: Confidence score of the result in question.
            upstream_scores: Confidence scores of this result's upstream
                dependencies. Used to compound uncertainty.
            threshold: Override the default threshold.
        """
        # Apply risk tolerance modifier to threshold
        effective_threshold = (threshold or self.default_threshold) + \
            self._tolerance_modifiers.get(self.risk_tolerance, 0.0)
        effective_threshold = max(0.05, min(effective_threshold, 0.95))

        # Compound upstream uncertainty
        adjusted = self._compound_confidence(score, upstream_scores)

        # Classify risk level
        risk = self._classify_risk(adjusted)

        # Gate decision
        if adjusted >= effective_threshold:
            return Decision(
                use=True,
                reason=f"Confidence {adjusted:.3f} >= threshold {effective_threshold:.3f}",
                adjusted_confidence=adjusted,
                risk_level=risk,
            )
        else:
            return Decision(
                use=False,
                reason=f"Confidence {adjusted:.3f} < threshold {effective_threshold:.3f}",
                adjusted_confidence=adjusted,
                risk_level=risk,
            )

    def compound_chain_confidence(
        self, scores: list[ConfidenceScore]
    ) -> float:
        """Compute the compounded confidence of a chain of experiments.

        Each link in the chain multiplies the overall confidence, because
        the final result is only as strong as its weakest upstream premise.

        This captures the key insight: in a chain A->B->C, if A has 0.7
        confidence and B has 0.8, the effective confidence of C's premise
        is at most 0.7 * 0.8 = 0.56.
        """
        if not scores:
            return 1.0

        confidences = [s.overall for s in scores]
        # Product of confidences, with a floor
        compounded = float(np.prod(confidences))
        return max(compounded, 0.001)

    def _compound_confidence(
        self,
        score: ConfidenceScore,
        upstream_scores: Optional[list[ConfidenceScore]] = None,
    ) -> float:
        """Adjust confidence by compounding upstream uncertainty."""
        base = score.overall

        if not upstream_scores:
            return base

        # Multiply by the compounded upstream chain confidence
        upstream_compound = self.compound_chain_confidence(upstream_scores)
        return base * upstream_compound

    def _classify_risk(self, confidence: float) -> str:
        """Classify risk level based on adjusted confidence."""
        if confidence >= 0.7:
            return "low"
        elif confidence >= 0.4:
            return "medium"
        elif confidence >= 0.2:
            return "high"
        else:
            return "critical"
