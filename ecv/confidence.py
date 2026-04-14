"""Confidence Score Model.

Bayesian confidence scoring for experimental results. Combines multiple
evidence signals (reproduction trials, effect size, p-value, code quality)
into a single 0.0-1.0 confidence score with uncertainty breakdown.

Design philosophy:
  A "keep" result in autoresearch is only as good as its reproducibility.
  A false positive that propagates downstream is worse than a discard,
  because subsequent experiments build on it. This scorer quantifies how
  much to trust each result, enabling downstream consumers to treat prior
  outputs as uncertain premises rather than ground truth.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class ConfidenceScore:
    """Composite confidence score for an experimental result."""

    overall: float  # 0.0 - 1.0
    reproduction_component: float  # based on replication trials
    effect_size_component: float  # based on observed effect magnitude
    statistical_component: float  # based on p-value / Bayes factor
    code_quality_component: float  # based on code quality metrics
    uncertainty: float  # standard deviation of the posterior

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def __repr__(self) -> str:
        return (
            f"ConfidenceScore(overall={self.overall:.3f}, "
            f"uncertainty={self.uncertainty:.3f})"
        )


@dataclass
class EvidencePacket:
    """Evidence packet for a single experimental result."""

    reproduction_successes: int = 0  # number of successful reproductions
    reproduction_attempts: int = 0  # total reproduction attempts
    effect_size: float = 0.0  # Cohen's d or equivalent
    p_value: float = 1.0  # observed p-value
    sample_size: int = 1  # sample size of the experiment
    code_passes_lint: bool = True
    code_has_tests: bool = False
    code_test_coverage: float = 0.0  # 0.0 - 1.0
    code_complexity_score: float = 0.5  # 0.0 (simple) - 1.0 (complex)


@dataclass
class ConfidenceScorer:
    """Bayesian confidence scorer.

    Uses Beta-Binomial model for reproduction, and transforms other signals
    via calibrated sigmoid mappings. The overall score is a weighted product
    that penalizes weak components.
    """

    # Component weights (must sum to 1.0)
    w_reproduction: float = 0.40
    w_effect_size: float = 0.20
    w_statistical: float = 0.25
    w_code_quality: float = 0.15

    # Beta prior parameters for reproduction (initially weakly informative)
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def score(self, evidence: EvidencePacket) -> ConfidenceScore:
        """Compute confidence score from an evidence packet."""
        repro = self._score_reproduction(evidence)
        effect = self._score_effect_size(evidence)
        stat = self._score_statistical(evidence)
        code = self._score_code_quality(evidence)

        # Weighted geometric mean -- penalizes any single weak component
        # more harshly than arithmetic mean.
        components = np.array([repro, effect, stat, code])
        weights = np.array([
            self.w_reproduction,
            self.w_effect_size,
            self.w_statistical,
            self.w_code_quality,
        ])

        # Avoid log(0) by clamping
        components_clamped = np.clip(components, 1e-10, 1.0)
        log_overall = np.sum(weights * np.log(components_clamped))
        overall = float(np.exp(log_overall))

        # Uncertainty from posterior on reproduction (dominant signal)
        uncertainty = self._reproduction_uncertainty(evidence)

        return ConfidenceScore(
            overall=overall,
            reproduction_component=repro,
            effect_size_component=effect,
            statistical_component=stat,
            code_quality_component=code,
            uncertainty=uncertainty,
        )

    def update_prior(self, successes: int, failures: int) -> None:
        """Bayesian update of the reproduction prior."""
        self.prior_alpha += successes
        self.prior_beta += failures

    def _score_reproduction(self, ev: EvidencePacket) -> float:
        """Beta-Binomial posterior mean for reproduction rate."""
        if ev.reproduction_attempts == 0:
            # No reproduction data -> use prior, which starts at 0.5
            # but we penalize lack of evidence
            return 0.3  # low confidence without any reproduction

        alpha_post = self.prior_alpha + ev.reproduction_successes
        beta_post = self.prior_beta + (
            ev.reproduction_attempts - ev.reproduction_successes
        )
        return float(alpha_post / (alpha_post + beta_post))

    def _reproduction_uncertainty(self, ev: EvidencePacket) -> float:
        """Standard deviation of the Beta posterior on reproduction rate."""
        if ev.reproduction_attempts == 0:
            alpha_post = self.prior_alpha
            beta_post = self.prior_beta
        else:
            alpha_post = self.prior_alpha + ev.reproduction_successes
            beta_post = self.prior_beta + (
                ev.reproduction_attempts - ev.reproduction_successes
            )

        variance = (alpha_post * beta_post) / (
            (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)
        )
        return float(np.sqrt(variance))

    def _score_effect_size(self, ev: EvidencePacket) -> float:
        """Map effect size to confidence via calibrated sigmoid.

        Small effects (d < 0.2) get low confidence.
        Medium effects (d ~ 0.5) get moderate confidence.
        Large effects (d > 0.8) get high confidence.
        """
        d = abs(ev.effect_size)
        # Sigmoid centered at d=0.3, steepness=6
        return float(1.0 / (1.0 + np.exp(-6.0 * (d - 0.3))))

    def _score_statistical(self, ev: EvidencePacket) -> float:
        """Transform p-value into confidence.

        Uses -log10(p) mapped through a sigmoid.
        Also factors in sample size: larger N -> more credible.
        """
        # p-value component: -log10(p), clamped
        log_p = -np.log10(max(ev.p_value, 1e-20))
        # Sigmoid: midpoint at -log10(0.05) ~ 1.3
        p_score = float(1.0 / (1.0 + np.exp(-2.0 * (log_p - 1.3))))

        # Sample size adjustment: penalize very small N
        n_factor = float(1.0 - np.exp(-ev.sample_size / 30.0))

        return p_score * 0.7 + n_factor * 0.3

    def _score_code_quality(self, ev: EvidencePacket) -> float:
        """Heuristic code quality score."""
        score = 0.3  # base score for having code at all

        if ev.code_passes_lint:
            score += 0.2
        if ev.code_has_tests:
            score += 0.2
        # Coverage contribution (up to 0.2)
        score += 0.2 * ev.code_test_coverage
        # Penalize high complexity
        score += 0.1 * (1.0 - ev.code_complexity_score)

        return min(score, 1.0)
