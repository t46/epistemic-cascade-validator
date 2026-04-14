"""Adapter for auto-research-evaluator pipeline output.

The auto-research-evaluator runs a multi-phase analysis on autoresearch-
generated papers. Each experiment directory (exp-YYYY-MM-DD[-suffix])
contains markdown artifacts with structured evaluation scores.

Mapping strategy:
  - reproduction_successes/attempts: Derived from reproducibility scores
    found in results_analysis.md and experiment_assessment.md. The evaluator
    rates reproducibility on a 1-10 scale; we convert to a simulated
    reproduction trial outcome.
  - effect_size: Derived from the "statistical validity" or "claims support"
    scores. Higher claim support suggests larger observable effects.
  - p_value: Derived from statistical rigor scores. Papers with strong
    statistical reporting get lower (better) p-values.
  - sample_size: Approximated from the evaluation detail (number of claims
    evaluated, number of phases completed).
  - code_quality: Derived from repository analysis presence and
    reproducibility scores.

Key limitation: These are meta-evaluations (evaluation OF papers), not
direct experimental data. The EvidencePacket fields are proxies, not
ground truth. This is documented honestly in the results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ecv.confidence import EvidencePacket
from ecv.adapters.base import BaseAdapter, AdaptedResult


@dataclass
class EvaluatorScores:
    """Scores extracted from evaluator markdown files."""

    experiment_id: str = ""
    paper_title: str = ""
    # Scores on 0-10 scale (or -1 if not found)
    overall_score: float = -1.0
    methodology_score: float = -1.0
    statistical_validity_score: float = -1.0
    reproducibility_score: float = -1.0
    claims_support_score: float = -1.0
    results_score: float = -1.0
    novelty_score: float = -1.0
    # Metadata
    has_repo_analysis: bool = False
    has_code_consistency: bool = False
    has_critical_review: bool = False
    has_comparative_analysis: bool = False
    num_phases_completed: int = 0
    # Raw text snippets for provenance
    score_sources: dict[str, str] = None

    def __post_init__(self):
        if self.score_sources is None:
            self.score_sources = {}


def _extract_scores_from_text(text: str) -> list[float]:
    """Extract X/10 scores from markdown text."""
    pattern = r'(\d+\.?\d*)\s*/\s*10'
    return [float(m) for m in re.findall(pattern, text)]


def _extract_first_score(text: str, patterns: list[str]) -> float:
    """Search for score patterns and return the first match."""
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    return -1.0


def _parse_experiment_dir(exp_dir: Path) -> EvaluatorScores:
    """Parse all artifacts in an experiment directory to extract scores."""
    scores = EvaluatorScores(experiment_id=exp_dir.name)
    artifacts = exp_dir / "artifacts"
    if not artifacts.exists():
        return scores

    # Count completed phases
    phase_files = list(artifacts.glob("phase*_completion_summary.md"))
    phase_files.extend(artifacts.glob("phase*-*_completion_summary.md"))
    scores.num_phases_completed = len(phase_files)

    # Check for presence of detailed analysis files
    scores.has_repo_analysis = (artifacts / "repository_analysis.md").exists()
    scores.has_code_consistency = (artifacts / "paper_code_consistency.md").exists()
    scores.has_critical_review = (artifacts / "critical_review.md").exists()
    scores.has_comparative_analysis = (artifacts / "comparative_analysis.md").exists()

    # --- comprehensive_evaluation_report.md ---
    report_path = artifacts / "comprehensive_evaluation_report.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8", errors="replace")
        scores.score_sources["comprehensive_evaluation_report"] = text[:500]

        # Extract paper title
        title_match = re.search(
            r'\*\*(?:Title|Paper Title|タイトル)\*\*[:\s]*(.+)',
            text, re.IGNORECASE,
        )
        if title_match:
            scores.paper_title = title_match.group(1).strip()

        # Overall / total score
        overall = _extract_first_score(text, [
            r'総合(?:評価)?スコア[:\s]*(\d+\.?\d*)\s*/\s*10',
            r'[Oo]verall\s+(?:[Ss]core|[Rr]ating|[Aa]ssessment)[:\s]*(\d+\.?\d*)\s*/\s*10',
            r'[Tt]otal\s+[Ss]core[:\s]*(\d+\.?\d*)\s*/\s*10',
            r'(\d+\.?\d*)\s*/\s*10\s*[（(](?:良好|overall|total)',
        ])
        if overall >= 0:
            scores.overall_score = overall

        # If no explicit overall, take the first X/10 score
        if scores.overall_score < 0:
            all_scores = _extract_scores_from_text(text)
            # Filter out scores that are clearly sub-components (like 6.33/10
            # which appears in AI-Scientist papers as reviewer scores)
            candidates = [s for s in all_scores if 1.0 <= s <= 10.0]
            if candidates:
                scores.overall_score = candidates[0]

    # --- methodology_evaluation.md ---
    meth_path = artifacts / "methodology_evaluation.md"
    if meth_path.exists():
        text = meth_path.read_text(encoding="utf-8", errors="replace")
        meth = _extract_first_score(text, [
            r'[Mm]ethodology\s+[Qq]uality[:\s]*(\d+\.?\d*)\s*/\s*10',
            r'方法論.*?(\d+\.?\d*)\s*/\s*10',
            r'[Oo]verall\s+[Mm]ethodology[:\s]*(\d+\.?\d*)\s*/\s*10',
        ])
        if meth >= 0:
            scores.methodology_score = meth

    # --- results_analysis.md ---
    results_path = artifacts / "results_analysis.md"
    if results_path.exists():
        text = results_path.read_text(encoding="utf-8", errors="replace")

        stat = _extract_first_score(text, [
            r'統計的妥当性[:\s]*(\d+\.?\d*)\s*/\s*10',
            r'[Ss]tatistical\s+(?:[Vv]alidity|[Rr]igor)[:\s]*(\d+\.?\d*)\s*/\s*10',
            # Table format: | 統計的妥当性 | 4/10 | ...
            r'統計的妥当性\s*\|\s*(\d+\.?\d*)\s*/\s*10',
            r'[Ss]tatistical\s+(?:validity|significance|rigor)\s*\|\s*(\d+\.?\d*)\s*/\s*10',
        ])
        if stat >= 0:
            scores.statistical_validity_score = stat

        repro = _extract_first_score(text, [
            r'再現性[:\s]*(\d+\.?\d*)\s*/\s*10',
            r'[Rr]eproducibility[:\s]*(\d+\.?\d*)\s*/\s*10',
            # Table format: | 再現性 | 7.5/10 | ...
            r'再現性\s*\|\s*(\d+\.?\d*)\s*/\s*10',
            r'[Rr]eproducibility\s*\|\s*(\d+\.?\d*)\s*/\s*10',
        ])
        if repro >= 0:
            scores.reproducibility_score = repro

        claims = _extract_first_score(text, [
            r'主張.*?支持度[:\s]*(\d+\.?\d*)\s*/\s*10',
            r'[Cc]laim.*?[Ss]upport[:\s]*(\d+\.?\d*)\s*/\s*10',
            # Table format
            r'主張の支持度\s*\|\s*(\d+\.?\d*)\s*/\s*10',
        ])
        if claims >= 0:
            scores.claims_support_score = claims

        results = _extract_first_score(text, [
            r'結果分析.*?総合スコア[:\s]*(\d+\.?\d*)\s*/\s*10',
            r'[Rr]esults?.*?[Ss]core[:\s]*(\d+\.?\d*)\s*/\s*10',
            # Table format
            r'総合スコア[:\s]*(\d+\.?\d*)\s*/\s*10',
        ])
        if results >= 0:
            scores.results_score = results

    # --- experiment_assessment.md ---
    assess_path = artifacts / "experiment_assessment.md"
    if assess_path.exists():
        text = assess_path.read_text(encoding="utf-8", errors="replace")

        # Reproducibility from assessment
        if scores.reproducibility_score < 0:
            repro = _extract_first_score(text, [
                r'[Rr]eproducibility[:\s]*(\d+\.?\d*)\s*/\s*10',
            ])
            if repro >= 0:
                scores.reproducibility_score = repro

        # Statistical validity from assessment
        if scores.statistical_validity_score < 0:
            stat = _extract_first_score(text, [
                r'[Ss]tatistical\s+(?:validity|significance|rigor)[:\s]*(\d+\.?\d*)\s*/\s*10',
            ])
            if stat >= 0:
                scores.statistical_validity_score = stat

    return scores


def _scores_to_evidence(scores: EvaluatorScores) -> tuple[EvidencePacket, list[str]]:
    """Convert evaluator scores to an EvidencePacket.

    Returns (packet, mapping_notes) where mapping_notes documents
    approximations and missing data.
    """
    notes: list[str] = []

    # --- Reproduction ---
    # The evaluator does not run actual reproductions. We approximate
    # using the reproducibility score: a 9/10 reproducibility score
    # maps to ~4/5 successful reproductions; a 3/10 maps to ~1/5.
    repro_score = scores.reproducibility_score
    if repro_score < 0:
        # Fallback: use overall score as rough proxy
        repro_score = scores.overall_score if scores.overall_score >= 0 else 5.0
        notes.append("reproducibility_score missing; used overall_score as proxy")

    # Map 0-10 score to reproduction outcomes (out of 5 trials)
    repro_rate = max(0.0, min(1.0, repro_score / 10.0))
    reproduction_attempts = 5
    reproduction_successes = round(repro_rate * reproduction_attempts)
    notes.append(
        f"reproduction derived from reproducibility_score={repro_score:.1f}/10 "
        f"-> {reproduction_successes}/{reproduction_attempts}"
    )

    # --- Effect size ---
    # Claims support score is our best proxy for effect size: if the paper's
    # claims are well-supported, the effects are presumably observable.
    claims = scores.claims_support_score
    if claims < 0:
        claims = scores.results_score if scores.results_score >= 0 else 5.0
        notes.append("claims_support_score missing; used results_score as proxy")

    # Map 0-10 to Cohen's d range (0.0 - 1.2)
    effect_size = max(0.0, (claims / 10.0) * 1.2)
    notes.append(
        f"effect_size derived from claims_support={claims:.1f}/10 -> d={effect_size:.3f}"
    )

    # --- p-value ---
    # Statistical validity score maps to p-value: high rigor -> low p.
    stat = scores.statistical_validity_score
    if stat < 0:
        stat = 5.0  # neutral prior
        notes.append("statistical_validity_score missing; defaulted to 5.0/10")

    # Map: 10/10 -> p=0.001, 5/10 -> p=0.05, 1/10 -> p=0.5
    # Using exponential mapping: p = 10^(-0.3 * stat)
    import math
    p_value = min(1.0, max(1e-10, 10 ** (-0.3 * stat)))
    notes.append(
        f"p_value derived from statistical_validity={stat:.1f}/10 -> p={p_value:.4f}"
    )

    # --- Sample size ---
    # Number of completed phases is a proxy for evidence depth.
    # More phases = more thorough evaluation = larger effective sample.
    n_phases = max(1, scores.num_phases_completed)
    sample_size = n_phases * 5  # each phase ~ 5 data points
    notes.append(
        f"sample_size derived from num_phases={n_phases} -> n={sample_size}"
    )

    # --- Code quality ---
    code_has_tests = scores.has_repo_analysis
    code_test_coverage = 0.0
    code_passes_lint = True  # assume true (evaluator doesn't report lint)
    code_complexity = 0.5

    if scores.has_code_consistency:
        code_test_coverage += 0.3
        notes.append("code_test_coverage +0.3 for paper_code_consistency analysis")
    if scores.has_repo_analysis:
        code_test_coverage += 0.2
        notes.append("code_test_coverage +0.2 for repository_analysis present")
    if scores.has_critical_review:
        code_test_coverage += 0.1
        notes.append("code_test_coverage +0.1 for critical_review present")
    if scores.has_comparative_analysis:
        code_test_coverage += 0.1
        notes.append("code_test_coverage +0.1 for comparative_analysis present")

    # Methodology score affects complexity assessment
    meth = scores.methodology_score
    if meth >= 0:
        # Higher methodology -> lower complexity (well-structured)
        code_complexity = max(0.1, 1.0 - (meth / 10.0) * 0.8)
        notes.append(
            f"code_complexity derived from methodology_score={meth:.1f}/10 "
            f"-> complexity={code_complexity:.2f}"
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


class AutoResearchEvaluatorAdapter(BaseAdapter):
    """Adapter for the auto-research-evaluator experiment outputs."""

    def source_name(self) -> str:
        return "auto-research-evaluator"

    def load(self, path: Path) -> list[AdaptedResult]:
        """Load all experiment directories from the evaluator root.

        Skips template directories and experiments without artifacts.
        """
        results = []
        path = Path(path)

        for exp_dir in sorted(path.iterdir()):
            if not exp_dir.is_dir():
                continue
            if "template" in exp_dir.name:
                continue
            if not (exp_dir / "artifacts").exists():
                continue

            scores = _parse_experiment_dir(exp_dir)

            # Skip experiments where we couldn't extract any meaningful score
            if scores.overall_score < 0 and scores.methodology_score < 0:
                # Try to get at least some data from phase completions
                if scores.num_phases_completed == 0:
                    continue

            packet, notes = _scores_to_evidence(scores)

            result = AdaptedResult(
                experiment_id=scores.experiment_id,
                source_path=str(exp_dir),
                evidence=packet,
                raw_scores={
                    "overall": scores.overall_score,
                    "methodology": scores.methodology_score,
                    "statistical_validity": scores.statistical_validity_score,
                    "reproducibility": scores.reproducibility_score,
                    "claims_support": scores.claims_support_score,
                    "results": scores.results_score,
                    "novelty": scores.novelty_score,
                    "num_phases": scores.num_phases_completed,
                    "has_repo_analysis": scores.has_repo_analysis,
                    "has_code_consistency": scores.has_code_consistency,
                },
                description=(
                    f"Paper: {scores.paper_title or 'unknown'} "
                    f"(overall={scores.overall_score}/10)"
                ),
                mapping_notes=notes,
            )
            results.append(result)

        return results
