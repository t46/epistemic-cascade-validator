"""Adapter for vanilla autoresearch pipeline output.

The vanilla autoresearch system (exp-2026-01-13-vanilla-autoresearch)
produces a structured JSON project file containing:
  - hypotheses: with status, confidence, estimated_novelty, testable_predictions
  - experiments: (empty in this dataset)
  - results: (empty in this dataset)
  - paper: draft sections with critiques

Mapping strategy:
  - Each hypothesis becomes an EvidencePacket
  - hypothesis.confidence -> reproduction proxy
  - hypothesis.estimated_novelty -> effect_size proxy
  - hypothesis.status (proposed/rejected) informs outcomes
  - testable_predictions count -> sample_size proxy
  - paper quality indicators -> code_quality proxy

Key limitation: This experiment produced hypotheses and a draft paper
but did not execute actual experiments or produce empirical results.
The confidence and novelty scores are self-assessed by the LLM, not
externally validated.
"""

from __future__ import annotations

import json
from pathlib import Path

from ecv.confidence import EvidencePacket
from ecv.adapters.base import BaseAdapter, AdaptedResult


def _parse_project_json(json_path: Path) -> list[dict]:
    """Parse the project JSON and extract hypothesis data."""
    text = json_path.read_text(encoding="utf-8", errors="replace")
    data = json.loads(text)

    hypotheses = []
    for h in data.get("hypotheses", []):
        hypotheses.append({
            "id": h.get("id", "unknown"),
            "statement": h.get("statement", ""),
            "rationale": h.get("rationale", ""),
            "status": h.get("status", "proposed"),
            "confidence": h.get("confidence", 0.5),
            "estimated_novelty": h.get("estimated_novelty", 0.5),
            "testable_predictions": h.get("testable_predictions", []),
            "required_resources": h.get("required_resources", []),
        })

    # Also extract paper-level metadata
    paper = data.get("paper", {})
    paper_meta = {
        "title": paper.get("title", ""),
        "has_abstract": bool(paper.get("abstract", "")),
        "num_sections": len(paper.get("sections", [])),
        "has_experiments": len(data.get("experiments", [])) > 0,
        "has_results": len(data.get("results", [])) > 0,
        "project_phase": data.get("phase", "unknown"),
        "research_area": data.get("research_area", ""),
    }

    return hypotheses, paper_meta


def _hypothesis_to_evidence(
    hyp: dict, paper_meta: dict,
) -> tuple[EvidencePacket, list[str]]:
    """Convert a hypothesis to an EvidencePacket."""
    notes: list[str] = []

    confidence = hyp["confidence"]
    novelty = hyp["estimated_novelty"]
    status = hyp["status"]
    n_predictions = len(hyp["testable_predictions"])

    # --- Reproduction ---
    # Status matters: rejected hypotheses have 0 successes.
    # Proposed hypotheses use their confidence as success probability.
    reproduction_attempts = max(3, n_predictions)
    if status == "rejected":
        reproduction_successes = 0
        notes.append(
            f"reproduction: hypothesis rejected, 0/{reproduction_attempts}"
        )
    else:
        reproduction_successes = round(confidence * reproduction_attempts)
        notes.append(
            f"reproduction: confidence={confidence:.2f} "
            f"-> {reproduction_successes}/{reproduction_attempts}"
        )

    # --- Effect size ---
    # estimated_novelty is our best proxy: novel results tend to have
    # larger effects (higher novelty = more deviation from baseline).
    # Modulated by confidence.
    effect_size = novelty * confidence * 1.0
    notes.append(
        f"effect_size: novelty={novelty:.2f} * confidence={confidence:.2f} "
        f"-> d={effect_size:.3f}"
    )

    # --- p-value ---
    # Confidence and number of testable predictions inform statistical
    # strength. More predictions = more ways to test = stronger evidence
    # potential.
    import math
    if status == "rejected":
        p_value = 0.8  # rejected -> weak evidence
        notes.append("p_value: rejected hypothesis -> p=0.80")
    else:
        exponent = confidence * (1 + n_predictions * 0.3)
        p_value = min(1.0, max(1e-10, 10 ** (-exponent)))
        notes.append(
            f"p_value: confidence * (1 + {n_predictions} predictions * 0.3) "
            f"-> p={p_value:.4f}"
        )

    # --- Sample size ---
    # Number of testable predictions * 10 as proxy.
    sample_size = max(10, n_predictions * 10)
    notes.append(f"sample_size: {n_predictions} predictions * 10 -> n={sample_size}")

    # --- Code quality ---
    # Based on paper completeness indicators.
    code_has_tests = paper_meta["has_experiments"]
    code_test_coverage = 0.0
    if paper_meta["has_abstract"]:
        code_test_coverage += 0.2
    if paper_meta["num_sections"] > 0:
        code_test_coverage += min(0.3, paper_meta["num_sections"] * 0.1)
    if paper_meta["has_results"]:
        code_test_coverage += 0.3
    if paper_meta["has_experiments"]:
        code_test_coverage += 0.2

    code_complexity = 0.5
    if len(hyp.get("required_resources", [])) > 0:
        # Detailed resource requirements suggest more complex setup
        code_complexity = 0.6
    notes.append(
        f"code_quality: tests={code_has_tests}, "
        f"coverage={code_test_coverage:.2f}, "
        f"complexity={code_complexity:.2f}"
    )

    packet = EvidencePacket(
        reproduction_successes=reproduction_successes,
        reproduction_attempts=reproduction_attempts,
        effect_size=effect_size,
        p_value=p_value,
        sample_size=sample_size,
        code_passes_lint=True,
        code_has_tests=code_has_tests,
        code_test_coverage=min(1.0, code_test_coverage),
        code_complexity_score=code_complexity,
    )

    return packet, notes


class VanillaAutoresearchAdapter(BaseAdapter):
    """Adapter for the vanilla autoresearch hypothesis pipeline."""

    def source_name(self) -> str:
        return "vanilla-autoresearch"

    def load(self, path: Path) -> list[AdaptedResult]:
        """Load project JSON from the vanilla autoresearch workspace.

        Args:
            path: Path to the experiment root directory.
        """
        results = []
        path = Path(path)

        # Find project JSON files
        storage_dir = path / "workspace" / "storage" / "projects"
        if not storage_dir.exists():
            return results

        for json_path in sorted(storage_dir.glob("*.json")):
            try:
                hypotheses, paper_meta = _parse_project_json(json_path)
            except (json.JSONDecodeError, KeyError) as e:
                continue

            project_id = json_path.stem

            for hyp in hypotheses:
                packet, notes = _hypothesis_to_evidence(hyp, paper_meta)

                result = AdaptedResult(
                    experiment_id=f"{project_id}/{hyp['id']}",
                    source_path=str(json_path),
                    evidence=packet,
                    raw_scores={
                        "confidence": hyp["confidence"],
                        "estimated_novelty": hyp["estimated_novelty"],
                        "status": hyp["status"],
                        "n_predictions": len(hyp["testable_predictions"]),
                        "project_phase": paper_meta["project_phase"],
                    },
                    description=(
                        f"[{hyp['status']}] {hyp['statement'][:100]}"
                    ),
                    mapping_notes=notes,
                )
                results.append(result)

        return results

    def build_hypothesis_chain(
        self, results: list[AdaptedResult],
    ) -> list[tuple[str, str]]:
        """Build dependency edges between hypotheses.

        The vanilla autoresearch groups hypotheses by prefix (comb_, cons_,
        cont_). Within each group, hypotheses are sequential explorations
        of the same idea space, creating implicit dependencies.
        """
        # Group by prefix
        groups: dict[str, list[str]] = {}
        for r in results:
            hyp_id = r.experiment_id.split("/")[-1]
            prefix = hyp_id.split("_")[0] if "_" in hyp_id else hyp_id
            groups.setdefault(prefix, []).append(r.experiment_id)

        edges = []
        for prefix, ids in groups.items():
            # Sort by suffix number
            sorted_ids = sorted(ids)
            for i in range(len(sorted_ids) - 1):
                edges.append((sorted_ids[i], sorted_ids[i + 1]))

        return edges
