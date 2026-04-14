"""Adapter for Autonomous Research Agent (ARA) belief store.

The ARA maintains a structured belief store with:
  - ledger/ledger.jsonl: Action log (orient, literature-search, belief-update)
  - beliefs/claims/*.yaml: Claims with confidence, evidence, and dependencies

Mapping strategy:
  - Each claim becomes an EvidencePacket
  - confidence (0-1) maps to reproduction rate proxy
  - evidence count and fidelity levels inform sample_size and effect_size
  - depends_on relationships enable cascade graph construction
  - evidence direction (for/against) affects p_value proxy

Key limitation: ARA data is from a single literature-review session with
only abstract-level reads (F1 fidelity). No actual experiments were run,
so all EvidencePacket fields are rough proxies of epistemic confidence
rather than empirical measurements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ecv.confidence import EvidencePacket
from ecv.adapters.base import BaseAdapter, AdaptedResult

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass
class ARAClaim:
    """Parsed ARA claim."""

    claim_id: str = ""
    claim_text: str = ""
    status: str = "active"
    confidence: float = 0.5
    evidence_for: int = 0
    evidence_against: int = 0
    evidence_total: int = 0
    max_fidelity: str = "F0"
    depends_on: list[str] = field(default_factory=list)
    related_claims: list[str] = field(default_factory=list)


@dataclass
class ARALedgerEntry:
    """Parsed ARA ledger entry."""

    timestamp: str = ""
    session: str = ""
    action: str = ""
    detail: str = ""


def _parse_yaml_claim(claim_path: Path) -> ARAClaim | None:
    """Parse a single claim YAML file."""
    if yaml is None:
        # Fallback: basic parsing without PyYAML
        return _parse_yaml_claim_fallback(claim_path)

    text = claim_path.read_text(encoding="utf-8", errors="replace")
    try:
        data = yaml.safe_load(text)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    claim = ARAClaim(
        claim_id=claim_path.stem,
        claim_text=data.get("claim", ""),
        status=data.get("status", "active"),
        confidence=float(data.get("confidence", 0.5)),
        depends_on=data.get("depends_on", []) or [],
        related_claims=data.get("related_claims", []) or [],
    )

    evidence_list = data.get("evidence", []) or []
    for ev in evidence_list:
        if not isinstance(ev, dict):
            continue
        direction = ev.get("direction", "")
        if direction == "for":
            claim.evidence_for += 1
        elif direction == "against":
            claim.evidence_against += 1
        claim.evidence_total += 1

        fidelity = ev.get("fidelity", "F0")
        # F0 < F1 < F2 < F3 (increasing fidelity)
        if fidelity > claim.max_fidelity:
            claim.max_fidelity = fidelity

    return claim


def _parse_yaml_claim_fallback(claim_path: Path) -> ARAClaim | None:
    """Parse claim YAML without PyYAML (regex-based)."""
    import re
    text = claim_path.read_text(encoding="utf-8", errors="replace")

    claim = ARAClaim(claim_id=claim_path.stem)

    # Extract confidence
    conf_match = re.search(r'^confidence:\s*([\d.]+)', text, re.MULTILINE)
    if conf_match:
        claim.confidence = float(conf_match.group(1))

    # Extract status
    status_match = re.search(r'^status:\s*(\w+)', text, re.MULTILINE)
    if status_match:
        claim.status = status_match.group(1)

    # Extract claim text
    claim_match = re.search(r'^claim:\s*"(.+?)"', text, re.MULTILINE | re.DOTALL)
    if claim_match:
        claim.claim_text = claim_match.group(1)[:200]

    # Count evidence directions
    claim.evidence_for = len(re.findall(r'direction:\s*for', text))
    claim.evidence_against = len(re.findall(r'direction:\s*against', text))
    claim.evidence_total = claim.evidence_for + claim.evidence_against

    # Extract fidelity levels
    fidelities = re.findall(r'fidelity:\s*"?(F\d)"?', text)
    if fidelities:
        claim.max_fidelity = max(fidelities)

    # Extract depends_on
    deps_match = re.search(r'depends_on:\s*\[(.+?)\]', text)
    if deps_match:
        claim.depends_on = [
            d.strip().strip('"').strip("'")
            for d in deps_match.group(1).split(",")
            if d.strip()
        ]
    else:
        # Check for list-style depends_on
        deps = re.findall(r'depends_on:.*?\n((?:\s*-\s*.+\n)*)', text)
        if deps:
            claim.depends_on = [
                d.strip().strip('"').strip("'").lstrip("- ")
                for d in deps[0].strip().split("\n")
                if d.strip()
            ]

    return claim


def _claim_to_evidence(claim: ARAClaim) -> tuple[EvidencePacket, list[str]]:
    """Convert an ARA claim to EvidencePacket."""
    notes: list[str] = []

    # --- Reproduction ---
    # Confidence maps to reproduction success rate.
    # We model as if there were N "consistency checks" with the evidence.
    reproduction_attempts = max(1, claim.evidence_total + 1)
    reproduction_successes = round(claim.confidence * reproduction_attempts)
    notes.append(
        f"reproduction from confidence={claim.confidence:.2f} "
        f"and evidence_count={claim.evidence_total} "
        f"-> {reproduction_successes}/{reproduction_attempts}"
    )

    # --- Effect size ---
    # More supporting evidence with less contradiction -> larger effect.
    if claim.evidence_total > 0:
        support_ratio = claim.evidence_for / claim.evidence_total
    else:
        support_ratio = 0.5
    effect_size = support_ratio * 0.8 * claim.confidence
    notes.append(
        f"effect_size from support_ratio={support_ratio:.2f} * confidence "
        f"-> d={effect_size:.3f}"
    )

    # --- p-value ---
    # Higher confidence with more evidence -> lower p-value.
    # F0 (no read) is weak; F1 (abstract) is moderate; F2+ is strong.
    fidelity_weight = {"F0": 0.2, "F1": 0.5, "F2": 0.8, "F3": 1.0}
    fw = fidelity_weight.get(claim.max_fidelity, 0.2)
    # p = 10^(-(confidence * fidelity_weight * evidence_count * 0.5))
    import math
    exponent = claim.confidence * fw * max(1, claim.evidence_total) * 0.5
    p_value = min(1.0, max(1e-10, 10 ** (-exponent)))
    notes.append(
        f"p_value from confidence*fidelity*evidence "
        f"(fidelity={claim.max_fidelity}, fw={fw}) -> p={p_value:.4f}"
    )

    # --- Sample size ---
    # Number of evidence items is the closest analog to sample size.
    sample_size = max(1, claim.evidence_total * 10)
    notes.append(f"sample_size from evidence_count*10 -> n={sample_size}")

    # --- Code quality ---
    # ARA is a literature review agent -- no code produced.
    code_passes_lint = True
    code_has_tests = False
    code_test_coverage = 0.0
    code_complexity = 0.3
    notes.append("code_quality: N/A for literature-review claims, using defaults")

    packet = EvidencePacket(
        reproduction_successes=reproduction_successes,
        reproduction_attempts=reproduction_attempts,
        effect_size=effect_size,
        p_value=p_value,
        sample_size=sample_size,
        code_passes_lint=code_passes_lint,
        code_has_tests=code_has_tests,
        code_test_coverage=code_test_coverage,
        code_complexity_score=code_complexity,
    )

    return packet, notes


class ARAAdapter(BaseAdapter):
    """Adapter for the Autonomous Research Agent belief store."""

    def source_name(self) -> str:
        return "autonomous-research-agent"

    def load(self, path: Path) -> list[AdaptedResult]:
        """Load claims from the ARA research workspace.

        Args:
            path: Path to the research-workspace/ directory.
        """
        results = []
        path = Path(path)

        claims_dir = path / "beliefs" / "claims"
        if not claims_dir.exists():
            return results

        for claim_path in sorted(claims_dir.glob("*.yaml")):
            claim = _parse_yaml_claim(claim_path)
            if claim is None:
                continue

            packet, notes = _claim_to_evidence(claim)

            result = AdaptedResult(
                experiment_id=claim.claim_id,
                source_path=str(claim_path),
                evidence=packet,
                raw_scores={
                    "confidence": claim.confidence,
                    "evidence_for": claim.evidence_for,
                    "evidence_against": claim.evidence_against,
                    "evidence_total": claim.evidence_total,
                    "max_fidelity": claim.max_fidelity,
                    "status": claim.status,
                    "depends_on": claim.depends_on,
                    "related_claims": claim.related_claims,
                },
                description=claim.claim_text[:120],
                mapping_notes=notes,
            )
            results.append(result)

        return results

    def load_ledger(self, path: Path) -> list[ARALedgerEntry]:
        """Load the action ledger for provenance analysis."""
        ledger_path = Path(path) / "ledger" / "ledger.jsonl"
        entries = []
        if not ledger_path.exists():
            return entries

        for line in ledger_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                entries.append(ARALedgerEntry(
                    timestamp=data.get("ts", ""),
                    session=data.get("session", ""),
                    action=data.get("action", ""),
                    detail=data.get("detail", ""),
                ))
            except json.JSONDecodeError:
                continue

        return entries

    def build_dependency_graph(
        self, results: list[AdaptedResult]
    ) -> list[tuple[str, str]]:
        """Extract claim dependency edges for cascade analysis.

        Returns list of (upstream_id, downstream_id) edges.
        """
        edges = []
        for r in results:
            deps = r.raw_scores.get("depends_on", [])
            for dep in deps:
                edges.append((dep, r.experiment_id))
        return edges
