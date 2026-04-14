"""Base adapter interface for real-data sources."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

from ecv.confidence import EvidencePacket


@dataclass
class AdaptedResult:
    """A real-world experimental result mapped to ECV format.

    Preserves provenance (source file, original scores) alongside
    the derived EvidencePacket so that analysis can trace back to
    the raw data.
    """

    experiment_id: str
    source_path: str
    evidence: EvidencePacket
    # Original scores/metadata from the source, for auditing
    raw_scores: dict = field(default_factory=dict)
    # Human-readable description
    description: str = ""
    # Mapping notes (what was approximated, what was missing)
    mapping_notes: list[str] = field(default_factory=list)


class BaseAdapter(abc.ABC):
    """Abstract base for data source adapters."""

    @abc.abstractmethod
    def load(self, path: Path) -> list[AdaptedResult]:
        """Load and convert all available results from the data source.

        Args:
            path: Root directory of the data source.

        Returns:
            List of AdaptedResult objects, one per experiment/claim.
        """
        ...

    @abc.abstractmethod
    def source_name(self) -> str:
        """Human-readable name of the data source."""
        ...
