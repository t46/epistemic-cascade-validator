"""Adapters for converting real autoresearch pipeline outputs to EvidencePackets.

Each adapter is responsible for parsing a specific data source and mapping
its output fields to the ECV EvidencePacket schema.

Adapter hierarchy:
  BaseAdapter  (abstract)
    AutoResearchEvaluatorAdapter  - 14 experiments from auto-research-evaluator
    ARAAdapter                    - ARA belief store (claims + ledger)
    VanillaAutoresearchAdapter    - vanilla autoresearch hypothesis pipeline
"""

from ecv.adapters.base import BaseAdapter
from ecv.adapters.evaluator import AutoResearchEvaluatorAdapter
from ecv.adapters.ara import ARAAdapter
from ecv.adapters.vanilla import VanillaAutoresearchAdapter

__all__ = [
    "BaseAdapter",
    "AutoResearchEvaluatorAdapter",
    "ARAAdapter",
    "VanillaAutoresearchAdapter",
]
