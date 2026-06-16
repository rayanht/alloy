"""Speculative decoding for alloy"""

from __future__ import annotations

from .contract import Drafter, Proposal, RoundStats, SpecMetrics, TapBatch, TargetTaps
from .session import SpecSession

__all__ = [
    "Drafter",
    "Proposal",
    "RoundStats",
    "SpecMetrics",
    "TapBatch",
    "TargetTaps",
    "SpecSession",
    "SPEC_DRAFTERS",
]
