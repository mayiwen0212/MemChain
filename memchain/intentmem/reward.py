"""Reward utilities for active-memory exposure policy optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveMemoryRewardWeights:
    answer_correctness: float = 0.40
    evidence_recall: float = 0.25
    evidence_precision: float = 0.20
    stop_correctness: float = 0.10
    token_cost: float = 0.05


def active_memory_reward(
    *,
    answer_correctness: float,
    evidence_recall: float,
    evidence_precision: float,
    stop_correctness: float,
    active_tokens: int,
    token_budget: int = 512,
    weights: ActiveMemoryRewardWeights = ActiveMemoryRewardWeights(),
) -> float:
    """Compute the utility score for read-time memory exposure.

    Inputs should already be normalized to [0, 1] except ``active_tokens``.
    Token cost is clipped at 1.0 so very long contexts do not dominate the score.
    """

    cost = min(1.0, max(0.0, active_tokens / max(1, token_budget)))
    return (
        weights.answer_correctness * _clip01(answer_correctness)
        + weights.evidence_recall * _clip01(evidence_recall)
        + weights.evidence_precision * _clip01(evidence_precision)
        + weights.stop_correctness * _clip01(stop_correctness)
        - weights.token_cost * cost
    )


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
