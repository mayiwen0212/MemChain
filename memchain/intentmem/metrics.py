"""Metrics for active-memory exposure and answer evaluation."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from memchain.intentmem.schema import IntentMemExample, MemoryAction

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = TOKEN_RE.findall(normalize_text(prediction))
    gold_tokens = TOKEN_RE.findall(normalize_text(gold))
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum(min(pred_counts.get(tok, 0), count) for tok, count in gold_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def string_match(prediction: str, gold: str) -> bool:
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    if not pred_norm or not gold_norm:
        return pred_norm == gold_norm
    return gold_norm in pred_norm or pred_norm in gold_norm


def action_counts(actions: Iterable[MemoryAction]) -> Counter[str]:
    return Counter(action.action for action in actions)


def selected_ids(actions: Iterable[MemoryAction]) -> set[str]:
    ids: set[str] = set()
    for action in actions:
        if action.action in {"KEEP", "MERGE", "REFINE", "ADD"}:
            ids.update(action.referenced_ids())
    return ids


def memory_precision_recall(predicted: Iterable[MemoryAction], gold: Iterable[MemoryAction]) -> dict[str, float]:
    pred_ids = selected_ids(predicted)
    gold_ids = selected_ids(gold)
    if not pred_ids and not gold_ids:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = len(pred_ids & gold_ids) / max(1, len(pred_ids))
    recall = len(pred_ids & gold_ids) / max(1, len(gold_ids))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def approx_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def active_memory_token_count(example: IntentMemExample) -> int:
    return sum(approx_token_count(text) for text in example.active_memories)


def memory_chain_validity(example: IntentMemExample) -> dict[str, float]:
    selected = selected_ids(example.memory_actions)
    if not example.memory_chain:
        valid_empty = 1.0 if not selected else 0.0
        return {"step_validity": valid_empty, "citation_validity": valid_empty}

    candidate_ids = example.candidate_id_set()
    step_ok = True
    citation_ok = True
    for step in example.memory_chain:
        if not step.step_id or not step.statement or not step.memory_ids:
            step_ok = False
        for memory_id in step.memory_ids:
            if memory_id not in candidate_ids:
                citation_ok = False
    return {
        "step_validity": 1.0 if step_ok else 0.0,
        "citation_validity": 1.0 if citation_ok else 0.0,
    }


def dataset_summary(examples: Iterable[IntentMemExample], *, require_labels: bool = False) -> dict[str, object]:
    rows = list(examples)
    intents: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    candidate_counts: list[int] = []
    active_counts: list[int] = []
    active_tokens: list[int] = []
    for example in rows:
        if example.intent_plan:
            intents[example.intent_plan.intent] += 1
        actions.update(action_counts(example.memory_actions))
        candidate_counts.append(len(example.candidate_memories))
        active_counts.append(len(example.active_memories))
        active_tokens.append(active_memory_token_count(example))

    def avg(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    out: dict[str, object] = {
        "num_examples": len(rows),
        "avg_candidates": avg(candidate_counts),
        "avg_active_memories": avg(active_counts),
        "avg_active_tokens": avg(active_tokens),
        "intent_counts": dict(intents),
        "action_counts": dict(actions),
    }
    if require_labels:
        total_actions = max(1, sum(actions.values()))
        out["action_rates"] = {k: v / total_actions for k, v in actions.items()}
    return out
