"""Schema for MemChain active-memory policy data.

MemChain trains a read-time policy over candidate memories. The policy
returns an intent plan, memory-chain evidence steps, per-memory actions, and
compact active memories. This module is deliberately strict about the public action
and intent space so data generation cannot drift back into a broad memory-OS
schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

ACTIONS = ("KEEP", "DROP", "MERGE", "REFINE", "ADD", "STOP")
INTENTS = (
    "fact_lookup",
    "preference_recall",
    "temporal_state_tracking",
    "multi_hop_relation",
    "conflict_update",
    "open_domain_contextual_qa",
)
SUFFICIENCY_LABELS = ("enough", "insufficient", "needs_more")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


@dataclass
class CandidateMemory:
    memory_id: str
    content: str
    type: str = "semantic"
    time: str | None = None
    source_turns: list[int] = field(default_factory=list)
    retrieval_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateMemory":
        memory_id = _clean_string(data.get("memory_id") or data.get("id"))
        typ = _clean_string(data.get("type") or data.get("dimension") or "semantic")
        score = data.get("retrieval_score", data.get("score"))
        return cls(
            memory_id=memory_id,
            content=_clean_string(data.get("content") or data.get("text")),
            type=typ or "semantic",
            time=data.get("time") or data.get("timestamp") or data.get("created_at"),
            source_turns=[int(x) for x in _as_list(data.get("source_turns")) if str(x).strip()],
            retrieval_score=float(score) if score is not None else None,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntentPlan:
    intent: str
    needed_types: list[str] = field(default_factory=list)
    time_scope: str = "any"
    evidence_need: str = ""
    budget: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "IntentPlan":
        data = data or {}
        intent = _clean_string(data.get("intent") or data.get("query_intent") or "fact_lookup")
        needed = data.get("needed_types", data.get("needed_memory_types", []))
        budget = data.get("budget", data.get("retrieval_budget", 5))
        try:
            budget_int = int(budget)
        except (TypeError, ValueError):
            budget_int = 5
        return cls(
            intent=intent,
            needed_types=[_clean_string(x) for x in _as_list(needed) if _clean_string(x)],
            time_scope=_clean_string(data.get("time_scope") or "any"),
            evidence_need=_clean_string(data.get("evidence_need") or data.get("reason") or ""),
            budget=max(0, budget_int),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryAction:
    action: str
    memory_id: str | None = None
    memory_ids: list[str] = field(default_factory=list)
    reason: str = ""
    output: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryAction":
        action = _clean_string(data.get("action") or data.get("type")).upper()
        ids = data.get("memory_ids", data.get("target_ids", []))
        memory_id = data.get("memory_id", data.get("target_id"))
        return cls(
            action=action,
            memory_id=_clean_string(memory_id) or None,
            memory_ids=[_clean_string(x) for x in _as_list(ids) if _clean_string(x)],
            reason=_clean_string(data.get("reason") or data.get("rationale")),
            output=(
                _clean_string(data.get("output") or data.get("new_content") or data.get("content"))
                or None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def referenced_ids(self) -> list[str]:
        ids: list[str] = []
        if self.memory_id:
            ids.append(self.memory_id)
        ids.extend(self.memory_ids)
        return ids


@dataclass
class MemoryChainStep:
    step_id: str
    role: str = ""
    memory_ids: list[str] = field(default_factory=list)
    statement: str = ""
    relation_to_next: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryChainStep":
        ids = data.get("memory_ids", data.get("evidence_ids", data.get("memory_id", [])))
        return cls(
            step_id=_clean_string(data.get("step_id") or data.get("id")),
            role=_clean_string(data.get("role") or data.get("type")),
            memory_ids=[_clean_string(x) for x in _as_list(ids) if _clean_string(x)],
            statement=_clean_string(data.get("statement") or data.get("content") or data.get("summary")),
            relation_to_next=_clean_string(data.get("relation_to_next") or data.get("next_relation")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def referenced_ids(self) -> list[str]:
        return list(self.memory_ids)


@dataclass
class IntentMemExample:
    sample_id: str
    question: str
    gold_answer: str = ""
    candidate_memories: list[CandidateMemory] = field(default_factory=list)
    intent_plan: IntentPlan | None = None
    memory_actions: list[MemoryAction] = field(default_factory=list)
    memory_chain: list[MemoryChainStep] = field(default_factory=list)
    active_memories: list[str] = field(default_factory=list)
    sufficiency: str = "insufficient"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentMemExample":
        candidates = [
            CandidateMemory.from_dict(row)
            for row in data.get("candidate_memories", data.get("candidates", []))
            if isinstance(row, dict)
        ]
        actions = [
            MemoryAction.from_dict(row)
            for row in data.get("memory_actions", data.get("actions", []))
            if isinstance(row, dict)
        ]
        chain_raw = data.get("memory_chain", data.get("memchain", []))
        chain = [
            MemoryChainStep.from_dict(row)
            for row in _as_list(chain_raw)
            if isinstance(row, dict)
        ]
        intent_raw = data.get("intent_plan")
        intent = IntentPlan.from_dict(intent_raw) if isinstance(intent_raw, dict) else None
        active_raw = data.get("active_memories", [])
        active = [
            _clean_string(item.get("content") if isinstance(item, dict) else item)
            for item in _as_list(active_raw)
            if _clean_string(item.get("content") if isinstance(item, dict) else item)
        ]
        return cls(
            sample_id=_clean_string(data.get("sample_id") or data.get("id")),
            question=_clean_string(data.get("question")),
            gold_answer=_clean_string(data.get("gold_answer") or data.get("answer")),
            candidate_memories=candidates,
            intent_plan=intent,
            memory_actions=actions,
            memory_chain=chain,
            active_memories=active,
            sufficiency=_clean_string(data.get("sufficiency") or "insufficient"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "candidate_memories": [m.to_dict() for m in self.candidate_memories],
            "intent_plan": self.intent_plan.to_dict() if self.intent_plan else None,
            "memory_actions": [a.to_dict() for a in self.memory_actions],
            "memory_chain": [step.to_dict() for step in self.memory_chain],
            "active_memories": list(self.active_memories),
            "sufficiency": self.sufficiency,
            "metadata": dict(self.metadata),
        }

    def candidate_id_set(self) -> set[str]:
        return {m.memory_id for m in self.candidate_memories}

    def selected_memory_ids(self) -> set[str]:
        selected: set[str] = set()
        for action in self.memory_actions:
            if action.action in {"KEEP", "MERGE", "REFINE", "ADD"}:
                selected.update(action.referenced_ids())
        return selected


MemChainExample = IntentMemExample


def validate_example(example: IntentMemExample, *, require_labels: bool = False) -> list[str]:
    errors: list[str] = []
    if not example.sample_id:
        errors.append("missing sample_id")
    if not example.question:
        errors.append("missing question")
    candidate_ids = [m.memory_id for m in example.candidate_memories]
    if any(not mid for mid in candidate_ids):
        errors.append("candidate memory_id is empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        errors.append("candidate memory_id values are not unique")
    if require_labels:
        if example.intent_plan is None:
            errors.append("missing intent_plan")
        elif example.intent_plan.intent not in INTENTS:
            errors.append(f"unknown intent: {example.intent_plan.intent}")
        if example.sufficiency not in SUFFICIENCY_LABELS:
            errors.append(f"unknown sufficiency: {example.sufficiency}")
        if not example.memory_actions:
            errors.append("missing memory_actions")
        if not example.active_memories and example.sufficiency == "enough":
            errors.append("sufficiency=enough but active_memories is empty")
    candidate_set = set(candidate_ids)
    for idx, action in enumerate(example.memory_actions):
        if action.action not in ACTIONS:
            errors.append(f"memory_actions[{idx}] unknown action: {action.action}")
        refs = action.referenced_ids()
        if action.action == "STOP":
            continue
        if not refs:
            errors.append(f"memory_actions[{idx}] missing memory id")
        for ref in refs:
            if ref not in candidate_set:
                errors.append(f"memory_actions[{idx}] references unknown memory id: {ref}")
    for idx, step in enumerate(example.memory_chain):
        if not step.step_id:
            errors.append(f"memory_chain[{idx}] missing step_id")
        if not step.statement:
            errors.append(f"memory_chain[{idx}] missing statement")
        for ref in step.referenced_ids():
            if ref not in candidate_set:
                errors.append(f"memory_chain[{idx}] references unknown memory id: {ref}")
    return errors


def read_jsonl(path: str | Path) -> list[IntentMemExample]:
    path = Path(path)
    rows: list[IntentMemExample] = []
    with path.open("r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                rows.append(IntentMemExample.from_dict(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, examples: Iterable[IntentMemExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for example in examples:
            writer.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
