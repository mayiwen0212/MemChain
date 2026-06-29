"""Deterministic heuristics for smoke tests and non-API debugging."""

from __future__ import annotations

from collections import Counter

from memchain.retriever import tokenize
from memchain.schema import (
    CandidateMemory,
    MemChainExample,
    IntentPlan,
    MemoryAction,
)


def infer_intent(question: str, question_type: str | None = None) -> str:
    q = question.lower()
    qtype = (question_type or "").lower()
    if any(k in qtype for k in ["open_domain", "open domain"]):
        return "open_domain_contextual_qa"
    if any(k in q or k in qtype for k in ["prefer", "favorite", "like", "preference"]):
        return "preference_recall"
    if any(k in q or k in qtype for k in ["when", "before", "after", "current", "latest", "now"]):
        return "temporal_state_tracking"
    if any(k in q or k in qtype for k in ["conflict", "change", "update", "instead", "retract"]):
        return "conflict_update"
    if any(k in q or k in qtype for k in ["why", "how", "relationship", "connect"]):
        return "multi_hop_relation"
    return "fact_lookup"


def overlap_score(query: str, memory: CandidateMemory, gold_answer: str = "") -> float:
    q = Counter(tokenize(query))
    g = Counter(tokenize(gold_answer))
    doc = Counter(tokenize(memory.content))
    if not q and not g:
        return 0.0
    q_overlap = sum(min(count, doc.get(tok, 0)) for tok, count in q.items()) / max(1, sum(q.values()))
    g_overlap = sum(min(count, doc.get(tok, 0)) for tok, count in g.items()) / max(1, sum(g.values()))
    source_bonus = 0.0
    score = memory.retrieval_score if memory.retrieval_score is not None else 0.0
    return q_overlap + 0.75 * g_overlap + 0.02 * score + source_bonus


def active_statement(memory: CandidateMemory, *, max_chars: int = 360) -> str:
    content = " ".join(memory.content.split())
    if len(content) <= max_chars:
        return content
    return content[: max_chars - 3].rstrip() + "..."


def heuristic_policy(example: MemChainExample, *, keep_k: int = 5) -> MemChainExample:
    question_type = str(example.metadata.get("question_type", "") or "")
    intent = infer_intent(example.question, question_type)
    ranked = sorted(
        example.candidate_memories,
        key=lambda memory: overlap_score(example.question, memory, example.gold_answer),
        reverse=True,
    )
    kept = [memory for memory in ranked[:keep_k] if overlap_score(example.question, memory, example.gold_answer) > 0]
    if not kept and ranked:
        kept = ranked[:1]

    kept_ids = {memory.memory_id for memory in kept}
    actions: list[MemoryAction] = []
    active: list[str] = []
    for memory in example.candidate_memories:
        if memory.memory_id in kept_ids:
            action = "REFINE" if len(memory.content) > 500 else "KEEP"
            statement = active_statement(memory)
            active.append(statement)
            actions.append(
                MemoryAction(
                    memory_id=memory.memory_id,
                    action=action,
                    reason="highest lexical/evidence overlap for the inferred intent",
                    output=statement if action == "REFINE" else None,
                )
            )
        else:
            actions.append(
                MemoryAction(
                    memory_id=memory.memory_id,
                    action="DROP",
                    reason="lower evidence utility for the inferred intent",
                )
            )
    actions.append(
        MemoryAction(
            action="STOP",
            reason="selected evidence is enough" if active else "no supporting memory was found",
        )
    )
    return MemChainExample(
        sample_id=example.sample_id,
        question=example.question,
        gold_answer=example.gold_answer,
        candidate_memories=list(example.candidate_memories),
        intent_plan=IntentPlan(
            intent=intent,
            needed_types=sorted({memory.type for memory in kept}) or ["semantic"],
            time_scope="recent" if intent in {"temporal_state_tracking", "conflict_update"} else "any",
            evidence_need=f"select evidence for {intent}",
            budget=keep_k,
        ),
        memory_actions=actions,
        active_memories=active,
        sufficiency="enough" if active else "insufficient",
        metadata=dict(example.metadata),
    )
