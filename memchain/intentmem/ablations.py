"""Ablation transforms for IntentMem policy outputs.

These transforms operate on existing labeled or predicted policy outputs.  They
do not create new teacher labels; they provide reproducible variants for
diagnostics, ablation smoke tests, and derived SFT export pipelines.
"""

from __future__ import annotations

from dataclasses import replace

from memchain.intentmem.heuristics import active_statement
from memchain.intentmem.schema import IntentMemExample, IntentPlan, MemoryAction

ABLATION_MODES = (
    "full",
    "no_intent_plan",
    "no_memory_chain",
    "no_memory_actions",
    "no_refine",
    "no_merge",
    "no_stop",
    "raw_selected_memories",
    "fixed_topk1",
    "fixed_topk3",
    "fixed_topk5",
)


def _copy(example: IntentMemExample) -> IntentMemExample:
    return IntentMemExample.from_dict(example.to_dict())


def _with_metadata(example: IntentMemExample, mode: str) -> IntentMemExample:
    example.metadata = {**example.metadata, "intentmem_ablation": mode}
    return example


def _fallback_intent_plan(example: IntentMemExample, *, budget: int) -> IntentPlan:
    if example.intent_plan is not None:
        return replace(example.intent_plan, budget=budget)
    return IntentPlan(intent="fact_lookup", needed_types=["episodic"], time_scope="any", budget=budget)


def _fixed_topk(example: IntentMemExample, *, k: int, mode: str) -> IntentMemExample:
    out = _copy(example)
    kept = out.candidate_memories[:k]
    kept_ids = {memory.memory_id for memory in kept}
    active = [active_statement(memory) for memory in kept]
    actions: list[MemoryAction] = []
    for memory in out.candidate_memories:
        if memory.memory_id in kept_ids:
            actions.append(MemoryAction(action="KEEP", memory_id=memory.memory_id, reason=f"{mode} ablation"))
        else:
            actions.append(MemoryAction(action="DROP", memory_id=memory.memory_id, reason=f"{mode} ablation"))
    actions.append(MemoryAction(action="STOP", reason="fixed top-k evidence budget reached"))
    out.intent_plan = _fallback_intent_plan(out, budget=len(kept))
    out.memory_actions = actions
    out.active_memories = active
    out.sufficiency = "enough" if active else "insufficient"
    return _with_metadata(out, mode)


def apply_ablation(example: IntentMemExample, mode: str) -> IntentMemExample:
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode: {mode}")
    if mode == "full":
        return _with_metadata(_copy(example), mode)
    if mode == "fixed_topk1":
        return _fixed_topk(example, k=1, mode=mode)
    if mode == "fixed_topk3":
        return _fixed_topk(example, k=3, mode=mode)
    if mode == "fixed_topk5":
        return _fixed_topk(example, k=5, mode=mode)

    out = _copy(example)
    if mode == "no_intent_plan":
        out.intent_plan = IntentPlan(
            intent="fact_lookup",
            needed_types=[],
            time_scope="any",
            evidence_need="intent plan removed for ablation",
            budget=len(out.active_memories),
        )
    elif mode == "no_memory_chain":
        out.memory_chain = []
    elif mode == "no_memory_actions":
        out.memory_actions = [
            MemoryAction(action="STOP", reason="memory actions removed for ablation")
        ]
        out.sufficiency = "enough" if out.active_memories else "insufficient"
    elif mode == "no_refine":
        for action in out.memory_actions:
            if action.action == "REFINE":
                action.action = "KEEP"
                action.output = None
        selected = out.selected_memory_ids()
        out.active_memories = [
            active_statement(memory)
            for memory in out.candidate_memories
            if memory.memory_id in selected
        ]
    elif mode == "no_merge":
        expanded: list[MemoryAction] = []
        for action in out.memory_actions:
            if action.action == "MERGE":
                for memory_id in action.referenced_ids():
                    expanded.append(MemoryAction(action="KEEP", memory_id=memory_id, reason="MERGE disabled"))
            else:
                expanded.append(action)
        out.memory_actions = expanded
        selected = out.selected_memory_ids()
        out.active_memories = [
            active_statement(memory)
            for memory in out.candidate_memories
            if memory.memory_id in selected
        ]
    elif mode == "no_stop":
        out.memory_actions = [action for action in out.memory_actions if action.action != "STOP"]
        out.memory_actions.append(MemoryAction(action="STOP", reason="synthetic stop for schema validity"))
        out.sufficiency = "needs_more"
    elif mode == "raw_selected_memories":
        selected = out.selected_memory_ids()
        out.active_memories = [
            active_statement(memory, max_chars=700)
            for memory in out.candidate_memories
            if memory.memory_id in selected
        ]
    return _with_metadata(out, mode)
