"""MemChain framework components.

This module gives the project a stable implementation surface:

MemorySubstrate -> CandidateMemoryGenerator -> IntentPlanner ->
MemChainBuilder -> MemoryActionPolicy -> ActiveMemoryComposer ->
SufficiencyController -> frozen answer model.

The trainable policy itself can be a local checkpoint, an OpenAI-compatible
teacher, or a deterministic debug policy. This file intentionally keeps the
memory substrate lightweight so the method remains focused on read-time
active-memory construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from memchain.data.benchmarks.base import Dialogue, QAPair
from memchain.intentmem.heuristics import heuristic_policy
from memchain.intentmem.retriever import raw_session_memories, retrieve_candidates
from memchain.intentmem.schema import (
    CandidateMemory,
    IntentMemExample,
    MemoryAction,
    MemoryChainStep,
)


@dataclass(frozen=True)
class MemorySubstrate:
    """Project-owned representation of historical dialogue memory units."""

    memories: tuple[CandidateMemory, ...]
    substrate_id: str = "intentmem_memory_substrate"
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dialogue(cls, dialogue: Dialogue) -> "MemorySubstrate":
        return cls(
            memories=tuple(raw_session_memories(dialogue)),
            metadata={
                "dialogue_id": dialogue.dialogue_id,
                "num_memory_units": len(dialogue.sessions),
                "substrate_source": "historical_dialogue_sessions",
            },
        )


@dataclass(frozen=True)
class CandidateMemoryGenerator:
    """Hybrid-recall entry point for building a query-conditioned candidate pool."""

    top_k: int = 12

    def generate(self, question: str, substrate: MemorySubstrate) -> list[CandidateMemory]:
        return retrieve_candidates(question, substrate.memories, top_k=self.top_k)


@dataclass(frozen=True)
class PolicyInputBuilder:
    """Build MemChain examples from a question and generated candidates."""

    def build(
        self,
        *,
        sample_id: str,
        question: str,
        candidate_memories: Sequence[CandidateMemory],
        gold_answer: str = "",
        metadata: dict[str, object] | None = None,
    ) -> IntentMemExample:
        return IntentMemExample(
            sample_id=sample_id,
            question=question,
            gold_answer=gold_answer,
            candidate_memories=list(candidate_memories),
            metadata=dict(metadata or {}),
        )


def selected_action_ids(actions: Iterable[MemoryAction]) -> set[str]:
    ids: set[str] = set()
    for action in actions:
        if action.action in {"KEEP", "MERGE", "REFINE", "ADD"}:
            ids.update(action.referenced_ids())
    return ids


@dataclass(frozen=True)
class MemChainBuilder:
    """Build an ordered evidence chain from selected memory actions.

    Teacher and trained policies should emit ``memory_chain`` directly.  This
    fallback keeps deterministic debug runs and older checkpoints on the same
    framework contract by turning selected actions into auditable chain steps.
    """

    max_statement_chars: int = 180

    def build(self, example: IntentMemExample) -> list[MemoryChainStep]:
        by_id = {memory.memory_id: memory for memory in example.candidate_memories}
        steps: list[MemoryChainStep] = []
        selected_actions = [
            action
            for action in example.memory_actions
            if action.action in {"KEEP", "MERGE", "REFINE"}
        ]
        for action in selected_actions:
            refs = [memory_id for memory_id in action.referenced_ids() if memory_id in by_id]
            if not refs:
                continue
            statement = action.output or by_id[refs[0]].content
            statement = " ".join(statement.split())
            if len(statement) > self.max_statement_chars:
                statement = statement[: self.max_statement_chars - 3].rstrip() + "..."
            steps.append(
                MemoryChainStep(
                    step_id=f"c{len(steps) + 1}",
                    role=self._role_for(example, action, is_first=not steps),
                    memory_ids=refs,
                    statement=statement,
                    relation_to_next="supports",
                )
            )
        if steps:
            steps[-1].relation_to_next = "none"
        return steps

    @staticmethod
    def _role_for(example: IntentMemExample, action: MemoryAction, *, is_first: bool) -> str:
        intent = example.intent_plan.intent if example.intent_plan else ""
        if intent == "temporal_state_tracking":
            return "seed_evidence" if is_first else "temporal_update"
        if intent == "conflict_update":
            return "resolution"
        if action.action == "MERGE":
            return "bridge"
        if is_first:
            return "seed_evidence"
        return "supporting_context"


@dataclass(frozen=True)
class ActiveMemoryComposer:
    """Compose answer-ready active memories from policy actions."""

    max_raw_chars: int = 360

    def compose(self, example: IntentMemExample) -> list[str]:
        by_id = {memory.memory_id: memory for memory in example.candidate_memories}
        active: list[str] = []
        for action in example.memory_actions:
            if action.action == "STOP":
                continue
            if action.output:
                active.append(action.output)
                continue
            if action.action not in {"KEEP", "MERGE", "REFINE"}:
                continue
            for memory_id in action.referenced_ids():
                memory = by_id.get(memory_id)
                if memory is None:
                    continue
                content = " ".join(memory.content.split())
                if len(content) > self.max_raw_chars:
                    content = content[: self.max_raw_chars - 3].rstrip() + "..."
                active.append(content)
        return active


PolicyFn = Callable[[IntentMemExample], IntentMemExample]


@dataclass(frozen=True)
class IntentMemFramework:
    """End-to-end framework wrapper for non-generation experiments."""

    candidate_generator: CandidateMemoryGenerator = field(default_factory=CandidateMemoryGenerator)
    input_builder: PolicyInputBuilder = field(default_factory=PolicyInputBuilder)
    chain_builder: MemChainBuilder = field(default_factory=MemChainBuilder)
    composer: ActiveMemoryComposer = field(default_factory=ActiveMemoryComposer)

    def build_policy_input(self, dialogue: Dialogue, qa: QAPair, *, benchmark: str) -> IntentMemExample:
        substrate = MemorySubstrate.from_dialogue(dialogue)
        candidates = self.candidate_generator.generate(qa.question, substrate)
        sample_id = f"{benchmark}:{dialogue.dialogue_id}:{qa.question_id}"
        return self.input_builder.build(
            sample_id=sample_id,
            question=qa.question,
            gold_answer=qa.gold_answer,
            candidate_memories=candidates,
            metadata={
                "benchmark": benchmark,
                "dialogue_id": dialogue.dialogue_id,
                "question_id": qa.question_id,
                "question_type": qa.question_type,
                "answer_session_ids": list(qa.answer_session_ids),
                "candidate_source": "intentmem_memory_substrate",
                "candidate_top_k": self.candidate_generator.top_k,
            },
        )

    def run_policy(self, example: IntentMemExample, policy: PolicyFn | None = None) -> IntentMemExample:
        out = policy(example) if policy is not None else heuristic_policy(example)
        if not out.memory_chain:
            out.memory_chain = self.chain_builder.build(out)
        if not out.active_memories:
            out.active_memories = self.composer.compose(out)
        return out


MemChainFramework = IntentMemFramework
