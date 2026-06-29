from __future__ import annotations

from datetime import datetime

from memchain.data.benchmarks.base import Benchmark, Dialogue, QAPair, Session, Utterance
from memchain.intentmem.framework import CandidateMemoryGenerator, IntentMemFramework
from memchain.intentmem.heuristics import heuristic_policy
from memchain.intentmem.metrics import active_memory_token_count, memory_precision_recall
from memchain.intentmem.schema import CandidateMemory, IntentMemExample, MemoryAction, validate_example
from memchain.memory_pool.intent_guided import (
    AtomicMemoryStore,
    IntentGuidedMemoryPoolBuilder,
    MemoryPoolConfig,
)


def toy_dialogue() -> Dialogue:
    return Dialogue(
        dialogue_id="d1",
        sessions=[
            Session(
                session_id="s1",
                timestamp=datetime(2026, 1, 1, 10, 0, 0),
                utterances=[
                    Utterance(speaker="user", text="I chose MemChain as the project direction."),
                    Utterance(speaker="assistant", text="It builds evidence chains before active memories."),
                ],
            ),
            Session(
                session_id="s2",
                timestamp=datetime(2026, 1, 2, 10, 0, 0),
                utterances=[
                    Utterance(speaker="user", text="The memory pool should keep temporal neighbors."),
                ],
            ),
        ],
        qa_pairs=[
            QAPair(
                question_id="q1",
                question="What project direction did the user choose?",
                gold_answer="MemChain",
                question_type="fact_lookup",
                answer_session_ids=["s1"],
            )
        ],
    )


def test_intent_guided_memory_pool_is_answer_blind_and_valid() -> None:
    dialogue = toy_dialogue()
    store = AtomicMemoryStore.from_dialogue_windows(dialogue)
    builder = IntentGuidedMemoryPoolBuilder(store, config=MemoryPoolConfig(max_candidates=4, use_dense=False))
    candidates, intent, trace = builder.build_pool(dialogue.qa_pairs[0].question, group=dialogue.dialogue_id)

    assert candidates
    assert intent.intent in {"fact_lookup", "temporal_state_tracking", "multi_hop_relation"}
    assert trace["retrieval_stage"] == "intent_guided_multiview"
    assert all(memory.metadata["candidate_pool_strategy"] == "intent_guided_candidate_memory" for memory in candidates)
    assert all("gold_answer" not in memory.metadata for memory in candidates)


def test_framework_builds_policy_output() -> None:
    dialogue = toy_dialogue()
    framework = IntentMemFramework(candidate_generator=CandidateMemoryGenerator(top_k=2))
    example = framework.build_policy_input(dialogue, dialogue.qa_pairs[0], benchmark="toy")
    out = framework.run_policy(example, lambda row: heuristic_policy(row, keep_k=1))

    assert validate_example(out, require_labels=True) == []
    assert out.memory_chain
    assert out.active_memories
    assert active_memory_token_count(out) > 0


def test_action_metric_uses_selected_ids_only() -> None:
    pred = [MemoryAction(action="KEEP", memory_id="m1"), MemoryAction(action="DROP", memory_id="m2")]
    gold = [MemoryAction(action="REFINE", memory_id="m1"), MemoryAction(action="KEEP", memory_id="m3")]
    metrics = memory_precision_recall(pred, gold)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5


def test_benchmark_jsonl_roundtrip(tmp_path) -> None:
    benchmark = Benchmark(name="toy", dialogues=[toy_dialogue()])
    path = tmp_path / "toy.jsonl"
    benchmark.to_jsonl(path)
    loaded = Benchmark.from_jsonl(path)
    assert loaded.name == "toy"
    assert loaded.total_qa_pairs() == 1


def test_schema_validation_rejects_unknown_candidate_reference() -> None:
    example = IntentMemExample(
        sample_id="s1",
        question="What changed?",
        candidate_memories=[CandidateMemory(memory_id="m1", content="A useful memory.")],
        memory_actions=[MemoryAction(action="KEEP", memory_id="missing")],
        active_memories=["A useful memory."],
    )
    errors = validate_example(example, require_labels=False)
    assert any("unknown memory id" in err for err in errors)
