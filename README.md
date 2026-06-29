# MemChain

MemChain is an intent-guided active-memory policy for long-dialogue agents.
Given a question and a broad historical memory pool, it builds an explicit evidence
chain and a compact set of answer-ready active memories for a separate frozen answer
model.

This repository is the cleaned open-source implementation of the project-owned
method only. It intentionally excludes third-party comparison code, local
evaluation outputs, trained checkpoints, private datasets, and drafting artifacts.

## Method Surface

The public pipeline is:

1. Convert dialogue history into provenance-preserving memory units.
2. Build a query-conditioned candidate memory pool with intent-guided multi-view recall.
3. Run a MemChain policy that emits:
   - `intent_plan`
   - `memory_actions`
   - `memory_chain`
   - `active_memories`
4. Pass only `question + active_memories` to a frozen answer model.

The candidate pool builder in `memchain/memory_pool/intent_guided.py` is
answer-blind: it can use the question and dialogue provenance, but not the gold
answer.

## Install

```bash
pip install -e ".[dev]"
```

Optional dense retrieval:

```bash
pip install -e ".[dense]"
```

## Quick Start

Build candidate pools from the toy dialogue:

```bash
python scripts/build_memory_pool.py \
  --input examples/toy_dialogue.jsonl \
  --output outputs/toy_memory_pool.jsonl
```

Run the deterministic policy smoke path:

```bash
python scripts/run_heuristic_policy.py \
  --input outputs/toy_memory_pool.jsonl \
  --output outputs/toy_policy.jsonl
```

Run tests:

```bash
pytest -q
```

If `pytest` is not installed, install the development extras first:

```bash
pip install -e ".[dev]"
```

## Python API

```python
from memchain.data.benchmarks.base import Benchmark
from memchain.memory_pool.intent_guided import (
    AtomicMemoryStore,
    IntentGuidedMemoryPoolBuilder,
)

benchmark = Benchmark.from_jsonl("examples/toy_dialogue.jsonl")
dialogue = benchmark.dialogues[0]

store = AtomicMemoryStore.from_dialogue_windows(dialogue)
builder = IntentGuidedMemoryPoolBuilder(store)
candidates, intent, trace = builder.build_pool(
    dialogue.qa_pairs[0].question,
    group=dialogue.dialogue_id,
)
```

Each candidate memory keeps provenance metadata such as source session, turn
window, retrieval channels, rank, fusion score, and inferred question intent.

## Data Format

MemChain uses one normalized JSONL file per benchmark split. The first row can
be an optional header:

```json
{"_header": true, "name": "toy", "metadata": {"description": "demo"}}
```

Each following row is one dialogue:

```json
{
  "dialogue_id": "dialogue_1",
  "sessions": [
    {
      "session_id": "s1",
      "timestamp": "2026-01-01T10:00:00",
      "utterances": [
        {"speaker": "user", "text": "I chose MemChain."}
      ]
    }
  ],
  "qa_pairs": [
    {
      "question_id": "q1",
      "question": "What project direction did the user choose?",
      "gold_answer": "MemChain",
      "question_type": "fact_lookup",
      "answer_session_ids": ["s1"]
    }
  ]
}
```

The memory-pool construction step does not read `gold_answer`; that field is
kept for evaluation and supervised-label generation.

## Repository Layout

```text
memchain/
  data/benchmarks/base.py        # dialogue, session, QA dataclasses
  memory_pool/intent_guided.py   # answer-blind candidate memory pool builder
  intentmem/schema.py            # policy schema
  intentmem/framework.py         # end-to-end framework wrapper
  intentmem/prompts.py           # policy and teacher prompts
  intentmem/policy_io.py         # JSON repair, API client, SFT row conversion
  intentmem/reward.py            # lightweight active-memory reward utility
  llm/                          # optional OpenAI-compatible clients
scripts/
  build_memory_pool.py
  run_heuristic_policy.py
tests/
  test_core.py
```

## What Is Not Included

The release does not include:

- third-party comparison implementations or adapters
- benchmark raw data
- evaluation outputs
- local model checkpoints
- drafting folders
- private endpoints, API keys, logs, or machine-specific paths

Use your own benchmark data in the normalized JSONL format shown in
`examples/toy_dialogue.jsonl`.

## License

MIT.
