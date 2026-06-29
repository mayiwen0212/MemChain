#!/usr/bin/env python
"""Build answer-blind MemChain candidate pools from normalized dialogues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memchain.data.benchmarks.base import Benchmark
from memchain.memory_pool.intent_guided import (
    AtomicMemoryStore,
    IntentGuidedMemoryPoolBuilder,
    MemoryPoolConfig,
    build_dialogue_memory_pool_examples,
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Normalized Benchmark JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with candidate_memories.")
    parser.add_argument("--max-candidates", type=int, default=18)
    parser.add_argument("--no-dense", action="store_true", help="Run the lightweight BM25/metadata path only.")
    args = parser.parse_args()

    benchmark = Benchmark.from_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as writer:
        for dialogue in benchmark.dialogues:
            store = AtomicMemoryStore.from_dialogue_windows(dialogue)
            builder = IntentGuidedMemoryPoolBuilder(
                store,
                config=MemoryPoolConfig(max_candidates=args.max_candidates, use_dense=not args.no_dense),
            )
            examples = build_dialogue_memory_pool_examples(
                dialogue,
                builder,
                benchmark_name=benchmark.name,
            )
            for example in examples:
                writer.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                count += 1

    print(f"wrote {count} examples to {args.output}")


if __name__ == "__main__":
    main()
