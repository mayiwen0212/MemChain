#!/usr/bin/env python
"""Run a deterministic MemChain policy for smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memchain.heuristics import heuristic_policy
from memchain.schema import MemChainExample, validate_example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-k", type=int, default=5)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open("r", encoding="utf-8") as reader, args.output.open("w", encoding="utf-8") as writer:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            example = MemChainExample.from_dict(json.loads(line))
            labeled = heuristic_policy(example, keep_k=args.keep_k)
            errors = validate_example(labeled, require_labels=True)
            if errors:
                raise ValueError(f"{example.sample_id} failed validation: {errors}")
            writer.write(json.dumps(labeled.to_dict(), ensure_ascii=False) + "\n")
            count += 1

    print(f"wrote {count} policy outputs to {args.output}")


if __name__ == "__main__":
    main()
