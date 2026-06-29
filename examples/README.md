# MemChain Minimal Example

This folder contains a compact runnable example for the public MemChain API.

## Files

- `minimal_dialogue.jsonl`: normalized long-dialogue benchmark input.
- `out/`: generated candidate pools and policy outputs; this directory is
  created when you run the commands below.

The input contains three timestamped sessions and two QA pairs. It is small
enough to inspect by hand, but it uses the same schema as full benchmark
adapters.

## Run

From the repository root:

```bash
python scripts/build_memory_pool.py \
  --input examples/minimal_dialogue.jsonl \
  --output examples/out/minimal_candidates.jsonl \
  --max-candidates 8 \
  --no-dense

python scripts/run_heuristic_policy.py \
  --input examples/out/minimal_candidates.jsonl \
  --output examples/out/minimal_policy_outputs.jsonl \
  --keep-k 3
```

The first command builds an answer-blind candidate memory pool. The second
command runs a deterministic policy that emits the public MemChain fields:
`intent_plan`, `memory_actions`, `memory_chain`, and `active_memories`.

## Inspect

```bash
head -n 1 examples/out/minimal_policy_outputs.jsonl | python -m json.tool | head -80
```
