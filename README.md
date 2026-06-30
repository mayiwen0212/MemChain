<div align="center">

<h1>MemChain</h1>

<p>
  <b>Learning Interpretable Memory Traces for Memory-Augmented LLM Agents</b>
</p>

<p>
  Query-guided active-memory construction for long-dialogue agents.
</p>

<p>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2EA44F?style=flat&labelColor=555" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat&labelColor=555&logo=python&logoColor=white" alt="Python"></a>
  <a href="tests"><img src="https://img.shields.io/badge/tests-pytest-0A7E8C?style=flat&labelColor=555" alt="Tests"></a>
  <a href="assets/figure2_memchain_overview.png"><img src="https://img.shields.io/badge/figure-MemChain_overview-B72A3F?style=flat&labelColor=555" alt="Figure"></a>
</p>

<p>
  <a href="#quick-start">Quick Start</a> ·
  <a href="#method-overview">Method Overview</a> ·
  <a href="#run-the-example">Run the Example</a> ·
  <a href="#training">Training</a> ·
  <a href="#code-layout">Code Layout</a>
</p>

<table>
<tr>
<td align="center" width="120">
  <img src="assets/icons/candidate-pool.svg" width="44" height="44" alt="Candidate Pool"><br>
  <sub><b>Candidate Pool</b></sub>
</td>
<td align="center" width="120">
  <img src="assets/icons/intent-plan.svg" width="44" height="44" alt="Intent Plan"><br>
  <sub><b>Intent Plan</b></sub>
</td>
<td align="center" width="120">
  <img src="assets/icons/memory-chain.svg" width="44" height="44" alt="Memory Chain"><br>
  <sub><b>Memory Chain</b></sub>
</td>
<td align="center" width="120">
  <img src="assets/icons/active-memory.svg" width="44" height="44" alt="Active Memory"><br>
  <sub><b>Active Memory</b></sub>
</td>
<td align="center" width="120">
  <img src="assets/icons/rl-training.svg" width="44" height="44" alt="SFT to RL"><br>
  <sub><b>SFT + RL</b></sub>
</td>
</tr>
</table>

<p>
  <img src="assets/figure2_memchain_overview.png" alt="MemChain overview" width="94%">
</p>

</div>

<h2 id="highlights">✨ Highlights</h2>

MemChain is a read-time memory policy for long-dialogue agents. Given a user
question and a candidate memory pool, it produces an explicit memory trace
before the answer model is called:

- 🧭 `intent_plan`: what evidence the question requires.
- 🧩 `memory_actions`: which memories to keep, drop, refine, merge, or stop on.
- 🔗 `memory_chain`: ordered evidence steps with memory citations.
- 📝 `active_memories`: compact answer-ready memories passed to a frozen answer model.

The candidate-memory pool is answer-blind: it uses the question and dialogue
provenance, not the gold answer. This keeps retrieval, policy learning, and
answer generation separated and auditable.

<h2 id="quick-start">🚀 Quick Start</h2>

**📦 Install**

```bash
git clone https://github.com/mayiwen0212/MemChain.git
cd MemChain
pip install -e ".[dev]"
pytest -q
```

**🧠 Build a candidate pool and run the policy**

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

**🔍 Inspect one MemChain policy output**

```bash
head -n 1 examples/out/minimal_policy_outputs.jsonl | python -m json.tool | head -80
```

The example is compact, but it follows the same public contract as the full
experiments: normalized dialogue history, answer-blind candidate construction,
structured policy output, and active-memory composition.

<h2 id="method-overview">🧭 Method Overview</h2>

MemChain follows this read-time pipeline:

1. Convert long dialogue history into provenance-preserving candidate memories.
2. Infer the evidence need from the question.
3. Build a bounded candidate pool with lexical, entity, temporal, and
   multi-hop retrieval views.
4. Produce a structured MemChain trace with actions and cited chain steps.
5. Pass only `question + active_memories` to the answer model.

This design makes memory use explicit instead of silently stuffing raw retrieved
notes into the answer context.

<h2 id="run-the-example">🧪 Run the Example</h2>

The repository includes a complete compact benchmark file:

```text
examples/minimal_dialogue.jsonl
```

It contains:

- three timestamped dialogue sessions,
- two QA pairs,
- answer-session provenance,
- one fact lookup question and one temporal tracking question.

After running the quick-start commands, the generated policy rows contain:

```json
{
  "intent_plan": {"intent": "fact_lookup", "...": "..."},
  "memory_actions": [{"action": "KEEP", "memory_id": "..."}],
  "memory_chain": [{"step_id": "c1", "memory_ids": ["..."]}],
  "active_memories": ["..."]
}
```

<h2 id="training">🏋️ Training</h2>

The intended training setup is H200 eight-card training.

In our experiments, the MemChain policy is trained with an SFT-to-RL workflow on
8 x H200 GPUs. The released repository keeps the method-facing modules, schema,
memory-pool construction, policy IO, reward utilities, metrics, and smoke tests.

Private dataset paths, model checkpoints, API keys, service endpoints, raw
benchmark dumps, and machine-specific launch scripts are intentionally not
included.

<h2 id="code-layout">🗂️ Code Layout</h2>

```text
memchain/
  data/benchmarks/base.py        # normalized dialogue/session/QA dataclasses
  memory_pool/intent_guided.py   # answer-blind candidate memory pool
  schema.py                      # public MemChain data schema
  framework.py                   # framework wrapper and active-memory composer
  prompts.py                     # policy and teacher prompts
  policy_io.py                   # policy JSON parsing and SFT row export
  reward.py                      # active-memory reward utility
  metrics.py                     # trace and active-memory metrics
  llm/                           # optional OpenAI-compatible clients
scripts/
  build_memory_pool.py           # build candidate pools from normalized JSONL
  run_heuristic_policy.py        # deterministic policy sanity check
examples/
  minimal_dialogue.jsonl         # runnable open-source example
tests/
  test_core.py
```

<h2 id="data-interface">🔌 Data Interface</h2>

Input data should be normalized as dialogues with sessions and QA pairs. The
core dataclasses are in `memchain/data/benchmarks/base.py`.

Each policy input uses:

- `sample_id`
- `question`
- `candidate_memories`
- optional `gold_answer`
- optional `metadata`

Each policy output uses:

- `intent_plan`
- `memory_actions`
- `memory_chain`
- `active_memories`
- `sufficiency`

<h2 id="scope">📦 Scope</h2>

This repository contains the MemChain core implementation for open-source
inspection and extension. It does not include third-party comparison code,
benchmark raw data, trained checkpoints, evaluation outputs, paper drafts,
private API keys, private endpoints, or machine-specific paths.

<h2 id="citation">📝 Citation</h2>

```bibtex
@misc{memchain2026,
  title  = {MemChain: Learning Interpretable Memory Traces for Memory-Augmented LLM Agents},
  author = {Ma, Yiwen},
  year   = {2026},
  note   = {Open-source code release}
}
```

<h2 id="license">📄 License</h2>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

<h2 id="acknowledgments">🙏 Acknowledgments</h2>

We would like to thank the following projects and teams:

- 🔍 **Embedding Backend:** [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) and compatible embedding models, such as [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding), for optional dense candidate-memory search.
- 🧮 **Retrieval Core:** NumPy-backed in-memory dense scoring plus MemChain's BM25, entity, temporal, and feedback retrieval views for provenance-grounded candidate pools.
- 📊 **Benchmark:** [LoCoMo](https://github.com/snap-research/locomo) - long-context memory evaluation framework.
