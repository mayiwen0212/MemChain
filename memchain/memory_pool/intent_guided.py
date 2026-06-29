"""Intent-guided construction of per-question candidate memory pools.

This module builds compact, provenance-grounded candidate memories for a
question before MemChain policy labeling or inference. The builder is
answer-blind: it can use the question and dialogue/memory provenance, but never
the gold answer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import numpy as np

from memchain.data.benchmarks.base import Benchmark, Dialogue, QAPair
from memchain.intentmem.schema import CandidateMemory, IntentMemExample


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
DATE_RE = re.compile(
    r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|20\d{2}|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday)\b",
    flags=re.IGNORECASE,
)
PROPER_RE = re.compile(r"\b[A-Z][A-Za-z0-9_'-]{2,}(?:\s+[A-Z][A-Za-z0-9_'-]{2,}){0,2}\b")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}

MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

LOCOMO_DIALOGUE_GROUPS = {
    f"locomo_dialogue_{idx}": group
    for idx, group in enumerate(
        ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]
    )
}


@dataclass
class QuestionIntent:
    """Answer-blind evidence intent inferred from a question."""

    intent: str
    queries: list[str]
    needs_temporal: bool = False
    needs_multihop: bool = False
    needs_entity_focus: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryPoolConfig:
    """Retrieval and assembly limits for one candidate memory pool."""

    max_candidates: int = 18
    initial_candidates: int = 14
    bm25_top_k: int = 80
    dense_top_k: int = 80
    feedback_top_k: int = 8
    use_dense: bool = True
    keep_related_negatives: int = 4
    min_content_chars: int = 8


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if len(tok) >= 3 and tok not in STOPWORDS]


def compact_text(text: str, *, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def infer_question_intent(question: str) -> QuestionIntent:
    q_lower = question.lower()
    q_terms = content_tokens(question)
    names = [name.strip() for name in PROPER_RE.findall(question) if name.strip().lower() not in STOPWORDS]

    needs_temporal = any(word in q_lower for word in ("when", "date", "before", "after", "first", "last", "recent"))
    needs_multihop = any(word in q_lower for word in ("both", "also", "same", "relationship", "between", "compare"))
    if q_lower.startswith(("why ", "how ")):
        intent = "open_domain_contextual_qa"
    elif needs_temporal:
        intent = "temporal_state_tracking"
    elif needs_multihop:
        intent = "multi_hop_relation"
    elif any(word in q_lower for word in ("like", "prefer", "favorite", "enjoy", "interested")):
        intent = "preference_recall"
    else:
        intent = "fact_lookup"

    queries = [question]
    if q_terms:
        queries.append(" ".join(q_terms))
    if names:
        queries.append(" ".join(names + q_terms[:4]))
    if needs_temporal:
        temporal_terms = [tok for tok in q_terms if tok not in {"when", "date", "time"}]
        queries.append(" ".join(names + temporal_terms[:6] + ["date"]))
    if needs_multihop:
        queries.append(" ".join(names + q_terms[:8] + ["connection"]))

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = " ".join(content_tokens(query))
        if key and key not in seen:
            seen.add(key)
            deduped.append(query)
    return QuestionIntent(
        intent=intent,
        queries=deduped or [question],
        needs_temporal=needs_temporal,
        needs_multihop=needs_multihop,
        metadata={"names": names, "content_terms": q_terms},
    )


class AtomicMemoryStore:
    """A grouped store of atomic memories with optional dense embeddings."""

    def __init__(self, groups: dict[str, list[CandidateMemory]]) -> None:
        self.groups = groups
        self._doc_tokens: dict[str, list[list[str]]] = {}
        self._dense_embeddings: dict[str, np.ndarray] = {}

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        group_key: str | None = None,
    ) -> "AtomicMemoryStore":
        groups: dict[str, list[CandidateMemory]] = {}
        for idx, row in enumerate(records):
            content = compact_text(str(row.get("content") or row.get("text") or ""))
            if len(content) < 8:
                continue
            session_id = str(row.get("session_id") or row.get("source_session") or "")
            if group_key is not None:
                group = group_key
            elif "::" in session_id:
                group = session_id.split("::", 1)[0]
            elif session_id:
                group = session_id
            else:
                group = "__global__"
            memory_id = str(row.get("memory_id") or row.get("id") or f"{session_id or group}:atom:{idx}")
            memory = CandidateMemory(
                memory_id=memory_id,
                content=content,
                type=str(row.get("type") or "atomic_fact"),
                time=row.get("timestamp") or row.get("time"),
                source_turns=list(row.get("source_turns") or []),
                metadata={
                    "source": "atomic_memory",
                    "source_session": session_id or row.get("source_session"),
                    "keywords": list(row.get("keywords") or []),
                    "persons": list(row.get("persons") or []),
                    "entities": list(row.get("entities") or []),
                    "topic": row.get("topic"),
                    "group": group,
                },
            )
            groups.setdefault(group, []).append(memory)
        return cls(groups)

    @classmethod
    def from_dialogue_windows(cls, dialogue: Dialogue, *, radius: int = 1, max_chars: int = 900) -> "AtomicMemoryStore":
        memories: list[CandidateMemory] = []
        for session_idx, session in enumerate(dialogue.sessions):
            timestamp = session.timestamp.isoformat() if isinstance(session.timestamp, datetime) else None
            turns = session.utterances
            for turn_idx in range(len(turns)):
                start = max(0, turn_idx - radius)
                end = min(len(turns), turn_idx + radius + 1)
                lines = []
                if timestamp:
                    lines.append(f"timestamp: {timestamp}")
                lines.append(f"session: {session.session_id}")
                for utterance in turns[start:end]:
                    lines.append(f"{utterance.speaker}: {utterance.text}")
                memories.append(
                    CandidateMemory(
                        memory_id=f"{dialogue.dialogue_id}:{session.session_id}:window:{start}-{end - 1}",
                        content=compact_text("\n".join(lines), max_chars=max_chars),
                        type="dialogue_window",
                        time=timestamp,
                        source_turns=[session_idx],
                        metadata={
                            "source": "dialogue_window",
                            "source_session": session.session_id,
                            "session_index": session_idx,
                            "turn_start": start,
                            "turn_end": end - 1,
                            "group": dialogue.dialogue_id,
                        },
                    )
                )
        return cls({dialogue.dialogue_id: memories})

    def get(self, group: str) -> list[CandidateMemory]:
        return self.groups.get(group, [])

    def doc_tokens(self, group: str) -> list[list[str]]:
        if group not in self._doc_tokens:
            self._doc_tokens[group] = [content_tokens(memory.content) or ["<empty>"] for memory in self.get(group)]
        return self._doc_tokens[group]

    def build_dense(self, embedder: Any, *, group: str) -> None:
        memories = self.get(group)
        if not memories or group in self._dense_embeddings:
            return
        texts = [memory.content for memory in memories]
        vectors = embedder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._dense_embeddings[group] = np.asarray(vectors, dtype=np.float32)

    def dense_embeddings(self, group: str) -> np.ndarray | None:
        return self._dense_embeddings.get(group)


class IntentGuidedMemoryPoolBuilder:
    """Build one compact memory pool per question."""

    def __init__(
        self,
        store: AtomicMemoryStore,
        *,
        config: MemoryPoolConfig | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.store = store
        self.config = config or MemoryPoolConfig()
        self.embedder = embedder

    def build_pool(self, question: str, *, group: str) -> tuple[list[CandidateMemory], QuestionIntent, dict[str, Any]]:
        memories = self.store.get(group)
        if self.embedder is not None and self.config.use_dense:
            self.store.build_dense(self.embedder, group=group)
        intent = infer_question_intent(question)
        fused, channels = self._initial_retrieve(question, intent, group=group)
        self._feedback_backfill(question, intent, group=group, fused=fused, channels=channels)
        selected = self._assemble(memories, fused, channels, intent)
        trace = {
            "intent": intent.intent,
            "queries": intent.queries,
            "retrieval_stage": "intent_guided_multiview",
            "feedback_enabled": True,
            "candidate_count": len(selected),
        }
        return selected, intent, trace

    def _initial_retrieve(
        self,
        question: str,
        intent: QuestionIntent,
        *,
        group: str,
    ) -> tuple[dict[int, float], dict[int, list[str]]]:
        memories = self.store.get(group)
        fused: dict[int, float] = {}
        channels: dict[int, list[str]] = {}
        for qidx, query in enumerate(intent.queries):
            for rank, (idx, score) in enumerate(self._bm25_rank(query, group=group)[: self.config.bm25_top_k], start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.2 / (40 + rank) + 0.035 * score
                channels.setdefault(idx, []).append(f"lexical_intent_{qidx}")
            for rank, (idx, score) in enumerate(self._dense_rank(query, group=group)[: self.config.dense_top_k], start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.6 / (40 + rank) + 0.09 * score
                channels.setdefault(idx, []).append(f"semantic_intent_{qidx}")

        for idx, memory in enumerate(memories):
            boost = self._structured_boost(question, intent, memory)
            if boost:
                fused[idx] = fused.get(idx, 0.0) + boost
                channels.setdefault(idx, []).append("structured_intent")
        return fused, channels

    def _feedback_backfill(
        self,
        question: str,
        intent: QuestionIntent,
        *,
        group: str,
        fused: dict[int, float],
        channels: dict[int, list[str]],
    ) -> None:
        memories = self.store.get(group)
        if not fused:
            return
        top_indices = sorted(fused, key=fused.get, reverse=True)[: self.config.initial_candidates]
        top_text = "\n".join(memories[idx].content for idx in top_indices)
        feedback_queries: list[str] = []
        names = intent.metadata.get("names") or []
        terms = intent.metadata.get("content_terms") or content_tokens(question)

        if intent.needs_temporal and not DATE_RE.search(top_text):
            feedback_queries.append(" ".join(list(names) + terms[:6] + ["date", "time"]))
        if intent.needs_multihop and len(top_indices) < max(8, self.config.initial_candidates // 2):
            feedback_queries.append(" ".join(list(names) + terms[:8] + ["related", "evidence"]))
        if not feedback_queries and names:
            feedback_queries.append(" ".join(list(names) + terms[:5]))

        for qidx, query in enumerate(feedback_queries):
            for rank, (idx, score) in enumerate(self._bm25_rank(query, group=group)[: self.config.feedback_top_k], start=1):
                fused[idx] = fused.get(idx, 0.0) + 0.9 / (20 + rank) + 0.03 * score
                channels.setdefault(idx, []).append(f"feedback_lexical_{qidx}")
            for rank, (idx, score) in enumerate(self._dense_rank(query, group=group)[: self.config.feedback_top_k], start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.1 / (20 + rank) + 0.07 * score
                channels.setdefault(idx, []).append(f"feedback_semantic_{qidx}")

    def _assemble(
        self,
        memories: list[CandidateMemory],
        fused: dict[int, float],
        channels: dict[int, list[str]],
        intent: QuestionIntent,
    ) -> list[CandidateMemory]:
        ranked = sorted(fused, key=fused.get, reverse=True)
        selected: list[CandidateMemory] = []
        seen_content: set[str] = set()
        for idx in ranked:
            memory = CandidateMemory.from_dict(memories[idx].to_dict())
            content_key = re.sub(r"\s+", " ", memory.content.lower()).strip()
            if not content_key or content_key in seen_content:
                continue
            seen_content.add(content_key)
            memory.retrieval_score = float(fused[idx])
            metadata = dict(memory.metadata)
            metadata.update(
                {
                    "candidate_pool_strategy": "intent_guided_candidate_memory",
                    "candidate_rank": len(selected) + 1,
                    "retrieval_channels": sorted(set(channels.get(idx, []))),
                    "retrieval_fusion_score": float(fused[idx]),
                    "question_intent": intent.intent,
                }
            )
            memory.metadata = metadata
            selected.append(memory)
            if len(selected) >= self.config.max_candidates:
                break
        return selected

    def _bm25_rank(self, query: str, *, group: str) -> list[tuple[int, float]]:
        query_terms = content_tokens(query)
        docs = self.store.doc_tokens(group)
        if not query_terms or not docs:
            return []
        n_docs = len(docs)
        avgdl = sum(len(doc) for doc in docs) / max(1, n_docs)
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc))
        scores: list[tuple[int, float]] = []
        k1 = 1.5
        b = 0.75
        for idx, doc in enumerate(docs):
            tf = Counter(doc)
            dl = len(doc)
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                denom = tf[term] + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
                score += idf * (tf[term] * (k1 + 1)) / denom
            if score > 0:
                scores.append((idx, score))
        return sorted(scores, key=lambda pair: pair[1], reverse=True)

    def _dense_rank(self, query: str, *, group: str) -> list[tuple[int, float]]:
        embeddings = self.store.dense_embeddings(group)
        if self.embedder is None or embeddings is None or len(embeddings) == 0:
            return []
        query_vector = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = np.dot(embeddings, query_vector)
        order = np.argsort(-scores)
        return [(int(idx), float(scores[idx])) for idx in order if float(scores[idx]) > 0]

    @staticmethod
    def _structured_boost(question: str, intent: QuestionIntent, memory: CandidateMemory) -> float:
        q_terms = set(content_tokens(question))
        m_text = memory.content.lower()
        m_terms = set(content_tokens(memory.content))
        score = 0.0
        if q_terms:
            score += 0.14 * len(q_terms & m_terms)
        for name in intent.metadata.get("names") or []:
            if name.lower() in m_text:
                score += 0.7
        if intent.needs_temporal:
            if DATE_RE.search(memory.content) or DATE_RE.search(str(memory.time or "")):
                score += 0.45
            for month, value in MONTHS.items():
                if month in question.lower() and f"-{value}-" in str(memory.time or ""):
                    score += 1.0
        return score


def build_locomo_memory_pool_examples(
    benchmark: Benchmark,
    builder: IntentGuidedMemoryPoolBuilder,
    *,
    four_class: bool = True,
    limit: int | None = None,
    stratified_counts: dict[str, int] | None = None,
    dialogue_groups: dict[str, str] | None = None,
) -> list[IntentMemExample]:
    """Build LoCoMo candidate-pool examples with one pool per QA item."""

    dialogue_groups = dialogue_groups or LOCOMO_DIALOGUE_GROUPS
    used_by_type: Counter[str] = Counter()
    examples: list[IntentMemExample] = []
    for dialogue in benchmark.dialogues:
        group = dialogue_groups.get(dialogue.dialogue_id, dialogue.dialogue_id)
        for qa in dialogue.qa_pairs:
            if four_class and qa.question_type == "Adversarial":
                continue
            if stratified_counts is not None:
                qtype = qa.question_type or "unknown"
                if used_by_type[qtype] >= stratified_counts.get(qtype, 0):
                    continue
            candidates, intent, trace = builder.build_pool(qa.question, group=group)
            examples.append(_to_example(dialogue, qa, candidates, intent, trace))
            if stratified_counts is not None:
                used_by_type[qa.question_type or "unknown"] += 1
                if all(used_by_type[key] >= value for key, value in stratified_counts.items()):
                    return examples
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def build_dialogue_memory_pool_examples(
    dialogue: Dialogue,
    builder: IntentGuidedMemoryPoolBuilder,
    *,
    benchmark_name: str = "synthetic_long_dialogue_qa",
    sample_prefix: str = "intent_guided_memory_pool",
) -> list[IntentMemExample]:
    """Build candidate-pool examples for any dialogue-shaped benchmark item."""

    examples: list[IntentMemExample] = []
    for qa in dialogue.qa_pairs:
        candidates, intent, trace = builder.build_pool(qa.question, group=dialogue.dialogue_id)
        examples.append(
            IntentMemExample(
                sample_id=f"{sample_prefix}:{benchmark_name}:{dialogue.dialogue_id}:{qa.question_id}",
                question=qa.question,
                gold_answer=qa.gold_answer,
                candidate_memories=candidates,
                metadata={
                    "benchmark": benchmark_name,
                    "dialogue_id": dialogue.dialogue_id,
                    "question_id": qa.question_id,
                    "question_type": qa.question_type,
                    "answer_session_ids": list(qa.answer_session_ids),
                    "candidate_source": "dialogue_windows",
                    "candidate_strategy": "intent_guided_candidate_memory",
                    "intent": intent.intent,
                    **trace,
                },
            )
        )
    return examples


def _to_example(
    dialogue: Dialogue,
    qa: QAPair,
    candidates: list[CandidateMemory],
    intent: QuestionIntent,
    trace: dict[str, Any],
) -> IntentMemExample:
    return IntentMemExample(
        sample_id=f"intent_guided_memory_pool:locomo:{dialogue.dialogue_id}:{qa.question_id}",
        question=qa.question,
        gold_answer=qa.gold_answer,
        candidate_memories=candidates,
        metadata={
            "benchmark": "locomo",
            "dialogue_id": dialogue.dialogue_id,
            "question_id": qa.question_id,
            "question_type": qa.question_type,
            "answer_session_ids": list(qa.answer_session_ids),
            "candidate_source": "atomic_memories",
            "candidate_strategy": "intent_guided_candidate_memory",
            "intent": intent.intent,
            **trace,
        },
    )
