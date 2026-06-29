"""Project-owned candidate-memory construction and lightweight retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Sequence

from memchain.data.benchmarks.base import Dialogue, Session
from memchain.schema import CandidateMemory

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
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


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if len(tok) >= 3 and tok not in STOPWORDS]


def session_text(session: Session) -> str:
    lines = [f"SESSION {session.session_id}"]
    if session.timestamp is not None:
        lines.append(f"timestamp: {session.timestamp.isoformat()}")
    if session.tags:
        lines.append(f"tags: {session.tags}")
    for utterance in session.utterances:
        lines.append(f"{utterance.speaker}: {utterance.text}")
    return "\n".join(lines)


def raw_session_memories(dialogue: Dialogue) -> list[CandidateMemory]:
    """Convert each dialogue session into a provenance-preserving memory unit."""

    memories: list[CandidateMemory] = []
    for idx, session in enumerate(dialogue.sessions):
        memories.append(
            CandidateMemory(
                memory_id=f"{dialogue.dialogue_id}:session:{idx}",
                content=session_text(session),
                type="episodic",
                time=session.timestamp.isoformat() if isinstance(session.timestamp, datetime) else None,
                source_turns=[idx],
                metadata={
                    "dialogue_id": dialogue.dialogue_id,
                    "session_id": session.session_id,
                    "session_index": idx,
                    "source": "raw_session",
                },
            )
        )
    return memories


def bm25_scores(query: str, corpus: Sequence[CandidateMemory]) -> list[float]:
    query_tokens = content_tokens(query)
    docs = [content_tokens(memory.content) or ["<empty>"] for memory in corpus]
    if not query_tokens:
        return [0.0 for _ in docs]
    n_docs = len(docs)
    if n_docs == 0:
        return []
    avgdl = sum(len(doc) for doc in docs) / max(1, n_docs)
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for doc in docs:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            score += idf * (tf[term] * (k1 + 1)) / denom
        scores.append(score)
    return scores


def retrieve_candidates(
    query: str,
    corpus: Sequence[CandidateMemory],
    *,
    top_k: int,
) -> list[CandidateMemory]:
    """Return a bounded BM25 candidate pool for framework smoke runs."""

    if top_k <= 0 or not corpus:
        return []
    scores = bm25_scores(query, corpus)
    ranked = sorted(zip(scores, corpus), key=lambda pair: pair[0], reverse=True)
    out: list[CandidateMemory] = []
    for rank, (score, memory) in enumerate(ranked[:top_k], start=1):
        row = CandidateMemory.from_dict(memory.to_dict())
        row.retrieval_score = float(score)
        metadata = dict(row.metadata)
        metadata.update(
            {
                "candidate_pool_strategy": "memchain_bm25",
                "candidate_rank": rank,
                "retrieval_channels": ["bm25"],
            }
        )
        row.metadata = metadata
        out.append(row)
    return out

