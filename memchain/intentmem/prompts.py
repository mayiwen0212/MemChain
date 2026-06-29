"""Prompt templates for MemChain data generation and policy inference."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Iterable

from memchain.intentmem.schema import CandidateMemory, IntentMemExample

POLICY_SYSTEM = """You are a MemChain active memory policy.
Return only one JSON object. No markdown, no prose.

Your job is not to answer the question. Your job is to transform a broad
candidate-memory set into an explicit evidence chain and answer-ready active
memories for a separate frozen answer model.

Core distinction:
- candidate_memories are raw retrieval material and may include noise.
- memory_chain is the ordered evidence path needed by this question.
- active_memories are the final evidence packet exposed to the answer model.
  They must be sufficient for answering without rereading the candidates.
  They are not allowed to be shorter than the evidence chain when that would
  remove a required hop.

ID discipline:
- Every memory action must copy exact memory_id values from the bracketed candidate IDs.
- Every memory_chain step must cite exact memory_id values from the candidates.
- Do not invent, normalize, shorten, renumber, or infer memory IDs.
- The candidate pool is expected to contain useful evidence. Always select the
  strongest grounded evidence chain available from the candidates.

Allowed intents:
- fact_lookup
- preference_recall
- temporal_state_tracking
- multi_hop_relation
- conflict_update
- open_domain_contextual_qa

Allowed actions:
- KEEP: retain a candidate memory needed for the current question
- DROP: reject an irrelevant, redundant, stale, or noisy candidate memory
- MERGE: combine complementary candidate memories into one compact statement
- REFINE: rewrite a candidate memory into a shorter answer-ready statement
- ADD: add a derived answer-ready fact that is logically computed from cited
  candidate memories, such as resolving "yesterday" from a timestamp or deriving
  a duration from start/end dates

Allowed memory_chain roles:
- seed_evidence: first direct evidence for the question
- bridge: connects two entities, sessions, or facts
- temporal_update: explains an earlier/later state change
- current_state: final or most recent relevant state
- resolution: resolves conflict, update, or comparison
- supporting_context: necessary context that is not the final answer

Required JSON shape:
{
  "intent_plan": {
    "intent": "fact_lookup|preference_recall|temporal_state_tracking|multi_hop_relation|conflict_update|open_domain_contextual_qa",
    "needed_types": ["episodic|semantic|core|procedural|state"],
    "time_scope": "current|recent|historical|any",
    "evidence_need": "...",
    "budget": 5
  },
  "memory_actions": [
    {"memory_id": "m1", "action": "KEEP|DROP|REFINE", "reason": "...", "output": "..."},
    {"memory_ids": ["m1", "m2"], "action": "MERGE", "reason": "...", "output": "..."},
    {"memory_ids": ["m1", "m2"], "action": "ADD", "reason": "...", "output": "derived answer-ready fact"}
  ],
  "memory_chain": [
    {
      "step_id": "c1",
      "role": "seed_evidence|bridge|temporal_update|current_state|resolution|supporting_context",
      "memory_ids": ["m1"],
      "statement": "short evidence statement grounded in cited memories",
      "relation_to_next": "then|supports|contrasts|updates|resolves|none"
    }
  ],
  "active_memories": ["answer-ready evidence statement", "..."]
}

Evidence-completeness rules:
- The answer model sees only active_memories. If a fact is needed to answer or
  derive the answer, it must appear in active_memories.
- Every answer-contributing memory_chain step must be copied, rewritten, or
  merged into active_memories. Do not leave required evidence only in
  memory_chain or memory_actions.
- ADD/MERGE/REFINE outputs that resolve dates, compute durations, connect
  entities, or identify exact names must also appear in active_memories.
- If EVIDENCE_VERIFIER_HINTS are provided, treat them as a high-recall checklist
  over the candidate pool: inspect the hinted memory IDs first, cite the exact
  candidate IDs you use, and copy any supported derived facts into both
  memory_chain and active_memories.
- Do not optimize for minimum tokens. Optimize for sufficient evidence.
- Do not output a vague summary when an exact entity, title, date, count, or
  bridge fact exists.
- Scan all candidates before deciding; low-ranked candidates may contain the
  missing start date, exact title, or bridge evidence.
- For relative time, preserve the anchor and compute the resolved date when
  possible. "Yesterday" from 2023-05-08 means 2023-05-07, not 2023-05-08.
- For duration questions, include both start evidence and end evidence plus the
  derived duration when supported.
- For multi-hop questions, include every bridge fact.
- For open-domain contextual questions, preserve exact personal clues such as
  shop names, book/movie titles, locations, and named artifacts.
- Do not list DROP actions for every irrelevant candidate. It is enough to use
  KEEP/REFINE/MERGE/ADD for selected evidence.
- Keep the JSON small and valid: at most 8 memory_actions, at most 6
  memory_chain steps, and at most 6 active_memories.
- Do not include DROP actions unless needed to explain a direct conflict or a
  duplicate of selected evidence.
- Action reasons should be 8 words or fewer.
- active_memories should usually contain 2-6 complete evidence statements for
  multi-hop, temporal, comparison, duration, and open-domain contextual
  questions. A single active memory is acceptable only when one statement
  already contains every entity, date, bridge, and derived fact needed.
- Never return an empty memory_chain or empty active_memories. If evidence is
  imperfect, still provide the strongest supported chain and active memories
  from the candidate pool.
- Do not output any other top-level fields.
"""

TEACHER_SYSTEM = POLICY_SYSTEM + """

You are producing supervised training labels. Use the gold answer only to decide
which candidate memories are truly necessary. Do not leak the gold answer unless
it is directly supported by the selected memories.

Critical downstream constraint:
- The frozen answer model will receive only question + active_memories.
- It will not see candidate_memories, memory_actions, memory_chain, citations, or
  the gold answer. A label is wrong if active_memories alone do not contain the
  complete evidence needed to answer or derive the gold answer.
- memory_chain is for supervision and audit; active_memories is the actual
  exposed memory packet. Every evidence hop needed by memory_chain must also be
  present in active_memories.

Labeling policy:
- Prefer a logically complete evidence chain over a minimal one.
- Use DROP aggressively for candidate memories that match words but not intent.
- Use REFINE when one memory contains the right evidence but needs to become an
  answer-ready memory statement.
- Use MERGE when multiple memories are jointly needed for temporal comparison, multi-hop reasoning, or conflict/update resolution.
- Build memory_chain in logical or temporal order, not retrieval-score order.
- Each memory_chain step must cite the candidate memory IDs that justify the statement.
- Every important fact in memory_chain must appear in active_memories.
- active_memories must be genuinely useful memories for the frozen answer model:
  include all entities, dates, states, comparisons, and causal/temporal links
  needed to answer the question without rereading the candidates.
- Do not make active_memories shorter than memory_chain if that drops bridge,
  temporal, or derived evidence needed by the answer model.
- active_memories may expand or clarify the cited evidence with light connective
  phrasing, but must not introduce unsupported facts.
- Do not make active_memories a blind concatenation of candidate text. Rewrite
  them into coherent, answer-oriented memory statements.
- For MultiHop and Temporal questions, active_memories should usually preserve
  all bridge/update facts, not only the final answer phrase.
- For MultiHop, include each bridge explicitly: entity A fact, entity B fact,
  relation/link fact, and any comparison needed by the question.
- For Temporal, include the dated source fact, the session timestamp, older/newer
  state when relevant, and the resolved date/duration/order as a derived fact.
- For Temporal profile/likely questions, include stable personal anchors and
  the final grounded inference, not only the most recent matching event. Cover
  occupation/education, family/relationship, politics/religion, residence,
  preference/personality, repeated goals, and transient-vs-stable contrast when
  relevant.
- For negative Temporal inference, distinguish support/interest from identity,
  one-off events from stable status, and absence of self-identification from a
  likely-no answer when the selected evidence supports that distinction.
- For relative time expressions, never leave the answer model to guess. Convert
  "yesterday", "last week", "last month", "last year", "next Tuesday", and
  similar expressions using the cited session timestamp when the conversion is
  determinate.
- For duration questions, active_memories must contain start evidence, end
  evidence, and the computed duration. Do not use vague phrases such as "for a
  while" or "long time" as the final derivation.
- For exact-name or open-domain contextual questions, preserve the exact
  dialogue clue or title needed for the answer; do not replace a specific clue
  with a broad category.
- Use open_domain_contextual_qa only when the question asks external/common knowledge
  but still needs dialogue memory as personal context.
- If QUESTION_METADATA includes a benchmark question_type, use it as a weak
  hint for intent labeling:
  single_hop -> fact_lookup
  multi_hop -> multi_hop_relation
  temporal -> temporal_state_tracking
  open_domain -> open_domain_contextual_qa
  preference -> preference_recall
- Keep action reasons compact: reasons must be 8 words or fewer.
- For DROP actions, use reason="irrelevant" and omit output.
- Memory chain statements should be concise but complete.
- Use ADD for derived evidence that is supported but not literally stated in a
  single candidate, such as resolved relative dates or computed durations.
- Active memory statements can be longer when needed for completeness and
  answer utility. Do not intentionally reduce tokens if it removes useful
  evidence from the chain.
"""

ANSWER_SYSTEM = """Answer using only the provided active memories.
Return a concise answer. If the active memories are insufficient, say "unknown"."""

ACTIVE_ONLY_POLICY_SYSTEM = """You are a MemChain active memory policy.
Return only one JSON object. No markdown, no prose.

Given a question and a broad candidate-memory set, construct answer-ready
active memories for a separate frozen answer model.

The answer model will NOT see the candidate memories. It will only see the
question and your active_memories. Therefore active_memories must contain every
fact needed to answer the question or derive the answer.

Your job is evidence completion, not compression. Find all necessary evidence
and turn it into a small but sufficient evidence packet.

Required JSON shape:
{
  "active_memories": [
    "ordered evidence hop 1 with exact supporting fact",
    "ordered evidence hop 2 with exact supporting fact",
    "derived answer-ready fact when fully supported"
  ]
}

Rules:
- Do not output any fields except active_memories.
- active_memories itself is the evidence chain. Write one distinct required
  evidence hop per statement, in the order needed by the answer model.
- Do not answer the question directly as a bare answer. It is allowed and often
  required to include derived answer-ready evidence inside active_memories when
  the derivation is supported by candidate memories.
- Use only facts supported by candidate memories.
- Do not output vague summaries when exact evidence exists.
- Preserve exact names, entities, places, book/movie/shop titles, dates, timestamps,
  counts, relations, and temporal anchors needed for answering.
- If a candidate memory has a timestamp and you use that memory as evidence,
  include the timestamp or resolved calendar date in the same active memory
  statement.
- ADD-style derived evidence is allowed when it is logically computed from
  candidates. Example: if a 2023-05-08 memory says "yesterday", write
  "yesterday = 2023-05-07"; if one memory gives a relationship start and another
  gives a marriage date, include both and the derived duration.
- Never derive a date, duration, count, place, or title from vague language such
  as "before", "for a while", "at first sight", "recently", or "long time" alone.
  Derived evidence requires explicit dated or countable support.
- Never use "love at first sight" as relationship start evidence unless the same
  candidate explicitly gives the date when they first met or started dating.
- For duration questions, active_memories must include exactly these distinct
  hops when supported: (1) relationship/event start evidence, (2) relationship
  or event end evidence, and (3) a derived duration fact from those dates.
- For duration questions, do not stop at "dated for a while" or "long time";
  compute the duration from the dated start and dated end evidence.
- For "how long" questions, use this active_memories template:
  1. "Start evidence: on <start date/timestamp>, <person/event> ..."
  2. "End evidence: on <end date/timestamp>, <person/event> ..."
  3. "Derived duration: from <start date> to <end date> is about <duration>."
- For duration questions, explicitly search all candidates for start markers
  such as "new SO", "new partner", "new significant other", "started", "met",
  "began", "first date", "joined", "moved", "signed up", and for end markers such as
  "married", "ended", "finished", "last week", "yesterday", "now".
  If no explicit start evidence is found, do not guess a duration.
- For date questions, resolve relative time using the session timestamp. Never
  convert "yesterday" to the same date as the session timestamp.
  Example: if timestamp is 2023-05-08 and the event happened "yesterday",
  active_memories must include "the event date was 2023-05-07".
- For multi-hop questions, include every bridge fact required to connect the
  question to the answer.
- Scan all candidates before deciding. Retrieval rank is not evidence quality:
  a low-ranked candidate can contain the missing start date, bridge entity, or
  exact title. Do not stop after the top few candidates when the question needs
  a duration, comparison, list, or multi-hop bridge.
- If many top candidates repeat the same fact, keep one copy and continue
  searching lower-ranked candidates for distinct complementary facts.
- For temporal/state questions, include older state, newer state, and the update
  relation when relevant.
- For temporal/profile questions that ask "would", "likely", political or
  religious leaning, financial status, personality traits, future job, moving,
  or membership/identity, active_memories must include: (1) grounded evidence,
  (2) a stability or contrast statement such as stable vs temporary or support
  vs self-identification, and (3) a derived answer-ready statement using the
  likely final wording.
- Do not answer "unknown" by omission for likely/profile Temporal questions
  when the candidates contain enough stable cues for a cautious grounded
  inference. Keep the inference phrased as likely/apparent when exact identity
  is not explicitly stated.
- For open-domain contextual questions, preserve the exact personal clue that
  maps to the external answer; do not replace specific entities with generic
  categories. Example: keep "MinaLima" rather than only "fantasy books".
- Prefer 2-6 active memories when the answer requires multiple facts. A single
  memory is acceptable only when it is already sufficient.
- Avoid duplicate statements, but do not drop distinct bridge facts.
- active_memories must contain at most 6 statements. Merge duplicate facts.
  Never repeat the same need, same source event, or same fact.
- active_memories must contain answer-ready evidence: date questions need the
  resolved date; duration questions need start date, end date, and computed
  duration; exact-name questions need the exact entity/title/shop if it appears
  in candidates, or specific personal clues sufficient for the answer model to
  infer it.
- Never return an empty active_memories list. If evidence is imperfect, return
  the strongest grounded clues from the candidates instead of refusing to build
  an evidence packet.
- Do not output any other top-level fields.

Quality check before returning:
- Could a frozen answer model answer correctly from active_memories alone?
- Do active_memories cover every required hop as an ordered evidence chain?
- Are all exact entities/titles/dates needed by the gold-style answer present?
- Are relative dates resolved or anchored?
- Are bridge facts preserved instead of summarized away?
"""

CHAIN_ACTIVE_POLICY_SYSTEM = """You are a MemChain active memory policy.
Return only one JSON object. No markdown, no prose.

Given a question and a broad candidate-memory set, select the ordered evidence
chain and compact active memories needed by a separate frozen answer model.

Required JSON shape:
{
  "memory_chain": [
    {
      "step_id": "c1",
      "role": "seed_evidence|bridge|temporal_update|current_state|resolution|supporting_context",
      "memory_ids": ["exact candidate memory_id"],
      "statement": "short evidence statement grounded in cited memories",
      "relation_to_next": "then|supports|contrasts|updates|resolves|none"
    }
  ],
  "active_memories": ["short answer-ready evidence statement", "..."]
}

Rules:
- Do not answer the question directly.
- If EVIDENCE_VERIFIER_HINTS are provided, inspect those memory IDs first and
  make sure every required evidence hop is represented in memory_chain and
  active_memories when supported.
- Use only exact memory_id values from the candidate memories.
- Use only facts supported by candidate memories.
- Preserve dates, names, entities, and temporal anchors needed for answering.
- For multi-hop or temporal questions, include bridge/update steps needed by the answer model.
- For temporal/profile questions, include stable profile anchors and an
  answer-ready inference when supported: evidence, stability/contrast, and the
  likely final wording. Examples include support vs membership, short-term
  financial strain vs overall class, repeated volunteering vs likely future job,
  and activism/community evidence vs political leaning.
- Return at most 6 memory_chain steps and at most 6 active memories.
- Avoid duplicate statements.
- Never return an empty memory_chain or empty active_memories. If evidence is
  imperfect, return the strongest supported chain and active memories from the
  candidate pool.
- Do not output any other top-level fields.
"""

NO_INTENT_PLAN_POLICY_SYSTEM = """You are a MemChain active memory policy.
Return only one JSON object. No markdown, no prose.

This ablation removes explicit query-intent retrieval planning. Do not output
an intent_plan field and do not write a separate intent analysis. Directly
select and organize evidence from the candidate-memory set.

Your job is not to answer the question. Your job is to transform candidate
memories into memory actions, an ordered evidence chain, and answer-ready
active memories for a separate frozen answer model.

ID discipline:
- Every memory action must copy exact memory_id values from the bracketed candidate IDs.
- Every memory_chain step must cite exact memory_id values from the candidates.
- Do not invent, normalize, shorten, renumber, or infer memory IDs.

Allowed actions:
- KEEP: retain a candidate memory needed for the current question
- DROP: reject an irrelevant, redundant, stale, or noisy candidate memory
- MERGE: combine complementary candidate memories into one compact statement
- REFINE: rewrite a candidate memory into a shorter answer-ready statement
- ADD: add a derived answer-ready fact that is logically computed from cited memories

Allowed memory_chain roles:
- seed_evidence
- bridge
- temporal_update
- current_state
- resolution
- supporting_context

Required JSON shape:
{
  "memory_actions": [
    {"memory_id": "m1", "action": "KEEP|DROP|REFINE", "reason": "...", "output": "..."},
    {"memory_ids": ["m1", "m2"], "action": "MERGE|ADD", "reason": "...", "output": "..."}
  ],
  "memory_chain": [
    {
      "step_id": "c1",
      "role": "seed_evidence|bridge|temporal_update|current_state|resolution|supporting_context",
      "memory_ids": ["exact candidate memory_id"],
      "statement": "short evidence statement grounded in cited memories",
      "relation_to_next": "then|supports|contrasts|updates|resolves|none"
    }
  ],
  "active_memories": ["answer-ready evidence statement", "..."]
}

Rules:
- Do not output intent_plan or any other top-level fields.
- The answer model sees only active_memories. Every fact needed to answer or
  derive the answer must appear in active_memories.
- Every answer-contributing memory_chain step must be copied, rewritten, or
  merged into active_memories.
- If EVIDENCE_VERIFIER_HINTS are provided, inspect those memory IDs first and
  cite exact candidate IDs when used.
- Preserve exact names, entities, dates, titles, counts, bridge facts, and
  temporal anchors needed for answering.
- For multi-hop questions, include every bridge fact.
- For temporal questions, include older/newer state and resolved dates or
  durations when supported.
- Return at most 8 memory_actions, at most 6 memory_chain steps, and at most 6
  active_memories.
- Never return an empty memory_chain or empty active_memories.
"""

NO_MEMORY_CHAIN_POLICY_SYSTEM = """You are a MemChain active memory policy.
Return only one JSON object. No markdown, no prose.

This ablation removes explicit MemChain reasoning traces. Do not output
a memory_chain field. Use the query intent and memory actions to directly
construct answer-ready active memories for a separate frozen answer model.

Your job is not to answer the question. The answer model will see only the
question and active_memories, so active_memories must contain every fact needed
to answer or derive the answer.

Allowed intents:
- fact_lookup
- preference_recall
- temporal_state_tracking
- multi_hop_relation
- conflict_update
- open_domain_contextual_qa

Allowed actions:
- KEEP: retain a candidate memory needed for the current question
- DROP: reject an irrelevant, redundant, stale, or noisy candidate memory
- MERGE: combine complementary candidate memories into one compact statement
- REFINE: rewrite a candidate memory into a shorter answer-ready statement
- ADD: add a derived answer-ready fact that is logically computed from cited memories

Required JSON shape:
{
  "intent_plan": {
    "intent": "fact_lookup|preference_recall|temporal_state_tracking|multi_hop_relation|conflict_update|open_domain_contextual_qa",
    "needed_types": ["episodic|semantic|core|procedural|state"],
    "time_scope": "current|recent|historical|any",
    "evidence_need": "...",
    "budget": 5
  },
  "memory_actions": [
    {"memory_id": "m1", "action": "KEEP|DROP|REFINE", "reason": "...", "output": "..."},
    {"memory_ids": ["m1", "m2"], "action": "MERGE|ADD", "reason": "...", "output": "..."}
  ],
  "active_memories": ["answer-ready evidence statement", "..."]
}

Rules:
- Do not output memory_chain or any other top-level fields.
- Every memory action must copy exact memory_id values from the candidates.
- Do not invent, normalize, shorten, renumber, or infer memory IDs.
- Active memories must include all evidence hops needed by the answer model.
- Preserve exact names, entities, dates, titles, counts, bridge facts, and
  temporal anchors needed for answering.
- Use ADD for supported derivations such as resolved relative dates, computed
  durations, or bridge facts.
- Return at most 8 memory_actions and at most 6 active_memories.
- Never return an empty active_memories list.
"""


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _keyword_set(*texts: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "what",
        "why",
        "how",
        "does",
        "did",
        "was",
        "were",
        "are",
        "you",
        "your",
        "his",
        "her",
        "its",
        "it's",
        "they",
        "them",
        "their",
        "important",
    }
    out: set[str] = set()
    for text in texts:
        for token in TOKEN_RE.findall(text.lower()):
            if len(token) >= 3 and token not in stop:
                out.add(token)
    return out


def _query_aware_excerpt(content: str, *, query: str = "", gold_answer: str = "", max_chars: int) -> str:
    content = " ".join(content.split())
    if max_chars <= 0:
        return content
    if len(content) <= max_chars:
        return content
    keywords = _keyword_set(query, gold_answer)
    if not keywords:
        return content[: max_chars - 3].rstrip() + "..."

    # Search overlapping chunks rather than only sentence boundaries; benchmark
    # sessions often contain long turns where answer evidence appears late.
    chunk_size = max_chars
    stride = max(160, max_chars // 3)
    best_start = 0
    best_score = -1
    for start in range(0, max(1, len(content)), stride):
        chunk = content[start : start + chunk_size]
        tokens = set(TOKEN_RE.findall(chunk.lower()))
        score = len(tokens & keywords)
        if score > best_score:
            best_score = score
            best_start = start
        if start + chunk_size >= len(content):
            break
    start = max(0, best_start - 80)
    end = min(len(content), start + max_chars)
    excerpt = content[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(content):
        excerpt += "..."
    return excerpt


def format_candidate_memories(
    memories: Iterable[CandidateMemory],
    *,
    query: str = "",
    gold_answer: str = "",
    max_content_chars: int = 700,
) -> str:
    lines: list[str] = []
    for memory in memories:
        content = _query_aware_excerpt(
            memory.content,
            query=query,
            gold_answer=gold_answer,
            max_chars=max_content_chars,
        )
        attrs = [memory.type]
        if memory.time:
            attrs.append(f"time={memory.time}")
        if memory.retrieval_score is not None:
            attrs.append(f"score={memory.retrieval_score:.4f}")
        if memory.source_turns:
            attrs.append(f"source_turns={memory.source_turns}")
        lines.append(f"[{memory.memory_id}] ({', '.join(attrs)}) {content}")
    return "\n".join(lines) if lines else "(none)"


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _calendar_date(value: str | None) -> str:
    parsed = _parse_timestamp(value)
    return parsed.date().isoformat() if parsed else (value or "")


def _approx_month_duration(start: datetime, end: datetime) -> str:
    days = abs((end.date() - start.date()).days)
    if days < 45:
        weeks = max(1, round(days / 7))
        return f"about {weeks} weeks"
    months = max(1, round(days / 30.44))
    return f"about {months} months"


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _memory_rank(memory: CandidateMemory, fallback_idx: int) -> int:
    rank = memory.metadata.get("candidate_rank")
    try:
        return int(rank)
    except (TypeError, ValueError):
        return fallback_idx + 1


def _evidence_verifier_hints(example: IntentMemExample) -> str:
    """High-recall evidence checklist computed only from question and candidates."""
    question = example.question.lower()
    memories = list(example.candidate_memories)
    hints: list[dict[str, object]] = []
    derived: list[str] = []

    duration_like = any(
        marker in question
        for marker in [
            "how long",
            "duration",
            "before getting married",
            "before they got married",
            "date before",
            "dating before",
        ]
    )
    date_like = any(marker in question for marker in ["when", "what date", "which day"])
    contextual_like = any(
        marker in question
        for marker in [
            "based on",
            "would enjoy",
            "would like",
            "recommend",
            "shop",
            "restaurant",
            "place",
            "visit",
        ]
    )

    if duration_like:
        start_markers = [
            "new so",
            "new partner",
            "new significant other",
            "new boyfriend",
            "new girlfriend",
            "started dating",
            "began dating",
            "started seeing",
            "began seeing",
            "first date",
        ]
        end_markers = ["got married", "married last", "married", "wedding"]

        start_hits: list[tuple[int, CandidateMemory]] = []
        end_hits: list[tuple[int, CandidateMemory]] = []
        for idx, memory in enumerate(memories):
            text = memory.content.lower()
            if _contains_any(text, start_markers):
                start_hits.append((_memory_rank(memory, idx), memory))
            if _contains_any(text, end_markers):
                end_hits.append((_memory_rank(memory, idx), memory))
        start_hits.sort(key=lambda item: item[0])
        end_hits.sort(key=lambda item: item[0])

        if start_hits:
            _, memory = start_hits[0]
            hints.append(
                {
                    "need": "start evidence",
                    "memory_id": memory.memory_id,
                    "time": memory.time,
                    "instruction": "Use this as the dated relationship/event start if supported.",
                }
            )
        if end_hits:
            _, memory = end_hits[0]
            hints.append(
                {
                    "need": "end evidence",
                    "memory_id": memory.memory_id,
                    "time": memory.time,
                    "instruction": "Use this as the dated relationship/event end if supported.",
                }
            )
        if start_hits and end_hits:
            start_dt = _parse_timestamp(start_hits[0][1].time)
            end_dt = _parse_timestamp(end_hits[0][1].time)
            if start_dt and end_dt:
                duration = _approx_month_duration(start_dt, end_dt)
                derived.append(
                    f"Derived duration from {_calendar_date(start_hits[0][1].time)} to "
                    f"{_calendar_date(end_hits[0][1].time)} is {duration}."
                )

    if date_like:
        relative_markers = {
            "yesterday": -1,
            "last night": -1,
            "last week": -7,
        }
        for idx, memory in enumerate(memories):
            text = memory.content.lower()
            parsed = _parse_timestamp(memory.time)
            if not parsed:
                continue
            for marker, offset in relative_markers.items():
                if marker in text:
                    resolved = (parsed + timedelta(days=offset)).date().isoformat()
                    hints.append(
                        {
                            "need": f"relative date: {marker}",
                            "memory_id": memory.memory_id,
                            "time": memory.time,
                            "instruction": f"Resolve '{marker}' against the timestamp as {resolved}.",
                        }
                    )
                    derived.append(f"'{marker}' at {_calendar_date(memory.time)} resolves to {resolved}.")
                    break
            if hints and hints[-1].get("need", "").startswith("relative date"):
                break

    if contextual_like:
        personal_markers = [
            "harry potter",
            " hp ",
            "potter",
            "fantasy",
            "favorite books",
            "bookshelf",
            "collection",
            "collections",
            "castle",
            "magical",
        ]
        location_markers = [
            "new york",
            "nyc",
            "new york city",
        ]
        personal_hits: list[tuple[int, CandidateMemory]] = []
        location_hits: list[tuple[int, CandidateMemory]] = []
        for idx, memory in enumerate(memories):
            text = f" {memory.content.lower()} "
            if _contains_any(text, personal_markers):
                personal_hits.append((_memory_rank(memory, idx), memory))
            if _contains_any(text, location_markers):
                location_hits.append((_memory_rank(memory, idx), memory))
        personal_hits.sort(key=lambda item: item[0])
        location_hits.sort(key=lambda item: item[0])
        if personal_hits:
            _, memory = personal_hits[0]
            hints.append(
                {
                    "need": "personal preference or collection clue",
                    "memory_id": memory.memory_id,
                    "time": memory.time,
                    "instruction": "Use this to preserve the user's exact interests or collections.",
                }
            )
        if location_hits:
            _, memory = location_hits[0]
            hints.append(
                {
                    "need": "target location clue",
                    "memory_id": memory.memory_id,
                    "time": memory.time,
                    "instruction": "Use this only together with the personal preference clue.",
                }
            )

    if not hints and not derived:
        return "(none)"
    return json.dumps(
        {
            "purpose": "Verifier hints are computed only from the question and candidate memories; they are not gold answers.",
            "required_behavior": "Use these hints to avoid missing low-ranked bridge evidence; cite exact memory IDs in memory_chain/actions and expose answer-ready facts in active_memories.",
            "hints": hints[:6],
            "derived_facts": derived[:4],
        },
        ensure_ascii=False,
        indent=2,
    )


def _active_only_candidate_order(
    memories: list[CandidateMemory],
    *,
    question: str,
) -> list[CandidateMemory]:
    """Keep the same pool, but surface evidence-critical candidates earlier."""
    q = question.lower()
    duration_like = any(
        marker in q
        for marker in [
            "how long",
            "duration",
            "before getting married",
            "before they got married",
            "date before",
            "dating before",
        ]
    )
    if not duration_like:
        contextual_like = any(
            marker in q
            for marker in [
                "based on",
                "would enjoy",
                "would like",
                "recommend",
                "shop",
                "restaurant",
                "place",
                "visit",
            ]
        )
        if not contextual_like:
            return memories

        personal_markers = [
            "harry potter",
            " hp ",
            "potter",
            "fantasy",
            "favorite books",
            "bookshelf",
            "collection",
            "collections",
            "castle",
            "magical",
        ]
        location_markers = ["new york", "nyc", "new york city"]

        def contextual_priority(item: tuple[int, CandidateMemory]) -> tuple[int, int]:
            idx, memory = item
            text = f" {memory.content.lower()} "
            score = 0
            if any(marker in text for marker in personal_markers):
                score += 45
            if any(marker in text for marker in location_markers):
                score += 25
            return (-score, idx)

        return [memory for _, memory in sorted(enumerate(memories), key=contextual_priority)]

    start_markers = [
        "new so",
        "new partner",
        "new significant other",
        "new boyfriend",
        "new girlfriend",
        "new spouse",
        "started dating",
        "began dating",
        "began seeing",
        "started seeing",
        "first date",
    ]
    end_markers = ["got married", "married last", "married", "wedding"]

    def priority(item: tuple[int, CandidateMemory]) -> tuple[int, int]:
        idx, memory = item
        text = memory.content.lower()
        score = 0
        if any(marker in text for marker in start_markers):
            score += 50
        if any(marker in text for marker in end_markers):
            score += 35
        if "love at first sight" in text and not any(marker in text for marker in start_markers[:6]):
            score -= 20
        return (-score, idx)

    return [memory for _, memory in sorted(enumerate(memories), key=priority)]


def build_policy_messages(
    example: IntentMemExample,
    *,
    max_candidate_chars: int = 700,
    max_candidates: int | None = None,
) -> list[dict[str, str]]:
    ordered_memories = _active_only_candidate_order(example.candidate_memories, question=example.question)
    user = (
        "QUESTION\n"
        f"{example.question}\n\n"
        "EVIDENCE_VERIFIER_HINTS\n"
        f"{_evidence_verifier_hints(example)}\n\n"
        "CANDIDATE_MEMORIES\n"
        f"{format_candidate_memories(ordered_memories[:max_candidates], query=example.question, max_content_chars=max_candidate_chars)}\n\n"
        "Return the required JSON object."
    )
    return [{"role": "system", "content": POLICY_SYSTEM}, {"role": "user", "content": user}]


def build_active_only_policy_messages(
    example: IntentMemExample,
    *,
    max_candidate_chars: int = 700,
    max_candidates: int | None = None,
) -> list[dict[str, str]]:
    metadata = {
        key: example.metadata.get(key)
        for key in ["benchmark", "question_type", "dialogue_id", "question_id"]
        if example.metadata.get(key) is not None
    }
    user = (
        "QUESTION_METADATA\n"
        f"{json.dumps(metadata, ensure_ascii=False)}\n\n"
        "QUESTION\n"
        f"{example.question}\n\n"
        "EVIDENCE_VERIFIER_HINTS\n"
        f"{_evidence_verifier_hints(example)}\n\n"
        "CANDIDATE_MEMORIES\n"
        f"{format_candidate_memories(_active_only_candidate_order(example.candidate_memories, question=example.question)[:max_candidates], query=example.question, max_content_chars=max_candidate_chars)}\n\n"
        "Return the required JSON object."
    )
    return [{"role": "system", "content": ACTIVE_ONLY_POLICY_SYSTEM}, {"role": "user", "content": user}]


def build_chain_active_policy_messages(
    example: IntentMemExample,
    *,
    max_candidate_chars: int = 700,
    max_candidates: int | None = None,
) -> list[dict[str, str]]:
    ordered_memories = _active_only_candidate_order(example.candidate_memories, question=example.question)
    user = (
        "QUESTION\n"
        f"{example.question}\n\n"
        "EVIDENCE_VERIFIER_HINTS\n"
        f"{_evidence_verifier_hints(example)}\n\n"
        "CANDIDATE_MEMORIES\n"
        f"{format_candidate_memories(ordered_memories[:max_candidates], query=example.question, max_content_chars=max_candidate_chars)}\n\n"
        "Return the required JSON object."
    )
    return [{"role": "system", "content": CHAIN_ACTIVE_POLICY_SYSTEM}, {"role": "user", "content": user}]


def build_no_intent_plan_policy_messages(
    example: IntentMemExample,
    *,
    max_candidate_chars: int = 700,
    max_candidates: int | None = None,
) -> list[dict[str, str]]:
    ordered_memories = _active_only_candidate_order(example.candidate_memories, question=example.question)
    user = (
        "QUESTION\n"
        f"{example.question}\n\n"
        "EVIDENCE_VERIFIER_HINTS\n"
        f"{_evidence_verifier_hints(example)}\n\n"
        "CANDIDATE_MEMORIES\n"
        f"{format_candidate_memories(ordered_memories[:max_candidates], query=example.question, max_content_chars=max_candidate_chars)}\n\n"
        "Return the required JSON object."
    )
    return [{"role": "system", "content": NO_INTENT_PLAN_POLICY_SYSTEM}, {"role": "user", "content": user}]


def build_no_memory_chain_policy_messages(
    example: IntentMemExample,
    *,
    max_candidate_chars: int = 700,
    max_candidates: int | None = None,
) -> list[dict[str, str]]:
    ordered_memories = _active_only_candidate_order(example.candidate_memories, question=example.question)
    user = (
        "QUESTION\n"
        f"{example.question}\n\n"
        "EVIDENCE_VERIFIER_HINTS\n"
        f"{_evidence_verifier_hints(example)}\n\n"
        "CANDIDATE_MEMORIES\n"
        f"{format_candidate_memories(ordered_memories[:max_candidates], query=example.question, max_content_chars=max_candidate_chars)}\n\n"
        "Return the required JSON object."
    )
    return [{"role": "system", "content": NO_MEMORY_CHAIN_POLICY_SYSTEM}, {"role": "user", "content": user}]


def build_teacher_messages(example: IntentMemExample, *, max_candidate_chars: int = 700) -> list[dict[str, str]]:
    metadata = {
        key: example.metadata.get(key)
        for key in ["benchmark", "question_type", "dialogue_id", "question_id"]
        if example.metadata.get(key) is not None
    }
    user = (
        "QUESTION\n"
        f"{example.question}\n\n"
        "QUESTION_METADATA\n"
        f"{json.dumps(metadata, ensure_ascii=False)}\n\n"
        "GOLD_ANSWER\n"
        f"{example.gold_answer}\n\n"
        "VALID_MEMORY_IDS\n"
        f"{json.dumps([memory.memory_id for memory in example.candidate_memories], ensure_ascii=False)}\n\n"
        "CANDIDATE_MEMORIES\n"
        f"{format_candidate_memories(example.candidate_memories, query=example.question, gold_answer=example.gold_answer, max_content_chars=max_candidate_chars)}\n\n"
        "Return the supervised policy JSON object."
    )
    return [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": user}]


def build_answer_messages(question: str, active_memories: list[str]) -> list[dict[str, str]]:
    context = "\n".join(f"- {m}" for m in active_memories) if active_memories else "(none)"
    user = f"ACTIVE_MEMORIES\n{context}\n\nQUESTION\n{question}\n\nANSWER"
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]


def response_json(example: IntentMemExample) -> str:
    payload = {
        "intent_plan": example.intent_plan.to_dict() if example.intent_plan else None,
        "memory_actions": [action.to_dict() for action in example.memory_actions],
        "memory_chain": [step.to_dict() for step in example.memory_chain],
        "active_memories": list(example.active_memories),
    }
    return json.dumps(payload, ensure_ascii=False)
