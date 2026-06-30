"""Unified dialogue data model for MemChain.

The method code only needs a small normalized shape: dialogues contain sessions,
sessions contain utterances, and QA pairs point to questions over the dialogue
history. Loader-specific benchmark adapters can live outside this core package.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import jsonlines
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback for eval envs.
    jsonlines = None
try:
    from dataclasses_json import (
        CatchAll,
        DataClassJsonMixin,
        Undefined,
        config,
        dataclass_json,
    )
    from marshmallow import fields as mm_fields
except ModuleNotFoundError:  # pragma: no cover - eval env fallback.
    CatchAll = dict

    class DataClassJsonMixin:
        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

        @classmethod
        def from_dict(cls, data: dict[str, Any]):
            name = cls.__name__
            if name == "Utterance":
                return cls(
                    speaker=data.get("speaker", ""),
                    text=data.get("text", ""),
                    timestamp=_ts_decoder(data.get("timestamp")),
                )
            if name == "Session":
                return cls(
                    session_id=data.get("session_id", ""),
                    utterances=[Utterance.from_dict(row) for row in data.get("utterances", [])],
                    timestamp=_ts_decoder(data.get("timestamp")),
                    tags=list(data.get("tags", []) or []),
                )
            if name == "QAPair":
                return cls(
                    question_id=data.get("question_id", ""),
                    question=data.get("question", ""),
                    gold_answer=data.get("gold_answer", ""),
                    question_type=data.get("question_type"),
                    answer_session_ids=list(data.get("answer_session_ids", []) or []),
                    metadata=dict(data.get("metadata", {}) or {}),
                )
            if name == "Dialogue":
                known = {"dialogue_id", "sessions", "qa_pairs", "metadata"}
                return cls(
                    dialogue_id=data.get("dialogue_id", ""),
                    sessions=[Session.from_dict(row) for row in data.get("sessions", [])],
                    qa_pairs=[QAPair.from_dict(row) for row in data.get("qa_pairs", [])],
                    metadata=dict(data.get("metadata", {}) or {}),
                    _extras={k: v for k, v in data.items() if k not in known},
                )
            if name == "Benchmark":
                return cls(
                    name=data.get("name", ""),
                    dialogues=[Dialogue.from_dict(row) for row in data.get("dialogues", [])],
                    metadata=dict(data.get("metadata", {}) or {}),
                )
            return cls(**data)

    class Undefined:
        EXCLUDE = "exclude"

    def config(**_kwargs: Any) -> dict[str, Any]:
        return {}

    def dataclass_json(**_kwargs: Any):
        def decorator(cls):
            if not hasattr(cls, "to_dict"):
                cls.to_dict = DataClassJsonMixin.to_dict
            if not hasattr(cls, "from_dict"):
                cls.from_dict = classmethod(DataClassJsonMixin.from_dict.__func__)
            return cls

        return decorator

    class _FallbackFields:
        @staticmethod
        def DateTime(**_kwargs: Any) -> None:
            return None

    mm_fields = _FallbackFields()


def _ts_encoder(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _ts_decoder(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


_TIMESTAMP_CFG = config(
    encoder=_ts_encoder,
    decoder=_ts_decoder,
    mm_field=mm_fields.DateTime(allow_none=True),
)


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Utterance:
    """One turn in a session.

    ``timestamp`` is None when the source benchmark doesn't carry per-turn
    timestamps (LoCoMo has session-level only; LongMemEval has neither).
    Don't fabricate dates — None is more honest than a fake.
    """

    speaker: str
    text: str
    timestamp: Optional[datetime] = field(default=None, metadata=_TIMESTAMP_CFG)


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Session:
    session_id: str
    utterances: list[Utterance] = field(default_factory=list)
    timestamp: Optional[datetime] = field(default=None, metadata=_TIMESTAMP_CFG)
    # ``tags`` is the extension hook used by LoCoMo-Long ("original" / "filler")
    # and LoCoMo-Adversarial ("misleading" / "retraction"). Standard benchmarks
    # leave it empty.
    tags: list[str] = field(default_factory=list)


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class QAPair:
    question_id: str
    question: str
    gold_answer: str
    question_type: Optional[str] = None
    # Which session ids the gold answer is grounded in. Some benchmarks ship
    # this as ``answer_session_ids`` (LongMemEval), others don't.
    answer_session_ids: list[str] = field(default_factory=list)
    # Adversarial extension marks questions whose answer was retracted.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Dialogue:
    dialogue_id: str
    sessions: list[Session] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Catch-all bucket for unknown upstream fields so we don't silently lose
    # structure during round-trips. dataclasses_json populates this with any
    # leftover top-level keys.
    _extras: CatchAll = field(default_factory=dict)


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Benchmark:
    """A loaded benchmark — list of dialogues + a name."""

    name: str
    dialogues: list[Dialogue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------- aggregates ----------

    def total_qa_pairs(self) -> int:
        return sum(len(d.qa_pairs) for d in self.dialogues)

    def total_sessions(self) -> int:
        return sum(len(d.sessions) for d in self.dialogues)

    def total_utterances(self) -> int:
        return sum(len(s.utterances) for d in self.dialogues for s in d.sessions)

    def qa_type_counts(self) -> dict[str, int]:
        """Per-question_type counts. Useful for the LoCoMo 5-class assertion."""
        counts: dict[str, int] = {}
        for d in self.dialogues:
            for q in d.qa_pairs:
                key = q.question_type or "unknown"
                counts[key] = counts.get(key, 0) + 1
        return counts

    # ---------- filtering ----------

    def filter_by_type(self, q_type: str) -> "Benchmark":
        """Return a new Benchmark with only QA pairs of a given question_type.

        Dialogues with no surviving QA pairs are dropped. Sessions are kept
        intact — the memory state still depends on every session even when
        we're only scoring a subset of questions.
        """
        new_dialogues: list[Dialogue] = []
        for d in self.dialogues:
            kept = [q for q in d.qa_pairs if q.question_type == q_type]
            if not kept:
                continue
            new_dialogues.append(
                Dialogue(
                    dialogue_id=d.dialogue_id,
                    sessions=list(d.sessions),
                    qa_pairs=kept,
                    metadata=dict(d.metadata),
                )
            )
        return Benchmark(
            name=f"{self.name}/{q_type}",
            dialogues=new_dialogues,
            metadata={**self.metadata, "filter_by_type": q_type},
        )

    # ---------- (de)serialization ----------

    def to_jsonl(self, path: str | Path) -> None:
        """Write one Dialogue per line. Header line carries name + metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"_header": True, "name": self.name, "metadata": self.metadata}
        if jsonlines is not None:
            with jsonlines.open(path, mode="w") as writer:
                writer.write(header)
                for d in self.dialogues:
                    writer.write(d.to_dict())
            return
        with path.open("w", encoding="utf-8") as writer:
            writer.write(json.dumps(header, ensure_ascii=False) + "\n")
            for d in self.dialogues:
                writer.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Benchmark":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        name = path.stem
        metadata: dict[str, Any] = {}
        dialogues: list[Dialogue] = []
        if jsonlines is not None:
            with jsonlines.open(path, mode="r") as reader:
                for i, row in enumerate(reader):
                    if i == 0 and row.get("_header"):
                        name = row.get("name", name)
                        metadata = row.get("metadata", {})
                        continue
                    dialogues.append(Dialogue.from_dict(row))
        else:
            with path.open("r", encoding="utf-8") as reader:
                for i, line in enumerate(reader):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if i == 0 and row.get("_header"):
                        name = row.get("name", name)
                        metadata = row.get("metadata", {})
                        continue
                    dialogues.append(Dialogue.from_dict(row))
        return cls(name=name, dialogues=dialogues, metadata=metadata)


# ---------- helpers shared by loaders ----------


def default_cache_dir(benchmark_name: str) -> Path:
    """Conventional on-disk cache location for a benchmark's normalized jsonl."""
    return Path("data/benchmarks") / benchmark_name / "cache"


def iter_qa_pairs(benchmark: Benchmark) -> Iterable[tuple[Dialogue, QAPair]]:
    """Walk (dialogue, qa_pair) pairs across the whole benchmark."""
    for d in benchmark.dialogues:
        for q in d.qa_pairs:
            yield d, q
