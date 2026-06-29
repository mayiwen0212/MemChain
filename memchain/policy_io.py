"""Policy IO helpers shared by MemChain scripts."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from memchain.prompts import build_policy_messages, response_json
from memchain.schema import (
    INTENTS,
    SUFFICIENCY_LABELS,
    MemChainExample,
    IntentPlan,
    MemoryAction,
    MemoryChainStep,
)

INTENT_ALIASES = {
    "open_domain": "open_domain_contextual_qa",
    "open-domain": "open_domain_contextual_qa",
    "open_domain_qa": "open_domain_contextual_qa",
    "contextual_qa": "open_domain_contextual_qa",
    "temporal_reasoning": "temporal_state_tracking",
    "project_state_tracking": "temporal_state_tracking",
    "preference_query": "preference_recall",
    "single_hop": "fact_lookup",
    "single-hop": "fact_lookup",
    "multi_hop": "multi_hop_relation",
    "multi-hop": "multi_hop_relation",
    "conflict_resolution": "conflict_update",
}

SAFE_POLICY_REPAIR_PREFIXES = (
    "mapped_intent_alias:",
    "mapped_memory_id_suffix:",
    "retried_after_json_parse_failure",
    "added_missing_stop",
    "filled_active_from_selected_chain_or_actions",
    "filled_active_from_top_candidates",
    "recovered_missing_memory_id_quote",
)

OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"
OFFICIAL_OPENAI_HOSTS = {"api.openai.com"}
_MEMORY_ID_MISSING_QUOTE_RE = re.compile(
    r'("[A-Za-z0-9_.:-]+(?::fact|:temporal|:session):\d+),\s*(")'
)


def env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def is_official_openai_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.scheme == "https" and parsed.hostname in OFFICIAL_OPENAI_HOSTS


def unsafe_policy_repairs(repairs: list[str]) -> list[str]:
    return [
        repair
        for repair in repairs
        if not any(repair.startswith(prefix) for prefix in SAFE_POLICY_REPAIR_PREFIXES)
    ]


def load_dotenv(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_json_value(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        repaired = _MEMORY_ID_MISSING_QUOTE_RE.sub(r'\1", \2', stripped)
        if repaired != stripped:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        start_candidates = [idx for idx in [stripped.find("{"), stripped.find("[")] if idx >= 0]
        if not start_candidates:
            raise
        decoder = json.JSONDecoder()
        value, _ = decoder.raw_decode(repaired[min(start_candidates) :] if repaired != stripped else stripped[min(start_candidates) :])
        return value


def _resolve_memory_ref(ref: str, candidate_ids: set[str]) -> tuple[str | None, str | None]:
    if ref in candidate_ids:
        return ref, None
    matches = [candidate_id for candidate_id in candidate_ids if candidate_id.endswith(f":{ref}")]
    if len(matches) == 1:
        return matches[0], f"mapped_memory_id_suffix:{ref}->{matches[0]}"
    return None, None


def apply_policy_payload(
    base: MemChainExample,
    payload: dict[str, Any],
    *,
    preserve_missing_intent_plan: bool = False,
) -> MemChainExample:
    active_raw = payload.get("active_memories", [])
    active: list[str] = []
    if isinstance(active_raw, list):
        for item in active_raw:
            if isinstance(item, dict):
                content = str(item.get("content", "")).strip()
            else:
                content = str(item).strip()
            if content:
                active.append(content)
    actions_raw = payload.get("memory_actions", payload.get("actions", []))
    raw_actions = [MemoryAction.from_dict(row) for row in actions_raw if isinstance(row, dict)]
    repairs: list[str] = []
    candidate_ids = base.candidate_id_set()
    actions: list[MemoryAction] = []
    for action in raw_actions:
        if action.action == "STOP":
            continue
        refs = action.referenced_ids()
        resolved_refs: dict[str, str] = {}
        resolve_repairs: list[str] = []
        unknown: list[str] = []
        for ref in refs:
            resolved, repair = _resolve_memory_ref(ref, candidate_ids)
            if resolved is None:
                unknown.append(ref)
                continue
            resolved_refs[ref] = resolved
            if repair:
                resolve_repairs.append(repair)
        if unknown:
            repairs.append(f"dropped_action_unknown_memory_id:{','.join(unknown)}")
            continue
        if resolve_repairs:
            repairs.extend(resolve_repairs)
            if action.memory_id:
                action.memory_id = resolved_refs[action.memory_id]
            if action.memory_ids:
                action.memory_ids = [resolved_refs[ref] for ref in action.memory_ids]
        actions.append(action)

    chain_raw = payload.get("memory_chain", payload.get("memchain", []))
    memory_chain: list[MemoryChainStep] = []
    for idx, row in enumerate(chain_raw if isinstance(chain_raw, list) else []):
        if not isinstance(row, dict):
            continue
        step = MemoryChainStep.from_dict(row)
        if not step.step_id:
            step.step_id = f"c{idx + 1}"
        resolved_ids: list[str] = []
        unknown_chain_refs: list[str] = []
        for ref in step.memory_ids:
            resolved, repair = _resolve_memory_ref(ref, candidate_ids)
            if resolved is None:
                unknown_chain_refs.append(ref)
                continue
            resolved_ids.append(resolved)
            if repair:
                repairs.append(repair)
        if unknown_chain_refs:
            repairs.append(f"dropped_chain_step_unknown_memory_id:{','.join(unknown_chain_refs)}")
            continue
        step.memory_ids = resolved_ids
        if step.statement:
            memory_chain.append(step)

    if not active:
        for step in memory_chain:
            if step.statement:
                active.append(step.statement)
        for action in actions:
            output = str(action.output or "").strip()
            if action.action in {"KEEP", "MERGE", "REFINE", "ADD"} and output and output not in active:
                active.append(output)
        if active:
            repairs.append("filled_active_from_selected_chain_or_actions")
    if not active and base.candidate_memories:
        for memory in base.candidate_memories[:3]:
            content = memory.content.strip()
            if content:
                active.append(content)
        if active:
            repairs.append("filled_active_from_top_candidates")

    intent_plan: IntentPlan | None
    if preserve_missing_intent_plan and "intent_plan" not in payload:
        intent_plan = None
    else:
        intent_plan = IntentPlan.from_dict(payload.get("intent_plan"))
        if intent_plan.intent in INTENT_ALIASES:
            repairs.append(f"mapped_intent_alias:{intent_plan.intent}")
            intent_plan.intent = INTENT_ALIASES[intent_plan.intent]
        if intent_plan.intent not in INTENTS:
            repairs.append(f"mapped_unknown_intent:{intent_plan.intent}")
            intent_plan.intent = "fact_lookup"

    raw_sufficiency = payload.get("sufficiency")
    if raw_sufficiency is None:
        sufficiency = "enough" if active else "needs_more"
    else:
        sufficiency = str(raw_sufficiency).strip() or ("enough" if active else "needs_more")
        if sufficiency not in SUFFICIENCY_LABELS:
            repairs.append(f"mapped_unknown_sufficiency:{sufficiency}")
            sufficiency = "enough" if active else "needs_more"

    metadata = dict(base.metadata)
    if repairs:
        metadata["policy_repairs"] = repairs

    return MemChainExample(
        sample_id=base.sample_id,
        question=base.question,
        gold_answer=base.gold_answer,
        candidate_memories=list(base.candidate_memories),
        intent_plan=intent_plan,
        memory_actions=actions,
        memory_chain=memory_chain,
        active_memories=active,
        sufficiency=sufficiency,
        metadata=metadata,
    )


def to_sft_row(example: MemChainExample) -> dict[str, Any]:
    return {
        "messages": build_policy_messages(example),
        "response": response_json(example),
        "metadata": {
            **example.metadata,
            "sample_id": example.sample_id,
            "intent": example.intent_plan.intent if example.intent_plan else None,
        },
    }


class ChatClient:
    """Small OpenAI-compatible sync client for long CLI jobs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    @classmethod
    def from_env(cls, *, timeout: float = 120.0, max_retries: int = 4) -> "ChatClient":
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or OFFICIAL_OPENAI_BASE_URL
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set in env or .env")
        if env_flag_enabled("OPENAI_REQUIRE_OFFICIAL") and not is_official_openai_base_url(base_url):
            raise RuntimeError(
                "OPENAI_REQUIRE_OFFICIAL=1 but OPENAI_BASE_URL is not the official OpenAI endpoint. "
                f"Set OPENAI_BASE_URL={OFFICIAL_OPENAI_BASE_URL} or unset the third-party gateway before labeling."
            )
        return cls(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        json_object: bool = False,
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code >= 400 and "enable_thinking" in resp.text.lower():
                    payload.pop("chat_template_kwargs", None)
                    resp = requests.post(
                        self.url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=self.timeout,
                    )
                resp.raise_for_status()
                message = resp.json()["choices"][0]["message"]
                content = message.get("content")
                if content is None:
                    content = ""
                return str(content).strip()
            except Exception as exc:  # noqa: BLE001 - CLI jobs should retry broadly.
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(30.0, 2.0 * attempt))
        assert last_error is not None
        raise last_error
