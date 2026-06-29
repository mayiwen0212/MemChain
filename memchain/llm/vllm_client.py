"""Local vLLM backend for LLMClient.

Talks to a vLLM OpenAI-compatible HTTP server. We use httpx async rather than the
in-process vLLM library so that the same client works whether vLLM runs locally,
in a sidecar, or on a separate machine — and so that PyTorch is not required to
import this file. Phase 2 RL training uses this for the policy and the answer model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memchain.llm.base import CompletionResult, Message


@dataclass
class VLLMClient:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    timeout: float = 120.0

    def __post_init__(self) -> None:
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._client

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        client = self._ensure_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop is not None:
            payload["stop"] = stop
        payload.update(kwargs)

        resp = await client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return CompletionResult(
            text=choice["message"]["content"] or "",
            raw=data,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
