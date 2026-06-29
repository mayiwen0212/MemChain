"""OpenAI backend for LLMClient.

Delegates to the official ``openai`` SDK's async client. The class only constructs the
client lazily so that environments without the SDK installed can still import the
module (the import error is deferred to the first call).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memchain.llm.base import CompletionResult, Message


@dataclass
class OpenAIClient:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI  # local import keeps SDK optional at import-time

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
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
        resp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return CompletionResult(
            text=choice.message.content or "",
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )
