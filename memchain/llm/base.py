"""LLMClient abstraction.

Single async surface that all backends implement. The interface stays minimal:
`complete` plus dataclasses, so research code can plug in mocks easily and
backend-specific options can be added without breaking the protocol.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Protocol, TypeVar


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class CompletionResult:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    """Async LLM completion surface.

    Implementations are responsible for backend auth, retries, and rate limiting.
    """

    model: str

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> CompletionResult: ...


T = TypeVar("T")


def run_sync(coro: Awaitable[T]) -> T:
    """Run an async coroutine from synchronous code.

    Trainers and CLI scripts can call this without owning an event loop. If we're
    already inside a running loop (e.g., notebook context), we surface that as an
    error so the caller switches to ``await`` rather than silently double-entering.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    raise RuntimeError(
        "run_sync called from inside a running event loop; await the coroutine directly."
    ) from None
