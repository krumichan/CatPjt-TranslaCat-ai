from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.features.language_learning.listening.errors import ListeningStageException

T = TypeVar("T")


async def run_with_stage_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
) -> T:
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except ListeningStageException as exc:
            if not exc.retryable or attempt >= max_retries:
                raise
    raise AssertionError("unreachable")
