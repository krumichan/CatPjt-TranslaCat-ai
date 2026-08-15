from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.features.language_learning.speaking.errors import SpeakingStageException

T = TypeVar("T")


async def run_with_stage_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
) -> T:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except SpeakingStageException as exc:
            last_error = exc
            if not exc.retryable or attempt >= max_retries:
                raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                raise

    assert last_error is not None
    raise last_error
