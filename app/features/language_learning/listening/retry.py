from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.features.language_learning.listening.errors import ListeningStageException

T = TypeVar("T")

logger = logging.getLogger(__name__)


async def run_with_stage_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    context: str | None = None,
) -> T:
    total_attempts = max_retries + 1
    for attempt in range(total_attempts):
        try:
            return await operation()
        except ListeningStageException as exc:
            attempt_number = attempt + 1
            will_retry = exc.retryable and attempt < max_retries
            log = logger.warning if will_retry else logger.error
            log(
                "Listening stage operation failed. context=%s attempt=%d/%d code=%s "
                "stage=%s retryable=%s will_retry=%s message=%s",
                context or "-",
                attempt_number,
                total_attempts,
                exc.code.value,
                exc.stage.value,
                exc.retryable,
                will_retry,
                exc.message,
            )
            if not will_retry:
                raise
    raise AssertionError("unreachable")
