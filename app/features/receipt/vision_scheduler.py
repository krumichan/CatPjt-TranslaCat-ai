from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from app.core.config import settings


T = TypeVar("T")
VisionCallKind = Literal["WHOLE_IMAGE", "RECOVERY"]


class ReceiptVisionQueueFullError(RuntimeError):
    """The bounded receipt provider queue has no remaining capacity."""


class ReceiptVisionDeadlineExceeded(TimeoutError):
    """The request's single end-to-end Vision deadline was exhausted."""


@dataclass(frozen=True)
class ReceiptVisionCallStart:
    queue_wait_ms: int
    in_flight_at_start: int
    recovery_in_flight_at_start: int
    started_at: float


@dataclass(frozen=True)
class ReceiptVisionSchedulerSnapshot:
    active: int
    active_recoveries: int
    pending: int
    max_observed_active: int
    max_observed_recoveries: int


class ReceiptVisionScheduler(Generic[T]):
    """One-process limiter for full-image and selective-crop provider calls.

    The deployed AI image and the isolated launcher both use one Uvicorn worker.
    The process-wide limit is therefore also the server-wide limit for that
    topology. A multi-worker deployment must divide the configured capacity or
    add an external coordinator; this class intentionally makes no cross-process
    claim.
    """

    def __init__(
        self,
        *,
        max_in_flight: int,
        max_recovery_in_flight: int,
        max_pending: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        if not 1 <= max_recovery_in_flight <= max_in_flight:
            raise ValueError("max_recovery_in_flight must be within the total limit")
        if max_pending < max_in_flight:
            raise ValueError("max_pending must be at least max_in_flight")
        self._total = asyncio.Semaphore(max_in_flight)
        self._recovery = asyncio.Semaphore(max_recovery_in_flight)
        self._max_pending = max_pending
        self._clock = clock
        self._active = 0
        self._active_recoveries = 0
        self._pending = 0
        self._max_observed_active = 0
        self._max_observed_recoveries = 0

    @property
    def snapshot(self) -> ReceiptVisionSchedulerSnapshot:
        return ReceiptVisionSchedulerSnapshot(
            active=self._active,
            active_recoveries=self._active_recoveries,
            pending=self._pending,
            max_observed_active=self._max_observed_active,
            max_observed_recoveries=self._max_observed_recoveries,
        )

    async def run(
        self,
        *,
        kind: VisionCallKind,
        deadline: float,
        invoke: Callable[[ReceiptVisionCallStart], Awaitable[T]],
    ) -> T:
        if self._clock() >= deadline:
            raise ReceiptVisionDeadlineExceeded("Receipt Vision deadline exhausted")
        if self._pending >= self._max_pending:
            raise ReceiptVisionQueueFullError("Receipt Vision queue is full")

        self._pending += 1
        queued_at = self._clock()
        recovery_acquired = False
        total_acquired = False
        try:
            if kind == "RECOVERY":
                await self._acquire(self._recovery, deadline)
                recovery_acquired = True
            await self._acquire(self._total, deadline)
            total_acquired = True

            self._active += 1
            if kind == "RECOVERY":
                self._active_recoveries += 1
            self._max_observed_active = max(self._max_observed_active, self._active)
            self._max_observed_recoveries = max(
                self._max_observed_recoveries, self._active_recoveries
            )
            started_at = self._clock()
            start = ReceiptVisionCallStart(
                queue_wait_ms=max(0, round((started_at - queued_at) * 1000)),
                in_flight_at_start=self._active,
                recovery_in_flight_at_start=self._active_recoveries,
                started_at=started_at,
            )
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise ReceiptVisionDeadlineExceeded("Receipt Vision deadline exhausted")
            try:
                async with asyncio.timeout(remaining):
                    return await invoke(start)
            except TimeoutError as exc:
                raise ReceiptVisionDeadlineExceeded(
                    "Receipt Vision provider call exceeded the shared deadline"
                ) from exc
            finally:
                self._active = max(0, self._active - 1)
                if kind == "RECOVERY":
                    self._active_recoveries = max(0, self._active_recoveries - 1)
        finally:
            if total_acquired:
                self._total.release()
            if recovery_acquired:
                self._recovery.release()
            self._pending = max(0, self._pending - 1)

    async def _acquire(self, semaphore: asyncio.Semaphore, deadline: float) -> None:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise ReceiptVisionDeadlineExceeded("Receipt Vision deadline exhausted")
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=remaining)
        except TimeoutError as exc:
            raise ReceiptVisionDeadlineExceeded(
                "Receipt Vision queue wait exceeded the shared deadline"
            ) from exc


_SCHEDULERS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, ReceiptVisionScheduler[object]
] = weakref.WeakKeyDictionary()


def get_receipt_vision_scheduler() -> ReceiptVisionScheduler[object]:
    loop = asyncio.get_running_loop()
    scheduler = _SCHEDULERS.get(loop)
    if scheduler is None:
        scheduler = ReceiptVisionScheduler(
            max_in_flight=settings.RECEIPT_VISION_MAX_IN_FLIGHT,
            max_recovery_in_flight=settings.RECEIPT_VISION_MAX_RECOVERY_IN_FLIGHT,
            max_pending=settings.RECEIPT_VISION_MAX_PENDING,
        )
        _SCHEDULERS[loop] = scheduler
    return scheduler
