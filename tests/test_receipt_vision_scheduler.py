from __future__ import annotations

import asyncio
import time
from contextlib import suppress

import pytest

from app.core.config import settings
from app.features.receipt.vision_scheduler import (
    ReceiptVisionQueueFullError,
    ReceiptVisionScheduler,
)


def test_cancel_releases_receipt_provider_permit_for_follow_up_call():
    async def exercise() -> None:
        scheduler = ReceiptVisionScheduler(
            max_in_flight=1, max_recovery_in_flight=1, max_pending=2
        )
        entered = asyncio.Event()
        blocked = asyncio.Event()

        async def wait_forever(_slot):
            entered.set()
            await blocked.wait()

        task = asyncio.create_task(
            scheduler.run(
                kind="WHOLE_IMAGE",
                deadline=time.monotonic() + 1,
                invoke=wait_forever,
            )
        )
        await entered.wait()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        assert scheduler.snapshot.active == 0
        assert scheduler.snapshot.pending == 0
        result = await scheduler.run(
            kind="WHOLE_IMAGE",
            deadline=time.monotonic() + 1,
            invoke=lambda _slot: asyncio.sleep(0, result="ok"),
        )
        assert result == "ok"

    asyncio.run(exercise())


def test_bounded_pending_queue_rejects_before_starting_another_provider_call():
    async def exercise() -> None:
        scheduler = ReceiptVisionScheduler(
            max_in_flight=1, max_recovery_in_flight=1, max_pending=1
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        actual_calls = 0

        async def block(_slot):
            nonlocal actual_calls
            actual_calls += 1
            entered.set()
            await release.wait()

        first = asyncio.create_task(
            scheduler.run(
                kind="WHOLE_IMAGE",
                deadline=time.monotonic() + 1,
                invoke=block,
            )
        )
        await entered.wait()
        with pytest.raises(ReceiptVisionQueueFullError):
            await scheduler.run(
                kind="WHOLE_IMAGE",
                deadline=time.monotonic() + 1,
                invoke=block,
            )
        assert actual_calls == 1
        release.set()
        await first

    asyncio.run(exercise())


def test_two_crop_calls_leave_capacity_for_a_whole_image_call():
    async def exercise() -> None:
        scheduler = ReceiptVisionScheduler(
            max_in_flight=3, max_recovery_in_flight=2, max_pending=6
        )
        recoveries_entered = asyncio.Event()
        whole_entered = asyncio.Event()
        release = asyncio.Event()
        recovery_count = 0

        async def recovery(_slot):
            nonlocal recovery_count
            recovery_count += 1
            if recovery_count == 2:
                recoveries_entered.set()
            await release.wait()

        async def whole(_slot):
            whole_entered.set()
            await release.wait()

        recovery_tasks = [
            asyncio.create_task(
                scheduler.run(
                    kind="RECOVERY",
                    deadline=time.monotonic() + 1,
                    invoke=recovery,
                )
            )
            for _ in range(2)
        ]
        await recoveries_entered.wait()
        whole_task = asyncio.create_task(
            scheduler.run(
                kind="WHOLE_IMAGE",
                deadline=time.monotonic() + 1,
                invoke=whole,
            )
        )
        await asyncio.wait_for(whole_entered.wait(), timeout=0.2)
        assert scheduler.snapshot.active == 3
        release.set()
        await asyncio.gather(*recovery_tasks, whole_task)

    asyncio.run(exercise())


def test_receipt_vision_defaults_disable_ocr_warmup_and_bound_concurrency():
    assert settings.RECEIPT_VISION_DISABLE_OCR_WARMUP is True
    assert settings.RECEIPT_ANALYSIS_MODE == "VISION_ONLY"
    assert settings.RECEIPT_VISION_MAX_IMAGE_PIXELS == 24_000_000
    assert settings.RECEIPT_VISION_MAX_IN_FLIGHT == 3
    assert settings.RECEIPT_VISION_MAX_RECOVERY_IN_FLIGHT == 2
