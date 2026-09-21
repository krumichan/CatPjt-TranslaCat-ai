"""Offline shared-pool boundaries; these tests do not validate audio or language."""

import asyncio

import pytest

from app.ai.provider_pool import AiProviderPool
from app.common.idempotency import InMemoryIdempotencyStore


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["cancel", "timeout", "failure"])
async def test_one_mode_termination_does_not_cancel_peers_or_close_shared_pool(termination):
    entered = {mode: asyncio.Event() for mode in ("VOCABULARY", "LISTENING", "SPEAKING")}
    releases = {mode: asyncio.Event() for mode in entered}

    class Provider:
        ready = True
        closed = False

        async def call(self, type_name, data, schema=None):
            entered[type_name].set()
            await releases[type_name].wait()
            if type_name == "VOCABULARY" and termination == "failure":
                raise ValueError("invalid verifier structure")
            return type_name

        async def shutdown(self):
            self.closed = True

    provider = Provider()
    pool = AiProviderPool()
    pool.dock(name="fake", provider=provider, max_concurrency=3, cooldown_seconds=0)

    async def invoke(mode):
        if mode == "VOCABULARY" and termination == "timeout":
            async with asyncio.timeout(0.02):
                return await pool.call(mode, "no private content")
        return await pool.call(mode, "no private content")

    tasks = {mode: asyncio.create_task(invoke(mode)) for mode in entered}
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 1)
        if termination == "cancel":
            tasks["VOCABULARY"].cancel()
        elif termination == "failure":
            releases["VOCABULARY"].set()
        expected = {"cancel": asyncio.CancelledError, "timeout": TimeoutError, "failure": ValueError}
        with pytest.raises(expected[termination]):
            await tasks["VOCABULARY"]
        assert not provider.closed
        assert not tasks["LISTENING"].done()
        assert not tasks["SPEAKING"].done()
        releases["LISTENING"].set()
        releases["SPEAKING"].set()
        assert await tasks["LISTENING"] == "LISTENING"
        assert await tasks["SPEAKING"] == "SPEAKING"
        # The cancelled slot released its semaphore; the same pool remains usable.
        assert await asyncio.wait_for(pool.call("LISTENING", "next request"), 1) == "LISTENING"
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_idempotency_leader_does_not_cache_failure_or_cancel_waiter():
    store = InMemoryIdempotencyStore[str]()
    entered = asyncio.Event()

    async def cancelled_operation():
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    async def subsequent_operation():
        return "completed"

    leader = asyncio.create_task(store.execute("session:request", cancelled_operation))
    await entered.wait()
    waiter = asyncio.create_task(store.execute("session:request", subsequent_operation))
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    assert await asyncio.wait_for(waiter, 1) == ("completed", False)
    assert store.get("session:request") == "completed"
