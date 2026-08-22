from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from threading import Lock
from typing import Generic, TypeVar
from weakref import WeakValueDictionary

T = TypeVar("T")


class InMemoryIdempotencyStore(Generic[T]):
    """Small bounded TTL store that also coalesces concurrent duplicate calls."""

    def __init__(self, *, ttl_seconds: int = 600, max_entries: int = 1000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._items: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = Lock()
        self._async_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            value = self._items.get(key)
            if value is None:
                return None
            created_at, item = value
            self._items.move_to_end(key)
            if now - created_at > self.ttl_seconds:
                self._items.pop(key, None)
                return None
            return item

    def put(self, key: str, value: T) -> None:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            self._items[key] = (now, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    async def execute(
        self,
        key: str,
        operation: Callable[[], Awaitable[T]],
    ) -> tuple[T, bool]:
        existing = self.get(key)
        if existing is not None:
            return existing, True

        async_lock = self._async_locks.get(key)
        if async_lock is None:
            async_lock = asyncio.Lock()
            self._async_locks[key] = async_lock

        async with async_lock:
            existing = self.get(key)
            if existing is not None:
                return existing, True
            result = await operation()
            self.put(key, result)
            return result, False

    def _purge(self, now: float) -> None:
        expired = [
            key
            for key, (created_at, _) in self._items.items()
            if now - created_at > self.ttl_seconds
        ]
        for key in expired:
            self._items.pop(key, None)
