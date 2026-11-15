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

    # TTL は秒で保持し、結果の件数とキー単位の非同期ロックを管理する。
    def __init__(self, *, ttl_seconds: int = 600, max_entries: int = 1000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._items: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = Lock()
        self._async_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    # 単調時計で期限を判定し、期限内の結果を読み出す。
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

    # 結果に現在時刻を付け、上限を超えたら利用順の古い項目を追い出す。
    def put(self, key: str, value: T) -> None:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            self._items[key] = (now, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    # 同じキーの処理を直列化し、戻り値とキャッシュ再利用の有無を返す。
    async def execute(
        self,
        key: str,
        operation: Callable[[], Awaitable[T]],
    ) -> tuple[T, bool]:
        # 保存済みの結果があればロックを待たずに返す。
        existing = self.get(key)
        if existing is not None:
            return existing, True

        async_lock = self._async_locks.get(key)
        if async_lock is None:
            async_lock = asyncio.Lock()
            self._async_locks[key] = async_lock

        # 待機中に同じキーの処理が完了することがあるため、ロック取得後に再確認する。
        async with async_lock:
            existing = self.get(key)
            if existing is not None:
                return existing, True

            # 正常に完了した処理だけを後続の呼び出しで再利用する。
            result = await operation()
            self.put(key, result)
            return result, False

    # 利用順と作成時刻は別なので、先頭だけでなく全項目の期限を確認する。
    def _purge(self, now: float) -> None:
        expired = [
            key
            for key, (created_at, _) in self._items.items()
            if now - created_at > self.ttl_seconds
        ]
        for key in expired:
            self._items.pop(key, None)
