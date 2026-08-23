from __future__ import annotations

import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class StoredListeningAudio:
    reference: str
    path: Path
    content_type: str
    duration_seconds: float
    checksum: str
    created_at: float


class TemporaryListeningAudioStore:
    def __init__(self, *, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self.base_dir = Path(tempfile.gettempdir()) / "translacat-listening-tts"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, StoredListeningAudio] = {}
        self._lock = Lock()

    @staticmethod
    def reference_for(cache_key: str) -> str:
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:32]
        return f"listening-tts-{digest}"

    def put(
        self,
        audio_bytes: bytes,
        *,
        cache_key: str,
        content_type: str,
        duration_seconds: float,
    ) -> StoredListeningAudio:
        reference = self.reference_for(cache_key)
        path = self.base_dir / f"{reference}.audio"
        checksum = hashlib.sha256(audio_bytes).hexdigest()
        with self._lock:
            self._purge_locked()
            if not path.exists():
                temporary_path = path.with_suffix(".tmp")
                temporary_path.write_bytes(audio_bytes)
                os.replace(temporary_path, path)
            stored = StoredListeningAudio(
                reference=reference,
                path=path,
                content_type=content_type,
                duration_seconds=duration_seconds,
                checksum=checksum,
                created_at=time.time(),
            )
            self._items[reference] = stored
            return stored

    def get(self, reference: str) -> StoredListeningAudio | None:
        with self._lock:
            self._purge_locked()
            stored = self._items.get(reference)
            if stored is None or not stored.path.exists():
                return None
            return stored

    def _purge_locked(self) -> None:
        now = time.time()
        expired = [
            reference
            for reference, item in self._items.items()
            if now - item.created_at > self.ttl_seconds
        ]
        for reference in expired:
            item = self._items.pop(reference)
            item.path.unlink(missing_ok=True)
