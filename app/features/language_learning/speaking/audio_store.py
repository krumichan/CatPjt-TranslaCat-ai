from __future__ import annotations

import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class StoredAudio:
    reference: str
    path: Path
    content_type: str
    created_at: float


class TemporaryTtsAudioStore:
    def __init__(self, *, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self.base_dir = Path(tempfile.gettempdir()) / "translacat-speaking-tts"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, StoredAudio] = {}
        self._lock = Lock()

    def put(self, audio_bytes: bytes, *, cache_key: str) -> StoredAudio:
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:32]
        reference = f"tts-{digest}"
        path = self.base_dir / f"{reference}.wav"
        with self._lock:
            self._purge_locked()
            if not path.exists():
                temp_path = path.with_suffix(".tmp")
                temp_path.write_bytes(audio_bytes)
                os.replace(temp_path, path)
            stored = StoredAudio(
                reference=reference,
                path=path,
                content_type="audio/wav",
                created_at=time.time(),
            )
            self._items[reference] = stored
            return stored

    def get(self, reference: str) -> StoredAudio | None:
        with self._lock:
            self._purge_locked()
            item = self._items.get(reference)
            if item is None or not item.path.exists():
                return None
            return item

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
