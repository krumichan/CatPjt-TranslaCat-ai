from __future__ import annotations

import asyncio
import urllib.error
import urllib.request


class PresignedAudioUploadError(RuntimeError):
    pass


class PresignedAudioUploader:
    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def put(
        self,
        upload_url: str,
        audio_bytes: bytes,
        content_type: str,
    ) -> None:
        await asyncio.to_thread(
            self._put_sync,
            upload_url,
            audio_bytes,
            content_type,
        )

    def _put_sync(
        self,
        upload_url: str,
        audio_bytes: bytes,
        content_type: str,
    ) -> None:
        request = urllib.request.Request(
            upload_url,
            data=audio_bytes,
            method="PUT",
            headers={"Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                status = int(getattr(response, "status", 200) or 200)
                if status < 200 or status >= 300:
                    raise PresignedAudioUploadError(
                        f"Presigned audio upload failed with status {status}"
                    )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PresignedAudioUploadError(
                "Presigned audio upload failed"
            ) from exc
