from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import ssl
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from websockets.asyncio.client import connect

from app.features.voice_translation.audio import VadPartial, normalize_voice_pcm
from app.features.voice_translation.speech_detector import VoiceSpeechEvidenceGuard
from app.features.voice_translation.stt import VoiceSttProvider
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import SUPPORTED_VOICE_LANGUAGES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoicePreviewSnapshot:
    source_text: str
    translated_text: str


PreviewReceiver = Callable[[VoicePreviewSnapshot], Awaitable[None]]


class VoicePreviewProvider(Protocol):
    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        receive: PreviewReceiver,
    ) -> None: ...


class OpenAILivePreviewProvider:
    """발화별 연결을 사용해 지연된 인식 조각의 소유권을 보존한다."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        # 연결 도중 공유 인증서 저장소를 변경하지 않는다.
        self._ssl = ssl.create_default_context()

    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        receive: PreviewReceiver,
    ) -> None:
        import av
        import numpy as np

        url = "wss://api.openai.com/v1/realtime?intent=transcription"
        async with connect(
            url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            ssl=self._ssl,
            open_timeout=5,
            close_timeout=1,
            max_size=512 * 1024,
            max_queue=4,
        ) as socket:
            await socket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "transcription",
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 24000},
                                    "transcription": {
                                        "model": "gpt-live-transcribe",
                                        "delay": "low",
                                    },
                                    "noise_reduction": None,
                                    "turn_detection": None,
                                }
                            },
                        },
                    }
                )
            )
            # 설정이 수락되기 전에 음성/유료 추론을 시작하지 않는다.
            async with asyncio.timeout(5):
                while True:
                    event = json.loads(await socket.recv())
                    if event.get("type") == "error":
                        raise RuntimeError("Preview configuration rejected")
                    if event.get("type") == "session.updated":
                        break

            async def send_audio() -> None:
                # 상태를 가진 resampler로 프레임 경계의 필터 연속성을 보존한다.
                resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
                pending = bytearray()

                async def send_frames(frames: list[av.AudioFrame]) -> None:
                    for frame in frames:
                        pending.extend(frame.to_ndarray().astype("<i2").tobytes())
                    while len(pending) >= 9600:
                        chunk = bytes(pending[:9600])
                        del pending[:9600]
                        await socket.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(chunk).decode("ascii"),
                                }
                            )
                        )

                async for pcm in audio:
                    frame = av.AudioFrame.from_ndarray(
                        np.frombuffer(pcm, dtype="<i2").reshape(1, -1),
                        format="s16",
                        layout="mono",
                    )
                    frame.sample_rate = 16000
                    await send_frames(resampler.resample(frame))
                await send_frames(resampler.resample(None))
                if pending:
                    await socket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(pending).decode("ascii"),
                            }
                        )
                    )
                # 발화 끝을 commit하고 해당 item의 최종 인식까지 drain한다.
                await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))

            async def receive_text() -> None:
                source = ""
                event_ids: set[str] = set()
                item_id: str | None = None
                async for raw in socket:
                    event = json.loads(raw)
                    kind = event.get("type")
                    if kind in {"error", "conversation.item.input_audio_transcription.failed"}:
                        raise RuntimeError("Preview provider failed")
                    if kind not in {
                        "conversation.item.input_audio_transcription.delta",
                        "conversation.item.input_audio_transcription.completed",
                    }:
                        continue
                    event_item = event.get("item_id")
                    if not isinstance(event_item, str) or not event_item:
                        raise ValueError("Missing preview item identity")
                    if item_id is not None and item_id != event_item:
                        raise ValueError("Unexpected preview item")
                    item_id = event_item
                    identifier = event.get("event_id")
                    if not isinstance(identifier, str) or not identifier:
                        raise ValueError("Missing preview event identity")
                    if identifier in event_ids:
                        continue
                    if len(event_ids) >= 2048:
                        raise ValueError("Too many preview events")
                    event_ids.add(identifier)
                    complete = kind == "conversation.item.input_audio_transcription.completed"
                    delta = event.get("transcript" if complete else "delta")
                    if not isinstance(delta, str):
                        raise ValueError("Invalid preview text")
                    source = delta if complete else source + delta
                    if len(source) > 10_000:
                        raise ValueError("Preview text exceeds limit")
                    await receive(VoicePreviewSnapshot(source.strip(), ""))
                    if complete:
                        return
                raise RuntimeError("Preview stream closed before drain")

            # 송수신 어느 쪽이 실패해도 다른 쪽을 남기지 않는다. 재전송은 하지 않는다.
            async with asyncio.TaskGroup() as group:
                group.create_task(send_audio())
                group.create_task(receive_text())


@dataclass(frozen=True)
class VoicePreviewUpdate:
    revision: int
    source_text: str
    translated_text: str | None
    detected_language: str
    started_at_offset_ms: int
    ended_at_offset_ms: int


class _PreviewRejected(Exception):
    pass


class VoicePreviewJob:
    """최신 PCM snapshot 하나만 보관하며 확정 파이프라인과 독립적으로 실행한다."""

    def __init__(
        self,
        initial: VadPartial,
        *,
        provider: VoicePreviewProvider,
        language_provider: VoiceSttProvider,
        translation_service: VoiceTranslationService,
        speech_guard: VoiceSpeechEvidenceGuard,
        target_language: str,
        manual_language: str | None,
        emit: Callable[[VoicePreviewUpdate], Awaitable[None]],
        slots: asyncio.Semaphore,
    ) -> None:
        self._initial = initial
        self._provider = provider
        self._language_provider = language_provider
        self._translation_service = translation_service
        self._speech_guard = speech_guard
        self._target = target_language
        self._manual = manual_language
        self._emit = emit
        self._slots = slots
        self._queue: asyncio.Queue[tuple[VadPartial, bool]] = asyncio.Queue(maxsize=1)
        self._queue.put_nowait((initial, False))
        self._active = True
        self._finished = False
        self._language: str | None = None
        self._latest = VoicePreviewSnapshot("", "")
        self._revision = 0
        self._last_emitted: tuple[str, str | None] | None = None
        self._ended_at = initial.ended_at_offset_ms
        self._publish_lock = asyncio.Lock()
        self._source_changed = asyncio.Event()
        self._stream_finished = False
        self._translated_source = ""
        self._translated_text: str | None = None
        self._gain = 1.0
        self._dc = 0.0
        self._started = time.perf_counter()
        self._first_source_ms: int | None = None
        self._first_translation_ms: int | None = None
        self.task = asyncio.create_task(self._run(), name="voice-realtime-preview")

    def offer(self, partial: VadPartial, *, final: bool = False) -> None:
        if not self._active or self._finished or self.task.done():
            return
        self._finished = final
        # 누적 snapshot을 교체해도 소비자는 마지막 전송 byte부터 이어 보낸다.
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait((partial, final))

    def cancel(self) -> None:
        # 취소를 무시하고 돌아오는 공급자도 확정 이후 UI를 갱신하지 못한다.
        self._active = False
        self.task.cancel()

    async def _audio(self) -> AsyncIterator[bytes]:
        import numpy as np

        sent = 0
        while self._active:
            partial, final = await self._queue.get()
            pcm = partial.pcm_bytes
            if len(pcm) > 336_000 or len(pcm) % 2:
                raise ValueError("Invalid preview audio snapshot")
            # 마지막 VAD snapshot은 이미 보낸 후행 무음 일부를 잘라낼 수 있다.
            if len(pcm) < sent:
                if final:
                    return
                raise ValueError("Preview audio moved backwards")
            self._ended_at = partial.ended_at_offset_ms
            if len(pcm) > sent:
                samples = np.frombuffer(pcm[sent:], dtype="<i2").astype(np.float32)
                samples = (samples - self._dc) * self._gain
                yield np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()
                sent = len(pcm)
            if final:
                return

    async def _run(self) -> None:
        # 임시 작업은 슬롯을 기다려 오래된 음성을 재생하지 않고 즉시 양보한다.
        try:
            if self._slots.locked():
                return
            async with self._slots, asyncio.timeout(18):
                import numpy as np

                normalized = normalize_voice_pcm(self._initial.pcm_bytes)
                if not await asyncio.wait_for(self._speech_guard.has_speech(normalized), 1):
                    return
                # 조용한 발화도 동일한 초기 이득으로 보낸다. 프레임마다 음량을 흔들지 않는다.
                samples = np.frombuffer(self._initial.pcm_bytes, dtype="<i2").astype(np.float32)
                self._dc = float(samples.mean())
                rms = float(np.sqrt(np.mean((samples - self._dc) ** 2)))
                if rms > 0:
                    normalized_samples = np.frombuffer(normalized, dtype="<i2").astype(np.float32)
                    self._gain = float(np.sqrt(np.mean(normalized_samples**2))) / rms
                async with asyncio.TaskGroup() as group:
                    group.create_task(self._check_language(normalized))
                    group.create_task(self._transcribe())
                    group.create_task(self._translate())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 음성/원문/공급자 응답 본문을 로그에 남기지 않는다. 확정 처리는 계속된다.
            logger.info("Voice realtime preview skipped kind=%s", type(exc).__name__)
        finally:
            logger.info(
                "Voice realtime preview finished duration_ms=%s first_source_ms=%s "
                "first_translation_ms=%s revisions=%s",
                max(0, int((time.perf_counter() - self._started) * 1000)),
                self._first_source_ms,
                self._first_translation_ms,
                self._revision,
            )
            self._active = False
            self._initial = VadPartial(b"", 0, 0)
            while not self._queue.empty():
                self._queue.get_nowait()

    async def _check_language(self, pcm: bytes) -> None:
        result = await asyncio.wait_for(
            self._language_provider.transcribe_pcm(
                pcm,
                language=None,
                is_final=False,
                initial_prompt=None,
            ),
            3,
        )
        language = result.language
        if (
            not result.text.strip()
            or not result.trusted_language
            or language not in SUPPORTED_VOICE_LANGUAGES
            or self._manual not in (None, language)
        ):
            raise _PreviewRejected()
        self._language = language
        self._source_changed.set()
        await self._publish()

    async def _transcribe(self) -> None:
        try:
            await self._provider.transcribe_stream(self._audio(), receive=self._receive)
        finally:
            self._stream_finished = True
            self._source_changed.set()

    async def _receive(self, snapshot: VoicePreviewSnapshot) -> None:
        self._latest = snapshot
        self._source_changed.set()
        await self._publish()

    async def _translate(self) -> None:
        calls = 0
        last_source = ""
        last_called_at = 0.0
        while self._active and calls < 2:
            await self._source_changed.wait()
            self._source_changed.clear()
            # 조각마다 API를 호출하지 않고 문장/구 단위로 합쳐 두 번까지만 번역한다.
            await asyncio.sleep(0.3)
            # 두 번의 예산을 초반에 모두 쓰지 않고 뒤쪽 내용에도 번역 기회를 남긴다.
            # 종료 신호에서는 간격을 기다리지 않고 마지막 원문으로 갱신한다.
            if calls and not self._stream_finished and time.perf_counter() - last_called_at < 3:
                continue
            source = _translation_prefix(self._latest.source_text, final=self._stream_finished)
            if self._language is None or not source or source == last_source:
                if self._stream_finished and self._language is not None:
                    return
                continue
            if self._language == self._target:
                return
            calls += 1
            last_source = source
            last_called_at = time.perf_counter()
            try:
                result = await self._translation_service.translate(
                    source_text=source,
                    source_language=self._language,
                    target_language=self._target,
                )
            except Exception:
                if self._stream_finished and self._latest.source_text == source:
                    return
                continue
            # 앞부분이 바뀐 인식 결과에는 이전 번역을 붙이지 않는다.
            if self._latest.source_text.startswith(source):
                self._translated_source = source
                self._translated_text = result.translated_text
                await self._publish()
            # 이전 prefix를 번역하는 동안 live final이 교정/확장됐으면 남은 한 번을 사용한다.
            if self._stream_finished and self._latest.source_text == source:
                return

    async def _publish(self) -> None:
        async with self._publish_lock:
            if not self._active or self._language is None or not self._latest.source_text:
                return
            source = self._latest.source_text
            translated = (
                source
                if self._language == self._target
                else self._translated_text
                if source.startswith(self._translated_source)
                else None
            )
            if (source, translated) == self._last_emitted:
                return
            self._last_emitted = (source, translated)
            self._revision += 1
            elapsed_ms = max(0, int((time.perf_counter() - self._started) * 1000))
            if self._first_source_ms is None:
                self._first_source_ms = elapsed_ms
            if translated and self._first_translation_ms is None:
                self._first_translation_ms = elapsed_ms
            await self._emit(
                VoicePreviewUpdate(
                    revision=self._revision,
                    source_text=source,
                    translated_text=translated,
                    detected_language=self._language,
                    started_at_offset_ms=self._initial.started_at_offset_ms,
                    ended_at_offset_ms=self._ended_at,
                )
            )


def _translation_prefix(source: str, *, final: bool) -> str:
    # 문장 끝이 아직 없으면 충분한 길이의 구를 사용하되 영어 단어 중간은 자르지 않는다.
    if final:
        return source
    boundaries = list(re.finditer(r"[。！？、；]|[.!?,;](?:\s|$)", source))
    if boundaries:
        return source[: boundaries[-1].end()].strip()
    if re.search(r"[가-힣ぁ-ゖァ-ヺ一-龯]", source):
        return source if len(source) >= 18 else ""
    boundary = source.rfind(" ")
    return source[:boundary].strip() if boundary >= 32 else ""
