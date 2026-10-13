from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from app.features.voice_translation.audio import VadPartial, normalize_voice_pcm
from app.features.voice_translation.realtime_preview import (
    OpenAILivePreviewProvider,
    VoicePreviewJob,
    VoicePreviewSnapshot,
    VoicePreviewUpdate,
    _translation_prefix,
)
from app.features.voice_translation.stream import VoiceChannelStreamContext
from app.features.voice_translation.stt import VoiceSttResult
from app.schemas.voice_translation import VoiceStreamOpen, VoiceTranscriptFinalEvent
from tests.test_voice_translation import stream_open_payload

PCM = b"\xe8\x03\x18\xfc" * 800


class FakeSocket:
    def __init__(self, events: list[dict[str, Any]], *, reject_setup: bool = False) -> None:
        self.events = events
        self.reject_setup = reject_setup
        self.sent: list[dict[str, Any]] = []
        self.committed = asyncio.Event()
        self.exited = False

    async def __aenter__(self) -> FakeSocket:
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited = True

    async def send(self, raw: str) -> None:
        event = json.loads(raw)
        self.sent.append(event)
        if event["type"] == "input_audio_buffer.commit":
            self.committed.set()

    async def recv(self) -> str:
        return json.dumps({"type": "error" if self.reject_setup else "session.updated"})

    async def __aiter__(self) -> AsyncIterator[str]:
        await self.committed.wait()
        for event in self.events:
            yield json.dumps(event)


def delta(identifier: str, text: str, *, item_id: str = "item-1") -> dict[str, Any]:
    return {
        "type": "conversation.item.input_audio_transcription.delta",
        "event_id": identifier,
        "item_id": item_id,
        "delta": text,
    }


def completed(text: str) -> dict[str, Any]:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": "done-1",
        "item_id": "item-1",
        "transcript": text,
    }


async def audio_frames() -> AsyncIterator[bytes]:
    yield PCM
    yield PCM * 2


@pytest.mark.asyncio
async def test_live_provider_resamples_once_commits_drains_and_reconciles_final(
    monkeypatch: pytest.MonkeyPatch,
):
    # 준비: 중복 delta와 최종 교정 문장이 전달되는 메모리 소켓을 사용한다.
    socket = FakeSocket(
        [
            delta("d1", "We meet "),
            delta("d1", "We meet "),
            delta("d2", "on Friday."),
            completed("We meet on Thursday."),
        ]
    )
    connections: list[dict[str, Any]] = []

    def connect(url: str, **kwargs: Any) -> FakeSocket:
        connections.append({"url": url, **kwargs})
        return socket

    monkeypatch.setattr("app.features.voice_translation.realtime_preview.connect", connect)
    received: list[VoicePreviewSnapshot] = []

    async def receive(snapshot: VoicePreviewSnapshot) -> None:
        received.append(snapshot)

    # 실행
    await OpenAILivePreviewProvider("synthetic-key").transcribe_stream(
        audio_frames(), receive=receive
    )

    # 검증: 300ms 16k PCM이 24k PCM으로 변환되고 commit 뒤 final까지 수신한다.
    assert len(connections) == 1 and "intent=transcription" in connections[0]["url"]
    setup = socket.sent[0]["session"]["audio"]["input"]
    assert setup["format"]["rate"] == 24000
    assert setup["transcription"]["model"] == "gpt-live-transcribe"
    assert setup["turn_detection"] is None
    assert "languages" not in setup["transcription"]
    chunks = [base64.b64decode(item["audio"]) for item in socket.sent if "audio" in item]
    assert sum(map(len, chunks)) == 14_400
    assert all(len(chunk) <= 9600 and len(chunk) % 2 == 0 for chunk in chunks)
    assert socket.sent[-1]["type"] == "input_audio_buffer.commit"
    assert [item.source_text for item in received] == [
        "We meet",
        "We meet on Friday.",
        "We meet on Thursday.",
    ]
    assert all(item.translated_text == "" for item in received)
    assert socket.exited


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        [],
        [delta("d1", "text")],
        [delta("d1", "text"), delta("d2", "other", item_id="other-item")],
        [{"type": "error", "error": {"message": "private"}}],
        [delta("", "text")],
        [delta("d1", "x" * 10_001)],
    ],
)
async def test_live_provider_rejects_error_truncation_wrong_item_and_unbounded_text(
    monkeypatch: pytest.MonkeyPatch, events: list[dict[str, Any]]
):
    # 준비
    socket = FakeSocket(events)
    monkeypatch.setattr(
        "app.features.voice_translation.realtime_preview.connect", lambda *a, **k: socket
    )

    async def receive(_: VoicePreviewSnapshot) -> None:
        pass

    # 실행 / 검증: final 없는 transport 종료를 성공으로 취급하지 않는다.
    with pytest.raises(ExceptionGroup):
        await OpenAILivePreviewProvider("synthetic-key").transcribe_stream(
            audio_frames(),
            receive=receive,
        )
    assert socket.exited


@pytest.mark.asyncio
async def test_rejected_setup_never_uploads_audio(monkeypatch: pytest.MonkeyPatch):
    # 준비
    socket = FakeSocket([], reject_setup=True)
    monkeypatch.setattr(
        "app.features.voice_translation.realtime_preview.connect", lambda *a, **k: socket
    )

    async def receive(_: VoicePreviewSnapshot) -> None:
        pass

    # 실행 / 검증
    with pytest.raises(RuntimeError, match="configuration rejected"):
        await OpenAILivePreviewProvider("synthetic-key").transcribe_stream(
            audio_frames(),
            receive=receive,
        )
    assert [item["type"] for item in socket.sent] == ["session.update"]


class FakeLiveProvider:
    def __init__(self, source: str = "We will meet tomorrow.") -> None:
        self.source = source
        self.started = asyncio.Event()
        self.receive: Any = None
        self.pcm: list[bytes] = []
        self.calls = 0

    async def transcribe_stream(self, audio: AsyncIterator[bytes], *, receive: Any) -> None:
        self.calls += 1
        self.receive = receive
        await receive(VoicePreviewSnapshot(self.source, ""))
        self.started.set()
        async for chunk in audio:
            self.pcm.append(chunk)

    async def publish(self, source: str) -> None:
        await self.receive(VoicePreviewSnapshot(source, ""))


class FakeLanguage:
    ready = True
    model_version = "fake"

    def __init__(self, language: str | None = "en", *, trusted: bool = True) -> None:
        self.language = language
        self.trusted = trusted
        self.release = asyncio.Event()
        self.release.set()
        self.calls: list[dict[str, Any]] = []

    async def transcribe_pcm(self, _: bytes, **kwargs: Any) -> VoiceSttResult:
        self.calls.append(kwargs)
        await self.release.wait()
        return VoiceSttResult(
            "Synthetic source",
            self.language,
            None,
            None,
            "fake",
            "fake",
            trusted_language=self.trusted,
        )


class FakeGuard:
    ready = True

    def __init__(self, speech: bool = True) -> None:
        self.speech = speech

    async def has_speech(self, _: bytes) -> bool:
        return self.speech


class FakeTranslation:
    ready = True

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.release = asyncio.Event()
        self.release.set()

    async def translate(self, *, source_text: str, **kwargs: Any) -> Any:
        self.calls.append(source_text)
        self.started.put_nowait(source_text)
        await self.release.wait()
        return SimpleNamespace(translated_text="번역: " + source_text)


def make_job(
    *,
    provider: FakeLiveProvider | None = None,
    language: FakeLanguage | None = None,
    translation: FakeTranslation | None = None,
    speech: bool = True,
    slots: asyncio.Semaphore | None = None,
    manual: str | None = None,
    target: str = "ko",
) -> tuple[
    VoicePreviewJob, FakeLiveProvider, FakeLanguage, FakeTranslation, list[VoicePreviewUpdate]
]:
    provider = provider or FakeLiveProvider()
    language = language or FakeLanguage()
    translation = translation or FakeTranslation()
    updates: list[VoicePreviewUpdate] = []

    async def emit(update: VoicePreviewUpdate) -> None:
        updates.append(update)

    job = VoicePreviewJob(
        VadPartial(PCM * 18, 0, 1800),
        provider=provider,
        language_provider=language,
        translation_service=translation,
        speech_guard=FakeGuard(speech),
        target_language=target,
        manual_language=manual,
        emit=emit,
        slots=slots or asyncio.Semaphore(4),
    )
    return job, provider, language, translation, updates


async def cleanup(job: VoicePreviewJob) -> None:
    job.cancel()
    await asyncio.gather(job.task, return_exceptions=True)


@pytest.mark.asyncio
async def test_language_gate_holds_source_and_translation_until_independent_detection():
    # 준비: 실시간 source가 먼저 나와도 독립 언어 감지를 기다린다.
    language = FakeLanguage()
    language.release.clear()
    job, provider, _, translation, updates = make_job(language=language)
    try:
        await asyncio.wait_for(provider.started.wait(), 1)
        assert not updates and not translation.calls

        # 실행
        language.release.set()
        await asyncio.wait_for(translation.started.get(), 1)

        # 검증
        assert updates and updates[0].detected_language == "en"
        assert language.calls == [{"language": None, "is_final": False, "initial_prompt": None}]
    finally:
        await cleanup(job)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detected,trusted,manual",
    [(None, False, None), ("fr", False, None), ("en", True, "ja")],
)
async def test_rejected_language_never_publishes_or_translates(
    detected: str | None,
    trusted: bool,
    manual: str | None,
):
    # 준비
    job, _, _, translation, updates = make_job(
        language=FakeLanguage(detected, trusted=trusted),
        manual=manual,
    )

    # 실행
    await asyncio.wait_for(job.task, 1)

    # 검증
    assert not updates and not translation.calls
    assert job._initial.pcm_bytes == b""


@pytest.mark.asyncio
async def test_no_speech_skips_both_external_providers():
    # 준비
    job, provider, language, translation, updates = make_job(speech=False)

    # 실행
    await asyncio.wait_for(job.task, 1)

    # 검증
    assert not provider.calls and not language.calls and not translation.calls and not updates


@pytest.mark.asyncio
async def test_full_global_slots_skip_and_release_audio_snapshot():
    # 준비: 전역 네 자리를 이미 사용 중인 상황이다.
    slots = asyncio.Semaphore(4)
    for _ in range(4):
        await slots.acquire()
    job, provider, language, _, updates = make_job(slots=slots)

    # 실행
    await asyncio.wait_for(job.task, 1)

    # 검증: 자리 대기/외부 호출 없이 원본 PCM까지 버린다.
    assert not provider.calls and not language.calls and not updates
    assert job._initial.pcm_bytes == b""
    assert job._queue.empty() and not job._active


@pytest.mark.asyncio
async def test_cumulative_audio_snapshots_do_not_duplicate_pcm_and_final_drains():
    # 준비
    job, provider, _, _, updates = make_job(target="en")
    await asyncio.wait_for(provider.started.wait(), 1)
    await asyncio.sleep(0)

    # 실행: 대기 snapshot이 교체돼도 처음부터 다시 보내지 않는다.
    job.offer(VadPartial(PCM * 20, 0, 2000))
    job.offer(VadPartial(PCM * 25, 0, 2500), final=True)
    await asyncio.wait_for(job.task, 1)

    # 검증
    assert b"".join(provider.pcm) == normalize_voice_pcm(PCM * 25)
    assert updates and updates[-1].translated_text == updates[-1].source_text


@pytest.mark.asyncio
async def test_translation_is_coalesced_and_capped_at_two_calls_per_utterance(
    monkeypatch: pytest.MonkeyPatch,
):
    # 준비: API 간격은 모듈 시계로 진행시켜 실제 3초를 기다리지 않는다.
    clock = [100.0]
    monkeypatch.setattr(
        "app.features.voice_translation.realtime_preview.time",
        SimpleNamespace(perf_counter=lambda: clock[0]),
    )
    job, provider, _, translation, updates = make_job()
    try:
        await asyncio.wait_for(translation.started.get(), 1)

        # 실행 / 검증: 두 번째 문장이 생겨도 3초 간격 전에는 마지막 예산을 소비하지 않는다.
        clock[0] = 101.0
        await provider.publish("We will meet tomorrow. Bring the report.")
        await asyncio.sleep(0.35)
        assert len(translation.calls) == 1

        # 실행: 3초가 지난 최신 스냅샷에 두 번째 기회를 사용한다.
        clock[0] = 103.1
        await provider.publish("We will meet tomorrow. Bring the report.")
        await asyncio.wait_for(translation.started.get(), 1)
        await provider.publish("We will meet tomorrow. Bring the report. We start at ten.")
        job.offer(VadPartial(PCM * 30, 0, 3000), final=True)
        await asyncio.wait_for(job.task, 1)

        # 검증
        assert len(translation.calls) == 2
        assert translation.calls[1] == "We will meet tomorrow. Bring the report."
        assert [item.revision for item in updates] == list(range(1, len(updates) + 1))
        assert updates[-1].source_text.endswith("at ten.")
    finally:
        await cleanup(job)


@pytest.mark.asyncio
async def test_source_correction_discards_inflight_old_translation():
    # 준비: 번역 중 인식된 문장의 날짜가 교정된다.
    translation = FakeTranslation()
    translation.release.clear()
    job, provider, _, _, updates = make_job(translation=translation)
    try:
        await asyncio.wait_for(translation.started.get(), 1)

        # 실행
        await provider.publish("We will meet next week.")
        translation.release.set()
        job.offer(VadPartial(PCM * 22, 0, 2200), final=True)
        await asyncio.wait_for(job.task, 1)

        # 검증: 최신 원문에 이전 날짜 번역을 붙이지 않는다.
        corrected = [item for item in updates if item.source_text == "We will meet next week."]
        assert corrected
        assert all(
            item.translated_text is None or "tomorrow" not in item.translated_text
            for item in corrected
        )
        assert translation.calls == ["We will meet tomorrow.", "We will meet next week."]
        assert corrected[-1].translated_text == "번역: We will meet next week."
    finally:
        await cleanup(job)


@pytest.mark.asyncio
async def test_cancelled_job_cannot_publish_late_provider_callback():
    # 준비
    job, provider, _, _, updates = make_job(target="en")
    await asyncio.wait_for(provider.started.wait(), 1)

    # 실행: 최종 결과가 먼저 도착해 job 소유권을 취소한 뒤 늦은 callback을 전달한다.
    await cleanup(job)
    before = len(updates)
    await provider.publish("Late text after final.")

    # 검증
    assert len(updates) == before and job._initial.pcm_bytes == b""


@pytest.mark.asyncio
async def test_context_final_barrier_next_utterance_and_shutdown_keep_preview_ownership():
    # 준비: 실제 stream context에 fake preview/언어/번역 서비스만 연결한다.
    provider = FakeLiveProvider()
    translation = FakeTranslation()
    stream_open = VoiceStreamOpen.model_validate(stream_open_payload())
    context = VoiceChannelStreamContext(
        stream_open,
        stt_provider=FakeLanguage(),
        translation_service=translation,
        speech_evidence_guard=FakeGuard(),
        preview_provider=provider,
        preview_language_provider=FakeLanguage(),
        preview_translation_service=translation,
    )
    await context._start_utterance(0)
    context._schedule_partial(VadPartial(PCM * 18, 0, 1800))
    first_job = next(iter(context._preview_jobs.values()))
    try:
        await asyncio.wait_for(translation.started.get(), 1)
        first_events = []
        while not context._event_queue.empty():
            first_events.append(context._event_queue.get_nowait())
        assert any(
            event.type == "TRANSCRIPT_PARTIAL" and event.translated_text for event in first_events
        )
        old_receive = provider.receive
        first_key = next(iter(context._preview_jobs))

        # 실행: 최종 원문이 임시 작업을 제거하고 이미 수신 대기 중이던 callback을 차단한다.
        await context._emit(
            VoiceTranscriptFinalEvent(
                event_id="canonical-1",
                session_id=stream_open.session_id,
                channel=stream_open.channel,
                utterance_key=first_key,
                utterance_sequence=1,
                source_text="Canonical source.",
                started_at_offset_ms=0,
                ended_at_offset_ms=1800,
                speech_duration_ms=1800,
            )
        )
        await asyncio.gather(first_job.task, return_exceptions=True)
        before = context._event_queue.qsize()
        await old_receive(VoicePreviewSnapshot("Late source from first utterance.", ""))

        # 검증: 확정 카드와 DB에 쓰일 이벤트를 늦은 preview가 되돌리지 않는다.
        assert not context._preview_jobs
        assert context._event_queue.qsize() == before
        assert context._event_queue.get_nowait().type == "TRANSCRIPT_FINAL"

        # 실행: 다음 발화는 독립 revision/소유권을 갖고 shutdown 후 callback도 무시한다.
        await context._start_utterance(1800)
        context._schedule_partial(VadPartial(PCM * 18, 1800, 3600))
        second_job = next(iter(context._preview_jobs.values()))
        await asyncio.wait_for(translation.started.get(), 1)
        second_events = []
        while not context._event_queue.empty():
            second_events.append(context._event_queue.get_nowait())
        await context._cancel_background_tasks()
        await asyncio.gather(second_job.task, return_exceptions=True)
        await provider.publish("Late source after shutdown.")

        # 검증
        partials = [event for event in second_events if event.type == "TRANSCRIPT_PARTIAL"]
        assert partials and partials[0].revision == 1
        assert all(event.utterance_sequence == 2 for event in partials)
        assert not context._preview_jobs and context._event_queue.empty()
    finally:
        tasks = [job.task for job in context._preview_jobs.values()]
        await context._cancel_background_tasks()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_context_caps_preview_jobs_at_two_per_channel():
    # 준비
    context = VoiceChannelStreamContext(
        VoiceStreamOpen.model_validate(stream_open_payload()),
        stt_provider=FakeLanguage(),
        translation_service=FakeTranslation(),
        speech_evidence_guard=FakeGuard(),
        preview_provider=FakeLiveProvider(),
        preview_language_provider=FakeLanguage(),
        preview_translation_service=FakeTranslation(),
    )

    # 실행: 세 발화의 최종 처리가 겹쳐도 채널당 임시 작업은 두 개만 생성한다.
    for index in range(3):
        await context._start_utterance(index * 2000)
        context._schedule_partial(VadPartial(PCM * 18, index * 2000, index * 2000 + 1800))

    # 검증 / 정리
    assert len(context._preview_jobs) == 2
    tasks = [job.task for job in context._preview_jobs.values()]
    await context._cancel_background_tasks()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("明日は晴れです。午後は雨", "明日は晴れです。"),
        ("회의를 시작합니다. 다음 안건", "회의를 시작합니다."),
        ("Meeting begins at ten. Bring your", "Meeting begins at ten."),
        ("Short unfinished", ""),
    ],
)
def test_translation_prefix_uses_completed_clause_without_cutting_english_word(
    source: str,
    expected: str,
):
    # 준비 / 실행 / 검증
    assert _translation_prefix(source, final=False) == expected
    assert _translation_prefix(source, final=True) == source
