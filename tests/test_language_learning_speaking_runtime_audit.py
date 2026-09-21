from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from app.ai.ports import SpeechSynthesisResult
from app.ai.providers.openai.schema import build_openai_text_config
from app.features.language_learning.speaking.audio_store import TemporaryTtsAudioStore
from app.features.language_learning.speaking.audio_processor import SpeakingAudioProcessor
from app.features.language_learning.speaking.conversation_service import SpeakingConversationService
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.evaluation_service import SpeakingEvaluationService
from app.features.language_learning.speaking.stt_service import SpeakingSttService
from app.features.language_learning.speaking.tts_service import SpeakingTtsService
from app.features.language_learning.speaking.turn_service import SpeakingTurnService
from app.schemas.language_learning_speaking import (
    SpeakingErrorCode, SpeakingEvaluationPayload, SpeakingStage, TtsRequest,
)
from tests.test_language_learning_speaking import (
    FakeConversationServiceForTurn,
    FakeSpeechProvider,
    FakeSttServiceForTurn,
    FakeStructuredProvider,
    FakeTtsServiceForTurn,
    conversation_payload,
    conversation_request,
    evaluation_payload,
    evaluation_request,
    make_wav,
)


def _turn_service(
    stt: FakeSttServiceForTurn, conversation: FakeConversationServiceForTurn
) -> SpeakingTurnService:
    # These existing test doubles implement the methods used by TurnService.
    return SpeakingTurnService(
        audio_processor=SpeakingAudioProcessor(),
        stt_service=cast(SpeakingSttService, stt),
        conversation_service=cast(SpeakingConversationService, conversation),
        tts_service=cast(SpeakingTtsService, FakeTtsServiceForTurn()),
    )


@pytest.mark.asyncio
async def test_tts_cache_preserves_playback_duration_without_another_provider_call(
    tmp_path: Path,
) -> None:
    provider = FakeSpeechProvider()
    with patch("tempfile.gettempdir", return_value=str(tmp_path)):
        store = TemporaryTtsAudioStore(ttl_seconds=60)
    service = SpeakingTtsService(provider, store, timeout_seconds=1, automatic_retries=0)
    request = TtsRequest(
        request_id="speaking-audit-tts",
        idempotency_key="speaking-audit-tts",
        session_id="speaking-audit",
        text="こんにちは",
        learning_language="ja",
    )

    first = await service.synthesize(request)
    replay = await service.synthesize(request)

    assert first.audio.duration_seconds == 1.5
    assert replay.audio == first.audio
    assert provider.calls == 1
    # A cached playback does not create billable synthesis usage.
    assert replay.usage.tts is not None
    assert replay.usage.tts.tts_audio_seconds == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp_field", ["startMs", "endMs"])
async def test_evaluation_rejects_out_of_range_single_timestamp(
    timestamp_field: str,
) -> None:
    payload = evaluation_payload()
    evidence = payload["metrics"][0]["evidence"][0]
    evidence["startMs"] = None
    evidence["endMs"] = None
    evidence[timestamp_field] = 13_000
    provider = FakeStructuredProvider(results=[payload])
    service = SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=0)

    with pytest.raises(SpeakingStageException) as caught:
        await service.evaluate(evaluation_request())

    assert caught.value.code.value == "INVALID_RESPONSE_SCHEMA"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp_field", ["startMs", "endMs"])
async def test_evaluation_keeps_valid_single_timestamp_supported(
    timestamp_field: str,
) -> None:
    payload = evaluation_payload()
    evidence = payload["metrics"][0]["evidence"][0]
    evidence["startMs"] = None
    evidence["endMs"] = None
    evidence[timestamp_field] = 1_000
    provider = FakeStructuredProvider(results=[payload])
    service = SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=0)

    result = await service.evaluate(evaluation_request())

    assert result.status.value == "EVALUATED"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_conversation_cancellation_does_not_retry_cache_or_cancel_other_session() -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()

    class CancellableProvider(FakeStructuredProvider):
        async def call_with_metadata(self, type_name, data, schema=None):
            if '"sessionId":"cancelled-session"' in data:
                self.calls.append((type_name, data, schema))
                started.set()
                await blocked.wait()
            return await super().call_with_metadata(type_name, data, schema)

    provider = CancellableProvider(results=[conversation_payload()])
    service = SpeakingConversationService(provider, timeout_seconds=1, automatic_retries=2)
    cancelled_request = conversation_request(
        sessionId="cancelled-session", idempotencyKey="cancelled-turn"
    )
    cancelled_task = asyncio.create_task(service.generate(cancelled_request))
    await asyncio.wait_for(started.wait(), timeout=1)
    peer_task = asyncio.create_task(
        service.generate(conversation_request(sessionId="peer-session", idempotencyKey="peer-turn"))
    )
    cancelled_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_task
    peer = await asyncio.wait_for(peer_task, timeout=1)

    assert peer.session_id == "peer-session"
    assert len(provider.calls) == 2
    assert service.idempotency_store.get("cancelled-turn") is None
    assert service.idempotency_store.get("peer-turn") is not None


@pytest.mark.asyncio
async def test_turn_retry_reuses_successful_transcript_and_rerecord_retranscribes() -> None:
    stt = FakeSttServiceForTurn()
    conversation = FakeConversationServiceForTurn(
        error=SpeakingStageException(
            code=SpeakingErrorCode.CONVERSATION_GENERATION_FAILED,
            stage=SpeakingStage.CONVERSATION,
            message="offline failure",
            retryable=True,
        )
    )
    service = _turn_service(stt, conversation)
    first = await service.process_turn(
        context=conversation_request(idempotencyKey="recording-1:process:0"),
        audio_bytes=make_wav(),
        file_name="turn.wav",
        content_type="audio/wav",
    )
    assert first.status == "PARTIAL_FAILURE"
    assert first.failed_stage == "CONVERSATION"
    assert first.transcript is not None
    # Preserve low-confidence metadata; reuse must never upgrade STT confidence.
    persisted = first.transcript.model_copy(
        update={"confidence": 0.4, "is_low_confidence": True}, deep=True
    )
    conversation.error = None
    second = await service.process_turn(
        context=conversation_request(
            idempotencyKey="recording-1:process:1",
            manualRetryAttempt=1,
            transcript=persisted,
        ),
        audio_bytes=make_wav(),
        file_name="turn.wav",
        content_type="audio/wav",
    )
    assert second.status == "READY"
    assert second.transcript == persisted
    assert conversation.last_request is not None
    assert conversation.last_request.transcript == persisted
    assert second.usage.stt is None
    assert stt.calls == 1
    assert conversation.calls == 2

    rerecord = await service.process_turn(
        context=conversation_request(
            idempotencyKey="recording-2:process:0", transcript=None
        ),
        audio_bytes=make_wav(),
        file_name="turn.wav",
        content_type="audio/wav",
    )
    assert rerecord.status == "READY"
    assert stt.calls == 2
    assert rerecord.usage.stt is not None


@pytest.mark.asyncio
async def test_reused_transcript_does_not_skip_audio_validation() -> None:
    stt = FakeSttServiceForTurn()
    prior = await stt.transcribe(request_id="prior", session_id="s", turn_index=1)
    conversation = FakeConversationServiceForTurn()
    service = _turn_service(stt, conversation)

    response = await service.process_turn(
        context=conversation_request(transcript=prior.transcript, manualRetryAttempt=1),
        audio_bytes=make_wav(silence=True),
        file_name="turn.wav",
        content_type="audio/wav",
    )

    assert response.status == "PARTIAL_FAILURE"
    assert response.error is not None
    assert response.error.code.value == "SILENCE_DETECTED"
    assert stt.calls == 1
    assert conversation.calls == 0


@pytest.mark.asyncio
async def test_blank_existing_transcript_still_uses_stt() -> None:
    stt = FakeSttServiceForTurn()
    prior = await stt.transcribe(request_id="prior", session_id="s", turn_index=1)
    service = _turn_service(stt, FakeConversationServiceForTurn())

    response = await service.process_turn(
        context=conversation_request(
            transcript=prior.transcript.model_copy(update={"text": "   "}),
            manualRetryAttempt=1,
        ),
        audio_bytes=make_wav(),
        file_name="turn.wav",
        content_type="audio/wav",
    )

    assert response.status == "READY"
    assert response.transcript is not None
    assert response.transcript.text.strip()
    assert stt.calls == 2


@pytest.mark.asyncio
async def test_reused_transcript_does_not_bypass_configured_manual_retry_limit() -> None:
    from app.core.config import settings

    stt = FakeSttServiceForTurn()
    prior = await stt.transcribe(request_id="prior", session_id="s", turn_index=1)
    conversation = FakeConversationServiceForTurn()
    service = _turn_service(stt, conversation)
    with patch.object(settings, "AI_SPEAKING_MANUAL_RETRY_LIMIT", 0):
        response = await service.process_turn(
            context=conversation_request(transcript=prior.transcript, manualRetryAttempt=1),
            audio_bytes=make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
        )

    assert response.status == "PARTIAL_FAILURE"
    assert response.error is not None
    assert response.error.code.value == "MANUAL_RETRY_LIMIT_EXCEEDED"
    assert stt.calls == 1
    assert conversation.calls == 0


def test_tts_store_duplicate_preserves_first_bytes_with_their_duration(tmp_path: Path) -> None:
    with patch("tempfile.gettempdir", return_value=str(tmp_path)):
        store = TemporaryTtsAudioStore(ttl_seconds=60)

    first = store.put(b"first-audio", cache_key="same", duration_seconds=1.5)
    duplicate = store.put(b"second-audio", cache_key="same", duration_seconds=9.0)

    assert duplicate == first
    assert duplicate.path.read_bytes() == b"first-audio"
    assert duplicate.duration_seconds == 1.5


def test_tts_store_replaces_orphan_file_without_live_metadata(tmp_path: Path) -> None:
    with patch("tempfile.gettempdir", return_value=str(tmp_path)):
        previous_store = TemporaryTtsAudioStore(ttl_seconds=60)
        previous = previous_store.put(b"orphan-audio", cache_key="same", duration_seconds=9.0)
        restarted_store = TemporaryTtsAudioStore(ttl_seconds=60)

    current = restarted_store.put(b"current-audio", cache_key="same", duration_seconds=1.5)

    assert current.path == previous.path
    assert current.path.read_bytes() == b"current-audio"
    assert current.duration_seconds == 1.5
    assert not list(current.path.parent.glob("*.tmp"))


@pytest.mark.asyncio
async def test_concurrent_tts_responses_describe_retained_audio(tmp_path: Path) -> None:
    started = [asyncio.Event(), asyncio.Event()]
    results = [asyncio.Event(), asyncio.Event()]
    audio = [make_wav(1.5), make_wav(9.0)]

    class OrderedSpeechProvider:
        calls = 0

        async def synthesize_speech(self, **kwargs):
            index = self.calls
            self.calls += 1
            started[index].set()
            await results[index].wait()
            return SpeechSynthesisResult(
                audio_bytes=audio[index],
                content_type="audio/wav",
                provider="fake",
                model="fake",
                duration_seconds=[1.5, 9.0][index],
            )

    provider = OrderedSpeechProvider()
    with patch("tempfile.gettempdir", return_value=str(tmp_path)):
        store = TemporaryTtsAudioStore(ttl_seconds=60)
    service = SpeakingTtsService(provider, store, timeout_seconds=1, automatic_retries=0)
    request = TtsRequest(
        request_id="tts-concurrent",
        idempotency_key="tts-concurrent",
        session_id="same",
        text="こんにちは",
        learning_language="ja",
    )
    first_task = asyncio.create_task(service.synthesize(request))
    await asyncio.wait_for(started[0].wait(), timeout=1)
    second_task = asyncio.create_task(service.synthesize(request))
    await asyncio.wait_for(started[1].wait(), timeout=1)
    results[0].set()
    first = await first_task
    results[1].set()
    second = await second_task

    assert second.audio == first.audio
    assert second.audio.duration_seconds == 1.5
    retained = store.get(first.audio.audio_reference)
    assert retained is not None
    assert retained.path.read_bytes() == audio[0]


def test_speaking_profile_direction_wire_schema_expresses_existing_closed_values() -> None:
    public_schema = SpeakingEvaluationPayload.model_json_schema()
    config = build_openai_text_config(
        type_name="LANGUAGE_LEARNING_SPEAKING_EVALUATION",
        schema=public_schema, verbosity="low",
    )
    wire = config["format"]
    direction = wire["schema"]["$defs"]["SpeakingProfileSignal"]["properties"]["direction"]
    assert direction["enum"] == ["STRENGTH", "WEAKNESS", "IMPROVING"]
    assert "pattern" not in direction
    assert wire["strict"] is False
    # The adapter neither mutates public DTO schema nor accepts new directions.
    assert public_schema["$defs"]["SpeakingProfileSignal"]["properties"]["direction"]["pattern"] == "^(STRENGTH|WEAKNESS|IMPROVING)$"


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["POSITIVE", "NEUTRAL"])
async def test_speaking_profile_unapproved_direction_still_rejects_without_relabeling(direction: str) -> None:
    payload = evaluation_payload()
    payload["profileSignals"][0]["direction"] = direction
    provider = FakeStructuredProvider(results=[payload])
    service = SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=0)
    with pytest.raises(SpeakingStageException) as caught:
        await service.evaluate(evaluation_request())
    assert caught.value.code.value == "INVALID_RESPONSE_SCHEMA"
    assert len(provider.calls) == 1
    assert payload["profileSignals"][0]["direction"] == direction
