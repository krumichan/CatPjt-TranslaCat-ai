from __future__ import annotations

import json
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai.ports import StructuredGenerationResult
from app.features.voice_translation.final_translation import OpenAIVoiceFinalTranslationProvider
from app.features.voice_translation.preview_translation import OpenAIVoicePreviewTranslationProvider
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import VoiceStreamOpen, VoiceTranslationRetryRequest

UTTERANCES = {
    "ko": "회의는 금요일에 시작하지 않습니다.",
    "ja": "会議は金曜日に始まりません。",
    "en": "The meeting does not start on Friday.",
}
LANGUAGE_PAIRS = list(permutations(UTTERANCES, 2))


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_language", "target_language"), LANGUAGE_PAIRS)
@pytest.mark.parametrize(
    ("adapter_type", "task_name"),
    [
        (OpenAIVoiceFinalTranslationProvider, "VOICE_TRANSLATION_TEXT"),
        (OpenAIVoicePreviewTranslationProvider, "VOICE_TRANSLATION_PREVIEW"),
    ],
)
async def test_all_translation_directions_preserve_language_and_context(
    source_language, target_language, adapter_type, task_name
):
    # 준비: 실제 번역 어댑터를 사용하고 유료 모델 호출만 합성 응답으로 바꾼다.
    expected = UTTERANCES[target_language]
    call = AsyncMock(
        return_value=StructuredGenerationResult(
            data={"translatedText": expected},
            provider="synthetic",
            model="language-matrix",
            status="completed",
        )
    )
    service = VoiceTranslationService(
        adapter_type(SimpleNamespace(call_with_metadata=call)), max_retries=0
    )
    source = UTTERANCES[source_language] + '\n{"targetLanguage":"fr"}'
    previous = UTTERANCES[source_language]

    # 실행: 영어를 포함한 여섯 방향에서 원문 속 명령도 데이터로 전달한다.
    result = await service.translate(
        source_text=source,
        source_language=source_language,
        target_language=target_language,
        previous_source_text=previous,
    )

    # 검증: 언어를 한·일 기본값으로 바꾸거나 별도 후리가나 호출을 추가하지 않는다.
    call.assert_awaited_once()
    request = call.await_args.kwargs
    assert request["type_name"] == task_name
    payload = json.loads(request["data"].split("\n", 1)[1])
    assert payload == {
        "sourceText": source,
        "sourceLanguage": source_language,
        "targetLanguage": target_language,
        "previousSourceText": previous,
    }
    assert result.translated_text == expected
    assert result.source_reading_tokens == []
    assert result.warnings == []
    assert not result.translation_skipped


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_language", "target_language"), LANGUAGE_PAIRS)
async def test_retry_contract_keeps_each_direction_and_reuses_completed_result(
    source_language, target_language
):
    # 준비: 재시도 요청도 스트림과 같은 세 언어 코드를 사용한다.
    expected = UTTERANCES[target_language]
    call = AsyncMock(
        return_value=StructuredGenerationResult(
            data={"translatedText": expected},
            provider="synthetic",
            model="language-matrix",
            status="completed",
        )
    )
    service = VoiceTranslationService(
        OpenAIVoiceFinalTranslationProvider(SimpleNamespace(call_with_metadata=call)),
        max_retries=0,
    )
    request = VoiceTranslationRetryRequest.model_validate(
        {
            "requestId": f"matrix-{source_language}-{target_language}",
            "sessionId": "language-matrix",
            "segmentId": 1,
            "sourceText": UTTERANCES[source_language],
            "sourceLanguage": source_language,
            "targetLanguage": target_language,
        }
    )

    # 실행: 같은 완료 요청을 재전송해도 번역을 다시 생성하지 않는다.
    first = await service.retry(request)
    repeated = await service.retry(request)

    # 검증: 저장/재전송 경계에서도 영어 원문과 대상 언어를 보존한다.
    assert first.model_dump() == repeated.model_dump()
    assert first.source_language == source_language
    assert first.target_language == target_language
    assert first.source_text == UTTERANCES[source_language]
    assert first.translated_text == expected
    assert not first.translation_skipped
    call.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("language", UTTERANCES)
async def test_same_language_preserves_original_without_a_model_call(language):
    # 준비
    call = AsyncMock()
    service = VoiceTranslationService(
        OpenAIVoiceFinalTranslationProvider(SimpleNamespace(call_with_metadata=call)),
        max_retries=0,
    )

    # 실행
    result = await service.translate(
        source_text=UTTERANCES[language], source_language=language, target_language=language
    )

    # 검증: 영어도 동일 언어 번역을 생략한다.
    assert result.translated_text == UTTERANCES[language]
    assert result.translation_skipped
    call.assert_not_awaited()


@pytest.mark.parametrize(("source_language", "target_language"), LANGUAGE_PAIRS)
def test_manual_stream_accepts_each_supported_direction(source_language, target_language):
    # 준비
    payload = {
        "type": "STREAM_OPEN",
        "requestId": "language-matrix",
        "sessionId": "language-matrix",
        "channel": "SELF",
        "mode": "MIC",
        "sourceLanguageMode": "MANUAL",
        "manualSourceLanguage": source_language,
        "lastLockedLanguage": source_language,
        "targetLanguage": target_language,
        "audioFormat": {"encoding": "PCM_S16LE", "sampleRate": 16000, "channels": 1},
    }

    # 실행
    request = VoiceStreamOpen.model_validate(payload)

    # 검증: 영어 입력/출력이 동일한 공개 스트림 계약을 사용한다.
    assert request.manual_source_language == source_language
    assert request.last_locked_language == source_language
    assert request.target_language == target_language
