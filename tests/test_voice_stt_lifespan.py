from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import main


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["faster-whisper", "openai"])
async def test_lifespan_only_warms_required_voice_runtime_and_closes_hosted_client(provider):
    # 준비: 수명주기만 검증하며 모델·외부 서비스는 실행하지 않는다.
    runtime = MagicMock(warm_up=AsyncMock(), shutdown=AsyncMock())
    service = MagicMock(warm_up=AsyncMock(), shutdown=AsyncMock())
    hosted = MagicMock(shutdown=AsyncMock())
    getters = {
        "get_voice_translation_service": MagicMock(return_value=service),
        "get_voice_speech_evidence_guard": MagicMock(return_value=service),
        "get_speech_runtime": MagicMock(return_value=runtime),
        "get_speaking_speech_runtime": MagicMock(return_value=runtime),
        "get_voice_stream_service": MagicMock(return_value=service),
        "get_voice_stt_provider": MagicMock(return_value=hosted),
        "get_speech_evidence_runtime": MagicMock(return_value=service),
        "get_ai_provider": MagicMock(return_value=service),
        "get_model_execution_provider": MagicMock(return_value=service),
        "get_speech_synthesis_provider": MagicMock(return_value=service),
        "validate_chat_auth_configuration": MagicMock(),
        "validate_novel_auth_configuration": MagicMock(),
    }

    # 실행
    with (
        patch.multiple(main, **getters),
        patch.object(main.settings, "AI_VOICE_WARM_UP", True),
        patch.object(main.settings, "AI_VOICE_STT_PROVIDER", provider),
        patch.object(main.settings, "OCR_WARM_UP", False),
    ):
        async with main.lifespan(main.app):
            assert runtime.warm_up.await_count == (1 if provider == "faster-whisper" else 0)

    # 검증: 선택한 공급자와 기존 runtime 모두 안전하게 종료한다.
    hosted.shutdown.assert_awaited_once()
    runtime.shutdown.assert_awaited_once()
