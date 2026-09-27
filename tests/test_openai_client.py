import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.ai.providers.openai.client import OpenAIService


def test_general_text_generation_records_latency_and_usage_without_live_provider():
    # 준비: 실제 API 대신 완료 응답과 사용량을 반환하는 클라이언트를 주입한다.
    create = AsyncMock(
        return_value=SimpleNamespace(
            status="completed",
            output_text="Synthetic translation",
            output=[],
            model="synthetic-model",
            usage=SimpleNamespace(input_tokens=8, output_tokens=3),
        )
    )
    provider = OpenAIService()
    provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))

    # 실행: 일반 task 경로의 응답 생성과 지연 시간 측정을 함께 거친다.
    result = asyncio.run(
        provider.call_with_metadata(
            type_name="VOICE",
            data="Synthetic source",
        )
    )

    # 검증: time 참조 오류 없이 결과와 진단 정보를 반환하고 한 번만 호출한다.
    assert result.data == "Synthetic translation"
    assert result.input_tokens == 8
    assert result.output_tokens == 3
    assert result.model == "synthetic-model"
    assert result.status == "completed"
    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0
    create.assert_awaited_once()
