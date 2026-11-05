import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai.providers.openai.client import OpenAIService
from app.features.spending_analysis.prompts import SPENDING_ANALYSIS_PROMPT, SPENDING_REVIEW_PROMPT
from app.features.spending_analysis.service import SpendingAnalysisService
from app.schemas.spending_analysis import SpendingAnalysisRequest
from tests.test_spending_analysis import request_data
from tests.test_spending_coaching import coaching


@pytest.mark.asyncio
@pytest.mark.parametrize("half_year", [False, True])
async def test_real_provider_adapter_resolves_generation_and_review_before_sdk_requests(half_year):
    # 준비: 업무 서비스와 실제 OpenAI adapter를 연결하고 SDK 전송 경계만 대체한다.
    data = request_data()
    if half_year:
        data.update(periodType="HALF_YEARLY", month=None, half=2, locale="ja",
                    periodStart="2026-07-01", periodEndExclusive="2027-01-01", previousPeriod=None)
    draft = coaching()
    sdk_response = SimpleNamespace(
        status="completed", output=[], output_text=json.dumps(draft, ensure_ascii=False),
        usage=SimpleNamespace(input_tokens=10, output_tokens=10), model="gpt-6.1-sol", incomplete_details=None,
    )
    create = AsyncMock(side_effect=[sdk_response, sdk_response])
    sdk = SimpleNamespace(responses=SimpleNamespace(create=create))
    sdk.with_options = lambda **_options: sdk
    provider = OpenAIService()
    provider._client = sdk

    # 실행: 각 stage의 registry/model/schema/decoder를 실제로 통과한다. 외부 호출은 없다.
    result = await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(data))

    # 검증: 검토 작업 등록이 빠지면 두 번째 SDK 요청 전에 HTTP400으로 이 검사가 실패한다.
    assert create.await_count == 2
    first, second = [call.kwargs for call in create.call_args_list]
    assert first["instructions"] == SPENDING_ANALYSIS_PROMPT
    assert second["instructions"] == SPENDING_REVIEW_PROMPT
    assert json.loads(first["input"])["periodType"] == data["periodType"]
    for call in (first, second):
        assert call["model"] == "gpt-6.1-sol"
        assert call["store"] is False
        assert call["text"]["format"]["strict"] is True
    assert result.policyVersion == "purchase-context-v4"
    assert result.findings
