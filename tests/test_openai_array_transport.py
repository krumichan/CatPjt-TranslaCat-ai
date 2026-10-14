import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai.providers.openai.client import OpenAIService
from app.ai.providers.openai.response import OpenAIProviderResponseError


@pytest.mark.asyncio
async def test_legacy_translation_array_has_object_wire_schema_and_list_domain_result():
    # 준비: OpenAI는 최상위 배열 schema를 거절하므로 실제 요청과 decode 양쪽을 검사한다.
    response = SimpleNamespace(
        status="completed", output=[], output_text=json.dumps({"items": ["제목", "소개"]})
    )
    create = AsyncMock(return_value=response)
    service = OpenAIService()
    service._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    schema = {"type": "ARRAY", "items": {"type": "STRING"}}

    # 실행
    result = await service.call(type_name="RANK", data='["題名","紹介"]', schema=schema)

    # 검증: 내부 provider 형식만 바꾸며 기존 호출자의 배열 계약을 보존한다.
    sent = create.call_args.kwargs["text"]["format"]["schema"]
    assert sent == {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
        "required": ["items"],
        "additionalProperties": False,
    }
    assert result == ["제목", "소개"]
    assert schema == {"type": "ARRAY", "items": {"type": "STRING"}}
    create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", [{"items": "bad"}, {"wrong": []}, {"items": [], "extra": "bad"}, []]
)
async def test_invalid_array_envelope_is_not_repaired_or_retried(payload):
    # 준비
    response = SimpleNamespace(status="completed", output=[], output_text=json.dumps(payload))
    create = AsyncMock(return_value=response)
    service = OpenAIService()
    service._client = SimpleNamespace(responses=SimpleNamespace(create=create))

    # 실행 / 검증: 임의 보정이나 유료 재시도 없이 실패를 유지한다.
    with pytest.raises(OpenAIProviderResponseError, match="array envelope"):
        await service.call(
            type_name="NOVEL",
            data='["第一話","第二話"]',
            schema={"type": "ARRAY", "items": {"type": "STRING"}},
        )
    create.assert_awaited_once()
