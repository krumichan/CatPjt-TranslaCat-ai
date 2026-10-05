import asyncio

import pytest

from app.ai.providers.openai.response import OpenAIProviderResponseError
from app.features.translation.service import TranslationService


class FakeProvider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def call(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ['"안녕."', "'안녕.'", "[회상]", "『문』"])
async def test_single_preserves_dialogue_quotes_and_brackets(text):
    # 준비
    provider = FakeProvider(text)
    service = TranslationService(provider)

    # 실행
    result = await service.translate_single("原文", "EPISODE")

    # 검증
    assert result == text
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, 17, True, {}, ["문장"], "", " \n "])
async def test_single_rejects_invalid_values_without_coercion(result):
    # 준비
    provider = FakeProvider(result)
    service = TranslationService(provider)

    # 실행 / 검증
    with pytest.raises(ValueError, match="TRANSLATION_RESPONSE_INVALID"):
        await service.translate_single("原文", "EPISODE")
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        "문장",
        ["하나"],
        ["하나", "둘", "셋"],
        ["하나", None],
        ["하나", 2],
        ["하나", True],
        ["하나", {}],
        ["하나", " \n"],
    ],
)
async def test_batch_rejects_invalid_values_without_recursive_repair(result):
    # 준비
    provider = FakeProvider(result)
    service = TranslationService(provider)

    # 실행 / 검증
    with pytest.raises(ValueError, match="TRANSLATION_RESPONSE_INVALID"):
        await service.translate_batch(["一", "二"], "EPISODE")
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [["一"], ["一", "二"]])
@pytest.mark.parametrize("reason", ["REFUSAL", "OUTPUT_TOKEN_LIMIT", "JSON_INVALID"])
async def test_provider_failure_is_not_retried_or_converted_to_blank(texts, reason):
    # 준비: 잘못된 JSON과 거절도 동일한 호출에서 드러나야 한다.
    error = OpenAIProviderResponseError(
        "synthetic safe failure", reason_code=reason, retryable=False
    )
    provider = FakeProvider(error=error)
    service = TranslationService(provider)

    # 실행 / 검증
    with pytest.raises(OpenAIProviderResponseError) as caught:
        await service.translate_batch(texts, "EPISODE")
    assert caught.value is error
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_singleton_chunks_obey_concurrency_and_preserve_order():
    # 준비: 첫 호출을 지연시켜도 모든 단일 조각이 같은 동시성 제한을 거친다.
    active = 0
    peak = 0
    calls = []

    class OrderedProvider:
        async def call(self, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            calls.append(kwargs)
            await asyncio.sleep(0.01)
            active -= 1
            return kwargs["data"] + " 번역"

    service = TranslationService(OrderedProvider(), batch_size=1, concurrency=1)

    # 실행
    result = await service.translate_batch(["一", "二", "三"], "EPISODE")

    # 검증
    assert result == ["一 번역", "二 번역", "三 번역"]
    assert peak == 1
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_valid_batch_preserves_strings_and_empty_input_does_not_call():
    # 준비
    expected = ['"대사"', "[회상]"]
    provider = FakeProvider(expected)
    service = TranslationService(provider)

    # 실행
    result = await service.translate_batch(["一", "二"], "EPISODE")
    empty = await service.translate_batch([], "EPISODE")

    # 검증
    assert result == expected
    assert empty == []
    assert len(provider.calls) == 1
