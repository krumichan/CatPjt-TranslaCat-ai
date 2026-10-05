from types import SimpleNamespace

import pytest

from app.ai.providers.openai.profile_execution import decode_profile_response
from app.ai.providers.openai.response import OpenAIProviderResponseError


@pytest.mark.parametrize("raw", ['{"s0":"a","s0":"b"}', '{"items":{"s0":"a","s0":"b"}}'])
def test_profile_rejects_duplicate_raw_json_keys(raw):
    # 준비
    response = SimpleNamespace(status="completed", output=[], output_text=raw)

    # 실행 / 검증
    with pytest.raises(OpenAIProviderResponseError) as caught:
        decode_profile_response(response, structured=True)
    assert caught.value.reason_code == "DUPLICATE_JSON_KEY"
    assert caught.value.retryable is False


def test_profile_preserves_valid_keyed_strings_and_rejects_invalid_json():
    # 준비
    valid = SimpleNamespace(status="completed", output=[], output_text='{"s0":"\\"가자\\""}')
    invalid = SimpleNamespace(status="completed", output=[], output_text='{"s0":broken}')

    # 실행 / 검증
    assert decode_profile_response(valid, structured=True) == {"s0": '"가자"'}
    with pytest.raises(OpenAIProviderResponseError) as caught:
        decode_profile_response(invalid, structured=True)
    assert caught.value.reason_code == "JSON_INVALID"


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_profile_rejects_non_json_numeric_constants(constant):
    # 준비
    response = SimpleNamespace(
        status="completed", output=[], output_text='{"value":' + constant + "}"
    )

    # 실행 / 검증
    with pytest.raises(OpenAIProviderResponseError) as caught:
        decode_profile_response(response, structured=True)
    assert caught.value.reason_code == "JSON_INVALID"
