"""Speaking 원본 대조 HTTP 검사에서 모델 출력만 고정하는 합성 Provider."""
import json
from functools import lru_cache
from pathlib import Path

import httpx
from openai import APIConnectionError, APITimeoutError

from app.ai.ports import StructuredGenerationResult
from app.ai.providers.openai.response import OpenAIProviderResponseError


@lru_cache(maxsize=1)
def _cases():
    root = Path(__file__).resolve().parents[2] / "CatPjt-TranslaCat-ll"
    fixture = root / "src/test/resources/contracts/speaking-conversation-python-golden.json"
    return {row["name"]: row for row in json.loads(fixture.read_text(encoding="utf-8"))}


def execute_speaking_conversation_fixture(kwargs):
    if kwargs.get("schema_name_value") not in {
        "LANGUAGE_LEARNING_SPEAKING_CONVERSATION", "LANGUAGE_LEARNING_SPEAKING_ASSISTANCE",
    }:
        return None
    payload = json.loads(kwargs["messages"][0]["content"].rsplit("\n", 1)[-1])
    request_id = payload.get("requestId", "")
    if not request_id.startswith("speaking-http-"):
        return None
    case = _cases()[request_id.removeprefix("speaking-http-")]

    # 오류 객체도 실제 SDK 형태로 전달하되 외부 Provider 호출과 본문 로그는 만들지 않는다.
    failure = case["providerFailure"]
    if failure == "REFUSAL":
        raise OpenAIProviderResponseError("OpenAI declined the request", reason_code="REFUSAL",
                                         retryable=False)
    if failure == "TIMEOUT":
        raise TimeoutError()
    if failure == "SDK_TIMEOUT":
        raise APITimeoutError(request=httpx.Request("POST", "https://synthetic.invalid"))
    if failure == "SDK_CONNECTION":
        raise APIConnectionError(request=httpx.Request("POST", "https://synthetic.invalid"))
    markers = {"RATE_SIGNAL": "rate limit", "DEADLINE_SIGNAL": "deadline exceeded",
               "SAFETY_SIGNAL": "safety blocked", "VALUE_ERROR": "Synthetic value failure"}
    if failure in markers:
        error_type = ValueError if failure == "VALUE_ERROR" else RuntimeError
        raise error_type(markers[failure])
    if failure == "SAFETY_400":
        error = RuntimeError("safety blocked")
        error.status_code = 400
        raise error
    if failure:
        error = RuntimeError("Synthetic provider failure")
        error.status_code = int(failure)
        raise error
    return StructuredGenerationResult(case["output"], 7, 2, "synthetic", "fixed")
