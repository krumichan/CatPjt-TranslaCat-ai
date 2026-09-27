"""Speaking 평가·코칭의 원본 계약 HTTP 검사용 합성 모델 출력이다."""

import json
import os
from functools import lru_cache
from pathlib import Path

import httpx
from openai import APIConnectionError, APITimeoutError

from app.ai.ports import StructuredGenerationResult
from app.ai.providers.openai.response import OpenAIProviderResponseError


@lru_cache(maxsize=1)
def _cases():
    root = Path(__file__).resolve().parents[2] / "CatPjt-TranslaCat-ll"
    fixture = root / "src/test/resources/contracts/speaking-evaluation-python-golden.json"
    return {row["name"]: row for row in json.loads(fixture.read_text(encoding="utf-8"))}


def execute_speaking_evaluation_fixture(kwargs):
    if kwargs.get("schema_name_value") not in {
        "LANGUAGE_LEARNING_SPEAKING_EVALUATION",
        "LANGUAGE_LEARNING_SPEAKING_SESSION_COACHING",
    }:
        return None
    if os.environ.get("TRANSLACAT_TEST_MODEL_EXECUTION") != "1":
        raise RuntimeError("Synthetic provider requires explicit test marker")
    payload = json.loads(kwargs["messages"][0]["content"].rsplit("\n", 1)[-1])
    session_id = payload.get("sessionId", "")
    if not session_id.startswith("speaking-evaluation-http-"):
        return None
    case = _cases()[session_id.removeprefix("speaking-evaluation-http-")]

    # 원본 예외 종류만 합성하며 외부 모델 호출이나 업무 상태 전이는 하지 않는다.
    failure = case["providerFailure"]
    if failure == "REFUSAL":
        raise OpenAIProviderResponseError(
            "OpenAI declined the request", reason_code="REFUSAL", retryable=False
        )
    if failure == "TIMEOUT":
        raise TimeoutError()
    if failure == "SDK_TIMEOUT":
        raise APITimeoutError(request=httpx.Request("POST", "https://synthetic.invalid"))
    if failure == "SDK_CONNECTION":
        raise APIConnectionError(request=httpx.Request("POST", "https://synthetic.invalid"))
    markers = {
        "RATE_SIGNAL": "rate limit",
        "DEADLINE_SIGNAL": "deadline exceeded",
        "SAFETY_SIGNAL": "safety blocked",
        "VALUE_ERROR": "Synthetic value failure",
    }
    if failure in markers:
        error_type = ValueError if failure == "VALUE_ERROR" else RuntimeError
        raise error_type(markers[failure])
    if failure:
        error = RuntimeError(
            "safety blocked" if failure == "SAFETY_400" else "Synthetic provider failure"
        )
        error.status_code = 400 if failure == "SAFETY_400" else int(failure)
        raise error
    return StructuredGenerationResult(case["output"], 7, 2, "synthetic", "fixed")
