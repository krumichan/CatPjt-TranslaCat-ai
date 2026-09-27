import json

import pytest

from scripts.speaking_evaluation_test_provider import execute_speaking_evaluation_fixture


def _request(name="guided_supported"):
    return {
        "schema_name_value": "LANGUAGE_LEARNING_SPEAKING_EVALUATION",
        "messages": [
            {
                "role": "user",
                "content": "synthetic\n"
                + json.dumps({"sessionId": "speaking-evaluation-http-" + name}),
            }
        ],
    }


def test_requires_explicit_test_marker(monkeypatch):
    # 준비
    monkeypatch.delenv("TRANSLACAT_TEST_MODEL_EXECUTION", raising=False)

    # 실행 및 검증
    with pytest.raises(RuntimeError, match="explicit test marker"):
        execute_speaking_evaluation_fixture(_request())


def test_returns_model_output_only(monkeypatch):
    # 준비
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")

    # 실행
    result = execute_speaking_evaluation_fixture(_request())

    # 검증: 실제 업무 response를 반환하지 않고 원본 모델 출력만 고정한다.
    assert result.provider == "synthetic"
    assert len(result.data["metrics"]) == 6
    assert "overallScore" not in result.data
    assert "eligibility" not in result.data


def test_unrecognized_task_cannot_use_fixture():
    # 준비
    request = _request()
    request["schema_name_value"] = "UNRELATED"

    # 실행 및 검증
    assert execute_speaking_evaluation_fixture(request) is None
