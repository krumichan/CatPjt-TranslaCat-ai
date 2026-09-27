import json

import pytest

from scripts.leveltest_test_provider import _fixtures, execute_leveltest_fixture


def test_leveltest_fixture_requires_explicit_test_marker(monkeypatch):
    # 준비
    monkeypatch.delenv("TRANSLACAT_TEST_MODEL_EXECUTION", raising=False)

    # 실행·검증
    with pytest.raises(RuntimeError, match="explicit local test marker"):
        execute_leveltest_fixture({"schema_name_value": "ListeningINTERPRETATION"})


def test_leveltest_fixture_accepts_only_recorded_synthetic_text(monkeypatch):
    # 준비
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")
    fixture = _fixtures("writing")[0]
    kwargs = {
        "schema_name_value": "LANGUAGE_LEARNING_WRITING_EVALUATION",
        "messages": [{"role": "user", "content": fixture["prompt"]}],
    }

    # 실행
    result = execute_leveltest_fixture(kwargs)
    unknown = execute_leveltest_fixture(
        {**kwargs, "messages": [{"role": "user", "content": "Unknown synthetic input"}]}
    )

    # 검증
    assert result is not None and result.data == fixture["providerOutput"]
    assert result.provider == "test-provider"
    assert unknown is None


def test_leveltest_runtime_only_returns_outputs_for_local_synthetic_generation(monkeypatch):
    # 준비
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")
    payload = {
        "requestId": "lt:synthetic:q1",
        "originLanguage": "ko",
        "learningLanguage": "en",
        "questionNumber": 1,
        "itemType": "VOCAB_CONTEXT_CHOICE",
        "targetComplexityBand": 4,
        "preferredScenarioCategories": ["TRAVEL"],
    }
    kwargs = {
        "schema_name_value": "LevelTestQuestionGenerationPayload",
        "messages": [{"role": "user", "content": "Synthetic prompt\n" + json.dumps(payload)}],
    }

    # 실행
    result = execute_leveltest_fixture(kwargs)
    payload["requestId"] = "unrecognized-request"
    unknown = execute_leveltest_fixture(
        {
            **kwargs,
            "messages": [{"role": "user", "content": "Synthetic prompt\n" + json.dumps(payload)}],
        }
    )

    # 검증: 모델 후보만 공급하고 선택·점수·저장은 반환하지 않는다.
    assert result is not None
    assert set(result.data) == {"candidates"}
    assert result.data["candidates"][0]["complexityBand"] == 4
    assert result.data["candidates"][0]["diversityMetadata"]["scenarioCategory"] == "TRAVEL"
    assert unknown is None
