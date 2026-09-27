import json

import pytest

from scripts import speaking_runtime_test_provider as fixture


def _request(task, payload):
    return {
        "schema_name_value": "LANGUAGE_LEARNING_SPEAKING_" + task,
        "messages": [
            {
                "role": "user",
                "content": "synthetic\n"
                + json.dumps(
                    {
                        "topic": "TRANSLACAT_SYNTHETIC_SPEAKING_TEST",
                        "originLanguage": "ko",
                        "learningLanguage": "en",
                        **payload,
                    }
                ),
            }
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["FREE", "GUIDED", "READ_ALOUD"])
async def test_runtime_conversation_returns_only_model_payload(monkeypatch, tmp_path, mode):
    # 준비
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")
    monkeypatch.setattr(fixture, "ROOT", tmp_path)

    # 실행
    result = await fixture.execute_speaking_runtime_fixture(
        _request("CONVERSATION", {"practiceMode": mode})
    )

    # 검증
    assert result.provider == "synthetic"
    assert "usage" not in result.data
    assert "sessionId" not in result.data
    if mode == "READ_ALOUD":
        assert result.data["scriptText"] == result.data["assistantText"]
    if mode == "GUIDED":
        assert result.data["providedFacts"]


@pytest.mark.asyncio
async def test_runtime_evaluation_binds_only_submitted_evidence(monkeypatch, tmp_path):
    # 준비
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    request = _request(
        "EVALUATION",
        {
            "evaluationCapabilities": {
                "textEvidenceTurnIds": ["101", "102"],
                "modelAssessableMetrics": ["MEANING"],
            }
        },
    )

    # 실행
    result = await fixture.execute_speaking_runtime_fixture(request)

    # 검증: 점수·적격성·Growth는 Kotlin이 원래 업무 정책으로 결정한다.
    assert result.data["metrics"][0]["evidence"][0]["turnId"] == "101"
    assert "overallScore" not in result.data
    assert "eligibility" not in result.data


@pytest.mark.asyncio
async def test_coaching_failure_is_an_invalid_model_field_not_a_server_response(
    monkeypatch, tmp_path
):
    # 준비
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    folder = tmp_path
    (folder / "speaking-control.json").write_text(
        json.dumps({"scenario": "wrong_snapshot"}), encoding="utf-8"
    )

    # 실행
    result = await fixture.execute_speaking_runtime_fixture(
        _request(
            "SESSION_COACHING",
            {"eligibleLearnerTurns": [{"turnId": "101", "transcript": "Synthetic learner text."}]},
        )
    )

    # 검증
    assert result.data["sourceSnapshotHash"] == "forbidden-model-owned-snapshot"
    assert "evidence" not in result.data["items"][0]
    assert "recordingRevision" not in result.data["items"][0]
