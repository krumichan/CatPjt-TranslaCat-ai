import pytest
import json
import sys
from types import SimpleNamespace
from typing import Any, cast

from scripts.qa_campaign_be_api import Client as ApiClient
from scripts.qa_campaign_be_workflows import accepted_turn_unchanged, evaluation_terminal, listening_answer, preserve_prior_outcome
from scripts.qa_campaign_be_workflows import speaking


def test_synthetic_answers_do_not_relabel_intent_or_claim_human_performance():
    item = {"sourceText": "abcdefghij", "generationMetadata": {
        "correctOptionKey": "B", "options": [{"key": "A"}, {"key": "B"}],
        "referenceMeanings": ["first", "second"], "summaryKeyPoints": ["one", "two"],
    }}
    assert listening_answer(item, "DICTATION", "REFERENCE") == "abcdefghij"
    assert listening_answer(item, "DICTATION", "PARTIAL") == "abcde"
    assert listening_answer(item, "COMPREHENSION", "REFERENCE") == "B"
    assert listening_answer(item, "COMPREHENSION", "PARTIAL") == "A"
    assert listening_answer(item, "INTERPRETATION", "REFERENCE") == "first second"
    assert listening_answer(item, "INTERPRETATION", "PARTIAL") == "first"
    assert listening_answer(item, "SUMMARY", "PARTIAL") == "one"


def test_real_session_view_not_selected_tasks_do_not_block_selected_terminal_tasks():
    attempt = {"status": "EVALUATED", "tasks": [
        {"taskType": "COMPREHENSION", "status": "NOT_SELECTED"},
        {"taskType": "DICTATION", "status": "EVALUATED"},
        {"taskType": "INTERPRETATION", "status": "EVALUATED"},
        {"taskType": "REPEAT_AFTER_AUDIO", "status": "NOT_SELECTED"},
        {"taskType": "SUMMARY", "status": "NOT_SELECTED"},
    ]}
    assert evaluation_terminal(attempt, ["DICTATION", "INTERPRETATION"])
    attempt["tasks"][2]["status"] = "EVALUATING"
    assert not evaluation_terminal(attempt, ["DICTATION", "INTERPRETATION"])
    attempt["tasks"][2]["status"] = "EVALUATION_FAILED"
    assert evaluation_terminal(attempt, ["DICTATION", "INTERPRETATION"])
    assert not evaluation_terminal(attempt, ["SUMMARY"])
    with pytest.raises(ValueError, match="coverage"):
        evaluation_terminal({"tasks": []}, ["DICTATION"])


def test_resume_preserves_failed_observation_without_stale_current_error():
    report: dict[str, Any] = {"status": "FAILED", "errorType": "TimeoutError", "finishedAt": "past"}
    preserve_prior_outcome(report)
    report["status"] = "COMPLETED"
    assert "errorType" not in report
    assert report["priorObservations"][0]["status"] == "FAILED"
    assert report["priorObservations"][0]["errorType"] == "TimeoutError"


def test_java_nanos_to_mysql_datetime6_precision_does_not_fake_content_change():
    original = {"id": 1, "transcript": "unchanged", "completedAt": "2026-09-19T19:38:20.5282778"}
    observed = {**original, "completedAt": "2026-09-19T19:38:20.528278"}
    assert accepted_turn_unchanged(original, observed)
    assert not accepted_turn_unchanged(original, {**observed, "transcript": "changed"})
    assert not accepted_turn_unchanged(original, {**observed, "completedAt": "2026-09-19T19:38:20.528290"})


@pytest.mark.parametrize("silence_probe", [False, True])
@pytest.mark.parametrize("problem_two_status", ["EVALUATED", "INSUFFICIENT_EVIDENCE"])
def test_read_aloud_real_dto_shape_ten_turns_five_problem_evaluations_no_extra_completion(tmp_path, silence_probe, problem_two_status):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "speaking-read_aloud.json").write_text(json.dumps({"practiceMode": "READ_ALOUD"}), encoding="utf-8")

    class Client:
        output = tmp_path
        calls = []
        turn = 0
        problem = 0

        def ai_runtime_identity(self):
            return {"pid": 123, "sourceSha256": {"service": "fixed-test-hash"}}

        def call(self, action, **kwargs):
            self.calls.append((action, kwargs))
            if action.endswith("-audio"):
                return {"httpStatus": 200, "audio": {"format": "WAV", "mimeMatches": True,
                        "durationSeconds": 4.0, "path": str(tmp_path / "audio.wav")}}
            if action == "speaking-active":
                value = None
            elif action == "speaking-create":
                value = {"id": 7, "openingPromptGuide": {"scriptText": "script"}}
            elif action == "speaking-upload-grant":
                self.turn += 1
                value = {"turnId": self.turn, "uploadToken": "private-upload-secret"}
            elif action == "speaking-turn":
                if kwargs["payload"]["durationSeconds"] == 3.0:
                    assert silence_probe and self.turn == 1 and not kwargs["payload"]["rerecord"]
                    value = {"id": 1, "status": "PARTIAL_FAILURE", "errorCode": "SILENCE_DETECTED", "recordingRevision": 0}
                else:
                    assert kwargs["payload"]["durationSeconds"] == 4.0
                    value = {"id": self.turn, "status": "READY", "durationSeconds": 4.0,
                         "assistantAudioUrl": "local-reference" if self.turn % 2 == 0 and self.turn < 10 else None}
            elif action == "speaking-turn-status":
                value = {"id": 1, "status": "PARTIAL_FAILURE", "errorCode": "SILENCE_DETECTED", "recordingRevision": 0}
            elif action == "speaking-rerecord-grant":
                value = {"turnId": 1, "uploadToken": "private-rerecord-secret"}
            elif action == "speaking-problem-evaluate":
                self.problem += 1
                value = {"status": "PENDING"}
            elif action == "speaking-problem-evaluations":
                value = [{"problemIndex": self.problem, "status": problem_two_status if self.problem == 2 else "EVALUATED"}]
            elif action == "speaking-session":
                value = {"session": {"status": "COMPLETED", "evaluationStatus": "EVALUATED"}}
            elif action == "speaking-evaluation":
                value = {"overallScore": 99}
            else:
                pytest.fail(action)
            return {"httpStatus": 200, "response": {"body": value}}

    client = Client()
    # Deliberately no real constructor/network/auth; fake implements only the
    # surface exercised by this workflow contract test.
    result = speaking(cast(ApiClient, client), "READ_ALOUD", silence_rerecord_probe=silence_probe)
    assert result["status"] == ("COMPLETED" if problem_two_status == "EVALUATED" else "COMPLETED_WITH_NON_EVALUABLE_PROBLEM")
    assert result["problemEvaluations"][1]["status"] == problem_two_status
    assert len(result["turns"]) == 10 and len(result["problemEvaluations"]) == 5
    assert result["actualAcceptedAudioSeconds"] == 40.0
    actions = [row[0] for row in client.calls]
    assert actions.count("speaking-turn") == 10 + int(silence_probe)
    assert actions.count("speaking-problem-evaluate") == 5
    assert "speaking-complete" not in actions
    assert "private-upload-secret" not in (tmp_path / "speaking-read_aloud-workflow.json").read_text()


def test_repaired_problem_resume_reads_accepted_turns_and_never_resubmits_them(tmp_path):
    audio = {"path": str(tmp_path / "reference.wav"), "durationSeconds": 4.0}
    responses = [{"id": order, "status": "READY", "durationSeconds": 4.0} for order in (1, 2)]
    old = {"mode": "READ_ALOUD", "status": "EVALUATION_NOT_EVALUATED_NO_RETRY", "sessionId": 7,
           "opening": {}, "openingAudio": audio, "problemEvaluations": [{"problemIndex": 1, "status": "FAILED"}],
           "turns": [{"turnIndex": order, "turnId": order, "response": responses[order - 1], "assistantAudio": audio} for order in (1, 2)]}
    (tmp_path / "speaking-read_aloud-workflow.json").write_text(json.dumps(old), encoding="utf-8")

    class Client:
        output = tmp_path
        calls = []

        def ai_runtime_identity(self):
            return {"pid": 123}

        def call(self, action, **kwargs):
            self.calls.append((action, kwargs))
            if action == "speaking-session":
                value = {"session": {"status": "IN_PROGRESS"}}
            elif action == "speaking-problem-evaluations":
                value = [{"problemIndex": 1, "status": "EVALUATED", "manualRetryCount": 1}]
            elif action == "speaking-turn-status":
                value = responses[kwargs["identifiers"]["turn_id"] - 1]
            elif action == "speaking-upload-grant":
                assert kwargs["payload"]["turnIndex"] == 3
                raise RuntimeError("TEST_STOP_BEFORE_FIRST_UNSUBMITTED_TURN")
            else:
                pytest.fail("Accepted prefix must not replay: " + action)
            return {"httpStatus": 200, "response": {"body": value}}

    client = Client()
    with pytest.raises(RuntimeError, match="TEST_STOP_BEFORE"):
        speaking(cast(ApiClient, client), "READ_ALOUD", resume_existing=True)
    assert [row[0] for row in client.calls].count("speaking-turn-status") == 2
    report = json.loads((tmp_path / "speaking-read_aloud-workflow.json").read_text())
    assert report["manualRetries"] == 1
    assert report["priorProblemEvaluations"] == old["problemEvaluations"]
    assert all(row["resumedReadbackOnly"] for row in report["turns"][:2])


@pytest.mark.parametrize("changed_transcript", [False, True])
def test_guided_authorized_retry_readback_preserves_stt_and_original_audio_before_next_turn(tmp_path, monkeypatch, changed_transcript):
    from scripts import qa_campaign_be_workflows as workflow
    audio = {"format": "WAV", "mimeMatches": True, "durationSeconds": 14.0, "sha256": "same-audio", "path": "private.wav"}
    old_response = {"id": 11, "turnIndex": 1, "status": "PARTIAL_FAILURE", "recordingRevision": 0,
                    "transcript": "fixed successful STT", "sttConfidence": 0.75, "durationSeconds": 14.0,
                    "manualRetryCount": 0, "failedStage": "CONVERSATION", "errorCode": "INVALID_RESPONSE_SCHEMA"}
    old = {"mode": "GUIDED", "status": "TURN_FAILED_NO_RETRY", "sessionId": 2, "opening": {}, "openingAudio": audio,
           "problemEvaluations": [], "turns": [{"turnIndex": 1, "turnId": 11, "response": old_response,
                                                 "learnerAudio": audio, "status": "PARTIAL_FAILURE"}]}
    (tmp_path / "speaking-guided-workflow.json").write_text(json.dumps(old), encoding="utf-8")

    class Client:
        output = tmp_path
        calls = []

        def ai_runtime_identity(self):
            return {"pid": 123}

        def call(self, action, **kwargs):
            self.calls.append(action)
            if action.endswith("-audio"):
                return {"httpStatus": 200, "audio": audio}
            if action == "speaking-session":
                value = {"session": {"status": "IN_PROGRESS"}}
            elif action == "speaking-turn-status":
                value = {**old_response, "status": "READY", "manualRetryCount": 1,
                         "transcript": "changed" if changed_transcript else old_response["transcript"],
                         "assistantAudioUrl": "local", "failedStage": None, "errorCode": None}
            else:
                pytest.fail("Resume never replays first turn: " + action)
            return {"httpStatus": 200, "response": {"body": value}}

    async def stop_before_new_audio(client, mode, turn, text):
        assert turn == 2
        raise RuntimeError("TEST_STOP_BEFORE_SECOND_SYNTHETIC_AUDIO")

    monkeypatch.setitem(sys.modules, "scripts.qa_campaign_speaking", SimpleNamespace(_learner_text=lambda *args: "synthetic"))
    monkeypatch.setattr(workflow, "_synthesize_learner", stop_before_new_audio)
    client = Client()
    if changed_transcript:
        result = speaking(cast(ApiClient, client), "GUIDED", resume_existing=True)
        assert result["status"] == "AUTHORIZED_RETRY_DID_NOT_PRESERVE_STT_NO_FURTHER_RETRY"
        assert "speaking-user-audio" not in client.calls
    else:
        with pytest.raises(RuntimeError, match="TEST_STOP_BEFORE_SECOND"):
            speaking(cast(ApiClient, client), "GUIDED", resume_existing=True)
        saved = json.loads((tmp_path / "speaking-guided-workflow.json").read_text())
        assert saved["turns"][0]["failureBeforeAuthorizedRetry"] == old_response
        assert saved["turns"][0]["resumedReadbackOnly"] is True
        assert saved["manualRetries"] == 1
