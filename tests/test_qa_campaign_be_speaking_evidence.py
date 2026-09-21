from __future__ import annotations

import io
import json
import wave
from typing import Any, cast

import pytest

from scripts.qa_campaign_be_api import Client, audio_metadata
from scripts.qa_campaign_be_workflows import _audio, speaking, speaking_evidence_v2


def evaluation(mode: str = "GUIDED") -> dict[str, Any]:
    axes = ["MEANING"] if mode == "READ_ALOUD" else [
        "GRAMMAR", "VOCABULARY", "NATURALNESS", "MEANING", "EXPRESSIVENESS", "INTERACTION",
    ]
    return {
        "status": "EVALUATED", "overallScore": 70,
        "evidencePolicyVersion": "speaking-transcript-evidence-v2",
        "evidenceSource": "TRANSCRIPT_OBSERVATION", "evaluatedAxes": axes,
        "evaluationCoverage": 0.1 if mode == "READ_ALOUD" else 0.65,
        "pronunciationPracticeJson": "[]",
        "metrics": [{"metricType": axis, "state": "EVALUATED" if axis in axes else "NOT_EVALUABLE",
                     "score": 70 if axis in axes else None}
                    for axis in ("GRAMMAR", "VOCABULARY", "NATURALNESS", "MEANING", "EXPRESSIVENESS",
                                 "FLUENCY", "PRONUNCIATION", "INTERACTION")],
    }


@pytest.mark.parametrize("mode", ["READ_ALOUD", "GUIDED", "FREE"])
def test_observed_v2_result_preserves_axes_coverage_and_two_unsupported_axes(mode):
    value = evaluation(mode)
    observed = speaking_evidence_v2(value, mode, metrics_required=True)
    assert observed["verifiedNotEvaluableAxes"] == ["PRONUNCIATION", "FLUENCY"]
    assert observed["evaluatedAxes"] == value["evaluatedAxes"]
    assert observed["evaluationCoverage"] == value["evaluationCoverage"]
    assert value["overallScore"] == 70
    if mode == "READ_ALOUD":
        assert observed["evaluatedAxes"] == ["MEANING"]


def test_read_aloud_problem_dto_does_not_fabricate_unexposed_metric_rows():
    value = evaluation("READ_ALOUD")
    value.pop("metrics")
    value.pop("pronunciationPracticeJson")
    observed = speaking_evidence_v2(value, "READ_ALOUD", metrics_required=False)
    assert observed["evaluatedAxes"] == ["MEANING"]
    assert observed["verifiedNotEvaluableAxes"] == []
    assert observed["metricRowsExposed"] is False
    with pytest.raises(ValueError, match="METRIC_ROWS_MISSING"):
        speaking_evidence_v2(value, "READ_ALOUD", metrics_required=True)


@pytest.mark.parametrize("mutation,code", [
    (lambda value: value.pop("evidencePolicyVersion"), "POLICY_MISSING"),
    (lambda value: value.update(evaluationCoverage=1), "COVERAGE_MISMATCH"),
    (lambda value: value["evaluatedAxes"].append("PRONUNCIATION"), "AXES_INVALID"),
    (lambda value: value["metrics"][5].update(score=50), "SCORED_UNSUPPORTED_ACOUSTICS"),
    (lambda value: value["metrics"][3].update(state="NOT_EVALUABLE"), "ROWS_DISAGREE"),
    (lambda value: value.update(pronunciationPracticeJson='[{"instruction":"unsupported"}]'), "PRONUNCIATION_PRACTICE"),
])
def test_invalid_evidence_is_reported_not_relabelled(mutation, code):
    value = evaluation()
    mutation(value)
    with pytest.raises(ValueError, match=code):
        speaking_evidence_v2(value, "GUIDED", metrics_required=True)


def test_insufficient_evidence_is_observed_without_requiring_fake_positive_score():
    value = evaluation()
    value.update(status="INSUFFICIENT_EVIDENCE", overallScore=None,
                 evaluatedAxes=[], evaluationCoverage=0, metrics=[])
    observed = speaking_evidence_v2(value, "FREE", metrics_required=True)
    assert observed["evaluatedAxes"] == []
    assert observed["verifiedNotEvaluableAxes"] == []
    # Existing service retains metric observations when global confidence is low.
    model_insufficient = evaluation()
    model_insufficient.update(status="INSUFFICIENT_EVIDENCE", overallScore=None)
    assert speaking_evidence_v2(model_insufficient, "FREE", metrics_required=True)["evaluationCoverage"] == 0.65


def wav_bytes() -> bytes:
    result = io.BytesIO()
    with wave.open(result, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x01\x00" * 16000)
    return result.getvalue()


def audio_record(tmp_path, raw: bytes):
    path = tmp_path / "download.wav"
    path.write_bytes(raw)
    return {"httpStatus": 200, "audio": {**audio_metadata(raw, "audio/wav"), "path": str(path)}}


def test_download_full_frames_and_recorded_sha_are_verified(tmp_path):
    record = audio_record(tmp_path, wav_bytes())
    result = _audio(record, verify_bytes=True)
    assert result["recordedSha256Verified"] is True
    assert result["fullWaveformDecoded"] is True
    assert result["sha256"] == record["audio"]["sha256"]


def test_download_checksum_and_truncated_frames_cannot_pass(tmp_path):
    record = audio_record(tmp_path, wav_bytes())
    record["audio"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="CHECKSUM"):
        _audio(record, verify_bytes=True)
    truncated = audio_record(tmp_path, wav_bytes()[:-2])
    with pytest.raises(ValueError, match="FRAMES_INVALID"):
        _audio(truncated, verify_bytes=True)


@pytest.mark.parametrize("mode", ["READ_ALOUD", "GUIDED", "FREE"])
def test_v2_full_fixed_workflow_records_evidence_and_sha_without_extra_actions(tmp_path, monkeypatch, mode):
    from scripts import qa_campaign_be_workflows as workflow

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x01\x00" * 16000 * 12)
    audio = audio_record(tmp_path, buffer.getvalue())
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "fixtures" / f"speaking-{mode.lower()}.json").write_text("{}", encoding="utf-8")

    class FakeClient:
        output = tmp_path

        def __init__(self):
            self.actions = []
            self.turn = 0
            self.problem = 0

        def ai_runtime_identity(self):
            return {"source": "explicit offline fake"}

        def call(self, action, **kwargs):
            self.actions.append(action)
            if action.endswith("-audio"):
                return audio
            if action == "speaking-active":
                value = None
            elif action == "speaking-create":
                value = {"id": 99}
            elif action == "speaking-upload-grant":
                self.turn += 1
                value = {"turnId": self.turn, "uploadToken": "fake-private-grant"}
            elif action == "speaking-turn":
                value = {"id": self.turn, "status": "READY", "durationSeconds": 12}
            elif action == "speaking-problem-evaluate":
                self.problem += 1
                value = {"status": "PENDING"}
            elif action == "speaking-problem-evaluations":
                value = [{"problemIndex": self.problem, **evaluation(mode)}]
            elif action == "speaking-session":
                value = {"session": {"status": "COMPLETED", "evaluationStatus": "EVALUATED"}}
            elif action == "speaking-complete":
                value = {"status": "COMPLETED"}
            elif action == "speaking-evaluation":
                value = evaluation(mode)
            else:
                raise AssertionError(action)
            return {"httpStatus": 200, "response": {"body": value}}

    async def synthetic(*args):
        return {**audio["audio"], "source": "fake audio, not linguistic quality evidence"}

    monkeypatch.setattr(workflow, "_synthesize_learner", synthetic)
    client = FakeClient()
    result = speaking(cast(Client, client), mode, verify_evidence_v2=True)
    expected = 10 if mode == "READ_ALOUD" else 6
    assert result["status"] == "COMPLETED"
    assert result["evidenceV2Check"]["verifiedNotEvaluableAxes"] == ["PRONUNCIATION", "FLUENCY"]
    assert len(result["turns"]) == expected
    assert client.actions.count("speaking-turn") == expected
    assert client.actions.count("speaking-user-audio") == expected
    assert client.actions.count("speaking-problem-evaluate") == (5 if mode == "READ_ALOUD" else 0)
    assert client.actions.count("speaking-complete") == (0 if mode == "READ_ALOUD" else 1)
    assert all(row["uploadedAudioSha256MatchesReadback"] for row in result["turns"])
    artifact = (tmp_path / f"speaking-{mode.lower()}-workflow.json").read_text(encoding="utf-8")
    assert "fake-private-grant" not in artifact
    assert json.loads(artifact)["evidenceV2VerificationRequired"] is True
