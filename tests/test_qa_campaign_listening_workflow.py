"""Offline QA-orchestration evidence, not real-provider content-quality evidence."""
from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
from typing import Any, cast
import wave

import pytest

from scripts.qa_campaign_be_api import Client, audio_metadata
from scripts.qa_campaign_be_workflows import (
    canonical_listening_items,
    listening,
    listening_case,
    validate_listening_audio,
)


def waveform(seconds: float) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(b"\0\0" * round(seconds * 8000))
    return stream.getvalue()


def fixture(tmp_path: Path, *, difficulty: str = "MY_LEVEL", seconds: float = 10.0,
            mode: str = "DICTATION") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audio_path = tmp_path / "reference.wav"
    raw = waveform(seconds)
    audio_path.write_bytes(raw)
    audio = {**audio_metadata(raw, "audio/wav"), "path": str(audio_path)}
    minimum, maximum = {"EASY": (5, 12), "MY_LEVEL": (8, 20), "CHALLENGE": (15, 30)}[difficulty]
    items = [{"itemId": index, "itemIndex": index, "replacementSequence": 0, "status": "READY",
              "sourceText": "synthetic source", "audioDurationMs": round(seconds * 1000),
              "audioContentType": "audio/wav", "audioChecksum": audio["sha256"], "audioObjectKey": f"qa/{index}",
              "generationMetadata": {
                  "durationDemand": {"minSeconds": minimum, "maxSeconds": maximum,
                                     "policyVersion": "listening-audio-duration-v1", "playbackSpeed": "NORMAL"},
                  "qualityCorrectionCount": 0, "referenceMeanings": ["synthetic reference"],
                  "summaryKeyPoints": ["synthetic point"],
                  "correctOptionKey": "A", "options": [{"key": "A"}, {"key": "B"}],
              }} for index in range(1, 6)]
    # Two rejected originals, both replaced successfully. Keep both originals.
    for index in (1, 3):
        replacement = deepcopy(items[index - 1])
        replacement.update(itemId=index + 10, replacementSequence=1)
        replacement["generationMetadata"]["qualityCorrectionCount"] = 1
        items[index - 1].update(status="NOT_EVALUABLE", failureReason="AUDIO_TOO_SHORT", audioDurationMs=None,
                                audioObjectKey=None, audioChecksum=None)
        items[index - 1]["generationMetadata"]["qualityCorrectionCount"] = 1
        items.append(replacement)
    snapshot = {"dailySetId": 42, "status": "READY", "mode": mode, "difficulty": difficulty,
                "physicalItemCount": 7, "targetItemCount": 5, "items": items}
    published = {"dailySetId": 42, "status": "READY", "learningMode": mode, "difficulty": difficulty,
                 "physicalItemCount": 7, "targetItemCount": 5, "readyItemCount": 5, "items": [
                     {key: item[key] for key in ("itemId", "itemIndex", "replacementSequence", "status", "audioDurationMs")}
                     | {"playable": True, "durationValidationStatus": "VALIDATED",
                        "durationPolicyVersion": "listening-audio-duration-v1"}
                     for item in sorted(items, key=lambda item: item["itemIndex"]) if item["status"] == "READY"]}
    return snapshot, published, audio


def test_case_paths_include_mode_and_difficulty_and_reject_unsafe_label():
    assert listening_case("SUMMARY", "EASY", None) == "summary-easy"
    assert listening_case("SUMMARY", "CHALLENGE", None) == "summary-challenge"
    assert listening_case("SUMMARY", "CHALLENGE", "qa-user2-summary-challenge") == "qa-user2-summary-challenge"
    with pytest.raises(ValueError, match="case ID"):
        listening_case("SUMMARY", "MY_LEVEL", "../elsewhere")
    with pytest.raises(ValueError, match="difficulty"):
        listening_case("SUMMARY", "EASIER", None)


def test_canonical_five_slots_with_seven_rows_preserve_all_replacement_history(tmp_path):
    snapshot, published, _ = fixture(tmp_path)
    original = deepcopy(snapshot)
    selected = canonical_listening_items(snapshot, published, "DICTATION", "MY_LEVEL")
    assert [row["itemId"] for row in selected] == [11, 2, 13, 4, 5]
    assert snapshot == original and len(snapshot["items"]) == 7


def test_active_selection_preserves_published_ready_even_with_later_failed_version(tmp_path):
    snapshot, published, _ = fixture(tmp_path)
    failed = deepcopy(snapshot["items"][1])
    failed.update(itemId=102, replacementSequence=1, status="NOT_EVALUABLE")
    snapshot["items"].append(failed)
    snapshot["physicalItemCount"] = published["physicalItemCount"] = 8
    assert canonical_listening_items(snapshot, published, "DICTATION", "MY_LEVEL")[1]["itemId"] == 2


@pytest.mark.parametrize("problem", ["api-id", "api-duplicate-slot", "duplicate-version", "wrong-difficulty", "legacy-audio"])
def test_db_api_identity_or_coverage_mismatch_never_picks_arbitrary_row(tmp_path, problem):
    snapshot, published, _ = fixture(tmp_path)
    if problem == "api-id":
        published["items"][0]["itemId"] = 1
    elif problem == "api-duplicate-slot":
        published["items"][1]["itemIndex"] = 1
    elif problem == "duplicate-version":
        snapshot["items"][-1]["replacementSequence"] = 0
    elif problem == "wrong-difficulty":
        published["difficulty"] = "EASY"
    else:
        published["items"][0]["durationValidationStatus"] = "LEGACY_UNVALIDATED"
    with pytest.raises(ValueError):
        canonical_listening_items(snapshot, published, "DICTATION", "MY_LEVEL")


@pytest.mark.parametrize(("difficulty", "seconds"), [("EASY", 5), ("EASY", 12), ("MY_LEVEL", 8),
                                                       ("MY_LEVEL", 20), ("CHALLENGE", 15), ("CHALLENGE", 30)])
def test_measured_waveform_boundary_acceptance_not_model_estimate(tmp_path, difficulty, seconds):
    snapshot, _, audio = fixture(tmp_path, difficulty=difficulty, seconds=seconds)
    item = snapshot["items"][1]
    item["generationMetadata"]["estimatedAudioSeconds"] = 999
    result = validate_listening_audio(item, audio, difficulty)
    assert result["withinEffectiveBounds"] and result["fullWaveformDecoded"]
    assert result["measuredSeconds"] == seconds


@pytest.mark.parametrize("problem", ["too-short", "effective-upper", "checksum", "duration-metadata", "truncated"])
def test_measured_audio_and_persisted_effective_intersection_must_match(tmp_path, problem):
    snapshot, _, audio = fixture(tmp_path, seconds=7.999 if problem == "too-short" else 10)
    item = snapshot["items"][1]
    if problem == "effective-upper":
        item["generationMetadata"]["durationDemand"]["maxSeconds"] = 9.999
    elif problem == "checksum":
        item["audioChecksum"] = "wrong"
    elif problem == "duration-metadata":
        item["audioDurationMs"] += 20
    elif problem == "truncated":
        path = Path(audio["path"])
        path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError):
        validate_listening_audio(item, audio, "MY_LEVEL")


@pytest.mark.parametrize(("difficulty", "seconds"), [("EASY", 6), ("MY_LEVEL", 10), ("CHALLENGE", 18)])
@pytest.mark.parametrize("audio_in_bounds", [True, False])
def test_complete_workflow_uses_only_published_five_and_keeps_all_history(tmp_path, difficulty, seconds, audio_in_bounds):
    if not audio_in_bounds:
        seconds = {"EASY": 4.999, "MY_LEVEL": 7.999, "CHALLENGE": 14.999}[difficulty]
    snapshot, published, audio = fixture(tmp_path, difficulty=difficulty, seconds=seconds, mode="SUMMARY")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    class FakeClient:
        output = tmp_path
        manifest = {"campaign": "offline-qa"}
        calls: list[tuple[str, dict[str, Any]]] = []
        attempts: list[dict[str, Any]] = []

        def listening_snapshot(self, set_id):
            assert set_id == 42
            return {"privateArtifact": str(snapshot_path)}

        def poll(self, action, *args, **kwargs):
            assert action == "listening-status"
            return {"httpStatus": 200, "response": {"body": published}}

        def call(self, action, **kwargs):
            self.calls.append((action, kwargs))
            if action == "listening-create":
                assert kwargs["payload"]["difficulty"] == difficulty
                assert difficulty.lower() in kwargs["payload"]["idempotencyKey"]
                value = {"dailySetId": 42}
            elif action == "listening-session-create":
                value = {"sessionId": 9}
            elif action == "listening-item":
                identity = kwargs["identifiers"]["item_id"]
                assert identity in {11, 2, 13, 4, 5}
                value = {"attempt": {"attemptId": identity, "tasks": [{"taskType": "SUMMARY", "status": "READY"}]}}
            elif action == "listening-audio":
                return {"httpStatus": 200, "audio": audio}
            elif action == "listening-submit":
                self.attempts.append({"attemptId": kwargs["identifiers"]["attempt_id"], "status": "EVALUATED",
                                      "tasks": [{"taskType": "SUMMARY", "status": "EVALUATED"}]})
                value = {}
            elif action == "listening-session":
                value = {"sessionId": 9, "attempts": self.attempts}
            elif action in {"listening-answer", "listening-complete", "listening-result"}:
                value = {}
            else:
                pytest.fail(action)
            return {"httpStatus": 200, "response": {"body": value}}

    fake = FakeClient()
    if not audio_in_bounds:
        with pytest.raises(ValueError, match="OUTSIDE_EFFECTIVE_BOUNDS"):
            listening(cast(Client, fake), "SUMMARY", difficulty=difficulty)
        partial = json.loads((tmp_path / f"listening-summary-{difficulty.lower()}-workflow.json").read_text())
        assert partial["status"] == "FAILED" and partial["partial"] is True
        assert partial["physicalItemHistory"] == snapshot["items"]
        assert partial["items"][0]["audio"] == audio
        assert not any(action in {"listening-answer", "listening-submit", "listening-complete"} for action, _ in fake.calls)
        return
    report = listening(cast(Client, fake), "SUMMARY", difficulty=difficulty)
    assert report["status"] == "COMPLETED" and len(report["items"]) == 5
    assert report["physicalItemHistory"] == snapshot["items"]
    assert len(report["physicalItemHistory"]) == 7
    assert report["activePublishedItemIds"] == [11, 2, 13, 4, 5]
    assert all(item["durationValidation"]["withinEffectiveBounds"] for item in report["items"])
    assert (tmp_path / f"listening-summary-{difficulty.lower()}-workflow.json").exists()
    assert sum(action == "listening-create" for action, _ in fake.calls) == 1
    assert sum(action == "listening-submit" for action, _ in fake.calls) == 5
    assert not any("retry" in action for action, _ in fake.calls)
    with pytest.raises(ValueError, match="may not rerun"):
        listening(cast(Client, fake), "SUMMARY", difficulty=difficulty)


def test_resume_never_relabels_other_difficulty_or_case(tmp_path):
    path = tmp_path / "listening-summary-my_level-workflow.json"
    path.write_text(json.dumps({"mode": "SUMMARY", "difficulty": "EASY", "caseId": "summary-my_level"}), encoding="utf-8")

    class NeverCalledClient:
        output = tmp_path

        def call(self, *args, **kwargs):
            pytest.fail("Case identity must be checked before any HTTP call")

    with pytest.raises(ValueError, match="identity mismatch"):
        listening(cast(Client, NeverCalledClient()), "SUMMARY", resume_existing=True)
