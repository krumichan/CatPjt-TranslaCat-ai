"""Fixed, no-retry synthetic QA scenarios through normal isolated BE HTTP APIs."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import wave
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_be_api import Client, audio_metadata, utc_now
from scripts.qa_campaign_integration import _write_json


def body(record: dict[str, Any]) -> Any:
    if record.get("httpStatus") != 200:
        raise RuntimeError("QA_HTTP_STATUS_" + str(record.get("httpStatus")))
    return record["response"]["body"]


def listening_answer(item: dict[str, Any], task: str, scenario: str) -> str:
    generated = item["generationMetadata"]
    if task == "COMPREHENSION":
        correct = generated["correctOptionKey"]
        return correct if scenario == "REFERENCE" else next(option["key"] for option in generated["options"] if option["key"] != correct)
    if scenario == "WRONG":
        return "今日は動物園で象を見ました。天気がよかったです。" if task == "DICTATION" else "오늘은 동물원에서 코끼리를 보았고 날씨가 좋았습니다."
    if task == "DICTATION":
        answer = item["sourceText"]
        return answer if scenario == "REFERENCE" else answer[:max(1, len(answer) // 2)]
    points = generated["referenceMeanings"] if task == "INTERPRETATION" else generated["summaryKeyPoints"]
    if not points:
        raise ValueError("Reference fixture lacks source-defined scoring oracle")
    return " ".join(points if scenario == "REFERENCE" else points[:1])


def evaluation_terminal(attempt: dict[str, Any], selected_tasks: list[str]) -> bool:
    selected = [item for item in attempt["tasks"] if item["taskType"] in selected_tasks]
    if len(selected) != len(selected_tasks) or {item["taskType"] for item in selected} != set(selected_tasks):
        raise ValueError("Selected task coverage missing from BE SessionView")
    return all(item["status"] in {"EVALUATED", "NOT_EVALUABLE", "EVALUATION_FAILED", "SKIPPED"} for item in selected)


def preserve_prior_outcome(report: dict[str, Any]) -> None:
    """Keep a failed observation distinct from the current resumed execution."""
    if report.get("errorType"):
        report.setdefault("priorObservations", []).append({
            "status": report["status"], "errorType": report.pop("errorType"),
            "finishedAt": report.get("finishedAt"), "source": "Preserved checkpoint",
        })


def accepted_turn_unchanged(original: dict[str, Any], observed: dict[str, Any]) -> bool:
    """Process response uses Java nanos; persisted DATETIME(6) rounds to micros."""
    if {key: value for key, value in original.items() if key != "completedAt"} != {
        key: value for key, value in observed.items() if key != "completedAt"
    }:
        return False
    if original.get("completedAt") == observed.get("completedAt"):
        return True
    try:
        delta = datetime.fromisoformat(original["completedAt"]) - datetime.fromisoformat(observed["completedAt"])
        return abs(delta.total_seconds()) <= 0.000001
    except (KeyError, TypeError, ValueError):
        return False


_LISTENING_DURATION_BOUNDS = {"EASY": (5.0, 12.0), "MY_LEVEL": (8.0, 20.0), "CHALLENGE": (15.0, 30.0)}


def listening_case(mode: str, difficulty: str, case_id: str | None) -> str:
    if mode not in {"DICTATION", "COMPREHENSION", "SUMMARY"}:
        raise ValueError("Unsupported source-defined mode")
    if difficulty not in _LISTENING_DURATION_BOUNDS:
        raise ValueError("Unsupported source-defined Listening difficulty")
    identity = case_id if case_id is not None else f"{mode.lower()}-{difficulty.lower()}"
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", identity):
        raise ValueError("Safe unique Listening case ID required")
    return identity


def canonical_listening_items(snapshot: dict[str, Any], published: dict[str, Any],
                              mode: str, difficulty: str) -> list[dict[str, Any]]:
    """Mirror BE activeItems: READY wins, then greatest replacement sequence.

    All physical rows remain in the private snapshot/report. This only selects
    the five published logical slots, never turns a rejected row into a pass.
    """
    if (snapshot["dailySetId"] != published["dailySetId"] or snapshot["mode"] != mode
            or published["learningMode"] != mode or snapshot["difficulty"] != difficulty
            or published["difficulty"] != difficulty):
        raise ValueError("LISTENING_SET_DB_API_IDENTITY_MISMATCH")
    rows = snapshot["items"]
    if (snapshot["physicalItemCount"] != len(rows)
            or published["physicalItemCount"] != len(rows)
            or snapshot["targetItemCount"] != 5 or published["targetItemCount"] != 5):
        raise ValueError("LISTENING_PHYSICAL_OR_TARGET_COUNT_MISMATCH")
    selected: dict[int, dict[str, Any]] = {}
    identities: set[int] = set()
    versions: set[tuple[int, int]] = set()
    for item in sorted(rows, key=lambda row: (row["replacementSequence"], row["itemIndex"])):
        index, sequence, identity = item["itemIndex"], item["replacementSequence"], item["itemId"]
        if (type(index) is not int or index not in range(1, 6) or type(sequence) is not int
                or sequence < 0 or type(identity) is not int or identity <= 0
                or identity in identities or (index, sequence) in versions):
            raise ValueError("LISTENING_PHYSICAL_ROW_IDENTITY_INVALID")
        identities.add(identity)
        versions.add((index, sequence))
        previous = selected.get(index)
        if previous is None or item["status"] == "READY" or previous["status"] != "READY":
            selected[index] = item
    if set(selected) != set(range(1, 6)) or len(published["items"]) != 5:
        raise ValueError("LISTENING_ACTIVE_SLOT_COVERAGE_INVALID")
    by_index = {item["itemIndex"]: item for item in published["items"]}
    if set(by_index) != set(selected):
        raise ValueError("LISTENING_PUBLISHED_SLOT_COVERAGE_INVALID")
    for index, item in selected.items():
        api = by_index[index]
        if any(api[key] != item[key] for key in ("itemId", "itemIndex", "replacementSequence", "status", "audioDurationMs")):
            raise ValueError("LISTENING_ACTIVE_DB_API_MISMATCH")
        if (item["status"] != "READY" or api.get("playable") is not True
                or api.get("durationValidationStatus") != "VALIDATED"
                or api.get("durationPolicyVersion") != "listening-audio-duration-v1"):
            raise ValueError("LISTENING_ACTIVE_AUDIO_NOT_PUBLISHED_VALIDATED")
    return [selected[index] for index in range(1, 6)]


def validate_listening_audio(item: dict[str, Any], audio: dict[str, Any], difficulty: str) -> dict[str, Any]:
    """Independently decode the exact downloaded bytes; estimates are not evidence."""
    raw = Path(audio["path"]).read_bytes()
    measured = audio_metadata(raw, audio["contentType"])
    if measured.get("format") != "WAV" or not measured["mimeMatches"]:
        raise ValueError("LISTENING_REFERENCE_AUDIO_MIME_OR_STRUCTURE_INVALID")
    with wave.open(io.BytesIO(raw), "rb") as source:
        expected_bytes = source.getnframes() * source.getnchannels() * source.getsampwidth()
        if (source.getcomptype() != "NONE" or expected_bytes <= 0
                or len(source.readframes(source.getnframes())) != expected_bytes):
            raise ValueError("LISTENING_REFERENCE_AUDIO_FRAMES_INVALID")
    if measured["sha256"] != audio["sha256"] or measured["sha256"] != item["audioChecksum"]:
        raise ValueError("LISTENING_REFERENCE_AUDIO_CHECKSUM_MISMATCH")
    seconds = measured["durationSeconds"]
    if (item["audioContentType"] != audio["contentType"] or item["audioDurationMs"] is None
            or abs(seconds * 1000 - item["audioDurationMs"]) > 1):
        raise ValueError("LISTENING_REFERENCE_AUDIO_DB_METADATA_MISMATCH")
    demand = item["generationMetadata"].get("durationDemand") or {}
    minimum, maximum = demand.get("minSeconds"), demand.get("maxSeconds")
    lower, upper = _LISTENING_DURATION_BOUNDS[difficulty]
    if (not isinstance(minimum, (int, float)) or isinstance(minimum, bool)
            or not isinstance(maximum, (int, float)) or isinstance(maximum, bool)
            or not math.isfinite(minimum) or not math.isfinite(maximum)
            or not lower <= minimum <= maximum <= upper
            or demand.get("policyVersion") != "listening-audio-duration-v1"
            or demand.get("playbackSpeed") != "NORMAL"):
        raise ValueError("LISTENING_PERSISTED_DURATION_DEMAND_INVALID")
    if not minimum <= seconds <= maximum:
        raise ValueError("LISTENING_PUBLISHED_AUDIO_OUTSIDE_EFFECTIVE_BOUNDS")
    return {"measuredSeconds": seconds, "minSeconds": minimum, "maxSeconds": maximum,
            "policyVersion": demand["policyVersion"], "playbackSpeed": "NORMAL", "withinEffectiveBounds": True,
            "dbApiChecksumMatches": True, "fullWaveformDecoded": True,
            "basis": "Persisted generation demand; policy subset verified, not model estimated duration"}


def listening(client: Client, mode: str, *, difficulty: str = "MY_LEVEL", case_id: str | None = None,
              resume_existing: bool = False) -> dict[str, Any]:
    case = listening_case(mode, difficulty, case_id)
    artifact = client.output / f"listening-{case}-workflow.json"
    if artifact.exists() and not resume_existing:
        raise ValueError("Fixed BE mode workflow may not rerun")
    report: dict[str, Any] = {"mode": mode, "difficulty": difficulty, "caseId": case, "status": "STARTED", "partial": True,
                              "startedAt": utc_now(), "items": [], "syntheticReferenceOracle": True,
                              "humanListeningQualityEvidence": False, "manualRetries": 0,
                              "submittedDurationBasis": "Measured QA wall clock, NOT human learner effort"}
    if resume_existing:
        previous = json.loads(artifact.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in (("mode", mode), ("difficulty", difficulty), ("caseId", case))):
            raise ValueError("Listening checkpoint case identity mismatch")
        generation_pending = (previous["status"] == "GENERATION_NOT_FIVE_READY" and not previous.get("items")
                              and not previous.get("sessionId") and previous.get("generation", {}).get("generationInProgress"))
        if previous["status"] == "GENERATION_NOT_FIVE_READY" and not previous.get("items") and not previous.get("sessionId"):
            old_snapshot = json.loads(Path(previous["snapshotArtifact"]).read_text(encoding="utf-8"))
            generation_pending = generation_pending or any(item["status"] == "TTS_PENDING" for item in old_snapshot["items"])
        evaluation_poll_timeout = (previous["status"] == "FAILED" and previous.get("errorType") == "TimeoutError"
                                   and previous.get("sessionId") is not None)
        if not (generation_pending or evaluation_poll_timeout):
            raise ValueError("Only unfinished existing-generation observation may resume; no fresh generation")
        _write_json(artifact.with_name(artifact.stem + f"-checkpoint-{time.time_ns()}.json"), previous)
        preserve_prior_outcome(previous)
        report = {**previous, "status": "RESUMED_EXISTING_GENERATION_ONLY", "resumedAt": utc_now(),
                  "priorObservationPreserved": True,
                  "priorSubmittedDurationBasis": "Initial DICTATION attempt used reference-audio milliseconds, not measured learner time",
                  "submittedDurationBasis": "Subsequent submits/completion use QA monotonic wall clock, not human effort"}
    _write_json(artifact, report)
    try:
        if resume_existing:
            set_id = report["dailySetId"]
        else:
            created = body(client.call("listening-create", allow_provider=True, payload={
                "itemCount": 5, "difficulty": difficulty, "learningMode": mode,
                "idempotencyKey": f"{client.manifest['campaign']}-be-{case}-1",
            }))
            set_id = created["dailySetId"]
        report["dailySetId"] = set_id
        _write_json(artifact, report)
        terminal = body(client.poll("listening-status", {"set_id": set_id}, max_polls=60, max_seconds=600))
        report["generation"] = {key: terminal.get(key) for key in (
            "status", "targetItemCount", "physicalItemCount", "readyItemCount", "failureReason", "generationInProgress", "items",
        )}
        snapshot_summary = client.listening_snapshot(set_id)
        snapshot = json.loads(Path(snapshot_summary["privateArtifact"]).read_text(encoding="utf-8"))
        report["snapshotArtifact"] = snapshot_summary["privateArtifact"]
        report.setdefault("snapshotArtifacts", []).append(snapshot_summary["privateArtifact"])
        report["physicalItemHistory"] = snapshot["items"]
        report["physicalItemCount"] = len(snapshot["items"])
        if terminal["status"] != "READY" or terminal["readyItemCount"] != 5:
            report["status"] = "GENERATION_NOT_FIVE_READY"
            return report
        active = canonical_listening_items(snapshot, terminal, mode, difficulty)
        report["activePublishedItemIds"] = [item["itemId"] for item in active]
        report["dbApiPublishedItemsAgree"] = True
        tasks = ["DICTATION", "INTERPRETATION"] if mode == "DICTATION" else [mode]
        if report.get("sessionId") is not None:
            session = body(client.call("listening-session", identifiers={"session_id": report["sessionId"]}))
        else:
            session = body(client.call("listening-session-create", payload={
                "dailySetId": set_id, "selectedTaskTypes": tasks,
                "idempotencyKey": f"{client.manifest['campaign']}-be-{case}-session-1",
            }))
        session_id = session["sessionId"]
        session_began = time.monotonic()
        report["sessionId"] = session_id
        for item, scenario in zip(active, ("REFERENCE", "PARTIAL", "WRONG", "REFERENCE", "PARTIAL"), strict=True):
            attempt_began = time.monotonic()
            item_id = item["itemId"]
            current = body(client.call("listening-item", identifiers={"session_id": session_id, "item_id": item_id}))
            attempt_id = current["attempt"]["attemptId"]
            existing_row = next((row for row in report["items"] if row["itemId"] == item_id), None)
            if evaluation_terminal(current["attempt"], tasks):
                if existing_row is None:
                    raise ValueError("Unexpected already-submitted attempt outside this workflow checkpoint")
                existing_row.update(status=current["attempt"]["status"], evaluation=current["attempt"], resumedFromReadback=True)
                if any(task["status"] == "EVALUATION_FAILED" for task in current["attempt"]["tasks"]):
                    report["status"] = "EVALUATION_FAILED_NO_RETRY"
                    return report
                _write_json(artifact, report)
                continue
            if existing_row is not None:
                raise ValueError("Existing nonterminal submission requires readback, not resubmission")
            audio = client.call("listening-audio", identifiers={"item_id": item_id})
            if audio.get("httpStatus") != 200 or not audio.get("audio", {}).get("mimeMatches"):
                raise RuntimeError("LISTENING_REFERENCE_AUDIO_MIME_OR_STRUCTURE_INVALID")
            row: dict[str, Any] = {"itemId": item_id, "itemIndex": item["itemIndex"], "attemptId": attempt_id,
                                   "scenario": "WRONG" if mode == "COMPREHENSION" and scenario == "PARTIAL" else scenario,
                                   "audio": audio["audio"], "preSubmitAnswerHidden": current.get("correctOptionKey") is None and current.get("sourceText") is None,
                                   "status": "STARTED"}
            report["items"].append(row)
            _write_json(artifact, report)
            row["durationValidation"] = validate_listening_audio(item, audio["audio"], difficulty)
            _write_json(artifact, report)
            for task in tasks:
                body(client.call("listening-answer", identifiers={"attempt_id": attempt_id, "task_type": task}, payload={
                    "answer": listening_answer(item, task, scenario), "assistanceUsage": [],
                    "idempotencyKey": f"qa-be-{mode}-{attempt_id}-{task}-answer",
                }))
            body(client.call("listening-submit", identifiers={"attempt_id": attempt_id}, allow_provider=True,
                             payload={"idempotencyKey": f"qa-be-{mode}-{attempt_id}-submit", "actualDurationMs": round((time.monotonic() - attempt_began) * 1000)}))
            for _ in range(60):
                session = body(client.call("listening-session", identifiers={"session_id": session_id}))
                current_attempt = next(value for value in session["attempts"] if value["attemptId"] == attempt_id)
                if evaluation_terminal(current_attempt, tasks):
                    row.update(status=current_attempt["status"], evaluation=current_attempt)
                    break
                time.sleep(5)
            else:
                raise TimeoutError("LISTENING_EVALUATION_POLL_BOUND")
            _write_json(artifact, report)
            print(json.dumps({"mode": mode, "itemIndex": item["itemIndex"], "status": row["status"]}), flush=True)
            if any(task["status"] == "EVALUATION_FAILED" for task in row["evaluation"]["tasks"]):
                report["status"] = "EVALUATION_FAILED_NO_RETRY"
                return report
        report["completion"] = body(client.call("listening-complete", identifiers={"session_id": session_id}, payload={"actualDurationMs": round((time.monotonic() - session_began) * 1000)}))
        report["result"] = body(client.call("listening-result", identifiers={"session_id": session_id}))
        report.update(status="COMPLETED", partial=False)
    except BaseException as error:
        report.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", errorType=type(error).__name__)
        raise
    finally:
        report["finishedAt"] = utc_now()
        _write_json(artifact, report)
    return report


def _audio(record: dict[str, Any], *, verify_bytes: bool = False) -> dict[str, Any]:
    if record.get("httpStatus") != 200 or not record.get("audio", {}).get("mimeMatches") or record["audio"].get("format") != "WAV":
        raise ValueError("HTTP_AUDIO_NOT_VALID_WAV")
    audio = record["audio"]
    if not verify_bytes:
        return audio
    raw = Path(audio["path"]).read_bytes()
    measured = audio_metadata(raw, audio["contentType"])
    if (measured.get("format") != "WAV" or not measured.get("mimeMatches")
            or measured["sha256"] != audio["sha256"]):
        raise ValueError("SPEAKING_DOWNLOADED_AUDIO_CHECKSUM_OR_FORMAT_MISMATCH")
    with wave.open(io.BytesIO(raw), "rb") as source:
        expected_bytes = source.getnframes() * source.getnchannels() * source.getsampwidth()
        if (source.getcomptype() != "NONE" or expected_bytes <= 0
                or len(source.readframes(source.getnframes())) != expected_bytes):
            raise ValueError("SPEAKING_DOWNLOADED_AUDIO_FRAMES_INVALID")
    if any(measured[key] != audio[key] for key in ("durationSeconds", "frames", "sampleRate", "channels", "bytes")):
        raise ValueError("SPEAKING_DOWNLOADED_AUDIO_METADATA_MISMATCH")
    return {**audio, "fullWaveformDecoded": True, "recordedSha256Verified": True}


def speaking_evidence_v2(evaluation: dict[str, Any], mode: str, *, metrics_required: bool) -> dict[str, Any]:
    """Observe actual BE fields; never synthesize a missing policy or axis pass."""
    from app.features.language_learning.speaking.evidence_policy import (
        SPEAKING_EVIDENCE_POLICY_VERSION,
        TRANSCRIPT_EVIDENCE_SOURCE,
    )
    from app.features.language_learning.speaking.policy import SPEAKING_METRIC_WEIGHTS

    if mode not in {"READ_ALOUD", "GUIDED", "FREE"}:
        raise ValueError("SPEAKING_EVIDENCE_MODE_INVALID")
    if (evaluation.get("evidencePolicyVersion") != SPEAKING_EVIDENCE_POLICY_VERSION
            or evaluation.get("evidenceSource") != TRANSCRIPT_EVIDENCE_SOURCE):
        raise ValueError("SPEAKING_EVIDENCE_POLICY_MISSING_OR_MISMATCHED")
    weights = {metric.value: weight for metric, weight in SPEAKING_METRIC_WEIGHTS.items()}
    axes = evaluation.get("evaluatedAxes")
    coverage = evaluation.get("evaluationCoverage")
    if (not isinstance(axes, list) or any(not isinstance(axis, str) or axis not in weights for axis in axes)
            or len(set(axes)) != len(axes) or {"PRONUNCIATION", "FLUENCY"}.intersection(axes)
            or (mode == "READ_ALOUD" and any(axis != "MEANING" for axis in axes))):
        raise ValueError("SPEAKING_EVALUATED_AXES_INVALID")
    expected_coverage = sum(weights[axis] for axis in axes)
    if (isinstance(coverage, bool) or not isinstance(coverage, (float, int)) or not math.isfinite(coverage)
            or abs(coverage - expected_coverage) > 1e-9):
        raise ValueError("SPEAKING_EVALUATION_COVERAGE_MISMATCH")
    status = evaluation.get("status")
    if status not in {"EVALUATED", "INSUFFICIENT_EVIDENCE"} or (status == "EVALUATED" and not axes):
        raise ValueError("SPEAKING_EVALUATION_NOT_SUPPORTED_BY_POSITIVE_AXIS")
    verified_not_evaluable: list[str] = []
    if metrics_required:
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, list):
            raise ValueError("SPEAKING_METRIC_ROWS_MISSING")
        by_axis = {metric["metricType"]: metric for metric in metrics}
        if len(by_axis) != len(metrics) or (status == "EVALUATED" and set(by_axis) != set(weights)):
            raise ValueError("SPEAKING_METRIC_ROW_COVERAGE_INVALID")
        actual_axes = {axis for axis, metric in by_axis.items() if metric["state"] == "EVALUATED"}
        if actual_axes != set(axes):
            raise ValueError("SPEAKING_METRIC_ROWS_DISAGREE_WITH_AXES")
        for axis in ("PRONUNCIATION", "FLUENCY"):
            metric = by_axis.get(axis)
            if metric is not None:
                if metric["state"] != "NOT_EVALUABLE" or metric.get("score") is not None:
                    raise ValueError("SPEAKING_TEXT_EVALUATOR_SCORED_UNSUPPORTED_ACOUSTICS")
                verified_not_evaluable.append(axis)
        practices = json.loads(evaluation.get("pronunciationPracticeJson") or "[]")
        if practices:
            raise ValueError("SPEAKING_TEXT_EVALUATOR_PRODUCED_PRONUNCIATION_PRACTICE")
    return {"evidencePolicyVersion": evaluation["evidencePolicyVersion"],
            "evidenceSource": evaluation["evidenceSource"], "evaluatedAxes": axes,
            "evaluationCoverage": coverage, "verifiedNotEvaluableAxes": verified_not_evaluable,
            "metricRowsExposed": metrics_required,
            "basis": "Observed BE result, not acoustic quality or human pronunciation validation"}


async def _synthesize_learner(client: Client, mode: str, turn: int, text: str) -> dict[str, Any]:
    # The sole admitted client is wrapped before the first possible paid call.
    from app.ai.provider_factory import create_speech_synthesis_provider
    from scripts.qa_campaign_audio_budget import BudgetedOpenAISpeechProvider
    from scripts.qa_campaign_budget import CampaignLedger
    campaign = client.campaign_directory
    path = client.output / f"speaking-{mode.lower()}-learner-{turn}.wav"
    if path.exists():
        raise ValueError("Synthetic learner audio cannot be overwritten or regenerated")
    ledger = CampaignLedger(campaign / "budget-ledger.json", campaign.name)
    upstream = create_speech_synthesis_provider()
    provider = BudgetedOpenAISpeechProvider(upstream, ledger, phase="BE_SYNTHETIC_LEARNER")
    try:
        result = await provider.synthesize_speech(text=text, voice="marin", language="ja", speed="NORMAL")
        path.write_bytes(result.audio_bytes)
        meta = audio_metadata(result.audio_bytes, result.content_type)
        if meta.get("format") != "WAV" or not meta["mimeMatches"]:
            raise ValueError("SYNTHETIC_TTS_AUDIO_NOT_VALID_WAV")
        return {**meta, "path": str(path), "text": text, "source": "Budgeted synthetic TTS, not microphone"}
    finally:
        await provider.shutdown()


def speaking(client: Client, mode: str, *, silence_rerecord_probe: bool = False, resume_existing: bool = False,
             verify_evidence_v2: bool = False) -> dict[str, Any]:
    if mode not in {"READ_ALOUD", "GUIDED", "FREE"}:
        raise ValueError("Unsupported source-defined Speaking mode")
    if silence_rerecord_probe and mode != "READ_ALOUD":
        raise ValueError("Normal source rerecord flow is READ_ALOUD only")
    artifact = client.output / f"speaking-{mode.lower()}-workflow.json"
    if artifact.exists() and not resume_existing:
        raise ValueError("Speaking workflow is one-shot; inspect outcome rather than repeat")
    report: dict[str, Any] = {"mode": mode, "status": "STARTED", "partial": True, "startedAt": utc_now(),
                              "turns": [], "problemEvaluations": [], "manualRetries": 0,
                              "humanMicrophoneOrPronunciationEvidence": False,
                              "conversationTurnsPredeclared": 6, "durationPadding": False}

    def download_audio(record: dict[str, Any]) -> dict[str, Any]:
        return _audio(record, verify_bytes=verify_evidence_v2)

    if resume_existing:
        previous = json.loads(artifact.read_text(encoding="utf-8"))
        guard_observation_failed = (previous["status"] == "FAILED" and previous.get("errorType") == "ValueError"
                                    and previous.get("resumedAt") and previous.get("priorProblemEvaluations")
                                    and all(row["status"] == "READY" for row in previous["turns"]))
        failed_conversation = (mode in {"GUIDED", "FREE"} and previous["status"] == "TURN_FAILED_NO_RETRY"
                               and previous["turns"][-1]["response"].get("failedStage") == "CONVERSATION"
                               and previous["turns"][-1]["response"].get("errorCode") == "INVALID_RESPONSE_SCHEMA")
        if not failed_conversation and ((previous["status"] != "EVALUATION_NOT_EVALUATED_NO_RETRY" and not guard_observation_failed) or mode != "READ_ALOUD"):
            raise ValueError("Only diagnosed failed READ_ALOUD evaluation checkpoint may resume")
        _write_json(artifact.with_name(artifact.stem + f"-checkpoint-{time.time_ns()}.json"), previous)
        preserve_prior_outcome(previous)
        report = {**previous, "status": "RESUMED_EXISTING_SESSION_ONLY", "resumedAt": utc_now(),
                  "priorProblemEvaluations": previous.get("problemEvaluations") or previous.get("priorProblemEvaluations", []), "problemEvaluations": [],
                  "authorizedConversationRetryReadback": bool(failed_conversation)}
    report["aiRuntime"] = client.ai_runtime_identity()
    report["evidenceV2VerificationRequired"] = verify_evidence_v2
    _write_json(artifact, report)
    try:
        if resume_existing:
            session_id = report["sessionId"]
            opened = report["opening"]
            detail = body(client.call("speaking-session", identifiers={"session_id": session_id}))
            if detail["session"]["status"] != "IN_PROGRESS":
                raise ValueError("Existing session no longer IN_PROGRESS; do not reopen")
            # Retry admission belongs to a separate explicit one-shot API action.
            # Resume itself only reads its result; it never resubmits evaluation.
            for prior in report["priorProblemEvaluations"]:
                for _ in range(60):
                    evaluations = body(client.call("speaking-problem-evaluations", identifiers={"session_id": session_id}))
                    current = next(item for item in evaluations if item["problemIndex"] == prior["problemIndex"])
                    if current["status"] not in {"PENDING", "EVALUATING"}:
                        break
                    time.sleep(5)
                else:
                    raise TimeoutError("REPAIRED_EVALUATION_READBACK_BOUND")
                report["problemEvaluations"].append(current)
                if current["status"] not in {"EVALUATED", "INSUFFICIENT_EVIDENCE"}:
                    report["status"] = "EVALUATION_NOT_EVALUATED_NO_RETRY"
                    return report
                if verify_evidence_v2:
                    report.setdefault("problemEvidenceChecks", {})[str(current["problemIndex"])] = speaking_evidence_v2(current, mode, metrics_required=False)
            report["manualRetries"] = sum(item.get("manualRetryCount", 0) for item in report["problemEvaluations"])
            reference = report["openingAudio"]
        else:
            active = body(client.call("speaking-active"))
            if active is not None:
                raise ValueError("Existing active Speaking session must not be displaced")
            payload = json.loads((client.output / "fixtures" / f"speaking-{mode.lower()}.json").read_text(encoding="utf-8"))
            opened = body(client.call("speaking-create", payload=payload, allow_provider=True, timeout=300))
            session_id = opened["id"]
            report.update(sessionId=session_id, opening=opened)
            _write_json(artifact, report)
            reference = download_audio(client.call("speaking-opening-audio", identifiers={"session_id": session_id}))
            report["openingAudio"] = reference
        guide = opened.get("openingPromptGuide") or {}
        scenarios = ["REFERENCE"] * 10 if mode == "READ_ALOUD" else ["RIGHT_INTENT", "PARTIAL", "UNRELATED", "RIGHT_INTENT", "RIGHT_INTENT", "PREDECLARED_FOLLOW_UP"]
        total_audio = 0.0
        for turn_index, scenario in enumerate(scenarios, start=1):
            problem = (turn_index + 1) // 2 if mode == "READ_ALOUD" else None
            attempt = (turn_index - 1) % 2 + 1 if problem else None
            existing = next((row for row in report["turns"] if row["turnIndex"] == turn_index), None)
            if existing is not None:
                observed = body(client.call("speaking-turn-status", identifiers={"session_id": session_id, "turn_id": existing["turnId"]}))
                if report.get("authorizedConversationRetryReadback") and existing["response"]["status"] == "PARTIAL_FAILURE":
                    old = existing["response"]
                    stable = ("id", "turnIndex", "recordingRevision", "transcript", "sttConfidence", "durationSeconds")
                    if (observed["status"] != "READY" or observed.get("manualRetryCount") != old.get("manualRetryCount", 0) + 1
                            or any(observed.get(key) != old.get(key) for key in stable)):
                        report["status"] = "AUTHORIZED_RETRY_DID_NOT_PRESERVE_STT_NO_FURTHER_RETRY"
                        return report
                    existing.update(failureBeforeAuthorizedRetry=old, response=observed, status="READY")
                    report["manualRetries"] = observed["manualRetryCount"]
                    existing["userAudioReadback"] = download_audio(client.call("speaking-user-audio", identifiers={"session_id": session_id, "turn_id": existing["turnId"]}))
                    if existing["userAudioReadback"]["sha256"] != existing["learnerAudio"]["sha256"]:
                        raise ValueError("Authorized retry changed original user audio")
                    if observed.get("assistantAudioUrl"):
                        existing["assistantAudio"] = download_audio(client.call("speaking-assistant-audio", identifiers={"session_id": session_id, "turn_id": existing["turnId"]}))
                if observed["status"] != "READY" or not accepted_turn_unchanged(existing["response"], observed):
                    raise ValueError("Existing accepted turn changed; no replay or repair admitted")
                if observed.get("completedAt") != existing["response"].get("completedAt"):
                    existing["persistenceTimestampPrecision"] = {
                        "processResponse": existing["response"].get("completedAt"), "persistedReadback": observed.get("completedAt"),
                        "maximumDifferenceMicroseconds": 1, "allOtherFieldsExact": True,
                    }
                total_audio += observed["durationSeconds"]
                if existing.get("assistantAudio"):
                    reference = existing["assistantAudio"]
                if observed.get("promptGuide"):
                    guide = observed["promptGuide"]
                existing["resumedReadbackOnly"] = True
                continue
            row: dict[str, Any] = {"turnIndex": turn_index, "problemIndex": problem, "attemptIndex": attempt,
                                   "scenario": scenario, "status": "PREPARING_SYNTHETIC_AUDIO"}
            report["turns"].append(row)
            _write_json(artifact, report)
            if mode == "READ_ALOUD":
                learner = {**reference, "source": "Same BE reference WAV reuploaded, synthetic read-aloud oracle"}
            else:
                from scripts.qa_campaign_speaking import _learner_text
                text = _learner_text(scenario, mode, "", guide.get("providedFacts") or [])
                row["syntheticLearnerText"] = text
                _write_json(artifact, report)
                learner = asyncio.run(_synthesize_learner(client, mode, turn_index, text))
            row.update(learnerAudio=learner, status="STARTED")
            _write_json(artifact, report)
            grant = body(client.call("speaking-upload-grant", identifiers={"session_id": session_id}, payload={
                "turnIndex": turn_index, "problemIndex": problem, "attemptIndex": attempt,
                "idempotencyKey": f"qa-be-{mode}-{session_id}-{turn_index}-upload",
            }))
            row["turnId"] = grant["turnId"]
            _write_json(artifact, report)
            rerecord = False
            if silence_rerecord_probe and turn_index == 1:
                silence_path = client.output / "speaking-read-aloud-silence-probe.wav"
                with wave.open(str(silence_path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes(b"\x00\x00" * 48000)
                silent = client.call("speaking-turn", identifiers={"session_id": session_id}, allow_provider=True,
                                    audio_path=silence_path, timeout=300, payload={
                    "turnId": grant["turnId"], "uploadToken": grant["uploadToken"],
                    "durationSeconds": 3.0, "assistanceUsage": [], "rerecord": False,
                })
                observed = body(client.call("speaking-turn-status", identifiers={"session_id": session_id, "turn_id": grant["turnId"]}))
                probe: dict[str, Any] = {"httpStatus": silent.get("httpStatus"), "response": observed,
                                         "input": audio_metadata(silence_path.read_bytes(), "audio/wav")}
                row["silenceProbe"] = probe
                _write_json(artifact, report)
                if observed["status"] not in {"PARTIAL_FAILURE", "FAILED"} or observed.get("errorCode") != "SILENCE_DETECTED":
                    report["status"] = "SILENCE_REJECTION_NOT_AS_EXPECTED"
                    return report
                probe["storedAudioBeforeRerecord"] = download_audio(client.call("speaking-user-audio", identifiers={"session_id": session_id, "turn_id": grant["turnId"]}))
                grant = body(client.call("speaking-rerecord-grant", identifiers={"session_id": session_id, "turn_id": grant["turnId"]}))
                rerecord = True
            result = body(client.call("speaking-turn", identifiers={"session_id": session_id}, allow_provider=True,
                                      audio_path=Path(learner["path"]), timeout=300, payload={
                "turnId": grant["turnId"], "uploadToken": grant["uploadToken"],
                "durationSeconds": learner["durationSeconds"], "assistanceUsage": [], "rerecord": rerecord,
            }))
            row.update(status=result["status"], response=result)
            _write_json(artifact, report)
            if result["status"] != "READY":
                report["status"] = "TURN_FAILED_NO_RETRY"
                return report
            row["userAudioReadback"] = download_audio(client.call("speaking-user-audio", identifiers={"session_id": session_id, "turn_id": grant["turnId"]}))
            if verify_evidence_v2:
                if row["userAudioReadback"]["sha256"] != learner["sha256"]:
                    raise ValueError("SPEAKING_UPLOADED_AUDIO_READBACK_MISMATCH")
                row["uploadedAudioSha256MatchesReadback"] = True
            total_audio += result["durationSeconds"]
            report["actualAcceptedAudioSeconds"] = total_audio
            if result.get("assistantAudioUrl"):
                reference = download_audio(client.call("speaking-assistant-audio", identifiers={"session_id": session_id, "turn_id": grant["turnId"]}))
                row["assistantAudio"] = reference
            if result.get("promptGuide"):
                guide = result["promptGuide"]
            if problem and attempt == 2:
                body(client.call("speaking-problem-evaluate", identifiers={"session_id": session_id, "problem_index": problem}, allow_provider=True))
                for _ in range(60):
                    evaluations = body(client.call("speaking-problem-evaluations", identifiers={"session_id": session_id}))
                    evaluated = next((item for item in evaluations if item["problemIndex"] == problem), None)
                    if evaluated and evaluated["status"] in {"EVALUATED", "INSUFFICIENT_EVIDENCE", "FAILED"}:
                        report["problemEvaluations"].append(evaluated)
                        break
                    time.sleep(5)
                else:
                    raise TimeoutError("READ_ALOUD_EVALUATION_POLL_BOUND")
                if evaluated["status"] == "FAILED":
                    report["status"] = "EVALUATION_NOT_EVALUATED_NO_RETRY"
                    return report
                if verify_evidence_v2:
                    report.setdefault("problemEvidenceChecks", {})[str(problem)] = speaking_evidence_v2(evaluated, mode, metrics_required=False)
            _write_json(artifact, report)
            print(json.dumps({"mode": mode, "turnIndex": turn_index, "status": row["status"]}), flush=True)
        detail = body(client.call("speaking-session", identifiers={"session_id": session_id}))
        report["preCompletionDetail"] = detail
        if mode != "READ_ALOUD":
            report["requiredRealAudioDurationMet"] = total_audio >= 60
            report["completion"] = body(client.call("speaking-complete", identifiers={"session_id": session_id}, payload={"skipEvaluation": False}, allow_provider=True))
        for _ in range(60):
            detail = body(client.call("speaking-session", identifiers={"session_id": session_id}))
            report["finalDetail"] = detail
            if detail["session"]["evaluationStatus"] in {"EVALUATED", "INSUFFICIENT_EVIDENCE", "FAILED"}:
                report["evaluation"] = body(client.call("speaking-evaluation", identifiers={"session_id": session_id}))
                if verify_evidence_v2 and detail["session"]["evaluationStatus"] in {"EVALUATED", "INSUFFICIENT_EVIDENCE"}:
                    report["evidenceV2Check"] = speaking_evidence_v2(report["evaluation"], mode, metrics_required=True)
                non_evaluable = any(item["status"] == "INSUFFICIENT_EVIDENCE" for item in report["problemEvaluations"])
                outcome = "COMPLETED_WITH_NON_EVALUABLE_PROBLEM" if non_evaluable else "COMPLETED"
                report.update(status=outcome if detail["session"]["evaluationStatus"] == "EVALUATED" else "EVALUATION_NOT_EVALUATED_NO_RETRY", partial=False)
                if mode != "READ_ALOUD" and not report["requiredRealAudioDurationMet"]:
                    report["status"] = "INSUFFICIENT_REAL_AUDIO_DURATION_NO_PADDING"
                return report
            time.sleep(5)
        raise TimeoutError("SPEAKING_EVALUATION_POLL_BOUND")
    except BaseException as error:
        report.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", errorType=type(error).__name__)
        raise
    finally:
        report["finishedAt"] = utc_now()
        _write_json(artifact, report)


def reconcile_listening_observations(client: Client) -> dict[str, Any]:
    """Artifact-only reconciliation; no HTTP, SQL, model call, or result relabeling."""
    summaries = []
    for artifact in client.output.glob("listening-*-workflow.json"):
        report = json.loads(artifact.read_text(encoding="utf-8"))
        if report["status"] == "COMPLETED" and report.get("errorType"):
            checkpoints = [json.loads(p.read_text(encoding="utf-8")) for p in client.output.glob(artifact.stem + "-checkpoint-*.json")]
            original = next((row for row in checkpoints if row.get("status") == "FAILED" and row.get("errorType") == report["errorType"]), None)
            if original is None:
                raise ValueError("Historical failure must have preserved checkpoint proof")
            report.setdefault("priorObservations", []).append({key: original[key] for key in ("status", "errorType", "finishedAt")})
            report.pop("errorType")
            _write_json(artifact, report)
        counts: dict[str, int] = {}
        for row in report["items"]:
            for task in row.get("evaluation", {}).get("tasks", []):
                if task["status"] != "NOT_SELECTED":
                    key = task["taskType"] + ":" + task["status"]
                    counts[key] = counts.get(key, 0) + 1
        summaries.append({"mode": report["mode"], "status": report["status"], "items": len(report["items"]), "taskCounts": counts})
    return {"modes": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["listening", "speaking", "reconcile-listening"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--run-label", help="Separate owned QA evidence run (letters, digits, hyphens only)")
    parser.add_argument("--mode", choices=["DICTATION", "COMPREHENSION", "SUMMARY", "READ_ALOUD", "GUIDED", "FREE"])
    parser.add_argument("--difficulty", choices=list(_LISTENING_DURATION_BOUNDS), default="MY_LEVEL")
    parser.add_argument("--case-id", help="Unique Listening case label; defaults to mode-difficulty")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--silence-rerecord-probe", action="store_true")
    args = parser.parse_args()
    if args.run_label is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_label):
        raise ValueError("Invalid QA run label")
    output_directory = args.directory / "be-api-runs" / args.run_label if args.run_label else None
    if args.action == "reconcile-listening":
        print(json.dumps(reconcile_listening_observations(Client(args.directory, output_directory=output_directory))))
        return
    if not args.live:
        raise SystemExit("Explicit live opt-in and root serial GO required")
    if args.action == "listening":
        result = listening(Client(args.directory, output_directory=output_directory), args.mode, difficulty=args.difficulty, case_id=args.case_id,
                           resume_existing=args.resume_existing)
    else:
        result = speaking(Client(args.directory, output_directory=output_directory), args.mode, silence_rerecord_probe=args.silence_rerecord_probe,
                          resume_existing=args.resume_existing, verify_evidence_v2=True)
    print(json.dumps({"mode": result["mode"], "status": result["status"], "itemCount": len(result.get("items", result.get("turns", [])))}))


if __name__ == "__main__":
    main()
