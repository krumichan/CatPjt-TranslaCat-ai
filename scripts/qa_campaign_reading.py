"""Injected-client, one-shot Reading QA; never creates auth or provider clients.

The caller prepares isolated account/date/profile/keyword fixtures explicitly.
Public Reading start has no topic/band override. Persisted request identity is
checked, not relabelled. DB answers are a synthetic workflow oracle, not proof
of Reading content quality or human comprehension. No generation retry exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Protocol

from scripts.qa_campaign_be_api import utc_now
from scripts.qa_campaign_integration import _write_json


@dataclass(frozen=True)
class ReadingCase:
    case_id: str
    mode: str
    keyword: str
    band: int


NEW_RUN_CASES = (
    ReadingCase("comprehension-park-b3", "COMPREHENSION", "公園", 3),
    ReadingCase("structure-work-b3", "STRUCTURE", "仕事", 3),
    ReadingCase("context-inference-travel-b3", "CONTEXT_INFERENCE", "旅行", 3),
    ReadingCase("comprehension-shopping-b1", "COMPREHENSION", "買い物", 1),
    ReadingCase("structure-environment-b5", "STRUCTURE", "環境", 5),
)
PASSAGE_ORDERS = {"p1": (1, 2, 3), "p2": (4, 5)}


class ReadingClient(Protocol):
    output: Path

    def call(self, action: str, *, identifiers: dict[str, Any] | None = None,
             payload: dict[str, Any] | None = None, allow_provider: bool = False,
             timeout: float = 40) -> dict[str, Any]: ...

    def practice_snapshot(self, set_id: int) -> dict[str, Any]: ...


def _body(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("httpStatus") != 200:
        raise ValueError("READING_QA_HTTP_STATUS_" + str(record.get("httpStatus")))
    result = record["response"]["body"]
    if not isinstance(result, dict):
        raise ValueError("READING_QA_RESPONSE_OBJECT_REQUIRED")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _snapshot(client: ReadingClient, set_id: int, case: ReadingCase, learning_date: str) -> dict[str, Any]:
    summary = client.practice_snapshot(set_id)
    snapshot = json.loads(Path(summary["privateArtifact"]).read_text(encoding="utf-8"))
    request = snapshot["generationRequestSnapshot"]
    if (snapshot["practiceSetId"] != set_id or request["domain"] != "READING"
            or request["mode"] != case.mode or request["questionCount"] != 5
            or request["complexityBand"] != case.band
            # Internal AI DTO uses generationDate; public PracticeSet uses learningDate.
            or request["generationDate"] != learning_date
            or case.keyword not in request.get("selectedKeywords", [])):
        raise ValueError("READING_QA_PERSISTED_REQUEST_FIXTURE_MISMATCH")
    return {"summary": summary, "snapshot": snapshot}


def _observe(view: dict[str, Any], case: ReadingCase, set_id: int,
             question_identity: dict[int, str], passage_identity: dict[str, str]) -> dict[str, Any]:
    if (view["practiceSetId"] != set_id or view["domain"] != "READING"
            or view["mode"] != case.mode or view["questionCount"] != 5
            or view["complexityBand"] != case.band):
        raise ValueError("READING_QA_SET_IDENTITY_MISMATCH")
    questions = view["questions"]
    orders = [question["order"] for question in questions]
    if orders != list(range(1, len(questions) + 1)) or len(questions) > 5:
        raise ValueError("READING_QA_PROGRESSIVE_PREFIX_INVALID")
    if any(order not in orders for order in question_identity):
        raise ValueError("READING_QA_ACCEPTED_QUESTION_DISAPPEARED")
    exposure: dict[str, Any] = {}
    for question in questions:
        order = question["order"]
        passage_id = "p1" if order <= 3 else "p2"
        if question["passageId"] != passage_id or not question["passageText"]:
            raise ValueError("READING_QA_PASSAGE_BINDING_INVALID")
        passage_hash = _digest(question["passageText"])
        if passage_id in passage_identity and passage_identity[passage_id] != passage_hash:
            raise ValueError("READING_QA_PASSAGE_CHANGED")
        passage_identity[passage_id] = passage_hash
        immutable = {key: question[key] for key in (
            "questionId", "order", "questionType", "difficulty", "complexityBand",
            "passageId", "passageText", "prompt", "options", "skillTag",
        )}
        fingerprint = _digest(immutable)
        if order in question_identity and question_identity[order] != fingerprint:
            raise ValueError("READING_QA_ACCEPTED_QUESTION_CHANGED")
        question_identity[order] = fingerprint
    for passage_id, expected in PASSAGE_ORDERS.items():
        related = [question for question in questions if question["passageId"] == passage_id]
        completed = [question["order"] for question in related if question["answered"]] == list(expected)
        candidates = []
        for question in related:
            values = question.get("vocabularyCandidates", [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError("READING_QA_OPTIONAL_EXPRESSION_PUBLIC_SHAPE_INVALID")
            if values and not completed:
                raise ValueError("READING_QA_EXPRESSION_RELEASED_BEFORE_PASSAGE_ANSWERS")
            if any(not value or value not in question["passageText"] for value in values):
                raise ValueError("READING_QA_EXPRESSION_NOT_EXACT_PASSAGE_SURFACE")
            candidates.extend(values)
        exposure[passage_id] = {"allPlannedQuestionsAnswered": completed,
                                "expressions": list(dict.fromkeys(candidates)),
                                "optionalAbsenceAllowed": True}
    return {"generationStatus": view["generationStatus"], "setStatus": view["status"],
            "generatedQuestionCount": len(questions), "answeredCount": view["answeredCount"],
            "passageHashes": dict(passage_identity), "passageExpressions": exposure}


def reading(client: ReadingClient, case: ReadingCase, *, max_polls: int = 60,
            max_seconds: float = 600, interval_seconds: float = 10,
            expected_learning_date: str | None = None,
            clock: Callable[[], float] = time.monotonic,
            sleep: Callable[[float], None] = time.sleep,
            _resume_set_id: int | None = None) -> dict[str, Any]:
    """One start, bounded GET polling, one answer per row, no automatic retries."""
    if case not in NEW_RUN_CASES:
        raise ValueError("Only predeclared NEW_RUN cases are admitted")
    if not 1 <= max_polls <= 60 or not 1 <= max_seconds <= 600 or not 5 <= interval_seconds <= 30:
        raise ValueError("Bounded read-only polling parameters required")
    if expected_learning_date is not None and date.fromisoformat(expected_learning_date).isoformat() != expected_learning_date:
        raise ValueError("Expected fixture date must use YYYY-MM-DD")
    artifact = client.output / f"reading-{case.case_id}-workflow.json"
    if artifact.exists() and _resume_set_id is None:
        raise ValueError("Reading case already started; inspect artifact rather than rerun")
    report: dict[str, Any] = {
        "campaignPhase": "NEW_RUN", "caseId": case.case_id, "mode": case.mode,
        "requestedKeyword": case.keyword, "requestedBand": case.band,
        "startedAt": utc_now(), "status": "STARTED", "partial": True,
        "syntheticReferenceAnswerOracle": True, "humanComprehensionEvidence": False,
        "generationRetries": 0, "questionCountPolicy": 5, "observations": [], "answers": [],
    }
    if _resume_set_id is not None:
        if type(_resume_set_id) is not int or _resume_set_id <= 0 or not artifact.exists():
            raise ValueError("Existing positive set ID and preserved failed artifact required")
        previous = json.loads(artifact.read_text(encoding="utf-8"))
        if (previous.get("status") != "FAILED" or previous.get("errorType") != "KeyError"
                or previous.get("practiceSetId") != _resume_set_id or previous.get("caseId") != case.case_id
                or previous.get("answers") or previous.get("observations") or previous.get("resumedAt")):
            raise ValueError("Only the diagnosed pre-observation date-contract failure may resume once")
        checkpoint = artifact.with_name(artifact.stem + f"-failed-checkpoint-{time.time_ns()}.json")
        _write_json(checkpoint, previous)
        report = {**previous, "status": "RESUMED_EXISTING_SET_ONLY", "resumedAt": utc_now(),
                  "originalFailureCheckpoint": str(checkpoint),
                  "priorObservations": [{"status": previous["status"], "errorType": previous["errorType"],
                                         "finishedAt": previous.get("finishedAt")}],
                  "resumeBoundary": "Same existing set; no fixture/start/generation retry; require zero submitted answers"}
        report.pop("errorType", None)
        expected_learning_date = expected_learning_date or previous.get("expectedFixtureLearningDate")
    identities: dict[int, str] = {}
    passages: dict[str, str] = {}
    _write_json(artifact, report)
    try:
        if _resume_set_id is None:
            view = _body(client.call("reading-start", identifiers={"mode": case.mode}, allow_provider=True))
        else:
            view = _body(client.call("reading-status", identifiers={"set_id": _resume_set_id}))
            if view["practiceSetId"] != _resume_set_id or view["answeredCount"] != 0 or any(
                question["answered"] for question in view["questions"]
            ):
                raise ValueError("READING_QA_RESUME_REQUIRES_SAME_UNANSWERED_SET")
        set_id = view["practiceSetId"]
        report["practiceSetId"] = set_id
        learning_date = view["learningDate"]
        report["learningDate"] = learning_date
        report["expectedFixtureLearningDate"] = expected_learning_date
        if expected_learning_date is not None and learning_date != expected_learning_date:
            raise ValueError("READING_QA_LEARNING_DATE_FIXTURE_MISMATCH")
        first = _snapshot(client, set_id, case, learning_date)
        report["initialDatabaseSnapshot"] = first["summary"]
        request_hash = _digest(first["snapshot"]["generationRequestSnapshot"])
        report["persistedRequestSha256"] = request_hash
        begin = clock()
        for index in range(max_polls + 1):
            report["observations"].append(_observe(view, case, set_id, identities, passages))
            _write_json(artifact, report)
            if view["generationStatus"] in {"READY", "FAILED", "PARTIAL"}:
                break
            remaining = max_seconds - (clock() - begin)
            if index == max_polls or remaining < interval_seconds + 1:
                raise TimeoutError("READING_QA_GENERATION_POLL_BOUND")
            sleep(interval_seconds)
            remaining = max_seconds - (clock() - begin)
            if remaining < 1:
                raise TimeoutError("READING_QA_GENERATION_POLL_BOUND")
            view = _body(client.call("reading-status", identifiers={"set_id": set_id}, timeout=min(40, remaining)))
        if view["generationStatus"] != "READY":
            report.update(status="GENERATION_TERMINAL_NO_RETRY", generationFailureMessage=view.get("generationFailureMessage"))
            report["terminalDatabaseSnapshot"] = _snapshot(client, set_id, case, learning_date)["summary"]
            return report
        ready = _snapshot(client, set_id, case, learning_date)
        stored = ready["snapshot"]["persistedQuestions"]
        if (len(view["questions"]) != 5 or [question["order"] for question in stored] != list(range(1, 6))
                or ready["snapshot"]["generationStatus"] != "READY"
                or any(question["answered"] for question in view["questions"])):
            raise ValueError("READING_QA_NEW_READY_FIVE_UNANSWERED_ROWS_REQUIRED")
        latest = view
        for question, oracle in zip(view["questions"], stored, strict=True):
            if question["questionId"] != oracle["questionId"]:
                raise ValueError("READING_QA_DATABASE_QUESTION_ID_MISMATCH")
            result = _body(client.call("reading-answer", identifiers={"question_id": question["questionId"]},
                                       payload={"answer": oracle["correctAnswer"]}))
            report["answers"].append({"order": question["order"], "questionId": question["questionId"], "result": result})
            _write_json(artifact, report)
            latest = _body(client.call("reading-status", identifiers={"set_id": set_id}))
            report["observations"].append(_observe(latest, case, set_id, identities, passages))
            _write_json(artifact, report)
        final = _snapshot(client, set_id, case, learning_date)
        report["finalDatabaseSnapshot"] = final["summary"]
        if (final["snapshot"]["persistedQuestions"] != stored
                or _digest(final["snapshot"]["generationRequestSnapshot"]) != request_hash
                or final["snapshot"]["generationStatus"] != "READY"
                or latest["answeredCount"] != 5 or latest["status"] != "COMPLETED"):
            raise ValueError("READING_QA_PERSISTENCE_OR_COMPLETION_INVARIANT_FAILED")
        report.update(status="COMPLETED", partial=False, officialScore=latest["officialScore"],
                      metrics=latest.get("metrics", []), stablePassageCount=len(passages), persistedQuestionCount=5)
    except BaseException as error:
        report.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", errorType=type(error).__name__)
        raise
    finally:
        report["finishedAt"] = utc_now()
        _write_json(artifact, report)
    return report


def resume_reading(client: ReadingClient, case: ReadingCase, set_id: int, *,
                   expected_learning_date: str | None = None, max_polls: int = 60,
                   max_seconds: float = 600, interval_seconds: float = 10,
                   clock: Callable[[], float] = time.monotonic,
                   sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Explicit one-time continuation after the diagnosed QA-only date KeyError.

Never calls Reading start or fixture preparation. Generation stays on the
existing BE worker; only still-unanswered rows may receive their first answer.
"""
    return reading(client, case, expected_learning_date=expected_learning_date,
                   max_polls=max_polls, max_seconds=max_seconds, interval_seconds=interval_seconds,
                   clock=clock, sleep=sleep, _resume_set_id=set_id)
