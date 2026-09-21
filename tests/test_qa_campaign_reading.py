"""Offline API-shape tests only; fake outputs are not Japanese quality evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import qa_campaign_reading
from scripts.qa_campaign_reading import NEW_RUN_CASES, ReadingCase, reading, resume_reading


class FakeReadingClient:
    def __init__(self, path: Path, case: ReadingCase):
        self.output = path
        self.case = case
        self.calls: list[dict[str, Any]] = []
        self.answers: dict[int, list[str]] = {}
        self.progress = 1
        self.status = "GENERATING"
        self.early_expression = False
        self.changed_passage = False
        self.request_band = case.band
        self.snapshot_count = 0
        self.failure: BaseException | None = None
        self.stall = False

    def _questions(self) -> list[dict[str, Any]]:
        result = []
        for order in range(1, self.progress + 1):
            passage = "p1" if order <= 3 else "p2"
            expected = (1, 2, 3) if passage == "p1" else (4, 5)
            visible = all(value in self.answers for value in expected)
            result.append({
                "questionId": 100 + order, "order": order,
                "questionType": "ORDERING" if self.case.mode == "STRUCTURE" and order == 5 else "SINGLE_CHOICE",
                "difficulty": "CURRENT", "complexityBand": self.case.band, "passageId": passage,
                "passageText": ("変更した" if self.changed_passage and self.progress > 1 else "") + f"{passage}の計画を確認する。\n\n計画を共有する。",
                "prompt": "内容に合うものを選びなさい。", "options": [{"key": "A", "text": "一"}, {"key": "B", "text": "二"}],
                "skillTag": "DETAIL", "answered": order in self.answers,
                "vocabularyCandidates": ["計画"] if visible or self.early_expression else [],
            })
        return result

    def _view(self) -> dict[str, Any]:
        return {"practiceSetId": 91, "domain": "READING", "mode": self.case.mode,
                "learningDate": "2026-09-19",
                "complexityBand": self.case.band, "questionCount": 5, "generationStatus": self.status,
                "status": "COMPLETED" if len(self.answers) == 5 else "ACTIVE",
                "answeredCount": len(self.answers), "questions": self._questions(),
                "officialScore": 100.0 if len(self.answers) == 5 else None, "metrics": [],
                "generationFailureMessage": None}

    def call(self, action: str, *, identifiers: dict[str, Any] | None = None,
             payload: dict[str, Any] | None = None, allow_provider: bool = False,
             timeout: float = 40) -> dict[str, Any]:
        self.calls.append({"action": action, "identifiers": identifiers, "payload": payload,
                           "allowProvider": allow_provider, "timeout": timeout})
        if self.failure is not None and action == "reading-status":
            raise self.failure
        if action == "reading-status" and not self.stall:
            self.progress = min(5, self.progress + 2)
            self.status = "READY" if self.progress == 5 else self.status
        if action == "reading-answer":
            assert identifiers is not None and payload is not None
            order = identifiers["question_id"] - 100
            assert order not in self.answers
            self.answers[order] = payload["answer"]
            result = {"questionId": 100 + order, "attemptNo": 1, "correct": True,
                      "official": True, "setCompleted": len(self.answers) == 5,
                      "officialScore": 100.0 if len(self.answers) == 5 else None}
        else:
            result = self._view()
        return {"httpStatus": 200, "response": {"body": copy.deepcopy(result)}}

    def practice_snapshot(self, set_id: int) -> dict[str, Any]:
        self.snapshot_count += 1
        path = self.output / f"snapshot-{self.snapshot_count}.json"
        path.write_text(json.dumps({
            "practiceSetId": set_id, "generationStatus": self.status,
            "generationRequestSnapshot": {"domain": "READING", "mode": self.case.mode,
                "generationDate": "2026-09-19",
                "questionCount": 5, "complexityBand": self.request_band,
                "selectedKeywords": [self.case.keyword]},
            "persistedQuestions": [{"questionId": 100 + order, "order": order,
                "correctAnswer": ["B", "A"] if self.case.mode == "STRUCTURE" and order == 5 else ["A"],
                "contentSha256": f"unchanged-{order}"} for order in range(1, self.progress + 1)],
        }), encoding="utf-8")
        return {"privateArtifact": str(path), "persistedQuestionCount": self.progress}


def _run(client: FakeReadingClient, **kwargs: Any) -> dict[str, Any]:
    elapsed = [0.0]

    def advance(seconds: float) -> None:
        elapsed[0] += seconds

    return reading(client, client.case, sleep=advance, clock=lambda: elapsed[0], **kwargs)


@pytest.mark.parametrize("case", NEW_RUN_CASES)
def test_reading_progressive_five_rows_answers_and_passage_expression_boundary(tmp_path: Path, case: ReadingCase):
    client = FakeReadingClient(tmp_path, case)
    report = _run(client)
    assert report["status"] == "COMPLETED" and report["partial"] is False
    assert report["persistedQuestionCount"] == 5 and report["stablePassageCount"] == 2
    assert report["generationRetries"] == 0
    assert [row["generatedQuestionCount"] for row in report["observations"][:3]] == [1, 3, 5]
    after_answers = report["observations"][3:]
    assert [row["passageExpressions"]["p1"]["expressions"] for row in after_answers] == [[], [], ["計画"], ["計画"], ["計画"]]
    assert [row["passageExpressions"]["p2"]["expressions"] for row in after_answers] == [[], [], [], [], ["計画"]]
    assert [call["action"] for call in client.calls].count("reading-start") == 1
    assert sum(call["allowProvider"] for call in client.calls) == 1
    assert [call["identifiers"]["question_id"] for call in client.calls if call["action"] == "reading-answer"] == [101, 102, 103, 104, 105]
    assert client.answers[5] == (["B", "A"] if case.mode == "STRUCTURE" else ["A"])
    assert report["humanComprehensionEvidence"] is False
    with pytest.raises(ValueError, match="already started"):
        _run(client)


@pytest.mark.parametrize("fault,match", [
    ("early_expression", "RELEASED_BEFORE"), ("changed_passage", "PASSAGE_CHANGED"),
])
def test_reading_observer_rejects_premature_expressions_and_passage_mutation(tmp_path: Path, fault: str, match: str):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[0])
    setattr(client, fault, True)
    with pytest.raises(ValueError, match=match):
        _run(client)
    assert not client.answers
    artifact = json.loads(next(tmp_path.glob("reading-*-workflow.json")).read_text(encoding="utf-8"))
    assert artifact["status"] == "FAILED" and artifact["partial"] is True


def test_reading_fixture_mismatch_does_not_relabel_band_or_start_another_generation(tmp_path: Path):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[0])
    client.request_band = 5
    with pytest.raises(ValueError, match="REQUEST_FIXTURE_MISMATCH"):
        _run(client)
    assert len(client.calls) == 1
    assert client.request_band == 5


def test_reading_records_real_date_and_rejects_mismatched_fixture_without_relabelling(tmp_path: Path):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[0])
    with pytest.raises(ValueError, match="LEARNING_DATE_FIXTURE_MISMATCH"):
        _run(client, expected_learning_date="2026-09-18")
    artifact = json.loads(next(tmp_path.glob("reading-*-workflow.json")).read_text(encoding="utf-8"))
    assert artifact["learningDate"] == "2026-09-19"
    assert artifact["expectedFixtureLearningDate"] == "2026-09-18"
    assert len(client.calls) == 1 and not client.answers


@pytest.mark.parametrize("error", [KeyboardInterrupt(), TimeoutError("bounded network")])
def test_reading_flushes_partial_observations_on_interruption(tmp_path: Path, error: BaseException):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[0])
    client.failure = error
    with pytest.raises(type(error)):
        _run(client)
    artifact = json.loads(next(tmp_path.glob("reading-*-workflow.json")).read_text(encoding="utf-8"))
    assert artifact["errorType"] == type(error).__name__
    assert artifact["observations"][0]["generatedQuestionCount"] == 1
    assert artifact["partial"] is True and not client.answers


def test_reading_bounded_poll_never_retries_generation(tmp_path: Path):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[0])
    client.stall = True
    with pytest.raises(TimeoutError, match="POLL_BOUND"):
        _run(client, max_polls=2, max_seconds=60)
    assert [call["action"] for call in client.calls] == ["reading-start", "reading-status", "reading-status"]


def test_reading_terminal_partial_is_preserved_not_resubmitted(tmp_path: Path):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[0])
    client.status = "PARTIAL"
    report = _run(client)
    assert report["status"] == "GENERATION_TERMINAL_NO_RETRY" and report["partial"] is True
    assert len(client.calls) == 1 and not client.answers


def test_reading_missing_optional_expressions_does_not_fail_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[0])
    original = client._questions

    def without_expressions() -> list[dict[str, Any]]:
        return [{**question, "vocabularyCandidates": []} for question in original()]

    monkeypatch.setattr(client, "_questions", without_expressions)
    assert _run(client)["status"] == "COMPLETED"


def test_internal_generation_date_is_not_public_learning_date(tmp_path: Path):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[0])
    report = _run(client, expected_learning_date="2026-09-19")
    snapshot = json.loads(Path(report["initialDatabaseSnapshot"]["privateArtifact"]).read_text(encoding="utf-8"))
    assert "learningDate" not in snapshot["generationRequestSnapshot"]
    assert snapshot["generationRequestSnapshot"]["generationDate"] == report["learningDate"]


def _date_bug_checkpoint(client: FakeReadingClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def old_date_contract(*args: Any, **kwargs: Any) -> None:
        raise KeyError("learningDate")

    with monkeypatch.context() as patch:
        patch.setattr(qa_campaign_reading, "_snapshot", old_date_contract)
        with pytest.raises(KeyError, match="learningDate"):
            _run(client, expected_learning_date="2026-09-19")


def test_explicit_resume_preserves_failure_and_never_restarts_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[1])
    _date_bug_checkpoint(client, monkeypatch)
    assert [call["action"] for call in client.calls] == ["reading-start"]
    report = resume_reading(client, client.case, 91, sleep=lambda _: None, clock=lambda: 0)
    assert report["status"] == "COMPLETED" and "errorType" not in report
    checkpoint = json.loads(Path(report["originalFailureCheckpoint"]).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "FAILED" and checkpoint["errorType"] == "KeyError"
    assert checkpoint["answers"] == []
    assert [call["action"] for call in client.calls].count("reading-start") == 1
    assert sum(call["allowProvider"] for call in client.calls) == 1
    assert len(client.answers) == 5
    assert report["priorObservations"][0]["errorType"] == "KeyError"
    with pytest.raises(ValueError, match="Only the diagnosed"):
        resume_reading(client, client.case, 91)


def test_resume_refuses_existing_answer_instead_of_resubmitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[1])
    _date_bug_checkpoint(client, monkeypatch)
    client.answers[1] = ["A"]
    with pytest.raises(ValueError, match="SAME_UNANSWERED_SET"):
        resume_reading(client, client.case, 91)
    assert all(call["action"] != "reading-answer" for call in client.calls)


def test_resume_wrong_set_id_has_no_http_side_effect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = FakeReadingClient(tmp_path, NEW_RUN_CASES[1])
    _date_bug_checkpoint(client, monkeypatch)
    with pytest.raises(ValueError, match="Only the diagnosed"):
        resume_reading(client, client.case, 92)
    assert len(client.calls) == 1
