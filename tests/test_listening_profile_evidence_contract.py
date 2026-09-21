"""Profile-reference bounds must not turn a long wrong answer into HTTP 500."""
from __future__ import annotations

import pytest

from app.features.language_learning.listening.dictation_service import ListeningDictationService
from app.schemas.language_learning_listening import (
    DictationEvaluationRequest, ListeningEvaluationResponse, ListeningProfileSignal, ListeningTaskType,
)


@pytest.mark.parametrize("evidence_count", [1, 49, 50, 51, 53, 120])
@pytest.mark.asyncio
async def test_all_dictation_evidence_and_scores_survive_bounded_profile_references(evidence_count):
    payload = {"requestId": "synthetic-evidence-bound", "idempotencyKey": "synthetic-evidence-bound",
               "itemId": "qa-item", "attemptId": "qa-attempt", "evaluationPurpose": "OFFICIAL",
               "sourceText": "あ" * evidence_count, "answer": "い" * evidence_count, "learningLanguage": "ja"}
    request = DictationEvaluationRequest.model_validate(payload)
    service = ListeningDictationService()
    official = await service.evaluate(request)
    practice = await service.evaluate(DictationEvaluationRequest.model_validate({**payload, "evaluationPurpose": "PRACTICE"}))
    task = next(row for row in official.tasks if row.task_type == ListeningTaskType.DICTATION)
    practice_task = next(row for row in practice.tasks if row.task_type == ListeningTaskType.DICTATION)
    assert task.evaluable and task.status.value == "EVALUATED" and task.profile_eligible
    assert task.score == practice_task.score == 25  # Pure substitutions retain the unchanged structure score.
    assert task.alignment == practice_task.alignment and len(task.alignment) == evidence_count
    assert task.evidence == practice_task.evidence and len(task.evidence) == evidence_count
    assert task.metrics == practice_task.metrics
    assert practice_task.profile_signals == []
    expected = [f"evidence-{index + 1}" for index in range(min(evidence_count, 50))]
    assert len(task.profile_signals) == 3
    assert all(signal.evidence_ids == expected for signal in task.profile_signals)
    assert all(int(identity.removeprefix("evidence-")) <= len(task.evidence)
               for signal in task.profile_signals for identity in signal.evidence_ids)
    summary = task.debug_metadata.get("profileSignalEvidenceReferences")
    if evidence_count > 50:
        assert summary == {"totalTaskEvidenceCount": evidence_count, "referencedEvidenceCount": 50,
                           "unreferencedEvidenceCount": evidence_count - 50,
                           "selection": "FIRST_IN_EXISTING_EVIDENCE_ORDER", "fullTaskEvidencePreserved": True}
    else:
        assert summary is None
    # Keep the existing public contract intact; response JSON can be reparsed.
    assert ListeningProfileSignal.model_json_schema()["properties"]["evidenceIds"]["maxItems"] == 50
    ListeningEvaluationResponse.model_validate_json(official.model_dump_json(by_alias=True))
    again = await service.evaluate(request.model_copy(update={"request_id": "same-idempotent-request"}))
    assert again.tasks == official.tasks
