from __future__ import annotations

import pytest

from app.ai.model_policy import AiModelTier, get_task_model_policy
from app.features.language_learning.speaking.coaching_service import (
    SPEAKING_COACHING_POLICY_VERSION,
    SpeakingSessionCoachingService,
    build_coaching_prompt,
)
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.schemas.language_learning_speaking import SpeakingCoachingRequest
from tests.test_language_learning_speaking import FakeStructuredProvider, evaluation_turn


def coaching_request(*, excluded: bool = False, transcript: str = "昨日、友達と映画を見ますた。"):
    turn = evaluation_turn(1, excluded=excluded)
    turn.update({
        "transcript": transcript,
        "recordingRevision": 3,
        "assistanceUsage": [{"type": "HINT", "count": 1}],
    })
    return SpeakingCoachingRequest.model_validate({
        "requestId": "coach-1",
        "idempotencyKey": "coach-idem-1",
        "sessionId": "session-1",
        "topic": "주말 경험",
        "practiceMode": "FREE",
        "evaluationScope": "SESSION",
        "originLanguage": "ko",
        "learningLanguage": "ja",
        "userTurns": [turn],
        "assistantTurns": [
            {"turnId": "opening", "turnIndex": 0, "text": "週末は何をしましたか。"},
            {"turnId": "trailing", "turnIndex": 1, "text": "映画はどうでしたか。"},
        ],
        "evaluationPolicyVersion": "speaking-evaluation-policy-v2",
        "resultKind": "SESSION_COACHING",
        "resultPolicyVersion": SPEAKING_COACHING_POLICY_VERSION,
        "sourceSnapshotHash": "0123456789abcdef0123456789abcdef",
    })


def valid_payload():
    return {
        "contentStatus": "GROUNDED",
        "limitationReasons": [],
        "items": [{
            "observationId": "obs-1",
            "kind": "CORRECTION",
            "turnId": "turn-1",
            "sourceExcerpt": "映画を見ますた",
            "message": "과거 경험에는 과거형을 쓰면 뜻이 더 정확해집니다.",
            "suggestedExpression": "映画を見ました。",
        }],
    }


def test_coaching_task_is_explicit_mini_low() -> None:
    policy = get_task_model_policy("LANGUAGE_LEARNING_SPEAKING_SESSION_COACHING")
    assert policy.tier == AiModelTier.MINI
    assert policy.reasoning_effort == "low"


def test_prompt_excludes_trailing_unanswered_assistant_and_preserves_source_contract() -> None:
    prompt = build_coaching_prompt(coaching_request())
    assert '"referenceAssistantTurnId":"opening"' in prompt
    assert "trailing" not in prompt
    assert '"recordingRevision":3' in prompt
    assert '"sourceProvenance":"AUTOMATIC_SPEECH_RECOGNITION"' in prompt
    assert "Use CORRECTION only when" in prompt
    assert "use ALTERNATIVE for optional refinement" in prompt
    assert "messages in originLanguage" in prompt


@pytest.mark.asyncio
async def test_no_usable_evidence_is_typed_without_provider_call() -> None:
    provider = FakeStructuredProvider(results=[valid_payload()])
    response = await SpeakingSessionCoachingService(provider).coach(
        coaching_request(excluded=True)
    )
    assert response.content_status.value == "NO_USABLE_EVIDENCE"
    assert response.items == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_understandable_error_becomes_grounded_correction_with_server_owned_provenance() -> None:
    provider = FakeStructuredProvider(results=[valid_payload()])
    response = await SpeakingSessionCoachingService(
        provider, automatic_retries=0
    ).coach(coaching_request())
    assert response.content_status.value == "GROUNDED"
    assert response.result_kind.value == "SESSION_COACHING"
    assert response.items[0].evidence.recording_revision == 3
    assert response.items[0].evidence.reference_assistant_turn_id == "opening"
    assert response.items[0].evidence.source_provenance == "AUTOMATIC_SPEECH_RECOGNITION"
    assert response.items[0].suggestion_is_learner_evidence is False
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_fabricated_quote_is_rejected_instead_of_becoming_coaching() -> None:
    payload = valid_payload()
    payload["items"][0]["sourceExcerpt"] = "実際には話していない文"
    provider = FakeStructuredProvider(results=[payload])
    with pytest.raises(SpeakingStageException) as raised:
        await SpeakingSessionCoachingService(provider, automatic_retries=0).coach(
            coaching_request()
        )
    assert raised.value.code.value == "INVALID_RESPONSE_SCHEMA"


@pytest.mark.asyncio
async def test_normalized_but_not_exact_quote_is_not_accepted_as_source_evidence() -> None:
    payload = valid_payload()
    payload["items"][0]["sourceExcerpt"] = "映画を  見ますた"
    provider = FakeStructuredProvider(results=[payload])
    with pytest.raises(SpeakingStageException) as raised:
        await SpeakingSessionCoachingService(provider, automatic_retries=0).coach(
            coaching_request()
        )
    assert raised.value.code.value == "INVALID_RESPONSE_SCHEMA"


@pytest.mark.asyncio
async def test_idempotent_source_policy_reuses_accepted_result_without_second_call() -> None:
    provider = FakeStructuredProvider(results=[valid_payload()])
    service = SpeakingSessionCoachingService(provider, automatic_retries=0)
    first = await service.coach(coaching_request())
    second = await service.coach(coaching_request())
    assert first.source_snapshot_hash == second.source_snapshot_hash
    assert len(provider.calls) == 1
