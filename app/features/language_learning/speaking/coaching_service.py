from __future__ import annotations

import asyncio
from hashlib import sha256
import json
import time
from typing import Any

from pydantic import ValidationError

from app.ai.ports import StructuredTextGenerationProvider
from app.core.config import settings
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.evidence_policy import transcript_is_usable
from app.features.language_learning.speaking.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.speaking.provider_error import map_provider_exception
from app.features.language_learning.speaking.retry import run_with_stage_retry
from app.schemas.language_learning_speaking import (
    SpeakingCoachingContentStatus,
    SpeakingCoachingDraftPayload,
    SpeakingCoachingEvidence,
    SpeakingCoachingItem,
    SpeakingCoachingRequest,
    SpeakingCoachingResponse,
    SpeakingErrorCode,
    SpeakingStage,
    SpeakingUsage,
    StageUsage,
)

SPEAKING_COACHING_POLICY_VERSION = "free-session-coaching-v1"
SPEAKING_COACHING_SCHEMA_VERSION = "speaking-session-coaching-schema-v1"
SPEAKING_COACHING_PROMPT_VERSION = "speaking-session-coaching-prompt-v1"


def _reference_turns(request: SpeakingCoachingRequest) -> dict[str, str | None]:
    last_user_index = max(turn.turn_index for turn in request.user_turns)
    by_index = {
        turn.turn_index: turn.turn_id
        for turn in request.assistant_turns
        if turn.turn_index < last_user_index
    }
    return {
        turn.turn_id: by_index.get(turn.turn_index - 1)
        for turn in request.user_turns
    }


def build_coaching_prompt(request: SpeakingCoachingRequest) -> str:
    references = _reference_turns(request)
    usable = [turn for turn in request.user_turns if transcript_is_usable(turn)]
    payload = {
        "sessionId": request.session_id,
        "originLanguage": request.origin_language,
        "learningLanguage": request.learning_language,
        "topic": request.topic,
        "goal": request.goal,
        "resultPolicyVersion": request.result_policy_version,
        "sourceSnapshotHash": request.source_snapshot_hash,
        "eligibleLearnerTurns": [
            {
                "turnId": turn.turn_id,
                "turnIndex": turn.turn_index,
                "transcript": turn.transcript,
                "sttConfidence": turn.stt_confidence,
                "recordingRevision": turn.recording_revision,
                "referenceAssistantTurnId": references[turn.turn_id],
                "assistanceUsage": [
                    item.model_dump(mode="json", by_alias=True)
                    for item in turn.assistance_usage
                ],
                "sourceProvenance": "AUTOMATIC_SPEECH_RECOGNITION",
                "verbatimAccuracyVerified": False,
            }
            for turn in usable
        ],
        "evidenceContract": {
            "allowedTurnIds": [turn.turn_id for turn in usable],
            "sourceExcerptMustBeExactContiguousTranscriptSpan": True,
            "assistantAndSuggestedTextAreNotLearnerEvidence": True,
            "maxItems": 3,
        },
    }
    return (
        "Create evidence-grounded session coaching from this fixed source snapshot. "
        "Do not score the learner and do not infer acoustic pronunciation or fluency. "
        "Write coaching messages in originLanguage and suggested expressions in learningLanguage. "
        "Use CORRECTION only when the exact quoted learner expression has an objective grammar, "
        "word-choice, or expression error. Never label a grammatical, natural sentence as a "
        "CORRECTION merely because another phrasing is stylistically preferable; use ALTERNATIVE "
        "for optional refinement. Do not invent a correction quota.\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


class SpeakingSessionCoachingService:
    TYPE_NAME = "LANGUAGE_LEARNING_SPEAKING_SESSION_COACHING"

    def __init__(
        self,
        provider: StructuredTextGenerationProvider,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
        idempotency_store: InMemoryIdempotencyStore[SpeakingCoachingResponse] | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds or settings.AI_SPEAKING_EVALUATION_TIMEOUT_SECONDS
        self.automatic_retries = (
            settings.AI_SPEAKING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None else automatic_retries
        )
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def coach(self, request: SpeakingCoachingRequest) -> SpeakingCoachingResponse:
        if request.result_policy_version != SPEAKING_COACHING_POLICY_VERSION:
            raise SpeakingStageException(
                code=SpeakingErrorCode.INVALID_RESPONSE_SCHEMA,
                stage=SpeakingStage.EVALUATION,
                message="지원하지 않는 Speaking coaching policy입니다.",
                retryable=False,
            )
        key = f"{request.session_id}:{request.result_policy_version}:{request.source_snapshot_hash}"
        response, _ = await self.idempotency_store.execute(key, lambda: self._coach_once(request))
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _coach_once(self, request: SpeakingCoachingRequest) -> SpeakingCoachingResponse:
        usable = [turn for turn in request.user_turns if transcript_is_usable(turn)]
        if not usable:
            return self._response(
                request,
                content_status=SpeakingCoachingContentStatus.NO_USABLE_EVIDENCE,
                limitation_reasons=["NO_USABLE_LEARNER_TRANSCRIPT"],
                items=[],
                usage=SpeakingUsage(),
            )
        prompt = build_coaching_prompt(request)
        schema = SpeakingCoachingDraftPayload.model_json_schema()
        schema["$defs"]["SpeakingCoachingDraftItem"]["properties"]["turnId"]["enum"] = [
            turn.turn_id for turn in usable
        ]
        started = time.perf_counter()

        async def operation() -> tuple[Any, SpeakingCoachingDraftPayload]:
            try:
                result = await asyncio.wait_for(
                    self.provider.call_with_metadata(
                        type_name=self.TYPE_NAME, data=prompt, schema=schema
                    ),
                    timeout=self.timeout_seconds,
                )
                payload = SpeakingCoachingDraftPayload.model_validate(result.data)
                self._validate_payload(request, payload, usable)
                return result, payload
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise SpeakingStageException(
                    code=SpeakingErrorCode.PROVIDER_TIMEOUT,
                    stage=SpeakingStage.EVALUATION,
                    message="Speaking coaching Provider 응답 시간이 초과되었습니다.",
                    retryable=True,
                ) from exc
            except (ValidationError, ValueError) as exc:
                raise SpeakingStageException(
                    code=SpeakingErrorCode.INVALID_RESPONSE_SCHEMA,
                    stage=SpeakingStage.EVALUATION,
                    message="Speaking coaching 응답 근거가 유효하지 않습니다.",
                    retryable=True,
                ) from exc
            except SpeakingStageException:
                raise
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=SpeakingStage.EVALUATION,
                    fallback_code=SpeakingErrorCode.EVALUATION_FAILED,
                    fallback_message="Speaking session coaching 생성에 실패했습니다.",
                ) from exc

        result, draft = await run_with_stage_retry(operation, max_retries=self.automatic_retries)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        turn_by_id = {turn.turn_id: turn for turn in usable}
        references = _reference_turns(request)
        items = [
            SpeakingCoachingItem(
                observation_id=item.observation_id,
                kind=item.kind,
                evidence=SpeakingCoachingEvidence(
                    turn_id=item.turn_id,
                    turn_index=turn_by_id[item.turn_id].turn_index,
                    recording_revision=turn_by_id[item.turn_id].recording_revision,
                    transcript_excerpt=item.source_excerpt,
                    transcript_hash=sha256(
                        turn_by_id[item.turn_id].transcript.encode("utf-8")
                    ).hexdigest(),
                    reference_assistant_turn_id=references[item.turn_id],
                    assistance_usage=turn_by_id[item.turn_id].assistance_usage,
                ),
                message=item.message,
                suggested_expression=item.suggested_expression,
                suggestion_is_learner_evidence=False,
            )
            for item in draft.items
        ]
        return self._response(
            request,
            content_status=draft.content_status,
            limitation_reasons=draft.limitation_reasons,
            items=items,
            usage=SpeakingUsage(
                coaching=StageUsage(
                    latency_ms=elapsed_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    provider=result.provider,
                    model=result.model,
                    prompt_version=SPEAKING_COACHING_PROMPT_VERSION,
                    evaluation_version=SPEAKING_COACHING_SCHEMA_VERSION,
                )
            ),
        )

    @staticmethod
    def _validate_payload(request, payload, usable) -> None:
        by_id = {turn.turn_id: turn for turn in usable}
        forbidden = ("turnId", "recordingRevision", "sourceSnapshotHash", "metadata")
        for item in payload.items:
            turn = by_id.get(item.turn_id)
            if turn is None:
                raise ValueError("Coaching evidence references an ineligible turn")
            if item.source_excerpt not in turn.transcript:
                raise ValueError("Coaching excerpt is not present in the learner transcript")
            if any(marker in item.message for marker in forbidden):
                raise ValueError("Coaching message exposes internal metadata")
            if item.suggested_expression and any(
                marker in item.suggested_expression for marker in forbidden
            ):
                raise ValueError("Coaching suggestion exposes internal metadata")

    @staticmethod
    def _response(request, *, content_status, limitation_reasons, items, usage):
        return SpeakingCoachingResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            result_policy_version=request.result_policy_version,
            schema_version=SPEAKING_COACHING_SCHEMA_VERSION,
            source_snapshot_hash=request.source_snapshot_hash,
            content_status=content_status,
            limitation_reasons=limitation_reasons,
            items=items,
            prompt_version=SPEAKING_COACHING_PROMPT_VERSION,
            usage=usage,
        )
