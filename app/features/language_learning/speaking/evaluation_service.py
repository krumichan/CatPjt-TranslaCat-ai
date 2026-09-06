from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from app.ai.ports import StructuredTextGenerationProvider
from app.core.config import settings
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.speaking.policy import (
    SPEAKING_EVALUATION_PROMPT_VERSION,
    SPEAKING_EVALUATION_VERSION,
    SPEAKING_SCORING_POLICY_VERSION,
    calculate_evaluation_eligibility,
    calculate_speaking_overall,
    has_pronunciation_evidence,
    validate_metric_set,
)
from app.features.language_learning.speaking.prompts import build_evaluation_prompt
from app.features.language_learning.speaking.provider_error import map_provider_exception
from app.features.language_learning.speaking.retry import run_with_stage_retry
from app.schemas.language_learning_speaking import (
    SpeakingErrorCode,
    SpeakingEvaluationPayload,
    SpeakingEvaluationRequest,
    SpeakingEvaluationResponse,
    SpeakingEvaluationStatus,
    SpeakingStage,
    SpeakingUsage,
    StageUsage,
)


class SpeakingEvaluationService:
    TYPE_NAME = "LANGUAGE_LEARNING_SPEAKING_EVALUATION"

    def __init__(
        self,
        provider: StructuredTextGenerationProvider,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
        idempotency_store: InMemoryIdempotencyStore[SpeakingEvaluationResponse]
        | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = (
            timeout_seconds or settings.AI_SPEAKING_EVALUATION_TIMEOUT_SECONDS
        )
        self.automatic_retries = (
            settings.AI_SPEAKING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else automatic_retries
        )
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def evaluate(
        self,
        request: SpeakingEvaluationRequest,
    ) -> SpeakingEvaluationResponse:
        if request.manual_retry_attempt > settings.AI_SPEAKING_MANUAL_RETRY_LIMIT:
            raise SpeakingStageException(
                code=SpeakingErrorCode.MANUAL_RETRY_LIMIT_EXCEEDED,
                stage=SpeakingStage.EVALUATION,
                message="Speaking 평가 수동 재시도 가능 횟수를 초과했습니다.",
                retryable=False,
            )

        evaluation_key = (
            f"{request.session_id}:{request.evaluation_scope}:"
            f"{request.evaluation_policy_version}"
        )
        response, _ = await self.idempotency_store.execute(
            evaluation_key,
            lambda: self._evaluate_once(request),
        )
        return response.model_copy(
            deep=True,
            update={"request_id": request.request_id},
        )

    async def _evaluate_once(
        self,
        request: SpeakingEvaluationRequest,
    ) -> SpeakingEvaluationResponse:
        required_user_turns = 5
        required_speech_seconds = 60.0
        if request.evaluation_scope == "READ_ALOUD_PROBLEM":
            required_user_turns = 2
            required_speech_seconds = 0.0
        elif request.practice_mode.value == "READ_ALOUD":
            required_user_turns = 10
            required_speech_seconds = 0.0

        eligibility = calculate_evaluation_eligibility(
            request.user_turns,
            min_stt_confidence=settings.AI_SPEAKING_STT_LOW_CONFIDENCE_THRESHOLD,
            required_user_turns=required_user_turns,
            required_speech_seconds=required_speech_seconds,
            required_stt_turn_ratio=0.80,
        )
        if not eligibility.eligible_before_ai:
            return SpeakingEvaluationResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                status=SpeakingEvaluationStatus.INSUFFICIENT_EVIDENCE,
                overall_score=None,
                evaluation_confidence=None,
                metrics=[],
                strengths=[],
                improvements=[],
                recommended_expressions=[],
                pronunciation_practice=[],
                profile_signals=[],
                eligibility=eligibility,
                evaluation_version=SPEAKING_EVALUATION_VERSION,
                scoring_policy_version=SPEAKING_SCORING_POLICY_VERSION,
                prompt_version=SPEAKING_EVALUATION_PROMPT_VERSION,
                usage=SpeakingUsage(),
            )

        prompt = build_evaluation_prompt(request)
        schema = SpeakingEvaluationPayload.model_json_schema()
        started = time.perf_counter()

        async def operation():
            try:
                result = await asyncio.wait_for(
                    self.provider.call_with_metadata(
                        type_name=self.TYPE_NAME,
                        data=prompt,
                        schema=schema,
                    ),
                    timeout=self.timeout_seconds,
                )
                if not isinstance(result.data, dict):
                    raise ValueError("structured evaluation response must be object")
                payload = SpeakingEvaluationPayload.model_validate(result.data)
                validate_metric_set(payload.metrics)
                self._validate_evidence(request, payload)
                self._enforce_pronunciation_safeguard(request, payload)
                return result, payload
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise SpeakingStageException(
                    code=SpeakingErrorCode.PROVIDER_TIMEOUT,
                    stage=SpeakingStage.EVALUATION,
                    message="Speaking 평가 Provider 응답 시간이 초과되었습니다.",
                    retryable=True,
                ) from exc
            except (ValidationError, ValueError) as exc:
                raise SpeakingStageException(
                    code=SpeakingErrorCode.INVALID_RESPONSE_SCHEMA,
                    stage=SpeakingStage.EVALUATION,
                    message="Speaking 평가 응답 Schema가 유효하지 않습니다.",
                    retryable=True,
                ) from exc
            except SpeakingStageException:
                raise
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=SpeakingStage.EVALUATION,
                    fallback_code=SpeakingErrorCode.EVALUATION_FAILED,
                    fallback_message="Speaking Session 평가에 실패했습니다.",
                ) from exc

        result, payload = await run_with_stage_retry(
            operation,
            max_retries=self.automatic_retries,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        confidence_threshold = settings.AI_SPEAKING_EVALUATION_CONFIDENCE_THRESHOLD
        status = (
            SpeakingEvaluationStatus.EVALUATED
            if payload.evaluation_confidence >= confidence_threshold
            else SpeakingEvaluationStatus.INSUFFICIENT_EVIDENCE
        )
        overall = (
            calculate_speaking_overall(payload.metrics)
            if status == SpeakingEvaluationStatus.EVALUATED
            else None
        )
        profile_signals = (
            payload.profile_signals
            if status == SpeakingEvaluationStatus.EVALUATED
            else []
        )

        return SpeakingEvaluationResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            status=status,
            overall_score=overall,
            evaluation_confidence=payload.evaluation_confidence,
            metrics=payload.metrics,
            strengths=payload.strengths,
            improvements=payload.improvements,
            recommended_expressions=payload.recommended_expressions,
            pronunciation_practice=payload.pronunciation_practice,
            profile_signals=profile_signals,
            eligibility=eligibility,
            evaluation_version=SPEAKING_EVALUATION_VERSION,
            scoring_policy_version=SPEAKING_SCORING_POLICY_VERSION,
            prompt_version=SPEAKING_EVALUATION_PROMPT_VERSION,
            usage=SpeakingUsage(
                evaluation=StageUsage(
                    latency_ms=elapsed_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    provider=result.provider,
                    model=result.model,
                    prompt_version=SPEAKING_EVALUATION_PROMPT_VERSION,
                    evaluation_version=SPEAKING_EVALUATION_VERSION,
                )
            ),
        )

    @staticmethod
    def _validate_evidence(
        request: SpeakingEvaluationRequest,
        payload: SpeakingEvaluationPayload,
    ) -> None:
        valid_turns = {
            turn.turn_id: turn
            for turn in request.user_turns
            if not turn.excluded_from_evaluation
        }
        for metric in payload.metrics:
            for evidence in metric.evidence:
                turn = valid_turns.get(evidence.turn_id)
                if turn is None:
                    raise ValueError(
                        "Evaluation evidence turnId가 유효하지 않습니다: "
                        f"{evidence.turn_id}"
                    )
                if evidence.start_ms is not None and evidence.end_ms is not None:
                    max_ms = int(turn.duration_seconds * 1000) + 250
                    if evidence.end_ms > max_ms:
                        raise ValueError(
                            "Evaluation evidence timestamp가 Turn Audio 범위를 초과했습니다."
                        )

        for signal in payload.profile_signals:
            if any(turn_id not in valid_turns for turn_id in signal.evidence_turn_ids):
                raise ValueError("Profile Signal evidenceTurnIds가 유효하지 않습니다.")

        for expression in payload.recommended_expressions:
            if any(turn_id not in valid_turns for turn_id in expression.evidence_turn_ids):
                raise ValueError("추천 표현 evidenceTurnIds가 유효하지 않습니다.")

        for practice in payload.pronunciation_practice:
            if any(turn_id not in valid_turns for turn_id in practice.evidence_turn_ids):
                raise ValueError("발음 연습 evidenceTurnIds가 유효하지 않습니다.")
            if practice.evidence_turn_ids and not any(
                valid_turns[turn_id].audio_available
                for turn_id in practice.evidence_turn_ids
            ):
                raise ValueError("발음 연습은 실제 Audio Evidence와 연결되어야 합니다.")

    @staticmethod
    def _enforce_pronunciation_safeguard(
        request: SpeakingEvaluationRequest,
        payload: SpeakingEvaluationPayload,
    ) -> None:
        pronunciation = next(
            metric
            for metric in payload.metrics
            if metric.type.value == "PRONUNCIATION"
        )
        has_audio_evidence = has_pronunciation_evidence(
            request.user_turns,
            min_stt_confidence=settings.AI_SPEAKING_STT_LOW_CONFIDENCE_THRESHOLD,
        )
        if pronunciation.state.value == "EVALUATED" and not has_audio_evidence:
            raise ValueError(
                "충분한 Audio/STT evidence 없이 Pronunciation을 평가할 수 없습니다."
            )
