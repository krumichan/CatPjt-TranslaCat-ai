from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from app.ai.ports import StructuredTextGenerationProvider
from app.common.idempotency import InMemoryIdempotencyStore
from app.core.config import settings
from app.features.language_learning.listening.errors import ListeningStageException
from app.features.language_learning.listening.evaluation_common import (
    answer_revealed_task,
    build_evaluation_response,
    effective_answer_revealed,
    finalize_task,
)
from app.features.language_learning.listening.policy import (
    INTERPRETATION_WEIGHTS,
    LISTENING_EVALUATION_VERSION,
    LISTENING_INTERPRETATION_PROMPT_VERSION,
    calculate_weighted_score,
    resolve_assistance_level,
)
from app.features.language_learning.listening.prompts import build_interpretation_prompt
from app.features.language_learning.listening.provider_error import (
    map_provider_exception,
)
from app.features.language_learning.listening.retry import run_with_stage_retry
from app.schemas.language_learning_listening import (
    InterpretationEvaluationPayload,
    InterpretationEvaluationRequest,
    InterpretationMetricType,
    ListeningErrorCode,
    ListeningEvaluationResponse,
    ListeningMetric,
    ListeningStage,
    ListeningTaskResult,
    ListeningTaskStatus,
    ListeningTaskType,
    ListeningUsage,
    StageUsage,
)


class ListeningInterpretationService:
    TYPE_NAME = "LANGUAGE_LEARNING_LISTENING_INTERPRETATION"

    def __init__(
        self,
        provider: StructuredTextGenerationProvider,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
        idempotency_store: InMemoryIdempotencyStore[ListeningEvaluationResponse]
        | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = (
            timeout_seconds or settings.AI_LISTENING_EVALUATION_TIMEOUT_SECONDS
        )
        self.automatic_retries = (
            settings.AI_LISTENING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else min(automatic_retries, 2)
        )
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def evaluate(
        self,
        request: InterpretationEvaluationRequest,
    ) -> ListeningEvaluationResponse:
        self._validate_manual_retry(request.manual_retry_attempt)
        key = "|".join(
            [
                request.idempotency_key,
                request.evaluation_purpose.value,
                LISTENING_EVALUATION_VERSION,
                request.policy_version,
                request.model_config_version,
            ]
        )
        response, _ = await self.idempotency_store.execute(
            key,
            lambda: self._evaluate_once(request),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _evaluate_once(
        self,
        request: InterpretationEvaluationRequest,
    ) -> ListeningEvaluationResponse:
        if effective_answer_revealed(request):
            task = answer_revealed_task(request, ListeningTaskType.INTERPRETATION)
            return build_evaluation_response(request, task)

        prompt = build_interpretation_prompt(request)
        schema = InterpretationEvaluationPayload.model_json_schema()
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
                    raise ValueError("Interpretation response must be an object")
                payload = InterpretationEvaluationPayload.model_validate(result.data)
                self._validate_payload(request, payload)
                return result, payload
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise ListeningStageException(
                    ListeningErrorCode.PROVIDER_TIMEOUT,
                    ListeningStage.INTERPRETATION,
                    "Interpretation 평가 Provider 응답 시간이 초과되었습니다.",
                    True,
                ) from exc
            except ListeningStageException:
                raise
            except (ValidationError, ValueError) as exc:
                raise ListeningStageException(
                    ListeningErrorCode.INVALID_RESPONSE_SCHEMA,
                    ListeningStage.INTERPRETATION,
                    "Interpretation 평가 응답 Schema가 유효하지 않습니다.",
                    True,
                ) from exc
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=ListeningStage.INTERPRETATION,
                    fallback_code=ListeningErrorCode.INTERPRETATION_FAILED,
                    fallback_message="Interpretation 의미 평가에 실패했습니다.",
                ) from exc

        result, payload = await run_with_stage_retry(
            operation,
            max_retries=self.automatic_retries,
        )
        metrics = [
            ListeningMetric(
                type=metric.type.value,
                score=metric.score,
                weight=INTERPRETATION_WEIGHTS[metric.type.value],
                confidence=metric.confidence,
                evidence=[
                    item.model_copy(update={"metric": metric.type.value})
                    for item in metric.evidence
                ],
            )
            for metric in payload.metrics
        ]
        score = calculate_weighted_score(metrics, INTERPRETATION_WEIGHTS)
        assert score is not None
        evidence = [item for metric in metrics for item in metric.evidence]
        task = ListeningTaskResult(
            task_type=ListeningTaskType.INTERPRETATION,
            status=ListeningTaskStatus.EVALUATED,
            evaluable=True,
            score=score,
            confidence=payload.evaluation_confidence,
            assistance_level=resolve_assistance_level(request.assistance_usage),
            metrics=metrics,
            evidence=evidence,
            strengths=payload.strengths,
            improvements=payload.improvements,
            recommended_interpretations=payload.recommended_interpretations,
            delivered_meaning_units=payload.delivered_meaning_units,
            omitted_meaning_units=payload.omitted_meaning_units,
            misunderstood_meaning_units=payload.misunderstood_meaning_units,
            added_information=payload.added_information,
            profile_eligible=False,
            assistance_usage=request.assistance_usage,
            debug_metadata={
                "promptVersion": LISTENING_INTERPRETATION_PROMPT_VERSION,
                "evaluationVersion": LISTENING_EVALUATION_VERSION,
            },
        )
        task = finalize_task(request, task)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return build_evaluation_response(
            request,
            task,
            usage=ListeningUsage(
                interpretation=StageUsage(
                    latency_ms=elapsed_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    provider=result.provider,
                    model=result.model,
                    prompt_version=LISTENING_INTERPRETATION_PROMPT_VERSION,
                    evaluation_version=LISTENING_EVALUATION_VERSION,
                )
            ),
        )

    @staticmethod
    def _validate_payload(
        request: InterpretationEvaluationRequest,
        payload: InterpretationEvaluationPayload,
    ) -> None:
        metric_types = [metric.type for metric in payload.metrics]
        if len(metric_types) != len(set(metric_types)) or set(metric_types) != set(
            InterpretationMetricType
        ):
            raise ValueError("Interpretation 3개 Metric이 중복 없이 필요합니다.")
        valid_units = set(request.key_meaning_units)
        returned_units = (
            payload.delivered_meaning_units
            + payload.omitted_meaning_units
            + payload.misunderstood_meaning_units
        )
        if any(unit not in valid_units for unit in returned_units):
            raise ValueError(
                "Meaning Unit 결과는 요청 keyMeaningUnits를 참조해야 합니다."
            )

    @staticmethod
    def _validate_manual_retry(attempt: int) -> None:
        if attempt > settings.AI_LISTENING_MANUAL_RETRY_LIMIT:
            raise ListeningStageException(
                ListeningErrorCode.MANUAL_RETRY_LIMIT_EXCEEDED,
                ListeningStage.INTERPRETATION,
                "Interpretation 평가 수동 재시도 가능 횟수를 초과했습니다.",
                False,
            )
