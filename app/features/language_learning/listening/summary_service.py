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
    LISTENING_SUMMARY_PROMPT_VERSION,
    SUMMARY_WEIGHTS,
    calculate_weighted_score,
    resolve_assistance_level,
)
from app.features.language_learning.listening.prompts import build_summary_prompt
from app.features.language_learning.listening.provider_error import map_provider_exception
from app.features.language_learning.listening.retry import run_with_stage_retry
from app.schemas.language_learning_listening import (
    ListeningAssistanceLevel,
    ListeningErrorCode,
    ListeningEvaluationResponse,
    ListeningMetric,
    ListeningStage,
    ListeningTaskResult,
    ListeningTaskStatus,
    ListeningTaskType,
    ListeningUsage,
    StageUsage,
    SummaryEvaluationPayload,
    SummaryEvaluationRequest,
)


class ListeningSummaryService:
    TYPE_NAME = "LANGUAGE_LEARNING_LISTENING_SUMMARY_EVALUATION"

    def __init__(
        self,
        provider: StructuredTextGenerationProvider,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
        idempotency_store: InMemoryIdempotencyStore[ListeningEvaluationResponse] | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds or settings.AI_LISTENING_EVALUATION_TIMEOUT_SECONDS
        self.automatic_retries = (
            settings.AI_LISTENING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else automatic_retries
        )
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def evaluate(self, request: SummaryEvaluationRequest) -> ListeningEvaluationResponse:
        key = "|".join(
            [
                request.idempotency_key,
                request.evaluation_purpose.value,
                request.policy_version,
                request.model_config_version,
                "SUMMARY",
            ]
        )
        response, _ = await self.idempotency_store.execute(
            key,
            lambda: self._evaluate_once(request),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _evaluate_once(self, request: SummaryEvaluationRequest) -> ListeningEvaluationResponse:
        if effective_answer_revealed(request):
            return build_evaluation_response(
                request,
                answer_revealed_task(request, ListeningTaskType.SUMMARY),
            )

        prompt = build_summary_prompt(request)
        schema = SummaryEvaluationPayload.model_json_schema()
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
                    raise ValueError("Summary response must be an object")
                return result, SummaryEvaluationPayload.model_validate(result.data)
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise ListeningStageException(
                    ListeningErrorCode.PROVIDER_TIMEOUT,
                    ListeningStage.SUMMARY,
                    "Listening 요약 평가 Provider 응답 시간이 초과되었습니다.",
                    True,
                ) from exc
            except (ValidationError, ValueError) as exc:
                raise ListeningStageException(
                    ListeningErrorCode.INVALID_RESPONSE_SCHEMA,
                    ListeningStage.SUMMARY,
                    "Listening 요약 평가 응답 Schema가 유효하지 않습니다.",
                    True,
                ) from exc
            except ListeningStageException:
                raise
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=ListeningStage.SUMMARY,
                    fallback_code=ListeningErrorCode.SUMMARY_FAILED,
                    fallback_message="Listening 요약 평가에 실패했습니다.",
                ) from exc

        result, payload = await run_with_stage_retry(
            operation,
            max_retries=self.automatic_retries,
            context=f"summary request_id={request.request_id}",
        )
        metric_map = {metric.type.value: metric for metric in payload.metrics}
        if set(metric_map) != set(SUMMARY_WEIGHTS):
            raise ListeningStageException(
                ListeningErrorCode.INVALID_RESPONSE_SCHEMA,
                ListeningStage.SUMMARY,
                "Listening 요약 Metric 구성이 유효하지 않습니다.",
                False,
            )
        metrics = [
            ListeningMetric(
                type=metric_type,
                score=metric_map[metric_type].score,
                weight=weight,
                confidence=metric_map[metric_type].confidence,
                evidence=metric_map[metric_type].evidence,
            )
            for metric_type, weight in SUMMARY_WEIGHTS.items()
        ]
        score = calculate_weighted_score(metrics, SUMMARY_WEIGHTS)
        confidence = min(
            payload.evaluation_confidence,
            min(metric.confidence for metric in metrics),
        )
        evidence = [item for metric in metrics for item in metric.evidence]
        task = ListeningTaskResult(
            task_type=ListeningTaskType.SUMMARY,
            status=ListeningTaskStatus.EVALUATED,
            evaluable=True,
            score=score,
            confidence=confidence,
            assistance_level=resolve_assistance_level(
                request.assistance_usage, answer_revealed=False
            ),
            metrics=metrics,
            evidence=evidence,
            strengths=payload.strengths,
            improvements=payload.improvements,
            recommended_interpretations=payload.recommended_summaries,
            delivered_meaning_units=payload.delivered_key_points,
            omitted_meaning_units=payload.omitted_key_points,
            assistance_usage=request.assistance_usage,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return build_evaluation_response(
            request,
            finalize_task(request, task),
            usage=ListeningUsage(
                summary=StageUsage(
                    latency_ms=elapsed_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    provider=result.provider,
                    model=result.model,
                    prompt_version=LISTENING_SUMMARY_PROMPT_VERSION,
                )
            ),
        )
