from __future__ import annotations

import asyncio
import copy
import json
import logging
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


logger = logging.getLogger(__name__)

_PROVIDER_PAYLOAD_LOG_LIMIT = 4000

_SEVERITY_ALIASES = {
    "NONE": "INFO",
    "MINOR": "LOW",
    "MODERATE": "MEDIUM",
    "MAJOR": "HIGH",
    "CRITICAL": "HIGH",
}


def _canonicalize_confidence(value: object) -> tuple[object, bool]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value, False

    numeric = float(value)
    if 0.0 <= numeric <= 1.0:
        return numeric, numeric != value
    if 1.0 < numeric <= 5.0:
        return numeric / 5.0, True
    return value, False


def _canonicalize_provider_payload(
    data: dict[str, object],
) -> tuple[dict[str, object], int, int, int]:
    normalized = copy.deepcopy(data)
    normalized_severity_count = 0
    normalized_confidence_count = 0
    normalized_score_count = 0

    confidence, changed = _canonicalize_confidence(
        normalized.get("evaluationConfidence")
    )
    if changed:
        normalized["evaluationConfidence"] = confidence
        normalized_confidence_count += 1

    metrics = normalized.get("metrics")
    if not isinstance(metrics, list):
        return (
            normalized,
            normalized_severity_count,
            normalized_confidence_count,
            normalized_score_count,
        )

    metric_scores = [
        metric.get("score")
        for metric in metrics
        if isinstance(metric, dict)
    ]
    normalized_score_scale = (
        len(metric_scores) == len(metrics)
        and bool(metric_scores)
        and all(
            not isinstance(score, bool)
            and isinstance(score, (int, float))
            and 0.0 <= float(score) <= 1.0
            for score in metric_scores
        )
    )
    if normalized_score_scale:
        for metric in metrics:
            assert isinstance(metric, dict)
            metric["score"] = float(metric["score"]) * 100.0
            normalized_score_count += 1

    for metric in metrics:
        if not isinstance(metric, dict):
            continue

        confidence, changed = _canonicalize_confidence(metric.get("confidence"))
        if changed:
            metric["confidence"] = confidence
            normalized_confidence_count += 1

        evidence = metric.get("evidence")
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, dict):
                continue
            severity = item.get("severity")
            if not isinstance(severity, str):
                continue
            canonical = _SEVERITY_ALIASES.get(
                severity.strip().upper(),
                severity.strip().upper(),
            )
            if canonical != severity:
                item["severity"] = canonical
                normalized_severity_count += 1

    return (
        normalized,
        normalized_severity_count,
        normalized_confidence_count,
        normalized_score_count,
    )


def _provider_payload_preview(data: object) -> str:
    try:
        rendered = json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(data)
    if len(rendered) <= _PROVIDER_PAYLOAD_LOG_LIMIT:
        return rendered
    return f"{rendered[:_PROVIDER_PAYLOAD_LOG_LIMIT]}...<truncated>"


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
        logger.info(
            "Listening interpretation evaluation request received. "
            "request_id=%s item_id=%s attempt_id=%s origin=%s learning=%s "
            "meaning_units=%d answer_chars=%d manual_retry_attempt=%d",
            request.request_id,
            request.item_id,
            request.attempt_id,
            request.origin_language,
            request.learning_language,
            len(request.key_meaning_units),
            len(request.answer),
            request.manual_retry_attempt,
        )
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
        response, cache_hit = await self.idempotency_store.execute(
            key,
            lambda: self._evaluate_once(request),
        )
        logger.info(
            "Listening interpretation evaluation request completed. "
            "request_id=%s item_id=%s attempt_id=%s cache_hit=%s overall_score=%s",
            request.request_id,
            request.item_id,
            request.attempt_id,
            cache_hit,
            response.overall.score,
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
        provider_attempt = 0

        async def operation():
            nonlocal provider_attempt
            provider_attempt += 1
            attempt_started = time.perf_counter()
            provider_data: object | None = None
            logger.info(
                "Listening interpretation provider call started. "
                "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d "
                "timeout_seconds=%s",
                request.request_id,
                request.item_id,
                request.attempt_id,
                provider_attempt,
                self.automatic_retries + 1,
                self.timeout_seconds,
            )
            try:
                result = await asyncio.wait_for(
                    self.provider.call_with_metadata(
                        type_name=self.TYPE_NAME,
                        data=prompt,
                        schema=schema,
                    ),
                    timeout=self.timeout_seconds,
                )
                provider_data = result.data
                provider_elapsed_ms = int(
                    (time.perf_counter() - attempt_started) * 1000
                )
                logger.info(
                    "Listening interpretation provider call completed. "
                    "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d "
                    "latency_ms=%d provider=%s model=%s input_tokens=%d "
                    "output_tokens=%d data_type=%s",
                    request.request_id,
                    request.item_id,
                    request.attempt_id,
                    provider_attempt,
                    self.automatic_retries + 1,
                    provider_elapsed_ms,
                    result.provider,
                    result.model,
                    result.input_tokens,
                    result.output_tokens,
                    type(result.data).__name__,
                )
                if not isinstance(result.data, dict):
                    raise ValueError("Interpretation response must be an object")
                (
                    provider_data,
                    normalized_severity_count,
                    normalized_confidence_count,
                    normalized_score_count,
                ) = _canonicalize_provider_payload(result.data)
                if normalized_severity_count:
                    logger.info(
                        "Listening interpretation evidence severity normalized. "
                        "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d "
                        "normalized_count=%d",
                        request.request_id,
                        request.item_id,
                        request.attempt_id,
                        provider_attempt,
                        self.automatic_retries + 1,
                        normalized_severity_count,
                    )
                if normalized_confidence_count:
                    logger.info(
                        "Listening interpretation confidence normalized. "
                        "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d "
                        "normalized_count=%d",
                        request.request_id,
                        request.item_id,
                        request.attempt_id,
                        provider_attempt,
                        self.automatic_retries + 1,
                        normalized_confidence_count,
                    )
                if normalized_score_count:
                    logger.info(
                        "Listening interpretation metric score scale normalized. "
                        "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d "
                        "normalized_count=%d",
                        request.request_id,
                        request.item_id,
                        request.attempt_id,
                        provider_attempt,
                        self.automatic_retries + 1,
                        normalized_score_count,
                    )
                payload = InterpretationEvaluationPayload.model_validate(provider_data)
                self._validate_payload(request, payload)
                logger.info(
                    "Listening interpretation response validated. "
                    "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d "
                    "metrics=%d delivered_units=%d omitted_units=%d "
                    "misunderstood_units=%d",
                    request.request_id,
                    request.item_id,
                    request.attempt_id,
                    provider_attempt,
                    self.automatic_retries + 1,
                    len(payload.metrics),
                    len(payload.delivered_meaning_units),
                    len(payload.omitted_meaning_units),
                    len(payload.misunderstood_meaning_units),
                )
                return result, payload
            except (TimeoutError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Listening interpretation provider timed out. "
                    "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d",
                    request.request_id,
                    request.item_id,
                    request.attempt_id,
                    provider_attempt,
                    self.automatic_retries + 1,
                )
                raise ListeningStageException(
                    ListeningErrorCode.PROVIDER_TIMEOUT,
                    ListeningStage.INTERPRETATION,
                    "Interpretation 평가 Provider 응답 시간이 초과되었습니다.",
                    True,
                ) from exc
            except ListeningStageException as exc:
                logger.warning(
                    "Listening interpretation stage failed. "
                    "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d "
                    "code=%s stage=%s retryable=%s message=%s",
                    request.request_id,
                    request.item_id,
                    request.attempt_id,
                    provider_attempt,
                    self.automatic_retries + 1,
                    exc.code.value,
                    exc.stage.value,
                    exc.retryable,
                    exc.message,
                )
                raise
            except ValidationError as exc:
                logger.error(
                    "Listening interpretation response schema validation failed. "
                    "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d "
                    "validation_errors=%s provider_payload=%s",
                    request.request_id,
                    request.item_id,
                    request.attempt_id,
                    provider_attempt,
                    self.automatic_retries + 1,
                    exc.errors(include_url=False, include_input=False),
                    _provider_payload_preview(provider_data),
                )
                raise ListeningStageException(
                    ListeningErrorCode.INVALID_RESPONSE_SCHEMA,
                    ListeningStage.INTERPRETATION,
                    "Interpretation 평가 응답 Schema가 유효하지 않습니다.",
                    False,
                ) from exc
            except ValueError as exc:
                logger.error(
                    "Listening interpretation response business validation failed. "
                    "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d "
                    "error=%s provider_payload=%s",
                    request.request_id,
                    request.item_id,
                    request.attempt_id,
                    provider_attempt,
                    self.automatic_retries + 1,
                    exc,
                    _provider_payload_preview(provider_data),
                )
                raise ListeningStageException(
                    ListeningErrorCode.INVALID_RESPONSE_SCHEMA,
                    ListeningStage.INTERPRETATION,
                    "Interpretation 평가 응답 Schema가 유효하지 않습니다.",
                    False,
                ) from exc
            except Exception as exc:
                mapped = map_provider_exception(
                    exc,
                    stage=ListeningStage.INTERPRETATION,
                    fallback_code=ListeningErrorCode.INTERPRETATION_FAILED,
                    fallback_message="Interpretation 의미 평가에 실패했습니다.",
                )
                logger.exception(
                    "Listening interpretation provider call failed. "
                    "request_id=%s item_id=%s attempt_id=%s provider_attempt=%d/%d "
                    "mapped_code=%s mapped_stage=%s retryable=%s",
                    request.request_id,
                    request.item_id,
                    request.attempt_id,
                    provider_attempt,
                    self.automatic_retries + 1,
                    mapped.code.value,
                    mapped.stage.value,
                    mapped.retryable,
                )
                raise mapped from exc

        result, payload = await run_with_stage_retry(
            operation,
            max_retries=self.automatic_retries,
            context=(
                f"interpretation request_id={request.request_id} "
                f"item_id={request.item_id} attempt_id={request.attempt_id}"
            ),
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
