from __future__ import annotations

import asyncio
from copy import deepcopy
import logging
import time
from typing import Any

from pydantic import ValidationError

from app.ai.ports import StructuredTextGenerationProvider
from app.core.config import settings
from app.features.language_learning.speaking.evidence_policy import (
    SPEAKING_EVIDENCE_POLICY_VERSION,
    TRANSCRIPT_EVIDENCE_SOURCE,
    transcript_is_usable,
    unsupported_metrics,
)
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.speaking.policy import (
    SPEAKING_EVALUATION_PROMPT_VERSION,
    SPEAKING_EVALUATION_VERSION,
    SPEAKING_SCORING_POLICY_VERSION,
    SPEAKING_METRIC_WEIGHTS,
    calculate_evaluation_eligibility,
    calculate_speaking_overall,
    validate_metric_set,
)
from app.features.language_learning.speaking.prompts import (
    build_evaluation_prompt,
    evaluation_evidence_bounds,
)
from app.features.language_learning.speaking.provider_error import map_provider_exception
from app.features.language_learning.speaking.retry import run_with_stage_retry
from app.schemas.language_learning_speaking import (
    MetricEvaluationState,
    SpeakingErrorCode,
    SpeakingEvaluationPayload,
    SpeakingEvaluationRequest,
    SpeakingEvaluationResponse,
    SpeakingEvaluationStatus,
    SpeakingPracticeMode,
    SpeakingStage,
    SpeakingUsage,
    StageUsage,
)

logger = logging.getLogger(__name__)


def _evaluation_response_schema(request: SpeakingEvaluationRequest) -> dict[str, Any]:
    """Expose existing request-bound evidence rules without changing public DTOs."""
    schema = SpeakingEvaluationPayload.model_json_schema()
    bounds = evaluation_evidence_bounds(request)
    if not bounds:
        raise ValueError("Speaking evaluation requires an eligible user turn")
    definitions = schema["$defs"]
    original_evidence = definitions["MetricEvidence"]
    alternatives = []
    for turn_id, maximum in bounds.items():
        evidence = deepcopy(original_evidence)
        properties = evidence["properties"]
        properties["turnId"]["enum"] = [turn_id]
        for field in ("startMs", "endMs"):
            # Optional/nullable timing remains supported. Each branch binds the
            # correct measured duration to its exact learner turn, not any turn.
            for variant in properties[field]["anyOf"]:
                if variant.get("type") == "integer":
                    variant["maximum"] = maximum
        alternatives.append(evidence)
    definitions["MetricEvidence"] = {"anyOf": alternatives}
    for name in ("SpeakingProfileSignal", "RecommendedExpression", "PronunciationPractice"):
        definitions[name]["properties"]["evidenceTurnIds"]["items"]["enum"] = list(bounds)
    # Match the actual text-only provider's capability before sampling. Runtime
    # validation remains authoritative even when a provider ignores this schema.
    metric_schema = definitions["SpeakingMetricPayload"]
    alternatives = []
    unavailable = unsupported_metrics(request)
    for metric_type in SPEAKING_METRIC_WEIGHTS:
        if metric_type in unavailable:
            continue
        branch = deepcopy(metric_schema)
        properties = branch["properties"]
        properties["type"] = {"type": "string", "enum": [metric_type.value]}
        alternatives.append(branch)
    definitions["SpeakingMetricPayload"] = (
        alternatives[0] if len(alternatives) == 1 else {"anyOf": alternatives}
    )
    schema["properties"]["metrics"]["minItems"] = len(alternatives)
    schema["properties"]["metrics"]["maxItems"] = len(alternatives)
    schema["properties"]["pronunciationPractice"]["maxItems"] = 0
    if request.practice_mode == SpeakingPracticeMode.READ_ALOUD:
        schema["properties"]["profileSignals"]["maxItems"] = 0
    return schema


def _complete_server_owned_metric_reasons(
    request: SpeakingEvaluationRequest, data: dict[str, Any],
) -> dict[str, Any]:
    """Fill only omitted reasons for axes the server already marked unavailable.

    A text-only Mini may omit the reason while correctly returning null score
    and no evidence. Never repair a model's state, score, evidence, or a
    supported metric; all of those remain subject to ordinary validation.
    """
    unavailable = {metric.value: reason for metric, reason in unsupported_metrics(request).items()}
    metrics = data.get("metrics")
    if not isinstance(metrics, list):
        return data
    completed = deepcopy(data)
    supported = {metric.value for metric in SPEAKING_METRIC_WEIGHTS if metric not in unsupported_metrics(request)}
    returned = [metric.get("type") if isinstance(metric, dict) else None for metric in metrics]
    if len(returned) == len(supported) and set(returned) == supported:
        by_type = {metric["type"]: metric for metric in completed["metrics"]}
        completed["metrics"] = [
            by_type[metric_type.value] if metric_type.value in by_type else {
                "type": metric_type.value,
                "state": "NOT_EVALUABLE",
                "score": None,
                "confidence": 1.0,
                "summary": "This evaluator has no evidence for this metric.",
                "evidence": [],
                "notEvaluableReason": unavailable[metric_type.value],
            }
            for metric_type in SPEAKING_METRIC_WEIGHTS
        ]
        return completed
    # Legacy full-metric responses are still validated, never silently trimmed.
    for metric in completed["metrics"]:
        if (isinstance(metric, dict)
                and metric.get("type") in unavailable
                and metric.get("state") == "NOT_EVALUABLE"
                and metric.get("score") is None
                and metric.get("evidence") == []
                and metric.get("notEvaluableReason") is None):
            metric["notEvaluableReason"] = unavailable[metric["type"]]
    return completed


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
                evaluated_axes=[],
                evaluation_coverage=0.0,
                evidence_policy_version=SPEAKING_EVIDENCE_POLICY_VERSION,
                evidence_source=TRANSCRIPT_EVIDENCE_SOURCE,
                usage=SpeakingUsage(),
            )

        prompt = build_evaluation_prompt(request)
        schema = _evaluation_response_schema(request)
        started = time.perf_counter()

        async def operation():
            validation_stage = "provider"
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
                validation_stage = "payload_schema"
                payload = SpeakingEvaluationPayload.model_validate(
                    _complete_server_owned_metric_reasons(request, result.data)
                )
                validation_stage = "metric_coverage"
                validate_metric_set(payload.metrics)
                validation_stage = "evidence_binding"
                self._validate_evidence(request, payload)
                validation_stage = "evidence_capability"
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
                # Provider output can contain learner text. Record only schema
                # locations/types and the application-owned validation stage.
                schema_errors = (
                    [{"loc": ".".join(map(str, item["loc"])), "type": item["type"]}
                     for item in exc.errors()]
                    if isinstance(exc, ValidationError) else []
                )
                logger.warning(
                    "Speaking evaluation output rejected. stage=%s errorType=%s schemaErrors=%s",
                    validation_stage, type(exc).__name__, schema_errors,
                )
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
        # A confident "not evaluable" judgement still has no numerical score.
        # Never emit EVALUATED/overallScore=null across the BE contract.
        if status == SpeakingEvaluationStatus.EVALUATED and overall is None:
            status = SpeakingEvaluationStatus.INSUFFICIENT_EVIDENCE

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
            evaluated_axes=[metric.type for metric in payload.metrics if metric.state == MetricEvaluationState.EVALUATED],
            evaluation_coverage=round(sum(
                SPEAKING_METRIC_WEIGHTS[metric.type] for metric in payload.metrics
                if metric.state == MetricEvaluationState.EVALUATED
            ), 4),
            evidence_policy_version=SPEAKING_EVIDENCE_POLICY_VERSION,
            evidence_source=TRANSCRIPT_EVIDENCE_SOURCE,
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
        bounds = evaluation_evidence_bounds(request)
        for metric in payload.metrics:
            for evidence in metric.evidence:
                turn = valid_turns.get(evidence.turn_id)
                if turn is None:
                    raise ValueError(
                        "Evaluation evidence turnId가 유효하지 않습니다: "
                        f"{evidence.turn_id}"
                    )
                max_ms = bounds[turn.turn_id]
                if any(
                    timestamp is not None and timestamp > max_ms
                    for timestamp in (evidence.start_ms, evidence.end_ms)
                ):
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
        unavailable = unsupported_metrics(request)
        usable = {turn.turn_id for turn in request.user_turns if transcript_is_usable(turn)}
        evaluated = {metric.type for metric in payload.metrics if metric.state == MetricEvaluationState.EVALUATED}
        for metric in payload.metrics:
            if metric.type in unavailable and (
                metric.state != MetricEvaluationState.NOT_EVALUABLE
                or metric.score is not None or metric.evidence
            ):
                raise ValueError("Speaking metric exceeds actual evaluator evidence capability")
            if metric.state == MetricEvaluationState.EVALUATED and any(
                evidence.turn_id not in usable for evidence in metric.evidence
            ):
                raise ValueError("Uncertain STT observation cannot support an evaluated metric")
        if payload.pronunciation_practice:
            raise ValueError("Text-only evaluator cannot prescribe pronunciation practice")
        if request.practice_mode == SpeakingPracticeMode.READ_ALOUD and payload.profile_signals:
            raise ValueError("Copied READ_ALOUD script cannot create language ability profile signals")
        for signal in payload.profile_signals:
            if signal.metric_type not in evaluated or signal.metric_type in unavailable:
                raise ValueError("Profile signal must use an evaluated supported metric")
            if len(set(signal.evidence_turn_ids)) < 2 or any(
                turn_id not in usable for turn_id in signal.evidence_turn_ids
            ):
                raise ValueError("Profile signal requires independent usable transcript evidence")
        for expression in payload.recommended_expressions:
            if not expression.evidence_turn_ids or any(turn_id not in usable for turn_id in expression.evidence_turn_ids):
                raise ValueError("Corrective recommendation requires usable transcript evidence")
        has_reference = request.practice_mode != SpeakingPracticeMode.READ_ALOUD or any(
            turn.script_text for turn in request.assistant_turns
        )
        if usable and has_reference and not evaluated:
            raise ValueError("Usable text evidence requires a supported content assessment")
