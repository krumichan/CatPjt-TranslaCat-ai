from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from app.ai.ports import StructuredTextGenerationProvider
from app.common.idempotency import InMemoryIdempotencyStore
from app.core.config import settings
from app.features.language_learning.listening.errors import ListeningStageException
from app.features.language_learning.listening.policy import (
    LISTENING_EXPLANATION_PROMPT_VERSION,
    LISTENING_EXPLANATION_VERSION,
)
from app.features.language_learning.listening.prompts import build_explanation_prompt
from app.features.language_learning.listening.provider_error import (
    map_provider_exception,
)
from app.features.language_learning.listening.retry import run_with_stage_retry
from app.schemas.language_learning_listening import (
    ListeningErrorCode,
    ListeningStage,
    ListeningUsage,
    RecommendationExplanationPayload,
    RecommendationExplanationRequest,
    RecommendationExplanationResponse,
    StageUsage,
)


class ListeningExplanationService:
    TYPE_NAME = "LANGUAGE_LEARNING_LISTENING_EXPLANATION"

    def __init__(
        self,
        provider: StructuredTextGenerationProvider,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
        idempotency_store: InMemoryIdempotencyStore[RecommendationExplanationResponse]
        | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = (
            timeout_seconds or settings.AI_LISTENING_EXPLANATION_TIMEOUT_SECONDS
        )
        self.automatic_retries = (
            settings.AI_LISTENING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else min(automatic_retries, 2)
        )
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def explain(
        self,
        request: RecommendationExplanationRequest,
    ) -> RecommendationExplanationResponse:
        key = "|".join(
            [
                request.idempotency_key,
                LISTENING_EXPLANATION_VERSION,
                request.policy_version,
                request.model_config_version,
                request.target_metric.value,
                request.recommended_activity,
                request.recommended_task,
            ]
        )
        response, _ = await self.idempotency_store.execute(
            key,
            lambda: self._explain_once(request),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _explain_once(
        self,
        request: RecommendationExplanationRequest,
    ) -> RecommendationExplanationResponse:
        prompt = build_explanation_prompt(request)
        schema = RecommendationExplanationPayload.model_json_schema()
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
                    raise ValueError("Explanation response must be an object")
                payload = RecommendationExplanationPayload.model_validate(result.data)
                self._validate_sentence_limit(payload.explanation)
                return result, payload
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise ListeningStageException(
                    ListeningErrorCode.PROVIDER_TIMEOUT,
                    ListeningStage.EXPLANATION,
                    "추천 설명 Provider 응답 시간이 초과되었습니다.",
                    True,
                ) from exc
            except ListeningStageException:
                raise
            except (ValidationError, ValueError) as exc:
                raise ListeningStageException(
                    ListeningErrorCode.INVALID_RESPONSE_SCHEMA,
                    ListeningStage.EXPLANATION,
                    "추천 설명 응답 Schema가 유효하지 않습니다.",
                    True,
                ) from exc
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=ListeningStage.EXPLANATION,
                    fallback_code=ListeningErrorCode.EXPLANATION_FAILED,
                    fallback_message="추천 설명 생성에 실패했습니다.",
                ) from exc

        result, payload = await run_with_stage_retry(
            operation,
            max_retries=self.automatic_retries,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RecommendationExplanationResponse(
            request_id=request.request_id,
            target_metric=request.target_metric,
            recommended_activity=request.recommended_activity,
            recommended_task=request.recommended_task,
            explanation=payload.explanation,
            cta_label=payload.cta_label,
            explanation_version=LISTENING_EXPLANATION_VERSION,
            usage=ListeningUsage(
                explanation=StageUsage(
                    latency_ms=elapsed_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    provider=result.provider,
                    model=result.model,
                    prompt_version=LISTENING_EXPLANATION_PROMPT_VERSION,
                )
            ),
        )

    @staticmethod
    def _validate_sentence_limit(explanation: str) -> None:
        terminators = sum(
            explanation.count(mark) for mark in (".", "!", "?", "。", "！", "？")
        )
        if terminators > 2:
            raise ValueError("추천 설명은 최대 두 문장이어야 합니다.")
