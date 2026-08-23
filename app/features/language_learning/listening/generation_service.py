from __future__ import annotations

import asyncio
import hashlib
import time
from difflib import SequenceMatcher

from pydantic import ValidationError

from app.ai.ports import StructuredTextGenerationProvider
from app.common.idempotency import InMemoryIdempotencyStore
from app.core.config import settings
from app.features.language_learning.listening.errors import ListeningStageException
from app.features.language_learning.listening.normalization import (
    normalize_text,
    similarity_key,
)
from app.features.language_learning.listening.policy import (
    DIFFICULTY_DURATION_RANGES,
    LISTENING_GENERATION_PROMPT_VERSION,
    LISTENING_GENERATION_VERSION,
)
from app.features.language_learning.listening.prompts import build_generation_prompt
from app.features.language_learning.listening.provider_error import (
    map_provider_exception,
)
from app.features.language_learning.listening.retry import run_with_stage_retry
from app.schemas.language_learning_listening import (
    ListeningErrorCode,
    ListeningGenerationPayload,
    ListeningItem,
    ListeningSetGenerationRequest,
    ListeningSetGenerationResponse,
    ListeningStage,
    ListeningUsage,
    StageUsage,
)


class ListeningGenerationService:
    TYPE_NAME = "LANGUAGE_LEARNING_LISTENING_GENERATION"

    def __init__(
        self,
        provider: StructuredTextGenerationProvider,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
        idempotency_store: InMemoryIdempotencyStore[ListeningSetGenerationResponse]
        | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = (
            timeout_seconds or settings.AI_LISTENING_GENERATION_TIMEOUT_SECONDS
        )
        self.automatic_retries = (
            settings.AI_LISTENING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else min(automatic_retries, 2)
        )
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def generate(
        self,
        request: ListeningSetGenerationRequest,
    ) -> ListeningSetGenerationResponse:
        self._validate_manual_retry(request.manual_retry_attempt)
        key = "|".join(
            [
                request.idempotency_key,
                LISTENING_GENERATION_VERSION,
                request.policy_version,
                request.model_config_version,
            ]
        )
        response, _ = await self.idempotency_store.execute(
            key,
            lambda: self._generate_once(request),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _generate_once(
        self,
        request: ListeningSetGenerationRequest,
    ) -> ListeningSetGenerationResponse:
        prompt = build_generation_prompt(request)
        schema = ListeningGenerationPayload.model_json_schema()
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
                    raise ValueError("Listening generation response must be an object")
                payload = ListeningGenerationPayload.model_validate(result.data)
                items = self._finalize_items(request, payload)
                return result, items
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise ListeningStageException(
                    ListeningErrorCode.PROVIDER_TIMEOUT,
                    ListeningStage.GENERATION,
                    "Listening 문항 생성 Provider 응답 시간이 초과되었습니다.",
                    True,
                ) from exc
            except ListeningStageException:
                raise
            except (ValidationError, ValueError) as exc:
                raise ListeningStageException(
                    ListeningErrorCode.INVALID_RESPONSE_SCHEMA,
                    ListeningStage.GENERATION,
                    "Listening 문항 생성 응답 Schema가 유효하지 않습니다.",
                    True,
                ) from exc
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=ListeningStage.GENERATION,
                    fallback_code=ListeningErrorCode.GENERATION_FAILED,
                    fallback_message="Listening 문항 생성에 실패했습니다.",
                ) from exc

        result, items = await run_with_stage_retry(
            operation,
            max_retries=self.automatic_retries,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ListeningSetGenerationResponse(
            request_id=request.request_id,
            generation_version=LISTENING_GENERATION_VERSION,
            policy_version=request.policy_version,
            model_config_version=request.model_config_version,
            items=items,
            usage=ListeningUsage(
                generation=StageUsage(
                    latency_ms=elapsed_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    provider=result.provider,
                    model=result.model,
                    prompt_version=LISTENING_GENERATION_PROMPT_VERSION,
                )
            ),
        )

    def _finalize_items(
        self,
        request: ListeningSetGenerationRequest,
        payload: ListeningGenerationPayload,
    ) -> list[ListeningItem]:
        expected_count = request.set_context.item_count
        if len(payload.items) != expected_count:
            raise ValueError("생성 Item 수가 itemCount와 일치하지 않습니다.")
        indexes = [item.item_index for item in payload.items]
        if sorted(indexes) != list(range(1, expected_count + 1)):
            raise ValueError("itemIndex는 1부터 itemCount까지 중복 없이 필요합니다.")

        minimum, maximum = self._duration_range(request)
        recent_hashes = set(request.constraints.recent_content_hashes)
        recent_keys = [
            similarity_key(summary, request.user_context.learning_language)
            for summary in request.constraints.recent_similarity_summaries
        ]
        seen_keys: list[str] = []
        finalized: list[ListeningItem] = []
        for item in sorted(payload.items, key=lambda candidate: candidate.item_index):
            if not item.safety.passed:
                raise ListeningStageException(
                    ListeningErrorCode.UNSAFE_CONTENT,
                    ListeningStage.GENERATION,
                    "안전 정책을 통과하지 못한 Listening 문항이 생성되었습니다.",
                    True,
                )
            if not minimum <= item.estimated_audio_seconds <= maximum:
                raise ValueError(
                    f"estimatedAudioSeconds는 {minimum:g}~{maximum:g}초여야 합니다."
                )
            normalized = normalize_text(
                item.source_text,
                request.user_context.learning_language,
            )
            content_hash = hashlib.sha256(normalized.text.encode("utf-8")).hexdigest()
            key = similarity_key(
                item.source_text, request.user_context.learning_language
            )
            if content_hash in recent_hashes or self._is_similar(
                key, recent_keys + seen_keys
            ):
                raise ListeningStageException(
                    ListeningErrorCode.DUPLICATE_CONTENT,
                    ListeningStage.GENERATION,
                    "최근 또는 현재 Set과 중복되는 Listening 문항이 생성되었습니다.",
                    True,
                )
            seen_keys.append(key)
            finalized.append(
                ListeningItem(
                    item_index=item.item_index,
                    source_text=item.source_text,
                    normalized_source_text=normalized.text,
                    reference_meanings=item.reference_meanings,
                    key_meaning_units=item.key_meaning_units,
                    target_keywords=item.target_keywords,
                    estimated_audio_seconds=item.estimated_audio_seconds,
                    content_hash=content_hash,
                    similarity_key=hashlib.sha256(key.encode("utf-8")).hexdigest(),
                    safety=item.safety,
                )
            )
        return finalized

    @staticmethod
    def _duration_range(request: ListeningSetGenerationRequest) -> tuple[float, float]:
        policy_minimum, policy_maximum = DIFFICULTY_DURATION_RANGES[
            request.set_context.difficulty.value
        ]
        minimum = request.constraints.audio_seconds_min or policy_minimum
        maximum = request.constraints.audio_seconds_max or policy_maximum
        minimum = max(minimum, policy_minimum)
        maximum = min(maximum, policy_maximum)
        if minimum > maximum:
            raise ListeningStageException(
                ListeningErrorCode.INVALID_REQUEST,
                ListeningStage.GENERATION,
                "요청 Duration 범위와 Difficulty 정책 범위가 겹치지 않습니다.",
                False,
            )
        return minimum, maximum

    @staticmethod
    def _is_similar(candidate: str, existing: list[str]) -> bool:
        if not candidate:
            return True
        return any(
            SequenceMatcher(a=candidate, b=other).ratio() >= 0.90
            for other in existing
            if other
        )

    @staticmethod
    def _validate_manual_retry(attempt: int) -> None:
        if attempt > settings.AI_LISTENING_MANUAL_RETRY_LIMIT:
            raise ListeningStageException(
                ListeningErrorCode.MANUAL_RETRY_LIMIT_EXCEEDED,
                ListeningStage.GENERATION,
                "Listening 생성 수동 재시도 가능 횟수를 초과했습니다.",
                False,
            )
