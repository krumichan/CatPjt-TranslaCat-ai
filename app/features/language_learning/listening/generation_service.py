from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

from pydantic import ValidationError

from app.ai.ports import StructuredTextGenerationProvider
from app.common.idempotency import InMemoryIdempotencyStore
from app.core.config import settings
from app.features.language_learning.listening.difficulty_adapter import (
    validate_listening_candidate,
)
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
from app.features.language_learning.quality import (
    CONTENT_DIVERSITY_POLICY_VERSION,
    LANGUAGE_COMPLEXITY_POLICY_VERSION,
    DiversityCandidate,
    DiversityValidator,
    DiversityValidationStats,
)
from app.features.language_learning.listening.prompts import build_generation_prompt
from app.features.language_learning.listening.provider_error import (
    map_provider_exception,
)
from app.schemas.language_learning_listening import (
    GeneratedListeningItemPayload,
    ListeningErrorCode,
    ListeningGenerationPayload,
    ListeningItem,
    ListeningSetGenerationRequest,
    ListeningSetGenerationResponse,
    ListeningStage,
    ListeningUsage,
    StageUsage,
)
from app.schemas.language_learning_quality import (
    DiversityContext,
    DiversityHistoryEntry,
    DiversitySummary,
    GenerationSourceType,
)


logger = logging.getLogger(__name__)

_PROVIDER_PAYLOAD_LOG_LIMIT = 4000


def _provider_payload_preview(data: object) -> str:
    """Return a bounded JSON-ish preview without logging the request prompt."""
    try:
        rendered = json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(data)
    if len(rendered) <= _PROVIDER_PAYLOAD_LOG_LIMIT:
        return rendered
    return f"{rendered[:_PROVIDER_PAYLOAD_LOG_LIMIT]}...<truncated>"


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
        logger.info(
            "Listening generation request received. request_id=%s origin=%s learning=%s "
            "item_count=%d difficulty=%s manual_retry_attempt=%d policy_version=%s "
            "model_config_version=%s",
            request.request_id,
            request.user_context.origin_language,
            request.user_context.learning_language,
            request.set_context.item_count,
            request.set_context.difficulty.value,
            request.manual_retry_attempt,
            request.policy_version,
            request.model_config_version,
        )
        self._validate_manual_retry(request.manual_retry_attempt)
        key = "|".join(
            [
                request.idempotency_key,
                request.policy_version,
                request.model_config_version,
            ]
        )
        response, cache_hit = await self.idempotency_store.execute(
            key,
            lambda: self._generate_diverse(request),
        )
        logger.info(
            "Listening generation request completed. request_id=%s cache_hit=%s items=%d",
            request.request_id,
            cache_hit,
            len(response.items),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _generate_diverse(
        self,
        request: ListeningSetGenerationRequest,
    ) -> ListeningSetGenerationResponse:
        expected_count = request.set_context.item_count
        effective_context = self._effective_diversity_context(request)
        accepted_items: list[ListeningItem] = []
        accepted_diversity: list[DiversityCandidate] = []
        all_candidates = []
        stats = DiversityValidationStats()
        total_latency_ms = 0
        total_input_tokens = 0
        total_output_tokens = 0
        last_provider = None
        last_model = None

        for provider_attempt in range(3):
            missing = expected_count - len(accepted_items)
            if missing <= 0:
                break
            pool_size = max(missing, min(missing * 2, 40))
            current_session = list(effective_context.current_session)
            current_session.extend(
                DiversityHistoryEntry(
                    source_type=GenerationSourceType.LISTENING,
                    content=item.source_text,
                    content_hash=item.content_hash,
                    scenario_category=(
                        item.diversity_metadata.scenario_category
                        if item.diversity_metadata
                        else None
                    ),
                    communicative_intent=(
                        item.diversity_metadata.communicative_intent
                        if item.diversity_metadata
                        else None
                    ),
                    task_archetype=(
                        item.diversity_metadata.task_archetype
                        if item.diversity_metadata
                        else None
                    ),
                    grammar_focus_codes=(
                        item.diversity_metadata.grammar_focus_codes
                        if item.diversity_metadata
                        else []
                    ),
                    semantic_summary=(
                        item.diversity_metadata.semantic_summary
                        if item.diversity_metadata
                        else None
                    ),
                    age_days=0,
                )
                for item in accepted_items
            )
            candidate_request = request.model_copy(
                deep=True,
                update={
                    "set_context": request.set_context.model_copy(
                        deep=True,
                        update={"item_count": pool_size},
                    ),
                    "diversity_context": effective_context.model_copy(
                        deep=True,
                        update={"current_session": current_session},
                    ),
                },
            )
            prompt = build_generation_prompt(candidate_request)
            schema = ListeningGenerationPayload.model_json_schema()
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    self.provider.call_with_metadata(
                        type_name=self.TYPE_NAME,
                        data=prompt,
                        schema=schema,
                    ),
                    timeout=self.timeout_seconds,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                if provider_attempt < 2:
                    continue
                raise ListeningStageException(
                    ListeningErrorCode.PROVIDER_TIMEOUT,
                    ListeningStage.GENERATION,
                    "Listening 문항 생성 Provider 응답 시간이 초과되었습니다.",
                    True,
                ) from exc
            except ListeningStageException:
                raise
            except Exception as exc:
                mapped = map_provider_exception(
                    exc,
                    stage=ListeningStage.GENERATION,
                    fallback_code=ListeningErrorCode.GENERATION_FAILED,
                    fallback_message="Listening 문항 생성에 실패했습니다.",
                )
                if mapped.retryable and provider_attempt < 2:
                    continue
                raise mapped from exc

            total_latency_ms += int((time.perf_counter() - started) * 1000)
            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens
            last_provider = result.provider
            last_model = result.model
            if not isinstance(result.data, dict):
                raise ListeningStageException(
                    ListeningErrorCode.INVALID_RESPONSE_SCHEMA,
                    ListeningStage.GENERATION,
                    "Listening 문항 생성 응답 Schema가 유효하지 않습니다.",
                    False,
                )
            candidates, _ = self._salvage_candidates(
                result.data,
                request_id=request.request_id,
                provider_attempt=provider_attempt + 1,
            )
            if not candidates:
                if provider_attempt < 2:
                    continue
                raise ListeningStageException(
                    ListeningErrorCode.INVALID_RESPONSE_SCHEMA,
                    ListeningStage.GENERATION,
                    "Listening 문항 생성 응답 Schema가 유효하지 않습니다.",
                    False,
                )

            validator = DiversityValidator(effective_context)
            for candidate in candidates:
                all_candidates.append(candidate)
                finalized = self._finalize_candidate(request, candidate)
                if finalized is None:
                    continue
                assert candidate.diversity_metadata is not None
                decision = validator.validate(
                    DiversityCandidate(finalized.source_text, candidate.diversity_metadata),
                    accepted_diversity,
                )
                stats.record(decision)
                if not decision.accepted:
                    continue
                metadata = decision.metadata.model_copy(
                    update={
                        "content_hash": finalized.content_hash,
                        "similarity_key": finalized.similarity_key,
                    }
                )
                finalized = finalized.model_copy(update={"diversity_metadata": metadata})
                accepted_items.append(finalized)
                accepted_diversity.append(DiversityCandidate(finalized.source_text, metadata))
                if len(accepted_items) >= expected_count:
                    break

        fallback_used = False
        if len(accepted_items) < expected_count:
            fallback_used = True
            validator = DiversityValidator(effective_context, relaxed_history=True)
            for candidate in all_candidates:
                finalized = self._finalize_candidate(request, candidate)
                if finalized is None or candidate.diversity_metadata is None:
                    continue
                if any(item.content_hash == finalized.content_hash for item in accepted_items):
                    continue
                decision = validator.validate(
                    DiversityCandidate(finalized.source_text, candidate.diversity_metadata),
                    accepted_diversity,
                )
                if not decision.accepted:
                    continue
                metadata = decision.metadata.model_copy(
                    update={
                        "content_hash": finalized.content_hash,
                        "similarity_key": finalized.similarity_key,
                    }
                )
                accepted_items.append(finalized.model_copy(update={"diversity_metadata": metadata}))
                accepted_diversity.append(DiversityCandidate(finalized.source_text, metadata))
                if len(accepted_items) >= expected_count:
                    break

        if len(accepted_items) < expected_count:
            raise ListeningStageException(
                ListeningErrorCode.CONTENT_DIVERSITY_EXHAUSTED,
                ListeningStage.GENERATION,
                "Listening 중복 방지 기준을 만족하는 문항이 부족합니다.",
                False,
            )

        finalized_items = [
            item.model_copy(update={"item_index": index})
            for index, item in enumerate(accepted_items[:expected_count], start=1)
        ]
        return ListeningSetGenerationResponse(
            request_id=request.request_id,
            generation_version=LISTENING_GENERATION_VERSION,
            policy_version=request.policy_version,
            model_config_version=request.model_config_version,
            items=finalized_items,
            usage=ListeningUsage(
                generation=StageUsage(
                    latency_ms=total_latency_ms,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    provider=last_provider,
                    model=last_model,
                    prompt_version=LISTENING_GENERATION_PROMPT_VERSION,
                )
            ),
            content_diversity_policy_version=CONTENT_DIVERSITY_POLICY_VERSION,
            language_complexity_policy_version=LANGUAGE_COMPLEXITY_POLICY_VERSION,
            diversity_summary=DiversitySummary(
                policy_version=CONTENT_DIVERSITY_POLICY_VERSION,
                candidate_count=stats.candidate_count,
                accepted_count=len(finalized_items),
                rejected_exact=stats.rejected_exact,
                rejected_similarity=stats.rejected_similarity,
                rejected_structural=stats.rejected_structural,
                rejected_background_knowledge=stats.rejected_background_knowledge,
                fallback_used=fallback_used,
            ),
        )

    @classmethod
    def _salvage_candidates(
        cls,
        data: dict,
        *,
        request_id: str,
        provider_attempt: int,
    ) -> tuple[list[GeneratedListeningItemPayload], list[str]]:
        raw_items = data.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            logger.warning(
                "Listening candidate schema salvage. request_id=%s "
                "attempt=%d/3 validCandidates=0 rejectedCandidates=1 "
                "reasons=%s",
                request_id,
                provider_attempt,
                ["items:not-a-non-empty-list"],
            )
            return [], ["items:not-a-non-empty-list"]

        valid: list[GeneratedListeningItemPayload] = []
        reasons: list[str] = []
        for index, raw_candidate in enumerate(raw_items):
            if not isinstance(raw_candidate, dict):
                reasons.append(f"items.{index}:candidate:not-an-object")
                continue
            candidate = dict(raw_candidate)
            if "comprehensionFocus" in candidate:
                candidate["comprehensionFocus"] = cls._normalize_comprehension_focus(
                    candidate.get("comprehensionFocus")
                )
            try:
                valid.append(GeneratedListeningItemPayload.model_validate(candidate))
            except ValidationError as exc:
                reasons.extend(
                    cls._validation_reasons(index, exc, raw_candidate)
                )

        if reasons:
            logger.info(
                "Listening candidate schema salvage. request_id=%s "
                "attempt=%d/3 validCandidates=%d rejectedCandidates=%d reasons=%s",
                request_id,
                provider_attempt,
                len(valid),
                len(raw_items) - len(valid),
                reasons[:12],
            )
        return valid, reasons

    @staticmethod
    def _normalize_comprehension_focus(value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        if normalized in {"GIST", "DETAIL", "INTENT", "INFERENCE", "NEXT_ACTION"}:
            return normalized
        return value

    @staticmethod
    def _validation_reasons(
        item_index: int,
        exc: ValidationError,
        raw_candidate: dict,
    ) -> list[str]:
        reasons: list[str] = []
        for error in exc.errors(include_url=False, include_input=False):
            location = ".".join(str(part) for part in error.get("loc", ())) or "candidate"
            reason = f"items.{item_index}:{location}:{error.get('type', 'validation_error')}"
            if location in {"comprehension_focus", "comprehensionFocus"}:
                value = raw_candidate.get("comprehensionFocus")
                if value is not None:
                    reason += f":value={str(value)[:40]}"
            reasons.append(reason)
        return reasons

    def _finalize_candidate(
        self,
        request: ListeningSetGenerationRequest,
        item,
    ) -> ListeningItem | None:
        _, validation = validate_listening_candidate(
            request,
            item,
            duration_range_resolver=lambda: self._duration_range(request),
            mode_payload_validator=lambda: self._valid_mode_payload(request, item),
        )
        if not validation.passed:
            return None
        normalized = normalize_text(item.source_text, request.user_context.learning_language)
        content_hash = hashlib.sha256(normalized.text.encode("utf-8")).hexdigest()
        key = similarity_key(item.source_text, request.user_context.learning_language)
        return ListeningItem(
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
            language_complexity_band=item.language_complexity_band,
            diversity_metadata=item.diversity_metadata,
            question=item.question,
            options=item.options,
            correct_option_key=item.correct_option_key,
            comprehension_focus=item.comprehension_focus,
            summary_key_points=item.summary_key_points,
        )

    @staticmethod
    def _effective_diversity_context(
        request: ListeningSetGenerationRequest,
    ) -> DiversityContext:
        exact = list(request.diversity_context.exact_content_hashes_90d)
        exact.extend(request.constraints.recent_content_hashes)
        same_feature = list(request.diversity_context.same_feature_recent)
        same_feature.extend(
            DiversityHistoryEntry(
                source_type=GenerationSourceType.LISTENING,
                content=summary,
            )
            for summary in request.constraints.recent_similarity_summaries
            if summary.strip()
        )
        return request.diversity_context.model_copy(
            deep=True,
            update={
                "exact_content_hashes_90d": list(dict.fromkeys(exact))[:200],
                "same_feature_recent": same_feature[:80],
            },
        )

    @staticmethod
    def _valid_mode_payload(request: ListeningSetGenerationRequest, item) -> bool:
        mode = request.set_context.learning_mode.value
        if mode == "DICTATION":
            return (
                item.question is None
                and not item.options
                and item.correct_option_key is None
                and item.comprehension_focus is None
                and not item.summary_key_points
            )
        if mode == "COMPREHENSION":
            keys = [option.key for option in item.options]
            return (
                bool(item.question and item.question.strip())
                and len(item.options) == 4
                and set(keys) == {"A", "B", "C", "D"}
                and item.correct_option_key in set(keys)
                and item.comprehension_focus in {
                    "GIST", "DETAIL", "INTENT", "INFERENCE", "NEXT_ACTION"
                }
                and not item.summary_key_points
            )
        if mode == "SUMMARY":
            points = [point.strip() for point in item.summary_key_points if point.strip()]
            return (
                2 <= len(points) <= 6
                and item.question is None
                and not item.options
                and item.correct_option_key is None
                and item.comprehension_focus is None
            )
        return False

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
    def _validate_manual_retry(attempt: int) -> None:
        if attempt > settings.AI_LISTENING_MANUAL_RETRY_LIMIT:
            raise ListeningStageException(
                ListeningErrorCode.MANUAL_RETRY_LIMIT_EXCEEDED,
                ListeningStage.GENERATION,
                "Listening 생성 수동 재시도 가능 횟수를 초과했습니다.",
                False,
            )
