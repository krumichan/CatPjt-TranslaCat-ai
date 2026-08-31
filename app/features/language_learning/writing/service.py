from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from app.ai.ports import TextGenerationProvider
from app.core.config import settings
from app.features.language_learning.quality import (
    CONTENT_DIVERSITY_POLICY_VERSION,
    LANGUAGE_COMPLEXITY_POLICY_VERSION,
    DiversityCandidate,
    DiversityValidator,
    DiversityValidationStats,
    resolve_daily_complexity_band,
)
from app.features.language_learning.writing.policy import (
    EVALUATION_RUBRIC_VERSION,
    SCORING_POLICY_VERSION,
    calculate_overall_score,
)
from app.features.language_learning.writing.prompts import (
    DAILY_WRITING_GENERATION_PROMPT_VERSION,
    DAILY_WRITING_GENERATION_V35_PROMPT_VERSION,
    LEVEL_TEST_QUESTION_PROMPT_VERSION,
    WRITING_EVALUATION_PROMPT_VERSION,
    build_daily_writing_generation_prompt,
    build_level_test_question_prompt,
    build_writing_evaluation_prompt,
)
from app.schemas.language_learning import (
    AiWritingEvaluationPayload,
    DailyWritingGenerationRequest,
    DifficultyDistribution,
    DailyWritingGenerationResponse,
    DailyWritingItem,
    LevelTestQuestionPayload,
    LevelTestQuestionRequest,
    LevelTestQuestionResponse,
    WritingEvaluationRequest,
    WritingEvaluationResponse,
    WritingEvaluationScores,
)
from app.schemas.language_learning_quality import (
    DiversityHistoryEntry,
    DiversitySummary,
    GenerationSourceType,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DAILY_WRITING_GENERATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "order": {"type": "INTEGER"},
                    "difficulty": {
                        "type": "STRING",
                        "enum": ["REVIEW", "NORMAL", "CHALLENGE"],
                    },
                    "originText": {"type": "STRING"},
                    "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "focusMetrics": {
                        "type": "ARRAY",
                        "items": {
                            "type": "STRING",
                            "enum": [
                                "MEANING",
                                "GRAMMAR",
                                "VOCABULARY",
                                "NATURALNESS",
                                "EXPRESSION",
                            ],
                        },
                    },
                    "focusReason": {"type": "STRING"},
                    "languageComplexityBand": {"type": "INTEGER"},
                    "diversityMetadata": {
                        "type": "OBJECT",
                        "properties": {
                            "scenarioCategory": {
                                "type": "STRING",
                                "enum": [
                                    "DAILY_LIFE", "WORK", "TRAVEL", "SHOPPING",
                                    "FOOD", "SERVICE", "LEARNING", "HOBBY",
                                    "DIGITAL_LIFE", "SOCIAL", "SCHEDULE", "HEALTH_GENERAL"
                                ],
                            },
                            "communicativeIntent": {
                                "type": "STRING",
                                "enum": [
                                    "DESCRIBE", "REQUEST", "CONFIRM", "REPORT", "SUGGEST",
                                    "DECLINE", "APOLOGIZE", "COMPARE", "EXPLAIN_REASON",
                                    "ASK_INFORMATION", "GIVE_INSTRUCTION", "EXPRESS_PREFERENCE",
                                    "SUMMARIZE"
                                ],
                            },
                            "taskArchetype": {"type": "STRING"},
                            "grammarFocusCodes": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "lexicalFocusCodes": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "semanticSummary": {"type": "STRING"},
                            "requiresBackgroundKnowledge": {"type": "BOOLEAN"},
                        },
                        "required": [
                            "scenarioCategory", "communicativeIntent", "taskArchetype",
                            "grammarFocusCodes", "lexicalFocusCodes", "semanticSummary",
                            "requiresBackgroundKnowledge"
                        ],
                    },
                },
                "required": [
                    "order",
                    "difficulty",
                    "originText",
                    "keywords",
                    "focusMetrics",
                    "focusReason",
                ],
            },
        }
    },
    "required": ["items"],
}

_BILINGUAL_MESSAGE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "originText": {"type": "STRING"},
        "learningText": {"type": "STRING"},
    },
    "required": ["originText", "learningText"],
}

_WRITING_EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "scores": {
            "type": "OBJECT",
            "properties": {
                "meaning": {"type": "NUMBER"},
                "grammar": {"type": "NUMBER"},
                "vocabulary": {"type": "NUMBER"},
                "naturalness": {"type": "NUMBER"},
                "expression": {"type": "NUMBER"},
            },
            "required": [
                "meaning",
                "grammar",
                "vocabulary",
                "naturalness",
                "expression",
            ],
        },
        "strengths": {"type": "ARRAY", "items": _BILINGUAL_MESSAGE_SCHEMA},
        "weaknesses": {"type": "ARRAY", "items": _BILINGUAL_MESSAGE_SCHEMA},
        "corrections": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "original": {"type": "STRING"},
                    "corrected": {"type": "STRING"},
                    "category": {"type": "STRING"},
                    "explanation": _BILINGUAL_MESSAGE_SCHEMA,
                },
                "required": ["original", "corrected", "category", "explanation"],
            },
        },
        "recommendedAnswers": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "minItems": 2,
            "maxItems": 3,
        },
        "explanation": _BILINGUAL_MESSAGE_SCHEMA,
        "profileSignals": {
            "type": "OBJECT",
            "properties": {
                "strengthTags": {"type": "ARRAY", "items": {"type": "STRING"}},
                "weaknessTags": {"type": "ARRAY", "items": {"type": "STRING"}},
                "grammarPatterns": {"type": "ARRAY", "items": {"type": "STRING"}},
                "vocabularyPatterns": {"type": "ARRAY", "items": {"type": "STRING"}},
                "naturalnessPatterns": {"type": "ARRAY", "items": {"type": "STRING"}},
                "expressionPatterns": {"type": "ARRAY", "items": {"type": "STRING"}},
                "meaningPatterns": {"type": "ARRAY", "items": {"type": "STRING"}},
                "recommendedFocus": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": [
                "strengthTags",
                "weaknessTags",
                "grammarPatterns",
                "vocabularyPatterns",
                "naturalnessPatterns",
                "expressionPatterns",
                "meaningPatterns",
                "recommendedFocus",
            ],
        },
    },
    "required": [
        "scores",
        "strengths",
        "weaknesses",
        "corrections",
        "recommendedAnswers",
        "explanation",
        "profileSignals",
    ],
}

_LEVEL_TEST_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "difficulty": {
            "type": "STRING",
            "enum": ["EASY", "NORMAL", "CHALLENGE"],
        },
        "originText": {"type": "STRING"},
        "focusMetrics": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": [
                    "MEANING",
                    "GRAMMAR",
                    "VOCABULARY",
                    "NATURALNESS",
                    "EXPRESSION",
                ],
            },
        },
        "focusReason": {"type": "STRING"},
    },
    "required": ["difficulty", "originText", "focusMetrics", "focusReason"],
}


class _DailyWritingPayload(BaseModel):
    items: list[DailyWritingItem]


class LanguageLearningWritingService:
    def __init__(
        self,
        provider: TextGenerationProvider,
        generation_timeout_seconds: float | None = None,
        evaluation_timeout_seconds: float | None = None,
        level_test_timeout_seconds: float | None = None,
        generation_max_retries: int | None = None,
        evaluation_max_retries: int | None = None,
    ) -> None:
        self.provider = provider
        self.generation_timeout_seconds = (
            generation_timeout_seconds
            if generation_timeout_seconds is not None
            else settings.AI_LANGUAGE_LEARNING_GENERATION_TIMEOUT_SECONDS
        )
        self.evaluation_timeout_seconds = (
            evaluation_timeout_seconds
            if evaluation_timeout_seconds is not None
            else settings.AI_LANGUAGE_LEARNING_EVALUATION_TIMEOUT_SECONDS
        )
        self.level_test_timeout_seconds = (
            level_test_timeout_seconds
            if level_test_timeout_seconds is not None
            else settings.AI_LANGUAGE_LEARNING_LEVEL_TEST_TIMEOUT_SECONDS
        )
        self.generation_max_retries = (
            generation_max_retries
            if generation_max_retries is not None
            else settings.AI_LANGUAGE_LEARNING_GENERATION_MAX_RETRIES
        )
        self.evaluation_max_retries = (
            evaluation_max_retries
            if evaluation_max_retries is not None
            else settings.AI_LANGUAGE_LEARNING_EVALUATION_MAX_RETRIES
        )

    async def generate_daily(
        self,
        request: DailyWritingGenerationRequest,
    ) -> DailyWritingGenerationResponse:
        if (
            request.content_diversity_policy_version
            == CONTENT_DIVERSITY_POLICY_VERSION
        ):
            return await self._generate_daily_v35(request)

        prompt = build_daily_writing_generation_prompt(request)

        payload = await self._call_and_validate(
            request_id=request.request_id,
            operation="daily writing generation",
            type_name="LANGUAGE_LEARNING_DAILY_WRITING_GENERATION",
            prompt=prompt,
            schema=_DAILY_WRITING_GENERATION_SCHEMA,
            model_type=_DailyWritingPayload,
            timeout_seconds=self.generation_timeout_seconds,
            max_retries=self.generation_max_retries,
            post_validate=lambda value: self._validate_daily_items(request, value.items),
        )

        return DailyWritingGenerationResponse(
            request_id=request.request_id,
            prompt_version=DAILY_WRITING_GENERATION_PROMPT_VERSION,
            items=payload.items,
        )

    async def _generate_daily_v35(
        self,
        request: DailyWritingGenerationRequest,
    ) -> DailyWritingGenerationResponse:
        remaining = {
            "REVIEW": request.difficulty_distribution.review,
            "NORMAL": request.difficulty_distribution.normal,
            "CHALLENGE": request.difficulty_distribution.challenge,
        }
        accepted: list[DailyWritingItem] = []
        accepted_diversity: list[DiversityCandidate] = []
        all_candidates: list[DailyWritingItem] = []
        stats = DiversityValidationStats()

        for provider_attempt in range(3):
            missing_total = sum(remaining.values())
            if missing_total <= 0:
                break
            candidate_distribution = self._expanded_distribution(remaining)
            current_history = list(request.diversity_context.current_session)
            current_history.extend(
                DiversityHistoryEntry(
                    source_type=GenerationSourceType.WRITING,
                    content=item.origin_text,
                    content_hash=(
                        item.diversity_metadata.content_hash
                        if item.diversity_metadata
                        else None
                    ),
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
                for item in accepted
            )
            candidate_request = request.model_copy(
                deep=True,
                update={
                    "sentence_count": candidate_distribution.total,
                    "difficulty_distribution": candidate_distribution,
                    "diversity_context": request.diversity_context.model_copy(
                        deep=True,
                        update={"current_session": current_history},
                    ),
                },
            )
            payload = await self._call_daily_candidate_batch(
                candidate_request,
                provider_attempt=provider_attempt,
            )
            all_candidates.extend(payload.items)
            validator = DiversityValidator(request.diversity_context)
            for candidate in payload.items:
                difficulty = candidate.difficulty.value
                if remaining.get(difficulty, 0) <= 0:
                    continue
                if not self._valid_phase35_daily_candidate(request, candidate):
                    continue
                assert candidate.diversity_metadata is not None
                decision = validator.validate(
                    DiversityCandidate(candidate.origin_text, candidate.diversity_metadata),
                    accepted_diversity,
                )
                stats.record(decision)
                if not decision.accepted:
                    continue
                finalized = candidate.model_copy(
                    deep=True,
                    update={"diversity_metadata": decision.metadata},
                )
                accepted.append(finalized)
                accepted_diversity.append(
                    DiversityCandidate(finalized.origin_text, decision.metadata)
                )
                remaining[difficulty] -= 1

        fallback_used = False
        if sum(remaining.values()) > 0:
            fallback_used = True
            validator = DiversityValidator(
                request.diversity_context,
                relaxed_history=True,
            )
            for candidate in all_candidates:
                difficulty = candidate.difficulty.value
                if remaining.get(difficulty, 0) <= 0:
                    continue
                if not self._valid_phase35_daily_candidate(request, candidate):
                    continue
                assert candidate.diversity_metadata is not None
                decision = validator.validate(
                    DiversityCandidate(candidate.origin_text, candidate.diversity_metadata),
                    accepted_diversity,
                )
                if not decision.accepted:
                    continue
                finalized = candidate.model_copy(
                    deep=True,
                    update={"diversity_metadata": decision.metadata},
                )
                accepted.append(finalized)
                accepted_diversity.append(
                    DiversityCandidate(finalized.origin_text, decision.metadata)
                )
                remaining[difficulty] -= 1
                if sum(remaining.values()) <= 0:
                    break

        if sum(remaining.values()) > 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "CONTENT_DIVERSITY_EXHAUSTED",
                    "retryable": False,
                    "message": "Daily Writing 중복 방지 기준을 만족하는 문항이 부족합니다.",
                },
            )

        ordered = [
            item.model_copy(update={"order": index})
            for index, item in enumerate(accepted[: request.sentence_count], start=1)
        ]
        self._validate_daily_items(request, ordered)
        return DailyWritingGenerationResponse(
            request_id=request.request_id,
            prompt_version=DAILY_WRITING_GENERATION_V35_PROMPT_VERSION,
            items=ordered,
            content_diversity_policy_version=CONTENT_DIVERSITY_POLICY_VERSION,
            language_complexity_policy_version=LANGUAGE_COMPLEXITY_POLICY_VERSION,
            diversity_summary=DiversitySummary(
                policy_version=CONTENT_DIVERSITY_POLICY_VERSION,
                candidate_count=stats.candidate_count,
                accepted_count=len(ordered),
                rejected_exact=stats.rejected_exact,
                rejected_similarity=stats.rejected_similarity,
                rejected_structural=stats.rejected_structural,
                rejected_background_knowledge=stats.rejected_background_knowledge,
                fallback_used=fallback_used,
            ),
        )

    async def _call_daily_candidate_batch(
        self,
        request: DailyWritingGenerationRequest,
        *,
        provider_attempt: int,
    ) -> _DailyWritingPayload:
        prompt = build_daily_writing_generation_prompt(request)
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.provider.call(
                    type_name="LANGUAGE_LEARNING_DAILY_WRITING_GENERATION",
                    data=prompt,
                    schema=_DAILY_WRITING_GENERATION_SCHEMA,
                ),
                timeout=self.generation_timeout_seconds,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise HTTPException(
                status_code=504,
                detail="daily writing generation 시간이 초과되었습니다.",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in {429, 500, 502, 503, 504} and provider_attempt < 2:
                logger.warning(
                    "Phase 3.5 daily writing transient provider failure. request_id=%s attempt=%d/3",
                    request.request_id,
                    provider_attempt + 1,
                )
                return _DailyWritingPayload(items=[])
            raise HTTPException(
                status_code=502,
                detail="daily writing generation에 실패했습니다.",
            ) from exc
        if not isinstance(result, dict):
            raise HTTPException(status_code=502, detail="daily writing generation 응답 Schema가 유효하지 않습니다.")
        try:
            payload = _DailyWritingPayload.model_validate(result)
        except ValidationError as exc:
            raise HTTPException(
                status_code=502,
                detail="daily writing generation 응답 Schema가 유효하지 않습니다.",
            ) from exc
        logger.info(
            "Phase 3.5 daily writing candidate batch completed. request_id=%s attempt=%d/3 candidates=%d latency_ms=%d",
            request.request_id,
            provider_attempt + 1,
            len(payload.items),
            int((time.perf_counter() - started) * 1000),
        )
        return payload

    @staticmethod
    def _expanded_distribution(remaining: dict[str, int]) -> DifficultyDistribution:
        counts = {
            "REVIEW": max(0, remaining.get("REVIEW", 0)),
            "NORMAL": max(0, remaining.get("NORMAL", 0)),
            "CHALLENGE": max(0, remaining.get("CHALLENGE", 0)),
        }
        missing_total = sum(counts.values())
        if missing_total <= 0:
            return DifficultyDistribution(review=0, normal=0, challenge=0)

        pool_total = min(missing_total * 2, 40)
        extra_total = pool_total - missing_total
        allocation = counts.copy()
        if extra_total > 0:
            raw_extra = {
                key: extra_total * value / missing_total
                for key, value in counts.items()
            }
            for key, raw in raw_extra.items():
                allocation[key] += int(raw)
            remainder = pool_total - sum(allocation.values())
            for key in sorted(
                counts,
                key=lambda item: (
                    raw_extra[item] - int(raw_extra[item]),
                    counts[item],
                ),
                reverse=True,
            ):
                if remainder <= 0:
                    break
                if counts[key] <= 0:
                    continue
                allocation[key] += 1
                remainder -= 1

        return DifficultyDistribution(
            review=allocation["REVIEW"],
            normal=allocation["NORMAL"],
            challenge=allocation["CHALLENGE"],
        )

    @staticmethod
    def _valid_phase35_daily_candidate(
        request: DailyWritingGenerationRequest,
        item: DailyWritingItem,
    ) -> bool:
        if item.diversity_metadata is None or item.language_complexity_band is None:
            return False
        expected_band = resolve_daily_complexity_band(
            item.difficulty.value,
            request.language_complexity,
        )
        return item.language_complexity_band == expected_band

    async def evaluate(
        self,
        request: WritingEvaluationRequest,
    ) -> WritingEvaluationResponse:
        prompt = build_writing_evaluation_prompt(request)

        payload = await self._call_and_validate(
            request_id=request.request_id,
            operation="writing evaluation",
            type_name="LANGUAGE_LEARNING_WRITING_EVALUATION",
            prompt=prompt,
            schema=_WRITING_EVALUATION_SCHEMA,
            model_type=AiWritingEvaluationPayload,
            timeout_seconds=self.evaluation_timeout_seconds,
            max_retries=self.evaluation_max_retries,
        )

        raw = payload.scores
        overall = calculate_overall_score(
            meaning=raw.meaning,
            grammar=raw.grammar,
            vocabulary=raw.vocabulary,
            naturalness=raw.naturalness,
            expression=raw.expression,
        )

        return WritingEvaluationResponse(
            request_id=request.request_id,
            scores=WritingEvaluationScores(
                overall=overall,
                meaning=self._round_score(raw.meaning),
                grammar=self._round_score(raw.grammar),
                vocabulary=self._round_score(raw.vocabulary),
                naturalness=self._round_score(raw.naturalness),
                expression=self._round_score(raw.expression),
            ),
            strengths=payload.strengths,
            weaknesses=payload.weaknesses,
            corrections=payload.corrections,
            recommended_answers=payload.recommended_answers,
            explanation=payload.explanation,
            profile_signals=payload.profile_signals,
            evaluation_rubric_version=EVALUATION_RUBRIC_VERSION,
            scoring_policy_version=SCORING_POLICY_VERSION,
            prompt_version=WRITING_EVALUATION_PROMPT_VERSION,
        )

    async def generate_level_test_question(
        self,
        request: LevelTestQuestionRequest,
    ) -> LevelTestQuestionResponse:
        prompt = build_level_test_question_prompt(request)

        payload = await self._call_and_validate(
            request_id=request.request_id,
            operation="level test question generation",
            type_name="LANGUAGE_LEARNING_LEVEL_TEST_QUESTION",
            prompt=prompt,
            schema=_LEVEL_TEST_QUESTION_SCHEMA,
            model_type=LevelTestQuestionPayload,
            timeout_seconds=self.level_test_timeout_seconds,
            max_retries=self.generation_max_retries,
        )

        return LevelTestQuestionResponse(
            request_id=request.request_id,
            question_number=request.question_number,
            total_questions=request.total_questions,
            difficulty=payload.difficulty,
            origin_text=payload.origin_text,
            focus_metrics=payload.focus_metrics,
            focus_reason=payload.focus_reason,
            prompt_version=LEVEL_TEST_QUESTION_PROMPT_VERSION,
        )

    async def _call_and_validate(
        self,
        *,
        request_id: str,
        operation: str,
        type_name: str,
        prompt: str,
        schema: dict[str, Any],
        model_type: type[T],
        timeout_seconds: float,
        max_retries: int,
        post_validate: Callable[[T], None] | None = None,
    ) -> T:
        last_timeout = False
        last_error: Exception | None = None
        total_attempts = max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                result = await asyncio.wait_for(
                    self.provider.call(
                        type_name=type_name,
                        data=prompt,
                        schema=schema,
                    ),
                    timeout=timeout_seconds,
                )
                if not isinstance(result, dict):
                    raise ValueError("AI structured response must be an object")

                parsed = model_type.model_validate(result)
                if post_validate is not None:
                    post_validate(parsed)
                return parsed
            except TimeoutError as exc:
                last_timeout = True
                last_error = exc
                logger.warning(
                    "%s timed out. request_id=%s attempt=%s/%s",
                    operation,
                    request_id,
                    attempt,
                    total_attempts,
                )
            except (ValidationError, ValueError) as exc:
                last_timeout = False
                last_error = exc
                logger.warning(
                    "%s returned invalid schema. request_id=%s attempt=%s/%s error=%s",
                    operation,
                    request_id,
                    attempt,
                    total_attempts,
                    exc,
                )
            except HTTPException:
                raise
            except Exception as exc:
                last_timeout = False
                last_error = exc
                logger.exception(
                    "%s failed. request_id=%s attempt=%s/%s",
                    operation,
                    request_id,
                    attempt,
                    total_attempts,
                )

        if last_timeout:
            raise HTTPException(
                status_code=504,
                detail=f"{operation} 시간이 초과되었습니다.",
            ) from last_error

        raise HTTPException(
            status_code=502,
            detail=f"{operation}에 실패했습니다.",
        ) from last_error

    @staticmethod
    def _validate_daily_items(
        request: DailyWritingGenerationRequest,
        items: list[DailyWritingItem],
    ) -> None:
        if len(items) != request.sentence_count:
            raise ValueError(
                "AI가 요청한 Daily Writing 문장 수를 반환하지 않았습니다."
            )

        orders = sorted(item.order for item in items)
        if orders != list(range(1, request.sentence_count + 1)):
            raise ValueError("Daily Writing order가 연속적이지 않습니다.")

        expected = request.difficulty_distribution
        actual = {
            "REVIEW": sum(item.difficulty.value == "REVIEW" for item in items),
            "NORMAL": sum(item.difficulty.value == "NORMAL" for item in items),
            "CHALLENGE": sum(item.difficulty.value == "CHALLENGE" for item in items),
        }
        if actual != {
            "REVIEW": expected.review,
            "NORMAL": expected.normal,
            "CHALLENGE": expected.challenge,
        }:
            raise ValueError("AI가 요청한 난이도 분배를 준수하지 않았습니다.")

    @staticmethod
    def _round_score(value: float) -> int:
        return max(0, min(100, int(value + 0.5)))
