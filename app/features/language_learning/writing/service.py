from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from app.ai.ports import TextGenerationProvider
from app.core.config import settings
from app.features.language_learning.writing.generation import VerifiedWritingGenerator
from app.features.language_learning.writing.policy import (
    EVALUATION_RUBRIC_VERSION,
    SCORING_POLICY_VERSION,
    calculate_overall_score,
)
from app.features.language_learning.writing.prompts import (
    LEVEL_TEST_QUESTION_PROMPT_VERSION,
    WRITING_EVALUATION_PROMPT_VERSION,
    build_level_test_question_prompt,
    build_writing_evaluation_prompt,
)
from app.schemas.language_learning import (
    AiWritingEvaluationPayload,
    DailyWritingGenerationRequest,
    DailyWritingGenerationResponse,
    LevelTestQuestionPayload,
    LevelTestQuestionRequest,
    LevelTestQuestionResponse,
    WritingEvaluationRequest,
    WritingEvaluationResponse,
    WritingEvaluationScores,
)


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

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


class LanguageLearningWritingService:
    def __init__(
        self,
        provider: TextGenerationProvider,
        generation_timeout_seconds: float | None = None,
        evaluation_timeout_seconds: float | None = None,
        level_test_timeout_seconds: float | None = None,
        generation_max_retries: int | None = None,
        evaluation_max_retries: int | None = None,
        verification_timeout_seconds: float | None = None,
        generation_total_timeout_seconds: float | None = None,
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

        self.verification_timeout_seconds = (
            verification_timeout_seconds if verification_timeout_seconds is not None
            else settings.AI_WRITING_VERIFICATION_TIMEOUT_SECONDS
        )
        self.generation_total_timeout_seconds = (
            generation_total_timeout_seconds if generation_total_timeout_seconds is not None
            else settings.AI_WRITING_GENERATION_TOTAL_TIMEOUT_SECONDS
        )

    async def generate_daily(
        self,
        request: DailyWritingGenerationRequest,
    ) -> DailyWritingGenerationResponse:
        generator = VerifiedWritingGenerator(
            self.provider,
            generation_timeout_seconds=self.generation_timeout_seconds,
            verification_timeout_seconds=self.verification_timeout_seconds,
            total_timeout_seconds=self.generation_total_timeout_seconds,
            max_retries=self.generation_max_retries,
        )
        return await generator.generate(request)

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
    def _round_score(value: float) -> int:
        return max(0, min(100, int(value + 0.5)))
