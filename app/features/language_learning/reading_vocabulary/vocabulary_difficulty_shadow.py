"""Awaited, sampled, fail-open Vocabulary difficulty-only observation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ai.ports import StructuredGenerationResult, TextGenerationProvider
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_adapter import (
    VOCABULARY_DIFFICULTY_ISSUE_CODES,
    VocabularySemanticDifficultyPolicy,
    normalize_vocabulary_semantic_difficulty_assessment,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_recipe import (
    VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION,
    usage_intent_for_skill,
    vocabulary_blind_rubric_payload,
    vocabulary_difficulty_recipe_version,
    vocabulary_task_subtype,
)
from app.schemas.language_learning_practice import (
    PracticeDomain,
    PracticeGeneratedQuestion,
    PracticeGenerationRequest,
    VocabularyMode,
)


logger = logging.getLogger(__name__)

VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME = (
    "LANGUAGE_LEARNING_VOCABULARY_DIFFICULTY_SHADOW"
)
VOCABULARY_DIFFICULTY_SHADOW_SYSTEM_PROMPT = r"""
You are a blind difficulty classifier for TranslaCat Vocabulary practice.

Classify cognitive and linguistic demand only from learner-visible prompt/options and the supplied
mode-specific measurement rubric. You do not receive the requested band, planned difficulty label,
correct answer, hidden targetExpression, canonicalKey, or review identity. Never infer that hidden metadata.

Difficulty comes from semantic precision, contextual/register/collocational/pragmatic cue integration,
or composition constraints. Do not reward word length, rarity, trivia, weak distractors, unsupported answers,
or ambiguity. If the task lacks one supported best answer, use NOT_ASSESSED rather than calling ambiguity hard.
For each item, return evidenceSegmentIds only from supplied segment IDs and one assessment per order.
Return only the response schema.
""".strip()

_ASSESSMENT_PROPERTIES: dict[str, Any] = {
    "difficultyStatus": {
        "type": "STRING",
        "enum": ["ASSESSED", "BORDERLINE", "UNSURE", "NOT_ASSESSED"],
    },
    "observedBand": {"type": ["INTEGER", "NULL"]},
    "alternativeBand": {"type": ["INTEGER", "NULL"]},
    "issueCodes": {
        "type": "ARRAY",
        "items": {
            "type": "STRING",
            "enum": list(VOCABULARY_DIFFICULTY_ISSUE_CODES),
        },
    },
    "evidenceSegmentIds": {"type": "ARRAY", "items": {"type": "STRING"}},
    "difficultyConfidence": {"type": ["NUMBER", "NULL"]},
}
_ASSESSMENT_REQUIRED = list(_ASSESSMENT_PROPERTIES)

VOCABULARY_DIFFICULTY_SHADOW_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "additionalProperties": False,
    "properties": {
        "questionAssessments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "additionalProperties": False,
                "properties": {
                    "order": {"type": "INTEGER"},
                    **_ASSESSMENT_PROPERTIES,
                },
                "required": ["order", *_ASSESSMENT_REQUIRED],
            },
        },
    },
    "required": ["questionAssessments"],
}

_Band = Annotated[int, Field(strict=True, ge=1, le=5)]
_Confidence = Annotated[float, Field(strict=True, ge=0, le=1)]


class _VocabularyDifficultyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)

    order: Annotated[int, Field(strict=True, ge=1, le=20)]
    difficulty_status: Literal[
        "ASSESSED", "BORDERLINE", "UNSURE", "NOT_ASSESSED"
    ] = Field(alias="difficultyStatus")
    observed_band: _Band | None = Field(alias="observedBand")
    alternative_band: _Band | None = Field(alias="alternativeBand")
    issue_codes: list[str] = Field(alias="issueCodes", max_length=12)
    evidence_segment_ids: list[str] = Field(alias="evidenceSegmentIds", max_length=20)
    difficulty_confidence: _Confidence | None = Field(alias="difficultyConfidence")

    @model_validator(mode="after")
    def validate_band_shape(self) -> "_VocabularyDifficultyAssessment":
        if any(
            issue_code not in VOCABULARY_DIFFICULTY_ISSUE_CODES
            for issue_code in self.issue_codes
        ):
            raise ValueError("Vocabulary difficulty issue code is not telemetry-safe")
        valid = (
            self.difficulty_status == "ASSESSED"
            and self.observed_band is not None
            and self.alternative_band is None
        ) or (
            self.difficulty_status == "BORDERLINE"
            and self.observed_band is not None
            and self.alternative_band is not None
            and abs(self.observed_band - self.alternative_band) == 1
        ) or (
            self.difficulty_status in {"UNSURE", "NOT_ASSESSED"}
            and self.observed_band is None
            and self.alternative_band is None
        )
        if not valid:
            raise ValueError("Vocabulary difficulty assessment band/status mismatch")
        return self


class _VocabularyDifficultyShadowPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)

    question_assessments: list[_VocabularyDifficultyAssessment] = Field(
        alias="questionAssessments"
    )


@dataclass(frozen=True)
class _CallMetadata:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


_SAMPLING_NAMESPACE = "vocabulary-difficulty-shadow-v1:"
_SAMPLING_BUCKETS = 1_000_000


def vocabulary_difficulty_shadow_sample_bucket(request_id: str) -> int:
    digest = hashlib.sha256(f"{_SAMPLING_NAMESPACE}{request_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SAMPLING_BUCKETS


def should_sample_vocabulary_difficulty_shadow(
    request_id: str,
    sample_percent: float,
) -> bool:
    if (
        isinstance(sample_percent, bool)
        or not isinstance(sample_percent, (int, float))
        or not math.isfinite(sample_percent)
        or not 0 <= sample_percent <= 100
    ):
        raise ValueError("Vocabulary difficulty shadow sample percent must be from 0 to 100")
    if sample_percent == 0:
        return False
    if sample_percent == 100:
        return True
    return vocabulary_difficulty_shadow_sample_bucket(request_id) < sample_percent * 10_000


def vocabulary_question_segment_ids(question: PracticeGeneratedQuestion) -> set[str]:
    return {
        f"question:{question.order}:prompt",
        *(f"question:{question.order}:option:{option.key}" for option in question.options),
    }


def build_vocabulary_difficulty_shadow_payload(
    request: PracticeGenerationRequest,
    questions: list[PracticeGeneratedQuestion],
) -> dict[str, Any]:
    if request.domain != PracticeDomain.VOCABULARY:
        raise ValueError("Vocabulary difficulty shadow requires a Vocabulary request")
    return {
        "learningLanguage": request.learning_language,
        "mode": request.mode,
        "vocabularyDifficultyRubric": vocabulary_blind_rubric_payload(request.mode),
        "questions": [
            {
                "order": question.order,
                "questionType": question.question_type.value,
                "skillTag": question.skill_tag,
                "subtype": vocabulary_task_subtype(
                    mode=request.mode,
                    skill_tag=question.skill_tag,
                    question_type=question.question_type.value,
                ),
                **(
                    {"usageIntent": usage_intent_for_skill(question.skill_tag)}
                    if request.mode == VocabularyMode.USAGE_DISTINCTION.value
                    else {}
                ),
                "prompt": {
                    "segmentId": f"question:{question.order}:prompt",
                    "text": question.prompt,
                },
                "options": [
                    {
                        "key": option.key,
                        "segmentId": f"question:{question.order}:option:{option.key}",
                        "text": option.text,
                    }
                    for option in question.options
                ],
            }
            for question in questions
        ],
    }


def build_vocabulary_difficulty_shadow_prompt(
    request: PracticeGenerationRequest,
    questions: list[PracticeGeneratedQuestion],
) -> str:
    payload = build_vocabulary_difficulty_shadow_payload(request, questions)
    return (
        "Blindly classify Vocabulary difficulty. Return assessments only.\n"
        "<vocabulary-difficulty-data>\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "</vocabulary-difficulty-data>"
    )


class VocabularyDifficultyShadowCollector:
    """Awaited, fail-open Vocabulary-only difficulty telemetry collector."""

    def __init__(
        self,
        provider: TextGenerationProvider,
        *,
        enabled: bool = False,
        sample_percent: float = 0.0,
        timeout_seconds: float = 12.0,
    ) -> None:
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
            raise ValueError("Vocabulary difficulty shadow timeout must be from 0 to 60 seconds")
        should_sample_vocabulary_difficulty_shadow(
            "configuration-validation", sample_percent
        )
        self.provider = provider
        self.enabled = enabled
        self.sample_percent = float(sample_percent)
        self.timeout_seconds = timeout_seconds

    async def collect_if_selected(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
    ) -> bool:
        if request.domain != PracticeDomain.VOCABULARY or not self.enabled:
            return False
        selected = should_sample_vocabulary_difficulty_shadow(
            request.request_id,
            self.sample_percent,
        )
        logger.debug(
            "Vocabulary difficulty shadow cohort. request_id=%s sample_percentage=%s "
            "sample_bucket=%d cohort=%s",
            request.request_id,
            self.sample_percent,
            vocabulary_difficulty_shadow_sample_bucket(request.request_id),
            "selected" if selected else "unselected",
        )
        if not selected:
            return False

        started = time.perf_counter()
        metadata = _CallMetadata("unknown", "unknown", 0, 0)
        try:
            raw, metadata = await asyncio.wait_for(
                self._call_provider(
                    build_vocabulary_difficulty_shadow_prompt(request, questions)
                ),
                timeout=self.timeout_seconds,
            )
            payload = _VocabularyDifficultyShadowPayload.model_validate(raw)
            self._validate_coverage(payload, questions)
            self._record_telemetry(
                request,
                questions,
                payload,
                metadata=metadata,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Vocabulary difficulty shadow failed open. request_id=%s mode=%s "
                "sample_percentage=%s cohort=selected provider=%s model=%s latency_ms=%.3f "
                "input_tokens=%d output_tokens=%d type=%s",
                request.request_id,
                request.mode,
                self.sample_percent,
                metadata.provider,
                metadata.model,
                (time.perf_counter() - started) * 1000,
                metadata.input_tokens,
                metadata.output_tokens,
                type(exc).__name__,
            )
        return True

    async def _call_provider(self, prompt: str) -> tuple[Any, _CallMetadata]:
        call_with_metadata = getattr(self.provider, "call_with_metadata", None)
        supports = getattr(self.provider, "supports", None)
        metadata_supported = callable(call_with_metadata)
        if metadata_supported and callable(supports):
            try:
                metadata_supported = bool(supports("call_with_metadata"))
            except Exception:
                metadata_supported = False
        if metadata_supported:
            assert callable(call_with_metadata)
            metadata_call = cast(
                Callable[..., Awaitable[StructuredGenerationResult]],
                call_with_metadata,
            )
            result = await metadata_call(
                type_name=VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME,
                data=prompt,
                schema=VOCABULARY_DIFFICULTY_SHADOW_SCHEMA,
            )
            if not isinstance(result, StructuredGenerationResult):
                raise TypeError("shadow provider metadata result has an invalid type")
            return result.data, _CallMetadata(
                provider=result.provider,
                model=result.model,
                input_tokens=max(0, result.input_tokens),
                output_tokens=max(0, result.output_tokens),
            )

        raw = await self.provider.call(
            type_name=VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME,
            data=prompt,
            schema=VOCABULARY_DIFFICULTY_SHADOW_SCHEMA,
        )
        return raw, _CallMetadata(
            provider=self._provider_label("provider_name_for", "provider_name"),
            model=self._provider_label("model_name_for"),
            input_tokens=0,
            output_tokens=0,
        )

    def _provider_label(self, method_name: str, attribute_name: str | None = None) -> str:
        method = getattr(self.provider, method_name, None)
        if callable(method):
            try:
                return str(method(VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME))
            except Exception:
                return "unknown"
        if attribute_name is not None:
            return str(getattr(self.provider, attribute_name, "unknown"))
        return "unknown"

    @staticmethod
    def _validate_coverage(
        payload: _VocabularyDifficultyShadowPayload,
        questions: list[PracticeGeneratedQuestion],
    ) -> None:
        expected_orders = [question.order for question in questions]
        observed_orders = [item.order for item in payload.question_assessments]
        if (
            len(observed_orders) != len(expected_orders)
            or set(observed_orders) != set(expected_orders)
        ):
            raise ValueError("Vocabulary difficulty shadow question coverage mismatch")

    def _record_telemetry(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
        payload: _VocabularyDifficultyShadowPayload,
        *,
        metadata: _CallMetadata,
        latency_ms: float,
    ) -> None:
        policy = VocabularySemanticDifficultyPolicy()
        question_by_order = {question.order: question for question in questions}
        for raw in payload.question_assessments:
            question = question_by_order[raw.order]
            assessment = normalize_vocabulary_semantic_difficulty_assessment(
                raw.model_dump(by_alias=True),
                allowed_evidence_refs=vocabulary_question_segment_ids(question),
            )
            comparison = policy.decide(
                requested_band=question.complexity_band,
                assessment=assessment,
            )
            subtype = vocabulary_task_subtype(
                mode=request.mode,
                skill_tag=question.skill_tag,
                question_type=question.question_type.value,
            )
            usage_intent = (
                usage_intent_for_skill(question.skill_tag)
                if request.mode == VocabularyMode.USAGE_DISTINCTION.value
                else None
            )
            logger.info(
                "Vocabulary difficulty shadow. request_id=%s learning_language=%s mode=%s "
                "order=%d subtype=%s usage_intent=%s requested_band=%d recipe_version=%s "
                "measurement_rubric_version=%s difficulty_status=%s observed_band=%s "
                "alternative_band=%s comparison=%s difficulty_confidence=%s issue_codes=%s "
                "evidence_refs=%s provider=%s model=%s latency_ms=%.3f input_tokens=%d "
                "output_tokens=%d sample_percentage=%s cohort=selected",
                request.request_id,
                request.learning_language,
                request.mode,
                question.order,
                subtype,
                usage_intent,
                question.complexity_band,
                vocabulary_difficulty_recipe_version(request.mode),
                VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION,
                assessment.difficulty_status,
                assessment.observed_target,
                assessment.alternative_target,
                comparison.action,
                assessment.difficulty_confidence,
                assessment.issue_codes,
                assessment.evidence_refs,
                metadata.provider,
                metadata.model,
                latency_ms,
                metadata.input_tokens,
                metadata.output_tokens,
                self.sample_percent,
            )
