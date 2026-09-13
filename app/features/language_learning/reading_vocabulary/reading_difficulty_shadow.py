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
from app.features.language_learning.reading_vocabulary.reading_difficulty_adapter import (
    ReadingSemanticDifficultyPolicy,
    measure_reading_passage,
    measure_reading_question,
    normalize_reading_semantic_difficulty_assessment,
    reading_passage_segments,
    reading_question_segment_ids,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_recipe import (
    READING_DIFFICULTY_RECIPE_VERSION,
    reading_blind_rubric_payload,
)
from app.schemas.language_learning_practice import (
    PracticeDomain,
    PracticeGeneratedQuestion,
    PracticeGenerationRequest,
)

logger = logging.getLogger(__name__)

READING_DIFFICULTY_SHADOW_TYPE_NAME = "LANGUAGE_LEARNING_READING_DIFFICULTY_SHADOW"
_SAMPLING_NAMESPACE = "reading-difficulty-shadow-v1:"
_SAMPLING_BUCKETS = 1_000_000

READING_DIFFICULTY_SHADOW_SYSTEM_PROMPT = r"""
You are a blind difficulty classifier for TranslaCat Reading practice.

Classify the visible passage complexity and composite question demand against the complete Band 1-5 rubric.
- Passage difficulty is the linguistic and discourse complexity of the passage itself.
- Question difficulty includes evidence explicitness and location, inference and discourse-relation demand, and
  distractor discrimination demand.
- Ambiguity, weak distractors and required external knowledge never increase difficulty.
- Do not decide answer correctness or semantic-quality acceptance.
- Return one assessment for every supplied passageId and every supplied question order, with no extras.
- Use only supplied passage/question segment IDs in evidenceSegmentIds; never quote source content.
- ASSESSED names one observedBand and no alternativeBand.
- BORDERLINE names exactly two adjacent bands.
- UNSURE uses null for both bands when visible evidence is insufficient.
The requested band, relative difficulty label, generation recipe and answer key are intentionally absent.
""".strip()

_ASSESSMENT_PROPERTIES: dict[str, Any] = {
    "difficultyStatus": {
        "type": "STRING",
        "enum": ["ASSESSED", "BORDERLINE", "UNSURE"],
    },
    "observedBand": {"type": ["INTEGER", "NULL"]},
    "alternativeBand": {"type": ["INTEGER", "NULL"]},
    "issueCodes": {"type": "ARRAY", "items": {"type": "STRING"}},
    "evidenceSegmentIds": {"type": "ARRAY", "items": {"type": "STRING"}},
    "difficultyConfidence": {"type": ["NUMBER", "NULL"]},
}
_ASSESSMENT_REQUIRED = list(_ASSESSMENT_PROPERTIES)

READING_DIFFICULTY_SHADOW_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "additionalProperties": False,
    "properties": {
        "passageAssessments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "additionalProperties": False,
                "properties": {
                    "passageId": {"type": "STRING"},
                    **_ASSESSMENT_PROPERTIES,
                },
                "required": ["passageId", *_ASSESSMENT_REQUIRED],
            },
        },
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
    "required": ["passageAssessments", "questionAssessments"],
}

_Band = Annotated[int, Field(strict=True, ge=1, le=5)]
_Confidence = Annotated[float, Field(strict=True, ge=0, le=1)]


class _DifficultyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)

    difficulty_status: Literal["ASSESSED", "BORDERLINE", "UNSURE"] = Field(
        alias="difficultyStatus"
    )
    observed_band: _Band | None = Field(alias="observedBand")
    alternative_band: _Band | None = Field(alias="alternativeBand")
    issue_codes: list[str] = Field(alias="issueCodes", max_length=12)
    evidence_segment_ids: list[str] = Field(alias="evidenceSegmentIds", max_length=20)
    difficulty_confidence: _Confidence | None = Field(alias="difficultyConfidence")

    @model_validator(mode="after")
    def validate_band_shape(self) -> "_DifficultyAssessment":
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
            self.difficulty_status == "UNSURE"
            and self.observed_band is None
            and self.alternative_band is None
        )
        if not valid:
            raise ValueError("difficulty assessment band/status mismatch")
        return self


class _PassageAssessment(_DifficultyAssessment):
    passage_id: str = Field(alias="passageId", min_length=1, max_length=80)


class _QuestionAssessment(_DifficultyAssessment):
    order: Annotated[int, Field(strict=True, ge=1, le=20)]


class _DifficultyShadowPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)

    passage_assessments: list[_PassageAssessment] = Field(alias="passageAssessments")
    question_assessments: list[_QuestionAssessment] = Field(alias="questionAssessments")


@dataclass(frozen=True)
class _CallMetadata:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


def reading_difficulty_shadow_sample_bucket(request_id: str) -> int:
    digest = hashlib.sha256(f"{_SAMPLING_NAMESPACE}{request_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _SAMPLING_BUCKETS


def should_sample_reading_difficulty_shadow(
    request_id: str,
    sample_percent: float,
) -> bool:
    if (
        isinstance(sample_percent, bool)
        or not isinstance(sample_percent, (int, float))
        or not math.isfinite(sample_percent)
        or not 0 <= sample_percent <= 100
    ):
        raise ValueError("Reading difficulty shadow sample percent must be from 0 to 100")
    if sample_percent == 0:
        return False
    if sample_percent == 100:
        return True
    return reading_difficulty_shadow_sample_bucket(request_id) < sample_percent * 10_000


def build_reading_difficulty_shadow_payload(
    request: PracticeGenerationRequest,
    questions: list[PracticeGeneratedQuestion],
) -> dict[str, Any]:
    passage_text_by_id: dict[str, str] = {}
    for question in questions:
        if not question.passage_id or not question.passage_text:
            raise ValueError("Reading difficulty shadow requires passage-bound questions")
        existing = passage_text_by_id.setdefault(question.passage_id, question.passage_text)
        if existing != question.passage_text:
            raise ValueError("Reading difficulty shadow passage binding mismatch")

    passages = []
    for passage_id, passage_text in passage_text_by_id.items():
        segments = reading_passage_segments(passage_id, passage_text)
        passages.append(
            {
                "passageId": passage_id,
                "passageText": passage_text,
                "passageSegments": [
                    {"id": segment["id"], "ordinal": index}
                    for index, segment in enumerate(segments, 1)
                ],
            }
        )

    return {
        "learningLanguage": request.learning_language,
        "readingDifficultyRubric": reading_blind_rubric_payload(),
        "passages": passages,
        "questions": [
            {
                "order": question.order,
                "passageId": question.passage_id,
                "skillTag": question.skill_tag,
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


def build_reading_difficulty_shadow_prompt(
    request: PracticeGenerationRequest,
    questions: list[PracticeGeneratedQuestion],
) -> str:
    payload = build_reading_difficulty_shadow_payload(request, questions)
    return (
        "Blindly classify Reading difficulty. Return assessments only.\n"
        f"<reading-difficulty-data>\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "</reading-difficulty-data>"
    )


class ReadingDifficultyShadowCollector:
    """Awaited, fail-open Reading-only difficulty telemetry collector."""

    def __init__(
        self,
        provider: TextGenerationProvider,
        *,
        enabled: bool = False,
        sample_percent: float = 0.0,
        timeout_seconds: float = 12.0,
    ) -> None:
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
            raise ValueError("Reading difficulty shadow timeout must be from 0 to 60 seconds")
        should_sample_reading_difficulty_shadow("configuration-validation", sample_percent)
        self.provider = provider
        self.enabled = enabled
        self.sample_percent = float(sample_percent)
        self.timeout_seconds = timeout_seconds

    async def collect_if_selected(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
    ) -> bool:
        if request.domain != PracticeDomain.READING or not self.enabled:
            return False
        selected = should_sample_reading_difficulty_shadow(
            request.request_id,
            self.sample_percent,
        )
        logger.debug(
            "Reading difficulty shadow cohort. request_id=%s sample_percentage=%s "
            "sample_bucket=%d cohort=%s",
            request.request_id,
            self.sample_percent,
            reading_difficulty_shadow_sample_bucket(request.request_id),
            "selected" if selected else "unselected",
        )
        if not selected:
            return False

        started = time.perf_counter()
        metadata = _CallMetadata("unknown", "unknown", 0, 0)
        try:
            raw, metadata = await asyncio.wait_for(
                self._call_provider(
                    build_reading_difficulty_shadow_prompt(request, questions)
                ),
                timeout=self.timeout_seconds,
            )
            payload = _DifficultyShadowPayload.model_validate(raw)
            self._validate_coverage(payload, questions)
            latency_ms = (time.perf_counter() - started) * 1000
            self._record_telemetry(
                request,
                questions,
                payload,
                metadata=metadata,
                latency_ms=latency_ms,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Reading difficulty shadow failed open. request_id=%s sample_percentage=%s "
                "cohort=selected provider=%s model=%s latency_ms=%.3f input_tokens=%d "
                "output_tokens=%d type=%s",
                request.request_id,
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
                type_name=READING_DIFFICULTY_SHADOW_TYPE_NAME,
                data=prompt,
                schema=READING_DIFFICULTY_SHADOW_SCHEMA,
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
            type_name=READING_DIFFICULTY_SHADOW_TYPE_NAME,
            data=prompt,
            schema=READING_DIFFICULTY_SHADOW_SCHEMA,
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
                return str(method(READING_DIFFICULTY_SHADOW_TYPE_NAME))
            except Exception:
                return "unknown"
        if attribute_name is not None:
            return str(getattr(self.provider, attribute_name, "unknown"))
        return "unknown"

    @staticmethod
    def _validate_coverage(
        payload: _DifficultyShadowPayload,
        questions: list[PracticeGeneratedQuestion],
    ) -> None:
        expected_passage_ids = list(
            dict.fromkeys(
                question.passage_id for question in questions if question.passage_id
            )
        )
        observed_passage_ids = [item.passage_id for item in payload.passage_assessments]
        if (
            len(observed_passage_ids) != len(expected_passage_ids)
            or set(observed_passage_ids) != set(expected_passage_ids)
        ):
            raise ValueError("Reading difficulty shadow passage coverage mismatch")

        expected_orders = [question.order for question in questions]
        observed_orders = [item.order for item in payload.question_assessments]
        if (
            len(observed_orders) != len(expected_orders)
            or set(observed_orders) != set(expected_orders)
        ):
            raise ValueError("Reading difficulty shadow question coverage mismatch")

    def _record_telemetry(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
        payload: _DifficultyShadowPayload,
        *,
        metadata: _CallMetadata,
        latency_ms: float,
    ) -> None:
        policy = ReadingSemanticDifficultyPolicy()
        question_by_order = {question.order: question for question in questions}
        passage_text_by_id = {
            question.passage_id: question.passage_text
            for question in questions
            if question.passage_id and question.passage_text
        }
        for raw in payload.passage_assessments:
            passage_text = passage_text_by_id[raw.passage_id]
            allowed_refs = {
                segment["id"]
                for segment in reading_passage_segments(raw.passage_id, passage_text)
            }
            assessment = normalize_reading_semantic_difficulty_assessment(
                raw.model_dump(by_alias=True),
                allowed_evidence_refs=allowed_refs,
            )
            comparison = policy.decide(
                requested_band=request.complexity_band,
                assessment=assessment,
            )
            logger.info(
                "Reading difficulty shadow. request_id=%s learning_language=%s mode=%s "
                "scope=passage passage_id=%s requested_band=%d recipe_version=%s "
                "measurements=%s difficulty_status=%s observed_band=%s "
                "alternative_band=%s comparison=%s difficulty_confidence=%s "
                "issue_codes=%s evidence_refs=%s provider=%s model=%s latency_ms=%.3f "
                "input_tokens=%d output_tokens=%d sample_percentage=%s cohort=selected",
                request.request_id,
                request.learning_language,
                request.mode,
                raw.passage_id,
                request.complexity_band,
                READING_DIFFICULTY_RECIPE_VERSION,
                measure_reading_passage(passage_text).log_fields(),
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

        for raw in payload.question_assessments:
            question = question_by_order[raw.order]
            passage_id = question.passage_id or "unbound"
            passage_text = question.passage_text or ""
            allowed_refs = {
                segment["id"]
                for segment in reading_passage_segments(passage_id, passage_text)
            } | reading_question_segment_ids(question)
            assessment = normalize_reading_semantic_difficulty_assessment(
                raw.model_dump(by_alias=True),
                allowed_evidence_refs=allowed_refs,
            )
            comparison = policy.decide(
                requested_band=question.complexity_band,
                assessment=assessment,
            )
            logger.info(
                "Reading difficulty shadow. request_id=%s learning_language=%s mode=%s "
                "scope=question passage_id=%s order=%d skill_tag=%s requested_band=%d "
                "recipe_version=%s measurements=%s difficulty_status=%s observed_band=%s "
                "alternative_band=%s comparison=%s difficulty_confidence=%s "
                "issue_codes=%s evidence_refs=%s provider=%s model=%s latency_ms=%.3f "
                "input_tokens=%d output_tokens=%d sample_percentage=%s cohort=selected",
                request.request_id,
                request.learning_language,
                request.mode,
                passage_id,
                raw.order,
                question.skill_tag,
                question.complexity_band,
                READING_DIFFICULTY_RECIPE_VERSION,
                measure_reading_question(
                    question,
                    passage_id=passage_id,
                    passage_text=passage_text,
                ).log_fields(),
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
