from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from app.ai.ports import TextGenerationProvider
from app.features.language_learning.reading_vocabulary.reading_difficulty_adapter import (
    ReadingAcceptanceContext,
    ReadingSemanticQualityPolicy,
    build_reading_passage_difficulty_spec,
    build_reading_question_difficulty_spec,
    normalize_reading_semantic_assessment,
    project_reading_question_validation,
    validate_reading_passage,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_adapter import (
    VocabularyAcceptanceContext,
    build_vocabulary_difficulty_spec,
    normalize_vocabulary_semantic_assessment,
    project_vocabulary_validation,
    semantic_quality_policy_for_vocabulary_mode,
)
from app.features.language_learning.reading_vocabulary.prompts import (
    PRACTICE_GENERATION_PROMPT_VERSION,
    build_origin_explanation_prompt,
    build_practice_generation_prompt,
    build_practice_verification_prompt,
    build_reading_passage_prompt,
    build_usage_prescreen_prompt,
)
from app.schemas.language_learning_practice import (
    PracticeDifficulty,
    PracticeDomain,
    PracticeGeneratedQuestion,
    PracticeGenerationRequest,
    PracticeGenerationResponse,
    PracticeQuestionType,
    PracticeReviewTarget,
    ReadingMode,
    ReadingSkill,
    VocabularyMode,
    VocabularySkill,
)

logger = logging.getLogger(__name__)

_MAX_CANDIDATE_ATTEMPTS_PER_SLOT = 3
_MAX_READING_GENERATION_ATTEMPTS_PER_SLOT = 8
_MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT = 3
_MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT = 10
_MAX_USAGE_DISTINCTION_SEMANTIC_ATTEMPTS_PER_SLOT = 5
_MAX_COMPOSITION_GENERATION_ATTEMPTS_PER_SLOT = 8
_MAX_COMPOSITION_SEMANTIC_ATTEMPTS_PER_SLOT = 3
_MAX_PASSAGE_ATTEMPTS = 3
_MAX_ORIGIN_EXPLANATION_ATTEMPTS = 2
_CANDIDATE_BATCH_SIZE = 2
_USAGE_DISTINCTION_BATCH_SIZE = 1
_ORIGIN_EXPLANATION_BATCH_SIZE = 4

# ASCII-only tokens that are genuinely used as lexical items inside Japanese/Korean practical
# language. Ordinary English words are intentionally NOT accepted here: selected keywords may
# seed a topic, but they must never silently become the vocabulary answer in another language.
_ASCII_VOCABULARY_ALLOWLIST = {
    "AI", "API", "AWS", "B2B", "B2C", "CD", "CI", "CRM", "CPU", "DB", "DNS",
    "ERP", "GCP", "GPU", "HTTP", "HTTPS", "IP", "IT", "JSON", "KPI", "OKR",
    "PR", "QA", "RAM", "SaaS", "SDK", "SLA", "SQL", "SSH", "SSL", "TCP",
    "TLS", "UDP", "UI", "URI", "URL", "UX", "VPN", "XML",
    "BtoB", "BtoC", "DevOps", "Docker", "Git", "GitHub", "IoT", "Java",
    "JavaScript", "Kubernetes", "Next.js", "NoSQL", "PoC", "Python", "React",
    "RPA", "SRE", "TypeScript",
}
_ASCII_ACRONYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9+./#_-]{1,11}$")

_USAGE_INTENT_BY_SKILL = {
    VocabularySkill.DISTINCTION.value: "CONTEXTUAL_NEAR_EXPRESSION_CHOICE",
    VocabularySkill.COLLOCATION.value: "COLLOCATION_CHOICE",
    VocabularySkill.REGISTER.value: "REGISTER_CHOICE",
    VocabularySkill.CONTEXT_USAGE.value: "CONTEXTUAL_USAGE_CHOICE",
}

_OPTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {"key": {"type": "STRING"}, "text": {"type": "STRING"}},
    "required": ["key", "text"],
}

# explanationOrigin is intentionally absent. It is generated only after the question has
# passed deterministic + independent semantic verification, so a translation/language
# failure can never invalidate an otherwise-good question candidate.
_PRACTICE_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "order": {"type": "INTEGER"},
                    "questionType": {"type": "STRING", "enum": ["SINGLE_CHOICE", "ORDERING"]},
                    "difficulty": {"type": "STRING", "enum": ["EASIER", "CURRENT", "CHALLENGE"]},
                    "complexityBand": {"type": "INTEGER"},
                    "passageId": {"type": ["STRING", "NULL"]},
                    "passageText": {"type": ["STRING", "NULL"]},
                    "prompt": {"type": "STRING"},
                    "options": {"type": "ARRAY", "items": _OPTION_SCHEMA},
                    "correctAnswer": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "skillTag": {"type": "STRING"},
                    "evidenceText": {"type": ["STRING", "NULL"]},
                    "explanationLearning": {"type": "STRING"},
                    "targetExpression": {"type": ["STRING", "NULL"]},
                    "canonicalKey": {"type": ["STRING", "NULL"]},
                    "reviewTarget": {"type": "BOOLEAN"},
                    "vocabularyCandidates": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": [
                    "order", "questionType", "difficulty", "complexityBand", "passageId",
                    "passageText", "prompt", "options", "correctAnswer", "skillTag",
                    "evidenceText", "explanationLearning", "targetExpression", "canonicalKey",
                    "reviewTarget", "vocabularyCandidates",
                ],
            },
        }
    },
    "required": ["questions"],
}

_READING_PASSAGE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "passageId": {"type": "STRING"},
        "passageText": {"type": "STRING"},
    },
    "required": ["passageId", "passageText"],
}

_PRACTICE_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdicts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "order": {"type": "INTEGER"},
                    "bestAnswerKey": {"type": "STRING"},
                    "ambiguous": {"type": "BOOLEAN"},
                    "supported": {"type": "BOOLEAN"},
                    "reason": {"type": "STRING"},
                    "modeFit": {"type": "BOOLEAN"},
                    "answerLeakage": {"type": "BOOLEAN"},
                    "contextDependent": {"type": "BOOLEAN"},
                    "distractorsPlausible": {"type": "BOOLEAN"},
                },
                "required": [
                    "order", "bestAnswerKey", "ambiguous", "supported", "reason",
                    "modeFit", "answerLeakage", "contextDependent", "distractorsPlausible",
                ],
            },
        }
    },
    "required": ["verdicts"],
}

_USAGE_PRESCREEN_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdicts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "order": {"type": "INTEGER"},
                    "modeFit": {"type": "BOOLEAN"},
                    "answerLeakage": {"type": "BOOLEAN"},
                    "contextDependent": {"type": "BOOLEAN"},
                    "reason": {"type": "STRING"},
                },
                "required": ["order", "modeFit", "answerLeakage", "contextDependent", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}


_ORIGIN_EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "explanations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "order": {"type": "INTEGER"},
                    "text": {"type": "STRING"},
                },
                "required": ["order", "text"],
            },
        }
    },
    "required": ["explanations"],
}


class _PracticeVerificationVerdict(BaseModel):
    order: int
    bestAnswerKey: str
    ambiguous: bool
    supported: bool
    reason: str
    modeFit: bool
    answerLeakage: bool
    contextDependent: bool
    distractorsPlausible: bool


class _PracticeVerificationPayload(BaseModel):
    verdicts: list[_PracticeVerificationVerdict]


class _UsagePrescreenVerdict(BaseModel):
    order: int
    modeFit: bool = True
    answerLeakage: bool = False
    contextDependent: bool = True
    reason: str = ""


class _UsagePrescreenPayload(BaseModel):
    verdicts: list[_UsagePrescreenVerdict]


class _PracticeCandidatePayload(BaseModel):
    questions: list[dict[str, Any]]


class _ReadingPassagePayload(BaseModel):
    passageId: str
    passageText: str


class _OriginExplanationItem(BaseModel):
    order: int
    text: str


class _OriginExplanationPayload(BaseModel):
    explanations: list[_OriginExplanationItem]


@dataclass(frozen=True)
class _QuestionSlot:
    order: int
    difficulty: PracticeDifficulty
    complexity_band: int
    skill_tag: str
    question_type: PracticeQuestionType | None = None
    passage_id: str | None = None
    passage_text: str | None = None
    review_canonical_key: str | None = None
    review_expression: str | None = None
    previous_question_types: tuple[str, ...] = ()
    usage_intent: str | None = None

    @property
    def review_target(self) -> bool:
        return self.review_canonical_key is not None

    def prompt_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "order": self.order,
            "difficulty": self.difficulty.value,
            "complexityBand": self.complexity_band,
            "skillTag": self.skill_tag,
            "passageId": self.passage_id,
            "passageText": self.passage_text,
            "reviewTarget": self.review_target,
        }
        if self.question_type is not None:
            payload["questionType"] = self.question_type.value
        if self.usage_intent is not None:
            payload["usageIntent"] = self.usage_intent
            payload["answerVisibilityPolicy"] = "HIDE_TARGET_FROM_STEM"
        if self.review_target:
            payload["boundReviewTarget"] = {
                "canonicalKey": self.review_canonical_key,
                "expression": self.review_expression,
                "previousQuestionTypes": list(self.previous_question_types),
            }
        return payload


class ReadingVocabularyGenerationService:
    TYPE_NAME = "LANGUAGE_LEARNING_READING_VOCABULARY_GENERATION"
    PASSAGE_TYPE_NAME = "LANGUAGE_LEARNING_READING_PASSAGE_GENERATION"
    PRESCREEN_TYPE_NAME = "LANGUAGE_LEARNING_READING_VOCABULARY_USAGE_PRESCREEN"
    VERIFICATION_TYPE_NAME = "LANGUAGE_LEARNING_READING_VOCABULARY_VERIFICATION"
    ORIGIN_EXPLANATION_TYPE_NAME = "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION"
    ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME = (
        "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION_FALLBACK"
    )

    def __init__(self, provider: TextGenerationProvider, timeout_seconds: float = 45.0) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds

    async def generate(self, request: PracticeGenerationRequest) -> PracticeGenerationResponse:
        """Generate with application-owned planning, validation and smallest-unit retries.

        Core questions, semantic verification and origin-language explanations are independent
        stages. A bad explanation therefore repairs only the explanation; an ambiguous question
        regenerates only that slot; successful slots are retained.
        """
        try:
            passages = await self._generate_reading_passages(request)
            slots = self._build_slots(request, passages)
            questions = await self._generate_verified_questions(request, slots)
            questions = await self._attach_origin_explanations(request, questions)
            questions = sorted(questions, key=lambda item: item.order)
            self._validate(request, questions)
            return PracticeGenerationResponse(
                request_id=request.request_id,
                prompt_version=PRACTICE_GENERATION_PROMPT_VERSION,
                domain=request.domain,
                mode=request.mode,
                complexity_band=request.complexity_band,
                questions=questions,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(
                "Reading/Vocabulary generation pipeline failed. request_id=%s stage=final type=%s",
                request.request_id,
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "AI_GENERATION_FAILED",
                    "retryable": True,
                    "message": "Reading/Vocabulary 문제 생성에 실패했습니다.",
                    "cause": type(exc).__name__,
                },
            ) from exc

    async def _generate_reading_passages(
        self,
        request: PracticeGenerationRequest,
    ) -> dict[str, str]:
        if request.domain != PracticeDomain.READING:
            return {}

        passages = {
            question.passage_id: question.passage_text
            for question in request.previous_questions
            if question.passage_id and question.passage_text
        }
        required_passages = {
            1 if index <= 3 else 2
            for index in range(
                request.question_offset + 1,
                request.question_offset + request.question_count + 1,
            )
        }
        for passage_number in sorted(required_passages):
            passage_id = f"p{passage_number}"
            if passage_id in passages:
                self._assert_language_lane(
                    request.learning_language,
                    [passages[passage_id]],
                    reason="Reading passage is not in learningLanguage",
                )
                continue
            passage_spec = build_reading_passage_difficulty_spec(
                request,
                passage_id=passage_id,
                complexity_band=request.complexity_band,
            )
            last_error: Exception | None = None
            for attempt in range(1, _MAX_PASSAGE_ATTEMPTS + 1):
                try:
                    raw = await asyncio.wait_for(
                        self.provider.call(
                            type_name=self.PASSAGE_TYPE_NAME,
                            data=build_reading_passage_prompt(
                                request,
                                passage_id=passage_id,
                                passage_number=passage_number,
                            ),
                            schema=_READING_PASSAGE_SCHEMA,
                        ),
                        timeout=self.timeout_seconds,
                    )
                    payload = _ReadingPassagePayload.model_validate(raw)
                    passage_text = payload.passageText.strip()
                    passage_validation = validate_reading_passage(
                        passage_spec,
                        observed_passage_id=payload.passageId,
                        passage_text=passage_text,
                        language_validator=lambda: self._assert_language_lane(
                            request.learning_language,
                            [passage_text],
                            reason="Reading passage is not in learningLanguage",
                        ),
                    )
                    if not passage_validation.passed:
                        raise ValueError(passage_validation.primary_issue)
                    passages[passage_id] = passage_text
                    break
                except (ValidationError, ValueError, TimeoutError, asyncio.TimeoutError) as exc:
                    last_error = exc
                    logger.warning(
                        "Reading passage rejected. request_id=%s passage=%s attempt=%d/%d reason=%s",
                        request.request_id,
                        passage_id,
                        attempt,
                        _MAX_PASSAGE_ATTEMPTS,
                        str(exc)[:300],
                    )
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Reading passage provider failure. request_id=%s passage=%s attempt=%d/%d type=%s",
                        request.request_id,
                        passage_id,
                        attempt,
                        _MAX_PASSAGE_ATTEMPTS,
                        type(exc).__name__,
                    )
            if passage_id not in passages:
                raise ValueError(
                    f"reading passage generation exhausted passage={passage_id} "
                    f"cause={type(last_error).__name__ if last_error else 'unknown'}"
                )
        return passages

    def _eligible_review_targets(
        self,
        request: PracticeGenerationRequest,
        *,
        log_rejections: bool = True,
    ) -> list[PracticeReviewTarget]:
        if request.domain != PracticeDomain.VOCABULARY:
            return []
        committed_keys = {question.canonical_key for question in request.previous_questions}
        committed_expressions = {
            self._normalize_for_leak_check(question.target_expression or "")
            for question in request.previous_questions
        }
        eligible = []
        for target in request.review_targets:
            if target.canonical_key in committed_keys or self._normalize_for_leak_check(
                target.expression
            ) in committed_expressions:
                continue
            try:
                self._assert_vocabulary_surface_language(
                    request.learning_language,
                    target.expression,
                    reason=(
                        "review target expression is not a lexical expression in learningLanguage"
                    ),
                )
            except ValueError as exc:
                if log_rejections:
                    logger.warning(
                        "Reading/Vocabulary review target skipped by language lane. "
                        "request_id=%s canonical_key=%s expression=%s reason=%s",
                        request.request_id,
                        target.canonical_key,
                        target.expression[:120],
                        str(exc)[:220],
                    )
                continue
            eligible.append(target)
        return eligible

    def _build_slots(
        self,
        request: PracticeGenerationRequest,
        passages: dict[str, str],
    ) -> list[_QuestionSlot]:
        difficulties = self._difficulty_plan(request)
        if request.domain == PracticeDomain.READING:
            skill_cycle = {
                ReadingMode.COMPREHENSION.value: [
                    ReadingSkill.CONTENT.value,
                    ReadingSkill.DETAIL.value,
                    ReadingSkill.INFERENCE.value,
                    ReadingSkill.DETAIL.value,
                    ReadingSkill.INFERENCE.value,
                ],
                ReadingMode.STRUCTURE.value: [
                    ReadingSkill.GIST.value,
                    ReadingSkill.STRUCTURE.value,
                    ReadingSkill.STRUCTURE.value,
                    ReadingSkill.GIST.value,
                    ReadingSkill.STRUCTURE.value,
                ],
                ReadingMode.CONTEXT_INFERENCE.value: [
                    ReadingSkill.CONTEXT_INFERENCE.value,
                    ReadingSkill.INFERENCE.value,
                    ReadingSkill.CONTEXT_INFERENCE.value,
                    ReadingSkill.INFERENCE.value,
                    ReadingSkill.CONTEXT_INFERENCE.value,
                ],
            }[request.mode]
            return [
                _QuestionSlot(
                    order=index,
                    difficulty=difficulty,
                    complexity_band=self._band_for(request, difficulty),
                    skill_tag=skill_cycle[request.question_offset + index - 1],
                    passage_id="p1" if request.question_offset + index <= 3 else "p2",
                    passage_text=passages[
                        "p1" if request.question_offset + index <= 3 else "p2"
                    ],
                )
                for index, difficulty in enumerate(difficulties, 1)
            ]

        skill_cycle = {
            VocabularyMode.MEANING_RELATION.value: [
                VocabularySkill.MEANING.value,
                VocabularySkill.SYNONYM.value,
                VocabularySkill.ANTONYM.value,
                VocabularySkill.DISTINCTION.value,
            ],
            VocabularyMode.USAGE_DISTINCTION.value: [
                VocabularySkill.DISTINCTION.value,
                VocabularySkill.COLLOCATION.value,
                VocabularySkill.REGISTER.value,
                VocabularySkill.CONTEXT_USAGE.value,
            ],
            VocabularyMode.COMPOSITION.value: [
                VocabularySkill.COMPOSITION.value,
                VocabularySkill.COLLOCATION.value,
                VocabularySkill.COMPOSITION.value,
                VocabularySkill.CONTEXT_USAGE.value,
            ],
        }[request.mode]
        ordering_orders = {1, 3, 6, 8} if request.mode == VocabularyMode.COMPOSITION.value else set()
        eligible_review_targets = self._eligible_review_targets(request)
        review_targets = list(eligible_review_targets[: request.review_question_count])
        slots: list[_QuestionSlot] = []
        for index, difficulty in enumerate(difficulties, 1):
            global_index = request.question_offset + index
            review = review_targets[index - 1] if index <= len(review_targets) else None
            slots.append(
                _QuestionSlot(
                    order=index,
                    difficulty=difficulty,
                    complexity_band=self._band_for(request, difficulty),
                    skill_tag=skill_cycle[(global_index - 1) % len(skill_cycle)],
                    question_type=(
                        PracticeQuestionType.ORDERING
                        if global_index in ordering_orders
                        else PracticeQuestionType.SINGLE_CHOICE
                    ),
                    review_canonical_key=review.canonical_key if review else None,
                    review_expression=review.expression if review else None,
                    previous_question_types=tuple(
                        item.value for item in (review.previous_question_types if review else [])
                    ),
                    usage_intent=(
                        self._usage_intent_for_skill(skill_cycle[(global_index - 1) % len(skill_cycle)])
                        if request.mode == VocabularyMode.USAGE_DISTINCTION.value
                        else None
                    ),
                )
            )
        return slots

    @staticmethod
    def _difficulty_plan(request: PracticeGenerationRequest) -> list[PracticeDifficulty]:
        remaining = {
            PracticeDifficulty.EASIER: request.easier_count,
            PracticeDifficulty.CURRENT: request.current_count,
            PracticeDifficulty.CHALLENGE: request.challenge_count,
        }
        # CURRENT first keeps the set anchored at the learner estimate; easier/challenge are
        # distributed around it instead of relying on the model to invent the mix.
        pattern = [
            PracticeDifficulty.CURRENT,
            PracticeDifficulty.EASIER,
            PracticeDifficulty.CURRENT,
            PracticeDifficulty.CHALLENGE,
        ]
        result: list[PracticeDifficulty] = []
        while len(result) < request.question_count:
            progressed = False
            for difficulty in pattern:
                if remaining[difficulty] <= 0:
                    continue
                result.append(difficulty)
                remaining[difficulty] -= 1
                progressed = True
                if len(result) == request.question_count:
                    break
            if not progressed:
                break
        if len(result) != request.question_count:
            raise ValueError("failed to build deterministic difficulty plan")
        return result

    @staticmethod
    def _band_for(request: PracticeGenerationRequest, difficulty: PracticeDifficulty) -> int:
        if difficulty == PracticeDifficulty.EASIER:
            return max(1, request.complexity_band - 1)
        if difficulty == PracticeDifficulty.CHALLENGE:
            return min(5, request.complexity_band + 1)
        return request.complexity_band

    async def _generate_verified_questions(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
    ) -> list[PracticeGeneratedQuestion]:
        if request.domain == PracticeDomain.READING:
            return await self._generate_verified_reading_questions(request, slots)

        if request.domain == PracticeDomain.VOCABULARY:
            if request.mode == VocabularyMode.USAGE_DISTINCTION.value:
                return await self._generate_verified_usage_distinction_questions(request, slots)
            if request.mode == VocabularyMode.COMPOSITION.value:
                return await self._generate_verified_composition_questions(request, slots)

        slot_by_order = {slot.order: slot for slot in slots}
        attempts = {slot.order: 0 for slot in slots}
        accepted: dict[int, PracticeGeneratedQuestion] = {}
        semantically_verified: set[int] = set()
        retry_feedback: dict[int, str] = {}
        attempt_limit = self._candidate_attempt_limit(request)
        batch_size = self._candidate_batch_size(request)

        while len(semantically_verified) < len(slots):
            pending_generation = [
                order
                for order in sorted(slot_by_order)
                if order not in accepted and attempts[order] < attempt_limit
            ]
            if pending_generation:
                for start in range(0, len(pending_generation), batch_size):
                    batch_orders = pending_generation[start : start + batch_size]
                    batch_slots = [slot_by_order[order] for order in batch_orders]
                    for order in batch_orders:
                        attempts[order] += 1
                    candidate_failures = await self._generate_candidate_batch(
                        request,
                        batch_slots,
                        accepted,
                        retry_feedback=retry_feedback,
                    )
                    for order, reason in candidate_failures.items():
                        retry_feedback[order] = reason
                    for order in batch_orders:
                        if order in accepted:
                            retry_feedback.pop(order, None)

            exhausted = [
                order
                for order in slot_by_order
                if order not in accepted and attempts[order] >= attempt_limit
            ]
            if exhausted:
                raise ValueError(f"candidate generation exhausted orders={sorted(exhausted)}")

            unverified = [
                accepted[order]
                for order in sorted(accepted)
                if order not in semantically_verified
            ]
            if not unverified:
                continue

            # Nano is deliberately a soft prescreen. It is useful for cheap diagnostics, but it
            # must never be the authority that burns a scarce candidate attempt. Deterministic
            # checks already reject direct leakage and Mini remains the independent semantic gate.
            prescreen_flags = await self._usage_prescreen_failures(request, unverified)
            for question in unverified:
                failure = prescreen_flags.get(question.order)
                if failure is None:
                    continue
                logger.warning(
                    "Reading/Vocabulary Nano prescreen flagged candidate; deferring to Mini. "
                    "request_id=%s order=%d attempt=%d/%d reason=%s",
                    request.request_id,
                    question.order,
                    attempts[question.order],
                    attempt_limit,
                    failure[:300],
                )

            semantic_failures = await self._semantic_failures(request, unverified)
            for question in unverified:
                if question.question_type == PracticeQuestionType.ORDERING:
                    semantically_verified.add(question.order)
                    retry_feedback.pop(question.order, None)
                    continue
                failure = semantic_failures.get(question.order)
                if failure is None:
                    semantically_verified.add(question.order)
                    retry_feedback.pop(question.order, None)
                    continue
                logger.warning(
                    "Reading/Vocabulary semantic candidate rejected. request_id=%s order=%d "
                    "attempt=%d/%d reason=%s",
                    request.request_id,
                    question.order,
                    attempts[question.order],
                    attempt_limit,
                    failure[:300],
                )
                accepted.pop(question.order, None)
                semantically_verified.discard(question.order)
                retry_feedback[question.order] = failure

            semantic_exhausted = [
                order
                for order in slot_by_order
                if order not in semantically_verified
                and order not in accepted
                and attempts[order] >= attempt_limit
            ]
            if semantic_exhausted:
                raise ValueError(f"semantic verification exhausted orders={sorted(semantic_exhausted)}")

        return [accepted[order] for order in sorted(accepted)]

    async def _generate_verified_reading_questions(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
    ) -> list[PracticeGeneratedQuestion]:
        """Generate Reading with independent structural and semantic retry budgets.

        Reading candidate generation can fail for deterministic/schema/language reasons while an
        otherwise-good question can fail Mini only for semantic quality. Those are different
        failure classes and must not consume the same scarce retry counter. Keep successful slots,
        regenerate only failed slots, and preserve the existing small generation batches.
        """
        slot_by_order = {slot.order: slot for slot in slots}
        generation_attempts = {slot.order: 0 for slot in slots}
        semantic_attempts = {slot.order: 0 for slot in slots}
        accepted: dict[int, PracticeGeneratedQuestion] = {}
        semantically_verified: set[int] = set()
        retry_feedback: dict[int, str] = {}

        while len(semantically_verified) < len(slots):
            pending_generation = [
                order
                for order in sorted(slot_by_order)
                if order not in accepted
                and order not in semantically_verified
                and generation_attempts[order] < _MAX_READING_GENERATION_ATTEMPTS_PER_SLOT
                and semantic_attempts[order] < _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT
            ]
            for start in range(0, len(pending_generation), _CANDIDATE_BATCH_SIZE):
                batch_orders = pending_generation[start : start + _CANDIDATE_BATCH_SIZE]
                batch_slots = [slot_by_order[order] for order in batch_orders]
                for order in batch_orders:
                    generation_attempts[order] += 1
                candidate_failures = await self._generate_candidate_batch(
                    request,
                    batch_slots,
                    accepted,
                    retry_feedback=(retry_feedback or None),
                )
                for order in batch_orders:
                    if order in accepted:
                        retry_feedback.pop(order, None)
                        continue
                    reason = candidate_failures.get(
                        order, "candidate generation did not yield an accepted item"
                    )
                    retry_feedback[order] = self._reading_retry_feedback(reason)

            generation_exhausted = [
                order
                for order in sorted(slot_by_order)
                if order not in accepted
                and order not in semantically_verified
                and generation_attempts[order] >= _MAX_READING_GENERATION_ATTEMPTS_PER_SLOT
            ]
            if generation_exhausted:
                raise ValueError(
                    "reading candidate generation exhausted orders="
                    f"{generation_exhausted}"
                )

            unverified = [
                accepted[order]
                for order in sorted(accepted)
                if order not in semantically_verified
            ]
            if not unverified:
                continue

            for question in unverified:
                semantic_attempts[question.order] += 1
            semantic_failures = await self._semantic_failures(request, unverified)

            semantic_exhausted: list[int] = []
            for question in unverified:
                failure = semantic_failures.get(question.order)
                if failure is None:
                    semantically_verified.add(question.order)
                    retry_feedback.pop(question.order, None)
                    continue

                logger.warning(
                    "Reading/Vocabulary semantic candidate rejected. request_id=%s order=%d "
                    "generation_attempt=%d/%d semantic_attempt=%d/%d reason=%s",
                    request.request_id,
                    question.order,
                    generation_attempts[question.order],
                    _MAX_READING_GENERATION_ATTEMPTS_PER_SLOT,
                    semantic_attempts[question.order],
                    _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT,
                    failure[:300],
                )
                accepted.pop(question.order, None)
                retry_feedback[question.order] = self._reading_retry_feedback(failure)
                if semantic_attempts[question.order] >= _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT:
                    semantic_exhausted.append(question.order)

            if semantic_exhausted:
                raise ValueError(
                    "reading semantic verification exhausted orders="
                    f"{sorted(semantic_exhausted)}"
                )

        return [accepted[order] for order in sorted(accepted)]

    async def _generate_verified_usage_distinction_questions(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
    ) -> list[PracticeGeneratedQuestion]:
        """Run the strict usage pipeline end-to-end one slot at a time.

        Generation uses one-item-per-Luna-call and previously accumulated all ten
        candidates before asking Mini to judge them together. In production that caused
        synchronized batch rejections and made retry feedback much less targeted. Keep
        already-approved slots, and finish deterministic -> Nano -> Mini for one slot
        before moving to the next.
        """
        accepted: dict[int, PracticeGeneratedQuestion] = {}

        for slot in slots:
            retry_feedback: str | None = None
            retry_target_expression: str | None = None
            generation_attempts = 0
            semantic_attempts = 0
            last_stage = "candidate"

            while (
                generation_attempts < _MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT
                and semantic_attempts < _MAX_USAGE_DISTINCTION_SEMANTIC_ATTEMPTS_PER_SLOT
            ):
                generation_attempts += 1
                candidate_failures = await self._generate_candidate_batch(
                    request,
                    [slot],
                    accepted,
                    retry_feedback=(
                        {slot.order: retry_feedback} if retry_feedback else None
                    ),
                    retry_target_expressions=(
                        {slot.order: retry_target_expression}
                        if retry_target_expression
                        else None
                    ),
                )
                question = accepted.get(slot.order)
                if question is None:
                    retry_feedback = candidate_failures.get(
                        slot.order, "candidate generation did not yield an accepted item"
                    )
                    last_stage = "candidate"
                    continue

                prescreen_flags = await self._usage_prescreen_failures(request, [question])
                prescreen_failure = prescreen_flags.get(slot.order)
                if prescreen_failure is not None:
                    logger.warning(
                        "Reading/Vocabulary Nano prescreen flagged candidate; deferring to Mini. "
                        "request_id=%s order=%d generation_attempt=%d/%d "
                        "semantic_attempt=%d/%d reason=%s",
                        request.request_id,
                        slot.order,
                        generation_attempts,
                        _MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT,
                        semantic_attempts + 1,
                        _MAX_USAGE_DISTINCTION_SEMANTIC_ATTEMPTS_PER_SLOT,
                        prescreen_failure[:300],
                    )

                semantic_attempts += 1
                semantic_failures = await self._semantic_failures(request, [question])
                semantic_failure = semantic_failures.get(slot.order)
                if semantic_failure is None:
                    retry_feedback = None
                    retry_target_expression = None
                    break

                logger.warning(
                    "Reading/Vocabulary semantic candidate rejected. request_id=%s order=%d "
                    "generation_attempt=%d/%d semantic_attempt=%d/%d reason=%s",
                    request.request_id,
                    slot.order,
                    generation_attempts,
                    _MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT,
                    semantic_attempts,
                    _MAX_USAGE_DISTINCTION_SEMANTIC_ATTEMPTS_PER_SLOT,
                    semantic_failure[:300],
                )
                accepted.pop(slot.order, None)
                retry_target_expression = question.target_expression
                retry_feedback = self._usage_retry_feedback(semantic_failure)
                last_stage = "semantic"

            if slot.order not in accepted:
                raise ValueError(
                    f"{last_stage} verification exhausted orders={[slot.order]} "
                    f"generationAttempts={generation_attempts}/"
                    f"{_MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT} "
                    f"semanticAttempts={semantic_attempts}/"
                    f"{_MAX_USAGE_DISTINCTION_SEMANTIC_ATTEMPTS_PER_SLOT}"
                )

        return [accepted[order] for order in sorted(accepted)]

    async def _generate_verified_composition_questions(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
    ) -> list[PracticeGeneratedQuestion]:
        """Generate COMPOSITION slots independently with separate structural/semantic budgets.

        COMPOSITION has strict application-owned ORDERING shapes. A malformed Luna payload must
        not consume the same three-attempt budget used for semantic quality, otherwise one slot can
        exhaust before Mini ever gets a fair chance to judge it. ORDERING items need structural
        validation only; SINGLE_CHOICE items still pass the independent Mini verifier.
        """
        accepted: dict[int, PracticeGeneratedQuestion] = {}

        for slot in slots:
            retry_feedback: str | None = None
            retry_target_expression: str | None = None
            generation_attempts = 0
            semantic_attempts = 0
            last_stage = "candidate"

            while (
                generation_attempts < _MAX_COMPOSITION_GENERATION_ATTEMPTS_PER_SLOT
                and semantic_attempts < _MAX_COMPOSITION_SEMANTIC_ATTEMPTS_PER_SLOT
            ):
                generation_attempts += 1
                candidate_failures = await self._generate_candidate_batch(
                    request,
                    [slot],
                    accepted,
                    retry_feedback=({slot.order: retry_feedback} if retry_feedback else None),
                    retry_target_expressions=(
                        {slot.order: retry_target_expression}
                        if retry_target_expression
                        else None
                    ),
                )
                question = accepted.get(slot.order)
                if question is None:
                    retry_feedback = self._composition_retry_feedback(
                        candidate_failures.get(
                            slot.order, "candidate generation did not yield an accepted item"
                        ),
                        slot,
                    )
                    last_stage = "candidate"
                    continue

                if question.question_type == PracticeQuestionType.ORDERING:
                    retry_feedback = None
                    retry_target_expression = None
                    break

                semantic_attempts += 1
                semantic_failures = await self._semantic_failures(request, [question])
                semantic_failure = semantic_failures.get(slot.order)
                if semantic_failure is None:
                    retry_feedback = None
                    retry_target_expression = None
                    break

                logger.warning(
                    "Reading/Vocabulary semantic candidate rejected. request_id=%s order=%d "
                    "generation_attempt=%d/%d semantic_attempt=%d/%d reason=%s",
                    request.request_id,
                    slot.order,
                    generation_attempts,
                    _MAX_COMPOSITION_GENERATION_ATTEMPTS_PER_SLOT,
                    semantic_attempts,
                    _MAX_COMPOSITION_SEMANTIC_ATTEMPTS_PER_SLOT,
                    semantic_failure[:300],
                )
                accepted.pop(slot.order, None)
                retry_target_expression = question.target_expression
                retry_feedback = self._composition_retry_feedback(semantic_failure, slot)
                last_stage = "semantic"

            if slot.order not in accepted:
                raise ValueError(
                    f"{last_stage} verification exhausted orders={[slot.order]} "
                    f"generationAttempts={generation_attempts}/"
                    f"{_MAX_COMPOSITION_GENERATION_ATTEMPTS_PER_SLOT} "
                    f"semanticAttempts={semantic_attempts}/"
                    f"{_MAX_COMPOSITION_SEMANTIC_ATTEMPTS_PER_SLOT}"
                )

        return [accepted[order] for order in sorted(accepted)]

    @staticmethod
    def _candidate_attempt_limit(request: PracticeGenerationRequest) -> int:
        if (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.USAGE_DISTINCTION.value
        ):
            return _MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT
        return _MAX_CANDIDATE_ATTEMPTS_PER_SLOT

    @staticmethod
    def _candidate_batch_size(request: PracticeGenerationRequest) -> int:
        if (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.USAGE_DISTINCTION.value
        ):
            # Contextual usage questions are the strictest mode. Generate one slot per call so a
            # difficult slot cannot contaminate the rest of the batch and retries stay minimal.
            return _USAGE_DISTINCTION_BATCH_SIZE
        return _CANDIDATE_BATCH_SIZE

    @staticmethod
    def _usage_intent_for_skill(skill_tag: str) -> str | None:
        return _USAGE_INTENT_BY_SKILL.get(skill_tag)

    async def _generate_candidate_batch(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
        accepted: dict[int, PracticeGeneratedQuestion],
        *,
        retry_feedback: dict[int, str] | None = None,
        retry_target_expressions: dict[int, str] | None = None,
    ) -> dict[int, str]:
        expected_orders = {slot.order for slot in slots}
        excluded_keys = sorted(
            {
                question.canonical_key
                for question in [*request.previous_questions, *accepted.values()]
                if question.canonical_key
            }
            | {target.canonical_key for target in request.review_targets}
        )
        excluded_target_expressions = sorted(
            {
                question.target_expression.strip()
                for question in [*request.previous_questions, *accepted.values()]
                if question.target_expression and question.target_expression.strip()
            }
            | {
                target.expression.strip()
                for target in self._eligible_review_targets(request, log_rejections=False)
                if target.expression.strip()
            }
        )
        slot_payloads: list[dict[str, Any]] = []
        for slot in slots:
            payload = slot.prompt_payload()
            if retry_feedback and retry_feedback.get(slot.order):
                payload["retryFeedback"] = retry_feedback[slot.order][:600]
            if retry_target_expressions and retry_target_expressions.get(slot.order):
                payload["retryTargetExpression"] = retry_target_expressions[slot.order]
                payload["preserveTargetOnRetry"] = True
            slot_payloads.append(payload)

        failures: dict[int, str] = {}
        try:
            raw = await asyncio.wait_for(
                self.provider.call(
                    type_name=self.TYPE_NAME,
                    data=build_practice_generation_prompt(
                        request,
                        slot_payloads,
                        excluded_canonical_keys=excluded_keys,
                        excluded_target_expressions=excluded_target_expressions,
                    ),
                    schema=_PRACTICE_CANDIDATE_SCHEMA,
                ),
                timeout=self.timeout_seconds,
            )
            payload = _PracticeCandidatePayload.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "Reading/Vocabulary candidate batch provider/schema failure. request_id=%s orders=%s type=%s",
                request.request_id,
                sorted(expected_orders),
                type(exc).__name__,
            )
            return {order: f"provider/schema failure: {type(exc).__name__}" for order in expected_orders}

        raw_by_order: dict[int, dict[str, Any]] = {}
        duplicate_orders: set[int] = set()
        for item in payload.questions:
            order = item.get("order")
            if not isinstance(order, int) or order not in expected_orders:
                continue
            if order in raw_by_order:
                duplicate_orders.add(order)
            raw_by_order[order] = item
        for order in duplicate_orders:
            raw_by_order.pop(order, None)

        slot_by_order = {slot.order: slot for slot in slots}
        for order in sorted(expected_orders):
            item = raw_by_order.get(order)
            if item is None:
                logger.warning(
                    "Reading/Vocabulary candidate missing/duplicate. request_id=%s order=%d",
                    request.request_id,
                    order,
                )
                failures[order] = "candidate missing or duplicate in provider response"
                continue
            try:
                normalized_item = item
                if request.domain == PracticeDomain.READING:
                    normalized_item = {
                        **item,
                        "vocabularyCandidates": self._sanitize_reading_vocabulary_candidates(
                            item.get("vocabularyCandidates", []),
                            slot_by_order[order].passage_text or "",
                        ),
                    }
                # Placeholder is internal only and always overwritten after semantic acceptance.
                question = PracticeGeneratedQuestion.model_validate(
                    {**normalized_item, "explanationOrigin": "PENDING"}
                )
                self._normalize_reading_candidate_metadata(
                    request,
                    question,
                    slot_by_order[order],
                )
                self._normalize_usage_distinction_candidate_metadata(
                    request,
                    question,
                    slot_by_order[order],
                )
                self._normalize_composition_candidate_metadata(
                    request,
                    question,
                    slot_by_order[order],
                )
                expected_retry_target = (
                    retry_target_expressions.get(order)
                    if retry_target_expressions
                    else None
                )
                if (
                    expected_retry_target
                    and question.target_expression != expected_retry_target
                ):
                    raise ValueError("semantic retry changed targetExpression")
                if request.domain == PracticeDomain.READING:
                    slot = slot_by_order[order]
                    assert slot.passage_id is not None
                    question_spec = build_reading_question_difficulty_spec(
                        difficulty=slot.difficulty.value,
                        complexity_band=slot.complexity_band,
                        skill_tag=slot.skill_tag,
                        passage_id=slot.passage_id,
                    )
                    validation = project_reading_question_validation(
                        question_spec,
                        lambda: self._validate_candidate(
                            request,
                            question,
                            slot,
                            accepted,
                        ),
                    )
                    if not validation.passed:
                        raise ValueError(validation.primary_issue)
                else:
                    slot = slot_by_order[order]
                    vocabulary_spec = build_vocabulary_difficulty_spec(
                        mode=request.mode,
                        difficulty=slot.difficulty.value,
                        complexity_band=slot.complexity_band,
                        skill_tag=slot.skill_tag,
                        question_type=question.question_type.value,
                        target_expression=question.target_expression or "",
                    )
                    validation = project_vocabulary_validation(
                        vocabulary_spec,
                        lambda: self._validate_candidate(
                            request,
                            question,
                            slot,
                            accepted,
                        ),
                    )
                    if not validation.passed:
                        raise ValueError(validation.primary_issue)
                accepted[order] = question
            except (ValidationError, ValueError) as exc:
                reason = str(exc)
                failures[order] = reason
                logger.warning(
                    "Reading/Vocabulary candidate rejected. request_id=%s order=%d reason=%s",
                    request.request_id,
                    order,
                    reason[:300],
                )
        return failures

    def _normalize_reading_candidate_metadata(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
        slot: _QuestionSlot,
    ) -> None:
        if request.domain != PracticeDomain.READING:
            return

        # vocabularyCandidates is optional enrichment, never part of the question's correctness.
        # Keep only useful exact surface forms from the assigned passage instead of discarding an
        # otherwise-valid Reading question because Luna returned an inflected/paraphrased form.
        passage_text = slot.passage_text or question.passage_text or ""
        question.vocabulary_candidates = self._sanitize_reading_vocabulary_candidates(
            question.vocabulary_candidates,
            passage_text,
        )
        question.review_target = False

    @staticmethod
    def _sanitize_reading_vocabulary_candidates(
        values: Any,
        passage_text: str,
    ) -> list[str]:
        if not isinstance(values, list):
            return []

        sanitized: list[str] = []
        seen: set[str] = set()
        for raw_value in values:
            if not isinstance(raw_value, str):
                continue
            value = raw_value.strip()
            if not value or value in seen or value not in passage_text:
                continue
            seen.add(value)
            sanitized.append(value)
            if len(sanitized) == 3:
                break
        return sanitized

    def _normalize_usage_distinction_candidate_metadata(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
        slot: _QuestionSlot,
    ) -> None:
        if not (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.USAGE_DISTINCTION.value
        ):
            return

        # These fields are application-owned plan metadata, not learner-authored semantics.
        # Normalize harmless model drift instead of wasting a scarce semantic retry.
        question.difficulty = slot.difficulty
        question.complexity_band = slot.complexity_band
        question.skill_tag = slot.skill_tag
        if slot.question_type is not None:
            question.question_type = slot.question_type
        question.passage_id = None
        question.passage_text = None
        question.evidence_text = None
        question.vocabulary_candidates = []

        target = (question.target_expression or "").strip()
        if slot.review_target:
            question.review_target = True
            # Only canonical metadata is repairable. A different target changes the learning item
            # and must still be rejected by deterministic validation below.
            if target and target == (slot.review_expression or ""):
                question.canonical_key = slot.review_canonical_key
        else:
            question.review_target = False
            if target:
                # New mastery identity is application-owned and derived from the exact expression.
                # Do not let an LLM hallucinated/reused key poison uniqueness or review binding.
                question.canonical_key = self._canonical_key_for_expression(target)

        # targetExpression is the trained answer by contract. If it appears exactly once, bind
        # correctAnswer to that option deterministically; Mini still verifies whether the context
        # actually makes that target the best answer.
        normalized_target = self._normalize_for_leak_check(target)
        target_option_keys = [
            option.key
            for option in question.options
            if self._normalize_for_leak_check(option.text) == normalized_target
        ]
        if normalized_target and len(target_option_keys) == 1:
            question.correct_answer = [target_option_keys[0]]

    def _normalize_composition_candidate_metadata(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
        slot: _QuestionSlot,
    ) -> None:
        if not (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.COMPOSITION.value
        ):
            return

        # Difficulty/skill/question type/review flag are deterministic plan metadata. Repair model
        # drift instead of burning a generation attempt on fields the application already owns.
        question.difficulty = slot.difficulty
        question.complexity_band = slot.complexity_band
        question.skill_tag = slot.skill_tag
        if slot.question_type is not None:
            question.question_type = slot.question_type
        question.passage_id = None
        question.passage_text = None
        question.evidence_text = None
        question.vocabulary_candidates = []
        question.review_target = slot.review_target

        target = (question.target_expression or "").strip()
        if slot.review_target:
            # Review identity may never be invented. Only repair canonical metadata after the model
            # has actually preserved the bound expression; a different target still fails below.
            if target and target == (slot.review_expression or ""):
                question.canonical_key = slot.review_canonical_key
            return

        # For NEW ORDERING, the trained expression is the assembled result. If Luna omitted only
        # targetExpression/canonicalKey but returned a valid permutation, reconstruct the mastery
        # identity deterministically instead of discarding an otherwise-good ordering task.
        if not target and question.question_type == PracticeQuestionType.ORDERING:
            target = self._assembled_ordering_expression(question) or ""
            if target:
                question.target_expression = target

        if target:
            question.canonical_key = self._canonical_key_for_expression(target)

    @staticmethod
    def _assembled_ordering_expression(question: PracticeGeneratedQuestion) -> str | None:
        option_by_key = {option.key: option.text for option in question.options}
        option_keys = list(option_by_key)
        if (
            len(option_keys) != len(set(option_keys))
            or len(question.correct_answer) != len(option_keys)
            or set(question.correct_answer) != set(option_keys)
        ):
            return None
        return "".join(option_by_key[key] for key in question.correct_answer).strip() or None

    @staticmethod
    def _reading_retry_feedback(reason: str) -> str:
        if reason == "ambiguous single-choice item":
            return (
                "REPAIR_READING_AMBIGUITY: keep the exact assigned passage. Rewrite only the "
                "question/options so one answer is uniquely supported by a visible passage cue; "
                "replace any rival that could also be accepted."
            )
        if reason == "distractors are too weak or unrelated":
            return (
                "REPAIR_READING_DISTRACTORS: keep the exact assigned passage and intended skill. "
                "Use plausible passage-related distractors that require careful reading to reject, "
                "while preserving exactly one clearly supported answer."
            )
        if reason == "answer is not sufficiently supported":
            return (
                "REPAIR_READING_SUPPORT: keep the exact assigned passage. Rebuild the prompt and "
                "options around evidence that is explicitly present or safely inferable from that passage."
            )
        if reason.startswith("answer mismatch"):
            return (
                "REPAIR_READING_ANSWER: keep the exact assigned passage. Rebuild the question/options "
                "so the declared answer is the single best answer under the passage evidence."
            )
        if reason == "question does not fit requested mode/skill":
            return (
                "REPAIR_READING_MODE: keep the exact assigned passage and fixed skillTag. Rebuild the "
                "question to test that reading skill rather than vocabulary recall or outside knowledge."
            )
        if reason == "semantic verifier detected answer leakage":
            return (
                "REPAIR_READING_LEAKAGE: keep the exact assigned passage but remove wording that "
                "directly gives away the answer; preserve the same reading skill."
            )
        return (
            "REPAIR_READING_CANDIDATE: keep the exact assigned passageId/passageText and fixed slot "
            f"metadata, then repair only this question: {reason}. vocabularyCandidates are optional; "
            "return [] unless each value is copied exactly from passageText."
        )

    @staticmethod
    def _composition_retry_feedback(reason: str, slot: _QuestionSlot) -> str:
        if reason == "ordering answer must use each option exactly once":
            return (
                "REPAIR_ORDERING_KEYS: keep this fixed ORDERING slot. Return 3-8 unique chunks "
                "and correctAnswer containing every option key exactly once, with no duplicate, "
                "missing, or extra key."
            )
        if reason == "ordering answer length mismatch":
            return (
                "REPAIR_ORDERING_LENGTH: keep this fixed ORDERING slot. correctAnswer must have "
                "exactly the same number of keys as options and use every chunk once."
            )
        if reason == "vocabulary question requires targetExpression/canonicalKey":
            if slot.question_type == PracticeQuestionType.ORDERING:
                return (
                    "REPAIR_ORDERING_TARGET: targetExpression is mandatory. For a new ORDERING "
                    "slot, use the exact assembled expression; for a review slot, use the exact "
                    "bound review expression/canonicalKey."
                )
            return "REPAIR_TARGET: return the exact vocabulary targetExpression for this slot."
        if reason == "ambiguous single-choice item":
            return (
                "REPAIR_AMBIGUITY: keep the same targetExpression and rebuild the completion/"
                "collocation context so exactly one option is clearly best. Replace any tied rival."
            )
        if reason == "distractors are too weak or unrelated":
            return (
                "REPAIR_DISTRACTORS: keep the same targetExpression. Use plausible same-neighborhood "
                "completion/collocation rivals that are wrong for a visible lexical or grammatical cue."
            )
        if reason.startswith("answer mismatch"):
            return (
                "REPAIR_ANSWER_MISMATCH: keep the same targetExpression and rebuild the completion "
                "context/options until the target-aligned answer is the single best choice."
            )
        if "duplicate" in reason or "reused a review" in reason:
            return (
                "REPAIR_DUPLICATE_TARGET: generate a genuinely different targetExpression for this "
                "new slot; do not reuse any accepted or review expression."
            )
        return f"REPAIR_COMPOSITION: fix this slot without changing its planned type/skill: {reason}"

    @staticmethod
    def _canonical_key_for_expression(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).strip().casefold()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized[:200]

    @staticmethod
    def _usage_retry_feedback(reason: str) -> str:
        if reason == "ambiguous single-choice item":
            return (
                "REPAIR_AMBIGUITY: keep the same targetExpression. Rewrite the learner-visible "
                "context so one explicit semantic/register/collocation cue makes the target clearly "
                "better than every rival, and replace any tied rival option."
            )
        if reason == "distractors are too weak or unrelated":
            return (
                "REPAIR_DISTRACTORS: keep the same targetExpression. Replace all weak/unrelated "
                "distractors with same-part-of-speech competitors from the same usage neighborhood; "
                "each should be plausible in isolation but clearly lose on a visible context cue."
            )
        if reason.startswith("answer mismatch"):
            return (
                "REPAIR_ANSWER_MISMATCH: keep the same targetExpression. The independent verifier "
                "preferred another option, so strengthen the decisive context and/or replace that "
                "rival until the target is unmistakably the single best answer."
            )
        if reason == "USAGE_DISTINCTION does not require context":
            return (
                "REPAIR_CONTEXT: keep the same targetExpression. Add a learner-visible cue required "
                "to choose among plausible alternatives. For collocation the local lexical frame is "
                "enough; for register show relationship/formality/channel; otherwise add semantic or "
                "pragmatic situation cues."
            )
        if reason == "semantic verifier detected answer leakage":
            return (
                "REPAIR_LEAKAGE: keep the same targetExpression but remove direct quotation, "
                "definition, synonym/paraphrase hints, or wording that gives the answer away."
            )
        if reason == "question does not fit requested mode/skill":
            return (
                "REPAIR_MODE_FIT: keep the same targetExpression and rebuild the item around the "
                "supplied usageIntent/skillTag rather than meaning recall or generic synonym matching."
            )
        return f"REPAIR_SEMANTIC_QUALITY: keep the same targetExpression and fix: {reason}"

    def _validate_candidate(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
        slot: _QuestionSlot,
        accepted: dict[int, PracticeGeneratedQuestion],
    ) -> None:
        if question.order != slot.order:
            raise ValueError("candidate order mismatch")
        if question.difficulty != slot.difficulty:
            raise ValueError("candidate difficulty mismatch")
        if question.complexity_band != slot.complexity_band:
            raise ValueError("candidate complexityBand mismatch")
        if question.skill_tag != slot.skill_tag:
            raise ValueError("candidate skillTag mismatch")
        if slot.question_type is not None and question.question_type != slot.question_type:
            raise ValueError("candidate questionType mismatch")

        self._validate_question_shape(question)
        self._validate_learning_language(request, question)

        if request.domain == PracticeDomain.READING:
            if question.passage_id != slot.passage_id or question.passage_text != slot.passage_text:
                raise ValueError("reading candidate must reuse the exact planned passage")
            if question.target_expression or question.canonical_key:
                raise ValueError("reading question must not become vocabulary item")
            normalized_candidates = [value.strip() for value in question.vocabulary_candidates]
            if len(normalized_candidates) != len(set(normalized_candidates)):
                raise ValueError("reading vocabularyCandidates must be unique")
            if any(not value for value in normalized_candidates):
                raise ValueError("reading vocabularyCandidates must not contain blanks")
            if any(value not in (question.passage_text or "") for value in normalized_candidates):
                raise ValueError("reading vocabularyCandidates must use passage surface forms")
            return

        if question.passage_id is not None or question.passage_text is not None:
            raise ValueError("vocabulary candidate must not include reading passage")
        if not question.target_expression or not question.canonical_key:
            raise ValueError("vocabulary question requires targetExpression/canonicalKey")
        self._validate_vocabulary_surface_language(request, question)
        if question.vocabulary_candidates:
            raise ValueError("vocabulary question must not return reading vocabularyCandidates")

        accepted_keys = {
            item.canonical_key for item in accepted.values() if item.canonical_key and item.order != question.order
        } | {item.canonical_key for item in request.previous_questions if item.canonical_key}
        if question.canonical_key in accepted_keys:
            raise ValueError("vocabulary canonicalKey duplicates an accepted slot")

        normalized_target = self._normalize_for_leak_check(question.target_expression)
        accepted_targets = {
            self._normalize_for_leak_check(item.target_expression or "")
            for item in accepted.values()
            if item.order != question.order and item.target_expression
        } | {
            self._normalize_for_leak_check(item.target_expression)
            for item in request.previous_questions
            if item.target_expression
        }
        if normalized_target and normalized_target in accepted_targets:
            raise ValueError("vocabulary targetExpression duplicates an accepted slot")

        eligible_review_targets = self._eligible_review_targets(request, log_rejections=False)
        review_target_expressions = {
            self._normalize_for_leak_check(target.expression)
            for target in eligible_review_targets
            if target.expression.strip()
        }
        if (
            not slot.review_target
            and normalized_target
            and normalized_target in review_target_expressions
        ):
            raise ValueError("new vocabulary slot reused a review targetExpression")

        if slot.review_target:
            if not question.review_target:
                raise ValueError("planned review slot must set reviewTarget=true")
            if question.canonical_key != slot.review_canonical_key:
                raise ValueError("review slot canonicalKey does not match bound review target")
            if question.target_expression != slot.review_expression:
                raise ValueError("review slot targetExpression does not match bound review target")
        else:
            if question.review_target:
                raise ValueError("new vocabulary slot must set reviewTarget=false")
            review_keys = {target.canonical_key for target in request.review_targets}
            if question.canonical_key in review_keys:
                raise ValueError("new vocabulary slot reused a review canonicalKey")

        if request.mode == VocabularyMode.USAGE_DISTINCTION.value:
            self._validate_usage_distinction_candidate(question)

    def _validate_vocabulary_surface_language(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
    ) -> None:
        assert question.target_expression is not None
        self._assert_vocabulary_surface_language(
            request.learning_language,
            question.target_expression,
            reason=(
                f"vocabulary targetExpression order={question.order} is not a lexical "
                "expression in learningLanguage"
            ),
        )
        for option in question.options:
            self._assert_vocabulary_surface_language(
                request.learning_language,
                option.text,
                reason=(
                    f"vocabulary option order={question.order} key={option.key} is not a lexical "
                    "expression in learningLanguage"
                ),
            )

    @classmethod
    def _assert_vocabulary_surface_language(
        cls,
        language_code: str,
        value: str,
        *,
        reason: str,
    ) -> None:
        text = value.strip()
        if not text:
            raise ValueError(reason)
        language = cls._base_language(language_code)
        has_hangul = bool(re.search(r"[\uac00-\ud7a3]", text))
        has_kana = bool(re.search(r"[\u3040-\u30ff]", text))
        has_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
        has_ascii = bool(re.search(r"[A-Za-z]", text))

        if language == "ja":
            if has_hangul:
                raise ValueError(reason)
            if has_kana or has_cjk:
                return
            if has_ascii and cls._is_allowed_ascii_vocabulary_token(text):
                return
            raise ValueError(reason)
        if language == "ko":
            if has_kana:
                raise ValueError(reason)
            if has_hangul:
                return
            if has_ascii and cls._is_allowed_ascii_vocabulary_token(text):
                return
            raise ValueError(reason)
        if language == "en":
            if has_hangul or has_kana or has_cjk or not has_ascii:
                raise ValueError(reason)
            return
        # Same-script/unknown languages cannot be proven from Unicode ranges. Keep the existing
        # generic language-lane validation as the fallback rather than inventing a false rule.

    @staticmethod
    def _is_allowed_ascii_vocabulary_token(value: str) -> bool:
        compact = value.strip()
        if compact in _ASCII_VOCABULARY_ALLOWLIST:
            return True
        return bool(_ASCII_ACRONYM_RE.fullmatch(compact))

    @staticmethod
    def _base_language(language_code: str) -> str:
        return language_code.strip().lower().split("-")[0].split("_")[0]

    @classmethod
    def _validate_usage_distinction_candidate(cls, question: PracticeGeneratedQuestion) -> None:
        if question.question_type != PracticeQuestionType.SINGLE_CHOICE:
            raise ValueError("USAGE_DISTINCTION must use single-choice questions")
        target = cls._normalize_for_leak_check(question.target_expression or "")
        stem = cls._normalize_for_leak_check(question.prompt)
        if target and target in stem:
            raise ValueError("USAGE_DISTINCTION stem leaks targetExpression")

        option_by_key = {option.key: option.text for option in question.options}
        correct_text = option_by_key.get(question.correct_answer[0], "")
        normalized_correct = cls._normalize_for_leak_check(correct_text)
        if target != normalized_correct:
            raise ValueError("USAGE_DISTINCTION correct option must equal targetExpression")
        if len(normalized_correct) >= 4 and normalized_correct in stem:
            raise ValueError("USAGE_DISTINCTION stem leaks correct option text")

        semantic_relation_patterns = (
            r"同じ意味",
            r"最も近い意味",
            r"近い意味",
            r"言い換え",
            r"意味(?:は|として)",
            r"같은\s*의미",
            r"가장\s*가까운\s*의미",
            r"(?:뜻|의미)(?:은|는)",
            r"same\s+meaning",
            r"closest\s+meaning",
            r"synonym",
        )
        if any(re.search(pattern, question.prompt, re.IGNORECASE) for pattern in semantic_relation_patterns):
            raise ValueError("USAGE_DISTINCTION became a meaning/synonym question")

    @staticmethod
    def _normalize_for_leak_check(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7a3]+", "", value).casefold()

    @staticmethod
    def _validate_question_shape(question: PracticeGeneratedQuestion) -> None:
        option_keys = [option.key for option in question.options]
        option_texts = [option.text.strip().casefold() for option in question.options]
        if len(option_keys) != len(set(option_keys)):
            raise ValueError("duplicate option keys")
        if len(option_texts) != len(set(option_texts)):
            raise ValueError("duplicate option texts")
        if question.question_type == PracticeQuestionType.SINGLE_CHOICE:
            if len(question.options) != 4 or len(question.correct_answer) != 1:
                raise ValueError("single choice must have 4 options and one answer")
            if question.correct_answer[0] not in option_keys:
                raise ValueError("single choice answer key missing from options")
            return
        if len(question.options) < 3 or len(question.options) > 8:
            raise ValueError("ordering must have 3-8 chunks")
        if len(question.correct_answer) != len(option_keys):
            raise ValueError("ordering answer length mismatch")
        if set(question.correct_answer) != set(option_keys):
            raise ValueError("ordering answer must use each option exactly once")

    def _validate_learning_language(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
    ) -> None:
        values = [
            question.passage_text,
            question.prompt,
            question.target_expression,
            question.evidence_text,
            question.explanation_learning,
            *[option.text for option in question.options],
        ]
        self._assert_language_lane(
            request.learning_language,
            [value for value in values if value and value.strip()],
            reason=f"question order={question.order} learner content is not in learningLanguage",
        )

    async def _usage_prescreen_failures(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
    ) -> dict[int, str]:
        if not (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.USAGE_DISTINCTION.value
        ):
            return {}
        single_choice = [
            question for question in questions
            if question.question_type == PracticeQuestionType.SINGLE_CHOICE
        ]
        if not single_choice:
            return {}
        payload_questions = [
            {
                "order": question.order,
                "prompt": question.prompt,
                "options": [option.model_dump(by_alias=True) for option in question.options],
                "skillTag": question.skill_tag,
                **(
                    {"usageIntent": self._usage_intent_for_skill(question.skill_tag)}
                    if request.mode == VocabularyMode.USAGE_DISTINCTION.value
                    else {}
                ),
            }
            for question in single_choice
        ]
        try:
            raw = await asyncio.wait_for(
                self.provider.call(
                    type_name=self.PRESCREEN_TYPE_NAME,
                    data=build_usage_prescreen_prompt(request, payload_questions),
                    schema=_USAGE_PRESCREEN_SCHEMA,
                ),
                timeout=self.timeout_seconds,
            )
            payload = _UsagePrescreenPayload.model_validate(raw)
            verdict_by_order = {verdict.order: verdict for verdict in payload.verdicts}
            expected_orders = {question.order for question in single_choice}
            if set(verdict_by_order) != expected_orders:
                raise ValueError("usage prescreen verdict coverage mismatch")
        except Exception as exc:
            # Nano is a cheap quality gate, not an availability dependency. If it is unavailable,
            # fail open to the Mini verifier rather than failing the whole daily set.
            logger.warning(
                "Reading/Vocabulary Nano prescreen unavailable. request_id=%s type=%s reason=%s",
                request.request_id,
                type(exc).__name__,
                str(exc)[:300],
            )
            return {}

        failures: dict[int, str] = {}
        for order, verdict in verdict_by_order.items():
            reasons: list[str] = []
            if not verdict.modeFit:
                reasons.append("not a usage-distinction task")
            if verdict.answerLeakage:
                reasons.append("answer/target leakage")
            if not verdict.contextDependent:
                reasons.append("context is not required")
            if reasons:
                failures[order] = "; ".join(reasons) + f" ({verdict.reason})"
        return failures

    async def _semantic_failures(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
    ) -> dict[int, str]:
        single_choice = [question for question in questions if question.question_type == PracticeQuestionType.SINGLE_CHOICE]
        if not single_choice:
            return {}
        verification_input = [
            {
                "order": question.order,
                "passageText": question.passage_text,
                "prompt": question.prompt,
                "options": [option.model_dump(by_alias=True) for option in question.options],
                "skillTag": question.skill_tag,
                **(
                    {"usageIntent": self._usage_intent_for_skill(question.skill_tag)}
                    if request.mode == VocabularyMode.USAGE_DISTINCTION.value
                    else {}
                ),
            }
            for question in single_choice
        ]
        expected_by_order = {question.order: question.correct_answer[0] for question in single_choice}

        last_error: Exception | None = None
        for verification_attempt in (1, 2, 3):
            try:
                raw = await asyncio.wait_for(
                    self.provider.call(
                        type_name=self.VERIFICATION_TYPE_NAME,
                        data=build_practice_verification_prompt(request, verification_input),
                        schema=_PRACTICE_VERIFICATION_SCHEMA,
                    ),
                    timeout=self.timeout_seconds,
                )
                payload = _PracticeVerificationPayload.model_validate(raw)
                verdict_by_order = {verdict.order: verdict for verdict in payload.verdicts}
                if set(verdict_by_order) != set(expected_by_order):
                    raise ValueError("semantic verifier verdict coverage mismatch")
                failures: dict[int, str] = {}
                for order, expected_key in expected_by_order.items():
                    verdict = verdict_by_order[order]
                    if request.domain == PracticeDomain.READING:
                        assessment = normalize_reading_semantic_assessment(
                            best_answer_key=verdict.bestAnswerKey,
                            ambiguous=verdict.ambiguous,
                            supported=verdict.supported,
                            mode_fit=verdict.modeFit,
                            answer_leakage=verdict.answerLeakage,
                            distractors_plausible=verdict.distractorsPlausible,
                        )
                        decision = ReadingSemanticQualityPolicy().decide(
                            assessment=assessment,
                            context=ReadingAcceptanceContext(expected_answer_key=expected_key),
                        )
                        if decision.action != "ACCEPT":
                            failures[order] = decision.reason
                    else:
                        assessment = normalize_vocabulary_semantic_assessment(
                            best_answer_key=verdict.bestAnswerKey,
                            ambiguous=verdict.ambiguous,
                            supported=verdict.supported,
                            mode_fit=verdict.modeFit,
                            answer_leakage=verdict.answerLeakage,
                            context_dependent=verdict.contextDependent,
                            distractors_plausible=verdict.distractorsPlausible,
                        )
                        decision = semantic_quality_policy_for_vocabulary_mode(
                            request.mode
                        ).decide(
                            assessment=assessment,
                            context=VocabularyAcceptanceContext(
                                expected_answer_key=expected_key
                            ),
                        )
                        if decision.action != "ACCEPT":
                            failures[order] = decision.reason
                return failures
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Reading/Vocabulary semantic verifier failure. request_id=%s attempt=%d/3 type=%s",
                    request.request_id,
                    verification_attempt,
                    type(exc).__name__,
                )
        raise ValueError(
            "semantic verifier unavailable after provider retries: "
            f"{type(last_error).__name__ if last_error else 'unknown'}"
        )

    async def _attach_origin_explanations(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
    ) -> list[PracticeGeneratedQuestion]:
        by_order = {question.order: question for question in questions}
        explanations: dict[int, str] = {}
        attempts = {question.order: 0 for question in questions}

        while len(explanations) < len(questions):
            pending = [
                order
                for order in sorted(by_order)
                if order not in explanations and attempts[order] < _MAX_ORIGIN_EXPLANATION_ATTEMPTS
            ]
            if not pending:
                break
            for start in range(0, len(pending), _ORIGIN_EXPLANATION_BATCH_SIZE):
                batch_orders = pending[start : start + _ORIGIN_EXPLANATION_BATCH_SIZE]
                for order in batch_orders:
                    attempts[order] += 1
                await self._generate_origin_explanation_batch(
                    request,
                    [by_order[order] for order in batch_orders],
                    explanations,
                    type_name=self.ORIGIN_EXPLANATION_TYPE_NAME,
                )

        remaining = [order for order in sorted(by_order) if order not in explanations]
        if remaining:
            logger.warning(
                "Reading/Vocabulary Nano explanation exhausted; using Mini fallback. request_id=%s orders=%s",
                request.request_id,
                remaining,
            )
            for start in range(0, len(remaining), _ORIGIN_EXPLANATION_BATCH_SIZE):
                batch_orders = remaining[start : start + _ORIGIN_EXPLANATION_BATCH_SIZE]
                await self._generate_origin_explanation_batch(
                    request,
                    [by_order[order] for order in batch_orders],
                    explanations,
                    type_name=self.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME,
                )

        remaining = [order for order in sorted(by_order) if order not in explanations]
        if remaining:
            raise ValueError(f"origin explanation generation exhausted orders={remaining}")

        return [
            question.model_copy(update={"explanation_origin": explanations[question.order]})
            for question in questions
        ]

    async def _generate_origin_explanation_batch(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
        explanations: dict[int, str],
        *,
        type_name: str,
    ) -> None:
        expected_orders = {question.order for question in questions}
        payload_questions = []
        for question in questions:
            correct_text = self._correct_answer_text(question)
            payload_questions.append(
                {
                    "order": question.order,
                    "passageText": question.passage_text,
                    "prompt": question.prompt,
                    "targetExpression": question.target_expression,
                    "correctAnswerText": correct_text,
                    "evidenceText": question.evidence_text,
                    "explanationLearning": question.explanation_learning,
                }
            )
        try:
            raw = await asyncio.wait_for(
                self.provider.call(
                    type_name=type_name,
                    data=build_origin_explanation_prompt(request, payload_questions),
                    schema=_ORIGIN_EXPLANATION_SCHEMA,
                ),
                timeout=self.timeout_seconds,
            )
            payload = _OriginExplanationPayload.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "Reading/Vocabulary origin explanation provider/schema failure. request_id=%s "
                "modelTask=%s orders=%s type=%s",
                request.request_id,
                type_name,
                sorted(expected_orders),
                type(exc).__name__,
            )
            return

        seen: set[int] = set()
        for item in payload.explanations:
            if item.order not in expected_orders or item.order in seen:
                continue
            seen.add(item.order)
            text = item.text.strip()
            if not text:
                continue
            try:
                # Validate each explanation independently. One good Korean explanation can no
                # longer hide nine Japanese explanations in a combined-language check.
                self._assert_language_lane(
                    request.origin_language,
                    [text],
                    reason=f"origin explanation order={item.order} is not in originLanguage",
                    allow_mixed_scripts=True,
                )
            except ValueError as exc:
                logger.warning(
                    "Reading/Vocabulary origin explanation rejected. request_id=%s order=%d "
                    "modelTask=%s reason=%s",
                    request.request_id,
                    item.order,
                    type_name,
                    str(exc)[:240],
                )
                continue
            explanations[item.order] = text

    @staticmethod
    def _correct_answer_text(question: PracticeGeneratedQuestion) -> str:
        option_by_key = {option.key: option.text for option in question.options}
        if question.question_type == PracticeQuestionType.SINGLE_CHOICE:
            return option_by_key.get(question.correct_answer[0], "")
        return " ".join(option_by_key.get(key, "") for key in question.correct_answer).strip()

    def _validate(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
    ) -> None:
        if len(questions) != request.question_count:
            raise ValueError("question count mismatch")
        if [q.order for q in questions] != list(range(1, request.question_count + 1)):
            raise ValueError("question order must be contiguous")

        allowed_skills = (
            {item.value for item in ReadingSkill}
            if request.domain == PracticeDomain.READING
            else {item.value for item in VocabularySkill}
        )
        expected_bands = sorted(
            [max(1, request.complexity_band - 1)] * request.easier_count
            + [request.complexity_band] * request.current_count
            + [min(5, request.complexity_band + 1)] * request.challenge_count
        )
        actual_bands = sorted(q.complexity_band for q in questions)
        if actual_bands != expected_bands:
            raise ValueError(f"complexity mix mismatch: expected={expected_bands} actual={actual_bands}")

        expected_difficulty_counts = {
            PracticeDifficulty.EASIER: request.easier_count,
            PracticeDifficulty.CURRENT: request.current_count,
            PracticeDifficulty.CHALLENGE: request.challenge_count,
        }
        actual_difficulty_counts = {
            difficulty: sum(1 for q in questions if q.difficulty == difficulty)
            for difficulty in expected_difficulty_counts
        }
        if actual_difficulty_counts != expected_difficulty_counts:
            raise ValueError("difficulty label mix mismatch")

        for question in questions:
            if question.skill_tag not in allowed_skills:
                raise ValueError(f"invalid skillTag: {question.skill_tag}")
            self._validate_question_shape(question)
            self._validate_learning_language(request, question)
            self._assert_language_lane(
                request.origin_language,
                [question.explanation_origin],
                reason=f"origin explanation order={question.order} is not in originLanguage",
                allow_mixed_scripts=True,
            )

            if request.domain == PracticeDomain.READING:
                if not question.passage_id or not question.passage_text:
                    raise ValueError("reading question requires passageId/passageText")
                if question.target_expression or question.canonical_key:
                    raise ValueError("reading question must not become vocabulary item")
            else:
                if question.passage_id is not None or question.passage_text is not None:
                    raise ValueError("vocabulary question must not include passage")
                if not question.target_expression or not question.canonical_key:
                    raise ValueError("vocabulary question requires targetExpression/canonicalKey")
                self._validate_vocabulary_surface_language(request, question)

        if request.domain == PracticeDomain.READING:
            passage_text_by_id: dict[str, str] = {}
            for question in [*request.previous_questions, *questions]:
                assert question.passage_id is not None
                assert question.passage_text is not None
                previous = passage_text_by_id.setdefault(question.passage_id, question.passage_text)
                if previous != question.passage_text:
                    raise ValueError("same passageId must reuse identical passageText")
            expected_passages = (
                {"p1", "p2"}
                if request.question_offset + request.question_count > 3
                else {"p1"}
            )
            if set(passage_text_by_id) != expected_passages:
                raise ValueError("reading set must use planned passages")
        else:
            canonical_keys = [
                q.canonical_key for q in [*request.previous_questions, *questions]
            ]
            if len(canonical_keys) != len(set(canonical_keys)):
                raise ValueError("vocabulary set must use unique canonicalKey values")
            review_count = sum(1 for q in questions if q.review_target)
            eligible_reviews = self._eligible_review_targets(request, log_rejections=False)
            expected_review_count = min(request.review_question_count, len(eligible_reviews))
            if review_count != expected_review_count:
                raise ValueError(
                    f"review mix mismatch expected={expected_review_count} actual={review_count}"
                )
            review_keys = {target.canonical_key for target in eligible_reviews}
            for question in questions:
                if question.review_target and question.canonical_key not in review_keys:
                    raise ValueError("review question canonicalKey is not in reviewTargets")
            if request.mode == VocabularyMode.COMPOSITION.value:
                ordering_count = sum(
                    1 for q in questions if q.question_type == PracticeQuestionType.ORDERING
                )
                expected_ordering_count = sum(
                    index in {1, 3, 6, 8}
                    for index in range(
                        request.question_offset + 1,
                        request.question_offset + request.question_count + 1,
                    )
                )
                if ordering_count < expected_ordering_count:
                    raise ValueError("composition requires its planned ordering questions")

    @staticmethod
    def _assert_language_lane(
        language_code: str,
        texts: list[str],
        *,
        reason: str,
        allow_mixed_scripts: bool = False,
    ) -> None:
        language = ReadingVocabularyGenerationService._base_language(language_code)
        normalized = "\n".join(text for text in texts if text and text.strip())
        if not normalized:
            raise ValueError(reason)

        has_hangul = bool(re.search(r"[\uac00-\ud7a3]", normalized))
        has_kana = bool(re.search(r"[\u3040-\u30ff]", normalized))
        has_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", normalized))
        has_ascii = bool(re.search(r"[A-Za-z]", normalized))

        mismatch = False
        missing_expected_script = False
        if language == "ja":
            mismatch = has_hangul
            missing_expected_script = not (has_kana or has_cjk)
        elif language == "ko":
            mismatch = has_kana
            missing_expected_script = not has_hangul
        elif language == "en":
            mismatch = has_hangul or has_kana or has_cjk
            missing_expected_script = not has_ascii
        else:
            # Unknown/same-script languages cannot be proven safely with Unicode ranges.
            # Keep the content non-empty and rely on the generation/explanation stage contract.
            return

        if (mismatch and not allow_mixed_scripts) or missing_expected_script:
            raise ValueError(reason)
