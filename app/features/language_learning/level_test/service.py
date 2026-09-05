from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from difflib import SequenceMatcher

from fastapi import HTTPException
from pydantic import ValidationError

from app.ai.ports import SpeechSynthesisProvider, StructuredTextGenerationProvider
from app.common.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.level_test.audio_upload import (
    PresignedAudioUploadError,
    PresignedAudioUploader,
)
from app.features.language_learning.level_test.normalizer import LevelTestGenerationNormalizer
from app.core.config import settings
from app.features.language_learning.level_test.policy import (
    LEVEL_TEST_EVALUATION_VERSION,
    LEVEL_TEST_GENERATION_VERSION,
    LEVEL_TEST_PROMPT_VERSION,
    LEVEL_TEST_SPEAKING_EVALUATION_VERSION,
    LEVEL_TEST_SPEAKING_PROMPT_VERSION,
    SPEAKING_REPEAT_WEIGHTS,
    SPEAKING_RESPONSE_WEIGHTS,
    validate_recipe,
)
from app.features.language_learning.level_test.prompts import (
    build_level_test_choice_semantic_verification_prompt,
    build_level_test_generation_prompt,
    build_level_test_speaking_evaluation_prompt,
    build_level_test_task_sufficiency_verification_prompt,
    build_level_test_vocab_context_design_prompt,
    build_level_test_vocab_context_repair_prompt,
)
from app.features.language_learning.listening.dictation_service import ListeningDictationService
from app.features.language_learning.listening.interpretation_service import ListeningInterpretationService
from app.features.language_learning.quality import (
    CONTENT_DIVERSITY_POLICY_VERSION,
    DiversityCandidate,
    DiversityValidator,
    DiversityValidationStats,
)
from app.features.language_learning.speaking.audio_processor import SpeakingAudioProcessor
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.stt_service import SpeakingSttService
from app.features.language_learning.writing.service import LanguageLearningWritingService
from app.schemas.language_learning import WritingEvaluationRequest, WritingMetric
from app.schemas.language_learning_level_test import (
    LevelTestAnswerMode,
    LevelTestAssessmentSignal,
    LevelTestBestOptionAdvantage,
    LevelTestChoiceSelectionPolicy,
    LevelTestChoiceSemanticVerificationPayload,
    LevelTestDomain,
    LevelTestEvaluationResponse,
    LevelTestFeedbackDetail,
    LevelTestItemType,
    LevelTestMetricResult,
    LevelTestMetricState,
    LevelTestQuestionCandidate,
    LevelTestQuestionGenerationPayload,
    LevelTestQuestionGenerationRequest,
    LevelTestQuestionGenerationResponse,
    LevelTestReferenceAudio,
    LevelTestScoredInternalAnswerKey,
    LevelTestSpeakingEvaluationContext,
    LevelTestSpeakingEvaluationPayload,
    LevelTestTaskSufficiencyVerificationPayload,
    LevelTestTextEvaluationRequest,
    LevelTestUsage,
    LevelTestVocabContextDesign,
    LevelTestVocabContextDesignPayload,
    LevelTestVocabContextRepairPayload,
)
from app.schemas.language_learning_listening import (
    DictationEvaluationRequest,
    EvaluationPurpose,
    InterpretationEvaluationRequest,
    ListeningTaskType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StageUsage:
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _GenerationTelemetry:
    design_calls: int = 0
    design_fallbacks: int = 0
    generation_calls: int = 0
    verifier_calls: int = 0
    task_verifier_calls: int = 0
    task_rejections: int = 0
    repair_calls: int = 0
    repair_accepted: int = 0
    candidates_seen: int = 0
    deterministic_rejections: int = 0
    semantic_rejections: int = 0
    semantic_unverifiable: int = 0
    schema_rejections: int = 0


@dataclass
class _GenerationTelemetryAggregate:
    requests: int = 0
    accepted: int = 0
    total_ai_calls: int = 0
    design_calls: int = 0
    design_fallbacks: int = 0
    generation_calls: int = 0
    verifier_calls: int = 0
    task_verifier_calls: int = 0
    task_rejections: int = 0
    repair_calls: int = 0
    repair_accepted: int = 0
    semantic_rejections: int = 0
    semantic_unverifiable: int = 0
    deterministic_rejections: int = 0
    schema_rejections: int = 0


class LevelTestService:
    GENERATION_TYPE = "LANGUAGE_LEARNING_LEVEL_TEST_V2_GENERATION"
    VOCAB_CONTEXT_DESIGN_TYPE = "LANGUAGE_LEARNING_LEVEL_TEST_V2_VOCAB_CONTEXT_DESIGN"
    VOCAB_CONTEXT_REPAIR_TYPE = "LANGUAGE_LEARNING_LEVEL_TEST_V2_VOCAB_CONTEXT_REPAIR"
    CHOICE_VERIFICATION_TYPE = "LANGUAGE_LEARNING_LEVEL_TEST_V2_CHOICE_VERIFICATION"
    TASK_SUFFICIENCY_VERIFICATION_TYPE = "LANGUAGE_LEARNING_LEVEL_TEST_V2_TASK_SUFFICIENCY_VERIFICATION"
    SPEAKING_EVALUATION_TYPE = "LANGUAGE_LEARNING_LEVEL_TEST_V2_SPEAKING_EVALUATION"

    def __init__(
        self,
        *,
        provider: StructuredTextGenerationProvider,
        writing_service: LanguageLearningWritingService,
        dictation_service: ListeningDictationService,
        interpretation_service: ListeningInterpretationService,
        audio_processor: SpeakingAudioProcessor,
        stt_service: SpeakingSttService,
        speech_provider: SpeechSynthesisProvider | None = None,
        audio_uploader: PresignedAudioUploader | None = None,
        generation_timeout_seconds: float | None = None,
        evaluation_timeout_seconds: float | None = None,
        question_store: InMemoryIdempotencyStore[LevelTestQuestionGenerationResponse] | None = None,
        text_evaluation_store: InMemoryIdempotencyStore[LevelTestEvaluationResponse] | None = None,
        speaking_evaluation_store: InMemoryIdempotencyStore[LevelTestEvaluationResponse] | None = None,
        verify_vocab_context_semantics: bool = True,
        enable_vocab_context_multistage: bool = True,
        verify_task_sufficiency: bool = True,
        telemetry_summary_interval: int = 20,
    ) -> None:
        self.provider = provider
        self.writing_service = writing_service
        self.dictation_service = dictation_service
        self.interpretation_service = interpretation_service
        self.audio_processor = audio_processor
        self.stt_service = stt_service
        self.speech_provider = speech_provider
        self.audio_uploader = audio_uploader or PresignedAudioUploader()
        self.generation_timeout_seconds = generation_timeout_seconds or settings.AI_LANGUAGE_LEARNING_LEVEL_TEST_TIMEOUT_SECONDS
        self.evaluation_timeout_seconds = evaluation_timeout_seconds or settings.AI_SPEAKING_EVALUATION_TIMEOUT_SECONDS
        self.question_store = question_store or InMemoryIdempotencyStore()
        self.text_evaluation_store = text_evaluation_store or InMemoryIdempotencyStore()
        self.speaking_evaluation_store = speaking_evaluation_store or InMemoryIdempotencyStore()
        # Backward-compatible constructor name: this guard now covers all semantic
        # multiple-choice types, not only VOCAB_CONTEXT_CHOICE.
        self.verify_vocab_context_semantics = verify_vocab_context_semantics
        self.verify_choice_semantics = verify_vocab_context_semantics
        self.enable_vocab_context_multistage = enable_vocab_context_multistage
        self.verify_task_sufficiency = verify_task_sufficiency
        self.telemetry_summary_interval = max(1, telemetry_summary_interval)
        self._generation_telemetry_aggregates: dict[tuple[LevelTestItemType, int], _GenerationTelemetryAggregate] = {}

    async def generate_question(
        self,
        request: LevelTestQuestionGenerationRequest,
    ) -> LevelTestQuestionGenerationResponse:
        try:
            validate_recipe(request.question_number, request.domain, request.item_type)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        key = "|".join(
            [
                request.idempotency_key,
                LEVEL_TEST_GENERATION_VERSION,
                request.policy_version,
                request.model_config_version,
                str(request.target_complexity_band),
            ]
        )
        response, _ = await self.question_store.execute(
            key,
            lambda: self._generate_question_once(request),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _generate_question_once(
        self,
        request: LevelTestQuestionGenerationRequest,
    ) -> LevelTestQuestionGenerationResponse:
        stats = DiversityValidationStats()
        telemetry = _GenerationTelemetry()
        all_candidates: list[tuple[LevelTestQuestionCandidate, dict[str, int]]] = []
        rejected_summaries: list[str] = []
        content_rejection_count = 0
        schema_rejection_count = 0
        rejection_reasons: Counter[str] = Counter()
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0
        last_provider: str | None = None
        last_model: str | None = None
        staged_vocab = (
            self.enable_vocab_context_multistage
            and self.verify_choice_semantics
            and request.item_type == LevelTestItemType.VOCAB_CONTEXT_CHOICE
        )

        for provider_attempt in range(3):
            vocab_designs: list[LevelTestVocabContextDesign] | None = None
            if staged_vocab:
                telemetry.design_calls += 1
                vocab_designs, design_usage = await self._plan_vocab_context_designs(
                    request,
                    refill_attempt=provider_attempt,
                    rejected_summaries=rejected_summaries,
                )
                total_latency_ms += design_usage.latency_ms
                total_input_tokens += design_usage.input_tokens
                total_output_tokens += design_usage.output_tokens
                if vocab_designs is None:
                    # Planning is an optimization, not an availability dependency.
                    # The existing independent verifier still fails closed.
                    telemetry.design_fallbacks += 1

            design_map = {
                design.design_id: design
                for design in (vocab_designs or [])
            }
            schema = self._generation_schema(
                request,
                require_vocab_plan_id=bool(design_map),
            )
            selection_policy = self._choice_selection_policy(
                request.item_type,
                request.target_complexity_band,
            )
            prompt = build_level_test_generation_prompt(
                request,
                refill_attempt=provider_attempt,
                rejected_summaries=rejected_summaries[-12:],
                vocab_context_designs=vocab_designs,
                selection_policy=(selection_policy.value if selection_policy is not None else None),
            )
            started = time.perf_counter()
            telemetry.generation_calls += 1
            try:
                result = await asyncio.wait_for(
                    self.provider.call_with_metadata(
                        type_name=self.GENERATION_TYPE,
                        data=prompt,
                        schema=schema,
                    ),
                    timeout=self.generation_timeout_seconds,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                if provider_attempt < 2:
                    continue
                self._log_generation_telemetry(request, telemetry, outcome="PROVIDER_TIMEOUT")
                raise HTTPException(status_code=504, detail="Level Test 문제 생성 시간이 초과되었습니다.") from exc
            except Exception as exc:
                if provider_attempt < 2 and self._is_transient_provider_error(exc):
                    continue
                self._log_generation_telemetry(request, telemetry, outcome="PROVIDER_ERROR")
                raise HTTPException(status_code=502, detail="Level Test 문제 생성 Provider 호출에 실패했습니다.") from exc

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            total_latency_ms += elapsed_ms
            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens
            last_provider = result.provider
            last_model = result.model

            if not isinstance(result.data, dict):
                self._log_generation_telemetry(request, telemetry, outcome="INVALID_SCHEMA")
                raise HTTPException(status_code=502, detail="Level Test 문제 생성 응답 Schema가 유효하지 않습니다.")
            provider_data, normalization = LevelTestGenerationNormalizer.normalize(
                result.data,
                origin_language=request.origin_language,
                learning_language=request.learning_language,
            )
            if normalization.total:
                logger.info(
                    "Level Test generation payload normalized. request_id=%s "
                    "promptFallbacks=%d taskArchetypeCompactions=%d semanticSummaryCompactions=%d "
                    "maxAudioAdjustments=%d maxAnswerLengthAdjustments=%d answerLanguageAdjustments=%d "
                    "promptMarkupSanitizations=%d instructionLanguageRepairs=%d sentenceOrderShuffles=%d "
                    "optionKeyCanonicalizations=%d listeningSourceAliasRepairs=%d referencePayloadShapeRepairs=%d "
                    "readingStructureRepairs=%d listeningStructureRepairs=%d vocabEmphasisRepairs=%d "
                    "generationPlanIdRepairs=%d enumTokenRepairs=%d writingTranslationRepairs=%d",
                    request.request_id,
                    normalization.prompt_text_fallbacks,
                    normalization.task_archetype_compactions,
                    normalization.semantic_summary_compactions,
                    normalization.max_audio_seconds_adjustments,
                    normalization.max_answer_length_adjustments,
                    normalization.answer_language_adjustments,
                    normalization.prompt_markup_sanitizations,
                    normalization.instruction_language_repairs,
                    normalization.sentence_order_shuffles,
                    normalization.option_key_canonicalizations,
                    normalization.listening_source_alias_repairs,
                    normalization.reference_payload_shape_repairs,
                    normalization.reading_structure_repairs,
                    normalization.listening_structure_repairs,
                    normalization.vocab_emphasis_repairs,
                    normalization.generation_plan_id_repairs,
                    normalization.enum_token_repairs,
                    normalization.writing_translation_repairs,
                )

            parsed_candidates, schema_reasons, rejected_candidates = (
                self._parse_generation_candidates(provider_data)
            )
            if schema_reasons:
                schema_rejection_count += rejected_candidates
                telemetry.schema_rejections += rejected_candidates
                rejection_reasons.update(schema_reasons)
                rejected_summaries.extend(schema_reasons)
                if parsed_candidates:
                    logger.info(
                        "Level Test generation candidate schema partial salvage. request_id=%s attempt=%d "
                        "validCandidates=%d rejectedCandidates=%d reasons=%s",
                        request.request_id,
                        provider_attempt + 1,
                        len(parsed_candidates),
                        rejected_candidates,
                        schema_reasons,
                    )
                else:
                    logger.warning(
                        "Level Test generation candidates rejected by schema. request_id=%s attempt=%d "
                        "validCandidates=0 rejectedCandidates=%d reasons=%s",
                        request.request_id,
                        provider_attempt + 1,
                        rejected_candidates,
                        schema_reasons,
                    )
            if not parsed_candidates:
                continue

            for candidate_index, generated_candidate in parsed_candidates:
                telemetry.candidates_seen += 1
                candidate = generated_candidate
                design: LevelTestVocabContextDesign | None = None
                quality_failure_occurred = False
                option_scores: dict[str, int] = {}
                try:
                    self._validate_question_candidate(request, candidate)
                    if design_map:
                        design = self._validate_vocab_context_design_binding(candidate, design_map)

                    selection_policy = self._choice_selection_policy(
                        candidate.item_type,
                        candidate.complexity_band,
                    )
                    if self.verify_choice_semantics and selection_policy is not None:
                        semantic_reason, verdict, verification_usage, verifier_calls, option_scores = (
                            await self._verify_choice_candidate(
                                request,
                                candidate,
                                selection_policy,
                            )
                        )
                        telemetry.verifier_calls += verifier_calls
                        total_latency_ms += verification_usage.latency_ms
                        total_input_tokens += verification_usage.input_tokens
                        total_output_tokens += verification_usage.output_tokens

                        if semantic_reason is not None:
                            quality_failure_occurred = True
                            telemetry.semantic_rejections += 1
                            if verdict is not None and not verdict.verifiable:
                                telemetry.semantic_unverifiable += 1

                            repaired_candidate: LevelTestQuestionCandidate | None = None
                            if (
                                staged_vocab
                                and design is not None
                                and verdict is not None
                                and self._vocab_context_repairable_semantic_failure(
                                    candidate,
                                    verdict,
                                    selection_policy,
                                )
                            ):
                                telemetry.repair_calls += 1
                                repaired_candidate, repair_usage = await self._repair_vocab_context_candidate(
                                    request,
                                    candidate,
                                    design,
                                    verdict,
                                    selection_policy,
                                )
                                total_latency_ms += repair_usage.latency_ms
                                total_input_tokens += repair_usage.input_tokens
                                total_output_tokens += repair_usage.output_tokens

                            if repaired_candidate is not None:
                                self._validate_question_candidate(request, repaired_candidate)
                                self._validate_vocab_context_design_binding(repaired_candidate, design_map)
                                repair_reason, repair_verdict, repair_usage, repair_verifier_calls, repair_option_scores = (
                                    await self._verify_choice_candidate(
                                        request,
                                        repaired_candidate,
                                        selection_policy,
                                    )
                                )
                                telemetry.verifier_calls += repair_verifier_calls
                                total_latency_ms += repair_usage.latency_ms
                                total_input_tokens += repair_usage.input_tokens
                                total_output_tokens += repair_usage.output_tokens
                                if repair_reason is None:
                                    candidate = repaired_candidate
                                    option_scores = repair_option_scores
                                    telemetry.repair_accepted += 1
                                else:
                                    telemetry.semantic_rejections += 1
                                    if repair_verdict is not None and not repair_verdict.verifiable:
                                        telemetry.semantic_unverifiable += 1
                                    raise ValueError(repair_reason)
                            else:
                                raise ValueError(semantic_reason)

                    if self.verify_task_sufficiency and self._requires_task_sufficiency_verification(candidate.item_type):
                        task_reason, task_usage = await self._verify_task_sufficiency_candidate(
                            request,
                            candidate,
                        )
                        telemetry.task_verifier_calls += 1
                        total_latency_ms += task_usage.latency_ms
                        total_input_tokens += task_usage.input_tokens
                        total_output_tokens += task_usage.output_tokens
                        if task_reason is not None:
                            quality_failure_occurred = True
                            telemetry.task_rejections += 1
                            raise ValueError(task_reason)
                except ValueError as exc:
                    content_rejection_count += 1
                    reason = str(exc) or exc.__class__.__name__
                    if not quality_failure_occurred:
                        telemetry.deterministic_rejections += 1
                    rejection_reasons[reason] += 1
                    rejected_summaries.append(reason)
                    logger.warning(
                        "Level Test generation candidate rejected. request_id=%s attempt=%d candidateIndex=%d "
                        "domain=%s itemType=%s band=%s reason=%s",
                        request.request_id,
                        provider_attempt + 1,
                        candidate_index,
                        generated_candidate.domain.value,
                        generated_candidate.item_type.value,
                        generated_candidate.complexity_band,
                        reason,
                    )
                    continue

                all_candidates.append((candidate, option_scores))
                decision = DiversityValidator(request.diversity_context).validate(
                    DiversityCandidate(
                        content=self._diversity_content(candidate),
                        metadata=candidate.diversity_metadata,
                    ),
                    [],
                )
                stats.record(decision)
                if decision.accepted:
                    self._log_generation_telemetry(request, telemetry, outcome="ACCEPTED")
                    return await self._question_response_with_reference_audio(
                        request,
                        candidate.model_copy(update={"diversity_metadata": decision.metadata}),
                        stats=stats,
                        fallback_used=False,
                        latency_ms=total_latency_ms,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        provider=last_provider,
                        model=last_model,
                        option_scores=option_scores,
                        selection_policy=selection_policy,
                    )
                rejected_summaries.append(candidate.diversity_metadata.semantic_summary)

        relaxed = DiversityValidator(request.diversity_context, relaxed_history=True)
        for candidate, option_scores in all_candidates:
            selection_policy = self._choice_selection_policy(
                candidate.item_type,
                candidate.complexity_band,
            )
            decision = relaxed.validate(
                DiversityCandidate(
                    content=self._diversity_content(candidate),
                    metadata=candidate.diversity_metadata,
                ),
                [],
            )
            if decision.accepted:
                self._log_generation_telemetry(request, telemetry, outcome="ACCEPTED_RELAXED_DIVERSITY")
                return await self._question_response_with_reference_audio(
                    request,
                    candidate.model_copy(update={"diversity_metadata": decision.metadata}),
                    stats=stats,
                    fallback_used=True,
                    latency_ms=total_latency_ms,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    provider=last_provider,
                    model=last_model,
                    option_scores=option_scores,
                    selection_policy=selection_policy,
                )

        if not all_candidates and (content_rejection_count > 0 or schema_rejection_count > 0):
            reasons = self._summarize_rejection_reasons(rejection_reasons)
            logger.warning(
                "Level Test generation exhausted by content quality. request_id=%s contentRejections=%d "
                "schemaRejections=%d reasons=%s",
                request.request_id,
                content_rejection_count,
                schema_rejection_count,
                reasons,
            )
            self._log_generation_telemetry(request, telemetry, outcome="CONTENT_REJECTED")
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "QUESTION_CONTENT_INVALID",
                    "retryable": True,
                    "message": "Level Test 문항 내용 품질 기준을 만족하는 문제를 생성하지 못했습니다.",
                    "reasons": reasons,
                },
            )

        self._log_generation_telemetry(request, telemetry, outcome="DIVERSITY_EXHAUSTED")
        raise HTTPException(
            status_code=422,
            detail={
                "code": "CONTENT_DIVERSITY_EXHAUSTED",
                "retryable": False,
                "message": "Level Test 중복 방지 기준을 만족하는 문제를 생성하지 못했습니다.",
            },
        )

    async def _plan_vocab_context_designs(
        self,
        request: LevelTestQuestionGenerationRequest,
        *,
        refill_attempt: int,
        rejected_summaries: list[str],
    ) -> tuple[list[LevelTestVocabContextDesign] | None, _StageUsage]:
        prompt = build_level_test_vocab_context_design_prompt(
            request,
            refill_attempt=refill_attempt,
            rejected_summaries=rejected_summaries,
        )
        schema = copy.deepcopy(LevelTestVocabContextDesignPayload.model_json_schema())
        preferred = [category.value for category in request.preferred_scenario_categories]
        design_schema = schema.get("$defs", {}).get("LevelTestVocabContextDesign")
        design_properties = (
            design_schema.get("properties")
            if isinstance(design_schema, dict)
            else None
        )
        if preferred and isinstance(design_properties, dict):
            design_properties["scenarioCategory"] = {
                "type": "string",
                "enum": preferred,
            }
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.provider.call_with_metadata(
                    type_name=self.VOCAB_CONTEXT_DESIGN_TYPE,
                    data=prompt,
                    schema=schema,
                ),
                timeout=self.generation_timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "Level Test vocab design unavailable; falling back to verified direct generation. "
                "request_id=%s attempt=%d error=%s",
                request.request_id,
                refill_attempt + 1,
                exc.__class__.__name__,
            )
            return None, _StageUsage(
                latency_ms=int((time.perf_counter() - started) * 1000)
            )

        usage = _StageUsage(
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        if not isinstance(result.data, dict):
            logger.warning(
                "Level Test vocab design schema invalid; falling back to verified direct generation. request_id=%s",
                request.request_id,
            )
            return None, usage
        normalized_design_data = self._normalize_vocab_context_design_payload(result.data)
        try:
            payload = LevelTestVocabContextDesignPayload.model_validate(normalized_design_data)
        except ValidationError:
            logger.warning(
                "Level Test vocab design validation failed; falling back to verified direct generation. request_id=%s",
                request.request_id,
            )
            return None, usage
        preferred_categories = set(request.preferred_scenario_categories)
        if preferred_categories and any(
            design.scenario_category not in preferred_categories
            for design in payload.designs
        ):
            logger.warning(
                "Level Test vocab design ignored because it violated scenario balance. request_id=%s preferred=%s",
                request.request_id,
                [category.value for category in request.preferred_scenario_categories],
            )
            return None, usage
        return payload.designs, usage

    @staticmethod
    def _normalize_vocab_context_design_payload(data: dict) -> dict:
        """Normalize only structural design noise; never invent semantic content."""

        raw_designs = data.get("designs")
        if not isinstance(raw_designs, list):
            return data

        allowed_fields = (
            ("targetExpression", "target_expression"),
            ("targetMeaning", "target_meaning"),
            ("semanticConstraint", "semantic_constraint"),
            ("scenarioCategory", "scenario_category"),
            ("communicativeIntent", "communicative_intent"),
        )
        normalized: list[dict] = []
        for index, raw in enumerate(raw_designs[:2]):
            if not isinstance(raw, dict):
                continue
            item: dict = {"designId": "AB"[index]}
            for camel, snake in allowed_fields:
                if camel in raw:
                    item[camel] = raw[camel]
                elif snake in raw:
                    item[camel] = raw[snake]
            normalized.append(item)
        return {"designs": normalized}

    async def _repair_vocab_context_candidate(
        self,
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
        design: LevelTestVocabContextDesign,
        verdict: LevelTestChoiceSemanticVerificationPayload,
        selection_policy: LevelTestChoiceSelectionPolicy,
    ) -> tuple[LevelTestQuestionCandidate | None, _StageUsage]:
        prompt = build_level_test_vocab_context_repair_prompt(
            candidate,
            design,
            verdict,
            selection_policy=selection_policy.value,
        )
        schema = LevelTestVocabContextRepairPayload.model_json_schema()
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.provider.call_with_metadata(
                    type_name=self.VOCAB_CONTEXT_REPAIR_TYPE,
                    data=prompt,
                    schema=schema,
                ),
                timeout=self.generation_timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "Level Test vocab repair failed; candidate will be discarded. request_id=%s planId=%s error=%s",
                request.request_id,
                design.design_id,
                exc.__class__.__name__,
            )
            return None, _StageUsage(
                latency_ms=int((time.perf_counter() - started) * 1000)
            )

        usage = _StageUsage(
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        if not isinstance(result.data, dict) or not isinstance(result.data.get("candidate"), dict):
            return None, usage
        normalized, _ = LevelTestGenerationNormalizer.normalize(
            {"candidates": [result.data["candidate"]]},
            origin_language=request.origin_language,
            learning_language=request.learning_language,
        )
        candidates = normalized.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None, usage
        try:
            repaired = LevelTestQuestionCandidate.model_validate(candidates[0])
        except ValidationError:
            return None, usage
        return repaired, usage

    @staticmethod
    def _validate_vocab_context_design_binding(
        candidate: LevelTestQuestionCandidate,
        design_map: dict[str, LevelTestVocabContextDesign],
    ) -> LevelTestVocabContextDesign:
        plan_id = candidate.generation_plan_id
        if not plan_id or plan_id not in design_map:
            raise ValueError("VOCAB_CONTEXT_CHOICE가 서버 승인 generationPlanId를 따르지 않았습니다.")
        design = design_map[plan_id]
        correct_key = candidate.internal_answer_key.correct_option_key
        correct = next((option for option in candidate.options if option.key == correct_key), None)
        if correct is None:
            raise ValueError("VOCAB_CONTEXT_CHOICE 정답 Option을 확인할 수 없습니다.")
        if LevelTestService._normalize_choice_text(correct.text) != LevelTestService._normalize_choice_text(
            design.target_expression
        ):
            raise ValueError("VOCAB_CONTEXT_CHOICE가 서버 승인 targetExpression을 정답으로 사용하지 않았습니다.")
        if candidate.diversity_metadata.scenario_category != design.scenario_category:
            raise ValueError("VOCAB_CONTEXT_CHOICE가 서버 승인 scenarioCategory를 변경했습니다.")
        if candidate.diversity_metadata.communicative_intent != design.communicative_intent:
            raise ValueError("VOCAB_CONTEXT_CHOICE가 서버 승인 communicativeIntent를 변경했습니다.")
        return design

    @staticmethod
    def _requires_task_sufficiency_verification(item_type: LevelTestItemType) -> bool:
        return item_type in {
            LevelTestItemType.WRITING_GUIDED_SENTENCE,
            LevelTestItemType.WRITING_SCENARIO_RESPONSE,
            LevelTestItemType.WRITING_SHORT_PARAGRAPH,
            LevelTestItemType.SPEAKING_GUIDED_RESPONSE,
            LevelTestItemType.SPEAKING_SHORT_RESPONSE,
        }

    async def _call_task_sufficiency_verifier(
        self,
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
    ) -> tuple[LevelTestTaskSufficiencyVerificationPayload, _StageUsage]:
        prompt = build_level_test_task_sufficiency_verification_prompt(
            candidate,
            origin_language=request.origin_language,
            learning_language=request.learning_language,
        )
        schema = LevelTestTaskSufficiencyVerificationPayload.model_json_schema()
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.provider.call_with_metadata(
                    type_name=self.TASK_SUFFICIENCY_VERIFICATION_TYPE,
                    data=prompt,
                    schema=schema,
                ),
                timeout=self.generation_timeout_seconds,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise HTTPException(
                status_code=504,
                detail="Level Test 가이드 충분성 검증 시간이 초과되었습니다.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Level Test 가이드 충분성 검증 Provider 호출에 실패했습니다.",
            ) from exc

        usage = _StageUsage(
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        if not isinstance(result.data, dict):
            raise HTTPException(
                status_code=502,
                detail="Level Test 가이드 충분성 검증 응답 Schema가 유효하지 않습니다.",
            )
        try:
            verdict = LevelTestTaskSufficiencyVerificationPayload.model_validate(result.data)
        except ValidationError as exc:
            raise HTTPException(
                status_code=502,
                detail="Level Test 가이드 충분성 검증 응답 Schema가 유효하지 않습니다.",
            ) from exc
        return verdict, usage

    async def _verify_task_sufficiency_candidate(
        self,
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
    ) -> tuple[str | None, _StageUsage]:
        verdict, usage = await self._call_task_sufficiency_verifier(request, candidate)
        if (
            verdict.sufficient
            and not verdict.requires_external_knowledge
            and not verdict.requires_problem_solving
            and verdict.provided_facts_sufficient
            and verdict.communicative_goals_clear
            and verdict.instruction_and_task_roles_separated
        ):
            return None, usage

        reasons: list[str] = []
        if verdict.requires_external_knowledge:
            reasons.append("외부 지식 필요")
        if verdict.requires_problem_solving:
            reasons.append("상황 판단/문제 해결 요구")
        if not verdict.provided_facts_sufficient:
            reasons.append("제공 사실 부족")
        if not verdict.communicative_goals_clear:
            reasons.append("의사기능 가이드 불명확")
        if not verdict.instruction_and_task_roles_separated:
            reasons.append("origin/learning 언어 역할 중복")
        if verdict.missing_information:
            reasons.append("누락=" + ",".join(verdict.missing_information[:3]))
        if not reasons:
            reasons.append("독립 검증 실패")
        return (
            f"{candidate.item_type.value} 가이드 충분성 검증에 실패했습니다: " + "; ".join(reasons),
            usage,
        )

    def _log_generation_telemetry(
        self,
        request: LevelTestQuestionGenerationRequest,
        telemetry: _GenerationTelemetry,
        *,
        outcome: str,
    ) -> None:
        total_ai_calls = (
            telemetry.design_calls
            + telemetry.generation_calls
            + telemetry.verifier_calls
            + telemetry.task_verifier_calls
            + telemetry.repair_calls
        )
        selection_policy = self._choice_selection_policy(
            request.item_type,
            request.target_complexity_band,
        )
        logger.info(
            "Level Test generation telemetry. request_id=%s domain=%s itemType=%s band=%s selectionPolicy=%s outcome=%s "
            "totalAiCalls=%d designCalls=%d designFallbacks=%d generationCalls=%d verifierCalls=%d taskVerifierCalls=%d "
            "taskRejections=%d repairCalls=%d repairAccepted=%d candidatesSeen=%d deterministicRejections=%d semanticRejections=%d "
            "semanticUnverifiable=%d schemaRejections=%d preferredScenarios=%s",
            request.request_id,
            request.domain.value,
            request.item_type.value,
            request.target_complexity_band,
            selection_policy.value if selection_policy is not None else "NONE",
            outcome,
            total_ai_calls,
            telemetry.design_calls,
            telemetry.design_fallbacks,
            telemetry.generation_calls,
            telemetry.verifier_calls,
            telemetry.task_verifier_calls,
            telemetry.task_rejections,
            telemetry.repair_calls,
            telemetry.repair_accepted,
            telemetry.candidates_seen,
            telemetry.deterministic_rejections,
            telemetry.semantic_rejections,
            telemetry.semantic_unverifiable,
            telemetry.schema_rejections,
            [category.value for category in request.preferred_scenario_categories],
        )

        key = (request.item_type, request.target_complexity_band)
        aggregate = self._generation_telemetry_aggregates.setdefault(
            key,
            _GenerationTelemetryAggregate(),
        )
        aggregate.requests += 1
        aggregate.accepted += int(outcome.startswith("ACCEPTED"))
        aggregate.total_ai_calls += total_ai_calls
        aggregate.design_calls += telemetry.design_calls
        aggregate.design_fallbacks += telemetry.design_fallbacks
        aggregate.generation_calls += telemetry.generation_calls
        aggregate.verifier_calls += telemetry.verifier_calls
        aggregate.task_verifier_calls += telemetry.task_verifier_calls
        aggregate.task_rejections += telemetry.task_rejections
        aggregate.repair_calls += telemetry.repair_calls
        aggregate.repair_accepted += telemetry.repair_accepted
        aggregate.semantic_rejections += telemetry.semantic_rejections
        aggregate.semantic_unverifiable += telemetry.semantic_unverifiable
        aggregate.deterministic_rejections += telemetry.deterministic_rejections
        aggregate.schema_rejections += telemetry.schema_rejections

        if aggregate.requests % self.telemetry_summary_interval != 0:
            return

        acceptance_rate = 100.0 * aggregate.accepted / aggregate.requests
        avg_ai_calls = aggregate.total_ai_calls / aggregate.requests
        design_fallback_rate = (
            0.0
            if aggregate.design_calls == 0
            else 100.0 * aggregate.design_fallbacks / aggregate.design_calls
        )
        repair_success_rate = (
            0.0
            if aggregate.repair_calls == 0
            else 100.0 * aggregate.repair_accepted / aggregate.repair_calls
        )
        logger.info(
            "Level Test generation quality summary. itemType=%s band=%s selectionPolicy=%s requests=%d "
            "accepted=%d acceptanceRate=%.1f avgAiCalls=%.2f designFallbackRate=%.1f verifierCalls=%d "
            "taskVerifierCalls=%d taskRejections=%d semanticRejections=%d semanticUnverifiable=%d repairCalls=%d repairAccepted=%d repairSuccessRate=%.1f "
            "deterministicRejections=%d schemaRejections=%d",
            request.item_type.value,
            request.target_complexity_band,
            selection_policy.value if selection_policy is not None else "NONE",
            aggregate.requests,
            aggregate.accepted,
            acceptance_rate,
            avg_ai_calls,
            design_fallback_rate,
            aggregate.verifier_calls,
            aggregate.task_verifier_calls,
            aggregate.task_rejections,
            aggregate.semantic_rejections,
            aggregate.semantic_unverifiable,
            aggregate.repair_calls,
            aggregate.repair_accepted,
            repair_success_rate,
            aggregate.deterministic_rejections,
            aggregate.schema_rejections,
        )

    @staticmethod
    def _generation_schema(
        request: LevelTestQuestionGenerationRequest,
        *,
        require_vocab_plan_id: bool = False,
    ) -> dict:
        """Specialize the provider schema for reference fields required by this recipe.

        Every generation request targets one item type, so the provider contract can
        be stricter than the reusable Pydantic model. This prevents LISTENING source
        scripts (and SPEAKING_REPEAT reference text) from depending on prompt wording
        alone while preserving nullable fields for unrelated item types.
        """

        schema = copy.deepcopy(LevelTestQuestionGenerationPayload.model_json_schema())
        root_properties = schema.get("properties")
        candidates_schema = (
            root_properties.get("candidates")
            if isinstance(root_properties, dict)
            else None
        )
        if isinstance(candidates_schema, dict):
            # Generation always asks for two. The parser can still salvage a valid
            # sibling when the other candidate violates the provider contract.
            candidates_schema["minItems"] = 2
            candidates_schema["maxItems"] = 2

        reference_schema = schema.get("$defs", {}).get("LevelTestReferencePayload")
        if not isinstance(reference_schema, dict):
            return schema
        properties = reference_schema.get("properties")
        if not isinstance(properties, dict):
            return schema

        candidate_schema = schema.get("$defs", {}).get("LevelTestQuestionCandidate")
        candidate_properties = (
            candidate_schema.get("properties")
            if isinstance(candidate_schema, dict)
            else None
        )
        candidate_required = (
            candidate_schema.get("required")
            if isinstance(candidate_schema, dict)
            else None
        )

        choice_types = {
            LevelTestItemType.VOCAB_CONTEXT_CHOICE,
            LevelTestItemType.VOCAB_PARAPHRASE_CHOICE,
            LevelTestItemType.GRAMMAR_FORM_CHOICE,
            LevelTestItemType.GRAMMAR_SENTENCE_ORDER,
            LevelTestItemType.READING_GIST,
            LevelTestItemType.READING_DETAIL,
            LevelTestItemType.READING_DISCOURSE_FUNCTION,
            LevelTestItemType.READING_TEXT_INFERENCE,
            LevelTestItemType.LISTENING_GIST_CHOICE,
            LevelTestItemType.LISTENING_DETAIL_CHOICE,
        }
        writing_types = {
            LevelTestItemType.WRITING_TRANSLATION,
            LevelTestItemType.WRITING_GUIDED_SENTENCE,
            LevelTestItemType.WRITING_SCENARIO_RESPONSE,
            LevelTestItemType.WRITING_SHORT_PARAGRAPH,
        }
        speaking_types = {
            LevelTestItemType.SPEAKING_REPEAT,
            LevelTestItemType.SPEAKING_GUIDED_RESPONSE,
            LevelTestItemType.SPEAKING_SHORT_RESPONSE,
        }

        if isinstance(candidate_properties, dict):
            candidate_properties["domain"] = {
                "type": "string",
                "enum": [request.domain.value],
            }
            candidate_properties["itemType"] = {
                "type": "string",
                "enum": [request.item_type.value],
            }
            candidate_properties["complexityBand"] = {
                "type": "integer",
                "enum": [request.target_complexity_band],
            }
            candidate_properties["instructionLanguage"] = {
                "type": "string",
                "enum": [request.learning_language],
            }

            if request.item_type == LevelTestItemType.VOCAB_CONTEXT_CHOICE:
                if require_vocab_plan_id:
                    candidate_properties["generationPlanId"] = {
                        "type": "string",
                        "enum": ["A", "B"],
                    }
                    if (
                        isinstance(candidate_required, list)
                        and "generationPlanId" not in candidate_required
                    ):
                        candidate_required.append("generationPlanId")
                else:
                    candidate_properties["generationPlanId"] = {
                        "anyOf": [
                            {"type": "string", "enum": ["A", "B"]},
                            {"type": "null"},
                        ]
                    }
            else:
                candidate_properties["generationPlanId"] = {"type": "null"}

            if request.item_type in choice_types:
                candidate_properties["answerMode"] = {
                    "type": "string",
                    "enum": ["CHOICE"],
                }
                candidate_properties["answerLanguage"] = {"type": "null"}
                options_schema = candidate_properties.get("options")
                if isinstance(options_schema, dict):
                    if request.item_type == LevelTestItemType.GRAMMAR_SENTENCE_ORDER:
                        options_schema["minItems"] = 2
                    else:
                        options_schema["minItems"] = 4
                        options_schema["maxItems"] = 4
                if (
                    isinstance(candidate_required, list)
                    and "internalAnswerKey" not in candidate_required
                ):
                    candidate_required.append("internalAnswerKey")
            elif request.item_type in writing_types:
                candidate_properties["answerMode"] = {
                    "type": "string",
                    "enum": ["TEXT"],
                }
                candidate_properties["answerLanguage"] = {
                    "type": "string",
                    "enum": [request.learning_language],
                }
            elif request.item_type == LevelTestItemType.LISTENING_DICTATION:
                candidate_properties["answerMode"] = {
                    "type": "string",
                    "enum": ["TEXT"],
                }
                candidate_properties["answerLanguage"] = {
                    "type": "string",
                    "enum": [request.learning_language],
                }
            elif request.item_type == LevelTestItemType.LISTENING_INTERPRETATION:
                candidate_properties["answerMode"] = {
                    "type": "string",
                    "enum": ["TEXT"],
                }
                candidate_properties["answerLanguage"] = {
                    "type": "string",
                    "enum": [request.origin_language],
                }
            elif request.item_type in speaking_types:
                candidate_properties["answerMode"] = {
                    "type": "string",
                    "enum": ["AUDIO"],
                }
                candidate_properties["answerLanguage"] = {
                    "type": "string",
                    "enum": [request.learning_language],
                }

        answer_key_schema = schema.get("$defs", {}).get("LevelTestInternalAnswerKey")
        answer_key_properties = (
            answer_key_schema.get("properties")
            if isinstance(answer_key_schema, dict)
            else None
        )
        answer_key_required = (
            answer_key_schema.get("required")
            if isinstance(answer_key_schema, dict)
            else None
        )
        if request.item_type in choice_types and isinstance(answer_key_properties, dict):
            if request.item_type == LevelTestItemType.GRAMMAR_SENTENCE_ORDER:
                answer_key_properties["correctOptionKey"] = {"type": "null"}
                correct_order_schema = answer_key_properties.get("correctOrder")
                if isinstance(correct_order_schema, dict):
                    correct_order_schema["minItems"] = 2
            else:
                answer_key_properties["correctOptionKey"] = {
                    "type": "string",
                    "enum": ["A", "B", "C", "D"],
                }
                if (
                    isinstance(answer_key_required, list)
                    and "correctOptionKey" not in answer_key_required
                ):
                    answer_key_required.append("correctOptionKey")

        if request.item_type.name.startswith("READING_"):
            properties["readingPassage"] = {"type": "string"}
            properties["readingQuestion"] = {"type": "string"}
        if request.item_type.name.startswith("LISTENING_"):
            properties["sourceText"] = {"type": "string"}
            properties["listeningQuestion"] = {"type": "string"}
        if request.item_type == LevelTestItemType.LISTENING_INTERPRETATION:
            properties["referenceMeanings"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 3,
            }
            properties["keyMeaningUnits"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 5,
            }
            if isinstance(candidate_properties, dict):
                candidate_properties["answerMode"] = {"type": "string", "enum": ["TEXT"]}
                candidate_properties["answerLanguage"] = {
                    "type": "string",
                    "enum": [request.origin_language],
                }
        if request.item_type == LevelTestItemType.SPEAKING_REPEAT:
            properties["referenceText"] = {"type": "string"}
        if request.item_type == LevelTestItemType.WRITING_TRANSLATION:
            properties["translationSourceText"] = {"type": "string"}
        if request.item_type == LevelTestItemType.READING_DISCOURSE_FUNCTION:
            properties["emphasisText"] = {"type": "string"}
        if LevelTestService._requires_task_sufficiency_verification(request.item_type):
            min_items = 2 if request.item_type in {
                LevelTestItemType.WRITING_SCENARIO_RESPONSE,
                LevelTestItemType.WRITING_SHORT_PARAGRAPH,
                LevelTestItemType.SPEAKING_SHORT_RESPONSE,
            } else 1
            properties["providedFacts"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": min_items,
            }
            properties["requiredIntents"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": min_items,
            }
            properties["responseConstraints"] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }

        preferred = [category.value for category in request.preferred_scenario_categories]
        diversity_schema = schema.get("$defs", {}).get("DiversityMetadata")
        diversity_properties = (
            diversity_schema.get("properties")
            if isinstance(diversity_schema, dict)
            else None
        )
        if isinstance(diversity_properties, dict):
            if preferred:
                diversity_properties["scenarioCategory"] = {
                    "type": "string",
                    "enum": preferred,
                }
            intent_definition = schema.get("$defs", {}).get("CommunicativeIntent")
            intent_values = (
                intent_definition.get("enum")
                if isinstance(intent_definition, dict)
                else None
            )
            if isinstance(intent_values, list) and intent_values:
                diversity_properties["communicativeIntent"] = {
                    "type": "string",
                    "enum": intent_values,
                }
        return schema

    @staticmethod
    def _parse_generation_candidates(
        provider_data: dict,
    ) -> tuple[list[tuple[int, LevelTestQuestionCandidate]], list[str], int]:
        """Validate candidates independently so one malformed sibling cannot waste the call."""

        raw_candidates = provider_data.get("candidates")
        if not isinstance(raw_candidates, list):
            return [], ["SCHEMA:candidates:list_type"], 1
        if not raw_candidates:
            return [], ["SCHEMA:candidates:too_short"], 1

        parsed: list[tuple[int, LevelTestQuestionCandidate]] = []
        reasons: list[str] = []
        rejected = 0

        for index, raw_candidate in enumerate(raw_candidates[:6]):
            if not isinstance(raw_candidate, dict):
                rejected += 1
                reasons.append(f"SCHEMA:candidates.{index}:dict_type")
                continue
            try:
                candidate = LevelTestQuestionCandidate.model_validate(raw_candidate)
            except ValidationError as exc:
                rejected += 1
                for error in exc.errors(
                    include_url=False,
                    include_input=False,
                )[:6]:
                    location = ".".join(
                        str(part) for part in error.get("loc", ())
                    ) or "candidate"
                    error_type = LevelTestService._candidate_schema_error_code(error)
                    reasons.append(
                        f"SCHEMA:candidates.{index}.{location}:{error_type}"
                    )
                continue
            parsed.append((index, candidate))

        if len(raw_candidates) > 6:
            rejected += len(raw_candidates) - 6
            reasons.append("SCHEMA:candidates:too_long")

        return parsed, reasons[:12], rejected

    @staticmethod
    def _candidate_schema_error_code(error: dict) -> str:
        """Return a stable, non-sensitive reason code for candidate validation errors."""

        error_type = str(error.get("type") or "validation_error")
        if error_type != "value_error":
            return error_type

        message = str(error.get("msg") or "")
        known_value_errors = (
            (
                "Sentence Order에는 최소 2개 Token Option",
                "sentence_order_too_few_options",
            ),
            (
                "Sentence Order Option key는 중복될 수 없습니다",
                "sentence_order_duplicate_option_key",
            ),
            (
                "correctOrder는 Option key를 정확히 한 번씩 포함해야 합니다",
                "sentence_order_correct_order_mismatch",
            ),
            ("Choice 문제는 정확히 4개의 Option", "choice_option_count_invalid"),
            (
                "Choice 정답 Key가 Option에 존재하지 않습니다",
                "choice_correct_option_missing",
            ),
            (
                "TEXT/AUDIO 문제에는 Choice Option을 둘 수 없습니다",
                "non_choice_has_options",
            ),
            (
                "TEXT 문제에는 answerLanguage가 필요합니다",
                "text_answer_language_missing",
            ),
            (
                "AUDIO 문제에는 answerLanguage가 필요합니다",
                "audio_answer_language_missing",
            ),
        )
        for fragment, code in known_value_errors:
            if fragment in message:
                return code
        return error_type

    @staticmethod
    def _schema_rejection_reasons(exc: ValidationError) -> list[str]:
        reasons: list[str] = []
        for error in exc.errors(include_url=False, include_input=False)[:12]:
            location = ".".join(str(part) for part in error.get("loc", ())) or "payload"
            error_type = str(error.get("type") or "validation_error")
            reasons.append(f"SCHEMA:{location}:{error_type}")
        return reasons or ["SCHEMA:UNKNOWN"]

    @staticmethod
    def _summarize_rejection_reasons(reasons: Counter[str]) -> list[str]:
        return [
            f"{reason} (x{count})" if count > 1 else reason
            for reason, count in reasons.most_common(8)
        ]

    async def evaluate_text(
        self,
        request: LevelTestTextEvaluationRequest,
    ) -> LevelTestEvaluationResponse:
        key = "|".join(
            [request.idempotency_key, str(request.session_id), str(request.item_id), LEVEL_TEST_EVALUATION_VERSION]
        )
        response, _ = await self.text_evaluation_store.execute(
            key,
            lambda: self._evaluate_text_once(request),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _evaluate_text_once(
        self,
        request: LevelTestTextEvaluationRequest,
    ) -> LevelTestEvaluationResponse:
        if request.domain == LevelTestDomain.WRITING:
            return await self._evaluate_writing(request)
        if request.item_type == LevelTestItemType.LISTENING_DICTATION:
            return await self._evaluate_dictation(request)
        if request.item_type == LevelTestItemType.LISTENING_INTERPRETATION:
            return await self._evaluate_interpretation(request)
        raise HTTPException(
            status_code=422,
            detail="Objective Level Test Item은 AI Text Evaluation 대상이 아닙니다.",
        )

    async def _evaluate_writing(
        self,
        request: LevelTestTextEvaluationRequest,
    ) -> LevelTestEvaluationResponse:
        focus = request.focus_metrics or list(WritingMetric)
        result = await self.writing_service.evaluate(
            WritingEvaluationRequest(
                request_id=request.request_id,
                context="LEVEL_TEST",
                origin_language=request.origin_language,
                learning_language=request.learning_language,
                origin_sentence=request.prompt_text,
                user_answer=request.answer,
                difficulty=f"BAND_{request.complexity_band}",
                keywords=[],
                focus_metrics=focus,
                task_type=request.item_type.value,
                translation_source_text=request.translation_source_text,
                provided_facts=request.provided_facts,
                required_intents=request.required_intents,
                response_constraints=request.response_constraints,
            )
        )
        scores = result.scores
        metrics = [
            LevelTestMetricResult(type="MEANING", score=scores.meaning, confidence=1.0),
            LevelTestMetricResult(type="GRAMMAR", score=scores.grammar, confidence=1.0),
            LevelTestMetricResult(type="VOCABULARY", score=scores.vocabulary, confidence=1.0),
            LevelTestMetricResult(type="NATURALNESS", score=scores.naturalness, confidence=1.0),
            LevelTestMetricResult(type="EXPRESSION", score=scores.expression, confidence=1.0),
        ]
        strengths = self._writing_level_test_feedback_texts(
            request.prompt_text,
            [item.origin_text for item in result.strengths],
        )
        improvements = self._writing_level_test_feedback_texts(
            request.prompt_text,
            [item.origin_text for item in result.weaknesses],
        )
        for correction in result.corrections:
            explanation = correction.explanation.origin_text.strip()
            detail = f"「{correction.original.strip()}」 → 「{correction.corrected.strip()}」"
            if explanation:
                detail += f": {explanation}"
            if detail not in improvements:
                improvements.append(detail)
        if scores.overall < 90:
            overall_explanation = result.explanation.origin_text.strip()
            if (
                overall_explanation
                and not self._looks_like_source_repetition(request.prompt_text, overall_explanation)
                and overall_explanation not in improvements
            ):
                improvements.append(overall_explanation)

        return LevelTestEvaluationResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            item_id=request.item_id,
            domain=request.domain,
            item_type=request.item_type,
            evaluable=True,
            score=scores.overall,
            confidence=1.0,
            metrics=metrics,
            strengths=strengths[:20],
            improvements=improvements[:20],
            recommended_answers=result.recommended_answers,
            detailed_feedback=self._writing_feedback_details(result),
            assessment_signals=self._signals(request.domain, metrics),
            evaluation_version=LEVEL_TEST_EVALUATION_VERSION,
            prompt_version=result.prompt_version,
        )

    @staticmethod
    def _looks_like_source_repetition(source_text: str, feedback: str) -> bool:
        def compact(value: str) -> str:
            normalized = unicodedata.normalize("NFKC", value).casefold()
            return "".join(char for char in normalized if char.isalnum())

        source = compact(source_text)
        value = compact(feedback)
        return bool(value) and len(value) >= 8 and value in source

    @classmethod
    def _writing_level_test_feedback_texts(
        cls,
        source_text: str,
        values: list[str],
    ) -> list[str]:
        result: list[str] = []
        for value in values:
            text = value.strip()
            if not text or cls._looks_like_source_repetition(source_text, text):
                continue
            if text not in result:
                result.append(text)
        return result

    async def _evaluate_dictation(
        self,
        request: LevelTestTextEvaluationRequest,
    ) -> LevelTestEvaluationResponse:
        if not request.source_text:
            raise HTTPException(status_code=422, detail="LISTENING_DICTATION에는 sourceText가 필요합니다.")
        result = await self.dictation_service.evaluate(
            DictationEvaluationRequest(
                request_id=request.request_id,
                idempotency_key=request.idempotency_key,
                item_id=request.item_id,
                attempt_id=request.session_id,
                evaluation_purpose=EvaluationPurpose.PRACTICE,
                answer_revealed=False,
                assistance_usage=[],
                source_text=request.source_text,
                answer=request.answer,
                learning_language=request.learning_language,
                policy_version="level-test-v2-multiskill",
                model_config_version="level-test-model-config-v1",
                manual_retry_attempt=request.manual_retry_attempt,
            )
        )
        task = next(item for item in result.tasks if item.task_type == ListeningTaskType.DICTATION)
        metrics = [
            LevelTestMetricResult(
                type=item.type,
                state=item.state.value,
                score=item.score,
                confidence=item.confidence,
                evidence=[entry.model_dump(mode="json", by_alias=True) for entry in item.evidence],
                not_evaluable_reason=item.not_evaluable_reason,
            )
            for item in task.metrics
        ]
        return self._listening_eval_response(request, task, metrics, result.evaluation_version)

    async def _evaluate_interpretation(
        self,
        request: LevelTestTextEvaluationRequest,
    ) -> LevelTestEvaluationResponse:
        if not request.source_text or len(request.reference_meanings) < 2 or not request.key_meaning_units:
            raise HTTPException(
                status_code=422,
                detail="LISTENING_INTERPRETATION에는 sourceText/referenceMeanings/keyMeaningUnits가 필요합니다.",
            )
        result = await self.interpretation_service.evaluate(
            InterpretationEvaluationRequest(
                request_id=request.request_id,
                idempotency_key=request.idempotency_key,
                item_id=request.item_id,
                attempt_id=request.session_id,
                evaluation_purpose=EvaluationPurpose.PRACTICE,
                answer_revealed=False,
                assistance_usage=[],
                source_text=request.source_text,
                reference_meanings=request.reference_meanings,
                key_meaning_units=request.key_meaning_units,
                answer=request.answer,
                origin_language=request.origin_language,
                learning_language=request.learning_language,
                policy_version="level-test-v2-multiskill",
                model_config_version="level-test-model-config-v1",
                manual_retry_attempt=request.manual_retry_attempt,
            )
        )
        task = next(item for item in result.tasks if item.task_type == ListeningTaskType.INTERPRETATION)
        metrics = [
            LevelTestMetricResult(
                type=item.type,
                state=item.state.value,
                score=item.score,
                confidence=item.confidence,
                evidence=[entry.model_dump(mode="json", by_alias=True) for entry in item.evidence],
                not_evaluable_reason=item.not_evaluable_reason,
            )
            for item in task.metrics
        ]
        return self._listening_eval_response(request, task, metrics, result.evaluation_version)

    async def evaluate_speaking(
        self,
        request: LevelTestSpeakingEvaluationContext,
        *,
        audio_bytes: bytes,
        file_name: str | None,
        content_type: str | None,
    ) -> LevelTestEvaluationResponse:
        key = "|".join(
            [request.idempotency_key, str(request.session_id), str(request.item_id), LEVEL_TEST_SPEAKING_EVALUATION_VERSION]
        )
        response, _ = await self.speaking_evaluation_store.execute(
            key,
            lambda: self._evaluate_speaking_once(
                request,
                audio_bytes=audio_bytes,
                file_name=file_name,
                content_type=content_type,
            ),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _evaluate_speaking_once(
        self,
        request: LevelTestSpeakingEvaluationContext,
        *,
        audio_bytes: bytes,
        file_name: str | None,
        content_type: str | None,
    ) -> LevelTestEvaluationResponse:
        try:
            normalized = self.audio_processor.validate_and_normalize(
                audio_bytes,
                file_name=file_name,
                content_type=content_type,
                min_seconds=settings.AI_SPEAKING_MIN_VALID_AUDIO_SECONDS,
                max_seconds=float(request.max_duration_seconds),
                max_bytes=settings.AI_SPEAKING_MAX_AUDIO_FILE_BYTES,
            )
            stt = await self.stt_service.transcribe(
                request_id=request.request_id,
                session_id=str(request.session_id),
                turn_index=1,
                learning_language=request.learning_language,
                normalized_audio=normalized,
                phrase_hints=request.phrase_hints,
                idempotency_key=f"{request.idempotency_key}:stt",
                manual_retry_attempt=request.manual_retry_attempt,
            )
        except SpeakingStageException as exc:
            return LevelTestEvaluationResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                item_id=request.item_id,
                domain=LevelTestDomain.SPEAKING,
                item_type=request.item_type,
                evaluable=False,
                score=None,
                confidence=None,
                metrics=[],
                strengths=[],
                improvements=[],
                assessment_signals=[],
                reason_code=exc.code.value,
                evaluation_version=LEVEL_TEST_SPEAKING_EVALUATION_VERSION,
                prompt_version=LEVEL_TEST_SPEAKING_PROMPT_VERSION,
            )

        if stt.transcript.confidence < 0.30:
            return LevelTestEvaluationResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                item_id=request.item_id,
                domain=LevelTestDomain.SPEAKING,
                item_type=request.item_type,
                evaluable=False,
                score=None,
                confidence=stt.transcript.confidence,
                transcript=stt.transcript.text,
                reason_code="LOW_STT_CONFIDENCE",
                evaluation_version=LEVEL_TEST_SPEAKING_EVALUATION_VERSION,
                prompt_version=LEVEL_TEST_SPEAKING_PROMPT_VERSION,
                usage=LevelTestUsage(
                    latency_ms=stt.usage.stt.latency_ms if stt.usage.stt else 0,
                    provider=stt.usage.stt.provider if stt.usage.stt else None,
                    model=stt.usage.stt.model if stt.usage.stt else None,
                ),
            )

        prompt = build_level_test_speaking_evaluation_prompt(
            request,
            transcript=stt.transcript.model_dump(mode="json", by_alias=True),
            acoustic_quality=normalized.quality.model_dump(mode="json", by_alias=True),
        )
        started = time.perf_counter()
        payload: LevelTestSpeakingEvaluationPayload | None = None
        provider_result = None
        total_input_tokens = 0
        total_output_tokens = 0
        last_error: Exception | None = None
        last_timeout = False

        for evaluation_attempt in range(1, 4):
            try:
                current_result = await asyncio.wait_for(
                    self.provider.call_with_metadata(
                        type_name=self.SPEAKING_EVALUATION_TYPE,
                        data=prompt,
                        schema=LevelTestSpeakingEvaluationPayload.model_json_schema(),
                    ),
                    timeout=self.evaluation_timeout_seconds,
                )
                total_input_tokens += current_result.input_tokens
                total_output_tokens += current_result.output_tokens
                provider_result = current_result
                if not isinstance(current_result.data, dict):
                    raise ValueError("Speaking evaluation response must be an object")
                normalized_payload = self._normalize_speaking_evaluation_payload(
                    request,
                    current_result.data,
                )
                payload = LevelTestSpeakingEvaluationPayload.model_validate(
                    normalized_payload
                )
                break
            except (TimeoutError, asyncio.TimeoutError) as exc:
                last_error = exc
                last_timeout = True
                logger.warning(
                    "Level Test Speaking evaluation timed out. request_id=%s item_id=%s attempt=%d/3",
                    request.request_id,
                    request.item_id,
                    evaluation_attempt,
                )
            except ValidationError as exc:
                last_error = exc
                last_timeout = False
                logger.warning(
                    "Level Test Speaking evaluation schema rejected. request_id=%s item_id=%s attempt=%d/3 reasons=%s",
                    request.request_id,
                    request.item_id,
                    evaluation_attempt,
                    self._schema_rejection_reasons(exc),
                )
            except ValueError as exc:
                last_error = exc
                last_timeout = False
                logger.warning(
                    "Level Test Speaking evaluation payload rejected. request_id=%s item_id=%s attempt=%d/3 reason=%s",
                    request.request_id,
                    request.item_id,
                    evaluation_attempt,
                    type(exc).__name__,
                )
            except Exception as exc:
                last_error = exc
                last_timeout = isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                if evaluation_attempt >= 3 or not self._is_transient_provider_error(exc):
                    raise HTTPException(
                        status_code=502,
                        detail="Level Test Speaking 평가 Provider 호출에 실패했습니다.",
                    ) from exc
                logger.warning(
                    "Level Test Speaking transient provider failure. request_id=%s item_id=%s attempt=%d/3 errorType=%s",
                    request.request_id,
                    request.item_id,
                    evaluation_attempt,
                    type(exc).__name__,
                )

        if payload is None or provider_result is None:
            if last_timeout:
                raise HTTPException(
                    status_code=504,
                    detail="Level Test Speaking 평가 시간이 초과되었습니다.",
                ) from last_error
            raise HTTPException(
                status_code=502,
                detail="Level Test Speaking 평가 응답 Schema가 유효하지 않습니다.",
            ) from last_error

        metrics = [
            LevelTestMetricResult(
                type=item.type,
                state=item.state,
                score=item.score,
                confidence=item.confidence,
                summary=item.summary,
                evidence=[{"message": evidence} for evidence in item.evidence],
                not_evaluable_reason=item.not_evaluable_reason,
            )
            for item in payload.metrics
        ]
        score = self._calculate_speaking_item_score(request.item_type, metrics)
        evaluable = score is not None
        latency_ms = int((time.perf_counter() - started) * 1000)
        return LevelTestEvaluationResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            item_id=request.item_id,
            domain=LevelTestDomain.SPEAKING,
            item_type=request.item_type,
            evaluable=evaluable,
            score=score,
            confidence=payload.evaluation_confidence,
            transcript=stt.transcript.text,
            metrics=metrics,
            strengths=payload.strengths,
            improvements=payload.improvements,
            recommended_answers=(
                [request.reference_text]
                if request.item_type == LevelTestItemType.SPEAKING_REPEAT and request.reference_text
                else payload.recommended_answers
            ),
            detailed_feedback=self._speaking_feedback_details(metrics, payload.improvements),
            assessment_signals=self._signals(LevelTestDomain.SPEAKING, metrics) if evaluable else [],
            reason_code=None if evaluable else "INSUFFICIENT_EVIDENCE",
            evaluation_version=LEVEL_TEST_SPEAKING_EVALUATION_VERSION,
            prompt_version=LEVEL_TEST_SPEAKING_PROMPT_VERSION,
            usage=LevelTestUsage(
                latency_ms=latency_ms + (stt.usage.stt.latency_ms if stt.usage.stt else 0),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                provider=provider_result.provider,
                model=provider_result.model,
                prompt_version=LEVEL_TEST_SPEAKING_PROMPT_VERSION,
                evaluation_version=LEVEL_TEST_SPEAKING_EVALUATION_VERSION,
            ),
        )

    async def _question_response_with_reference_audio(
        self,
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
        *,
        stats: DiversityValidationStats,
        fallback_used: bool,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        provider: str | None,
        model: str | None,
        option_scores: dict[str, int],
        selection_policy: LevelTestChoiceSelectionPolicy | None,
    ) -> LevelTestQuestionGenerationResponse:
        reference_audio = await self._prepare_reference_audio(request, candidate)
        return self._question_response(
            request,
            candidate,
            stats=stats,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider=provider,
            model=model,
            reference_audio=reference_audio,
            option_scores=option_scores,
            selection_policy=selection_policy,
        )

    async def _prepare_reference_audio(
        self,
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
    ) -> LevelTestReferenceAudio | None:
        upload = request.reference_audio_upload
        if upload is None:
            return None
        if not self._requires_reference_audio(candidate.item_type):
            raise HTTPException(
                status_code=422,
                detail="Reference Audio 업로드 정보가 필요하지 않은 ItemType입니다.",
            )
        if self.speech_provider is None:
            raise HTTPException(
                status_code=503,
                detail="Level Test Reference Audio TTS Provider가 준비되지 않았습니다.",
            )

        text = self._reference_audio_text(candidate)
        try:
            result = await asyncio.wait_for(
                self.speech_provider.synthesize_speech(
                    text=text,
                    voice=upload.voice,
                    language=request.learning_language,
                    speed=upload.playback_speed,
                ),
                timeout=settings.AI_LISTENING_TTS_TIMEOUT_SECONDS,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise HTTPException(
                status_code=504,
                detail="Level Test Reference Audio 생성 시간이 초과되었습니다.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Level Test Reference Audio 생성에 실패했습니다.",
            ) from exc

        if result.content_type != upload.content_type:
            raise HTTPException(
                status_code=502,
                detail="Level Test Reference Audio Content-Type이 업로드 계약과 일치하지 않습니다.",
            )
        try:
            await self.audio_uploader.put(
                upload.upload_url,
                result.audio_bytes,
                result.content_type,
            )
        except PresignedAudioUploadError as exc:
            raise HTTPException(
                status_code=502,
                detail="Level Test Reference Audio 업로드에 실패했습니다.",
            ) from exc

        duration_ms = (
            None
            if result.duration_seconds is None
            else int(result.duration_seconds * 1000 + 0.5)
        )
        logger.info(
            "Level Test reference audio uploaded. request_id=%s item_type=%s object_key=%s bytes=%d duration_ms=%s",
            request.request_id,
            candidate.item_type.value,
            upload.object_key,
            len(result.audio_bytes),
            duration_ms,
        )
        return LevelTestReferenceAudio(
            object_key=upload.object_key,
            content_type=result.content_type,
            duration_ms=duration_ms,
            checksum_sha256=hashlib.sha256(result.audio_bytes).hexdigest(),
        )

    @staticmethod
    def _requires_reference_audio(item_type: LevelTestItemType) -> bool:
        return item_type.name.startswith("LISTENING_") or item_type == LevelTestItemType.SPEAKING_REPEAT

    @staticmethod
    def _reference_audio_text(candidate: LevelTestQuestionCandidate) -> str:
        key = (
            "referenceText"
            if candidate.item_type == LevelTestItemType.SPEAKING_REPEAT
            else "sourceText"
        )
        value = candidate.reference_payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(
                status_code=502,
                detail="Level Test Reference Audio 원문이 없습니다.",
            )
        return value.strip()

    @staticmethod
    def _question_response(
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
        *,
        stats: DiversityValidationStats,
        fallback_used: bool,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        provider: str | None,
        model: str | None,
        reference_audio: LevelTestReferenceAudio | None = None,
        option_scores: dict[str, int] | None = None,
        selection_policy: LevelTestChoiceSelectionPolicy | None = None,
    ) -> LevelTestQuestionGenerationResponse:
        from app.schemas.language_learning_quality import DiversitySummary

        return LevelTestQuestionGenerationResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            question_number=request.question_number,
            total_questions=request.total_questions,
            domain=request.domain,
            item_type=request.item_type,
            complexity_band=candidate.complexity_band,
            instruction=candidate.instruction,
            instruction_language=candidate.instruction_language,
            answer_mode=candidate.answer_mode,
            answer_language=candidate.answer_language,
            prompt_text=candidate.prompt_text,
            options=candidate.options,
            internal_answer_key=LevelTestScoredInternalAnswerKey(
                correct_option_key=candidate.internal_answer_key.correct_option_key,
                correct_order=list(candidate.internal_answer_key.correct_order),
                selection_policy=selection_policy,
                option_scores=dict(option_scores or {}),
            ),
            reference_payload=candidate.reference_payload,
            diversity_metadata=candidate.diversity_metadata,
            max_answer_length=candidate.max_answer_length,
            max_audio_seconds=candidate.max_audio_seconds,
            generation_version=LEVEL_TEST_GENERATION_VERSION,
            prompt_version=LEVEL_TEST_PROMPT_VERSION,
            diversity_summary=DiversitySummary(
                policy_version=CONTENT_DIVERSITY_POLICY_VERSION,
                candidate_count=stats.candidate_count,
                accepted_count=1,
                rejected_exact=stats.rejected_exact,
                rejected_similarity=stats.rejected_similarity,
                rejected_structural=stats.rejected_structural,
                rejected_background_knowledge=stats.rejected_background_knowledge,
                fallback_used=fallback_used,
            ),
            usage=LevelTestUsage(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider=provider,
                model=model,
                prompt_version=LEVEL_TEST_PROMPT_VERSION,
            ),
            reference_audio=reference_audio,
        )

    @staticmethod
    def _validate_question_candidate(
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
    ) -> None:
        if candidate.domain != request.domain or candidate.item_type != request.item_type:
            raise ValueError("Provider가 요청과 다른 Domain/ItemType을 반환했습니다.")
        if candidate.complexity_band != request.target_complexity_band:
            raise ValueError("Provider가 요청과 다른 complexityBand를 반환했습니다.")
        if candidate.instruction_language != request.learning_language:
            raise ValueError("instructionLanguage는 learningLanguage여야 합니다.")

        preferred_categories = set(request.preferred_scenario_categories)
        if (
            preferred_categories
            and candidate.diversity_metadata.scenario_category not in preferred_categories
        ):
            allowed = ",".join(sorted(category.value for category in preferred_categories))
            raise ValueError(
                "scenarioCategory가 서버의 세션 균형 계획을 따르지 않았습니다. "
                f"allowed={allowed}"
            )

        choice_types = {
            LevelTestItemType.VOCAB_CONTEXT_CHOICE,
            LevelTestItemType.VOCAB_PARAPHRASE_CHOICE,
            LevelTestItemType.GRAMMAR_FORM_CHOICE,
            LevelTestItemType.GRAMMAR_SENTENCE_ORDER,
            LevelTestItemType.READING_GIST,
            LevelTestItemType.READING_DETAIL,
            LevelTestItemType.READING_DISCOURSE_FUNCTION,
            LevelTestItemType.READING_TEXT_INFERENCE,
            LevelTestItemType.LISTENING_GIST_CHOICE,
            LevelTestItemType.LISTENING_DETAIL_CHOICE,
        }
        writing_types = {
            LevelTestItemType.WRITING_TRANSLATION,
            LevelTestItemType.WRITING_GUIDED_SENTENCE,
            LevelTestItemType.WRITING_SCENARIO_RESPONSE,
            LevelTestItemType.WRITING_SHORT_PARAGRAPH,
        }
        speaking_types = {
            LevelTestItemType.SPEAKING_REPEAT,
            LevelTestItemType.SPEAKING_GUIDED_RESPONSE,
            LevelTestItemType.SPEAKING_SHORT_RESPONSE,
        }

        if candidate.item_type in choice_types and candidate.answer_mode != LevelTestAnswerMode.CHOICE:
            raise ValueError("Choice Item의 answerMode가 CHOICE가 아닙니다.")
        if candidate.item_type in choice_types and candidate.answer_language is not None:
            raise ValueError("Choice Item에는 answerLanguage를 설정하지 않습니다.")
        if (
            candidate.item_type in choice_types
            and candidate.item_type != LevelTestItemType.GRAMMAR_SENTENCE_ORDER
            and {option.key for option in candidate.options} != {"A", "B", "C", "D"}
        ):
            raise ValueError("4지선다 Choice Option key는 A/B/C/D여야 합니다.")
        if candidate.item_type in writing_types and (
            candidate.answer_mode != LevelTestAnswerMode.TEXT
            or candidate.answer_language != request.learning_language
        ):
            raise ValueError("Writing Item의 답변 언어/방식이 유효하지 않습니다.")
        if candidate.item_type == LevelTestItemType.LISTENING_DICTATION and (
            candidate.answer_mode != LevelTestAnswerMode.TEXT
            or candidate.answer_language != request.learning_language
        ):
            raise ValueError("Dictation은 learningLanguage TEXT 답변이어야 합니다.")
        if candidate.item_type == LevelTestItemType.LISTENING_INTERPRETATION and (
            candidate.answer_mode != LevelTestAnswerMode.TEXT
            or candidate.answer_language != request.origin_language
        ):
            raise ValueError("Interpretation은 originLanguage TEXT 답변이어야 합니다.")
        if candidate.item_type in speaking_types and (
            candidate.answer_mode != LevelTestAnswerMode.AUDIO
            or candidate.answer_language != request.learning_language
        ):
            raise ValueError("Speaking Item은 learningLanguage AUDIO 답변이어야 합니다.")

        LevelTestService._validate_inline_markup(candidate)

        if candidate.item_type == LevelTestItemType.VOCAB_CONTEXT_CHOICE:
            LevelTestService._validate_vocab_context_choice(candidate)
        if candidate.item_type == LevelTestItemType.VOCAB_PARAPHRASE_CHOICE:
            LevelTestService._validate_vocab_answer_not_exposed(candidate)
        if candidate.item_type == LevelTestItemType.GRAMMAR_FORM_CHOICE:
            LevelTestService._validate_grammar_form_choice(candidate)
        if candidate.item_type == LevelTestItemType.GRAMMAR_SENTENCE_ORDER:
            option_order = [option.key for option in candidate.options]
            if option_order == candidate.internal_answer_key.correct_order:
                raise ValueError("Sentence Order Option은 정답 순서 그대로 노출될 수 없습니다.")

        if candidate.item_type.name.startswith("READING_"):
            LevelTestService._validate_reading_candidate(candidate)

        if candidate.item_type == LevelTestItemType.WRITING_TRANSLATION:
            source = candidate.reference_payload.get("translationSourceText")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("WRITING_TRANSLATION에는 translationSourceText가 필요합니다.")
            if source.strip() != candidate.prompt_text.strip():
                raise ValueError("WRITING_TRANSLATION promptText는 translationSourceText와 정확히 일치해야 합니다.")

        if LevelTestService._requires_task_sufficiency_verification(candidate.item_type):
            LevelTestService._validate_guided_task_payload(candidate)

        if candidate.item_type.name.startswith("LISTENING_"):
            source_text = candidate.reference_payload.get("sourceText")
            listening_question = candidate.reference_payload.get("listeningQuestion")
            if not isinstance(source_text, str) or not source_text.strip():
                raise ValueError("Listening Item에는 referencePayload.sourceText가 필요합니다.")
            if not isinstance(listening_question, str) or not listening_question.strip():
                raise ValueError("Listening Item에는 referencePayload.listeningQuestion이 필요합니다.")
            if candidate.prompt_text.strip() != listening_question.strip():
                raise ValueError("Listening Item promptText는 listeningQuestion과 정확히 일치해야 합니다.")
            if LevelTestService._has_listening_script_leak(
                candidate.prompt_text,
                source_text,
            ):
                raise ValueError("Listening Item promptText에 음성 sourceText가 노출되어 있습니다.")

        LevelTestService._validate_learning_language_lane(request, candidate)

        if candidate.item_type == LevelTestItemType.LISTENING_INTERPRETATION:
            references = candidate.reference_payload.get("referenceMeanings")
            units = candidate.reference_payload.get("keyMeaningUnits")
            if not isinstance(references, list) or len(references) not in {2, 3}:
                raise ValueError("Interpretation referenceMeanings는 2~3개여야 합니다.")
            if not isinstance(units, list) or not 2 <= len(units) <= 5:
                raise ValueError("Interpretation keyMeaningUnits는 2~5개여야 합니다.")
        if candidate.item_type == LevelTestItemType.SPEAKING_REPEAT:
            reference = candidate.reference_payload.get("referenceText")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("SPEAKING_REPEAT에는 referenceText가 필요합니다.")
            LevelTestService._validate_speaking_repeat_load(request, candidate, reference)

    @staticmethod
    def _validate_speaking_repeat_load(
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
        reference_text: str,
    ) -> None:
        compact = " ".join(reference_text.split())
        # Q18 and Q19 intentionally share the same SPEAKING_REPEAT pool bucket.
        # Keep every pooled repeat candidate safe for the stricter audio-only slot.
        if request.question_number in {18, 19}:
            if len(compact) > 90:
                raise ValueError("SPEAKING_REPEAT referenceText가 지나치게 깁니다.")
            sentence_parts = [
                part for part in re.split(r"[.!?。！？]+", compact) if part.strip()
            ]
            if len(sentence_parts) > 1:
                raise ValueError("SPEAKING_REPEAT는 한 문장으로 구성해야 합니다.")
            if candidate.max_audio_seconds is not None and candidate.max_audio_seconds > 20:
                raise ValueError("SPEAKING_REPEAT maxAudioSeconds는 20초 이하여야 합니다.")

    @staticmethod
    def _validate_learning_language_lane(
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
    ) -> None:
        """Enforce the learner-facing Level Test language contract.

        Normal assessment UI is learning-language-first.  The only intentional
        cross-language task lanes are the WRITING_TRANSLATION source text and the
        LISTENING_INTERPRETATION answer/reference-meaning lane.
        """

        learning_texts: list[str] = [candidate.instruction]
        payload = candidate.reference_payload

        if candidate.item_type == LevelTestItemType.WRITING_TRANSLATION:
            source = payload.get("translationSourceText")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("WRITING_TRANSLATION 번역 원문이 없습니다.")
            LevelTestService._assert_language_lane(
                request.origin_language,
                [source],
                reason="WRITING_TRANSLATION sourceText 언어가 originLanguage와 일치하지 않습니다.",
            )
        else:
            learning_texts.append(candidate.prompt_text)

        if candidate.item_type.name.startswith("READING_"):
            for key in ("readingPassage", "readingQuestion"):
                value = payload.get(key)
                if isinstance(value, str):
                    learning_texts.append(value)
            learning_texts.extend(option.text for option in candidate.options)
        elif candidate.item_type.name.startswith("LISTENING_"):
            for key in ("sourceText", "listeningQuestion"):
                value = payload.get(key)
                if isinstance(value, str):
                    learning_texts.append(value)
            learning_texts.extend(option.text for option in candidate.options)
        elif candidate.item_type in {
            LevelTestItemType.VOCAB_CONTEXT_CHOICE,
            LevelTestItemType.VOCAB_PARAPHRASE_CHOICE,
            LevelTestItemType.GRAMMAR_FORM_CHOICE,
            LevelTestItemType.GRAMMAR_SENTENCE_ORDER,
        }:
            learning_texts.extend(option.text for option in candidate.options)

        if LevelTestService._requires_task_sufficiency_verification(candidate.item_type):
            for key in ("providedFacts", "requiredIntents", "responseConstraints"):
                values = payload.get(key)
                if isinstance(values, list):
                    learning_texts.extend(
                        value for value in values if isinstance(value, str)
                    )

        LevelTestService._assert_language_lane(
            request.learning_language,
            learning_texts,
            reason=(
                f"{candidate.item_type.value} 학습자 표시 텍스트가 "
                f"learningLanguage={request.learning_language}와 일치하지 않습니다."
            ),
        )

        if candidate.item_type == LevelTestItemType.LISTENING_INTERPRETATION:
            origin_texts: list[str] = []
            for key in ("referenceMeanings", "keyMeaningUnits"):
                values = payload.get(key)
                if isinstance(values, list):
                    origin_texts.extend(
                        value for value in values if isinstance(value, str)
                    )
            if origin_texts:
                LevelTestService._assert_language_lane(
                    request.origin_language,
                    origin_texts,
                    reason="LISTENING_INTERPRETATION 평가 기준 언어가 originLanguage와 일치하지 않습니다.",
                )

    @staticmethod
    def _assert_language_lane(
        language_code: str,
        texts: list[str],
        *,
        reason: str,
    ) -> None:
        combined = "\n".join(text for text in texts if text and text.strip())
        if not combined:
            raise ValueError(reason)

        language = language_code.strip().lower().split("-", 1)[0].split("_", 1)[0]
        has_hangul = bool(re.search(r"[\uac00-\ud7a3]", combined))
        has_kana = bool(re.search(r"[\u3040-\u30ff]", combined))
        has_ascii = bool(re.search(r"[A-Za-z]", combined))
        mismatch = False
        missing_expected_script = False
        if language == "ja":
            mismatch = has_hangul
            missing_expected_script = not has_kana
        elif language == "ko":
            mismatch = has_kana
            missing_expected_script = not has_hangul
        elif language == "en":
            mismatch = has_hangul or has_kana
            missing_expected_script = not has_ascii

        if mismatch or missing_expected_script:
            raise ValueError(reason)


    @staticmethod
    def _has_listening_script_leak(prompt_text: str, source_text: str) -> bool:
        """Reject learner-visible text that reproduces most or all of the audio script.

        Listening questions may legitimately quote a short name or phrase. The guard
        therefore uses compact NFKC text plus a conservative longest-common-substring
        threshold instead of rejecting any overlap at all.
        """

        def compact(value: str) -> str:
            normalized = unicodedata.normalize("NFKC", value).casefold()
            return "".join(char for char in normalized if char.isalnum())

        prompt = compact(prompt_text)
        source = compact(source_text)
        if not prompt or not source:
            return False
        if prompt == source:
            return True
        if len(source) >= 12 and source in prompt:
            return True

        if len(source) < 20:
            threshold = max(8, len(source) - 2)
        else:
            threshold = max(20, int(len(source) * 0.60 + 0.999))
        if len(prompt) < threshold:
            return False

        matcher = SequenceMatcher(a=source, b=prompt, autojunk=False)
        longest = matcher.find_longest_match(0, len(source), 0, len(prompt)).size
        return longest >= threshold


    @staticmethod
    def _validate_inline_markup(candidate: LevelTestQuestionCandidate) -> None:
        prompt = candidate.prompt_text
        if "**" in prompt:
            raise ValueError("promptText에는 Markdown 굵은 글씨를 사용할 수 없습니다.")
        if re.search(r"</?[A-Za-z][^>]*>", prompt):
            raise ValueError("promptText에는 HTML/XML 마크업을 사용할 수 없습니다.")
        if candidate.item_type == LevelTestItemType.VOCAB_PARAPHRASE_CHOICE:
            emphasis = candidate.reference_payload.get("emphasisText")
            if not isinstance(emphasis, str) or not emphasis.strip():
                raise ValueError("VOCAB_PARAPHRASE_CHOICE에는 referencePayload.emphasisText가 필요합니다.")
            target = emphasis.strip()
            if len(target) > 80 or candidate.prompt_text.count(target) != 1:
                raise ValueError("VOCAB_PARAPHRASE_CHOICE emphasisText는 promptText에 정확히 한 번 존재해야 합니다.")


    @staticmethod
    def _validate_guided_task_payload(candidate: LevelTestQuestionCandidate) -> None:
        facts = candidate.reference_payload.get("providedFacts")
        intents = candidate.reference_payload.get("requiredIntents")
        constraints = candidate.reference_payload.get("responseConstraints")
        if not isinstance(facts, list) or not all(isinstance(item, str) and item.strip() for item in facts):
            raise ValueError(f"{candidate.item_type.value} providedFacts가 유효하지 않습니다.")
        if not isinstance(intents, list) or not all(isinstance(item, str) and item.strip() for item in intents):
            raise ValueError(f"{candidate.item_type.value} requiredIntents가 유효하지 않습니다.")
        if any(re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", item.strip()) for item in intents):
            raise ValueError(
                f"{candidate.item_type.value} requiredIntents에 내부 enum token을 노출할 수 없습니다."
            )
        if not isinstance(constraints, list) or not all(isinstance(item, str) and item.strip() for item in constraints):
            raise ValueError(f"{candidate.item_type.value} responseConstraints가 유효하지 않습니다.")

        minimum_facts = 2 if candidate.item_type in {
            LevelTestItemType.WRITING_SCENARIO_RESPONSE,
            LevelTestItemType.WRITING_SHORT_PARAGRAPH,
            LevelTestItemType.SPEAKING_SHORT_RESPONSE,
        } else 1
        minimum_intents = 2 if candidate.item_type in {
            LevelTestItemType.WRITING_SCENARIO_RESPONSE,
            LevelTestItemType.WRITING_SHORT_PARAGRAPH,
            LevelTestItemType.SPEAKING_SHORT_RESPONSE,
        } else 1
        if len(facts) < minimum_facts:
            raise ValueError(f"{candidate.item_type.value}에는 최소 {minimum_facts}개의 제공 사실이 필요합니다.")
        if len(intents) < minimum_intents:
            raise ValueError(f"{candidate.item_type.value}에는 최소 {minimum_intents}개의 필수 의사기능이 필요합니다.")
        if not constraints:
            raise ValueError(f"{candidate.item_type.value}에는 최소 1개의 응답 제약이 필요합니다.")

    @staticmethod
    def _validate_vocab_context_choice(candidate: LevelTestQuestionCandidate) -> None:
        """Apply only deterministic checks here.

        The generation-side choiceQualityAudit is deliberately not trusted as a
        correctness proof. Semantic uniqueness is checked independently by
        _verify_vocab_context_choice_semantics.
        """

        normalized_options = [
            LevelTestService._normalize_choice_text(option.text)
            for option in candidate.options
        ]
        if len(set(normalized_options)) != len(normalized_options):
            raise ValueError("VOCAB_CONTEXT_CHOICE Option에 실질적으로 동일한 표현이 중복되어 있습니다.")
        LevelTestService._validate_vocab_answer_not_exposed(candidate)

    @staticmethod
    def _validate_vocab_answer_not_exposed(
        candidate: LevelTestQuestionCandidate,
    ) -> None:
        """Reject vocabulary items that print the correct option in the question.

        This is intentionally deterministic.  Vocabulary context/paraphrase items
        must test recognition from context; showing the exact correct option text in
        promptText turns the item into an answer giveaway.
        """

        correct_key = candidate.internal_answer_key.correct_option_key
        if not correct_key:
            return
        correct_text = next(
            (option.text for option in candidate.options if option.key == correct_key),
            None,
        )
        if not correct_text:
            return
        normalized_answer = LevelTestService._normalize_choice_text(correct_text)
        normalized_prompt = LevelTestService._normalize_choice_text(candidate.prompt_text)
        if normalized_answer and normalized_answer in normalized_prompt:
            raise ValueError(
                f"{candidate.item_type.value} promptText에 정답 표현이 직접 노출되어 있습니다."
            )

    @staticmethod
    def _normalize_choice_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return re.sub(r"\s+", "", normalized)

    @staticmethod
    def _choice_selection_policy(
        item_type: LevelTestItemType,
        complexity_band: int,
    ) -> LevelTestChoiceSelectionPolicy | None:
        """Return the fairness contract for semantic multiple-choice verification.

        Lower bands favor unambiguous single-answer discrimination. Higher bands
        may intentionally test nuance where several options are technically possible
        but one must still be clearly the most appropriate.
        """

        if item_type == LevelTestItemType.GRAMMAR_SENTENCE_ORDER:
            return None

        if item_type == LevelTestItemType.VOCAB_CONTEXT_CHOICE:
            return (
                LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER
                if complexity_band <= 3
                else LevelTestChoiceSelectionPolicy.BEST_ANSWER
            )
        if item_type == LevelTestItemType.VOCAB_PARAPHRASE_CHOICE:
            return (
                LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER
                if complexity_band <= 2
                else LevelTestChoiceSelectionPolicy.BEST_ANSWER
            )
        if item_type == LevelTestItemType.GRAMMAR_FORM_CHOICE:
            return (
                LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER
                if complexity_band <= 3
                else LevelTestChoiceSelectionPolicy.BEST_ANSWER
            )
        if item_type in {
            LevelTestItemType.READING_DISCOURSE_FUNCTION,
            LevelTestItemType.READING_TEXT_INFERENCE,
        }:
            return (
                LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER
                if complexity_band <= 2
                else LevelTestChoiceSelectionPolicy.BEST_ANSWER
            )
        if item_type in {
            LevelTestItemType.READING_GIST,
            LevelTestItemType.READING_DETAIL,
            LevelTestItemType.LISTENING_GIST_CHOICE,
            LevelTestItemType.LISTENING_DETAIL_CHOICE,
        }:
            return (
                LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER
                if complexity_band <= 3
                else LevelTestChoiceSelectionPolicy.BEST_ANSWER
            )
        return None

    @staticmethod
    def _choice_verifier_passes(
        policy: LevelTestChoiceSelectionPolicy,
        complexity_band: int,
    ) -> int:
        # Advanced BEST_ANSWER items are intentionally judged twice. The two
        # independent calls may disagree on secondary plausible options, but both
        # must still find the server-held answer clearly best.
        if policy == LevelTestChoiceSelectionPolicy.BEST_ANSWER:
            return 2
        return 1

    async def _call_choice_semantic_verifier(
        self,
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
        policy: LevelTestChoiceSelectionPolicy,
        *,
        verification_pass: int = 1,
    ) -> tuple[LevelTestChoiceSemanticVerificationPayload, _StageUsage]:
        """Call an independent semantic verifier without revealing the answer key."""

        prompt = build_level_test_choice_semantic_verification_prompt(
            candidate,
            origin_language=request.origin_language,
            learning_language=request.learning_language,
            selection_policy=policy.value,
            verification_pass=verification_pass,
        )
        schema = LevelTestChoiceSemanticVerificationPayload.model_json_schema()
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.provider.call_with_metadata(
                    type_name=self.CHOICE_VERIFICATION_TYPE,
                    data=prompt,
                    schema=schema,
                ),
                timeout=self.generation_timeout_seconds,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise HTTPException(
                status_code=504,
                detail="Level Test 객관식 의미 검증 시간이 초과되었습니다.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Level Test 객관식 의미 검증 Provider 호출에 실패했습니다.",
            ) from exc

        usage = _StageUsage(
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        if not isinstance(result.data, dict):
            raise HTTPException(
                status_code=502,
                detail="Level Test 객관식 의미 검증 응답 Schema가 유효하지 않습니다.",
            )
        try:
            verdict = LevelTestChoiceSemanticVerificationPayload.model_validate(result.data)
        except ValidationError as exc:
            raise HTTPException(
                status_code=502,
                detail="Level Test 객관식 의미 검증 응답 Schema가 유효하지 않습니다.",
            ) from exc
        return verdict, usage

    @staticmethod
    def _choice_semantic_rejection_reason(
        candidate: LevelTestQuestionCandidate,
        verdict: LevelTestChoiceSemanticVerificationPayload,
        policy: LevelTestChoiceSelectionPolicy,
    ) -> str | None:
        option_keys = {option.key for option in candidate.options}
        plausible = list(dict.fromkeys(verdict.plausible_option_keys))
        near_equivalent = list(dict.fromkeys(verdict.near_equivalent_option_keys))
        if any(key not in option_keys for key in plausible + near_equivalent):
            return f"{candidate.item_type.value} 의미 검증 결과에 존재하지 않는 Option key가 포함되어 있습니다."
        if verdict.best_option_key not in option_keys:
            return f"{candidate.item_type.value} 의미 검증 결과의 bestOptionKey가 유효하지 않습니다."
        if verdict.best_option_key in near_equivalent:
            return f"{candidate.item_type.value} 의미 검증의 nearEquivalentOptionKeys에는 bestOptionKey 자체를 포함할 수 없습니다."
        if not verdict.verifiable:
            return f"{candidate.item_type.value} 문항을 공정한 단일 정답 문제로 독립 검증할 수 없습니다."

        correct_key = candidate.internal_answer_key.correct_option_key
        if correct_key is None:
            return f"{candidate.item_type.value} 서버 정답 Key가 없습니다."

        if policy == LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER:
            if near_equivalent:
                return (
                    f"{candidate.item_type.value} UNIQUE_ANSWER 검증에서 기능적으로 동등한 경쟁 선택지가 발견되었습니다."
                )
            if (
                plausible != [correct_key]
                or verdict.best_option_key != correct_key
                or verdict.best_option_advantage != LevelTestBestOptionAdvantage.CLEAR
            ):
                return (
                    f"{candidate.item_type.value} UNIQUE_ANSWER 검증에서 복수 정답 가능성 또는 정답 불일치가 발견되었습니다."
                )
            return None

        if near_equivalent:
            return (
                f"{candidate.item_type.value} BEST_ANSWER 검증에서 정답과 기능적으로 거의 동등한 경쟁 선택지가 발견되었습니다."
            )
        if (
            correct_key not in plausible
            or verdict.best_option_key != correct_key
            or verdict.best_option_advantage != LevelTestBestOptionAdvantage.CLEAR
        ):
            return (
                f"{candidate.item_type.value} BEST_ANSWER 검증에서 서버 정답이 명백한 최선의 선택지로 확인되지 않았습니다."
            )
        return None

    @staticmethod
    def _vocab_context_repairable_semantic_failure(
        candidate: LevelTestQuestionCandidate,
        verdict: LevelTestChoiceSemanticVerificationPayload,
        policy: LevelTestChoiceSelectionPolicy,
    ) -> bool:
        """Repair only ambiguity that can plausibly be fixed by context/distractors."""

        if not verdict.verifiable or verdict.near_equivalent_option_keys:
            return False
        correct_key = candidate.internal_answer_key.correct_option_key
        plausible = list(dict.fromkeys(verdict.plausible_option_keys))
        if correct_key is None or correct_key not in plausible:
            return False
        if verdict.best_option_key != correct_key:
            return False
        if policy == LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER:
            return len(plausible) > 1
        return verdict.best_option_advantage != LevelTestBestOptionAdvantage.CLEAR

    async def _verify_choice_candidate(
        self,
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
        policy: LevelTestChoiceSelectionPolicy,
    ) -> tuple[
        str | None,
        LevelTestChoiceSemanticVerificationPayload | None,
        _StageUsage,
        int,
        dict[str, int],
    ]:
        total_usage = _StageUsage()
        verdicts: list[LevelTestChoiceSemanticVerificationPayload] = []
        calls = 0
        for verification_pass in range(1, self._choice_verifier_passes(policy, candidate.complexity_band) + 1):
            verdict, usage = await self._call_choice_semantic_verifier(
                request,
                candidate,
                policy,
                verification_pass=verification_pass,
            )
            calls += 1
            total_usage = _StageUsage(
                latency_ms=total_usage.latency_ms + usage.latency_ms,
                input_tokens=total_usage.input_tokens + usage.input_tokens,
                output_tokens=total_usage.output_tokens + usage.output_tokens,
            )
            verdicts.append(verdict)
            reason = self._choice_semantic_rejection_reason(candidate, verdict, policy)
            if reason is not None:
                return reason, verdict, total_usage, calls, {}

        return (
            None,
            verdicts[-1] if verdicts else None,
            total_usage,
            calls,
            self._derive_choice_option_scores(candidate, policy, verdicts),
        )

    @staticmethod
    def _derive_choice_option_scores(
        candidate: LevelTestQuestionCandidate,
        policy: LevelTestChoiceSelectionPolicy,
        verdicts: list[LevelTestChoiceSemanticVerificationPayload],
    ) -> dict[str, int]:
        """Convert independent verifier consensus into deterministic partial credit.

        UNIQUE_ANSWER remains binary.  BEST_ANSWER gives small credit only to
        alternatives independently judged plausible; the AI never authors a numeric
        score.  Two-pass consensus prevents one noisy verifier call from granting a
        large score.
        """

        correct_key = candidate.internal_answer_key.correct_option_key
        if correct_key is None:
            return {}
        keys = [option.key for option in candidate.options]
        if policy == LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER:
            return {key: (100 if key == correct_key else 0) for key in keys}
        if any(verdict.near_equivalent_option_keys for verdict in verdicts):
            return {key: (100 if key == correct_key else 0) for key in keys}

        plausible_counts = Counter(
            key
            for verdict in verdicts
            for key in set(verdict.plausible_option_keys)
            if key in keys and key != correct_key
        )
        pass_count = max(1, len(verdicts))
        scores: dict[str, int] = {}
        for key in keys:
            if key == correct_key:
                scores[key] = 100
                continue
            count = plausible_counts.get(key, 0)
            if count >= pass_count:
                scores[key] = 30
            elif count > 0:
                scores[key] = 10
            else:
                scores[key] = 0
        return scores

    async def _verify_vocab_context_choice_semantics(
        self,
        request: LevelTestQuestionGenerationRequest,
        candidate: LevelTestQuestionCandidate,
    ) -> tuple[int, int, int]:
        """Backward-compatible wrapper used by tests and direct callers."""

        policy = self._choice_selection_policy(candidate.item_type, candidate.complexity_band)
        if policy is None:
            return 0, 0, 0
        reason, _, usage, _, _ = await self._verify_choice_candidate(request, candidate, policy)
        if reason is not None:
            raise ValueError(reason)
        return usage.latency_ms, usage.input_tokens, usage.output_tokens

    @staticmethod
    def _validate_grammar_form_choice(candidate: LevelTestQuestionCandidate) -> None:
        matches = list(re.finditer(r"(?:_{2,}|＿{2,})", candidate.prompt_text))
        if len(matches) != 1:
            raise ValueError("GRAMMAR_FORM_CHOICE에는 정확히 하나의 빈칸이 필요합니다.")
        correct_key = candidate.internal_answer_key.correct_option_key
        correct = next((option.text for option in candidate.options if option.key == correct_key), None)
        if not correct or LevelTestService._has_boundary_duplication(
            candidate.prompt_text, matches[0], correct
        ):
            raise ValueError("GRAMMAR_FORM_CHOICE 정답을 빈칸에 삽입했을 때 활용 형태가 중복됩니다.")

    @staticmethod
    def _has_boundary_duplication(prompt: str, blank: re.Match[str], option: str) -> bool:
        left = prompt[: blank.start()].rstrip()
        right = prompt[blank.end() :].lstrip()
        answer = option.strip()
        if not answer:
            return True
        for size in range(min(3, len(answer), len(right)), 0, -1):
            overlap = answer[-size:]
            if overlap == right[:size] and any(char.isalnum() or char.isalpha() for char in overlap):
                return True
        for size in range(min(3, len(answer), len(left)), 0, -1):
            overlap = answer[:size]
            if overlap == left[-size:] and any(char.isalnum() or char.isalpha() for char in overlap):
                return True
        return False

    @staticmethod
    def _validate_reading_candidate(candidate: LevelTestQuestionCandidate) -> None:
        passage = candidate.reference_payload.get("readingPassage")
        question = candidate.reference_payload.get("readingQuestion")
        if not isinstance(passage, str) or not passage.strip():
            raise ValueError("Reading Item에는 referencePayload.readingPassage가 필요합니다.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Reading Item에는 referencePayload.readingQuestion이 필요합니다.")

        passage = passage.strip()
        question = question.strip()
        canonical_prompt = f"{passage}\n\n{question}"
        if candidate.prompt_text.strip() != canonical_prompt:
            raise ValueError("Reading Item promptText는 passage/question의 canonical 조합이어야 합니다.")
        if len(passage) < 24 or len(question) < 6:
            raise ValueError("Reading Item에는 학습자가 읽을 수 있는 충분한 지문과 질문이 필요합니다.")

        sentence_ends = sum(passage.count(mark) for mark in (".", "!", "?", "。", "！", "？"))
        if sentence_ends < 1:
            raise ValueError("Reading Item readingPassage에 완결된 문맥이 필요합니다.")

        if candidate.item_type == LevelTestItemType.READING_DISCOURSE_FUNCTION:
            emphasis = candidate.reference_payload.get("emphasisText")
            if not isinstance(emphasis, str) or not emphasis.strip():
                raise ValueError("READING_DISCOURSE_FUNCTION에는 emphasisText가 필요합니다.")
            if passage.count(emphasis.strip()) != 1:
                raise ValueError("READING_DISCOURSE_FUNCTION emphasisText는 readingPassage에 정확히 한 번 존재해야 합니다.")

    @staticmethod
    def _diversity_content(candidate: LevelTestQuestionCandidate) -> str:
        source = candidate.reference_payload.get("sourceText")
        if isinstance(source, str) and source.strip():
            return source
        return candidate.prompt_text

    @staticmethod
    def _signals(domain: LevelTestDomain, metrics: list[LevelTestMetricResult]) -> list[LevelTestAssessmentSignal]:
        return [
            LevelTestAssessmentSignal(
                domain=domain,
                metric=item.type,
                score=float(item.score),
                confidence=item.confidence,
            )
            for item in metrics
            if item.state == LevelTestMetricState.EVALUATED and item.score is not None
        ]

    def _listening_eval_response(
        self,
        request: LevelTestTextEvaluationRequest,
        task,
        metrics: list[LevelTestMetricResult],
        evaluation_version: str,
    ) -> LevelTestEvaluationResponse:
        return LevelTestEvaluationResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            item_id=request.item_id,
            domain=LevelTestDomain.LISTENING,
            item_type=request.item_type,
            evaluable=task.evaluable,
            score=task.score,
            confidence=task.confidence,
            metrics=metrics,
            strengths=task.strengths,
            improvements=task.improvements,
            recommended_answers=(
                [request.source_text]
                if request.item_type == LevelTestItemType.LISTENING_DICTATION and request.source_text
                else list(getattr(task, "recommended_interpretations", []) or [])
            ),
            detailed_feedback=self._listening_feedback_details(task),
            assessment_signals=self._signals(LevelTestDomain.LISTENING, metrics) if task.evaluable else [],
            reason_code=task.reason_code,
            evaluation_version=evaluation_version,
        )

    @staticmethod
    def _writing_feedback_details(result) -> list[LevelTestFeedbackDetail]:
        details: list[LevelTestFeedbackDetail] = []
        for correction in result.corrections:
            details.append(LevelTestFeedbackDetail(
                category=correction.category,
                severity="CORRECTION",
                original=correction.original,
                corrected=correction.corrected,
                explanation=correction.explanation.origin_text,
            ))
        for weakness in result.weaknesses:
            details.append(LevelTestFeedbackDetail(
                category="IMPROVEMENT",
                severity="IMPROVEMENT",
                explanation=weakness.origin_text,
            ))
        return details[:50]

    @staticmethod
    def _listening_feedback_details(task) -> list[LevelTestFeedbackDetail]:
        details: list[LevelTestFeedbackDetail] = []
        for evidence in getattr(task, "evidence", []) or []:
            details.append(LevelTestFeedbackDetail(
                category=str(getattr(evidence, "metric", "LISTENING")),
                severity="IMPROVEMENT" if getattr(evidence, "severity", "INFO") != "INFO" else "INFO",
                original=getattr(evidence, "recognized", None),
                corrected=getattr(evidence, "reference", None),
                explanation=getattr(evidence, "feedback", "확인할 부분이 있습니다."),
            ))
        for value in getattr(task, "omitted_meaning_units", []) or []:
            details.append(LevelTestFeedbackDetail(
                category="MEANING_OMISSION", severity="OMISSION", corrected=value,
                explanation=f"답변에서 핵심 의미가 빠졌습니다: {value}",
            ))
        for value in getattr(task, "misunderstood_meaning_units", []) or []:
            details.append(LevelTestFeedbackDetail(
                category="MEANING_MISMATCH", severity="CORRECTION", corrected=value,
                explanation=f"이 의미 단위를 다르게 이해했습니다: {value}",
            ))
        for value in getattr(task, "added_information", []) or []:
            details.append(LevelTestFeedbackDetail(
                category="ADDED_INFORMATION", severity="IMPROVEMENT", original=value,
                explanation=f"원문에 없는 정보가 추가되었습니다: {value}",
            ))
        return details[:50]

    @staticmethod
    def _speaking_feedback_details(
        metrics: list[LevelTestMetricResult],
        improvements: list[str],
    ) -> list[LevelTestFeedbackDetail]:
        details: list[LevelTestFeedbackDetail] = []
        for metric in metrics:
            if metric.summary:
                details.append(LevelTestFeedbackDetail(
                    category=metric.type,
                    severity="IMPROVEMENT" if metric.score is not None and metric.score < 90 else "INFO",
                    explanation=metric.summary,
                ))
            for evidence in metric.evidence:
                message = evidence.get("message") if isinstance(evidence, dict) else None
                if isinstance(message, str) and message.strip():
                    details.append(LevelTestFeedbackDetail(
                        category=metric.type, severity="IMPROVEMENT", explanation=message.strip(),
                    ))
        for improvement in improvements:
            if improvement and not any(item.explanation == improvement for item in details):
                details.append(LevelTestFeedbackDetail(
                    category="TASK", severity="IMPROVEMENT", explanation=improvement,
                ))
        return details[:50]

    @staticmethod
    def _calculate_speaking_item_score(
        item_type: LevelTestItemType,
        metrics: list[LevelTestMetricResult],
    ) -> int | None:
        weights = (
            SPEAKING_REPEAT_WEIGHTS
            if item_type == LevelTestItemType.SPEAKING_REPEAT
            else SPEAKING_RESPONSE_WEIGHTS
        )
        weighted = Decimal("0")
        available = Decimal("0")
        for metric in metrics:
            if metric.type not in weights or metric.state != LevelTestMetricState.EVALUATED or metric.score is None:
                continue
            weight = weights[metric.type]
            weighted += Decimal(str(metric.score)) * weight
            available += weight
        if available <= 0:
            return None
        return int((weighted / available).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    @classmethod
    def _normalize_speaking_evaluation_payload(
        cls,
        request: LevelTestSpeakingEvaluationContext,
        data: dict,
    ) -> dict:
        """Repair provider formatting drift without inventing evaluation scores."""

        normalized = cls._canonicalize_confidence(data)
        if request.item_type == LevelTestItemType.SPEAKING_REPEAT:
            normalized.setdefault("recommendedAnswers", [])
        recommended = normalized.get("recommendedAnswers")
        if isinstance(recommended, str):
            normalized["recommendedAnswers"] = [recommended]
        metrics = normalized.get("metrics")
        if isinstance(metrics, list):
            numeric_scores = [
                metric.get("score")
                for metric in metrics
                if isinstance(metric, dict)
                and isinstance(metric.get("score"), (int, float))
                and not isinstance(metric.get("score"), bool)
            ]
            normalized_score_scale = (
                bool(numeric_scores)
                and len(numeric_scores) == sum(
                    1 for metric in metrics
                    if isinstance(metric, dict) and metric.get("score") is not None
                )
                and all(0 <= float(score) <= 1 for score in numeric_scores)
                and any(0 < float(score) < 1 for score in numeric_scores)
            )
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                metric_type = metric.get("type")
                if isinstance(metric_type, str):
                    token = re.sub(r"[^A-Z0-9]+", "_", metric_type.strip().upper()).strip("_")
                    token = {
                        "TASKFULFILLMENT": "TASK_FULFILLMENT",
                        "TASK_FULFILMENT": "TASK_FULFILLMENT",
                    }.get(token, token)
                    metric["type"] = token
                state = metric.get("state")
                if isinstance(state, str):
                    token = re.sub(r"[^A-Z0-9]+", "_", state.strip().upper()).strip("_")
                    token = {
                        "UNEVALUABLE": "NOT_EVALUABLE",
                        "NOT_EVALUABLE": "NOT_EVALUABLE",
                        "EVALUABLE": "EVALUATED",
                    }.get(token, token)
                    metric["state"] = token
                if normalized_score_scale and isinstance(metric.get("score"), (int, float)):
                    metric["score"] = float(metric["score"]) * 100
                evidence = metric.get("evidence")
                if isinstance(evidence, str):
                    metric["evidence"] = [evidence]
                elif isinstance(evidence, list):
                    normalized_evidence: list[object] = []
                    for item in evidence:
                        if isinstance(item, str):
                            normalized_evidence.append(item)
                        elif isinstance(item, dict) and isinstance(item.get("message"), str):
                            normalized_evidence.append(item["message"])
                        else:
                            normalized_evidence.append(item)
                    metric["evidence"] = normalized_evidence

                if metric.get("state") == "NOT_EVALUABLE":
                    metric["score"] = None
                    reason = metric.get("notEvaluableReason")
                    if not isinstance(reason, str) or not reason.strip():
                        metric["notEvaluableReason"] = cls._not_evaluable_reason(
                            request.origin_language
                        )
                    summary = metric.get("summary")
                    if not isinstance(summary, str) or not summary.strip():
                        metric["summary"] = metric["notEvaluableReason"]

        for key in ("strengths", "improvements"):
            value = normalized.get(key)
            if isinstance(value, str):
                normalized[key] = [value]
            elif isinstance(value, list):
                normalized_list: list[object] = []
                for item in value:
                    if isinstance(item, str):
                        normalized_list.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("message"), str):
                        normalized_list.append(item["message"])
                    else:
                        normalized_list.append(item)
                normalized[key] = normalized_list
        return normalized

    @staticmethod
    def _not_evaluable_reason(origin_language: str) -> str:
        language = origin_language.strip().lower().split("-", 1)[0].split("_", 1)[0]
        return {
            "ko": "이 항목을 평가할 근거가 충분하지 않습니다.",
            "ja": "この項目を評価するための根拠が十分ではありません。",
            "en": "There is not enough evidence to evaluate this metric.",
        }.get(language, "Insufficient evidence for this metric.")

    @staticmethod
    def _canonicalize_confidence(data: dict) -> dict:
        normalized = copy.deepcopy(data)
        values = [(normalized, "evaluationConfidence")]
        metrics = normalized.get("metrics")
        if isinstance(metrics, list):
            values.extend((metric, "confidence") for metric in metrics if isinstance(metric, dict))
        for target, key in values:
            value = target.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if 1 < value <= 5:
                    target[key] = value / 5
                elif 5 < value <= 100:
                    target[key] = value / 100
        return normalized

    @staticmethod
    def _is_transient_provider_error(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        return status in {429, 500, 502, 503, 504} or isinstance(exc, (TimeoutError, asyncio.TimeoutError))
