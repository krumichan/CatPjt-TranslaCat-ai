from __future__ import annotations

import asyncio
import hashlib
import math
import time
from typing import Protocol

from app.common.idempotency import InMemoryIdempotencyStore
from app.core.config import settings
from app.features.language_learning.listening.acoustic import (
    ListeningAcousticEvidence,
    analyze_normalized_wav,
)
from app.features.language_learning.listening.alignment import (
    align_tokens,
    attach_segment_timestamps,
)
from app.features.language_learning.listening.errors import ListeningStageException
from app.features.language_learning.listening.evaluation_common import (
    answer_revealed_task,
    build_evaluation_response,
    effective_answer_revealed,
    finalize_task,
    not_evaluable_task,
)
from app.features.language_learning.listening.normalization import normalize_text
from app.features.language_learning.listening.policy import (
    LISTENING_ALIGNMENT_VERSION,
    LISTENING_REPEAT_EVALUATOR_VERSION,
    LISTENING_STT_HINT_VERSION,
    REPEAT_WEIGHTS,
    calculate_weighted_score,
    resolve_assistance_level,
    round_score,
)
from app.features.language_learning.listening.provider_error import (
    map_provider_exception,
)
from app.features.language_learning.listening.retry import run_with_stage_retry
from app.features.language_learning.speaking.audio_processor import (
    NormalizedAudio,
)
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.stt_service import (
    SpeakingSttProvider,
    SttProviderResult,
)
from app.schemas.language_learning_listening import (
    AlignmentStatus,
    ListeningErrorCode,
    ListeningEvaluationResponse,
    ListeningMetric,
    ListeningStage,
    ListeningTaskResult,
    ListeningTaskStatus,
    ListeningTaskType,
    ListeningUsage,
    MetricEvidence,
    RepeatEvaluationContext,
    RepeatMetricType,
    StageUsage,
)


class ListeningAudioProcessor(Protocol):
    def validate_and_normalize(
        self,
        audio_bytes: bytes,
        *,
        file_name: str | None,
        content_type: str | None,
        min_seconds: float,
        max_seconds: float,
        max_bytes: int,
    ) -> NormalizedAudio: ...


class ListeningRepeatService:
    def __init__(
        self,
        audio_processor: ListeningAudioProcessor,
        stt_provider: SpeakingSttProvider,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
        idempotency_store: InMemoryIdempotencyStore[ListeningEvaluationResponse]
        | None = None,
    ) -> None:
        self.audio_processor = audio_processor
        self.stt_provider = stt_provider
        self.timeout_seconds = (
            timeout_seconds or settings.AI_LISTENING_STT_TIMEOUT_SECONDS
        )
        self.automatic_retries = (
            settings.AI_LISTENING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else min(automatic_retries, 2)
        )
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def evaluate(
        self,
        request: RepeatEvaluationContext,
        *,
        audio_bytes: bytes,
        file_name: str | None,
        content_type: str | None,
    ) -> ListeningEvaluationResponse:
        self._validate_manual_retry(request.manual_retry_attempt)
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        key = "|".join(
            [
                request.idempotency_key,
                request.evaluation_purpose.value,
                request.policy_version,
                request.model_config_version,
                audio_hash,
            ]
        )
        response, _ = await self.idempotency_store.execute(
            key,
            lambda: self._evaluate_once(
                request,
                audio_bytes=audio_bytes,
                file_name=file_name,
                content_type=content_type,
            ),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _evaluate_once(
        self,
        request: RepeatEvaluationContext,
        *,
        audio_bytes: bytes,
        file_name: str | None,
        content_type: str | None,
    ) -> ListeningEvaluationResponse:
        if effective_answer_revealed(request):
            task = answer_revealed_task(request, ListeningTaskType.REPEAT_AFTER_AUDIO)
            return build_evaluation_response(request, task)

        started = time.perf_counter()
        maximum_seconds = min(
            settings.AI_LISTENING_MAX_REPEAT_AUDIO_SECONDS,
            request.source_duration_seconds + 15,
        )
        try:
            normalized_audio = self.audio_processor.validate_and_normalize(
                audio_bytes,
                file_name=file_name,
                content_type=content_type,
                min_seconds=settings.AI_LISTENING_MIN_VALID_AUDIO_SECONDS,
                max_seconds=maximum_seconds,
                max_bytes=settings.AI_LISTENING_MAX_AUDIO_FILE_BYTES,
            )
        except SpeakingStageException as exc:
            task = not_evaluable_task(
                request,
                ListeningTaskType.REPEAT_AFTER_AUDIO,
                reason_code=self._map_audio_error(exc),
                improvements=[exc.message],
            )
            return build_evaluation_response(request, task)

        acoustic_evidence = analyze_normalized_wav(normalized_audio.wav_bytes)
        quality_score = self._audio_quality_score(
            normalized_audio.quality,
            acoustic_evidence,
        )
        if self._is_low_audio_quality(quality_score, acoustic_evidence):
            task = not_evaluable_task(
                request,
                ListeningTaskType.REPEAT_AFTER_AUDIO,
                reason_code=ListeningErrorCode.LOW_AUDIO_QUALITY.value,
                confidence=round(quality_score, 4),
                improvements=[
                    "소음이 적고 음량이 안정적인 환경에서 다시 녹음해 주세요."
                ],
                debug_metadata={
                    "audioQualityScore": round(quality_score, 4),
                    **self._acoustic_metadata(acoustic_evidence),
                },
            )
            return build_evaluation_response(request, task)

        try:
            stt_result = await self._transcribe(request, normalized_audio.wav_bytes)
        except ListeningStageException as exc:
            task = not_evaluable_task(
                request,
                ListeningTaskType.REPEAT_AFTER_AUDIO,
                reason_code=exc.code.value,
                improvements=[exc.message],
            )
            return build_evaluation_response(request, task)

        if not stt_result.text.strip():
            task = not_evaluable_task(
                request,
                ListeningTaskType.REPEAT_AFTER_AUDIO,
                reason_code=ListeningErrorCode.NON_SPEECH.value,
                improvements=["전사 가능한 발화가 확인되지 않았습니다."],
            )
            return build_evaluation_response(request, task)
        if self._is_language_mismatch(request.learning_language, stt_result):
            task = not_evaluable_task(
                request,
                ListeningTaskType.REPEAT_AFTER_AUDIO,
                reason_code=ListeningErrorCode.LANGUAGE_MISMATCH.value,
                confidence=round(stt_result.language_probability, 4),
                improvements=["학습 언어로 원문을 다시 따라 말해 주세요."],
                debug_metadata={
                    "expectedLanguage": request.learning_language,
                    "detectedLanguage": stt_result.language,
                },
            )
            return build_evaluation_response(request, task)

        reference = normalize_text(request.source_text, request.learning_language)
        transcript = normalize_text(stt_result.text, request.learning_language)
        alignment = align_tokens(reference.tokens, transcript.tokens)
        segment_ranges = [
            (
                max(0, int(segment.start_seconds * 1000)),
                max(0, int(segment.end_seconds * 1000)),
            )
            for segment in stt_result.segments
        ]
        timestamped_alignment = attach_segment_timestamps(
            alignment.entries,
            segment_ranges=segment_ranges,
        )
        if alignment.reference_coverage < 0.30:
            task = not_evaluable_task(
                request,
                ListeningTaskType.REPEAT_AFTER_AUDIO,
                reason_code=ListeningErrorCode.ALIGNMENT_INSUFFICIENT.value,
                confidence=round(alignment.reference_coverage, 4),
                improvements=[
                    "원문과 겹치는 발화가 부족합니다. 전체 문장을 다시 따라 말해 주세요."
                ],
                debug_metadata={
                    "alignmentVersion": LISTENING_ALIGNMENT_VERSION,
                    "referenceCoverage": round(alignment.reference_coverage, 4),
                },
            )
            return build_evaluation_response(request, task)

        segment_confidences = [
            max(0.0, min(1.0, math.exp(segment.avg_logprob)))
            for segment in stt_result.segments
        ]
        segment_confidence = (
            sum(segment_confidences) / len(segment_confidences)
            if segment_confidences
            else stt_result.language_probability
        )
        stt_confidence = max(
            0.0,
            min(
                1.0,
                segment_confidence * 0.70 + stt_result.language_probability * 0.30,
            ),
        )
        duration_ratio = (
            normalized_audio.duration_seconds / request.source_duration_seconds
        )
        tempo_score = self._tempo_score(duration_ratio)
        speech_ratio = 1.0 - normalized_audio.quality.silence_ratio
        recognition_ratio = alignment.recognition_ratio

        scores = {
            RepeatMetricType.PRONUNCIATION: round_score(
                (
                    recognition_ratio * 0.50
                    + stt_confidence * 0.25
                    + quality_score * 0.25
                )
                * 100
            ),
            RepeatMetricType.PROSODY_RHYTHM: round_score(
                (tempo_score * 0.60 + quality_score * 0.40) * 100
            ),
            RepeatMetricType.FLUENCY: round_score(
                (tempo_score * 0.45 + speech_ratio * 0.35 + stt_confidence * 0.20) * 100
            ),
            RepeatMetricType.COMPLETENESS: round_score(
                alignment.reference_coverage * 100
            ),
        }
        metrics = self._build_metrics(
            scores,
            confidence=min(stt_confidence, max(quality_score, 0.01)),
            quality_score=quality_score,
            duration_ratio=duration_ratio,
            speech_ratio=speech_ratio,
            alignment_coverage=alignment.reference_coverage,
        )
        score = calculate_weighted_score(metrics, REPEAT_WEIGHTS)
        assert score is not None
        evaluation_confidence = max(
            0.0,
            min(
                1.0,
                stt_confidence * 0.40
                + alignment.reference_coverage * 0.35
                + quality_score * 0.25,
            ),
        )
        evidence = self._alignment_evidence(timestamped_alignment)
        evidence.extend(item for metric in metrics for item in metric.evidence)
        task = ListeningTaskResult(
            task_type=ListeningTaskType.REPEAT_AFTER_AUDIO,
            status=ListeningTaskStatus.EVALUATED,
            evaluable=True,
            score=score,
            confidence=round(evaluation_confidence, 4),
            assistance_level=resolve_assistance_level(request.assistance_usage),
            metrics=metrics,
            alignment=timestamped_alignment,
            evidence=evidence,
            strengths=(
                ["원문의 핵심 Token과 흐름을 안정적으로 재현했습니다."]
                if score >= 80
                else ["인식된 구간을 끝까지 따라 말했습니다."]
            ),
            improvements=self._repeat_improvements(scores),
            profile_eligible=False,
            assistance_usage=request.assistance_usage,
            debug_metadata={
                "repeatEvaluatorVersion": LISTENING_REPEAT_EVALUATOR_VERSION,
                "alignmentVersion": LISTENING_ALIGNMENT_VERSION,
                "sttHintVersion": LISTENING_STT_HINT_VERSION,
                "supportedEvidence": [
                    "TOKEN_ALIGNMENT",
                    "STT_SEGMENT_CONFIDENCE",
                    "RMS",
                    "PEAK",
                    "SILENCE_RATIO",
                    "DURATION_RATIO",
                ],
                "phonemeProviderScoreAvailable": False,
                "audioQualityScore": round(quality_score, 4),
                "durationRatio": round(duration_ratio, 4),
                "detectedLanguage": stt_result.language,
                **self._acoustic_metadata(acoustic_evidence),
            },
        )
        task = finalize_task(request, task)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return build_evaluation_response(
            request,
            task,
            usage=ListeningUsage(
                stt=StageUsage(
                    latency_ms=elapsed_ms,
                    audio_seconds=normalized_audio.duration_seconds,
                    provider=stt_result.provider,
                    model=stt_result.model,
                    provider_version=stt_result.model_version,
                ),
                alignment=StageUsage(
                    latency_ms=0,
                    evaluation_version=LISTENING_ALIGNMENT_VERSION,
                ),
                pronunciation=StageUsage(
                    latency_ms=0,
                    evaluation_version=LISTENING_REPEAT_EVALUATOR_VERSION,
                ),
            ),
        )

    async def _transcribe(
        self,
        request: RepeatEvaluationContext,
        wav_bytes: bytes,
    ) -> SttProviderResult:
        async def operation() -> SttProviderResult:
            try:
                return await asyncio.wait_for(
                    self.stt_provider.transcribe(
                        wav_bytes,
                        language=request.learning_language,
                        phrase_hints=[request.source_text, *request.phrase_hints],
                    ),
                    timeout=self.timeout_seconds,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise ListeningStageException(
                    ListeningErrorCode.PROVIDER_TIMEOUT,
                    ListeningStage.STT,
                    "Repeat STT Provider 응답 시간이 초과되었습니다.",
                    True,
                ) from exc
            except ListeningStageException:
                raise
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=ListeningStage.STT,
                    fallback_code=ListeningErrorCode.STT_FAILED,
                    fallback_message="Repeat Audio 전사에 실패했습니다.",
                ) from exc

        return await run_with_stage_retry(
            operation,
            max_retries=self.automatic_retries,
        )

    @staticmethod
    def _audio_quality_score(
        quality,
        acoustic_evidence: ListeningAcousticEvidence | None = None,
    ) -> float:
        rms_score = max(0.0, min(1.0, quality.rms / 0.02))
        clipping_score = (
            1.0 if quality.peak < 0.98 else max(0.2, 1.0 - (quality.peak - 0.98) * 40)
        )
        speech_score = max(0.0, min(1.0, (1.0 - quality.silence_ratio) / 0.80))
        base_score = max(
            0.0,
            min(1.0, rms_score * 0.40 + clipping_score * 0.30 + speech_score * 0.30),
        )
        if acoustic_evidence is None:
            return base_score
        return max(
            0.0,
            min(
                1.0,
                base_score * 0.45
                + acoustic_evidence.clipping_score * 0.25
                + acoustic_evidence.noise_score * 0.30,
            ),
        )

    @staticmethod
    def _is_low_audio_quality(
        quality_score: float,
        acoustic_evidence: ListeningAcousticEvidence | None,
    ) -> bool:
        return quality_score < 0.25 or (
            acoustic_evidence is not None
            and (
                acoustic_evidence.clipping_ratio > 0.10
                or acoustic_evidence.noise_score < 0.15
            )
        )

    @staticmethod
    def _acoustic_metadata(
        evidence: ListeningAcousticEvidence | None,
    ) -> dict[str, float]:
        if evidence is None:
            return {}
        return {
            "clippingRatio": evidence.clipping_ratio,
            "zeroCrossingRate": evidence.zero_crossing_rate,
            "noiseScore": evidence.noise_score,
            "clippingScore": evidence.clipping_score,
        }

    @staticmethod
    def _tempo_score(duration_ratio: float) -> float:
        if duration_ratio <= 0:
            return 0.0
        return max(0.0, 1.0 - min(abs(math.log(duration_ratio, 2)), 1.0))

    @staticmethod
    def _is_language_mismatch(expected: str, result: SttProviderResult) -> bool:
        expected_primary = expected.lower().split("-")[0]
        actual_primary = result.language.lower().split("-")[0]
        return (
            bool(actual_primary)
            and actual_primary != expected_primary
            and result.language_probability >= 0.50
        )

    @staticmethod
    def _build_metrics(
        scores: dict[RepeatMetricType, int],
        *,
        confidence: float,
        quality_score: float,
        duration_ratio: float,
        speech_ratio: float,
        alignment_coverage: float,
    ) -> list[ListeningMetric]:
        feedback = {
            RepeatMetricType.PRONUNCIATION: (
                f"Token alignment와 음향 품질({quality_score:.2f})을 함께 확인했습니다."
            ),
            RepeatMetricType.PROSODY_RHYTHM: (
                f"Reference 대비 발화 길이 비율({duration_ratio:.2f})과 음향 흐름을 확인했습니다."
            ),
            RepeatMetricType.FLUENCY: (
                f"발화 구간 비율({speech_ratio:.2f})과 속도 안정성을 확인했습니다."
            ),
            RepeatMetricType.COMPLETENESS: (
                f"원문 Token Coverage({alignment_coverage:.2f})를 확인했습니다."
            ),
        }
        return [
            ListeningMetric(
                type=metric_type.value,
                score=scores[metric_type],
                weight=REPEAT_WEIGHTS[metric_type.value],
                confidence=round(confidence, 4),
                evidence=[
                    MetricEvidence(
                        metric=metric_type.value,
                        severity="INFO",
                        feedback=feedback[metric_type],
                    )
                ],
            )
            for metric_type in RepeatMetricType
        ]

    @staticmethod
    def _alignment_evidence(entries) -> list[MetricEvidence]:
        evidence: list[MetricEvidence] = []
        for entry in entries:
            if entry.status in {
                AlignmentStatus.MATCH,
                AlignmentStatus.ACCEPTED_VARIANT,
            }:
                continue
            evidence.append(
                MetricEvidence(
                    start_ms=entry.start_ms,
                    end_ms=entry.end_ms,
                    reference=entry.source,
                    recognized=entry.answer,
                    metric=RepeatMetricType.PRONUNCIATION.value,
                    severity=(
                        "HIGH"
                        if entry.status
                        in {AlignmentStatus.OMISSION, AlignmentStatus.SUBSTITUTION}
                        else "MEDIUM"
                    ),
                    feedback="원문과 다르게 정렬된 소리를 다시 따라 말해 보세요.",
                )
            )
        return evidence

    @staticmethod
    def _repeat_improvements(scores: dict[RepeatMetricType, int]) -> list[str]:
        labels = {
            RepeatMetricType.PRONUNCIATION: "소리의 명료도",
            RepeatMetricType.PROSODY_RHYTHM: "억양과 리듬",
            RepeatMetricType.FLUENCY: "끊김 없는 발화",
            RepeatMetricType.COMPLETENESS: "문장 완성도",
        }
        weakest = sorted(scores, key=lambda metric: scores[metric])[:2]
        return [
            f"다음 시도에서는 {labels[metric]}에 집중해 보세요." for metric in weakest
        ]

    @staticmethod
    def _map_audio_error(exc: SpeakingStageException) -> str:
        mapping = {
            "INVALID_AUDIO": ListeningErrorCode.AUDIO_DECODE_FAILED.value,
            "UNSUPPORTED_AUDIO_FORMAT": ListeningErrorCode.AUDIO_DECODE_FAILED.value,
            "AUDIO_TOO_SHORT": ListeningErrorCode.AUDIO_TOO_SHORT.value,
            "AUDIO_TOO_LONG": ListeningErrorCode.AUDIO_TOO_LONG.value,
            "AUDIO_TOO_LARGE": ListeningErrorCode.AUDIO_TOO_LARGE.value,
            "SILENCE_DETECTED": ListeningErrorCode.SILENCE.value,
        }
        return mapping.get(exc.code.value, ListeningErrorCode.AUDIO_DECODE_FAILED.value)

    @staticmethod
    def _validate_manual_retry(attempt: int) -> None:
        if attempt > settings.AI_LISTENING_MANUAL_RETRY_LIMIT:
            raise ListeningStageException(
                ListeningErrorCode.MANUAL_RETRY_LIMIT_EXCEEDED,
                ListeningStage.STT,
                "Repeat 평가 수동 재시도 가능 횟수를 초과했습니다.",
                False,
            )
