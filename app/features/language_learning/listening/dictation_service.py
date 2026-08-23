from __future__ import annotations

import hashlib
import time
from difflib import SequenceMatcher

from app.common.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.listening.alignment import align_tokens
from app.features.language_learning.listening.evaluation_common import (
    answer_revealed_task,
    build_evaluation_response,
    effective_answer_revealed,
    finalize_task,
)
from app.features.language_learning.listening.normalization import (
    canonicalize_accepted_variants,
    normalize_text,
    tokenize_normalized,
)
from app.features.language_learning.listening.policy import (
    DICTATION_WEIGHTS,
    LISTENING_ALIGNMENT_VERSION,
    LISTENING_NORMALIZATION_VERSION,
    calculate_weighted_score,
    resolve_assistance_level,
    round_score,
)
from app.schemas.language_learning_listening import (
    AlignmentStatus,
    DictationEvaluationRequest,
    DictationMetricType,
    ListeningEvaluationResponse,
    ListeningMetric,
    ListeningTaskResult,
    ListeningTaskStatus,
    ListeningTaskType,
    ListeningUsage,
    MetricEvidence,
    StageUsage,
)


class ListeningDictationService:
    def __init__(
        self,
        *,
        idempotency_store: InMemoryIdempotencyStore[ListeningEvaluationResponse]
        | None = None,
    ) -> None:
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def evaluate(
        self,
        request: DictationEvaluationRequest,
    ) -> ListeningEvaluationResponse:
        key = "|".join(
            [
                request.idempotency_key,
                request.evaluation_purpose.value,
                "DICTATION",
                request.policy_version,
                request.model_config_version,
            ]
        )
        response, _ = await self.idempotency_store.execute(
            key,
            lambda: self._evaluate_once(request),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _evaluate_once(
        self,
        request: DictationEvaluationRequest,
    ) -> ListeningEvaluationResponse:
        if effective_answer_revealed(request):
            task = answer_revealed_task(request, ListeningTaskType.DICTATION)
            return build_evaluation_response(request, task)

        started = time.perf_counter()
        source = normalize_text(request.source_text, request.learning_language)
        answer = normalize_text(request.answer, request.learning_language)
        canonical_answer = canonicalize_accepted_variants(
            answer,
            request.accepted_variants,
            request.learning_language,
        )
        answer_tokens = tokenize_normalized(
            canonical_answer.text,
            request.learning_language,
        )
        summary = align_tokens(
            source.tokens,
            answer_tokens,
            accepted_tokens=canonical_answer.accepted_tokens,
        )
        denominator = max(summary.reference_count, summary.answer_count, 1)
        token_score = round_score(summary.recognition_ratio * 100)
        structure_errors = (
            summary.omission_count + summary.addition_count + summary.order_count
        )
        structure_score = round_score(
            max(0.0, 1 - structure_errors / denominator) * 100
        )
        orthography_score = round_score(
            SequenceMatcher(a=source.text, b=canonical_answer.text).ratio() * 100
        )

        metrics = [
            self._metric(
                DictationMetricType.TOKEN_RECOGNITION,
                token_score,
                "들은 Token의 인식 정확도를 확인했습니다.",
            ),
            self._metric(
                DictationMetricType.OMISSION_ADDITION_ORDER,
                structure_score,
                "누락·추가·어순 Alignment를 확인했습니다.",
            ),
            self._metric(
                DictationMetricType.ORTHOGRAPHY,
                orthography_score,
                "Locale 정규화 후 표기 정확도를 확인했습니다.",
            ),
        ]
        score = calculate_weighted_score(metrics, DICTATION_WEIGHTS)
        assert score is not None
        alignment_evidence = self._alignment_evidence(summary.entries)
        task = ListeningTaskResult(
            task_type=ListeningTaskType.DICTATION,
            status=ListeningTaskStatus.EVALUATED,
            evaluable=True,
            score=score,
            confidence=1.0,
            assistance_level=resolve_assistance_level(request.assistance_usage),
            metrics=metrics,
            alignment=summary.entries,
            evidence=alignment_evidence,
            strengths=(
                ["원문의 Token과 순서를 정확히 받아썼습니다."]
                if score >= 90
                else ["정확히 인식한 Token을 기준으로 문장을 재구성했습니다."]
            ),
            improvements=self._improvements(summary),
            profile_eligible=False,
            assistance_usage=request.assistance_usage,
            debug_metadata={
                "normalizationVersion": LISTENING_NORMALIZATION_VERSION,
                "alignmentVersion": LISTENING_ALIGNMENT_VERSION,
                "normalizedSourceHash": hashlib.sha256(
                    source.text.encode("utf-8")
                ).hexdigest(),
                "normalizedAnswerHash": hashlib.sha256(
                    canonical_answer.text.encode("utf-8")
                ).hexdigest(),
                "sourceTokenCount": len(source.tokens),
                "answerTokenCount": len(answer_tokens),
                "operationCounts": {
                    "omission": summary.omission_count,
                    "addition": summary.addition_count,
                    "substitution": summary.substitution_count,
                    "order": summary.order_count,
                },
            },
        )
        task = finalize_task(request, task)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return build_evaluation_response(
            request,
            task,
            usage=ListeningUsage(
                dictation=StageUsage(
                    latency_ms=elapsed_ms,
                    evaluation_version=LISTENING_ALIGNMENT_VERSION,
                )
            ),
        )

    @staticmethod
    def _metric(
        metric_type: DictationMetricType,
        score: int,
        feedback: str,
    ) -> ListeningMetric:
        return ListeningMetric(
            type=metric_type.value,
            score=score,
            weight=DICTATION_WEIGHTS[metric_type.value],
            confidence=1.0,
            evidence=[
                MetricEvidence(
                    metric=metric_type.value,
                    severity="INFO",
                    feedback=feedback,
                )
            ],
        )

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
                    reference=entry.source,
                    recognized=entry.answer,
                    metric=DictationMetricType.TOKEN_RECOGNITION.value,
                    severity=(
                        "HIGH"
                        if entry.status
                        in {AlignmentStatus.OMISSION, AlignmentStatus.SUBSTITUTION}
                        else "MEDIUM"
                    ),
                    feedback=f"{entry.status.value} 항목을 다시 들어 보세요.",
                )
            )
        return evidence

    @staticmethod
    def _improvements(summary) -> list[str]:
        improvements: list[str] = []
        if summary.omission_count:
            improvements.append("빠뜨린 부분을 구간 반복으로 다시 확인해 보세요.")
        if summary.addition_count:
            improvements.append(
                "들리지 않은 말을 추측해 덧붙이지 않았는지 확인해 보세요."
            )
        if summary.order_count:
            improvements.append(
                "Token은 들었지만 순서가 바뀐 부분을 다시 확인해 보세요."
            )
        if summary.substitution_count:
            improvements.append("다르게 적은 Token의 소리를 집중해서 들어 보세요.")
        return improvements
