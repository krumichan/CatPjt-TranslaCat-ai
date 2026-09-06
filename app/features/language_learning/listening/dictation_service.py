from __future__ import annotations

import hashlib
import logging
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


logger = logging.getLogger(__name__)


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
        logger.info(
            "Listening dictation evaluation request received. "
            "request_id=%s item_id=%s attempt_id=%s learning=%s "
            "answer_chars=%d",
            request.request_id,
            request.item_id,
            request.attempt_id,
            request.learning_language,
            len(request.answer),
        )
        key = "|".join(
            [
                request.idempotency_key,
                request.evaluation_purpose.value,
                "DICTATION",
                request.policy_version,
                request.model_config_version,
            ]
        )
        response, cache_hit = await self.idempotency_store.execute(
            key,
            lambda: self._evaluate_once(request),
        )
        logger.info(
            "Listening dictation evaluation request completed. "
            "request_id=%s item_id=%s attempt_id=%s cache_hit=%s overall_score=%s",
            request.request_id,
            request.item_id,
            request.attempt_id,
            cache_hit,
            response.overall.score,
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
                if not alignment_evidence
                else (
                    ["대부분의 Token과 어순을 정확히 받아썼습니다."]
                    if score >= 90
                    else ["정확히 인식한 Token을 기준으로 문장을 재구성했습니다."]
                )
            ),
            improvements=self._improvements(
                summary,
                learning_language=request.learning_language,
            ),
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
            feedback = {
                AlignmentStatus.OMISSION: (
                    f"원문 「{entry.source}」가 답변에서 누락되었습니다."
                    if entry.source
                    else "원문의 일부가 답변에서 누락되었습니다."
                ),
                AlignmentStatus.ADDITION: (
                    f"원문에 없는 「{entry.answer}」가 답변에 추가되었습니다."
                    if entry.answer
                    else "원문에 없는 표현이 답변에 추가되었습니다."
                ),
                AlignmentStatus.SUBSTITUTION: (
                    f"원문 「{entry.source}」를 「{entry.answer}」로 다르게 받아썼습니다."
                    if entry.source or entry.answer
                    else "원문의 일부를 다른 표현으로 받아썼습니다."
                ),
                AlignmentStatus.ORDER: (
                    f"원문 「{entry.source}」와 답변 「{entry.answer}」의 순서가 다릅니다."
                    if entry.source or entry.answer
                    else "들은 Token의 순서가 바뀌었습니다."
                ),
            }.get(entry.status, "원문과 다른 부분을 다시 확인해 보세요.")
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
                    feedback=feedback,
                )
            )
        return evidence

    @classmethod
    def _improvements(cls, summary, *, learning_language: str) -> list[str]:
        improvements: list[str] = []

        def sample(status, *, limit: int = 3):
            return [entry for entry in summary.entries if entry.status == status][:limit]

        omissions = sample(AlignmentStatus.OMISSION)
        if omissions:
            tokens = "、".join(f"「{entry.source}」" for entry in omissions if entry.source)
            source_context, answer_context = cls._comparison_context(
                summary.entries, omissions[0], learning_language
            )
            improvements.append(
                cls._detailed_improvement(
                    label=f"누락: {tokens}" if tokens else "누락된 표현이 있습니다.",
                    source_context=source_context,
                    answer_context=answer_context,
                    advice="짧게 발음되는 표현이 빠지지 않았는지 해당 구간을 다시 들어 보세요.",
                )
            )

        additions = sample(AlignmentStatus.ADDITION)
        if additions:
            tokens = "、".join(f"「{entry.answer}」" for entry in additions if entry.answer)
            source_context, answer_context = cls._comparison_context(
                summary.entries, additions[0], learning_language
            )
            improvements.append(
                cls._detailed_improvement(
                    label=f"추가: {tokens}" if tokens else "원문에 없는 표현이 추가되었습니다.",
                    source_context=source_context,
                    answer_context=answer_context,
                    advice="익숙한 표현을 예상해서 들리지 않은 말을 덧붙이지 않았는지 확인해 보세요.",
                )
            )

        order_errors = sample(AlignmentStatus.ORDER)
        if order_errors:
            pairs = "、".join(
                f"「{entry.source}」→「{entry.answer}」"
                for entry in order_errors
                if entry.source or entry.answer
            )
            source_context, answer_context = cls._comparison_context(
                summary.entries, order_errors[0], learning_language
            )
            improvements.append(
                cls._detailed_improvement(
                    label=f"어순: {pairs}" if pairs else "Token 순서가 바뀌었습니다.",
                    source_context=source_context,
                    answer_context=answer_context,
                    advice="각 단어를 따로 맞히기보다 앞뒤 표현을 한 덩어리로 다시 들어 보세요.",
                )
            )

        substitutions = sample(AlignmentStatus.SUBSTITUTION)
        if substitutions:
            pairs = "、".join(
                f"원문 「{entry.source}」 / 내 답변 「{entry.answer}」"
                for entry in substitutions
                if entry.source or entry.answer
            )
            source_context, answer_context = cls._comparison_context(
                summary.entries, substitutions[0], learning_language
            )
            improvements.append(
                cls._detailed_improvement(
                    label=f"다르게 받아씀: {pairs}" if pairs else "다르게 받아쓴 표현이 있습니다.",
                    source_context=source_context,
                    answer_context=answer_context,
                    advice="비슷하게 들린 소리를 원문과 비교하며 해당 구간을 다시 확인해 보세요.",
                )
            )
        return improvements

    @classmethod
    def _comparison_context(
        cls,
        entries,
        target,
        learning_language: str,
        *,
        radius: int = 4,
    ) -> tuple[str, str]:
        try:
            target_index = entries.index(target)
        except ValueError:
            return "", ""
        window = entries[
            max(0, target_index - radius) : min(len(entries), target_index + radius + 1)
        ]
        source_tokens = [entry.source for entry in window if entry.source]
        answer_tokens = [entry.answer for entry in window if entry.answer]
        return (
            cls._join_context_tokens(source_tokens, learning_language),
            cls._join_context_tokens(answer_tokens, learning_language),
        )

    @staticmethod
    def _join_context_tokens(tokens: list[str], learning_language: str) -> str:
        language = learning_language.lower().split("-")[0]
        separator = "" if language in {"ja", "zh"} else " "
        return separator.join(tokens)

    @staticmethod
    def _detailed_improvement(
        *,
        label: str,
        source_context: str,
        answer_context: str,
        advice: str,
    ) -> str:
        comparisons: list[str] = []
        if source_context:
            comparisons.append(f"원문 구간 「{source_context}」")
        if answer_context:
            comparisons.append(f"내 답변 구간 「{answer_context}」")
        if comparisons:
            return f"{label}. {' / '.join(comparisons)}. {advice}"
        return f"{label}. {advice}"
