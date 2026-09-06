from __future__ import annotations

from app.common.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.listening.evaluation_common import (
    answer_revealed_task,
    build_evaluation_response,
    effective_answer_revealed,
    finalize_task,
)
from app.features.language_learning.listening.policy import COMPREHENSION_WEIGHTS, resolve_assistance_level
from app.schemas.language_learning_listening import (
    ComprehensionEvaluationRequest,
    ComprehensionMetricType,
    ListeningAssistanceLevel,
    ListeningEvaluationResponse,
    ListeningMetric,
    ListeningTaskResult,
    ListeningTaskStatus,
    ListeningTaskType,
    MetricEvidence,
)


class ListeningComprehensionService:
    """Evaluate the objective comprehension choice deterministically."""

    def __init__(
        self,
        idempotency_store: InMemoryIdempotencyStore[ListeningEvaluationResponse] | None = None,
    ) -> None:
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def evaluate(
        self,
        request: ComprehensionEvaluationRequest,
    ) -> ListeningEvaluationResponse:
        key = "|".join(
            [
                request.idempotency_key,
                request.evaluation_purpose.value,
                request.policy_version,
                request.model_config_version,
                "COMPREHENSION",
            ]
        )
        response, _ = await self.idempotency_store.execute(
            key,
            lambda: self._evaluate_once(request),
        )
        return response.model_copy(deep=True, update={"request_id": request.request_id})

    async def _evaluate_once(
        self,
        request: ComprehensionEvaluationRequest,
    ) -> ListeningEvaluationResponse:
        if effective_answer_revealed(request):
            return build_evaluation_response(
                request,
                answer_revealed_task(request, ListeningTaskType.COMPREHENSION),
            )

        correct = request.selected_option_key == request.correct_option_key
        correct_option = next(
            option for option in request.options if option.key == request.correct_option_key
        )
        score = 100 if correct else 0
        feedback, strength, improvement = self._feedback(
            request.origin_language, correct, correct_option.key
        )
        metric = ListeningMetric(
            type=ComprehensionMetricType.ANSWER_ACCURACY.value,
            score=score,
            weight=COMPREHENSION_WEIGHTS[ComprehensionMetricType.ANSWER_ACCURACY.value],
            confidence=1.0,
            evidence=[
                MetricEvidence(
                    metric=ComprehensionMetricType.ANSWER_ACCURACY.value,
                    severity="INFO" if correct else "HIGH",
                    reference=correct_option.key,
                    recognized=request.selected_option_key,
                    feedback=feedback,
                )
            ],
        )
        task = ListeningTaskResult(
            task_type=ListeningTaskType.COMPREHENSION,
            status=ListeningTaskStatus.EVALUATED,
            evaluable=True,
            score=score,
            confidence=1.0,
            assistance_level=resolve_assistance_level(
                request.assistance_usage, answer_revealed=False
            ),
            metrics=[metric],
            evidence=metric.evidence,
            strengths=[strength] if correct else [],
            improvements=[] if correct else [improvement],
            recommended_interpretations=[correct_option.text],
            assistance_usage=request.assistance_usage,
        )
        return build_evaluation_response(request, finalize_task(request, task))

    @staticmethod
    def _feedback(origin_language: str, correct: bool, correct_key: str) -> tuple[str, str, str]:
        language = (origin_language or "").lower()
        if language.startswith("ja"):
            return (
                "正解を選べました。" if correct else f"正解は{correct_key}です。音声の重要な手掛かりをもう一度確認しましょう。",
                "音声の中心内容を正確に判断できました。",
                "選択肢を見る前に、音声の中心となる行動や意図を一文で整理してみましょう。",
            )
        if language.startswith("ko"):
            return (
                "정답을 정확하게 골랐습니다." if correct else f"정답은 {correct_key}입니다. 오디오의 핵심 단서를 다시 확인해 보세요.",
                "오디오의 핵심 내용을 정확히 판단했습니다.",
                "선택지보다 먼저 오디오의 핵심 행동·의도를 한 문장으로 정리해 보세요.",
            )
        return (
            "You selected the correct answer." if correct else f"The correct answer is {correct_key}. Review the key clue in the audio.",
            "You identified the main content of the audio accurately.",
            "Before checking the options, summarize the main action or intent in one sentence.",
        )
