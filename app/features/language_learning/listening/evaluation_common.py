from __future__ import annotations

from app.features.language_learning.listening.policy import (
    CONFIDENCE_THRESHOLD,
    LISTENING_EVALUATION_VERSION,
    LISTENING_PROFILE_POLICY_VERSION,
    LISTENING_SCORING_POLICY_VERSION,
    build_profile_signals,
    resolve_assistance_level,
)
from app.schemas.language_learning_listening import (
    EvaluationBaseRequest,
    ListeningAssistanceType,
    ListeningEvaluationOverall,
    ListeningEvaluationResponse,
    ListeningTaskResult,
    ListeningTaskStatus,
    ListeningTaskType,
    ListeningUsage,
)


def effective_answer_revealed(request: EvaluationBaseRequest) -> bool:
    return request.answer_revealed or any(
        usage.type == ListeningAssistanceType.SHOW_ANSWER
        for usage in request.assistance_usage
    )


def answer_revealed_task(
    request: EvaluationBaseRequest,
    task_type: ListeningTaskType,
) -> ListeningTaskResult:
    return ListeningTaskResult(
        task_type=task_type,
        status=ListeningTaskStatus.NOT_EVALUABLE,
        evaluable=False,
        score=None,
        confidence=None,
        reason_code="ANSWER_REVEALED",
        assistance_level=resolve_assistance_level(
            request.assistance_usage,
            answer_revealed=True,
        ),
        improvements=["정답 공개 후의 시도는 공식 평가와 Profile 반영에서 제외됩니다."],
        profile_signals=[],
        profile_eligible=False,
        assistance_usage=request.assistance_usage,
    )


def not_evaluable_task(
    request: EvaluationBaseRequest,
    task_type: ListeningTaskType,
    *,
    reason_code: str,
    confidence: float | None = None,
    improvements: list[str] | None = None,
    debug_metadata: dict | None = None,
) -> ListeningTaskResult:
    return ListeningTaskResult(
        task_type=task_type,
        status=ListeningTaskStatus.NOT_EVALUABLE,
        evaluable=False,
        score=None,
        confidence=confidence,
        reason_code=reason_code,
        assistance_level=resolve_assistance_level(
            request.assistance_usage,
            answer_revealed=effective_answer_revealed(request),
        ),
        improvements=improvements or [],
        profile_signals=[],
        profile_eligible=False,
        assistance_usage=request.assistance_usage,
        debug_metadata=debug_metadata or {},
    )


def finalize_task(
    request: EvaluationBaseRequest,
    task: ListeningTaskResult,
) -> ListeningTaskResult:
    confidence = task.confidence or 0.0
    if confidence < CONFIDENCE_THRESHOLD:
        task = task.model_copy(
            deep=True,
            update={
                "status": ListeningTaskStatus.NOT_EVALUABLE,
                "evaluable": False,
                "score": None,
                "reason_code": "LOW_CONFIDENCE",
                "profile_signals": [],
                "profile_eligible": False,
            },
        )
        return task

    signals = build_profile_signals(
        task,
        purpose=request.evaluation_purpose,
        answer_revealed=effective_answer_revealed(request),
    )
    return task.model_copy(
        deep=True,
        update={
            "profile_signals": signals,
            "profile_eligible": bool(signals),
        },
    )


def build_evaluation_response(
    request: EvaluationBaseRequest,
    selected_task: ListeningTaskResult,
    *,
    usage: ListeningUsage | None = None,
) -> ListeningEvaluationResponse:
    tasks: list[ListeningTaskResult] = []
    for task_type in ListeningTaskType:
        if task_type == selected_task.task_type:
            tasks.append(selected_task)
        else:
            tasks.append(
                ListeningTaskResult(
                    task_type=task_type,
                    status=ListeningTaskStatus.NOT_SELECTED,
                    evaluable=False,
                    score=None,
                    confidence=None,
                    assistance_level=resolve_assistance_level(
                        request.assistance_usage,
                        answer_revealed=effective_answer_revealed(request),
                    ),
                    profile_eligible=False,
                    assistance_usage=request.assistance_usage,
                )
            )
    evaluated = [
        task
        for task in tasks
        if task.status == ListeningTaskStatus.EVALUATED and task.score is not None
    ]
    overall_score = (
        int(
            sum(task.score for task in evaluated if task.score is not None)
            / len(evaluated)
            + 0.5
        )
        if evaluated
        else None
    )
    return ListeningEvaluationResponse(
        request_id=request.request_id,
        item_id=request.item_id,
        attempt_id=request.attempt_id,
        evaluation_version=LISTENING_EVALUATION_VERSION,
        scoring_policy_version=LISTENING_SCORING_POLICY_VERSION,
        profile_policy_version=LISTENING_PROFILE_POLICY_VERSION,
        tasks=tasks,
        overall=ListeningEvaluationOverall(
            score=overall_score,
            evaluated_task_count=len(evaluated),
            total_task_count=3,
        ),
        usage=usage or ListeningUsage(),
    )
