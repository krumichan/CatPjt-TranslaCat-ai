from __future__ import annotations

from collections.abc import Iterable

from app.core.config import settings
from app.schemas.language_learning_listening import (
    AssistanceUsage,
    ComprehensionMetricType,
    DictationMetricType,
    EvaluationPurpose,
    InterpretationMetricType,
    ListeningAssistanceLevel,
    ListeningAssistanceType,
    ListeningMetric,
    ListeningProfileSignal,
    ListeningTaskResult,
    ListeningTaskType,
    ProfileMetric,
    RepeatMetricType,
    SummaryMetricType,
)

LISTENING_GENERATION_VERSION = "listening-generation"
LISTENING_GENERATION_PROMPT_VERSION = "listening-generation-prompt"
LISTENING_TTS_VERSION = "listening-tts"
LISTENING_NORMALIZATION_VERSION = "listening-normalization"
LISTENING_ALIGNMENT_VERSION = "listening-alignment"
LISTENING_EVALUATION_VERSION = "listening-evaluation"
LISTENING_SCORING_POLICY_VERSION = "listening-scoring-half-up"
LISTENING_PROFILE_POLICY_VERSION = "listening-profile"
LISTENING_INTERPRETATION_PROMPT_VERSION = "listening-interpretation-prompt"
LISTENING_SUMMARY_PROMPT_VERSION = "listening-summary-prompt"
LISTENING_REPEAT_EVALUATOR_VERSION = "listening-repeat-acoustic"
LISTENING_EXPLANATION_VERSION = "listening-explanation"
LISTENING_EXPLANATION_PROMPT_VERSION = "listening-explanation-prompt"
LISTENING_AUDIO_NORMALIZATION_VERSION = "listening-audio-normalization"
LISTENING_STT_HINT_VERSION = "listening-stt-hint"

CONFIDENCE_THRESHOLD = settings.AI_LISTENING_EVALUATION_CONFIDENCE_THRESHOLD

DIFFICULTY_DURATION_RANGES: dict[str, tuple[float, float]] = {
    "EASY": (5.0, 12.0),
    "MY_LEVEL": (8.0, 20.0),
    "CHALLENGE": (15.0, 30.0),
}

DICTATION_WEIGHTS: dict[str, float] = {
    DictationMetricType.TOKEN_RECOGNITION.value: 0.60,
    DictationMetricType.OMISSION_ADDITION_ORDER.value: 0.25,
    DictationMetricType.ORTHOGRAPHY.value: 0.15,
}

INTERPRETATION_WEIGHTS: dict[str, float] = {
    InterpretationMetricType.MEANING_FIDELITY.value: 0.60,
    InterpretationMetricType.DETAIL_AND_NUANCE.value: 0.25,
    InterpretationMetricType.ORIGIN_NATURALNESS.value: 0.15,
}

COMPREHENSION_WEIGHTS: dict[str, float] = {
    ComprehensionMetricType.ANSWER_ACCURACY.value: 1.00,
}

SUMMARY_WEIGHTS: dict[str, float] = {
    SummaryMetricType.GIST_COVERAGE.value: 0.55,
    SummaryMetricType.KEY_POINT_COVERAGE.value: 0.30,
    SummaryMetricType.LANGUAGE_CLARITY.value: 0.15,
}

REPEAT_WEIGHTS: dict[str, float] = {
    RepeatMetricType.PRONUNCIATION.value: 0.40,
    RepeatMetricType.PROSODY_RHYTHM.value: 0.25,
    RepeatMetricType.FLUENCY.value: 0.20,
    RepeatMetricType.COMPLETENESS.value: 0.15,
}

PROFILE_SIGNAL_WEIGHTS: dict[ListeningTaskType, dict[ProfileMetric, float]] = {
    ListeningTaskType.DICTATION: {
        ProfileMetric.LISTENING_RECOGNITION: 1.00,
        ProfileMetric.VOCABULARY: 0.60,
        ProfileMetric.ORTHOGRAPHY: 0.80,
    },
    ListeningTaskType.INTERPRETATION: {
        ProfileMetric.MEANING: 1.00,
        ProfileMetric.ORIGIN_NATURALNESS: 0.30,
    },
    ListeningTaskType.COMPREHENSION: {
        ProfileMetric.MEANING: 1.00,
    },
    ListeningTaskType.SUMMARY: {
        ProfileMetric.MEANING: 1.00,
    },
    ListeningTaskType.REPEAT_AFTER_AUDIO: {
        ProfileMetric.PRONUNCIATION: 1.00,
        ProfileMetric.FLUENCY: 0.80,
        ProfileMetric.LISTENING_RECOGNITION: 0.50,
    },
}


def round_score(value: float) -> int:
    return max(0, min(100, int(value + 0.5)))


def calculate_weighted_score(
    metrics: list[ListeningMetric],
    expected_weights: dict[str, float],
) -> int | None:
    validate_metric_set(metrics, expected_weights)
    if any(metric.score is None for metric in metrics):
        return None
    weighted_sum = 0.0
    for metric in metrics:
        assert metric.score is not None
        weighted_sum += metric.score * expected_weights[metric.type]
    return round_score(weighted_sum)


def validate_metric_set(
    metrics: list[ListeningMetric],
    expected_weights: dict[str, float],
) -> None:
    metric_types = [metric.type for metric in metrics]
    if len(metric_types) != len(set(metric_types)):
        raise ValueError("Listening 평가 Metric type이 중복되었습니다.")
    if set(metric_types) != set(expected_weights):
        raise ValueError(
            "Listening 평가 Metric 구성이 유효하지 않습니다: "
            f"expected={sorted(expected_weights)} actual={sorted(metric_types)}"
        )
    for metric in metrics:
        if abs(metric.weight - expected_weights[metric.type]) > 1e-9:
            raise ValueError(f"{metric.type} Metric weight가 정책과 다릅니다.")
    if abs(sum(expected_weights.values()) - 1.0) > 1e-9:
        raise ValueError("Listening Metric weight 합은 1.0이어야 합니다.")


def resolve_assistance_level(
    usages: Iterable[AssistanceUsage],
    *,
    answer_revealed: bool = False,
) -> ListeningAssistanceLevel:
    types = {usage.type for usage in usages}
    if answer_revealed or ListeningAssistanceType.SHOW_ANSWER in types:
        return ListeningAssistanceLevel.GUIDED
    if (
        ListeningAssistanceType.TOPIC_HINT in types
        or ListeningAssistanceType.KEYWORD_HINT in types
    ):
        return ListeningAssistanceLevel.ASSISTED
    return ListeningAssistanceLevel.INDEPENDENT


def can_emit_profile_signals(
    *,
    purpose: EvaluationPurpose,
    answer_revealed: bool,
    confidence: float,
    task: ListeningTaskResult,
) -> bool:
    return (
        purpose == EvaluationPurpose.OFFICIAL
        and not answer_revealed
        and task.evaluable
        and task.score is not None
        and confidence >= CONFIDENCE_THRESHOLD
    )


def build_profile_signals(
    task: ListeningTaskResult,
    *,
    purpose: EvaluationPurpose,
    answer_revealed: bool,
) -> list[ListeningProfileSignal]:
    confidence = task.confidence or 0.0
    if not can_emit_profile_signals(
        purpose=purpose,
        answer_revealed=answer_revealed,
        confidence=confidence,
        task=task,
    ):
        return []

    scores = _profile_scores(task)
    evidence_ids = [f"evidence-{index + 1}" for index, _ in enumerate(task.evidence)]
    return [
        ListeningProfileSignal(
            metric=metric,
            score=scores[metric],
            confidence=confidence,
            evidence_weight=weight,
            evidence_ids=evidence_ids,
            source_task=task.task_type,
            policy_version=LISTENING_PROFILE_POLICY_VERSION,
        )
        for metric, weight in PROFILE_SIGNAL_WEIGHTS[task.task_type].items()
    ]


def _profile_scores(task: ListeningTaskResult) -> dict[ProfileMetric, float]:
    metric_scores = {
        metric.type: float(metric.score)
        for metric in task.metrics
        if metric.score is not None
    }
    overall = float(task.score or 0)
    if task.task_type == ListeningTaskType.DICTATION:
        return {
            ProfileMetric.LISTENING_RECOGNITION: overall,
            ProfileMetric.VOCABULARY: metric_scores[
                DictationMetricType.TOKEN_RECOGNITION.value
            ],
            ProfileMetric.ORTHOGRAPHY: metric_scores[
                DictationMetricType.ORTHOGRAPHY.value
            ],
        }
    if task.task_type == ListeningTaskType.INTERPRETATION:
        return {
            ProfileMetric.MEANING: metric_scores[
                InterpretationMetricType.MEANING_FIDELITY.value
            ],
            ProfileMetric.ORIGIN_NATURALNESS: metric_scores[
                InterpretationMetricType.ORIGIN_NATURALNESS.value
            ],
        }
    if task.task_type == ListeningTaskType.COMPREHENSION:
        return {ProfileMetric.MEANING: overall}
    if task.task_type == ListeningTaskType.SUMMARY:
        return {ProfileMetric.MEANING: overall}
    return {
        ProfileMetric.PRONUNCIATION: metric_scores[
            RepeatMetricType.PRONUNCIATION.value
        ],
        ProfileMetric.FLUENCY: metric_scores[RepeatMetricType.FLUENCY.value],
        ProfileMetric.LISTENING_RECOGNITION: overall,
    }
