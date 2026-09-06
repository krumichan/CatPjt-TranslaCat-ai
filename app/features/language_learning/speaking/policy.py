from __future__ import annotations

from collections.abc import Iterable

from app.schemas.language_learning_speaking import (
    AssistanceLevel,
    AssistanceType,
    AssistanceUsage,
    EvaluationEligibility,
    MetricEvaluationState,
    SpeakingEvaluationTurn,
    SpeakingMetricPayload,
    SpeakingMetricType,
)

SPEAKING_SCORING_POLICY_VERSION = "speaking-scoring-policy-v1"
SPEAKING_EVALUATION_VERSION = "speaking-evaluation-v1"
SPEAKING_EVALUATION_PROMPT_VERSION = "speaking-evaluation-prompt-v1"
SPEAKING_CONVERSATION_PROMPT_VERSION = "speaking-conversation-v3"
SPEAKING_TTS_VERSION = "speaking-tts-v1"
AUDIO_NORMALIZATION_VERSION = "speaking-audio-normalization-v1"
STT_HINT_VERSION = "speaking-stt-hint-v1"

SPEAKING_METRIC_WEIGHTS: dict[SpeakingMetricType, float] = {
    SpeakingMetricType.GRAMMAR: 0.10,
    SpeakingMetricType.VOCABULARY: 0.10,
    SpeakingMetricType.NATURALNESS: 0.10,
    SpeakingMetricType.MEANING: 0.10,
    SpeakingMetricType.EXPRESSIVENESS: 0.10,
    SpeakingMetricType.FLUENCY: 0.20,
    SpeakingMetricType.PRONUNCIATION: 0.15,
    SpeakingMetricType.INTERACTION: 0.15,
}


def resolve_assistance_level(usages: Iterable[AssistanceUsage]) -> AssistanceLevel:
    types = {usage.type for usage in usages}
    if AssistanceType.SAMPLE_ANSWER in types:
        return AssistanceLevel.GUIDED
    if AssistanceType.HINT in types or AssistanceType.TRANSLATION in types:
        return AssistanceLevel.ASSISTED
    return AssistanceLevel.NONE


def calculate_evaluation_eligibility(
    turns: list[SpeakingEvaluationTurn],
    *,
    min_stt_confidence: float = 0.55,
    required_user_turns: int = 5,
    required_speech_seconds: float = 60.0,
    required_stt_turn_ratio: float = 0.80,
) -> EvaluationEligibility:
    submitted_turns = list(turns)
    evaluation_turns = [turn for turn in submitted_turns if not turn.excluded_from_evaluation]
    valid_turns = [turn for turn in evaluation_turns if turn.transcript.strip()]
    valid_user_turns = len(valid_turns)
    valid_user_speech_seconds = sum(
        turn.duration_seconds
        for turn in evaluation_turns
        if turn.duration_seconds > 0
    )
    valid_stt_turns = sum(
        bool(turn.transcript.strip()) and turn.stt_confidence >= min_stt_confidence
        for turn in evaluation_turns
    )
    valid_stt_turn_ratio = (
        valid_stt_turns / len(evaluation_turns) if evaluation_turns else 0.0
    )

    missing: list[str] = []
    if valid_user_turns < required_user_turns:
        missing.append("VALID_USER_TURNS")
    if valid_user_speech_seconds < required_speech_seconds:
        missing.append("VALID_SPEECH_SECONDS")
    if valid_stt_turn_ratio < required_stt_turn_ratio:
        missing.append("VALID_STT_TURN_RATIO")

    return EvaluationEligibility(
        valid_user_turns=valid_user_turns,
        valid_user_speech_seconds=round(valid_user_speech_seconds, 3),
        valid_stt_turn_ratio=round(valid_stt_turn_ratio, 4),
        required_user_turns=required_user_turns,
        required_speech_seconds=required_speech_seconds,
        required_stt_turn_ratio=required_stt_turn_ratio,
        eligible_before_ai=not missing,
        missing_requirements=missing,
    )



def has_pronunciation_evidence(
    turns: list[SpeakingEvaluationTurn],
    *,
    min_stt_confidence: float = 0.55,
) -> bool:
    for turn in turns:
        if turn.excluded_from_evaluation or not turn.audio_available:
            continue
        quality = turn.audio_quality_signals
        if quality is None or not turn.segments:
            continue
        if turn.duration_seconds < 1.0 or turn.stt_confidence < min_stt_confidence:
            continue
        if quality.rms < 0.003 or quality.silence_ratio >= 0.98:
            continue
        return True
    return False

def calculate_speaking_overall(metrics: list[SpeakingMetricPayload]) -> int | None:
    weighted_sum = 0.0
    available_weight = 0.0

    for metric in metrics:
        if metric.state != MetricEvaluationState.EVALUATED or metric.score is None:
            continue
        weight = SPEAKING_METRIC_WEIGHTS[metric.type]
        weighted_sum += metric.score * weight
        available_weight += weight

    if available_weight <= 0:
        return None

    normalized = weighted_sum / available_weight
    return max(0, min(100, int(normalized + 0.5)))


def validate_metric_set(metrics: list[SpeakingMetricPayload]) -> None:
    expected = set(SPEAKING_METRIC_WEIGHTS)
    actual = [metric.type for metric in metrics]
    if len(actual) != len(set(actual)):
        raise ValueError("Speaking evaluation metric type이 중복되었습니다.")
    if set(actual) != expected:
        missing = sorted(metric.value for metric in expected - set(actual))
        extra = sorted(metric.value for metric in set(actual) - expected)
        raise ValueError(
            f"Speaking evaluation 8축이 유효하지 않습니다. missing={missing} extra={extra}"
        )
