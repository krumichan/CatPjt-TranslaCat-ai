"""Server-owned capabilities of the current text-only Speaking evaluator.

Audio availability is not audio consumption. STT confidence is a decoder signal,
not a calibrated probability that the learner uttered the recognized sentence.
"""
from __future__ import annotations

from app.core.config import settings
from app.schemas.language_learning_speaking import (
    SpeakingEvaluationRequest,
    SpeakingEvaluationTurn,
    SpeakingMetricType,
    SpeakingPracticeMode,
)

SPEAKING_EVIDENCE_POLICY_VERSION = "speaking-transcript-evidence-v2"
TRANSCRIPT_EVIDENCE_SOURCE = "TRANSCRIPT_OBSERVATION"


def transcript_is_usable(turn: SpeakingEvaluationTurn) -> bool:
    threshold = settings.AI_SPEAKING_STT_LOW_CONFIDENCE_THRESHOLD
    return (
        not turn.excluded_from_evaluation
        and bool(turn.transcript.strip())
        and turn.stt_confidence >= threshold
        and all(segment.confidence >= threshold for segment in turn.segments)
    )


def unsupported_metrics(request: SpeakingEvaluationRequest) -> dict[SpeakingMetricType, str]:
    unavailable = {
        SpeakingMetricType.PRONUNCIATION: "TEXT_EVALUATOR_HAS_NO_ACOUSTIC_EVIDENCE",
        SpeakingMetricType.FLUENCY: "NO_VERIFIED_TEMPORAL_FLUENCY_MEASUREMENTS",
    }
    if request.practice_mode == SpeakingPracticeMode.READ_ALOUD:
        unavailable.update({
            metric: "READ_ALOUD_COPIED_TEXT_IS_NOT_SPONTANEOUS_LANGUAGE_EVIDENCE"
            for metric in SpeakingMetricType
            if metric != SpeakingMetricType.MEANING and metric not in unavailable
        })
    return unavailable


def evaluation_capabilities(request: SpeakingEvaluationRequest) -> dict:
    unavailable = unsupported_metrics(request)
    return {
        "policyVersion": SPEAKING_EVIDENCE_POLICY_VERSION,
        "source": TRANSCRIPT_EVIDENCE_SOURCE,
        "acousticEvidenceConsumed": False,
        "verifiedTemporalFluencyEvidenceConsumed": False,
        "audioReferencesAreNotConsumedByEvaluator": True,
        "pronunciationPracticeAllowed": False,
        "unsupportedMetrics": {metric.value: reason for metric, reason in unavailable.items()},
        "modelAssessableMetrics": [
            metric.value for metric in SpeakingMetricType if metric not in unavailable
        ],
        "evaluationConfidenceScope": "MODEL_ASSESSABLE_METRICS_AND_USABLE_TEXT_EVIDENCE",
        "textEvidenceTurnIds": [turn.turn_id for turn in request.user_turns if transcript_is_usable(turn)],
        "uncertainTranscriptTurnIds": [
            turn.turn_id for turn in request.user_turns
            if not turn.excluded_from_evaluation and not transcript_is_usable(turn)
        ],
        "meaningInterpretation": (
            "STT_OBSERVED_SCRIPT_ACCURACY_NOT_PRONUNCIATION"
            if request.practice_mode == SpeakingPracticeMode.READ_ALOUD
            else "OBSERVED_TASK_FULFILLMENT"
        ),
        "profileSignalsRequireIndependentTextEvidence": True,
        "readAloudProfileSignalsAllowed": False,
    }
