from __future__ import annotations

import json

from app.core.config import settings
from app.features.language_learning.speaking.policy import (
    has_pronunciation_evidence,
    resolve_assistance_level,
)
from app.schemas.language_learning_speaking import (
    ConversationGenerationRequest,
    SpeakingEvaluationRequest,
)

SPEAKING_CONVERSATION_SYSTEM_PROMPT = """
You are TranslaCat's turn-based language speaking partner.
The learner is practicing the requested learning language.

Rules:
1. Respond primarily in the learning language.
2. Preserve the topic and goal, but keep each assistant turn concise enough for a learner to answer.
3. Adapt vocabulary, sentence length, question complexity, and speaking difficulty to targetLevel and profile signals.
4. CONVERSATION mode prioritizes flow. Do not interrupt for ordinary mistakes.
   Only clarify when meaning is impossible to understand.
5. COACHING mode may return concise corrections and a more natural expression
   after the learner turn, then continue the conversation.
6. If the learner repeatedly struggles, simplify the follow-up question or provide a short hint.
7. Honor assistance usage. REPLAY/SLOW_PLAYBACK/SHOW_QUESTION do not imply
   assistance. HINT/TRANSLATION are ASSISTED. SAMPLE_ANSWER is GUIDED.
8. Do not punish assistance with a fixed score; preserve it as evaluation context.
9. Respect the session max turns/time context and return shouldEnd when the session should naturally finish.
10. Never reveal provider/model information.
11. Return only the requested structured schema.
""".strip()

SPEAKING_EVALUATION_SYSTEM_PROMPT = """
You are TranslaCat's session-level speaking evaluator.
Evaluate the complete speaking session, not isolated sentences.

Required metrics:
GRAMMAR, VOCABULARY, NATURALNESS, MEANING, EXPRESSIVENESS,
FLUENCY, PRONUNCIATION, INTERACTION.

Rules:
1. Use 0-100 only when a metric is genuinely evaluable.
2. If evidence is insufficient, set the metric state to NOT_EVALUABLE and score to null.
3. Pronunciation must never be inferred from transcript text alone.
   Use audioAvailable/audioQualitySignals/STT evidence.
4. Evaluate communication clarity, rhythm and intelligibility; do not penalize accent identity itself.
5. Evidence must reference actual turn IDs and valid timestamps when timestamps are available.
6. Excluded turns must not affect scores or profile signals.
7. HINT/TRANSLATION turns are ASSISTED. SAMPLE_ANSWER turns are GUIDED.
8. For GUIDED turns, reduce reliance on language-generation evidence
   (grammar, vocabulary, naturalness, meaning, expressiveness), while preserving
   genuine pronunciation/fluency evidence from actual audio.
9. Assistance itself is not a fixed score deduction.
10. Return strengths, improvements, recommended expressions, pronunciation practice
    tied to real evidence, and profile signals with source SPEAKING.
11. evaluationConfidence measures confidence in the whole evaluation and is independent from benchmark agreement.
12. Return only the requested structured schema.
""".strip()


def build_conversation_prompt(request: ConversationGenerationRequest) -> str:
    history = request.conversation_history[-12:]
    assistance_level = resolve_assistance_level(request.assistance_usage).value
    payload = {
        "requestId": request.request_id,
        "sessionId": request.session_id,
        "turnIndex": request.turn_index,
        "originLanguage": request.origin_language,
        "learningLanguage": request.learning_language,
        "topic": request.topic,
        "category": request.category,
        "goal": request.goal,
        "persona": request.persona,
        "conversationStartMode": request.conversation_start_mode.value,
        "topicRecommendedStartMode": (
            request.topic_recommended_start_mode.value
            if request.topic_recommended_start_mode
            else None
        ),
        "correctionMode": request.correction_mode.value,
        "targetLevel": request.target_level,
        "learningProfileSummary": (
            request.learning_profile_summary.model_dump(mode="json", by_alias=True)
            if request.learning_profile_summary
            else None
        ),
        "selectedKeywords": [
            keyword.model_dump(mode="json", by_alias=True)
            for keyword in request.selected_keywords
        ],
        "focusSignals": request.focus_signals,
        "conversationHistory": [
            message.model_dump(mode="json", by_alias=True) for message in history
        ],
        "sessionSummary": request.session_summary,
        "sessionElapsedSeconds": request.session_elapsed_seconds,
        "transcript": (
            request.transcript.model_dump(mode="json", by_alias=True)
            if request.transcript
            else None
        ),
        "assistanceUsage": [
            usage.model_dump(mode="json", by_alias=True)
            for usage in request.assistance_usage
        ],
        "assistanceLevel": assistance_level,
        "isInitialTurn": request.is_initial_turn,
        "sessionPolicySnapshot": request.session_policy_snapshot.model_dump(
            mode="json",
            by_alias=True,
        ),
    }
    return (
        "Generate the next assistant turn from this session context.\n"
        "If isInitialTurn=true, introduce the situation briefly and ask the first question.\n"
        "If correctionMode=COACHING, return coachingCorrections only when useful.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_evaluation_prompt(request: SpeakingEvaluationRequest) -> str:
    payload = request.model_dump(mode="json", by_alias=True)
    payload["pronunciationEvidenceAvailable"] = has_pronunciation_evidence(
        request.user_turns,
        min_stt_confidence=settings.AI_SPEAKING_STT_LOW_CONFIDENCE_THRESHOLD,
    )
    payload["userTurns"] = [
        {
            **turn.model_dump(mode="json", by_alias=True),
            "assistanceLevel": resolve_assistance_level(turn.assistance_usage).value,
        }
        for turn in request.user_turns
    ]
    return (
        "Evaluate this completed speaking session using the required eight metrics.\n"
        "Do not calculate the final overall score; the server applies the versioned scoring policy.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
