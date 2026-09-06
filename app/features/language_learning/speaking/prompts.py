from __future__ import annotations

import json

from app.core.config import settings
from app.features.language_learning.speaking.policy import (
    has_pronunciation_evidence,
    resolve_assistance_level,
)
from app.schemas.language_learning_speaking import (
    AssistanceRequest,
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
12. Follow practiceMode exactly:
    - READ_ALOUD: each assistant script is one item in the learner's daily repeat set. Generate one
      short self-contained learning-language sentence to repeat. assistantText and scriptText must
      be identical. Do not ask an open question and leave providedFacts/requiredIntents/
      responseConstraints empty. Keep pronunciation load appropriate to targetLevel and normally
      under 90 characters where the language permits. Make the new script meaningfully different
      from earlier READ_ALOUD scripts in conversationHistory while staying on the resolved topic.
    - GUIDED: generate a concrete speaking task. providedFacts, requiredIntents, and
      responseConstraints must all be non-empty and learner-visible. The learner must be able to
      answer using only the supplied guidance plus ordinary personal expression; never require
      unknown real-world facts. scriptText must be null.
    - FREE: ask for an opinion, experience, plan, preference, or explicitly imaginary situation.
      Guidance should be minimal or empty; do not require the learner to invent facts about a real
      incident they cannot know. scriptText must be null.
13. Interpret selectedKeywords by type:
    - TOPIC defines the broad conversation context and does not need to appear literally.
    - VOCABULARY defines a specific learning focus and should be used naturally when appropriate.
    - When both types are present, preserve the TOPIC context while emphasizing VOCABULARY.
    - SYSTEM and CUSTOM have equal priority; source is metadata only.
    - An empty selectedKeywords list adds no keyword constraint.
    Never force every selected keyword into one turn.
14. When category is KEYWORDS, the topic field is only a keyword seed, not a fixed final topic.
    - If isInitialTurn=true, choose one coherent, concrete session topic or situation from selectedKeywords
      that fits targetLevel and practiceMode. Return that concise learning-language title in resolvedTopic.
    - Prefer a natural combination of one TOPIC keyword and at most a few useful VOCABULARY keywords.
    - Do not create an unnatural scenario just to include every keyword.
    - On later turns, preserve the already resolved topic and resolvedTopic may be null.
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
12. Apply practiceMode when interpreting MEANING:
    - READ_ALOUD: MEANING means script accuracy/completeness against the preceding assistant script.
      Grammar/vocabulary/naturalness/interaction may be NOT_EVALUABLE when they would only judge copied text;
      pronunciation and fluency are primary.
    - GUIDED: MEANING means fulfillment of providedFacts, requiredIntents, and responseConstraints.
    - FREE: MEANING means on-topic task fulfillment and clarity of the learner's own message.
13. Return only the requested structured schema.
""".strip()


def build_conversation_prompt(request: ConversationGenerationRequest) -> str:
    history = request.conversation_history[-12:]
    assistance_level = resolve_assistance_level(request.assistance_usage).value
    payload = {
        "requestId": request.request_id,
        "sessionId": request.session_id,
        "turnIndex": request.turn_index,
        "problemIndex": request.problem_index,
        "attemptIndex": request.attempt_index,
        "readAloudGenerateNextProblem": request.read_aloud_generate_next_problem,
        "originLanguage": request.origin_language,
        "learningLanguage": request.learning_language,
        "topic": request.topic,
        "practiceMode": request.practice_mode.value,
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
        "If isInitialTurn=true, generate the first task/prompt for the selected practiceMode.\n"
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
        "Evaluate this speaking evidence using the required eight metrics. "
        "When evaluationScope=READ_ALOUD_PROBLEM, compare the repeated attempts for the same script "
        "and prioritize pronunciation, fluency, and script accuracy consistency.\n"
        "Do not calculate the final overall score; the server applies the versioned scoring policy.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


SPEAKING_ASSISTANCE_SYSTEM_PROMPT = """
You are TranslaCat's language speaking assistance generator.
Return exactly one requested assistance item and do not advance the conversation.

Rules by assistanceType:
- HINT: write one concise hint primarily in originLanguage. Do not reveal a complete answer.
  You may include 1-3 useful learningLanguage words or short phrases.
- TRANSLATION: translate assistantText into originLanguage faithfully. Do not add commentary.
- SAMPLE_ANSWER: write one natural learner answer in learningLanguage, usually 1-2 sentences.
  Match the learner level and topic. Do not include explanations.

Keyword rules:
- TOPIC supplies broad context; VOCABULARY supplies a specific learning focus.
- When both are present, keep the TOPIC context and use VOCABULARY naturally when useful.
- SYSTEM and CUSTOM have equal priority, and an empty selectedKeywords list adds no constraint.

Do not change the question, do not simulate a new assistant turn, and return only the requested structured schema.
""".strip()


def build_assistance_prompt(request: AssistanceRequest) -> str:
    history = request.conversation_history[-8:]
    payload = {
        "requestId": request.request_id,
        "sessionId": request.session_id,
        "turnIndex": request.turn_index,
        "assistanceType": request.assistance_type.value,
        "originLanguage": request.origin_language,
        "learningLanguage": request.learning_language,
        "topic": request.topic,
        "targetLevel": request.target_level,
        "assistantText": request.assistant_text,
        "conversationHistory": [
            message.model_dump(mode="json", by_alias=True) for message in history
        ],
        "selectedKeywords": [
            keyword.model_dump(mode="json", by_alias=True)
            for keyword in request.selected_keywords
        ],
        "sessionSummary": request.session_summary,
    }
    return (
        SPEAKING_ASSISTANCE_SYSTEM_PROMPT
        + "\n\nGenerate the requested assistance.\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
