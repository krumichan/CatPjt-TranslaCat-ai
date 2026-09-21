from __future__ import annotations

import json

from app.features.language_learning.speaking.evidence_policy import (
    evaluation_capabilities,
    transcript_is_usable,
)
from app.features.language_learning.speaking.policy import (
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
   Corrections concern recognized text, not proof of what the learner actually said.
   When STT is uncertain, clarify instead of declaring a learner error. improvementLink
   is an optional legacy field: return null unless a real supported linkage was supplied;
   never invent a URL, identifier, blank placeholder or prose to satisfy that field.
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

Return only the metrics named in evaluationCapabilities.modelAssessableMetrics.
The server adds its own NOT_EVALUABLE entries for unsupported metrics after
your response; do not return those entries yourself.

Rules:
1. Use 0-100 only when a metric is genuinely evaluable.
2. If evidence is insufficient, set the metric state to NOT_EVALUABLE and score to null.
3. This evaluator receives TRANSCRIPT_OBSERVATION only, not waveform or acoustic analysis.
   Audio references, RMS, silence ratio, STT confidence and approximate segments do NOT
   establish pronunciation, accent, phonemes, intonation, rhythm, pauses or speaking speed.
   Never score or return evaluationCapabilities.unsupportedMetrics.
   PRONUNCIATION and FLUENCY have no supported scorer here.
   pronunciationPractice must be []; never emit acoustic profile signals or acoustic claims
   in strengths, improvements, recommendedExpressions or metric summaries.
4. STT text is an observation, not a verified verbatim account or proof of learner error.
   Do not silently correct or substitute transcript text. State lexical/grammar observations
   conditionally as 'in the recognized transcript', not proven pronunciation or learner deficits.
   Uncertain transcript turns may provide context, but cannot justify evaluated metric evidence,
   penalties, corrective recommendations or durable profile signals. Decoder confidence is not
   calibrated accuracy and can remain high for incorrect recognition.
5. Evidence and all evidenceTurnIds must reference only evidenceContract.allowedUserTurnIds.
   Assistant turns are reference context, never learner evidence IDs. Follow the
   server-owned per-user-turn timestamp bounds in evidenceContract. STT segment
   timing is approximate: timestampUsableForEvidence=false means its times were
   withheld because they contradict measured audio bounds. Keep its transcript
   content, but use null timestamps when no trustworthy timing supports a claim;
   never invent, round up, or copy an unavailable timestamp.
6. Excluded turns must not affect scores or profile signals.
7. HINT/TRANSLATION turns are ASSISTED. SAMPLE_ANSWER turns are GUIDED.
8. For GUIDED turns, reduce reliance on language-generation evidence
   (grammar, vocabulary, naturalness, meaning, expressiveness), while preserving
   only claims supported by the current evaluationCapabilities.
9. Assistance itself is not a fixed score deduction.
10. Return strengths, improvements and recommended expressions tied to usable text evidence.
    Profile signals require at least two distinct usable learner turns and an evaluated supported
    metric; do not infer a lasting weakness from one uncertain recognition or from copied script.
    Return profileSignals=[] for READ_ALOUD. Never fabricate unsupported acoustic practice.
11. evaluationConfidence measures confidence in the assessment of the requested
    modelAssessableMetrics as a whole, using only usable transcript evidence and
    its actual task references. It is NOT confidence in unavailable acoustic
    metrics, a percentage of all possible Speaking axes, or evaluationCoverage.
    Preserve uncertainty caused by STT errors, weak evidence or incomplete task
    responses; do not derive or inflate confidence from turn count or duration.
12. Apply practiceMode when interpreting MEANING:
    - READ_ALOUD: MEANING means script accuracy/completeness against the preceding assistant script.
      Evaluate MEANING only as observed script accuracy, explicitly noting transcription uncertainty.
      Only return MEANING. Copied script is not spontaneous vocabulary/grammar ability,
      and no acoustic scorer is present. This is not a pronunciation test score.
    - GUIDED: MEANING means fulfillment of providedFacts, requiredIntents, and responseConstraints.
    - FREE: MEANING means on-topic task fulfillment and clarity of the learner's own message.
13. Do not use all-NOT_EVALUABLE as a shortcut when usable task/text evidence exists. Assess the
    supported content axes honestly, including low task fulfillment for unrelated but clear speech.
    Return only the requested structured schema.
""".strip()


SPEAKING_SESSION_COACHING_SYSTEM_PROMPT = """
You are TranslaCat's evidence-grounded FREE speaking session coach.
Return at most three useful coaching items. Every item must cite one exact,
contiguous excerpt from an eligible learner transcript and its supplied turnId.

Separate these meanings:
- OBSERVATION: a concrete, evidenced communication strength or pattern.
- CORRECTION: a specific, understandable grammar, vocabulary, or expression issue.
- ALTERNATIVE: a context-sensitive alternative without claiming the learner was wrong.

Write message in originLanguage. Write suggestedExpression, when present, in
learningLanguage. Never invent pronunciation, accent, rhythm, acoustic fluency,
scores, confidence, ability bands, or learner speech. ASR is an observation and
may be imperfect; do not turn suspected recognition errors into learner faults.
Do not cite assistant turns, hints, sample answers, metadata, or a suggested
expression as learner evidence. Avoid generic praise and placeholder coaching.
If evidence is usable but narrow, use LIMITED with an explicit reason. Use
NO_USABLE_EVIDENCE only when no eligible learner transcript can support a result.
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


def evaluation_evidence_bounds(request: SpeakingEvaluationRequest) -> dict[str, int]:
    """Existing measured-duration tolerance, shared by input and final validation."""
    return {
        turn.turn_id: int(turn.duration_seconds * 1000) + 250
        for turn in request.user_turns
        if not turn.excluded_from_evaluation
    }


def build_evaluation_prompt(request: SpeakingEvaluationRequest) -> str:
    payload = request.model_dump(mode="json", by_alias=True)
    # The BE snapshot includes the assistant reply after the last learner turn.
    # It has no learner answer and must not be mistaken for that turn's task.
    last_user_index = max(turn.turn_index for turn in request.user_turns)
    task_references = {
        turn.turn_index: turn
        for turn in request.assistant_turns
        if turn.turn_index < last_user_index
    }
    payload["assistantTurns"] = [
        turn.model_dump(mode="json", by_alias=True)
        for turn in request.assistant_turns
        if turn.turn_index < last_user_index
    ]
    payload["evaluationTaskBindings"] = [
        {
            "userTurnId": turn.turn_id,
            "referenceAssistantTurnId": (
                task_references[turn.turn_index - 1].turn_id
                if turn.turn_index - 1 in task_references else None
            ),
        }
        for turn in request.user_turns
    ]
    bounds = evaluation_evidence_bounds(request)
    payload["evidenceContract"] = {
        "allowedUserTurnIds": list(bounds),
        "timestampBoundsByUserTurn": {
            turn_id: {"minMs": 0, "maxMs": maximum}
            for turn_id, maximum in bounds.items()
        },
        "assistantTurnsAreReferenceContextOnly": True,
        "unsupportedTimestampsMustBeNull": True,
    }
    payload["pronunciationEvidenceAvailable"] = False
    payload["evaluationCapabilities"] = evaluation_capabilities(request)
    payload["userTurns"] = [
        {
            **turn.model_dump(mode="json", by_alias=True),
            "assistanceLevel": resolve_assistance_level(turn.assistance_usage).value,
            "transcriptObservation": {
                "source": "AUTOMATIC_SPEECH_RECOGNITION",
                "verbatimAccuracyVerified": False,
                "usableForTextEvaluation": transcript_is_usable(turn),
                "confidenceIsCalibratedAccuracy": False,
            },
        }
        for turn in request.user_turns
    ]
    # Only the provider-facing copy changes. Preserve the original STT evidence
    # in the request/DB; do not clip or fabricate replacement timing anchors.
    for turn in payload["userTurns"]:
        maximum = bounds.get(turn["turnId"])
        for segment in turn["segments"]:
            start, end = segment["startMs"], segment["endMs"]
            if maximum is None or not (0 <= start <= end <= maximum):
                segment.pop("startMs")
                segment.pop("endMs")
                segment["timestampUsableForEvidence"] = False
    return (
        "Evaluate this speaking evidence using only evaluationCapabilities.modelAssessableMetrics. "
        "For task fulfillment, pair each userTurn with evaluationTaskBindings.referenceAssistantTurnId; "
        "an assistant reply after the last learner turn is not a task the learner answered. "
        "When evaluationScope=READ_ALOUD_PROBLEM, compare the repeated attempts for the same script "
        "and assess only STT-observed script accuracy, not pronunciation or spontaneous grammar.\n"
        "Do not calculate the final overall score; the server applies the server scoring policy.\n\n"
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
