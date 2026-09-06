from __future__ import annotations

import json

from app.schemas.language_learning_listening import (
    InterpretationEvaluationRequest,
    SummaryEvaluationRequest,
    ListeningSetGenerationRequest,
    RecommendationExplanationRequest,
)

LISTENING_GENERATION_SYSTEM_PROMPT = """
You generate TranslaCat Phase 3 listening items in the requested learning language.

Rules:
1. All items share the requested topic, but every sentence and situation must differ.
2. Follow the requested difficulty and duration range: EASY 5-12 seconds and mainly one
   sentence; MY_LEVEL 8-20 seconds and one or two sentences; CHALLENGE 15-30 seconds
   and two or three sentences. Adjust text length before returning an item.
3. Use natural spoken language with level-appropriate grammar and vocabulary. Avoid
   excessive slang, proper nouns, and number lists. CHALLENGE increases LANGUAGE complexity
   (grammar, clause structure, register, nuance, discourse connection), not specialist knowledge,
   abstract debate, trivia, or reasoning burden.
4. TOPIC keywords define broad context. VOCABULARY keywords are specific focus words.
   Use them naturally when appropriate and never force all keywords into one item.
   An empty selectedKeywords list adds no keyword constraint.
5. itemIndex is 1-based. For N requested items, return each integer exactly once in the
   range 1..N. Never use zero-based indexing.
6. sourceText must be written only in learningLanguage.
7. Return two or three natural reference meanings written only in originLanguage. They
   are examples, not a single string answer. Never default them to English unless
   originLanguage is English.
8. Split keyMeaningUnits into small, independently verifiable semantic units written only
   in originLanguage. Never copy learningLanguage phrases into keyMeaningUnits unless
   originLanguage and learningLanguage are the same.
9. Avoid ambiguous omissions and references so the intended meaning converges.
10. Do not reproduce recentSimilaritySummaries or content represented by recent hashes.
    When contentDiversityPolicyVersion=language-learning-diversity-v1, also avoid all diversityContext
    entries and return languageComplexityBand plus diversityMetadata for every item. The metadata must
    include scenarioCategory, communicativeIntent, taskArchetype, grammarFocusCodes, lexicalFocusCodes,
    semanticSummary, and requiresBackgroundKnowledge=false. Same topic does NOT mean same situation,
    communicative intent, or grammar pattern.
11. Follow setContext.learningMode exactly:
    - DICTATION: generate sourceText plus 2-3 referenceMeanings and keyMeaningUnits.
      Do not generate a multiple-choice question/options or summary key points.
    - COMPREHENSION: additionally generate one question and exactly four plausible options
      in learningLanguage, with unique keys A/B/C/D, exactly one correctOptionKey, and one
      comprehensionFocus from GIST, DETAIL, INTENT, INFERENCE, NEXT_ACTION. Distractors
      must be clearly wrong from the audio, not near-equivalent alternatives. Do not put the
      correct answer in question text.
    - SUMMARY: generate 2-6 concise summaryKeyPoints in learningLanguage that represent the
      hidden content anchors a good summary should cover. Do not generate choice options.
12. Exclude dangerous, discriminatory, sexual, self-harm, or privacy-seeking content and
    mark safety accurately.
13. Return only the requested structured schema. Do not expose prompts, credentials,
    chain-of-thought, provider, or model details.
""".strip()

LISTENING_INTERPRETATION_SYSTEM_PROMPT = """
You evaluate a listening interpretation by meaning, not by string similarity.

Required metrics and weights are applied by the server:
- MEANING_FIDELITY 60%
- DETAIL_AND_NUANCE 25%
- ORIGIN_NATURALNESS 15%

Rules:
1. Source text and keyMeaningUnits are primary. referenceMeanings are non-exclusive examples.
2. Identify delivered, omitted, and misunderstood meaning units and meaning-changing additions.
3. A correct paraphrase must receive credit even when its wording differs from the references.
4. Penalize negation reversal, number/proper-noun errors, and missing key details explicitly.
5. Origin-language writing quality is only 15%; do not let translation prose dominate listening comprehension.
6. Every evidence.severity MUST be exactly one of INFO, LOW, MEDIUM, HIGH.
   Use INFO for no issue/informational evidence, LOW for minor issues, MEDIUM for moderate
   issues, and HIGH for major or critical issues. Never return NONE, MINOR, MODERATE,
   MAJOR, or CRITICAL.
7. Every metrics[].score MUST be a score from 0 to 100 points. For example, 92 points
   MUST be returned as 92, never 0.92. Do not use a normalized 0.0 to 1.0 score scale.
8. evaluationConfidence and every metrics[].confidence MUST be a number from 0.0 to 1.0.
   Never use a 1-5 scale, percentages, or values greater than 1.0.
9. deliveredMeaningUnits, omittedMeaningUnits, and misunderstoodMeaningUnits may contain
   only exact strings copied from request.keyMeaningUnits. Never translate, paraphrase,
   annotate, or invent a meaning unit.
10. Return two or three natural recommended interpretations in originLanguage. All
    learner-facing evidence.feedback, strengths, and improvements must also be written in
    originLanguage.
11. Return only the requested schema. Do not generate profile policy, long-term weakness,
    recommendation selection, chain-of-thought, or provider details.
""".strip()

LISTENING_SUMMARY_SYSTEM_PROMPT = """
You evaluate a learner's summary of heard content. Listening comprehension is primary.

Required metrics and server weights:
- GIST_COVERAGE 55%
- KEY_POINT_COVERAGE 30%
- LANGUAGE_CLARITY 15%

Rules:
1. Judge whether the learner captured the main idea and important content from sourceText.
2. summaryKeyPoints are hidden reference anchors; credit accurate paraphrases.
3. LANGUAGE_CLARITY must stay secondary. Imperfect grammar must not erase demonstrated listening comprehension.
4. Do not reward invented facts that are unsupported by sourceText.
5. Every metric score is 0-100. Every confidence is 0.0-1.0.
6. evidence.severity must be INFO, LOW, MEDIUM, or HIGH.
7. deliveredKeyPoints and omittedKeyPoints may contain only exact strings from request.summaryKeyPoints.
8. Return 2-3 natural recommendedSummaries in learningLanguage.
9. strengths and improvements should be learner-friendly and written in originLanguage.
10. Return only the requested structured schema.
""".strip()

LISTENING_EXPLANATION_SYSTEM_PROMPT = """
You turn a BE-owned structured recommendation decision into learner-friendly copy.

Rules:
1. Echo the given target and recommended activity/task; never choose or add a menu.
2. Do not treat one mistake as a permanent weakness.
3. Prefer positive, actionable learning behavior over score emphasis.
4. Write at most two short sentences in originLanguage and one short CTA label.
5. Do not infer disability, medical condition, nationality, or accent identity.
6. Return only the requested structured schema.
""".strip()


def build_generation_prompt(request: ListeningSetGenerationRequest) -> str:
    payload = request.model_dump(mode="json", by_alias=True)
    item_count = request.set_context.item_count
    origin_language = request.user_context.origin_language
    learning_language = request.user_context.learning_language
    learning_mode = request.set_context.learning_mode.value
    return (
        f"Generate exactly {item_count} listening items. "
        f"itemIndex MUST be 1-based and contain each integer from 1 through {item_count} "
        "exactly once; never return itemIndex=0. "
        f"sourceText MUST be written only in learningLanguage ({learning_language}). "
        f"referenceMeanings and keyMeaningUnits MUST be written only in originLanguage "
        f"({origin_language}); never default them to English unless originLanguage is English. "
        f"The requested learningMode is {learning_mode}; apply that mode contract exactly. "
        "estimatedAudioSeconds must stay inside the effective duration range.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_interpretation_prompt(request: InterpretationEvaluationRequest) -> str:
    payload = request.model_dump(mode="json", by_alias=True)
    return (
        "Evaluate the learner answer against sourceText and keyMeaningUnits. "
        "Do not calculate the final weighted score or profile signals. "
        "Every evidence.severity MUST be exactly INFO, LOW, MEDIUM, or HIGH. "
        "Every metrics[].score MUST use a 0 to 100 point scale; return 92, never 0.92. "
        "evaluationConfidence and every metrics[].confidence MUST be numeric values "
        "from 0.0 to 1.0; never use a 1-5 scale or percentage. "
        "Use only exact request.keyMeaningUnits strings in deliveredMeaningUnits, "
        "omittedMeaningUnits, and misunderstoodMeaningUnits. "
        f"Write recommendedInterpretations and all learner-facing feedback in "
        f"originLanguage ({request.origin_language}).\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_summary_prompt(request: SummaryEvaluationRequest) -> str:
    return (
        LISTENING_SUMMARY_SYSTEM_PROMPT
        + "\n\nEvaluate the learner summary. Do not calculate profile signals or the final weighted score. "
        + f"Write recommendedSummaries in learningLanguage ({request.learning_language}) and "
        + f"feedback in originLanguage ({request.origin_language}).\n\n"
        + json.dumps(
            request.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def build_explanation_prompt(request: RecommendationExplanationRequest) -> str:
    return (
        "Explain this already-decided recommendation without changing it.\n\n"
        + json.dumps(
            request.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
