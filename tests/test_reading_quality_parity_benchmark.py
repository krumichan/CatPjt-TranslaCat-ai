from __future__ import annotations

import json
from collections import Counter

from app.features.language_learning.reading_vocabulary.prompts import (
    build_practice_verification_prompt,
)
from scripts.run_reading_quality_parity_benchmark import (
    DEFAULT_CORPUS,
    QUALITY_FIELDS,
    QUALITY_ONLY_SCHEMA,
    QUALITY_ONLY_SYSTEM_PROMPT,
    build_batch_inputs,
    load_corpus,
    quality_only_prompt,
    summarize,
)


def test_reading_quality_parity_corpus_has_required_coverage():
    corpus = load_corpus(DEFAULT_CORPUS)
    candidates = [candidate for batch in corpus.batches for candidate in batch.candidates]

    assert len(corpus.batches) == 5
    assert len(candidates) == 15
    assert {batch.requested_band for batch in corpus.batches} == {1, 2, 3, 4, 5}
    assert Counter(candidate.difficulty for candidate in candidates) == Counter(
        {"EASIER": 5, "CURRENT": 5, "CHALLENGE": 5}
    )
    assert {candidate.intended_quality_condition for candidate in candidates} == {
        "GOOD",
        "AMBIGUOUS",
        "WEAK_DISTRACTORS",
        "UNSUPPORTED_ANSWER",
        "ANSWER_LEAKAGE",
        "ANSWER_KEY_MISMATCH",
    }


def test_quality_only_and_combined_inputs_share_candidates_but_keep_task_boundary():
    corpus = load_corpus(DEFAULT_CORPUS)
    batch = corpus.batches[0]
    request, quality_questions, combined_questions = build_batch_inputs(corpus, batch)

    quality_payload = json.loads(
        quality_only_prompt(corpus, batch, quality_questions).split("\n\n", 1)[1]
    )
    combined_payload = json.loads(
        build_practice_verification_prompt(request, combined_questions).split("\n\n", 1)[1]
    )

    assert quality_payload["questions"] == quality_questions
    assert "readingDifficultyRubric" not in quality_payload
    assert "difficulty" not in QUALITY_ONLY_SCHEMA["properties"]["verdicts"]["items"][
        "properties"
    ]
    assert "difficulty classification" not in QUALITY_ONLY_SYSTEM_PROMPT.lower()
    assert combined_payload["readingDifficultyRubric"]["passageBands"][0]["band"] == 1
    assert [
        {
            key: question[key]
            for key in ("order", "passageText", "prompt", "options", "skillTag")
        }
        for question in combined_payload["questions"]
    ] == quality_questions
    assert all("expectedAnswerKey" not in question for question in quality_questions)
    assert all("expectedAnswerKey" not in question for question in combined_questions)


def test_parity_summary_uses_quality_action_and_each_quality_field_threshold():
    quality = {
        "bestAnswerKey": "A",
        "ambiguous": False,
        "supported": True,
        "modeFit": True,
        "answerLeakage": False,
        "distractorsPlausible": True,
        "qualityAction": "ACCEPT",
    }
    comparisons = [
        {
            "intendedQualityCondition": "GOOD",
            "requestedBand": 3,
            "qualityOnly": quality,
            "combined": quality,
            "agreement": {field: True for field in (*QUALITY_FIELDS, "qualityAction")},
            "combinedDifficulty": {
                "difficultyStatus": "ASSESSED",
                "observedBand": 3,
                "alternativeBand": None,
                "difficultyConfidence": 0.8,
            },
        }
    ]
    calls = [
        {
            "condition": condition,
            "latencyMs": 10,
            "inputTokens": 20,
            "outputTokens": 5,
        }
        for condition in ("QUALITY_ONLY", "QUALITY_AND_DIFFICULTY")
    ]
    passages = [
        {
            "requestedBand": 3,
            "difficulty": {
                "difficultyStatus": "ASSESSED",
                "observedBand": 3,
                "alternativeBand": None,
                "difficultyConfidence": 0.7,
            },
        }
    ]

    summary = summarize(comparisons, calls, passages)

    assert summary["verdict"] == "QUALITY-PARITY-PASS"
    assert all(value == 1 for value in summary["agreementRates"].values())
    assert summary["combinedDifficultyObservations"]["questionRequestedVsObserved"] == {
        "3": {"3": 1}
    }
