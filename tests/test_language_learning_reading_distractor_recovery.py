from __future__ import annotations

import asyncio
import json
from collections import defaultdict

import pytest
from fastapi import HTTPException

from app.ai.model_policy import AiModelTier, get_task_model_policy
from app.ai.prompt_registry import get_prompt_rule
from app.core.config import settings
from app.features.language_learning.reading_vocabulary.reading_difficulty_recipe import (
    READING_DIFFICULTY_RECIPE_VERSION,
    READING_DIFFICULTY_SHADOW_RUBRIC_VERSION,
)
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
    _READING_DISTRACTOR_REPAIR_SCHEMA,
    _PracticeVerificationVerdict,
    _ReadingDistractorRepairPayload,
)
from app.schemas.language_learning_practice import PracticeGeneratedQuestion
from tests import test_language_learning_reading_vocabulary as practice_fixtures


@pytest.fixture(autouse=True)
def historical_repair_fake_uses_luna(monkeypatch):
    """The legacy fake registers Luna tasks; production Reading selects Sol separately."""
    monkeypatch.setattr(settings, "AI_READING_GENERATION_MODEL", "LUNA")


def _pass_verdict(order: int, **updates: object) -> dict[str, object]:
    verdict: dict[str, object] = {
        "order": order,
        "bestAnswerKey": "A",
        "ambiguous": False,
        "supported": True,
        "reason": "single supported answer",
        "modeFit": True,
        "answerLeakage": False,
        "contextDependent": True,
        "distractorsPlausible": True,
    }
    verdict.update(updates)
    return verdict


class DistractorRecoveryProvider(practice_fixtures.PipelineProvider):
    def __init__(
        self,
        *,
        verdict_rounds_by_order: dict[int, list[dict[str, object]]] | None = None,
        repair_behavior: str = "valid",
    ) -> None:
        super().__init__()
        self.verdict_rounds_by_order = verdict_rounds_by_order or {}
        self.repair_behavior = repair_behavior
        self.quality_occurrences: defaultdict[int, int] = defaultdict(int)
        self.repair_payloads: list[dict[str, object]] = []
        self.generated_candidates: list[dict[str, object]] = []
        self.repair_started = asyncio.Event()
        self.repair_cleaned_up = asyncio.Event()

    def _candidate(self, payload: dict, slot: dict) -> dict:
        candidate = super()._candidate(payload, slot)
        self.generated_candidates.append(candidate.copy())
        return candidate

    async def call(self, type_name, data, schema=None):
        if type_name == ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME:
            self.calls[type_name] += 1
            payload = json.loads(data.split("\n\n", 1)[1])
            self.verification_payloads.append(payload)
            verdicts = []
            for question in payload["questions"]:
                order = question["order"]
                occurrence = self.quality_occurrences[order]
                self.quality_occurrences[order] += 1
                rounds = self.verdict_rounds_by_order.get(order, [])
                verdict = dict(rounds[occurrence] if occurrence < len(rounds) else _pass_verdict(order))
                if payload["domain"] == "READING":
                    verdict.setdefault("stemPresuppositionsSupported", True)
                    verdict.setdefault("stemEvidenceSpanIds", [next(
                        span["id"] for span in payload["readingEvidenceSpans"]
                        if span["passageId"] == question["passageId"]
                    )])
                    verdict.setdefault(
                        "readingOperation",
                        "DISCOURSE_STRUCTURE" if question["skillTag"] == "STRUCTURE"
                        else "INFERENCE" if question["skillTag"] in {"INFERENCE", "CONTEXT_INFERENCE"}
                        else "DIRECT_RETRIEVAL",
                    )
                    verdict.setdefault("distinctReadingTask", True)
                    if payload["mode"] == "STRUCTURE":
                        verdict.setdefault("boundedStructureScope", True)
                verdicts.append(verdict)
            return {"verdicts": verdicts}

        if type_name == ReadingVocabularyGenerationService.DISTRACTOR_REPAIR_TYPE_NAME:
            self.calls[type_name] += 1
            payload = practice_fixtures._practice_data(data)
            self.repair_payloads.append(payload)
            self.repair_started.set()
            if self.repair_behavior in {"timeout", "cancel"}:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.repair_cleaned_up.set()
            if self.repair_behavior == "provider_error":
                raise RuntimeError("synthetic repair provider failure")
            if self.repair_behavior == "malformed":
                return {"options": []}
            if self.repair_behavior == "duplicate":
                return {
                    "wrongOptions": [
                        {"key": "B", "text": "もっともらしい誤答"},
                        {"key": "C", "text": "もっともらしい誤答"},
                        {"key": "D", "text": "別の誤答"},
                    ]
                }
            if self.repair_behavior == "correct_copy":
                correct_text = payload["correctOption"]["text"]
                return {
                    "wrongOptions": [
                        {"key": "B", "text": correct_text},
                        {"key": "C", "text": "範囲が異なる説明"},
                        {"key": "D", "text": "関係を逆にした説明"},
                    ]
                }
            if self.repair_behavior == "blank":
                return {
                    "wrongOptions": [
                        {"key": "B", "text": " "},
                        {"key": "C", "text": "範囲が異なる説明"},
                        {"key": "D", "text": "関係を逆にした説明"},
                    ]
                }
            if self.repair_behavior == "wrong_keys":
                return {
                    "wrongOptions": [
                        {"key": "B", "text": "もっともらしい誤答"},
                        {"key": "C", "text": "範囲が異なる説明"},
                    ]
                }
            if self.repair_behavior == "wrong_language":
                return {
                    "wrongOptions": [
                        {"key": "B", "text": "범위가 다른 설명"},
                        {"key": "C", "text": "관계를 뒤집은 설명"},
                        {"key": "D", "text": "일부만 맞는 설명"},
                    ]
                }
            return {
                "wrongOptions": [
                    {"key": "B", "text": "本文にはあるが範囲が異なる説明"},
                    {"key": "C", "text": "原因と結果を逆にした説明"},
                    {"key": "D", "text": "一部だけ正しい説明"},
                ]
            }

        return await super().call(type_name, data, schema)


def _one_reading_request(*, mode: str = "COMPREHENSION"):
    return practice_fixtures._request(
        mode=mode,
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
    )


def _weak_only(order: int = 1) -> dict[str, object]:
    return _pass_verdict(
        order,
        reason="wrong options are mechanically easy",
        distractorsPlausible=False,
    )


@pytest.mark.parametrize(
    ("updates", "eligible"),
    [
        ({"distractorsPlausible": False}, True),
        ({"distractorsPlausible": False, "ambiguous": True}, False),
        ({"distractorsPlausible": False, "supported": False}, False),
        ({"distractorsPlausible": False, "modeFit": False}, False),
        ({"distractorsPlausible": False, "answerLeakage": True}, False),
        ({"distractorsPlausible": False, "bestAnswerKey": "B"}, False),
    ],
)
def test_repair_eligibility_requires_distractor_only_failure(updates, eligible):
    verdict = _PracticeVerificationVerdict.model_validate(_pass_verdict(1, **updates))

    assert ReadingVocabularyGenerationService._is_reading_distractor_only_failure(
        verdict,
        expected_answer_key="A",
    ) is eligible


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["COMPREHENSION", "STRUCTURE", "CONTEXT_INFERENCE"])
async def test_distractor_only_failure_repairs_wrong_options_and_preserves_immutables(mode):
    provider = DistractorRecoveryProvider(verdict_rounds_by_order={1: [_weak_only()]})
    service = ReadingVocabularyGenerationService(provider)

    response = await service.generate(_one_reading_request(mode=mode))
    question = response.questions[0]
    generated = provider.generated_candidates[0]
    payload = provider.repair_payloads[0]

    assert provider.calls[service.TYPE_NAME] == 1
    assert provider.calls[service.DISTRACTOR_REPAIR_TYPE_NAME] == 1
    assert provider.calls[service.VERIFICATION_TYPE_NAME] == 2
    assert question.passage_text == generated["passageText"]
    assert question.prompt == generated["prompt"]
    assert question.correct_answer == generated["correctAnswer"]
    assert question.evidence_text == generated["evidenceText"]
    assert question.skill_tag == generated["skillTag"]
    assert question.complexity_band == generated["complexityBand"]
    assert question.difficulty.value == generated["difficulty"]
    assert question.passage_id == generated["passageId"]
    assert question.question_type.value == generated["questionType"]
    assert next(option.text for option in question.options if option.key == "A") == (
        next(option["text"] for option in generated["options"] if option["key"] == "A")
    )
    assert payload["mode"] == mode
    assert payload["difficultyRecipe"]["version"] == READING_DIFFICULTY_RECIPE_VERSION


def _original_question() -> PracticeGeneratedQuestion:
    return PracticeGeneratedQuestion.model_validate(
        {
            "order": 1,
            "questionType": "SINGLE_CHOICE",
            "difficulty": "CURRENT",
            "complexityBand": 3,
            "passageId": "p1",
            "passageText": "本文です。根拠があります。",
            "prompt": "最も適切な内容は何ですか。",
            "options": [
                {"key": "A", "text": "正しい答え"},
                {"key": "B", "text": "以前の誤答一"},
                {"key": "C", "text": "以前の誤答二"},
                {"key": "D", "text": "以前の誤答三"},
            ],
            "correctAnswer": ["A"],
            "skillTag": "CONTENT",
            "evidenceText": "根拠があります。",
            "explanationOrigin": "근거에 맞습니다.",
            "explanationLearning": "根拠に合います。",
            "targetExpression": None,
            "canonicalKey": None,
            "reviewTarget": False,
            "vocabularyCandidates": [],
        }
    )


@pytest.mark.parametrize(
    ("wrong_options", "reason"),
    [
        (
            [
                {"key": "B", "text": "同じ誤答"},
                {"key": "C", "text": "同じ誤答"},
                {"key": "D", "text": "別の誤答"},
            ],
            "duplicate",
        ),
        (
            [
                {"key": "B", "text": "正しい答え"},
                {"key": "C", "text": "別の誤答"},
                {"key": "D", "text": "さらに別の誤答"},
            ],
            "correct answer",
        ),
        (
            [
                {"key": "B", "text": " "},
                {"key": "C", "text": "別の誤答"},
                {"key": "D", "text": "さらに別の誤答"},
            ],
            "blank",
        ),
        (
            [
                {"key": "B", "text": "別の誤答"},
                {"key": "C", "text": "さらに別の誤答"},
            ],
            "keys/cardinality",
        ),
    ],
)
def test_reassembly_rejects_invalid_wrong_options(wrong_options, reason):
    repair = _ReadingDistractorRepairPayload.model_validate(
        {"wrongOptions": wrong_options}
    )

    with pytest.raises(ValueError, match=reason):
        ReadingVocabularyGenerationService._reassemble_reading_distractors(
            _original_question(),
            repair,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repair_behavior",
    [
        "provider_error",
        "malformed",
        "duplicate",
        "correct_copy",
        "blank",
        "wrong_keys",
        "wrong_language",
    ],
)
async def test_invalid_repair_falls_back_to_full_slot_generation(repair_behavior):
    provider = DistractorRecoveryProvider(
        verdict_rounds_by_order={1: [_weak_only()]},
        repair_behavior=repair_behavior,
    )
    service = ReadingVocabularyGenerationService(provider)

    response = await service.generate(_one_reading_request())

    assert len(response.questions) == 1
    assert provider.calls[service.DISTRACTOR_REPAIR_TYPE_NAME] == 1
    assert provider.calls[service.TYPE_NAME] == 2
    assert provider.calls[service.VERIFICATION_TYPE_NAME] == 2


@pytest.mark.asyncio
async def test_repair_timeout_falls_back_to_full_generation_without_extra_mini_budget():
    provider = DistractorRecoveryProvider(
        verdict_rounds_by_order={1: [_weak_only()]},
        repair_behavior="timeout",
    )
    service = ReadingVocabularyGenerationService(provider, timeout_seconds=0.01)

    response = await service.generate(_one_reading_request())

    assert len(response.questions) == 1
    assert provider.repair_cleaned_up.is_set()
    assert provider.calls[service.DISTRACTOR_REPAIR_TYPE_NAME] == 1
    assert provider.calls[service.TYPE_NAME] == 2
    assert provider.calls[service.VERIFICATION_TYPE_NAME] == 2


@pytest.mark.asyncio
async def test_repaired_candidate_weak_again_uses_full_generation_without_repair_loop():
    provider = DistractorRecoveryProvider(
        verdict_rounds_by_order={1: [_weak_only(), _weak_only()]}
    )
    service = ReadingVocabularyGenerationService(provider)

    response = await service.generate(_one_reading_request())

    assert len(response.questions) == 1
    assert provider.calls[service.DISTRACTOR_REPAIR_TYPE_NAME] == 1
    assert provider.calls[service.TYPE_NAME] == 2
    assert provider.calls[service.VERIFICATION_TYPE_NAME] == 3


@pytest.mark.asyncio
async def test_repair_does_not_extend_semantic_exhaustion_budget():
    provider = DistractorRecoveryProvider(
        verdict_rounds_by_order={1: [_weak_only(), _weak_only(), _weak_only()]}
    )
    service = ReadingVocabularyGenerationService(provider)

    with pytest.raises(HTTPException):
        await service.generate(_one_reading_request())

    assert provider.calls[service.DISTRACTOR_REPAIR_TYPE_NAME] == 1
    assert provider.calls[service.TYPE_NAME] == 2
    assert provider.calls[service.VERIFICATION_TYPE_NAME] == 3


@pytest.mark.asyncio
async def test_repair_logs_structured_summary_without_learner_content(caplog):
    provider = DistractorRecoveryProvider(verdict_rounds_by_order={1: [_weak_only()]})

    with caplog.at_level("INFO"):
        await ReadingVocabularyGenerationService(provider).generate(
            _one_reading_request()
        )

    assert "Reading distractor repair attempted" in caplog.text
    assert "Reading distractor repair accepted" in caplog.text
    assert "distractorRepairCalls=1" in caplog.text
    assert "distractorRepairAccepted=1" in caplog.text
    assert "distractorRepairRejected=0" in caplog.text
    assert "今日は会社で会議があります" not in caplog.text
    assert "最も適切な内容" not in caplog.text


@pytest.mark.asyncio
async def test_repair_cancellation_propagates_and_cleans_up_provider_call():
    provider = DistractorRecoveryProvider(
        verdict_rounds_by_order={1: [_weak_only()]},
        repair_behavior="cancel",
    )
    task = asyncio.create_task(
        ReadingVocabularyGenerationService(provider).generate(_one_reading_request())
    )
    await asyncio.wait_for(provider.repair_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.repair_cleaned_up.is_set()
    assert provider.calls[ReadingVocabularyGenerationService.DISTRACTOR_REPAIR_TYPE_NAME] == 1


@pytest.mark.asyncio
async def test_non_distractor_or_multi_issue_failure_skips_repair():
    provider = DistractorRecoveryProvider(
        verdict_rounds_by_order={
            1: [_weak_only() | {"ambiguous": True, "supported": False}]
        }
    )
    service = ReadingVocabularyGenerationService(provider)

    response = await service.generate(_one_reading_request())

    assert len(response.questions) == 1
    assert provider.calls[service.DISTRACTOR_REPAIR_TYPE_NAME] == 0
    assert provider.calls[service.TYPE_NAME] == 2


@pytest.mark.asyncio
async def test_normal_reading_and_vocabulary_paths_add_no_repair_call():
    reading_provider = DistractorRecoveryProvider()
    vocabulary_provider = DistractorRecoveryProvider(
        verdict_rounds_by_order={1: [_weak_only()]}
    )

    reading = await ReadingVocabularyGenerationService(reading_provider).generate(
        _one_reading_request()
    )
    vocabulary = await ReadingVocabularyGenerationService(vocabulary_provider).generate(
        practice_fixtures._request(
            domain="VOCABULARY",
            mode="MEANING_RELATION",
            question_count=1,
            easier=0,
            current=1,
            challenge=0,
        )
    )

    assert len(reading.questions) == len(vocabulary.questions) == 1
    assert reading_provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 1
    assert reading_provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1
    assert reading_provider.calls[ReadingVocabularyGenerationService.DISTRACTOR_REPAIR_TYPE_NAME] == 0
    assert vocabulary_provider.calls[ReadingVocabularyGenerationService.DISTRACTOR_REPAIR_TYPE_NAME] == 0
    assert vocabulary_provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 2


def test_repair_uses_registered_luna_task_without_changing_difficulty_versions():
    task = ReadingVocabularyGenerationService.DISTRACTOR_REPAIR_TYPE_NAME

    assert get_task_model_policy(task).tier == AiModelTier.LUNA
    assert get_prompt_rule(task)
    assert set(_READING_DISTRACTOR_REPAIR_SCHEMA["properties"]) == {"wrongOptions"}
    assert READING_DIFFICULTY_RECIPE_VERSION == "reading-difficulty-recipe-v2"
    assert READING_DIFFICULTY_SHADOW_RUBRIC_VERSION == (
        "reading-difficulty-recipe-v1-shadow"
    )
