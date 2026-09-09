import json
import re
from collections import Counter, defaultdict

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.features.language_learning.reading_vocabulary.service import ReadingVocabularyGenerationService
from app.schemas.language_learning_practice import PracticeGeneratedQuestion, PracticeGenerationRequest


def _practice_data(prompt: str) -> dict:
    match = re.search(r"<practice-data>\n(.+?)\n</practice-data>", prompt, re.S)
    if not match:
        return {}
    return json.loads(match.group(1))


class PipelineProvider:
    def __init__(
        self,
        *,
        semantic_ambiguous_once: set[int] | None = None,
        bad_learning_lane_once: set[int] | None = None,
        nano_bad_language_calls: int = 0,
        duplicate_new_canonical_once: set[int] | None = None,
        usage_target_leak_once: set[int] | None = None,
        prescreen_reject_once: set[int] | None = None,
        prescreen_reject_always: set[int] | None = None,
        semantic_reject_rounds_by_order: dict[int, int] | None = None,
        candidate_reject_rounds_by_order: dict[int, int] | None = None,
        usage_metadata_drift_once: set[int] | None = None,
        composition_invalid_ordering_once: set[int] | None = None,
        composition_missing_target_once: set[int] | None = None,
        reading_invalid_vocabulary_candidate_rounds_by_order: dict[int, int] | None = None,
        english_target_once: set[int] | None = None,
        english_options_once: set[int] | None = None,
    ):
        self.calls = Counter()
        self.candidate_slot_calls: list[list[int]] = []
        self.candidate_slot_payloads: list[list[dict]] = []
        self.slot_occurrences = defaultdict(int)
        self.verification_round = 0
        self.verification_occurrences = defaultdict(int)
        self.semantic_ambiguous_once = set(semantic_ambiguous_once or set())
        self.bad_learning_lane_once = set(bad_learning_lane_once or set())
        self.nano_bad_language_calls = nano_bad_language_calls
        self.duplicate_new_canonical_once = set(duplicate_new_canonical_once or set())
        self.usage_target_leak_once = set(usage_target_leak_once or set())
        self.prescreen_reject_once = set(prescreen_reject_once or set())
        self.prescreen_reject_always = set(prescreen_reject_always or set())
        self.semantic_reject_rounds_by_order = dict(semantic_reject_rounds_by_order or {})
        self.candidate_reject_rounds_by_order = dict(candidate_reject_rounds_by_order or {})
        self.usage_metadata_drift_once = set(usage_metadata_drift_once or set())
        self.composition_invalid_ordering_once = set(composition_invalid_ordering_once or set())
        self.composition_missing_target_once = set(composition_missing_target_once or set())
        self.reading_invalid_vocabulary_candidate_rounds_by_order = dict(
            reading_invalid_vocabulary_candidate_rounds_by_order or {}
        )
        self.english_target_once = set(english_target_once or set())
        self.english_options_once = set(english_options_once or set())
        self.prescreen_round = 0
        self.prescreen_payloads: list[dict] = []
        self.verification_payloads: list[dict] = []

    async def call(self, type_name, data, schema=None):
        self.calls[type_name] += 1
        if type_name == ReadingVocabularyGenerationService.PASSAGE_TYPE_NAME:
            payload = _practice_data(data)
            passage_id = payload["passageId"]
            return {
                "passageId": passage_id,
                "passageText": (
                    "今日は会社で会議があります。担当者は資料を確認しました。"
                    "その後、顧客への説明方法について話し合いました。"
                ),
            }

        if type_name == ReadingVocabularyGenerationService.TYPE_NAME:
            payload = _practice_data(data)
            slots = payload["candidateSlots"]
            self.candidate_slot_calls.append([slot["order"] for slot in slots])
            self.candidate_slot_payloads.append(slots)
            return {"questions": [self._candidate(payload, slot) for slot in slots]}

        if type_name == ReadingVocabularyGenerationService.PRESCREEN_TYPE_NAME:
            self.prescreen_round += 1
            payload = json.loads(data.split("\n\n", 1)[1])
            self.prescreen_payloads.append(payload)
            verdicts = []
            for question in payload["questions"]:
                order = question["order"]
                reject = (
                    order in self.prescreen_reject_always
                    or (self.prescreen_round == 1 and order in self.prescreen_reject_once)
                )
                verdicts.append(
                    {
                        "order": order,
                        "modeFit": not reject,
                        "answerLeakage": reject,
                        "contextDependent": not reject,
                        "reason": "target repeated in stem" if reject else "contextual usage choice",
                    }
                )
            return {"verdicts": verdicts}

        if type_name == ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME:
            self.verification_round += 1
            payload = json.loads(data.split("\n\n", 1)[1])
            self.verification_payloads.append(payload)
            verdicts = []
            for question in payload["questions"]:
                order = question["order"]
                self.verification_occurrences[order] += 1
                occurrence = self.verification_occurrences[order]
                ambiguous = (
                    (occurrence == 1 and order in self.semantic_ambiguous_once)
                    or occurrence <= self.semantic_reject_rounds_by_order.get(order, 0)
                )
                verdicts.append(
                    {
                        "order": order,
                        "bestAnswerKey": "A",
                        "ambiguous": ambiguous,
                        "supported": True,
                        "reason": "rival option plausible" if ambiguous else "single supported answer",
                        "modeFit": True,
                        "answerLeakage": False,
                        "contextDependent": True,
                        "distractorsPlausible": True,
                    }
                )
            return {"verdicts": verdicts}

        if type_name in {
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME,
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME,
        }:
            payload = _practice_data(data)
            nano = type_name == ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME
            bad_language = nano and self.calls[type_name] <= self.nano_bad_language_calls
            return {
                "explanations": [
                    {
                        "order": question["order"],
                        "text": (
                            "この説明は日本語のままです。"
                            if bad_language
                            else "이 답은 제시된 문맥과 근거에 가장 잘 맞습니다."
                        ),
                    }
                    for question in payload["questions"]
                ]
            }

        raise AssertionError(f"unexpected type_name={type_name}")

    def _candidate(self, payload: dict, slot: dict) -> dict:
        order = slot["order"]
        self.slot_occurrences[order] += 1
        domain = payload["domain"]
        vocab = domain == "VOCABULARY"
        question_type = slot.get("questionType", "SINGLE_CHOICE")

        if question_type == "ORDERING":
            options = [
                {"key": "A", "text": "対応を"},
                {"key": "B", "text": "早急に"},
                {"key": "C", "text": "検討する"},
                {"key": "D", "text": "必要があります"},
            ]
            correct = ["A", "B", "C", "D"]
        else:
            options = [
                {"key": "A", "text": "最も適切な内容"},
                {"key": "B", "text": "別の内容"},
                {"key": "C", "text": "関係のない内容"},
                {"key": "D", "text": "反対の内容"},
            ]
            correct = ["A"]

        prompt = "最も適切なものはどれですか。"
        if (
            (order in self.bad_learning_lane_once and self.slot_occurrences[order] == 1)
            or self.slot_occurrences[order] <= self.candidate_reject_rounds_by_order.get(order, 0)
        ):
            prompt = "가장 적절한 것은 무엇입니까?"

        bound = slot.get("boundReviewTarget")
        if vocab and bound:
            canonical_key = bound["canonicalKey"]
            target_expression = bound["expression"]
            review_target = True
        elif vocab:
            if order in self.duplicate_new_canonical_once and self.slot_occurrences[order] == 1:
                canonical_key = "duplicate-key"
            else:
                canonical_key = f"new-key-{order}-{self.slot_occurrences[order]}"
            target_expression = f"新しい表現{order}"
            review_target = False
        else:
            canonical_key = None
            target_expression = None
            review_target = False

        if (
            vocab
            and order in self.english_target_once
            and self.slot_occurrences[order] == 1
        ):
            target_expression = "deployment"
            canonical_key = f"english-target-{order}"

        if (
            vocab
            and order in self.english_options_once
            and self.slot_occurrences[order] == 1
            and question_type == "SINGLE_CHOICE"
        ):
            options = [
                {"key": "A", "text": "business"},
                {"key": "B", "text": "customer service"},
                {"key": "C", "text": "schedule"},
                {"key": "D", "text": "development"},
            ]

        if (
            vocab
            and payload.get("mode") == "USAGE_DISTINCTION"
            and question_type == "SINGLE_CHOICE"
        ):
            retry_target = slot.get("retryTargetExpression")
            if retry_target:
                target_expression = retry_target
            options[0] = {"key": "A", "text": target_expression}

        skill_tag = slot["skillTag"]
        if (
            vocab
            and payload.get("mode") == "USAGE_DISTINCTION"
            and order in self.usage_metadata_drift_once
            and self.slot_occurrences[order] == 1
        ):
            skill_tag = "MEANING"
            canonical_key = "review-a"
            review_target = True
            correct = ["B"]

        if (
            vocab
            and payload.get("mode") == "USAGE_DISTINCTION"
            and order in self.usage_target_leak_once
            and self.slot_occurrences[order] == 1
        ):
            prompt = f"「{target_expression}」と同じ意味で最も自然な表現はどれですか。"

        if (
            vocab
            and payload.get("mode") == "COMPOSITION"
            and question_type == "ORDERING"
            and order in self.composition_invalid_ordering_once
            and self.slot_occurrences[order] == 1
        ):
            correct = ["A", "B", "B", "D"]

        if (
            vocab
            and payload.get("mode") == "COMPOSITION"
            and question_type == "ORDERING"
            and order in self.composition_missing_target_once
            and self.slot_occurrences[order] >= 2
        ):
            target_expression = None
            canonical_key = None

        vocabulary_candidates = ["資料"] if not vocab else []
        if (
            not vocab
            and self.slot_occurrences[order]
            <= self.reading_invalid_vocabulary_candidate_rounds_by_order.get(order, 0)
        ):
            vocabulary_candidates = ["資料", "確認する", "", "資料", "顧客への説明"]

        return {
            "order": order,
            "questionType": question_type,
            "difficulty": slot["difficulty"],
            "complexityBand": slot["complexityBand"],
            "passageId": slot.get("passageId") if not vocab else None,
            "passageText": slot.get("passageText") if not vocab else None,
            "prompt": prompt,
            "options": options,
            "correctAnswer": correct,
            "skillTag": skill_tag,
            "evidenceText": "担当者は資料を確認しました。" if not vocab else None,
            "explanationLearning": "この答えが文脈に最も合います。",
            "targetExpression": target_expression,
            "canonicalKey": canonical_key,
            "reviewTarget": review_target,
            "vocabularyCandidates": vocabulary_candidates,
        }


def _request(
    domain="READING",
    mode="COMPREHENSION",
    question_count=5,
    easier=1,
    current=3,
    challenge=1,
    *,
    review_targets=None,
    review_question_count=0,
):
    return PracticeGenerationRequest.model_validate(
        {
            "requestId": "r1",
            "domain": domain,
            "mode": mode,
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "questionCount": question_count,
            "complexityBand": 3,
            "easierCount": easier,
            "currentCount": current,
            "challengeCount": challenge,
            "reviewTargets": review_targets or [],
            "reviewQuestionCount": review_question_count,
            "generationDate": "2026-09-07",
        }
    )


def _vocab_request(**kwargs):
    return _request(
        domain="VOCABULARY",
        mode=kwargs.pop("mode", "MEANING_RELATION"),
        question_count=10,
        easier=kwargs.pop("easier", 2),
        current=kwargs.pop("current", 6),
        challenge=kwargs.pop("challenge", 2),
        **kwargs,
    )


class ProgressiveProvider(PipelineProvider):
    """Use distinct vocabulary for successive calls while returning local order 1."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generation_payloads = []

    def _candidate(self, payload, slot):
        self.generation_payloads.append(payload)
        global_order = len(payload["previousQuestions"]) + slot["order"]
        candidate = super()._candidate(payload, {**slot, "order": global_order})
        candidate["order"] = slot["order"]
        return candidate


def _single_request(base, previous=(), **updates):
    payload = base.model_dump(mode="json", by_alias=True)
    payload.update(
        requestId=f"progressive-{len(previous) + 1}",
        questionCount=1,
        easierCount=0,
        currentCount=1,
        challengeCount=0,
        previousQuestions=[question.model_dump(mode="json", by_alias=True) for question in previous],
    )
    payload.update(updates)
    return PracticeGenerationRequest.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["COMPREHENSION", "STRUCTURE", "CONTEXT_INFERENCE"])
async def test_progressive_reading_reuses_passages_and_continues_global_skill_plan(mode):
    provider = ProgressiveProvider()
    service = ReadingVocabularyGenerationService(provider)
    base = _request(mode=mode)
    expected = service._build_slots(base, {"p1": "仮の本文です。", "p2": "別の本文です。"})
    previous = []

    for global_order in range(1, 6):
        response = await service.generate(_single_request(base, previous))
        assert len(response.questions) == 1
        question = response.questions[0]
        assert question.order == 1
        assert question.passage_id == expected[global_order - 1].passage_id
        assert question.skill_tag == expected[global_order - 1].skill_tag
        assert provider.calls[service.PASSAGE_TYPE_NAME] == (1 if global_order <= 3 else 2)
        previous.append(question.model_copy(update={"order": global_order}))

    assert provider.calls[service.TYPE_NAME] == 5
    assert provider.calls[service.VERIFICATION_TYPE_NAME] == 5
    assert provider.calls[service.ORIGIN_EXPLANATION_TYPE_NAME] == 5
    assert previous[0].passage_text == previous[2].passage_text
    assert previous[3].passage_text == previous[4].passage_text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["MEANING_RELATION", "USAGE_DISTINCTION", "COMPOSITION"])
async def test_progressive_vocabulary_preserves_skill_cycle_ordering_and_uniqueness(mode):
    provider = ProgressiveProvider()
    service = ReadingVocabularyGenerationService(provider)
    base = _vocab_request(mode=mode)
    expected = service._build_slots(base, {})
    previous = []

    for global_order in range(1, 11):
        response = await service.generate(_single_request(base, previous))
        assert len(response.questions) == 1
        question = response.questions[0]
        assert question.order == 1
        assert question.question_type == expected[global_order - 1].question_type
        assert question.skill_tag == expected[global_order - 1].skill_tag
        payload = provider.generation_payloads[-1]
        assert {q.canonical_key for q in previous} <= set(payload["excludedCanonicalKeys"])
        assert {q.target_expression for q in previous} <= set(payload["excludedTargetExpressions"])
        previous.append(question.model_copy(update={"order": global_order}))

    assert len({q.canonical_key for q in previous}) == 10
    assert len({q.target_expression for q in previous}) == 10
    assert provider.calls[service.TYPE_NAME] == 10
    assert provider.calls[service.ORIGIN_EXPLANATION_TYPE_NAME] == 10
    for payload in provider.verification_payloads + provider.prescreen_payloads:
        assert "previousQuestions" not in payload
        for question in payload["questions"]:
            assert "targetExpression" not in question
            assert "correctAnswer" not in question


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate_field", ["canonicalKey", "targetExpression"])
async def test_progressive_vocabulary_rejects_identity_from_previous_request(duplicate_field):
    base = _vocab_request()
    first = (await ReadingVocabularyGenerationService(ProgressiveProvider()).generate(
        _single_request(base)
    )).questions[0]

    class DuplicatePreviousProvider(ProgressiveProvider):
        def _candidate(self, payload, slot):
            candidate = super()._candidate(payload, slot)
            if len(self.generation_payloads) == 1:
                candidate[duplicate_field] = first.model_dump(by_alias=True)[duplicate_field]
            return candidate

    provider = DuplicatePreviousProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _single_request(base, [first])
    )
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 2
    assert response.questions[0].canonical_key != first.canonical_key
    assert response.questions[0].target_expression != first.target_expression


@pytest.mark.asyncio
async def test_progressive_reading_failed_slot_can_resume_without_regenerating_previous_passage():
    base = _request()
    service = ReadingVocabularyGenerationService(ProgressiveProvider())
    previous = []
    for order in (1, 2):
        question = (await service.generate(_single_request(base, previous))).questions[0]
        previous.append(question.model_copy(update={"order": order}))
    snapshot = [question.model_dump() for question in previous]
    failing = ProgressiveProvider(candidate_reject_rounds_by_order={3: 20})
    with pytest.raises(HTTPException) as error:
        await ReadingVocabularyGenerationService(failing).generate(_single_request(base, previous))
    assert error.value.status_code == 502
    assert [question.model_dump() for question in previous] == snapshot
    assert failing.calls[service.PASSAGE_TYPE_NAME] == 0

    resumed = ProgressiveProvider()
    response = await ReadingVocabularyGenerationService(resumed).generate(
        _single_request(base, previous)
    )
    assert response.questions[0].passage_text == previous[0].passage_text
    assert resumed.calls[service.PASSAGE_TYPE_NAME] == 0
    assert resumed.calls[service.TYPE_NAME] == 1


@pytest.mark.asyncio
async def test_progressive_vocabulary_review_slot_retains_bound_identity():
    base = _vocab_request(mode="USAGE_DISTINCTION")
    service = ReadingVocabularyGenerationService(ProgressiveProvider())
    previous = (await service.generate(_single_request(base))).questions
    request = _single_request(
        base,
        previous,
        reviewTargets=[{"canonicalKey": "ja-確認", "expression": "確認"}],
        reviewQuestionCount=1,
    )
    question = (await service.generate(request)).questions[0]
    assert question.order == 1
    assert question.skill_tag == "COLLOCATION"
    assert question.review_target
    assert question.canonical_key == "ja-確認"
    assert question.target_expression == "確認"


@pytest.mark.asyncio
async def test_progressive_review_planning_skips_committed_and_invalid_review_targets():
    base = _vocab_request(mode="USAGE_DISTINCTION")
    service = ReadingVocabularyGenerationService(ProgressiveProvider())
    first = (await service.generate(_single_request(
        base,
        reviewTargets=[{"canonicalKey": "ja-確認", "expression": "確認"}],
        reviewQuestionCount=1,
    ))).questions[0]
    request = _single_request(
        base,
        [first],
        reviewTargets=[
            {"canonicalKey": "deployment", "expression": "deployment"},
            {"canonicalKey": "ja-確認", "expression": "確認"},
            {"canonicalKey": "ja-連絡", "expression": "連絡"},
        ],
        reviewQuestionCount=1,
    )
    question = (await service.generate(request)).questions[0]
    assert question.review_target
    assert question.canonical_key == "ja-連絡"
    assert question.target_expression == "連絡"


@pytest.mark.asyncio
async def test_progressive_contract_rejects_non_contiguous_or_overflowing_prefix():
    base = _request()
    first = (await ReadingVocabularyGenerationService(ProgressiveProvider()).generate(
        _single_request(base)
    )).questions[0]
    for previous in (
        [first.model_copy(update={"order": 2})],
        [first, first],
        [first.model_copy(update={"order": index}) for index in range(1, 6)],
    ):
        with pytest.raises(ValidationError):
            _single_request(base, previous)

    with pytest.raises(ValidationError, match="single-question"):
        _single_request(base, [first], questionCount=5, currentCount=5)
    with pytest.raises(ValidationError, match="planned Reading passages"):
        _single_request(base, [first.model_copy(update={"passage_id": "p2"})])


@pytest.mark.asyncio
async def test_progressive_reading_contract_rejects_conflicting_committed_passages():
    base = _request()
    first = (await ReadingVocabularyGenerationService(ProgressiveProvider()).generate(
        _single_request(base)
    )).questions[0]
    second = first.model_copy(update={"order": 2, "passage_text": "異なる内容です。"})
    with pytest.raises(ValidationError, match="identical passageText"):
        _single_request(base, [first, second])


@pytest.mark.asyncio
async def test_reading_uses_two_source_passages_and_small_candidate_batches():
    provider = PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert len(response.questions) == 5
    assert response.prompt_version == "reading-vocabulary-generation"
    assert provider.calls[ReadingVocabularyGenerationService.PASSAGE_TYPE_NAME] == 2
    assert provider.candidate_slot_calls == [[1, 2], [3, 4], [5]]
    assert {question.passage_id for question in response.questions} == {"p1", "p2"}
    assert all(question.explanation_origin.startswith("이 답은") for question in response.questions)


@pytest.mark.asyncio
async def test_reading_invalid_vocabulary_candidates_are_sanitized_without_regeneration():
    provider = PipelineProvider(
        reading_invalid_vocabulary_candidate_rounds_by_order={3: 99}
    )
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(mode="CONTEXT_INFERENCE")
    )

    question = next(item for item in response.questions if item.order == 3)
    assert provider.slot_occurrences[3] == 1
    assert question.vocabulary_candidates == ["資料", "顧客への説明"]


@pytest.mark.asyncio
async def test_reading_generation_failures_do_not_consume_semantic_retry_budget():
    provider = PipelineProvider(
        candidate_reject_rounds_by_order={3: 4},
        semantic_reject_rounds_by_order={3: 2},
    )
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(mode="CONTEXT_INFERENCE")
    )

    assert len(response.questions) == 5
    assert provider.slot_occurrences[3] == 7
    assert provider.verification_occurrences[3] == 3
    retry_payloads = [
        slot
        for slots in provider.candidate_slot_payloads
        for slot in slots
        if slot["order"] == 3 and "retryFeedback" in slot
    ]
    assert retry_payloads
    assert any(
        slot["retryFeedback"].startswith("REPAIR_READING_AMBIGUITY")
        for slot in retry_payloads
    )


@pytest.mark.asyncio
async def test_vocabulary_generates_ten_questions_in_two_slot_batches():
    provider = PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(_vocab_request())

    assert len(response.questions) == 10
    assert provider.calls[ReadingVocabularyGenerationService.PASSAGE_TYPE_NAME] == 0
    assert provider.candidate_slot_calls == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    assert len({question.canonical_key for question in response.questions}) == 10


@pytest.mark.asyncio
async def test_bad_learning_language_regenerates_only_failed_slot():
    provider = PipelineProvider(bad_learning_lane_once={4})
    response = await ReadingVocabularyGenerationService(provider).generate(_vocab_request())

    assert len(response.questions) == 10
    assert provider.slot_occurrences[4] == 2
    assert all(provider.slot_occurrences[order] == 1 for order in range(1, 11) if order != 4)
    assert provider.candidate_slot_calls[-1] == [4]


@pytest.mark.asyncio
async def test_ascii_english_vocabulary_target_is_rejected_and_only_that_slot_regenerates():
    provider = PipelineProvider(english_target_once={2})
    response = await ReadingVocabularyGenerationService(provider).generate(_vocab_request())

    assert len(response.questions) == 10
    assert provider.slot_occurrences[2] == 2
    assert all(provider.slot_occurrences[order] == 1 for order in range(1, 11) if order != 2)
    assert provider.candidate_slot_calls[-1] == [2]
    assert all(question.target_expression != "deployment" for question in response.questions)


@pytest.mark.asyncio
async def test_ascii_english_vocabulary_options_are_rejected_and_only_that_slot_regenerates():
    provider = PipelineProvider(english_options_once={6})
    response = await ReadingVocabularyGenerationService(provider).generate(_vocab_request())

    assert len(response.questions) == 10
    assert provider.slot_occurrences[6] == 2
    assert all(provider.slot_occurrences[order] == 1 for order in range(1, 11) if order != 6)
    assert provider.candidate_slot_calls[-1] == [6]


def test_japanese_vocabulary_surface_language_allows_native_forms_and_real_ascii_acronyms():
    service = ReadingVocabularyGenerationService(PipelineProvider())

    for value in ("デプロイ", "本番環境への展開", "API", "SLA", "API障害"):
        service._assert_vocabulary_surface_language(
            "ja",
            value,
            reason="unexpected vocabulary language rejection",
        )

    for value in ("deployment", "customer service", "schedule"):
        with pytest.raises(ValueError):
            service._assert_vocabulary_surface_language(
                "ja",
                value,
                reason="english lexical target must be rejected",
            )


@pytest.mark.asyncio
async def test_invalid_review_target_is_skipped_instead_of_poisoning_generation():
    reviews = [
        {
            "canonicalKey": "invalid-deployment",
            "expression": "deployment",
            "masteryScore": 42,
            "wrongCount": 3,
            "previousQuestionTypes": ["SINGLE_CHOICE"],
        },
        {
            "canonicalKey": "review-ja",
            "expression": "見直す",
            "masteryScore": 55,
            "wrongCount": 2,
            "previousQuestionTypes": ["SINGLE_CHOICE"],
        },
    ]
    provider = PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(review_targets=reviews, review_question_count=2)
    )

    review_questions = [question for question in response.questions if question.review_target]
    assert len(review_questions) == 1
    assert review_questions[0].canonical_key == "review-ja"
    assert review_questions[0].target_expression == "見直す"
    assert all(question.target_expression != "deployment" for question in response.questions)


@pytest.mark.asyncio
async def test_semantic_ambiguity_regenerates_only_ambiguous_slot():
    provider = PipelineProvider(semantic_ambiguous_once={2})
    response = await ReadingVocabularyGenerationService(provider).generate(_vocab_request())

    assert len(response.questions) == 10
    assert provider.slot_occurrences[2] == 2
    assert all(provider.slot_occurrences[order] == 1 for order in range(1, 11) if order != 2)
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 2
    assert provider.candidate_slot_calls[-1] == [2]


@pytest.mark.asyncio
async def test_origin_explanation_language_failure_does_not_regenerate_questions():
    # Vocabulary explanation batches are 4+4+2. Make every Nano batch in the first
    # explanation pass wrong-language, then let the second Nano pass succeed.
    provider = PipelineProvider(nano_bad_language_calls=3)
    response = await ReadingVocabularyGenerationService(provider).generate(_vocab_request())

    assert len(response.questions) == 10
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 5
    assert all(provider.slot_occurrences[order] == 1 for order in range(1, 11))
    assert provider.calls[ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME] == 6
    assert provider.calls[ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME] == 0
    assert all("이 답은" in question.explanation_origin for question in response.questions)


@pytest.mark.asyncio
async def test_origin_explanation_uses_mini_fallback_after_nano_exhaustion():
    provider = PipelineProvider(nano_bad_language_calls=99)
    response = await ReadingVocabularyGenerationService(provider).generate(_vocab_request())

    assert len(response.questions) == 10
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 5
    assert provider.calls[ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME] == 6
    assert provider.calls[ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME] == 3
    assert all(question.explanation_origin.startswith("이 답은") for question in response.questions)


@pytest.mark.asyncio
async def test_review_slots_are_bound_deterministically_before_generation():
    reviews = [
        {
            "canonicalKey": "review-a",
            "expression": "見直す",
            "masteryScore": 42,
            "wrongCount": 3,
            "previousQuestionTypes": ["SINGLE_CHOICE"],
        },
        {
            "canonicalKey": "review-b",
            "expression": "踏まえる",
            "masteryScore": 55,
            "wrongCount": 2,
            "previousQuestionTypes": ["SINGLE_CHOICE"],
        },
    ]
    provider = PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(review_targets=reviews, review_question_count=2)
    )

    assert response.questions[0].review_target is True
    assert response.questions[0].canonical_key == "review-a"
    assert response.questions[0].target_expression == "見直す"
    assert response.questions[1].review_target is True
    assert response.questions[1].canonical_key == "review-b"
    assert response.questions[1].target_expression == "踏まえる"
    assert all(not question.review_target for question in response.questions[2:])


@pytest.mark.asyncio
async def test_composition_question_types_are_application_planned_not_model_chosen():
    provider = PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(mode="COMPOSITION")
    )

    ordering_orders = [
        question.order for question in response.questions if question.question_type.value == "ORDERING"
    ]
    assert ordering_orders == [1, 3, 6, 8]
    assert all(len(batch) == 1 for batch in provider.candidate_slot_calls)


@pytest.mark.asyncio
async def test_composition_ordering_repairs_invalid_answer_and_reconstructs_missing_target():
    provider = PipelineProvider(
        composition_invalid_ordering_once={8},
        composition_missing_target_once={8},
    )
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(mode="COMPOSITION")
    )

    question = next(item for item in response.questions if item.order == 8)
    assert question.question_type.value == "ORDERING"
    assert provider.slot_occurrences[8] == 2
    assert provider.verification_occurrences[8] == 0
    assert question.correct_answer == ["A", "B", "C", "D"]
    assert question.target_expression == "対応を早急に検討する必要があります"
    assert question.canonical_key == "対応を早急に検討する必要があります"
    retry_slot = next(
        slots[0]
        for slots in provider.candidate_slot_payloads
        if slots[0]["order"] == 8 and "retryFeedback" in slots[0]
    )
    assert retry_slot["retryFeedback"].startswith("REPAIR_ORDERING_KEYS")


@pytest.mark.asyncio
async def test_composition_generation_failures_do_not_consume_semantic_retry_budget():
    provider = PipelineProvider(
        candidate_reject_rounds_by_order={7: 4},
        semantic_reject_rounds_by_order={7: 2},
    )
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(mode="COMPOSITION")
    )

    assert len(response.questions) == 10
    assert provider.slot_occurrences[7] == 7
    assert provider.verification_occurrences[7] == 3
    retry_payloads = [
        slots[0]
        for slots in provider.candidate_slot_payloads
        if slots[0]["order"] == 7 and slots[0].get("retryTargetExpression")
    ]
    assert retry_payloads
    assert all(slot["preserveTargetOnRetry"] is True for slot in retry_payloads)


def test_origin_language_validation_is_per_question_not_combined():
    service = ReadingVocabularyGenerationService(PipelineProvider())
    with pytest.raises(ValueError):
        service._assert_language_lane(
            "ko",
            ["これは日本語だけの説明です。"],
            reason="origin explanation mismatch",
            allow_mixed_scripts=True,
        )
    service._assert_language_lane(
        "ko",
        ["「確認する」는 문맥에서 확인한다는 의미입니다."],
        reason="origin explanation mismatch",
        allow_mixed_scripts=True,
    )


@pytest.mark.asyncio
async def test_usage_distinction_target_leak_is_rejected_deterministically_and_only_slot_regenerates():
    provider = PipelineProvider(usage_target_leak_once={4})
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(mode="USAGE_DISTINCTION")
    )

    assert len(response.questions) == 10
    assert provider.slot_occurrences[4] == 2
    assert all(provider.slot_occurrences[order] == 1 for order in range(1, 11) if order != 4)
    assert provider.candidate_slot_calls.count([4]) == 2
    assert all(
        (question.target_expression or "") not in question.prompt
        for question in response.questions
    )


@pytest.mark.asyncio
async def test_usage_distinction_nano_prescreen_is_soft_and_cannot_burn_candidate_attempts():
    provider = PipelineProvider(prescreen_reject_always=set(range(1, 11)))
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(mode="USAGE_DISTINCTION")
    )

    assert len(response.questions) == 10
    assert provider.calls[ReadingVocabularyGenerationService.PRESCREEN_TYPE_NAME] == 10
    assert all(provider.slot_occurrences[order] == 1 for order in range(1, 11))
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 10
    assert all(len(payload["questions"]) == 1 for payload in provider.verification_payloads)


@pytest.mark.asyncio
async def test_usage_distinction_prescreen_and_verifier_never_receive_hidden_target_expression():
    provider = PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(mode="USAGE_DISTINCTION")
    )

    assert len(response.questions) == 10
    assert provider.prescreen_payloads
    assert provider.verification_payloads
    assert all(
        "targetExpression" not in question
        for payload in provider.prescreen_payloads
        for question in payload["questions"]
    )
    assert all(
        "targetExpression" not in question
        for payload in provider.verification_payloads
        for question in payload["questions"]
    )
    assert all(
        question.get("usageIntent")
        for payload in provider.prescreen_payloads
        for question in payload["questions"]
    )
    assert all(
        question.get("usageIntent")
        for payload in provider.verification_payloads
        for question in payload["questions"]
    )


def test_usage_distinction_requires_target_expression_to_be_the_correct_option():
    service = ReadingVocabularyGenerationService(PipelineProvider())
    question = {
        "order": 1,
        "questionType": "SINGLE_CHOICE",
        "difficulty": "CURRENT",
        "complexityBand": 3,
        "passageId": None,
        "passageText": None,
        "prompt": "納期を早めたいので、予定を（　）必要があります。",
        "options": [
            {"key": "A", "text": "前倒しする"},
            {"key": "B", "text": "延期する"},
            {"key": "C", "text": "中止する"},
            {"key": "D", "text": "保留する"},
        ],
        "correctAnswer": ["A"],
        "skillTag": "CONTEXT_USAGE",
        "evidenceText": None,
        "explanationOrigin": "설명입니다.",
        "explanationLearning": "納期を早める文脈なので「前倒しする」が自然です。",
        "targetExpression": "延期する",
        "canonicalKey": "postpone",
        "reviewTarget": False,
        "vocabularyCandidates": [],
    }
    parsed = PracticeGeneratedQuestion.model_validate(question)
    with pytest.raises(ValueError, match="correct option must equal targetExpression"):
        service._validate_usage_distinction_candidate(parsed)


@pytest.mark.asyncio
async def test_usage_distinction_uses_single_slot_generation_and_extended_retry_budget():
    provider = PipelineProvider(semantic_reject_rounds_by_order={3: 4})
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(mode="USAGE_DISTINCTION")
    )

    assert len(response.questions) == 10
    assert provider.slot_occurrences[3] == 5
    assert all(len(batch) == 1 for batch in provider.candidate_slot_calls)
    assert provider.candidate_slot_calls.count([3]) == 5
    assert provider.verification_occurrences[3] == 5


@pytest.mark.asyncio
async def test_usage_distinction_candidate_failures_do_not_consume_semantic_retry_budget():
    provider = PipelineProvider(
        candidate_reject_rounds_by_order={3: 4},
        semantic_reject_rounds_by_order={3: 4},
    )
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(mode="USAGE_DISTINCTION")
    )

    assert len(response.questions) == 10
    assert provider.slot_occurrences[3] == 9
    assert provider.verification_occurrences[3] == 5
    assert provider.candidate_slot_calls.count([3]) == 9


@pytest.mark.asyncio
async def test_usage_distinction_normalizes_application_owned_metadata_without_retry():
    reviews = [
        {
            "canonicalKey": "review-a",
            "expression": "見直す",
            "masteryScore": 42,
            "wrongCount": 3,
            "previousQuestionTypes": ["SINGLE_CHOICE"],
        }
    ]
    provider = PipelineProvider(usage_metadata_drift_once={4})
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(
            mode="USAGE_DISTINCTION",
            review_targets=reviews,
            review_question_count=1,
        )
    )

    question = response.questions[3]
    assert provider.slot_occurrences[4] == 1
    assert question.skill_tag == "CONTEXT_USAGE"
    assert question.review_target is False
    assert question.correct_answer == ["A"]
    assert question.canonical_key == question.target_expression.casefold()


@pytest.mark.asyncio
async def test_usage_distinction_semantic_retry_preserves_target_expression():
    provider = PipelineProvider(semantic_ambiguous_once={2})
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(mode="USAGE_DISTINCTION")
    )

    order_two_payloads = [
        slots[0]
        for slots in provider.candidate_slot_payloads
        if len(slots) == 1 and slots[0]["order"] == 2
    ]
    assert len(order_two_payloads) == 2
    assert "retryTargetExpression" not in order_two_payloads[0]
    assert order_two_payloads[1]["retryTargetExpression"] == response.questions[1].target_expression
    assert order_two_payloads[1]["preserveTargetOnRetry"] is True
    assert order_two_payloads[1]["retryFeedback"].startswith("REPAIR_AMBIGUITY")


@pytest.mark.asyncio
async def test_non_usage_vocabulary_skips_nano_usage_prescreen():
    provider = PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(_vocab_request())

    assert len(response.questions) == 10
    assert provider.calls[ReadingVocabularyGenerationService.PRESCREEN_TYPE_NAME] == 0


def test_usage_distinction_rejects_meaning_relation_stem_even_without_exact_target_leak():
    service = ReadingVocabularyGenerationService(PipelineProvider())
    question = {
        "order": 1,
        "questionType": "SINGLE_CHOICE",
        "difficulty": "CURRENT",
        "complexityBand": 3,
        "passageId": None,
        "passageText": None,
        "prompt": "この表現と同じ意味になるものはどれですか。",
        "options": [
            {"key": "A", "text": "前倒しする"},
            {"key": "B", "text": "延期する"},
            {"key": "C", "text": "中止する"},
            {"key": "D", "text": "繰り返す"},
        ],
        "correctAnswer": ["A"],
        "skillTag": "DISTINCTION",
        "evidenceText": None,
        "explanationOrigin": "설명입니다.",
        "explanationLearning": "文脈に合う表現です。",
        "targetExpression": "前倒しする",
        "canonicalKey": "maedaoshi",
        "reviewTarget": False,
        "vocabularyCandidates": [],
    }
    parsed = PracticeGeneratedQuestion.model_validate(question)
    with pytest.raises(ValueError, match="meaning/synonym"):
        service._validate_usage_distinction_candidate(parsed)
