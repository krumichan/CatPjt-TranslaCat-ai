import json
import re
from collections import Counter, defaultdict

import pytest

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
        english_target_once: set[int] | None = None,
        english_options_once: set[int] | None = None,
    ):
        self.calls = Counter()
        self.candidate_slot_calls: list[list[int]] = []
        self.slot_occurrences = defaultdict(int)
        self.verification_round = 0
        self.semantic_ambiguous_once = set(semantic_ambiguous_once or set())
        self.bad_learning_lane_once = set(bad_learning_lane_once or set())
        self.nano_bad_language_calls = nano_bad_language_calls
        self.duplicate_new_canonical_once = set(duplicate_new_canonical_once or set())
        self.usage_target_leak_once = set(usage_target_leak_once or set())
        self.prescreen_reject_once = set(prescreen_reject_once or set())
        self.prescreen_reject_always = set(prescreen_reject_always or set())
        self.semantic_reject_rounds_by_order = dict(semantic_reject_rounds_by_order or {})
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
                ambiguous = (
                    (self.verification_round == 1 and order in self.semantic_ambiguous_once)
                    or self.verification_round <= self.semantic_reject_rounds_by_order.get(order, 0)
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
        if order in self.bad_learning_lane_once and self.slot_occurrences[order] == 1:
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
            and order in self.usage_target_leak_once
            and self.slot_occurrences[order] == 1
        ):
            prompt = f"「{target_expression}」と同じ意味で最も自然な表現はどれですか。"

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
            "skillTag": slot["skillTag"],
            "evidenceText": "担当者は資料を確認しました。" if not vocab else None,
            "explanationLearning": "この答えが文脈に最も合います。",
            "targetExpression": target_expression,
            "canonicalKey": canonical_key,
            "reviewTarget": review_target,
            "vocabularyCandidates": ["資料"] if not vocab else [],
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


@pytest.mark.asyncio
async def test_reading_uses_two_source_passages_small_candidate_batches_and_v3_contract():
    provider = PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert len(response.questions) == 5
    assert response.prompt_version == "reading-vocabulary-generation-v5"
    assert provider.calls[ReadingVocabularyGenerationService.PASSAGE_TYPE_NAME] == 2
    assert provider.candidate_slot_calls == [[1, 2], [3, 4], [5]]
    assert {question.passage_id for question in response.questions} == {"p1", "p2"}
    assert all(question.explanation_origin.startswith("이 답은") for question in response.questions)


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
async def test_invalid_legacy_review_target_is_skipped_instead_of_poisoning_generation():
    reviews = [
        {
            "canonicalKey": "legacy-deployment",
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
    assert provider.candidate_slot_calls[-1] == [4]
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
    assert provider.calls[ReadingVocabularyGenerationService.PRESCREEN_TYPE_NAME] == 1
    assert all(provider.slot_occurrences[order] == 1 for order in range(1, 11))
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1


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


@pytest.mark.asyncio
async def test_usage_distinction_uses_single_slot_generation_and_extended_retry_budget():
    provider = PipelineProvider(semantic_reject_rounds_by_order={3: 4})
    response = await ReadingVocabularyGenerationService(provider).generate(
        _vocab_request(mode="USAGE_DISTINCTION")
    )

    assert len(response.questions) == 10
    assert provider.slot_occurrences[3] == 5
    assert all(len(batch) == 1 for batch in provider.candidate_slot_calls)
    assert provider.candidate_slot_calls[-1] == [3]


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
        "targetExpression": "予定を前倒しする",
        "canonicalKey": "maedaoshi",
        "reviewTarget": False,
        "vocabularyCandidates": [],
    }
    parsed = PracticeGeneratedQuestion.model_validate(question)
    with pytest.raises(ValueError, match="meaning/synonym"):
        service._validate_usage_distinction_candidate(parsed)
