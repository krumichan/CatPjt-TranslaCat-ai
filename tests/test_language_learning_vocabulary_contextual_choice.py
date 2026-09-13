import json
import re
from collections import Counter
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.features.language_learning.reading_vocabulary.contextual_choice_task import (
    CONTEXTUAL_CHOICE_ANSWER_KEYS,
    CONTEXTUAL_CHOICE_RECIPE_VERSION,
    CONTEXTUAL_CHOICE_SCENARIO_FAMILIES,
    CONTEXTUAL_CHOICE_SHADOW_RUBRIC_VERSION,
    render_contextual_choice_prompt,
)
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_recipe import (
    ContextualChoiceDemand,
    vocabulary_blind_rubric_payload,
    vocabulary_difficulty_recipe,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_shadow import (
    VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME,
)
from app.schemas.language_learning_practice import PracticeGenerationRequest


def _tagged_data(prompt: str) -> dict:
    match = re.search(r"<[^>]+>\n(.+?)\n</[^>]+>", prompt, re.S)
    if not match:
        return {}
    return json.loads(match.group(1))


class ContextualChoiceV2Provider:
    def __init__(
        self,
        *,
        deterministic_failure: str | None = None,
        verification_failures: list[str | None] | None = None,
        origin_artifact_once: bool = False,
    ) -> None:
        self.calls = Counter()
        self.generation_payloads: list[dict] = []
        self.generated_batches: list[list[dict]] = []
        self.verification_payloads: list[dict] = []
        self.deterministic_failure = deterministic_failure
        self.verification_failures = verification_failures or []
        self.origin_artifact_once = origin_artifact_once

    async def call(self, type_name, data, schema=None):
        self.calls[type_name] += 1
        if type_name == ReadingVocabularyGenerationService.TYPE_NAME:
            payload = _tagged_data(data)
            self.generation_payloads.append(payload)
            candidate_schema = schema["properties"]["candidates"]["items"]
            assert "correctAnswer" not in candidate_schema["properties"]
            assert "canonicalKey" not in candidate_schema["properties"]
            assert "context" not in candidate_schema["properties"]
            assert "completeSentence" in candidate_schema["properties"]
            candidates = [self._candidate(payload, index) for index in range(1, 4)]
            self._mutate_generation(payload, candidates)
            self.generated_batches.append(candidates)
            return {"candidates": candidates}
        if (
            type_name
            == ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME
        ):
            payload = json.loads(data.split("\n\n", 1)[1])
            self.verification_payloads.append(payload)
            assert set(schema["properties"]["verdicts"]["items"]["properties"]) == {
                "candidateId",
                "bestAnswerKey",
                "ambiguous",
                "supported",
                "skillFit",
                "definitionLike",
                "lexicalConceptRepeated",
                "genuineCompetitorKeys",
                "reason",
            }
            call_index = self.calls[type_name] - 1
            failure = (
                self.verification_failures[call_index]
                if call_index < len(self.verification_failures)
                else None
            )
            return {
                "verdicts": [
                    self._verdict(payload, candidate, failure)
                    for candidate in payload["candidates"]
                ]
            }
        if type_name in {
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME,
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME,
        }:
            payload = _tagged_data(data)
            artifact = (
                self.origin_artifact_once
                and self.calls[type_name] == 1
                and type_name
                == ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME
            )
            return {
                "explanations": [
                    {
                        "order": item["order"],
                        "text": (
                            'JSON: {"explanation":"도구 응답입니다."}'
                            if artifact
                            else "문맥상 이 표현이 가장 자연스럽습니다."
                        ),
                    }
                    for item in payload["questions"]
                ]
            }
        if type_name == VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME:
            payload = _tagged_data(data)
            return {
                "questionAssessments": [
                    {
                        "order": item["order"],
                        "difficultyStatus": "ASSESSED",
                        "observedBand": 4,
                        "alternativeBand": None,
                        "issueCodes": [],
                        "evidenceSegmentIds": [],
                        "difficultyConfidence": 0.8,
                    }
                    for item in payload["questions"]
                ]
            }
        raise AssertionError(f"unexpected type_name={type_name}")

    @staticmethod
    def _candidate(payload: dict, index: int) -> dict:
        slot = payload["candidateSlot"]
        round_number = payload["round"]
        global_order = len(payload["previousQuestions"]) + 1
        bound = slot.get("boundReviewTarget")
        target = (
            bound["expression"]
            if bound
            else f"語彙{global_order}第{round_number}候補{index}"
        )
        return {
            "candidateId": f"r{round_number}-c{index}",
            "targetExpression": target,
            "completeSentence": (
                f"案件{global_order}の条件{index}では、担当者が{target}方針を選びました。"
            ),
            "distractors": [
                {"text": f"代替表現{global_order}{index}一"},
                {"text": f"代替表現{global_order}{index}二"},
                {"text": f"代替表現{global_order}{index}三"},
            ],
            "explanationLearning": "状況の決め手に最も合う表現です。",
        }

    def _mutate_generation(self, payload: dict, candidates: list[dict]) -> None:
        if payload["round"] != 1:
            return
        if self.deterministic_failure == "target_absent":
            candidates[0]["completeSentence"] = "対象語を含まない文です。"
        elif self.deterministic_failure == "target_twice":
            target = candidates[0]["targetExpression"]
            candidates[0]["completeSentence"] = f"{target}後に再び{target}。"
        elif self.deterministic_failure == "bad_distractor":
            candidates[0]["distractors"][0]["text"] = candidates[0][
                "targetExpression"
            ]
        elif self.deterministic_failure == "short_distractors":
            candidates[0]["distractors"].pop()
        elif self.deterministic_failure == "duplicate_id":
            candidates[1]["candidateId"] = candidates[0]["candidateId"]
        elif self.deterministic_failure == "empty_explanation":
            candidates[0]["explanationLearning"] = ""
        elif self.deterministic_failure == "wrong_target_language":
            target = "한국어표현"
            candidates[0]["targetExpression"] = target
            candidates[0]["completeSentence"] = f"担当者は{target}方針です。"
        elif self.deterministic_failure == "wrong_sentence_language":
            target = candidates[0]["targetExpression"]
            candidates[0]["completeSentence"] = f"담당자는 {target} 방침입니다."
        elif self.deterministic_failure == "wrong_distractor_language":
            candidates[0]["distractors"][0]["text"] = "한국어선택지"
        elif self.deterministic_failure == "batch_duplicate":
            old_target = candidates[1]["targetExpression"]
            candidates[1]["targetExpression"] = candidates[0]["targetExpression"]
            candidates[1]["completeSentence"] = candidates[1][
                "completeSentence"
            ].replace(old_target, candidates[0]["targetExpression"])
        elif self.deterministic_failure == "review_sentence_duplicate":
            candidates[1]["completeSentence"] = candidates[0]["completeSentence"]
        elif self.deterministic_failure == "previous_repeat":
            previous = payload["previousQuestions"]
            if previous:
                old_target = previous[0]["targetExpression"]
                candidates[0]["targetExpression"] = old_target
                candidates[0]["completeSentence"] = f"担当者は{old_target}方針です。"

    @staticmethod
    def _verdict(payload: dict, candidate: dict, failure: str | None) -> dict:
        global_order = len(payload["previousQuestions"]) + 1
        expected_key = CONTEXTUAL_CHOICE_ANSWER_KEYS[(global_order - 1) % 4]
        wrong_keys = [key for key in CONTEXTUAL_CHOICE_ANSWER_KEYS if key != expected_key]
        candidate_number = int(candidate["candidateId"].rsplit("c", 1)[1])
        competitor_count = {1: 1, 2: 3, 3: 2}[candidate_number]
        if failure == "competitor_one":
            competitor_count = 1
        elif failure == "competitor_two":
            competitor_count = 2
        return {
            "candidateId": candidate["candidateId"],
            "bestAnswerKey": (
                wrong_keys[0] if failure == "answer_mismatch" else expected_key
            ),
            "ambiguous": failure == "all_reject",
            "supported": failure != "unsupported",
            "skillFit": failure != "skill",
            "definitionLike": failure == "definition",
            "lexicalConceptRepeated": failure == "lexical_repeat",
            "genuineCompetitorKeys": wrong_keys[:competitor_count],
            "reason": "independent learner-visible assessment",
        }


def _request(
    *,
    question_count: int = 1,
    complexity_band: int = 4,
    previous_questions: list[dict] | None = None,
    review_targets: list[dict] | None = None,
    review_question_count: int = 0,
) -> PracticeGenerationRequest:
    previous = previous_questions or []
    difficulty = "CURRENT"
    if question_count == 1 and previous:
        plan = ("CURRENT", "EASIER", "CURRENT", "CHALLENGE")
        difficulty = plan[len(previous) % len(plan)]
    return PracticeGenerationRequest.model_validate(
        {
            "requestId": f"contextual-v2-{len(previous) + 1}",
            "domain": "VOCABULARY",
            "mode": "CONTEXTUAL_CHOICE",
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "questionCount": question_count,
            "complexityBand": complexity_band,
            "easierCount": 2 if question_count == 10 else int(difficulty == "EASIER"),
            "currentCount": 6 if question_count == 10 else int(difficulty == "CURRENT"),
            "challengeCount": 2 if question_count == 10 else int(difficulty == "CHALLENGE"),
            "selectedKeywords": ["업무", "협업"],
            "reviewTargets": review_targets or [],
            "reviewQuestionCount": review_question_count,
            "generationDate": "2026-09-13",
            "previousQuestions": previous,
        }
    )


def _previous_question(
    order: int,
    *,
    target: str,
    review_target: bool = False,
    skill: str = "MEANING",
) -> dict:
    correct_key = CONTEXTUAL_CHOICE_ANSWER_KEYS[(order - 1) % 4]
    distractors = [f"既存候補{order}{index}" for index in range(1, 4)]
    options = distractors[:]
    options.insert(CONTEXTUAL_CHOICE_ANSWER_KEYS.index(correct_key), target)
    return {
        "order": order,
        "questionType": "SINGLE_CHOICE",
        "difficulty": "CURRENT",
        "complexityBand": 4,
        "passageId": None,
        "passageText": None,
        "prompt": f"既存の状況{order}では______方針です。",
        "options": [
            {"key": key, "text": text}
            for key, text in zip(CONTEXTUAL_CHOICE_ANSWER_KEYS, options, strict=True)
        ],
        "correctAnswer": [correct_key],
        "skillTag": skill,
        "evidenceText": None,
        "explanationOrigin": "문맥에 맞습니다.",
        "explanationLearning": "文脈に合います。",
        "targetExpression": target,
        "canonicalKey": target,
        "reviewTarget": review_target,
        "vocabularyCandidates": [],
    }


def test_complete_sentence_first_requires_one_exact_target_and_round_trips():
    target = "予定を調整する"
    sentence = "全員の都合を確認してから予定を調整する必要がある。"
    prompt = render_contextual_choice_prompt(sentence, target)

    assert prompt == "全員の都合を確認してから______必要がある。"
    assert target not in prompt
    assert prompt.replace("______", target) == sentence
    assert "日程を予定を調整する" not in sentence
    with pytest.raises(ValueError, match="exactly once"):
        render_contextual_choice_prompt("対象語を含まない文です。", target)
    with pytest.raises(ValueError, match="exactly once"):
        render_contextual_choice_prompt(f"{target}後に再び{target}。", target)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "target_absent",
        "target_twice",
        "bad_distractor",
        "short_distractors",
        "duplicate_id",
        "empty_explanation",
        "wrong_target_language",
        "wrong_sentence_language",
        "wrong_distractor_language",
    ],
)
async def test_deterministic_filter_keeps_other_survivors(failure):
    provider = ContextualChoiceV2Provider(deterministic_failure=failure)
    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert len(response.questions) == 1
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 1
    assert provider.calls[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME
    ] == 1
    verified_ids = {
        item["candidateId"] for item in provider.verification_payloads[0]["candidates"]
    }
    assert verified_ids == (
        {"r1-c3"} if failure == "duplicate_id" else {"r1-c2", "r1-c3"}
    )


@pytest.mark.asyncio
async def test_generation_returns_three_candidates_and_uses_one_batch_mini():
    provider = ContextualChoiceV2Provider()
    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert len(provider.generated_batches[0]) == 3
    assert len(provider.verification_payloads[0]["candidates"]) == 3
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 1
    assert provider.calls[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME
    ] == 1
    assert response.questions[0].target_expression == "語彙1第1候補2"


@pytest.mark.asyncio
async def test_round_two_uses_new_targets_and_is_the_only_retry_round():
    provider = ContextualChoiceV2Provider(
        verification_failures=["all_reject", None]
    )
    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert response.questions[0].target_expression == "語彙1第2候補2"
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 2
    assert provider.calls[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME
    ] == 2
    first_targets = {
        candidate["targetExpression"] for candidate in provider.generated_batches[0]
    }
    second_payload = provider.generation_payloads[1]
    assert first_targets.issubset(set(second_payload["excludedTargetExpressions"]))
    assert first_targets.isdisjoint(
        {candidate["targetExpression"] for candidate in provider.generated_batches[1]}
    )
    assert "retryTargetExpression" not in second_payload["candidateSlot"]


@pytest.mark.asyncio
async def test_round_two_exhaustion_does_not_start_a_third_round():
    provider = ContextualChoiceV2Provider(
        verification_failures=["all_reject", "all_reject"]
    )
    with pytest.raises(HTTPException) as exc_info:
        await ReadingVocabularyGenerationService(provider).generate(_request())

    assert exc_info.value.status_code == 502
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 2
    assert provider.calls[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME
    ] == 2


@pytest.mark.asyncio
async def test_review_batch_preserves_bound_target_and_preferred_skill():
    review = {
        "canonicalKey": "review-key",
        "expression": "再確認する",
        "preferredSkill": "REGISTER",
    }
    provider = ContextualChoiceV2Provider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(review_targets=[review], review_question_count=1)
    )

    assert {
        candidate["targetExpression"] for candidate in provider.generated_batches[0]
    } == {review["expression"]}
    assert len(
        {
            candidate["completeSentence"]
            for candidate in provider.generated_batches[0]
        }
    ) == 3
    item = response.questions[0]
    assert item.target_expression == review["expression"]
    assert item.canonical_key == review["canonicalKey"]
    assert item.skill_tag == "REGISTER"
    assert item.review_target is True


@pytest.mark.asyncio
async def test_review_ignores_lexical_repeat_and_accepts_one_competitor():
    review = {"canonicalKey": "review", "expression": "再確認する"}
    provider = ContextualChoiceV2Provider(
        verification_failures=["lexical_repeat"]
    )
    await ReadingVocabularyGenerationService(provider).generate(
        _request(review_targets=[review], review_question_count=1)
    )
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 1

    one = ContextualChoiceV2Provider(
        verification_failures=["competitor_one"]
    )
    await ReadingVocabularyGenerationService(one).generate(
        _request(review_targets=[review], review_question_count=1)
    )
    assert one.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 1


@pytest.mark.asyncio
async def test_review_batch_filters_duplicate_complete_sentences_independently():
    review = {"canonicalKey": "review", "expression": "再確認する"}
    provider = ContextualChoiceV2Provider(
        deterministic_failure="review_sentence_duplicate"
    )
    await ReadingVocabularyGenerationService(provider).generate(
        _request(review_targets=[review], review_question_count=1)
    )

    assert {
        item["candidateId"] for item in provider.verification_payloads[0]["candidates"]
    } == {"r1-c3"}


@pytest.mark.asyncio
@pytest.mark.parametrize("complexity_band", [4, 5])
async def test_b4_b5_require_two_genuine_competitors(complexity_band):
    one = ContextualChoiceV2Provider(
        verification_failures=["competitor_one", "competitor_two"]
    )
    await ReadingVocabularyGenerationService(one).generate(
        _request(complexity_band=complexity_band)
    )
    assert one.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 2

    two = ContextualChoiceV2Provider(
        verification_failures=["competitor_two"]
    )
    await ReadingVocabularyGenerationService(two).generate(
        _request(complexity_band=complexity_band)
    )
    assert two.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 1


@pytest.mark.asyncio
async def test_lexical_repeat_is_hard_but_scenario_similarity_is_not_a_verdict():
    repeated = ContextualChoiceV2Provider(
        verification_failures=["lexical_repeat", None]
    )
    await ReadingVocabularyGenerationService(repeated).generate(_request())
    assert repeated.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 2

    scenario_only = ContextualChoiceV2Provider()
    await ReadingVocabularyGenerationService(scenario_only).generate(_request())
    assert "scenarioSimilarity" not in scenario_only.verification_payloads[0]
    assert scenario_only.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["all_reject", "unsupported", "skill", "definition", "answer_mismatch"]
)
async def test_v2_semantic_hard_gates_replace_the_candidate_pool(failure):
    provider = ContextualChoiceV2Provider(
        verification_failures=[failure, None]
    )
    await ReadingVocabularyGenerationService(provider).generate(_request())

    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 2
    assert provider.calls[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME
    ] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["batch_duplicate", "previous_repeat"])
async def test_exact_and_canonical_repeats_are_deterministically_filtered(failure):
    previous = (
        [_previous_question(1, target="既存表現")]
        if failure == "previous_repeat"
        else []
    )
    provider = ContextualChoiceV2Provider(deterministic_failure=failure)
    await ReadingVocabularyGenerationService(provider).generate(
        _request(previous_questions=previous)
    )
    verified_ids = {
        item["candidateId"] for item in provider.verification_payloads[0]["candidates"]
    }
    assert (
        verified_ids == {"r1-c3"}
        if failure == "batch_duplicate"
        else "r1-c1" not in verified_ids
    )


@pytest.mark.parametrize("review_count", [0, 1, 2])
def test_scenario_plan_starts_at_first_new_item(review_count):
    reviews = [
        {"canonicalKey": f"review-{index}", "expression": f"復習表現{index}"}
        for index in range(1, review_count + 1)
    ]
    request = _request(
        question_count=10,
        review_targets=reviews,
        review_question_count=review_count,
    )
    slots = ReadingVocabularyGenerationService(
        ContextualChoiceV2Provider()
    )._build_slots(request, {})

    new_slots = [slot for slot in slots if not slot.review_target]
    assert new_slots[0].scenario_family == "SYSTEM_OPERATION"
    assert [slot.scenario_family for slot in new_slots] == list(
        CONTEXTUAL_CHOICE_SCENARIO_FAMILIES[: len(new_slots)]
    )
    assert all(slot.scenario_family is None for slot in slots if slot.review_target)


def test_progressive_scenario_counts_prior_new_items_not_global_order():
    previous = [
        _previous_question(1, target="復習表現", review_target=True),
        _previous_question(2, target="新規表現", review_target=False, skill="NUANCE"),
    ]
    slot = ReadingVocabularyGenerationService(
        ContextualChoiceV2Provider()
    )._build_slots(_request(previous_questions=previous), {})[0]

    assert slot.scenario_family == "CUSTOMER_COMMUNICATION"


@pytest.mark.asyncio
async def test_verifier_hides_answer_target_band_and_v1_verdicts():
    provider = ContextualChoiceV2Provider()
    await ReadingVocabularyGenerationService(provider).generate(_request())

    candidate = provider.verification_payloads[0]["candidates"][0]
    serialized = json.dumps(provider.verification_payloads[0], ensure_ascii=False)
    assert set(candidate) == {
        "candidateId",
        "prompt",
        "options",
        "skillTag",
        "reviewTarget",
    }
    assert "correctAnswer" not in serialized
    assert "targetExpression" not in serialized
    assert "requestedBand" not in serialized
    schema = ReadingVocabularyGenerationService._candidate_schema_for(_request())
    properties = schema["properties"]["candidates"]["items"]["properties"]
    assert "reserveDistractors" not in properties
    assert "context" not in properties


@pytest.mark.asyncio
async def test_answer_rotation_target_authority_and_daily_skill_balance():
    reviews = [
        {
            "canonicalKey": "review-a",
            "expression": "再確認する",
            "preferredSkill": "MEANING",
        },
        {
            "canonicalKey": "review-b",
            "expression": "慎重に進める",
            "preferredSkill": "NUANCE",
        },
    ]
    provider = ContextualChoiceV2Provider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(question_count=10, review_targets=reviews, review_question_count=2)
    )

    assert response.prompt_version == CONTEXTUAL_CHOICE_RECIPE_VERSION
    assert [question.correct_answer[0] for question in response.questions] == [
        "A", "B", "C", "D", "A", "B", "C", "D", "A", "B"
    ]
    assert Counter(question.skill_tag for question in response.questions) == {
        "MEANING": 2,
        "COLLOCATION": 2,
        "NUANCE": 2,
        "REGISTER": 2,
        "PRAGMATIC_FIT": 2,
    }
    for question in response.questions:
        assert question.prompt.count("______") == 1
        assert question.target_expression not in question.prompt
        assert sum(
            option.text == question.target_expression for option in question.options
        ) == 1
        assert next(
            option.text
            for option in question.options
            if option.key == question.correct_answer[0]
        ) == question.target_expression


@pytest.mark.asyncio
async def test_ten_progressive_slots_keep_rotation_and_skill_balance():
    provider = ContextualChoiceV2Provider()
    service = ReadingVocabularyGenerationService(provider)
    previous: list[dict] = []
    generated = []
    reviews = [
        {
            "canonicalKey": "review-a",
            "expression": "再確認する",
            "preferredSkill": "MEANING",
        },
        {
            "canonicalKey": "review-b",
            "expression": "慎重に進める",
            "preferredSkill": "NUANCE",
        },
    ]
    for global_order in range(1, 11):
        current_reviews = [reviews[global_order - 1]] if global_order <= 2 else reviews
        response = await service.generate(
            _request(
                previous_questions=previous,
                review_targets=current_reviews,
                review_question_count=int(global_order <= 2),
            )
        )
        question = response.questions[0]
        generated.append(question)
        previous.append(
            question.model_copy(update={"order": global_order}).model_dump(
                mode="json", by_alias=True
            )
        )

    assert [question.correct_answer[0] for question in generated] == [
        "A", "B", "C", "D", "A", "B", "C", "D", "A", "B"
    ]
    assert Counter(question.skill_tag for question in generated) == {
        "MEANING": 2,
        "COLLOCATION": 2,
        "NUANCE": 2,
        "REGISTER": 2,
        "PRAGMATIC_FIT": 2,
    }


@pytest.mark.asyncio
async def test_shadow_stays_v1_and_runs_after_selection():
    provider = ContextualChoiceV2Provider()
    response = await ReadingVocabularyGenerationService(
        provider,
        vocabulary_difficulty_shadow_enabled=True,
        vocabulary_difficulty_shadow_sample_percent=100,
    ).generate(_request())

    assert len(response.questions) == 1
    assert provider.calls[VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME] == 1
    assert (
        vocabulary_blind_rubric_payload("CONTEXTUAL_CHOICE")["version"]
        == CONTEXTUAL_CHOICE_SHADOW_RUBRIC_VERSION
    )


@pytest.mark.asyncio
async def test_origin_explanation_validation_remains_independent():
    provider = ContextualChoiceV2Provider(origin_artifact_once=True)
    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert response.questions[0].explanation_origin == "문맥상 이 표현이 가장 자연스럽습니다."
    assert provider.calls[
        ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME
    ] == 2


@pytest.mark.asyncio
async def test_be_persisted_order3_wire_fixture_remains_compatible():
    fixture = Path(__file__).parent / "fixtures" / "contextual_choice_order3_be_request.json"
    request = PracticeGenerationRequest.model_validate_json(
        fixture.read_text(encoding="utf-8")
    )
    response = await ReadingVocabularyGenerationService(
        ContextualChoiceV2Provider()
    ).generate(request)

    item = response.questions[0]
    assert item.difficulty.value == "CURRENT"
    assert item.complexity_band == 4
    assert item.skill_tag == "COLLOCATION"
    assert item.correct_answer == ["C"]


def test_recipe_v2_review_cap_and_shadow_ruler_are_explicit():
    recipe = vocabulary_difficulty_recipe(
        mode="CONTEXTUAL_CHOICE",
        band=5,
        skill_tag="PRAGMATIC_FIT",
        question_type="SINGLE_CHOICE",
    )
    assert recipe.version == "vocabulary-contextual-choice-recipe-v2"
    assert isinstance(recipe.demand, ContextualChoiceDemand)
    assert recipe.demand.minimum_close_distractors == 2
    assert recipe.demand.maximum_close_distractors == 3
    assert (
        vocabulary_blind_rubric_payload("CONTEXTUAL_CHOICE")["version"]
        == "vocabulary-contextual-choice-shadow-rubric-v1"
    )

    with pytest.raises(ValidationError, match="must not exceed 2"):
        _request(
            question_count=10,
            review_targets=[
                {"canonicalKey": f"review-{index}", "expression": f"復習{index}"}
                for index in range(3)
            ],
            review_question_count=3,
        )
