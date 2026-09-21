"""Application boundaries for Reading content quality, not model-accuracy claims.

The original set-4 trace stays in the private QA artifact; these are synthetic,
CI-safe counterparts. In particular, no test rewrites the captured Mini PASS.
"""

import asyncio
import json

import pytest
from pydantic import ValidationError

from app.features.language_learning.reading_vocabulary.prompts import (
    PRACTICE_GENERATION_SYSTEM_PROMPT,
    PRACTICE_READING_PASSAGE_SYSTEM_PROMPT,
    build_practice_generation_prompt,
    build_practice_verification_prompt,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_recipe import (
    question_demand_recipe,
    structure_judgment_contract,
)
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
    _PracticeVerificationPayload,
    _READING_VERIFICATION_SCHEMA,
    _ReadingPracticeVerificationPayload,
    _QuestionSlot,
    _PlannedPassage,
)
from app.schemas.language_learning_practice import (
    PracticeDifficulty,
    PracticeGeneratedQuestion,
    ReadingQuestionPlan,
)
from tests.test_language_learning_reading_vocabulary import _practice_data, _request


PASSAGE = (
    "最初は電車で町へ行くつもりでした。駅からホテルまでは荷物を持って歩くと大変です。"
    "そこで、駅からホテルまではバスに乗ることにしました。"
)
EVIDENCE = "駅からホテルまではバスに乗ることにしました。"


def _question(*, prompt: str = "筆者が電車の代わりにバスを選んだのはなぜですか。",
              skill: str = "INFERENCE", order: int = 1) -> PracticeGeneratedQuestion:
    return PracticeGeneratedQuestion.model_validate({
        "order": order, "questionType": "SINGLE_CHOICE", "difficulty": "CURRENT",
        "complexityBand": 3, "passageId": "p1", "passageText": PASSAGE,
        "prompt": prompt, "skillTag": skill,
        "options": [
            {"key": "A", "text": "駅からホテルまで歩くのが大変だから"},
            {"key": "B", "text": "町までの電車が止まるから"},
            {"key": "C", "text": "ホテルが駅の中にあるから"},
            {"key": "D", "text": "荷物を持たずに行くから"},
        ],
        "correctAnswer": ["A"], "evidenceText": EVIDENCE,
        "explanationOrigin": "역에서 호텔까지 걷기 어렵기 때문입니다.",
        "explanationLearning": "駅からホテルまで歩くと大変だからです。",
    })


def _verdict(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "order": 1, "bestAnswerKey": "A", "ambiguous": False,
        "supported": True, "reason": "local bus choice is supported",
        "modeFit": True, "answerLeakage": False,
        "contextDependent": False, "distractorsPlausible": True,
        "stemPresuppositionsSupported": True,
        "stemEvidenceSpanIds": ["p1:s3"],
        "readingOperation": "INFERENCE", "distinctReadingTask": True,
        "boundedStructureScope": True,
    }
    result.update(changes)
    return result


class _Verifier:
    def __init__(self, verdict: dict[str, object]):
        self.verdict = verdict
        self.calls: list[tuple[str, dict, dict]] = []

    async def call(self, type_name, data, schema=None):
        self.calls.append((type_name, json.loads(data.split("\n\n", 1)[1]), schema))
        return {"verdicts": [self.verdict]}


def _outcome(verdict: dict[str, object], *, mode: str = "CONTEXT_INFERENCE",
             question: PracticeGeneratedQuestion | None = None,
             previous: list[PracticeGeneratedQuestion] | None = None):
    provider = _Verifier(verdict)
    request = _request(mode=mode, question_count=1, easier=0, current=1, challenge=0)
    result = asyncio.run(ReadingVocabularyGenerationService(provider)
                         ._semantic_verification_outcome(
                             request, [question or _question()],
                             previous_reading_questions=previous,
                         ))
    assert len(provider.calls) == 1
    return result, provider.calls[0]


def test_original_style_local_support_does_not_override_unsupported_stem_premise():
    result, (_, payload, schema) = _outcome(_verdict(stemPresuppositionsSupported=False))
    assert result.failures == {1: "question stem presupposition is not supported by passage"}
    assert payload["questions"][0]["prompt"] == _question().prompt
    assert "correctAnswer" not in payload["questions"][0]
    assert "stemPresuppositionsSupported" in schema["properties"]["verdicts"]["items"]["required"]


def test_old_quality_only_response_cannot_silently_default_new_reading_verdict_to_pass():
    old = _verdict()
    for field in ("stemPresuppositionsSupported", "stemEvidenceSpanIds", "readingOperation", "distinctReadingTask"):
        old.pop(field)
    with pytest.raises(ValidationError):
        _ReadingPracticeVerificationPayload.model_validate({"verdicts": [old]})
    assert _PracticeVerificationPayload.model_validate({"verdicts": [old]}).verdicts[0].supported


@pytest.mark.parametrize(("verdict", "reason"), [
    (_verdict(distinctReadingTask=False), "question repeats a previous reading judgment"),
    (_verdict(readingOperation="DIRECT_RETRIEVAL"), "inference question only requires direct retrieval"),
])
def test_reading_contract_rejects_grounding_repetition_and_direct_retrieval(verdict, reason):
    result, _ = _outcome(verdict)
    assert result.failures == {1: reason}


@pytest.mark.parametrize("ids", [[], ["p1:s999"], ["p2:s1"], ["p1:s3", "p1:s3"]])
def test_verifier_evidence_binding_error_is_provider_failure_not_candidate_repair(ids):
    provider = _Verifier(_verdict(stemEvidenceSpanIds=ids))
    request = _request(mode="CONTEXT_INFERENCE", question_count=1,
                       easier=0, current=1, challenge=0)
    with pytest.raises(ValueError, match="semantic verifier unavailable"):
        asyncio.run(ReadingVocabularyGenerationService(provider)
                    ._semantic_verification_outcome(request, [_question()]))
    assert len(provider.calls) == 3  # Existing verifier retry cap, no generator repair.
    assert "stemEvidenceSpanIds" in provider.calls[0][2]["properties"]["verdicts"]["items"]["required"]
    assert provider.calls[0][2]["properties"]["verdicts"]["items"]["properties"][
        "stemEvidenceSpanIds"
    ]["items"]["enum"] == ["p1:s1", "p1:s2", "p1:s3"]


def test_supported_distinct_inference_can_pass_without_extra_verifier():
    result, (_, payload, _) = _outcome(_verdict())
    assert result.failures == {}
    assert payload["questions"][0]["questionDemand"] is None  # Frozen CONTEXT_INFERENCE recipe.


def test_b5_whole_text_evidence_and_per_slot_judgment_scopes_are_distinct():
    demand = question_demand_recipe(5, mode="STRUCTURE", skill_tag="STRUCTURE")
    assert demand.question_demand is not None
    assert demand.question_demand.kind.value == "WHOLE_TEXT_SYNTHESIS"
    assert demand.question_demand.evidence_scope.value == "WHOLE_TEXT"
    assert demand.question_demand.inference_requirement == "WHOLE_ARGUMENT_STRUCTURE"
    assert [structure_judgment_contract(band=5, position=position, count=3)["judgmentScope"]
            for position in (1, 2, 3)] == [
        "ONE_BOUNDED_ARGUMENT_FUNCTION",
        "ONE_BOUNDED_ARGUMENT_FUNCTION",
        "WHOLE_ARGUMENT_PROGRESSION",
    ]
    assert structure_judgment_contract(band=5, position=1, count=2)[
        "minimumEvidenceParagraphs"
    ] == 2
    assert structure_judgment_contract(band=4, position=1, count=3) is None


def test_b5_verifier_receives_whole_evidence_but_early_bounded_judgment():
    question = _question(skill="STRUCTURE").model_copy(update={"complexity_band": 5})
    result, (_, payload, _) = _outcome(
        _verdict(readingOperation="DISCOURSE_STRUCTURE", boundedStructureScope=True),
        mode="STRUCTURE", question=question,
    )
    assert result.failures == {}
    assert payload["questions"][0]["questionDemand"]["evidenceScope"] == "WHOLE_TEXT"
    assert payload["questions"][0]["structureJudgmentContract"]["judgmentScope"] == (
        "ONE_BOUNDED_ARGUMENT_FUNCTION"
    )
    assert payload["readingEvidenceSpans"][2]["id"] == "p1:s3"
    assert payload["readingEvidenceSpans"][2]["passageId"] == "p1"
    assert EVIDENCE in payload["readingEvidenceSpans"][2]["text"]
    assert payload["readingEvidenceSpans"][2]["text"] in PASSAGE


@pytest.mark.asyncio
async def test_b5_passage_plan_and_question_generator_share_bounded_scope():
    request = _request(mode="STRUCTURE", question_count=3, easier=0,
                       current=3, challenge=0, complexity_band=5)

    class RejectingPassageProvider:
        def __init__(self):
            self.prompts = []

        async def call(self, type_name, data, schema=None):
            self.prompts.append(_practice_data(data))
            raise RuntimeError("synthetic passage stop before any model result")

    passage_provider = RejectingPassageProvider()
    with pytest.raises(ValueError):
        await ReadingVocabularyGenerationService(passage_provider)._generate_reading_passages(request)
    planned = passage_provider.prompts[0]["plannedQuestionSlots"]
    assert [slot["globalOrder"] for slot in planned] == [1, 2, 3]
    assert all(slot["questionDemand"]["questionDemand"]["evidenceScope"] == "WHOLE_TEXT"
               for slot in planned)
    assert [slot["structureJudgmentContract"]["judgmentScope"] for slot in planned] == [
        "ONE_BOUNDED_ARGUMENT_FUNCTION", "ONE_BOUNDED_ARGUMENT_FUNCTION",
        "WHOLE_ARGUMENT_PROGRESSION",
    ]
    plans = tuple(ReadingQuestionPlan.model_validate({
        "globalOrder": order, "skillTag": slot["skillTag"],
        "difficulty": "CURRENT", "complexityBand": 5,
        "clueQuote": EVIDENCE, "questionFocus": f"focus-{order}",
    }) for order, slot in enumerate(planned, 1))
    slots = ReadingVocabularyGenerationService(_CandidateRecorder())._build_slots(
        request, {"p1": _PlannedPassage(PASSAGE, plans)},
    )
    candidate_provider = _CandidateRecorder()
    await ReadingVocabularyGenerationService(candidate_provider)._generate_candidate_batch(
        request, [slots[0]], {},
    )
    early = candidate_provider.payload["candidateSlots"][0]
    assert early["structureJudgmentContract"]["judgmentScope"] == "ONE_BOUNDED_ARGUMENT_FUNCTION"
    assert early["reservedFutureQuestionFocuses"] == ["focus-2", "focus-3"]
    await ReadingVocabularyGenerationService(candidate_provider)._generate_candidate_batch(
        request, [slots[2]], {},
    )
    final = candidate_provider.payload["candidateSlots"][0]
    assert final["structureJudgmentContract"]["judgmentScope"] == "WHOLE_ARGUMENT_PROGRESSION"
    assert final["reservedFutureQuestionFocuses"] == []


def test_valid_exact_span_does_not_override_unsupported_stem_or_unique_answer():
    unsupported, _ = _outcome(_verdict(stemPresuppositionsSupported=False))
    assert unsupported.failures == {1: "question stem presupposition is not supported by passage"}
    ambiguous, _ = _outcome(_verdict(ambiguous=True))
    assert ambiguous.failures == {1: "ambiguous single-choice item"}


def test_verifier_reason_fragment_is_not_the_authoritative_passage_quote():
    result, (_, payload, _) = _outcome(_verdict(reason="Ellipsis quote: 本文にない…引用"))
    assert result.failures == {}
    assert EVIDENCE in payload["readingEvidenceSpans"][2]["text"]
    assert payload["readingEvidenceSpans"][2]["text"] in PASSAGE


def test_structure_requires_discourse_operation_not_event_reason_retrieval():
    question = _question(prompt="第1段落と第2段落はどのような関係ですか。", skill="STRUCTURE")
    result, (_, payload, _) = _outcome(
        _verdict(readingOperation="DIRECT_RETRIEVAL"),
        mode="STRUCTURE", question=question,
    )
    assert result.failures == {1: "structure question does not require discourse reasoning"}
    assert payload["questions"][0]["questionDemand"]["kind"]


def test_structure_mini_receives_future_slot_scope_and_rejects_consuming_the_whole_passage():
    question = _question(prompt="この文章全体の論の進み方として、最も適切なものはどれですか。",
                         skill="STRUCTURE")
    result, (_, payload, _) = _outcome(
        _verdict(readingOperation="DISCOURSE_STRUCTURE", distinctReadingTask=False),
        mode="STRUCTURE", question=question,
    )
    assert result.failures == {1: "question consumes later passage structure tasks"}
    assert payload["questions"][0]["samePassageTaskPosition"] == 1
    assert payload["questions"][0]["samePassageTaskCount"] == 3
    assert "boundedStructureScope" in (
        build_practice_verification_prompt(_request(mode="STRUCTURE"), [{}])
    )


def test_structure_bounded_scope_is_an_independent_mini_verdict_not_a_forced_pass():
    question = _question(prompt="この文章全体の第1段落から第3段落への議論はどう進みますか。",
                         skill="STRUCTURE")
    result, (_, payload, schema) = _outcome(
        _verdict(readingOperation="DISCOURSE_STRUCTURE", boundedStructureScope=False),
        mode="STRUCTURE", question=question,
    )
    assert result.failures == {1: "structure question consumes future passage tasks"}
    assert payload["questions"][0]["samePassageTaskCount"] == 3
    assert "boundedStructureScope" in schema["properties"]["verdicts"]["items"]["required"]


def test_previous_question_is_visible_without_its_answer_and_semantic_duplicate_rejects():
    prior = _question(prompt="第1段落の交通手段は何ですか。", order=2)
    result, (_, payload, _) = _outcome(_verdict(distinctReadingTask=False), previous=[prior])
    assert result.failures == {1: "question repeats a previous reading judgment"}
    assert payload["previousReadingQuestions"][0]["prompt"] == prior.prompt
    assert "correctAnswer" not in payload["previousReadingQuestions"][0]


def test_normalized_exact_duplicate_is_rejected_before_mini_but_different_task_is_allowed():
    request = _request(mode="COMPREHENSION", question_count=1, easier=0, current=1, challenge=0)
    service = ReadingVocabularyGenerationService(_Verifier(_verdict()))
    question = _question(prompt="駅からホテルまではどう行きますか。", skill="DETAIL")
    slot = _QuestionSlot(order=1, difficulty=PracticeDifficulty.CURRENT,
                         complexity_band=3, skill_tag="DETAIL", passage_id="p1",
                         passage_text=PASSAGE, reading_mode="COMPREHENSION")
    duplicate = _question(prompt="駅から ホテルまではどう行きますか。", skill="DETAIL", order=2)
    with pytest.raises(ValueError, match="repeats a normalized prompt"):
        service._validate_candidate(request, question, slot, {2: duplicate})
    different = _question(prompt="ホテルから町の中心へはどう行きますか。", skill="DETAIL", order=2)
    service._validate_candidate(request, question, slot, {2: different})


def test_generation_includes_verified_prefix_without_exposing_answers():
    request = _request(mode="COMPREHENSION", question_count=1, easier=0, current=1, challenge=0)
    prior = _question()
    prompt = build_practice_generation_prompt(request, [], previous_questions=[prior])
    payload = _practice_data(prompt)
    assert payload["previousQuestions"][0]["prompt"] == prior.prompt
    assert "correctAnswer" not in payload["previousQuestions"][0]
    assert "stemPresuppositionsSupported" in _READING_VERIFICATION_SCHEMA["properties"]["verdicts"]["items"]["required"]


def test_comprehension_passage_prompt_supports_planned_inference_without_leaking_answer():
    # Both passage groups contain an INFERENCE slot; B1 still needs an implicit clue.
    assert "Every COMPREHENSION passage is also used for an INFERENCE question" in (
        PRACTICE_READING_PASSAGE_SYSTEM_PROMPT
    )
    assert "Do not state that inference as an explicit fact" in (
        PRACTICE_READING_PASSAGE_SYSTEM_PROMPT
    )
    assert "Keep the language at" in PRACTICE_READING_PASSAGE_SYSTEM_PROMPT


def test_b1_inference_candidate_requires_grounded_unstated_evidence_in_existing_generator():
    request = _request(mode="COMPREHENSION", question_count=1,
                       easier=0, current=1, challenge=0).model_copy(
                           update={"complexity_band": 1})
    passage = "今日は雨です。友だちは傘を忘れました。私は大きな傘を持っています。"
    slot = _QuestionSlot(
        order=1, difficulty=PracticeDifficulty.CURRENT, complexity_band=1,
        skill_tag="INFERENCE", passage_id="p1", passage_text=passage,
        reading_mode="COMPREHENSION",
    )
    service = ReadingVocabularyGenerationService(_Verifier(_verdict()))
    schema = service._candidate_schema_for(request, [slot])
    required = schema["properties"]["questions"]["items"]["required"]
    assert "inferenceClueQuote" in required and "unstatedInference" in required
    assert "B1 COMPREHENSION candidates" in PRACTICE_GENERATION_SYSTEM_PROMPT
    valid = {
        "inferenceClueQuote": "友だちは傘を忘れました。",
        "unstatedInference": "二人は一緒に傘を使うかもしれません。",
    }
    service._validate_b1_inference_candidate_evidence(request, slot, valid)
    for invalid in (
        {**valid, "inferenceClueQuote": "本文にない引用"},
        {**valid, "unstatedInference": "今日は雨です。"},
        {"inferenceClueQuote": None, "unstatedInference": None},
    ):
        with pytest.raises(ValueError, match="unstated passage-grounded conclusion"):
            service._validate_b1_inference_candidate_evidence(request, slot, invalid)


@pytest.mark.asyncio
async def test_b1_internal_candidate_evidence_is_not_passed_to_public_question_schema():
    request = _request(mode="COMPREHENSION", question_count=1,
                       easier=0, current=1, challenge=0).model_copy(
                           update={"complexity_band": 1})
    passage = "今日は雨です。友だちは傘を忘れました。私は大きな傘を持っています。"
    slot = _QuestionSlot(
        order=1, difficulty=PracticeDifficulty.CURRENT, complexity_band=1,
        skill_tag="INFERENCE", passage_id="p1", passage_text=passage,
        reading_mode="COMPREHENSION",
    )
    candidate = _question().model_dump(by_alias=True)
    candidate.update({
        "passageId": "p1", "passageText": passage,
        "prompt": "友だちが傘を忘れたので、二人はどうするかもしれませんか。",
        "inferenceClueQuote": "友だちは傘を忘れました。",
        "unstatedInference": "二人は一緒に傘を使うかもしれません。",
    })

    class CandidateProvider:
        async def call(self, type_name, data, schema=None):
            return {"questions": [candidate]}

    service = ReadingVocabularyGenerationService(CandidateProvider())
    accepted: dict[int, PracticeGeneratedQuestion] = {}
    failures = await service._generate_candidate_batch(request, [slot], accepted)
    assert "extra_forbidden" not in str(failures)
    if 1 in accepted:
        assert "inferenceClueQuote" not in accepted[1].model_dump(by_alias=True)
        assert "unstatedInference" not in accepted[1].model_dump(by_alias=True)


@pytest.mark.asyncio
async def test_b1_passage_requires_a_visible_clue_and_unstated_inference_before_first_question():
    request = _request(mode="COMPREHENSION", question_count=1, easier=0, current=1, challenge=0)
    request = request.model_copy(update={"complexity_band": 1})
    simple = "母とスーパーに行きました。りんごを三つ買いました。帰る前にパンも買いました。"
    planned = (
        "明日は店が休みです。母は今夜の夕食の材料が足りないことに気づきました。"
        "そこで、母とスーパーに行き、野菜を買いました。"
    )

    class PassageProvider:
        def __init__(self):
            self.calls = []

        async def call(self, type_name, data, schema=None):
            self.calls.append((type_name, schema, data))
            if len(self.calls) == 1:
                return {"passageId": "p1", "passageText": simple,
                        "inferenceClueQuote": "", "unstatedInference": ""}
            return {"passageId": "p1", "passageText": planned,
                    "inferenceClueQuote": "明日は店が休みです。",
                    "unstatedInference": "今夜の食事のために今日買い物を済ませる必要がありました。"}

    provider = PassageProvider()
    service = ReadingVocabularyGenerationService(provider)
    passages = await service._generate_reading_passages(request)
    assert passages["p1"].text == planned
    assert passages["p1"].plans == ()  # Legacy single-item request has no private bundle.
    assert len(provider.calls) == 2  # Existing passage retry bound and stage only.
    assert all(call[0] == service.PASSAGE_TYPE_NAME for call in provider.calls)
    assert "inferenceClueQuote" in provider.calls[0][1]["required"]
    assert "upcomingInferenceRequirement" in provider.calls[0][2]


@pytest.mark.asyncio
async def test_context_inference_passage_reserves_distinct_unstated_judgments_for_all_future_slots():
    request = _request(mode="CONTEXT_INFERENCE", question_count=1,
                       easier=0, current=1, challenge=0)
    passage = (
        "来月、友達と海の近くへ旅行します。電車は安いですが、駅からホテルまで少し歩きます。"
        "車なら荷物を運べますが、高速道路が混みそうです。私たちは荷物を減らして電車を選びました。"
        "友達は有名な店を見つけましたが、私は駅の近くの小さな店も調べています。"
        "その店には地元の料理があり、予約なしでも入れる時間があるそうです。"
        "雨の予報を見て、二人は室内でも楽しめる場所の案内を保存しました。"
    )
    plans = [
        {"questionOrder": 1, "clueQuote": "電車は安いですが、駅からホテルまで少し歩きます。",
         "unstatedInference": "二人は交通費と歩く負担を比べて決めました。"},
        {"questionOrder": 2, "clueQuote": "私は駅の近くの小さな店も調べています。",
         "unstatedInference": "混雑した有名店以外の選択肢も用意したいようです。"},
        {"questionOrder": 3, "clueQuote": "室内でも楽しめる場所の案内を保存しました。",
         "unstatedInference": "雨が降れば観光先を変えるつもりです。"},
    ]

    class PassageProvider:
        def __init__(self):
            self.calls = []

        async def call(self, type_name, data, schema=None):
            self.calls.append((type_name, schema, data))
            return {"passageId": "p1", "passageText": passage,
                    "inferencePlans": plans[:1] if len(self.calls) == 1 else plans}

    provider = PassageProvider()
    result = await ReadingVocabularyGenerationService(provider)._generate_reading_passages(request)
    assert result["p1"].text == passage
    assert result["p1"].plans == ()
    assert len(provider.calls) == 2  # Existing passage attempts, no added provider stage.
    assert "inferencePlans" in provider.calls[0][1]["required"]
    assert '"distinctUnstatedJudgmentCount":3' in provider.calls[0][2]


def test_context_inference_candidate_receives_future_position_without_prior_answer():
    request = _request(mode="CONTEXT_INFERENCE", question_count=1,
                       easier=0, current=1, challenge=0)
    request = request.model_copy(update={"question_offset": 1, "previous_questions": [_question()]})
    slot = _QuestionSlot(order=1, difficulty=PracticeDifficulty.CURRENT,
                         complexity_band=3, skill_tag="INFERENCE", passage_id="p1",
                         passage_text=PASSAGE, reading_mode="CONTEXT_INFERENCE")
    provider = _CandidateRecorder()
    asyncio.run(ReadingVocabularyGenerationService(provider)
                ._generate_candidate_batch(request, [slot], {}))
    assert provider.payload is not None
    assert provider.payload["candidateSlots"][0]["samePassageTaskPosition"] == 2
    assert provider.payload["candidateSlots"][0]["samePassageTaskCount"] == 3
    assert "correctAnswer" not in provider.payload["previousQuestions"][0]
    assert "A clause that explicitly says why" in PRACTICE_GENERATION_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_new_p1_plan_survives_passage_validation_into_every_slot_prompt():
    request = _request(mode="COMPREHENSION", question_count=3,
                       easier=0, current=3, challenge=0, complexity_band=1)
    passage = "今日は雨です。友だちは傘を忘れました。私は大きな傘を持っています。"
    plans = [
        {"globalOrder": order, "skillTag": skill, "difficulty": "CURRENT",
         "complexityBand": 1, "clueQuote": clue, "questionFocus": focus,
         "unstatedInference": inference}
        for order, skill, clue, focus, inference in (
            (1, "CONTENT", "今日は雨です。", "今日の天気", None),
            (2, "DETAIL", "私は大きな傘を持っています。", "私の持ち物", None),
            (3, "INFERENCE", "友だちは傘を忘れました。", "友だちと私の行動",
             "二人は一緒に傘を使うかもしれません。"),
        )
    ]

    class PassageProvider:
        def __init__(self):
            self.schema = None
            self.prompt = None

        async def call(self, type_name, data, schema=None):
            self.schema, self.prompt = schema, data
            return {"passageId": "p1", "passageText": passage,
                    "inferenceClueQuote": "友だちは傘を忘れました。",
                    "unstatedInference": "二人は一緒に傘を使うかもしれません。",
                    "questionPlans": plans}

    provider = PassageProvider()
    service = ReadingVocabularyGenerationService(provider)
    result = await service._generate_reading_passages(request)
    slots = service._build_slots(request, result)
    assert [slot.reading_plan.global_order for slot in slots] == [1, 2, 3]
    assert [len(slot.same_passage_plans) for slot in slots] == [3, 3, 3]
    assert slots[2].prompt_payload()["currentQuestionPlan"]["unstatedInference"] == plans[2]["unstatedInference"]
    assert "questionPlans" in provider.schema["required"]
    assert [slot["globalOrder"] for slot in _practice_data(provider.prompt)["plannedQuestionSlots"]] == [1, 2, 3]
    candidate_provider = _CandidateRecorder()
    await ReadingVocabularyGenerationService(candidate_provider)._generate_candidate_batch(
        request, [slots[2]], {},
    )
    assert candidate_provider.payload is not None
    candidate_slot = candidate_provider.payload["candidateSlots"][0]
    assert candidate_slot["currentQuestionPlan"]["globalOrder"] == 3
    assert len(candidate_slot["samePassageQuestionPlans"]) == 3


class _CandidateRecorder:
    def __init__(self):
        self.payload: dict | None = None

    async def call(self, type_name, data, schema=None):
        self.payload = _practice_data(data)
        return {"questions": []}


@pytest.mark.parametrize(("offset", "position", "count"), [
    (0, 1, 3), (1, 2, 3), (2, 3, 3), (3, 1, 2), (4, 2, 2),
])
def test_structure_generation_reserves_distinct_same_passage_tasks(offset, position, count):
    request = _request(mode="STRUCTURE", question_count=1, easier=0, current=1, challenge=0)
    request = request.model_copy(update={
        "previous_questions": [_question(order=order) for order in range(1, offset + 1)],
    })
    slot = _QuestionSlot(
        order=1, difficulty=PracticeDifficulty.CURRENT, complexity_band=3,
        skill_tag="STRUCTURE", passage_id="p1" if offset < 3 else "p2",
        passage_text=PASSAGE, reading_mode="STRUCTURE",
    )
    provider = _CandidateRecorder()
    asyncio.run(ReadingVocabularyGenerationService(provider)
                ._generate_candidate_batch(request, [slot], {}))
    assert provider.payload is not None
    assert provider.payload["candidateSlots"][0]["samePassageTaskPosition"] == position
    assert provider.payload["candidateSlots"][0]["samePassageTaskCount"] == count
    assert "Do not use the first question to ask" in PRACTICE_GENERATION_SYSTEM_PROMPT


def test_explicit_conclusion_paraphrase_is_not_labeled_inference_by_contract():
    question = _question(prompt="予定が変わったことを筆者はどう受け止めていますか。")
    request = _request(mode="CONTEXT_INFERENCE", question_count=1,
                       easier=0, current=1, challenge=0)
    prompt = build_practice_verification_prompt(request, [{"order": 1}])
    assert "merely paraphrasing an explicitly stated fact or conclusion" in prompt
    outcome, _ = _outcome(_verdict(readingOperation="DIRECT_RETRIEVAL"), question=question)
    assert outcome.failures == {1: "inference question only requires direct retrieval"}
