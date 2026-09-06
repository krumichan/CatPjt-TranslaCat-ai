from app.features.language_learning.writing.prompts import (
    DAILY_WRITING_GENERATION_SYSTEM_PROMPT,
    WRITING_EVALUATION_SYSTEM_PROMPT,
    build_daily_writing_generation_prompt,
)
from app.features.language_learning.writing.service import LanguageLearningWritingService
from app.schemas.language_learning import (
    DailyWritingGenerationRequest,
    DailyWritingItem,
    WritingEvaluationRequest,
)


def _request(writing_type: str) -> DailyWritingGenerationRequest:
    return DailyWritingGenerationRequest.model_validate(
        {
            "requestId": f"mode-{writing_type.lower()}",
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "writingType": writing_type,
            "sentenceCount": 1,
            "difficultyDistribution": {"review": 0, "normal": 1, "challenge": 0},
            "selectedKeywords": [],
            "generationDate": "2026-09-06",
        }
    )


def _item(**updates) -> DailyWritingItem:
    payload = {
        "order": 1,
        "difficulty": "NORMAL",
        "originText": "친구에게 다음 주말 계획을 이야기해 보세요.",
        "keywords": [],
        "focusMetrics": ["MEANING", "NATURALNESS"],
        "focusReason": "의미 전달과 자연스러움 확인",
        "providedFacts": [],
        "requiredIntents": [],
        "responseConstraints": [],
    }
    payload.update(updates)
    return DailyWritingItem.model_validate(payload)


def test_generation_request_serializes_writing_type_in_camel_case():
    request = _request("TRANSLATION")

    payload = request.model_dump(mode="json", by_alias=True)

    assert payload["writingType"] == "TRANSLATION"
    assert '"writingType":"TRANSLATION"' in build_daily_writing_generation_prompt(request)


def test_guided_contract_requires_all_three_guidance_groups():
    request = _request("GUIDED")
    incomplete = _item(
        providedFacts=["회의는 금요일 오후 3시입니다."],
        requiredIntents=["참석 가능 여부를 정중하게 답한다"],
    )
    complete = incomplete.model_copy(
        update={"response_constraints": ["일본어로 2~3문장 작성"]}
    )

    assert (
        LanguageLearningWritingService._writing_type_contract_reason(request, incomplete)
        == "GUIDED_RESPONSE_CONSTRAINTS_MISSING"
    )
    assert LanguageLearningWritingService._writing_type_contract_reason(request, complete) is None


def test_translation_and_free_reject_hidden_guidance_payloads():
    item = _item(providedFacts=["숨겨진 사실"])

    assert (
        LanguageLearningWritingService._writing_type_contract_reason(
            _request("TRANSLATION"), item
        )
        == "TRANSLATION_GUIDANCE_MUST_BE_EMPTY"
    )
    assert (
        LanguageLearningWritingService._writing_type_contract_reason(_request("FREE"), item)
        == "FREE_GUIDANCE_MUST_BE_EMPTY"
    )


def test_free_mode_prompt_forbids_unknown_real_world_fact_assumptions():
    assert "recent system failure" in DAILY_WRITING_GENERATION_SYSTEM_PROMPT
    assert "explicitly hypothetical/imagined situation" in DAILY_WRITING_GENERATION_SYSTEM_PROMPT


def test_daily_evaluation_contract_keeps_detailed_metrics_and_mode_specific_meaning():
    request = WritingEvaluationRequest.model_validate(
        {
            "requestId": "daily-guided-eval",
            "context": "DAILY",
            "writingType": "GUIDED",
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "originSentence": "가이드를 바탕으로 답변하세요.",
            "userAnswer": "金曜日の会議に参加できます。",
            "difficulty": "NORMAL",
            "keywords": [],
            "focusMetrics": ["MEANING", "GRAMMAR", "NATURALNESS"],
            "providedFacts": ["회의는 금요일입니다."],
            "requiredIntents": ["참석 가능 여부를 답한다"],
            "responseConstraints": ["정중하게 작성한다"],
        }
    )

    assert request.writing_type.value == "GUIDED"
    assert "TRANSLATION: MEANING means semantic preservation" in WRITING_EVALUATION_SYSTEM_PROMPT
    assert "GUIDED: use providedFacts" in WRITING_EVALUATION_SYSTEM_PROMPT
    assert "FREE: MEANING means relevance" in WRITING_EVALUATION_SYSTEM_PROMPT
    for metric in ("MEANING", "GRAMMAR", "VOCABULARY", "NATURALNESS", "EXPRESSION"):
        assert metric in WRITING_EVALUATION_SYSTEM_PROMPT
