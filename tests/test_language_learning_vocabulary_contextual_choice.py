from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.features.language_learning.reading_vocabulary.contextual_choice_task import (
    CONTEXTUAL_CHOICE_ANSWER_KEYS,
    CONTEXTUAL_CHOICE_PLAN_VERSION,
    CONTEXTUAL_CHOICE_RECIPE_VERSION,
    CONTEXTUAL_CHOICE_SCENARIO_FAMILIES,
    contextual_choice_required_close_distractors,
    render_contextual_choice_template,
)
from app.features.language_learning.reading_vocabulary.prompts import (
    PRACTICE_CONTEXTUAL_CHOICE_PLAN_VERIFICATION_SYSTEM_PROMPT,
)
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_recipe import (
    ContextualChoiceDemand,
    vocabulary_difficulty_recipe,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_shadow import (
    VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME,
    vocabulary_blind_rubric_payload,
)
from app.schemas.language_learning_practice import (
    PersonalizedVocabularyPlan,
    PracticeGenerationRequest,
)


_ORDER_4_VERIFIER_FLIP_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "contextual-choice-order-4-lexical-verifier-flips-v1.json"
)


def _payload(data: str, tag: str = "practice-data") -> dict:
    if f"<{tag}>" in data:
        return json.loads(data.split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0])
    return json.loads(data.split("\n\n", 1)[1])


class ContextualChoiceV3Provider:
    def __init__(
        self,
        *,
        plan_duplicate: bool = False,
        plan_extra_order: bool = False,
        persistent_plan_duplicate: bool = False,
        plan_quality_failure: str | None = None,
        plan_quality_order: int = 1,
        persistent_plan_quality_failure: bool = False,
        plan_reason_code_override: str | None = None,
        lexical_repair_mutation: str | None = None,
        context_fallback_lexical_failure: bool = False,
        distractor_faults: dict[int, tuple[str, ...]] | None = None,
        revalidation_distractor_faults: dict[int, tuple[str, ...]] | None = None,
        persistent_distractor_faults: bool = False,
        target_viable: bool = True,
        natural_target: bool = True,
        origin_explanation_sequence: list[str] | None = None,
        origin_learning_explanation_sequence: list[str] | None = None,
        context_explanation: str | None = None,
        context_template_override: str | None = None,
        context_failure: str | None = None,
        context_failure_sequence: list[tuple[str, ...]] | None = None,
        persistent_context_failure: bool = False,
        marker_count: int = 1,
        context_contract_sequence_by_order: dict[int, list[str | None]] | None = None,
        context_verifier_fault: str | None = None,
    ) -> None:
        self.calls: Counter[str] = Counter()
        self.operations: Counter[str] = Counter()
        self.plan_generation_payloads: list[dict] = []
        self.generation_prompts: list[str] = []
        self.plan_validation_payloads: list[dict] = []
        self.plan_validation_prompts: list[str] = []
        self.plan_verdicts: list[dict] = []
        self.context_generation_payloads: list[dict] = []
        self.context_validation_payloads: list[dict] = []
        self.context_verdicts: list[dict] = []
        self.origin_payloads: list[dict] = []
        self.schemas: dict[str, list[dict]] = {}
        self.plan_duplicate = plan_duplicate
        self.plan_extra_order = plan_extra_order
        self.persistent_plan_duplicate = persistent_plan_duplicate
        self.plan_quality_failure = plan_quality_failure
        self.plan_quality_order = plan_quality_order
        self.plan_quality_hits = 0
        self.persistent_plan_quality_failure = persistent_plan_quality_failure
        self.plan_reason_code_override = plan_reason_code_override
        self.lexical_repair_mutation = lexical_repair_mutation
        self.context_fallback_lexical_failure = context_fallback_lexical_failure
        self.distractor_faults = distractor_faults or {}
        self.revalidation_distractor_faults = (
            revalidation_distractor_faults or {}
        )
        self.persistent_distractor_faults = persistent_distractor_faults
        self.target_viable = target_viable
        self.natural_target = natural_target
        self.origin_explanation_sequence = origin_explanation_sequence
        self.origin_learning_explanation_sequence = origin_learning_explanation_sequence
        self.context_explanation = context_explanation
        self.context_template_override = context_template_override
        self.context_failure = context_failure
        self.context_failure_sequence = context_failure_sequence
        self.persistent_context_failure = persistent_context_failure
        self.marker_count = marker_count
        self.context_contract_sequence_by_order = (
            context_contract_sequence_by_order or {}
        )
        self.context_generation_counts_by_order: Counter[int] = Counter()
        self.context_verifier_fault = context_verifier_fault

    async def call(self, type_name: str, data: str, schema: dict | None = None):
        self.calls[type_name] += 1
        self.schemas.setdefault(type_name, []).append(schema or {})
        if type_name == ReadingVocabularyGenerationService.TYPE_NAME:
            self.generation_prompts.append(data)
            payload = _payload(data)
            operation = payload["operation"]
            self.operations[operation] += 1
            if operation in {"PLAN", "PLAN_REPAIR"}:
                self.plan_generation_payloads.append(payload)
                return self._plan(payload)
            if operation == "LEXICAL_REPAIR":
                self.plan_generation_payloads.append(payload)
                return self._lexical_repair(payload)
            if operation in {"CONTEXT", "CONTEXT_REPAIR"}:
                self.context_generation_payloads.append(payload)
                context = self._context(payload)
                if self.context_explanation is not None:
                    context["explanationLearning"] = self.context_explanation
                return context
            raise AssertionError(f"unexpected generation operation: {operation}")
        if (
            type_name
            == ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME
        ):
            payload = _payload(data, "plan-data")
            self.plan_validation_prompts.append(data)
            self.plan_validation_payloads.append(payload)
            verdict = self._plan_verdicts(payload)
            self.plan_verdicts.append(verdict)
            return verdict
        if (
            type_name
            == ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME
        ):
            payload = _payload(data)
            self.context_validation_payloads.append(payload)
            verdict = self._context_verdict(payload)
            self.context_verdicts.append(verdict)
            return verdict
        if type_name in {
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME,
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME,
        }:
            payload = _payload(data)
            self.origin_payloads.append(payload)
            origin_attempt = (
                self.calls[ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME]
                + self.calls[
                    ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME
                ]
            )
            explanation = (
                self.origin_explanation_sequence[
                    min(origin_attempt - 1, len(self.origin_explanation_sequence) - 1)
                ]
                if self.origin_explanation_sequence
                else "문맥상 이 표현이 가장 자연스럽습니다."
            )
            learning_explanation = (
                self.origin_learning_explanation_sequence[
                    min(
                        origin_attempt - 1,
                        len(self.origin_learning_explanation_sequence) - 1,
                    )
                ]
                if self.origin_learning_explanation_sequence
                else "正答表現が文脈の条件に最も自然に合います。"
            )
            return {
                "explanations": [
                    {
                        "order": item["order"],
                        "text": explanation,
                        "explanationLearning": (
                            learning_explanation
                            if item.get("repairExplanationLearning") else None
                        ),
                    }
                    for item in payload["questions"]
                ]
            }
        if type_name == VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME:
            raise AssertionError("CONTEXTUAL_CHOICE V3 shadow must not block production")
        raise AssertionError(f"unexpected task: {type_name}")

    async def call_with_image(
        self, type_name: str, prompt: str, image_bytes: bytes,
        mime_type: str, schema: dict | None = None,
    ):
        raise AssertionError("CONTEXTUAL_CHOICE does not use image generation")

    def _plan(self, payload: dict) -> dict:
        repair = payload["operation"] == "PLAN_REPAIR"
        signals = payload["userSignals"]
        keyword = (
            signals["selectedKeywords"]
            or signals["weakSignals"]
            or signals["recentMistakes"]
            or [signals["learningProfileFallback"]]
        )[0]
        new_anchor_type = (
            "SELECTED_KEYWORD"
            if signals["selectedKeywords"]
            else "WEAK_SIGNAL"
            if signals["weakSignals"]
            else "RECENT_MISTAKE"
            if signals["recentMistakes"]
            else "LEARNING_PROFILE"
        )
        keyword_code = sum(ord(char) for char in keyword) % 997
        items = []
        for slot in payload["serverSlots"]:
            order = slot["globalOrder"]
            if slot["reviewTarget"]:
                target = None
                anchor_type = None
                anchor_value = None
            else:
                target_order = order
                if self.plan_duplicate and (not repair or self.persistent_plan_duplicate) and order == 2:
                    target_order = 1
                target = f"個別表現{keyword_code}-{target_order}"
                anchor_type = new_anchor_type
                anchor_value = keyword
            items.append(
                {
                    "globalOrder": order,
                    "targetExpression": target,
                    "distractors": [
                        f"関連候補{order}一",
                        f"関連候補{order}二",
                        f"関連候補{order}三",
                    ],
                    "anchorType": anchor_type,
                    "anchorValue": anchor_value,
                }
            )
        if self.plan_extra_order and not repair:
            items.append({**items[-1], "globalOrder": 11})
        return {"items": items}

    def _lexical_repair(self, payload: dict) -> dict:
        item = payload["currentPlanItem"]
        order = item["globalOrder"]
        target = None if payload["preserveTarget"] else item["targetExpression"] + "改"
        if self.lexical_repair_mutation == "review_target":
            target = "別の復習表現"
        if self.lexical_repair_mutation == "duplicate_plan_target":
            target = payload["otherCanonicalKeys"][0]
        replacements = [
            {"index": index, "text": f"修正候補{order}{index + 1}"}
            for index in payload["invalidDistractorIndexes"]
        ]
        repaired = {
            "globalOrder": order,
            "targetExpression": target,
            "distractorReplacements": replacements,
        }
        if self.lexical_repair_mutation == "anchor":
            repaired["anchorValue"] = "別のアンカー"
        if self.lexical_repair_mutation == "future_slot":
            repaired["globalOrder"] = order + 1
        if self.lexical_repair_mutation == "unexpected_index":
            repaired["distractorReplacements"].append(
                {"index": 2, "text": f"許可外候補{order}"}
            )
        if self.lexical_repair_mutation == "duplicate_index":
            repaired["distractorReplacements"].append(
                dict(repaired["distractorReplacements"][0])
            )
        if self.lexical_repair_mutation == "missing_index":
            repaired["distractorReplacements"] = []
        if replacements:
            replacement = replacements[0]
            replacement_index = replacement["index"]
            if self.lexical_repair_mutation == "target_equal":
                replacement["text"] = item["targetExpression"]
            elif self.lexical_repair_mutation in {
                "preserved_equal", "normalized_duplicate"
            }:
                preserved = next(
                    distractor
                    for index, distractor in enumerate(item["distractors"])
                    if index not in payload["invalidDistractorIndexes"]
                )
                replacement["text"] = (
                    preserved
                    if self.lexical_repair_mutation == "preserved_equal"
                    else f"{preserved}！"
                )
            elif self.lexical_repair_mutation == "unchanged":
                replacement["text"] = item["distractors"][replacement_index]
            elif self.lexical_repair_mutation == "empty":
                replacement["text"] = "   "
            elif self.lexical_repair_mutation == "wrong_language":
                replacement["text"] = "잘못된 표현"
            elif self.lexical_repair_mutation == "known_good_register":
                replacement["text"] = "ご不便をおかけして申し訳ございません"
        return repaired

    def _plan_verdicts(self, payload: dict) -> dict:
        verdicts = []
        for order in [payload["currentOrder"]]:
            fail = (
                self.plan_quality_failure
                and order == self.plan_quality_order
                and (self.plan_quality_hits == 0 or self.persistent_plan_quality_failure)
            )
            if self.context_fallback_lexical_failure and self.operations["LEXICAL_REPAIR"]:
                fail = True
            if self.plan_quality_failure and order == self.plan_quality_order:
                self.plan_quality_hits += 1
            values: dict[str, Any] = {
                "targetExpressionWellFormed": (
                    self.natural_target
                    if len(self.plan_validation_payloads) == 1 else True
                ),
                "learningValue": True,
                "sameSurfaceCategory": True,
                "skillContrastSupported": True,
                "coveredDecisiveDimensions": list(
                    payload["structuralDifficultyDemand"]["decisiveDimensions"]
                ),
                "definitionOnly": False,
                "lexicalConceptRepeated": False,
                "rareOrTrivia": False,
                "targetViableWithDifferentDistractors": (
                    self.target_viable if len(self.plan_validation_payloads) == 1 else True
                ),
            }
            apply_faults = (
                len(self.plan_validation_payloads) == 1
                or self.persistent_distractor_faults
                or bool(self.revalidation_distractor_faults)
            )
            active_distractor_faults = (
                self.revalidation_distractor_faults
                if len(self.plan_validation_payloads) > 1
                and self.revalidation_distractor_faults
                else self.distractor_faults
            )
            values["distractors"] = [
                {
                    "index": index,
                    "expressionWellFormed": "unnatural" not in faults,
                    "sameSurfaceCategory": "surface" not in faults,
                    "bundleRelevant": "plausible" not in faults,
                    "closeCompetitor": "competitor" not in faults,
                    "malformedByGrammar": "grammar" in faults,
                    "skillContrastRelevant": "skill" not in faults,
                    "coveredDecisiveDimensions": (
                        []
                        if "skill" in faults or "dimension" in faults
                        else list(
                            payload["structuralDifficultyDemand"][
                                "decisiveDimensions"
                            ]
                        )
                    ),
                }
                for index in range(3)
                for faults in [
                    active_distractor_faults.get(index, ()) if apply_faults else ()
                ]
            ]
            if fail:
                if self.context_fallback_lexical_failure and self.operations["LEXICAL_REPAIR"]:
                    values["sameSurfaceCategory"] = False
                elif self.plan_quality_failure == "lexical_duplicate":
                    values["lexicalConceptRepeated"] = True
                elif self.plan_quality_failure == "competitors":
                    values["distractors"][1]["closeCompetitor"] = False
                    values["distractors"][2]["closeCompetitor"] = False
                elif self.plan_quality_failure == "surface":
                    values["sameSurfaceCategory"] = False
                elif self.plan_quality_failure == "confusion":
                    values["distractors"][1]["bundleRelevant"] = False
                elif self.plan_quality_failure == "skill_bundle":
                    values["skillContrastSupported"] = False
                    values["coveredDecisiveDimensions"] = []
                    for distractor in values["distractors"]:
                        distractor["skillContrastRelevant"] = False
                        distractor["coveredDecisiveDimensions"] = []
            reason_code = self.plan_reason_code_override or "PASS"
            if not values["sameSurfaceCategory"]:
                reason_code = "SAME_SURFACE_CATEGORY"
            elif not values["skillContrastSupported"]:
                reason_code = "SKILL_FIT"
            elif any(
                not item["bundleRelevant"]
                for item in values["distractors"]
            ):
                reason_code = "DISTRACTOR_NOT_PLAUSIBLE"
            elif sum(
                bool(item["closeCompetitor"])
                for item in values["distractors"]
            ) < 2:
                reason_code = "INSUFFICIENT_CLOSE_DISTRACTORS"
            elif values["lexicalConceptRepeated"]:
                reason_code = "LEXICAL_REPEAT"
            if self.plan_reason_code_override is not None:
                reason_code = self.plan_reason_code_override
            # The verifier wire contract binds verdicts to server-owned IDs.
            values["distractors"] = {
                f"D{distractor['index']}": {
                    "id": f"D{distractor['index']}",
                    **{key: value for key, value in distractor.items() if key != "index"},
                }
                for distractor in values["distractors"]
            }
            verdicts.append({
                "globalOrder": order, "targetId": "TARGET", **values, "reasonCode": reason_code,
                "reason": "fixture plan verdict",
            })
        return {"verdicts": verdicts}

    def _context(self, payload: dict) -> dict:
        item = payload["fixedLexicalBundle"]
        order = item["globalOrder"]
        self.context_generation_counts_by_order[order] += 1
        generation_round = self.context_generation_counts_by_order[order]
        sequence = self.context_contract_sequence_by_order.get(order, [])
        contract_fault = (
            sequence[generation_round - 1]
            if generation_round <= len(sequence)
            else None
        )
        if contract_fault == "response_schema":
            return {"globalOrder": order, "contextTemplate": 123}
        if contract_fault == "wrong_order":
            return {
                "globalOrder": order + 1,
                "contextTemplate": "会議で{{TARGET}}ことにしました。",
                "explanationLearning": "文脈に合う表現です。",
            }
        if contract_fault == "missing_marker":
            template = "会議ではこの方針で進めることにしました。"
        elif contract_fault == "duplicate_marker":
            template = "{{TARGET}}を確認して{{TARGET}}ことにしました。"
        elif contract_fault == "unknown_marker":
            template = "{{OTHER}}を確認し、{{TARGET}}ことにしました。"
        elif contract_fault == "language":
            template = "회의에서 {{TARGET}}하기로 했습니다."
        elif contract_fault == "target_leak":
            template = (
                f"{item['targetExpression']}を確認し、{{{{TARGET}}}}ことにしました。"
            )
        elif contract_fault == "round_trip":
            template = "会議で______を確認し、{{TARGET}}ことにしました。"
        else:
            template = None
        if template is not None or contract_fault == "explanation":
            return {
                "globalOrder": order,
                "contextTemplate": template or "会議で{{TARGET}}ことにしました。",
                "explanationLearning": (
                    "" if contract_fault == "explanation" else "文脈に合う表現です。"
                ),
            }
        if self.context_template_override is not None:
            return {
                "globalOrder": item["globalOrder"],
                "contextTemplate": self.context_template_override,
                "explanationLearning": "文脈の条件に合う表現です。",
            }
        if payload["operation"] == "CONTEXT" and self.operations["CONTEXT"] > 1:
            template = "新しい文脈では、{{TARGET}}ことにしました。"
            return {
                "globalOrder": item["globalOrder"],
                "contextTemplate": template,
                "explanationLearning": "新しい文脈の条件に合う表現です。",
            }
        marker_count = self.marker_count if payload["operation"] == "CONTEXT" else 1
        if marker_count == 0:
            template = "会議ではこの方針で進めることにしました。"
        elif marker_count == 2:
            template = "{{TARGET}}を確認して{{TARGET}}ことにしました。"
        else:
            template = "会議で状況を確認し、{{TARGET}}ことにしました。"
        return {
            "globalOrder": item["globalOrder"],
            "contextTemplate": template,
            "explanationLearning": "文脈の条件に最も自然に合う表現です。",
        }

    def _context_verdict(self, payload: dict) -> dict:
        question = payload["question"]
        if self.context_verifier_fault == "malformed":
            return {"verdicts": [{"order": question["order"]}]}
        if self.context_verifier_fault == "coverage":
            return {"verdicts": [{
                "order": question["order"] + 1,
                "bestAnswerKey": "A",
                "ambiguous": False,
                "supported": True,
                "skillFit": True,
                "definitionLike": False,
                "structurallyWellFormedKeys": list(CONTEXTUAL_CHOICE_ANSWER_KEYS),
                "reason": "fixture context verdict",
            }]}
        validation_round = len(self.context_validation_payloads)
        fail = self.context_failure and (
            validation_round == 1 or self.persistent_context_failure
        )
        expected = CONTEXTUAL_CHOICE_ANSWER_KEYS[(question["order"] - 1) % 4]
        verdict = {
            "order": question["order"],
            "bestAnswerKey": expected,
            "ambiguous": False,
            "supported": True,
            "skillFit": True,
            "definitionLike": False,
            "structurallyWellFormedKeys": list(CONTEXTUAL_CHOICE_ANSWER_KEYS),
            "reason": "fixture context verdict",
        }
        failures = (
            self.context_failure_sequence[validation_round - 1]
            if self.context_failure_sequence is not None
            and validation_round <= len(self.context_failure_sequence)
            else (self.context_failure,) if fail else ()
        )
        for failure in failures:
            if failure == "answer_mismatch":
                verdict["bestAnswerKey"] = CONTEXTUAL_CHOICE_ANSWER_KEYS[
                    CONTEXTUAL_CHOICE_ANSWER_KEYS.index(expected) - 1
                ]
            elif failure == "ambiguous":
                verdict["ambiguous"] = True
            elif failure == "unsupported":
                verdict["supported"] = False
            elif failure == "skill":
                verdict["skillFit"] = False
            elif failure == "definition":
                verdict["definitionLike"] = True
            elif failure == "grammar":
                verdict["structurallyWellFormedKeys"] = [
                    key for key in CONTEXTUAL_CHOICE_ANSWER_KEYS if key != expected
                ]
            elif failure == "grammar_wrong":
                verdict["structurallyWellFormedKeys"] = [
                    key for key in CONTEXTUAL_CHOICE_ANSWER_KEYS if key != "C"
                ]
            elif failure == "structural":
                verdict["structurallyWellFormedKeys"] = [
                    key for key in CONTEXTUAL_CHOICE_ANSWER_KEYS if key != expected
                ]
        if self.context_verifier_fault == "unknown_key":
            verdict["structurallyWellFormedKeys"] = ["A", "B", "C", "Z"]
        if self.context_verifier_fault == "unknown_answer":
            verdict["bestAnswerKey"] = "Z"
        return {"verdicts": [verdict]}


def _review_targets() -> list[dict]:
    return [
        {
            "canonicalKey": "再確認する",
            "expression": "再確認する",
            "preferredSkill": "MEANING",
        },
        {
            "canonicalKey": "慎重に進める",
            "expression": "慎重に進める",
            "preferredSkill": "NUANCE",
        },
    ]


def _request(
    *,
    request_id: str = "contextual-v3-1",
    question_count: int = 1,
    complexity_band: int = 4,
    selected_keywords: list[str] | None = None,
    weak_signals: list[str] | None = None,
    recent_mistakes: list[str] | None = None,
    review_targets: list[dict] | None = None,
    review_question_count: int = 0,
    previous_questions: list[dict] | None = None,
    vocabulary_plan: dict | None = None,
    vocabulary_plan_only: bool = False,
) -> PracticeGenerationRequest:
    previous = previous_questions or []
    global_order = len(previous) + 1
    daily_difficulties = (
        "CURRENT",
        "EASIER",
        "CURRENT",
        "CHALLENGE",
        "CURRENT",
        "EASIER",
        "CURRENT",
        "CHALLENGE",
        "CURRENT",
        "CURRENT",
    )
    difficulty = daily_difficulties[global_order - 1]
    return PracticeGenerationRequest.model_validate(
        {
            "requestId": request_id,
            "domain": "VOCABULARY",
            "mode": "CONTEXTUAL_CHOICE",
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "questionCount": question_count,
            "complexityBand": complexity_band,
            "easierCount": 2 if question_count == 10 else int(difficulty == "EASIER"),
            "currentCount": 6 if question_count == 10 else int(difficulty == "CURRENT"),
            "challengeCount": 2 if question_count == 10 else int(difficulty == "CHALLENGE"),
            "selectedKeywords": ["협업"] if selected_keywords is None else selected_keywords,
            "weakSignals": ["회의 표현"] if weak_signals is None else weak_signals,
            "recentMistakes": (
                ["REGISTER:표현"] if recent_mistakes is None else recent_mistakes
            ),
            "reviewTargets": review_targets or [],
            "reviewQuestionCount": review_question_count,
            "generationDate": "2026-09-14",
            "previousQuestions": previous,
            "vocabularyPlan": vocabulary_plan,
            "vocabularyPlanOnly": vocabulary_plan_only,
        }
    )


async def _candidate_request_at_order(
    order: int, target: str, distractors: list[str]
) -> PracticeGenerationRequest:
    baseline = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(question_count=10)
    )
    assert baseline.vocabulary_plan is not None
    plan = baseline.vocabulary_plan.model_dump(mode="json", by_alias=True)
    plan["items"][order - 1]["targetExpression"] = target
    plan["items"][order - 1]["canonicalKey"] = target
    plan["items"][order - 1]["distractors"] = distractors
    return _request(
        previous_questions=[
            question.model_copy(update={"order": index}).model_dump(
                mode="json", by_alias=True
            )
            for index, question in enumerate(baseline.questions[: order - 1], start=1)
        ],
        vocabulary_plan=plan,
    )


@pytest.mark.asyncio
async def test_plan_uses_personalization_and_tracks_real_signal_anchor():
    provider = ContextualChoiceV3Provider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(
            selected_keywords=["deployment"],
            weak_signals=["incident response"],
            recent_mistakes=["REGISTER:handoff"],
        )
    )
    generation = provider.plan_generation_payloads[0]
    assert generation["userSignals"] == {
        "selectedKeywords": ["deployment"],
        "weakSignals": ["incident response"],
        "recentMistakes": ["REGISTER:handoff"],
        "learningProfileFallback": "ja",
    }
    assert response.vocabulary_plan is not None
    anchors = {
        "SELECTED_KEYWORD": "selectedKeywords",
        "WEAK_SIGNAL": "weakSignals",
        "RECENT_MISTAKE": "recentMistakes",
    }
    for item in response.vocabulary_plan.items:
        if item.review_target:
            continue
        assert item.anchor_type is not None
        assert item.anchor_value in generation["userSignals"][anchors[item.anchor_type.value]]


@pytest.mark.asyncio
async def test_different_user_keywords_can_produce_different_daily_targets():
    first = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(selected_keywords=["travel"])
    )
    second = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(selected_keywords=["deployment"])
    )
    assert first.vocabulary_plan is not None
    assert second.vocabulary_plan is not None
    assert first.vocabulary_plan.items[0].target_expression != second.vocabulary_plan.items[0].target_expression


@pytest.mark.asyncio
async def test_empty_signal_profile_uses_explicit_learning_profile_fallback():
    response = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(selected_keywords=[], weak_signals=[], recent_mistakes=[])
    )
    assert response.vocabulary_plan is not None
    assert all(
        item.anchor_type is None or item.anchor_type.value == "LEARNING_PROFILE"
        for item in response.vocabulary_plan.items
    )
    assert all(
        item.anchor_value is None or item.anchor_value == "ja"
        for item in response.vocabulary_plan.items
    )


@pytest.mark.asyncio
async def test_review_identity_and_preferred_skill_are_application_owned():
    reviews = _review_targets()
    response = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(review_targets=reviews, review_question_count=1)
    )
    assert response.vocabulary_plan is not None
    first, second = response.vocabulary_plan.items[:2]
    assert (first.target_expression, first.canonical_key, first.skill_tag) == (
        "再確認する",
        "再確認する",
        "MEANING",
    )
    assert (second.target_expression, second.canonical_key, second.skill_tag) == (
        "慎重に進める",
        "慎重に進める",
        "NUANCE",
    )


@pytest.mark.asyncio
async def test_daily_plan_has_fixed_difficulty_skill_and_scenario_shape():
    response = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request()
    )
    assert response.prompt_version == CONTEXTUAL_CHOICE_RECIPE_VERSION
    assert response.vocabulary_plan is not None
    plan = response.vocabulary_plan
    assert plan.version == CONTEXTUAL_CHOICE_PLAN_VERSION
    assert Counter(item.difficulty.value for item in plan.items) == {
        "EASIER": 2,
        "CURRENT": 6,
        "CHALLENGE": 2,
    }
    assert Counter(item.skill_tag for item in plan.items) == {
        "MEANING": 2,
        "COLLOCATION": 2,
        "NUANCE": 2,
        "REGISTER": 2,
        "PRAGMATIC_FIT": 2,
    }
    wire_plan = response.model_dump(mode="json", by_alias=True)["vocabularyPlan"]
    assert wire_plan["version"] == CONTEXTUAL_CHOICE_PLAN_VERSION
    assert set(wire_plan["items"][0]) == {
        "globalOrder",
        "reviewTarget",
        "targetExpression",
        "canonicalKey",
        "distractors",
        "skillTag",
        "difficulty",
        "complexityBand",
        "scenarioFamily",
        "anchorType",
        "anchorValue",
    }
    assert [item.scenario_family for item in plan.items] == list(
        CONTEXTUAL_CHOICE_SCENARIO_FAMILIES
    )


@pytest.mark.asyncio
async def test_plan_only_phase_returns_durable_contract_without_generating_context():
    provider = ContextualChoiceV3Provider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(vocabulary_plan_only=True)
    )

    assert response.questions == []
    assert response.vocabulary_plan is not None
    assert provider.operations == Counter({"PLAN": 1})
    assert provider.calls[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME
    ] == 0
    assert provider.operations["CONTEXT"] == 0


@pytest.mark.asyncio
async def test_future_lexical_quality_cannot_block_first_question():
    provider = ContextualChoiceV3Provider(
        plan_quality_failure="surface", plan_quality_order=7,
    )
    service = ReadingVocabularyGenerationService(provider)
    snapshot = await service.generate(_request(vocabulary_plan_only=True))
    assert snapshot.vocabulary_plan is not None
    assert provider.plan_validation_payloads == []
    first = await service.generate(_request(
        vocabulary_plan=snapshot.vocabulary_plan.model_dump(mode="json", by_alias=True),
    ))
    assert len(first.questions) == 1
    assert [payload["currentOrder"] for payload in provider.plan_validation_payloads] == [1]
    assert provider.operations["LEXICAL_REPAIR"] == 0


@pytest.mark.asyncio
async def test_progressive_order_five_validates_only_current_lexical_slot():
    full = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(question_count=10)
    )
    assert full.vocabulary_plan is not None
    provider = ContextualChoiceV3Provider()
    fifth = await ReadingVocabularyGenerationService(provider).generate(_request(
        previous_questions=[question.model_dump(mode="json", by_alias=True)
                            for question in full.questions[:4]],
        vocabulary_plan=full.vocabulary_plan.model_dump(mode="json", by_alias=True),
    ))
    assert len(fifth.questions) == 1
    assert [payload["currentOrder"] for payload in provider.plan_validation_payloads] == [5]
    assert "planItems" not in provider.plan_validation_payloads[0]
    assert "sameDayCanonicalKeys" not in provider.plan_validation_payloads[0]
    assert len(provider.plan_validation_payloads[0]["previousLexicalIdentities"]) == 4


@pytest.mark.asyncio
async def test_current_new_slot_repair_changes_only_lexical_bundle():
    snapshot = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(vocabulary_plan_only=True)
    )
    assert snapshot.vocabulary_plan is not None
    provider = ContextualChoiceV3Provider(plan_quality_failure="surface")
    response = await ReadingVocabularyGenerationService(provider).generate(_request(
        vocabulary_plan=snapshot.vocabulary_plan.model_dump(mode="json", by_alias=True),
    ))
    assert response.vocabulary_plan is not None
    before, after = snapshot.vocabulary_plan.items[0], response.vocabulary_plan.items[0]
    assert before.target_expression == after.target_expression
    assert before.distractors != after.distractors
    assert (before.skill_tag, before.difficulty, before.complexity_band,
            before.scenario_family, before.anchor_type, before.anchor_value) == (
            after.skill_tag, after.difficulty, after.complexity_band,
            after.scenario_family, after.anchor_type, after.anchor_value)
    assert response.vocabulary_plan.items[1:] == snapshot.vocabulary_plan.items[1:]
    assert response.questions[0].target_expression == after.target_expression
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 2
    assert provider.operations["CONTEXT"] == 1
    assert sum(provider.calls.values()) == 6  # plus one persisted plan-generation call


@pytest.mark.asyncio
async def test_current_review_repair_preserves_bound_target_and_canonical():
    reviews = _review_targets()
    snapshot = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(review_targets=reviews, review_question_count=1, vocabulary_plan_only=True)
    )
    assert snapshot.vocabulary_plan is not None
    provider = ContextualChoiceV3Provider(plan_quality_failure="surface")
    response = await ReadingVocabularyGenerationService(provider).generate(_request(
        review_targets=reviews, review_question_count=1,
        vocabulary_plan=snapshot.vocabulary_plan.model_dump(mode="json", by_alias=True),
    ))
    assert response.vocabulary_plan is not None
    before, after = snapshot.vocabulary_plan.items[0], response.vocabulary_plan.items[0]
    assert (after.target_expression, after.canonical_key, after.review_target) == (
        before.target_expression, before.canonical_key, True)
    assert after.distractors != before.distractors


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("review_target", "LEXICAL_REPAIR_TARGET_AUTHORITY_VIOLATION"),
        ("anchor", "LEXICAL_REPAIR_RESPONSE_SCHEMA_INVALID"),
        ("future_slot", "LEXICAL_REPAIR_ORDER_MISMATCH"),
        ("duplicate_plan_target", "LEXICAL_REPAIR_PLAN_AUTHORITY_VIOLATION"),
    ],
)
async def test_lexical_repair_cannot_escape_current_slot_authority(
    mutation, expected_code, caplog,
):
    caplog.set_level("INFO", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    reviews = _review_targets() if mutation == "review_target" else []
    snapshot = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(review_targets=reviews, review_question_count=int(bool(reviews)),
                 vocabulary_plan_only=True)
    )
    assert snapshot.vocabulary_plan is not None
    provider = ContextualChoiceV3Provider(
        plan_quality_failure=(
            None if mutation == "duplicate_plan_target" else "surface"
        ),
        natural_target=mutation != "duplicate_plan_target",
        target_viable=mutation != "duplicate_plan_target",
        lexical_repair_mutation=mutation,
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request(
            review_targets=reviews, review_question_count=int(bool(reviews)),
            vocabulary_plan=snapshot.vocabulary_plan.model_dump(mode="json", by_alias=True),
        ))
    assert caught.value.status_code == 422
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.operations["CONTEXT"] == 0
    assert expected_code in caplog.text
    assert "reason_codes=['STRUCTURAL_DISTRACTOR']" not in caplog.text


@pytest.mark.asyncio
async def test_persistent_current_lexical_failure_is_422_after_one_repair(caplog):
    caplog.set_level("INFO", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    provider = ContextualChoiceV3Provider(
        plan_quality_failure="surface", persistent_plan_quality_failure=True,
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    detail = cast(dict[str, Any], caught.value.detail)
    assert isinstance(detail, dict)
    assert detail["stage"] == "LEXICAL_VALIDATION"
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 2
    assert provider.operations["CONTEXT"] == 0
    text = caplog.text
    assert "reason_codes=['SAME_SURFACE_CATEGORY']" in text
    assert "phase=repair" in text
    assert "협업" not in text


@pytest.mark.asyncio
async def test_exact_or_canonical_plan_duplicate_is_repaired_once():
    provider = ContextualChoiceV3Provider(plan_duplicate=True)
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert response.vocabulary_plan is not None
    assert provider.operations["PLAN"] == 1
    assert provider.operations["PLAN_REPAIR"] == 1
    assert len({item.canonical_key for item in response.vocabulary_plan.items}) == 10


@pytest.mark.asyncio
async def test_structural_plan_repair_is_targeted_without_full_semantic_mini():
    provider = ContextualChoiceV3Provider(plan_duplicate=True)
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(vocabulary_plan_only=True)
    )
    assert response.questions == []
    assert provider.operations["PLAN_REPAIR"] == 1
    assert provider.plan_validation_payloads == []
    assert sorted(slot["globalOrder"] for slot in provider.plan_generation_payloads[1]["serverSlots"]) == [1, 2]


@pytest.mark.asyncio
async def test_plan_with_unowned_extra_order_is_rejected_before_persistence():
    provider = ContextualChoiceV3Provider(plan_extra_order=True)
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(
            _request(vocabulary_plan_only=True)
        )
    assert caught.value.status_code == 422
    detail = cast(dict[str, Any], caught.value.detail)
    assert isinstance(detail, dict)
    assert detail["stage"] == "PLAN_STRUCTURE"
    assert provider.plan_validation_payloads == []
    assert provider.operations["PLAN_REPAIR"] == 0


@pytest.mark.asyncio
async def test_persistent_plan_duplicate_is_terminal_after_one_repair():
    provider = ContextualChoiceV3Provider(
        plan_duplicate=True,
        persistent_plan_duplicate=True,
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    assert provider.operations["PLAN_REPAIR"] == 1


@pytest.mark.asyncio
async def test_lexical_repeat_repairs_but_scenario_similarity_is_not_a_gate():
    repeated = ContextualChoiceV3Provider(plan_quality_failure="lexical_duplicate")
    await ReadingVocabularyGenerationService(repeated).generate(_request())
    assert repeated.operations["LEXICAL_REPAIR"] == 1
    ordinary = ContextualChoiceV3Provider()
    await ReadingVocabularyGenerationService(ordinary).generate(_request())
    assert "scenarioSimilarity" not in json.dumps(ordinary.plan_validation_payloads[0])
    assert ordinary.operations["LEXICAL_REPAIR"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("complexity_band", [4, 5])
async def test_b4_b5_keep_current_slot_genuine_competitor_requirement(complexity_band):
    provider = ContextualChoiceV3Provider(plan_quality_failure="competitors")
    await ReadingVocabularyGenerationService(provider).generate(
        _request(complexity_band=complexity_band)
    )
    assert provider.operations["LEXICAL_REPAIR"] == 1


@pytest.mark.asyncio
async def test_plan_is_generated_once_then_reused_after_reconstruction():
    first_provider = ContextualChoiceV3Provider()
    first_response = await ReadingVocabularyGenerationService(first_provider).generate(
        _request()
    )
    assert first_response.vocabulary_plan is not None
    previous = [
        first_response.questions[0].model_copy(update={"order": 1}).model_dump(
            mode="json", by_alias=True
        )
    ]
    persisted_plan = json.loads(
        json.dumps(
            first_response.vocabulary_plan.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
        )
    )
    retry_provider = ContextualChoiceV3Provider()
    retry_request = _request(
        request_id="contextual-v3-2",
        previous_questions=previous,
        vocabulary_plan=persisted_plan,
    )
    retry_response = await ReadingVocabularyGenerationService(retry_provider).generate(
        retry_request
    )
    assert retry_provider.operations["PLAN"] == 0
    assert retry_provider.operations["PLAN_REPAIR"] == 0
    assert retry_response.vocabulary_plan == first_response.vocabulary_plan
    assert retry_provider.context_generation_payloads[0]["fixedLexicalBundle"][
        "targetExpression"
    ] == first_response.vocabulary_plan.items[1].target_expression


@pytest.mark.asyncio
async def test_persisted_plan_cannot_disagree_with_accepted_prefix():
    first = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request()
    )
    assert first.vocabulary_plan is not None
    corrupted = first.vocabulary_plan.model_dump(mode="json", by_alias=True)
    corrupted["items"][0]["targetExpression"] = "別の個別表現"
    corrupted["items"][0]["canonicalKey"] = "別の個別表現"
    provider = ContextualChoiceV3Provider()
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request(
            previous_questions=[first.questions[0].model_copy(update={"order": 1}).model_dump(
                mode="json", by_alias=True
            )],
            vocabulary_plan=corrupted,
        ))
    assert caught.value.status_code == 422
    detail = cast(dict[str, Any], caught.value.detail)
    assert isinstance(detail, dict)
    assert detail["code"] == "AI_SCHEMA_INVALID"
    assert detail["stage"] == "PLAN_AUTHORITY"
    assert sum(provider.calls.values()) == 0


@pytest.mark.parametrize("template", ["対象マーカーなし", "{{TARGET}}と{{TARGET}}"])
def test_context_template_requires_exactly_one_target_marker(template):
    with pytest.raises(ValueError, match="exactly once"):
        render_contextual_choice_template(template, "再確認する")


def test_context_template_deterministically_builds_blank_and_complete_sentence():
    complete, prompt = render_contextual_choice_template(
        "内容を{{TARGET}}ことにした。",
        "再確認する",
    )
    assert complete == "内容を再確認することにした。"
    assert prompt == "内容を______ことにした。"
    assert prompt.replace("______", "再確認する") == complete


@pytest.mark.asyncio
@pytest.mark.parametrize("marker_count", [0, 2])
async def test_bad_marker_is_repaired_once_without_changing_lexical_bundle(marker_count):
    provider = ContextualChoiceV3Provider(marker_count=marker_count)
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 1
    fixed = [payload["fixedLexicalBundle"] for payload in provider.context_generation_payloads]
    assert fixed[0] == fixed[1]
    assert response.questions[0].target_expression == fixed[0]["targetExpression"]


@pytest.mark.asyncio
async def test_context_schema_cannot_change_target_distractors_or_answer():
    provider = ContextualChoiceV3Provider()
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    generation_payload = provider.context_generation_payloads[0]
    assert generation_payload["operation"] == "CONTEXT"
    assert "repairClass" not in generation_payload
    assert "reasonCodes" not in generation_payload
    assert "repairDirective" not in generation_payload
    schema = provider.schemas[ReadingVocabularyGenerationService.TYPE_NAME][-1]
    assert set(schema["properties"]) == {
        "globalOrder",
        "contextTemplate",
        "explanationLearning",
    }
    assert response.vocabulary_plan is not None
    item = response.vocabulary_plan.items[0]
    question = response.questions[0]
    assert {option.text for option in question.options} == {
        item.target_expression,
        *item.distractors,
    }


@pytest.mark.asyncio
async def test_context_verifier_hides_answer_target_canonical_and_band():
    provider = ContextualChoiceV3Provider()
    await ReadingVocabularyGenerationService(provider).generate(_request())
    payload = provider.context_validation_payloads[0]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert set(payload["question"]) == {
        "order",
        "prompt",
        "options",
        "renderedOptions",
        "skillTag",
        "reviewTarget",
    }
    for hidden in (
        "correctAnswer",
        "targetExpression",
        "canonicalKey",
        "complexityBand",
        "requestedBand",
        "observedBand",
    ):
        assert hidden not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["answer_mismatch", "ambiguous", "unsupported", "skill", "definition", "grammar"],
)
async def test_context_quality_hard_gates_repair_only_context(failure):
    provider = ContextualChoiceV3Provider(context_failure=failure)
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert len(response.questions) == 1
    assert provider.operations["PLAN"] == 1
    assert provider.operations["PLAN_REPAIR"] == 0
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 1


@pytest.mark.asyncio
async def test_persistent_context_failure_stops_after_bounded_lexical_fallback():
    provider = ContextualChoiceV3Provider(
        context_failure="ambiguous",
        persistent_context_failure=True,
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    assert provider.operations["CONTEXT"] == 2
    assert provider.operations["CONTEXT_REPAIR"] == 1
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.context_validation_payloads) == 3


@pytest.mark.asyncio
async def test_initial_ambiguity_repaired_by_context_does_not_touch_lexical_bundle():
    provider = ContextualChoiceV3Provider(context_failure="ambiguous")
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert len(response.questions) == 1
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 1
    assert len(provider.context_validation_payloads) == 2
    assert provider.context_verdicts[0]["verdicts"][0][
        "structurallyWellFormedKeys"
    ] == list(CONTEXTUAL_CHOICE_ANSWER_KEYS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("ambiguous", "AMBIGUOUS"),
        ("answer_mismatch", "BEST_ANSWER_MISMATCH"),
        ("skill", "SKILL_FIT"),
    ],
)
async def test_bundle_suspect_uses_one_lexical_fallback_then_fresh_context(
    failure, expected_code,
):
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[(failure,), (failure,), ()],
    )
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert len(response.questions) == 1
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.operations["CONTEXT"] == 2
    assert provider.operations["CONTEXT_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 2
    assert len(provider.context_validation_payloads) == 3
    assert provider.plan_generation_payloads[1]["repairTrigger"] == "CONTEXT_VALIDATION"
    assert provider.plan_generation_payloads[1]["reasonCodes"] == [expected_code]
    old_bundle, repair_bundle, fresh_bundle = (
        item["fixedLexicalBundle"] for item in provider.context_generation_payloads
    )
    assert old_bundle == repair_bundle
    assert old_bundle != fresh_bundle
    assert response.questions[0].prompt.startswith("新しい文脈では")
    assert response.vocabulary_plan is not None
    revised = response.vocabulary_plan.items[0]
    assert {option.text for option in response.questions[0].options} == {
        revised.target_expression, *revised.distractors,
    }
    assert sum(provider.calls.values()) == 11  # plan, lexical x2, repair, context x3, Mini x3, origin


@pytest.mark.asyncio
async def test_nonpersistent_skill_fit_does_not_use_lexical_fallback():
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("skill",), ("unsupported",)],
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 1


@pytest.mark.asyncio
async def test_context_local_reason_vetoes_mixed_bundle_suspect_verdict():
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("ambiguous",), ("ambiguous", "unsupported")],
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert provider.operations["CONTEXT"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unsupported", "definition"])
async def test_context_local_failure_never_mutates_lexical_bundle(failure):
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[(failure,), (failure,)],
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    assert cast(dict[str, Any], caught.value.detail)["stage"] == "CONTEXT_VALIDATION"
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 1


@pytest.mark.asyncio
async def test_jit_lexical_repair_consumes_the_only_mutation_budget():
    provider = ContextualChoiceV3Provider(
        plan_quality_failure="surface",
        context_failure_sequence=[("ambiguous",), ("ambiguous",)],
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.plan_generation_payloads[1]["repairTrigger"] == "LEXICAL_VALIDATION"
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 1
    assert len(provider.context_validation_payloads) == 2


@pytest.mark.asyncio
async def test_context_fallback_lexical_mini_failure_is_terminal():
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("ambiguous",), ("ambiguous",)],
        context_fallback_lexical_failure=True,
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    assert cast(dict[str, Any], caught.value.detail)["stage"] == "LEXICAL_VALIDATION"
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 1


@pytest.mark.asyncio
async def test_fresh_context_failure_does_not_start_another_repair_round():
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("ambiguous",), ("ambiguous",), ("ambiguous",)],
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    assert cast(dict[str, Any], caught.value.detail)["stage"] == "CONTEXT_VALIDATION"
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.operations["CONTEXT"] == 2
    assert provider.operations["CONTEXT_REPAIR"] == 1
    assert len(provider.context_validation_payloads) == 3


@pytest.mark.asyncio
async def test_context_fallback_review_binding_and_current_slot_delta_only():
    reviews = _review_targets()
    snapshot = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(review_targets=reviews, review_question_count=1, vocabulary_plan_only=True)
    )
    assert snapshot.vocabulary_plan is not None
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("ambiguous",), ("ambiguous",), ()],
    )
    response = await ReadingVocabularyGenerationService(provider).generate(_request(
        review_targets=reviews, review_question_count=1,
        vocabulary_plan=snapshot.vocabulary_plan.model_dump(mode="json", by_alias=True),
    ))
    assert response.vocabulary_plan is not None
    before, after = snapshot.vocabulary_plan.items[0], response.vocabulary_plan.items[0]
    assert (after.target_expression, after.canonical_key, after.review_target) == (
        before.target_expression, before.canonical_key, True,
    )
    assert after.distractors != before.distractors
    assert response.vocabulary_plan.items[1:] == snapshot.vocabulary_plan.items[1:]
    assert response.questions[0].target_expression == before.target_expression
    assert {option.text for option in response.questions[0].options} == {
        before.target_expression, *after.distractors,
    }


@pytest.mark.asyncio
async def test_context_fallback_new_preserves_metadata_and_future_slots():
    snapshot = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(vocabulary_plan_only=True)
    )
    assert snapshot.vocabulary_plan is not None
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("answer_mismatch",), ("answer_mismatch",), ()],
    )
    response = await ReadingVocabularyGenerationService(provider).generate(_request(
        vocabulary_plan=snapshot.vocabulary_plan.model_dump(mode="json", by_alias=True),
    ))
    assert response.vocabulary_plan is not None
    before, after = snapshot.vocabulary_plan.items[0], response.vocabulary_plan.items[0]
    assert (after.target_expression, after.canonical_key, after.distractors) != (
        before.target_expression, before.canonical_key, before.distractors,
    )
    assert (after.global_order, after.review_target, after.skill_tag, after.difficulty,
            after.complexity_band, after.scenario_family, after.anchor_type, after.anchor_value) == (
            before.global_order, before.review_target, before.skill_tag, before.difficulty,
            before.complexity_band, before.scenario_family, before.anchor_type, before.anchor_value)
    assert response.vocabulary_plan.items[1:] == snapshot.vocabulary_plan.items[1:]
    assert response.questions[0].target_expression == after.target_expression


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["review_target", "anchor", "future_slot"])
async def test_context_fallback_cannot_escape_existing_plan_diff_authority(mutation):
    reviews = _review_targets() if mutation == "review_target" else []
    snapshot = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(review_targets=reviews, review_question_count=int(bool(reviews)),
                 vocabulary_plan_only=True)
    )
    assert snapshot.vocabulary_plan is not None
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("ambiguous",), ("ambiguous",)],
        lexical_repair_mutation=mutation,
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request(
            review_targets=reviews, review_question_count=int(bool(reviews)),
            vocabulary_plan=snapshot.vocabulary_plan.model_dump(mode="json", by_alias=True),
        ))
    assert caught.value.status_code == 422
    assert cast(dict[str, Any], caught.value.detail)["stage"] == "LEXICAL_VALIDATION"
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.operations["CONTEXT"] == 1


@pytest.mark.asyncio
async def test_context_reason_codes_collect_all_failures_without_learner_text(caplog):
    caplog.set_level("INFO", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("ambiguous", "answer_mismatch"),
                                  ("ambiguous", "answer_mismatch"), ()],
    )
    await ReadingVocabularyGenerationService(provider).generate(_request())
    assert (
        "phase=context_initial failure_class=SEMANTIC_CONTEXT "
        "reason_codes=['AMBIGUOUS', 'BEST_ANSWER_MISMATCH']"
    ) in caplog.text
    assert (
        "phase=context_semantic_repair failure_class=SEMANTIC_CONTEXT "
        "reason_codes=['AMBIGUOUS', 'BEST_ANSWER_MISMATCH']"
    ) in caplog.text
    assert provider.plan_generation_payloads[1]["reasonCodes"] == [
        "AMBIGUOUS", "BEST_ANSWER_MISMATCH",
    ]
    for hidden in ("個別表現", "関連候補", "修正候補", "会議で", "fixture context verdict"):
        assert hidden not in caplog.text


@pytest.mark.asyncio
async def test_structural_plan_repair_plus_context_fallback_remains_bounded():
    provider = ContextualChoiceV3Provider(
        plan_duplicate=True,
        context_failure_sequence=[("ambiguous",), ("ambiguous",), ()],
    )
    await ReadingVocabularyGenerationService(provider).generate(_request())
    assert provider.operations["PLAN"] == 1
    assert provider.operations["PLAN_REPAIR"] == 1
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.operations["CONTEXT"] == 2
    assert provider.operations["CONTEXT_REPAIR"] == 1
    assert sum(provider.calls.values()) == 12


@pytest.mark.asyncio
async def test_first_and_later_normal_call_counts_are_unchanged():
    first_provider = ContextualChoiceV3Provider()
    first = await ReadingVocabularyGenerationService(first_provider).generate(_request())
    assert first.vocabulary_plan is not None
    assert sum(first_provider.calls.values()) == 5
    later_provider = ContextualChoiceV3Provider()
    later = await ReadingVocabularyGenerationService(later_provider).generate(_request(
        previous_questions=[first.questions[0].model_copy(update={"order": 1}).model_dump(
            mode="json", by_alias=True,
        )],
        vocabulary_plan=first.vocabulary_plan.model_dump(mode="json", by_alias=True),
    ))
    assert len(later.questions) == 1
    assert sum(later_provider.calls.values()) == 4


@pytest.mark.asyncio
async def test_later_item_context_fallback_uses_same_current_slot_delta_contract():
    first = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request()
    )
    assert first.vocabulary_plan is not None
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("ambiguous",), ("ambiguous",), ()],
    )
    later = await ReadingVocabularyGenerationService(provider).generate(_request(
        previous_questions=[first.questions[0].model_copy(update={"order": 1}).model_dump(
            mode="json", by_alias=True,
        )],
        vocabulary_plan=first.vocabulary_plan.model_dump(mode="json", by_alias=True),
    ))
    assert later.vocabulary_plan is not None
    assert later.vocabulary_plan.items[0] == first.vocabulary_plan.items[0]
    assert later.vocabulary_plan.items[1] != first.vocabulary_plan.items[1]
    assert later.vocabulary_plan.items[2:] == first.vocabulary_plan.items[2:]
    assert later.questions[0].correct_answer == ["B"]
    assert next(option.text for option in later.questions[0].options if option.key == "B") == (
        later.vocabulary_plan.items[1].target_expression
    )
    assert sum(provider.calls.values()) == 10


@pytest.mark.asyncio
async def test_nuance_grammar_only_distractors_are_rejected_with_indexes_and_repaired(caplog):
    caplog.set_level("INFO", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    request = await _candidate_request_at_order(
        2, "承認を得る", ["承認を取る", "承認に得る", "承認が得る"]
    )
    provider = ContextualChoiceV3Provider(distractor_faults={
        1: ("unnatural", "surface", "grammar", "competitor", "skill"),
        2: ("unnatural", "surface", "grammar", "competitor", "skill"),
    })
    response = await ReadingVocabularyGenerationService(provider).generate(request)
    assert len(response.questions) == 1
    assert response.vocabulary_plan is not None
    assert request.vocabulary_plan is not None
    assert response.vocabulary_plan.items[1].target_expression == "承認を得る"
    assert response.vocabulary_plan.items[1].distractors != (
        "承認を取る", "承認に得る", "承認が得る"
    )
    assert response.vocabulary_plan.items[2:] == request.vocabulary_plan.items[2:]
    repair = provider.plan_generation_payloads[0]
    assert repair["operation"] == "LEXICAL_REPAIR"
    assert repair["invalidDistractorIndexes"] == [1, 2]
    assert repair["preserveTarget"] is True
    assert provider.plan_verdicts[0]["verdicts"][0]["reasonCode"] == (
        "INSUFFICIENT_CLOSE_DISTRACTORS"
    )
    assert repair["reasonCodes"] == [
        "DISTRACTOR_UNNATURAL", "DISTRACTOR_SURFACE_MISMATCH",
        "DISTRACTOR_GRAMMAR_ONLY",
        "DISTRACTOR_SKILL_MISMATCH",
    ]
    assert "invalid_distractor_indexes=[1, 2]" in caplog.text
    assert "DISTRACTOR_GRAMMAR_ONLY" in caplog.text
    assert "承認に得る" not in caplog.text
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 2


@pytest.mark.asyncio
async def test_repaired_bundle_with_remaining_confusion_failure_is_terminal():
    request = await _candidate_request_at_order(
        2, "承認を得る", ["承認を取る", "承認に得る", "承認が得る"]
    )
    provider = ContextualChoiceV3Provider(
        plan_quality_failure="confusion", plan_quality_order=2,
        persistent_plan_quality_failure=True,
        distractor_faults={1: ("grammar",), 2: ("grammar",)},
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(request)
    assert caught.value.status_code == 422
    assert cast(dict[str, Any], caught.value.detail)["stage"] == "LEXICAL_VALIDATION"
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 2
    assert provider.operations["CONTEXT"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("complexity_band", [4, 5])
async def test_b4_b5_close_competitor_count_excludes_grammar_only_options(
    complexity_band,
):
    provider = ContextualChoiceV3Provider(distractor_faults={
        1: ("grammar",), 2: ("grammar",),
    })
    await ReadingVocabularyGenerationService(provider).generate(_request(
        complexity_band=complexity_band,
    ))
    repair = provider.plan_generation_payloads[1]
    assert repair["invalidDistractorIndexes"] == [1, 2]
    assert "DISTRACTOR_GRAMMAR_ONLY" in repair["reasonCodes"]
    assert "INSUFFICIENT_CLOSE_DISTRACTORS" in repair["reasonCodes"]
    assert provider.operations["LEXICAL_REPAIR"] == 1


@pytest.mark.parametrize(
    ("band", "new_minimum"),
    [(1, 0), (2, 1), (3, 1), (4, 2), (5, 2)],
)
def test_close_distractor_policy_uses_one_helper_for_new_and_review(
    band, new_minimum,
):
    demand = vocabulary_difficulty_recipe(
        mode="CONTEXTUAL_CHOICE",
        band=band,
        skill_tag="MEANING",
        question_type="SINGLE_CHOICE",
    ).demand
    assert isinstance(demand, ContextualChoiceDemand)
    assert contextual_choice_required_close_distractors(
        demand.minimum_close_distractors, review_target=False
    ) == new_minimum
    assert contextual_choice_required_close_distractors(
        demand.minimum_close_distractors, review_target=True
    ) == 1


@pytest.mark.asyncio
async def test_b4_two_close_and_one_base_plausible_nonclose_distractor_passes():
    provider = ContextualChoiceV3Provider(
        distractor_faults={2: ("competitor",)},
    )
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(complexity_band=4)
    )

    assert len(response.questions) == 1
    assert provider.operations["LEXICAL_REPAIR"] == 0
    verdict = provider.plan_verdicts[0]["verdicts"][0]
    assert sum(item["closeCompetitor"] for item in verdict["distractors"].values()) == 2
    assert all(
        item["bundleRelevant"] for item in verdict["distractors"].values()
    )
    schema = provider.schemas[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME
    ][0]["properties"]["verdicts"]["items"]["properties"]
    assert "confusionSetValid" not in schema
    assert "genuineCompetitorCount" not in schema


@pytest.mark.asyncio
@pytest.mark.parametrize("base_fault", ["plausible", "unnatural", "grammar"])
async def test_base_invalid_distractor_rejects_even_when_two_close_remain(base_fault):
    provider = ContextualChoiceV3Provider(
        distractor_faults={2: (base_fault,)},
    )
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(complexity_band=4)
    )

    assert len(response.questions) == 1
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.plan_generation_payloads[1]["invalidDistractorIndexes"] == [2]


@pytest.mark.asyncio
async def test_register_sparse_repair_receives_structured_close_deficit_evidence():
    request = await _candidate_request_at_order(
        4,
        "ご不便をおかけしております",
        [
            "ご不便をおかけしました",
            "ご不便をおかけいたします",
            "ご不便をおかけしません",
        ],
    )
    assert request.vocabulary_plan is not None
    before = request.vocabulary_plan.items[3]
    provider = ContextualChoiceV3Provider(
        distractor_faults={
            1: ("competitor",),
            2: ("plausible", "competitor", "skill"),
        }
    )

    response = await ReadingVocabularyGenerationService(provider).generate(request)

    repair = provider.plan_generation_payloads[0]
    assert repair["invalidDistractorIndexes"] == [2]
    assert repair["distractorEvidence"] == {
        "2": {
            "expressionWellFormed": True,
            "sameSurfaceCategory": True,
            "bundleRelevant": False,
            "closeCompetitor": False,
            "malformedByGrammar": False,
            "skillContrastRelevant": False,
            "coveredDecisiveDimensions": [],
        }
    }
    assert repair["requiredCloseDistractors"] == 2
    assert repair["currentValidCloseDistractorCount"] == 1
    assert repair["additionalCloseDistractorsNeeded"] == 1
    assert repair["forbiddenExpressions"] == [
        "ご不便をおかけしております",
        "ご不便をおかけしました",
        "ご不便をおかけいたします",
        "ご不便をおかけしません",
    ]
    assert repair["repairRequirements"] == {
        "2": {
            "mustBeNonEmpty": True,
            "mustUseLearningLanguage": True,
            "mustDifferFromPrevious": True,
            "mustDifferFromTarget": True,
            "mustDifferFromPreservedDistractors": True,
            "mustBeExpressionWellFormed": True,
            "mustBeSameSurfaceCategory": True,
            "mustBeBundleRelevant": True,
            "mustBeSkillContrastRelevant": True,
            "mustBeMalformedByGrammar": False,
            "mustBeCloseCompetitor": True,
        }
    }
    assert repair["skillDemand"] == {
        "skill": "REGISTER",
        "decisiveDimensions": ["REGISTER", "ROLE_FORMALITY_OR_CHANNEL"],
    }
    assert "reason" not in repair
    assert "free-form reason" in provider.generation_prompts[0]
    assert "simple semantic reversal, negation, or tense difference" in (
        provider.generation_prompts[0]
    )
    assert "normalization-equivalent" in provider.generation_prompts[0]
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 2
    assert response.vocabulary_plan is not None
    after = response.vocabulary_plan.items[3]
    assert after.target_expression == before.target_expression
    assert after.distractors[:2] == before.distractors[:2]
    assert after.distractors[2] != before.distractors[2]


@pytest.mark.asyncio
async def test_repair_does_not_require_extra_close_competitor_when_quota_is_met():
    provider = ContextualChoiceV3Provider(
        distractor_faults={2: ("plausible", "competitor", "skill")},
    )

    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(complexity_band=4)
    )

    assert len(response.questions) == 1
    repair = provider.plan_generation_payloads[1]
    assert repair["requiredCloseDistractors"] == 2
    assert repair["currentValidCloseDistractorCount"] == 2
    assert repair["additionalCloseDistractorsNeeded"] == 0
    assert repair["repairRequirements"]["2"]["mustBeCloseCompetitor"] is False
    assert provider.operations["LEXICAL_REPAIR"] == 1


@pytest.mark.asyncio
async def test_sparse_lexical_repair_changes_only_the_allowed_distractor_index():
    request = await _candidate_request_at_order(
        2, "承認を得る", ["承認を取る", "許可を得る", "同意を得る"]
    )
    assert request.vocabulary_plan is not None
    before = request.vocabulary_plan.items[1]
    provider = ContextualChoiceV3Provider(
        distractor_faults={1: ("unnatural",)},
    )

    response = await ReadingVocabularyGenerationService(provider).generate(request)

    assert response.vocabulary_plan is not None
    after = response.vocabulary_plan.items[1]
    assert after.target_expression == before.target_expression
    assert after.canonical_key == before.canonical_key
    assert after.distractors[0] == before.distractors[0]
    assert after.distractors[1] != before.distractors[1]
    assert after.distractors[2] == before.distractors[2]
    assert response.vocabulary_plan.items[2:] == request.vocabulary_plan.items[2:]
    repair = provider.plan_generation_payloads[0]
    assert repair["repairScope"] == "DISTRACTOR_INDEXES"
    assert repair["invalidDistractorIndexes"] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["unexpected_index", "duplicate_index", "missing_index"]
)
async def test_sparse_lexical_repair_rejects_invalid_replacement_index_contract(
    mutation, caplog,
):
    caplog.set_level("INFO", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    request = await _candidate_request_at_order(
        2, "承認を得る", ["承認を取る", "許可を得る", "同意を得る"]
    )
    provider = ContextualChoiceV3Provider(
        distractor_faults={1: ("unnatural",)},
        lexical_repair_mutation=mutation,
    )

    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(request)

    assert caught.value.status_code == 422
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.operations["CONTEXT"] == 0
    assert "LEXICAL_REPAIR_SCOPE_MISMATCH" in caplog.text
    assert "reason_codes=['STRUCTURAL_DISTRACTOR']" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("target_equal", "LEXICAL_REPAIR_DUPLICATE_TARGET"),
        ("preserved_equal", "LEXICAL_REPAIR_DUPLICATE_DISTRACTOR"),
        ("normalized_duplicate", "LEXICAL_REPAIR_DUPLICATE_DISTRACTOR"),
        ("unchanged", "LEXICAL_REPAIR_UNCHANGED_DISTRACTOR"),
        ("empty", "LEXICAL_REPAIR_EMPTY_DISTRACTOR"),
        ("wrong_language", "LEXICAL_REPAIR_LANGUAGE_INVALID"),
    ],
)
async def test_sparse_lexical_repair_fault_records_typed_structural_code(
    mutation, expected_code, caplog,
):
    caplog.set_level("INFO", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    request = await _candidate_request_at_order(
        4,
        "ご不便をおかけしております",
        [
            "ご不便をおかけしました",
            "ご不便をおかけいたします",
            "ご不便をおかけしません",
        ],
    )
    provider = ContextualChoiceV3Provider(
        distractor_faults={2: ("plausible", "competitor", "skill")},
        lexical_repair_mutation=mutation,
    )

    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(request)

    assert caught.value.status_code == 422
    assert expected_code in caplog.text
    assert "reason_codes=['STRUCTURAL_DISTRACTOR']" not in caplog.text
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 1
    assert provider.operations["CONTEXT"] == 0


@pytest.mark.asyncio
async def test_known_good_register_sparse_replacement_passes_structure_and_revalidation():
    request = await _candidate_request_at_order(
        4,
        "ご不便をおかけしております",
        [
            "ご不便をおかけしました",
            "ご不便をおかけいたします",
            "ご不便をおかけしません",
        ],
    )
    provider = ContextualChoiceV3Provider(
        distractor_faults={2: ("plausible", "competitor", "skill")},
        lexical_repair_mutation="known_good_register",
    )

    response = await ReadingVocabularyGenerationService(provider).generate(request)

    assert response.vocabulary_plan is not None
    repaired = response.vocabulary_plan.items[3]
    assert repaired.distractors == [
        "ご不便をおかけしました",
        "ご不便をおかけいたします",
        "ご不便をおかけして申し訳ございません",
    ]
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 2
    assert provider.operations["CONTEXT"] == 1
    assert sum(provider.calls.values()) == 6
    repair_schema = provider.schemas[ReadingVocabularyGenerationService.TYPE_NAME][0]
    properties = repair_schema["properties"]
    assert properties["globalOrder"]["enum"] == [4]
    replacements = properties["distractorReplacements"]
    assert replacements["minItems"] == replacements["maxItems"] == 1
    assert replacements["items"]["properties"]["index"]["enum"] == [2]
    assert replacements["items"]["properties"]["text"]["minLength"] == 1


def test_order_4_diagnostics_preserve_verifier_flip_without_relabeling_pass():
    fixture = json.loads(_ORDER_4_VERIFIER_FLIP_FIXTURE.read_text(encoding="utf-8"))

    assert fixture["caseSha256"] == (
        "d7a355ab5adf516390678812355adfc4ef4a0c0ae99c340ff3cacf7b36d7a3b8"
    )
    assert [run["sourceArtifactSha256"] for run in fixture["runs"]] == [
        "271e6feeee617306fcc83955d67fb4c05fa2f87c7d80a70e4b9e85315cfddc37",
        "c4588dfe64819d9278510dcc890fe34b132d1bbe5aa56978f7a5c5bf6681776b",
    ]
    for run in fixture["runs"]:
        assert run["status"] == "FAILED"
        assert run["initialIndex1"] == {
            "naturalExpression": True,
            "plausibleWrongAlternative": True,
            "closeCompetitor": True,
            "wrongByGrammarOnly": False,
            "skillRelevant": True,
        }
        assert run["revalidationIndex1"]["naturalExpression"] is False
        assert run["revalidationIndex1"]["plausibleWrongAlternative"] is False
        assert run["revalidationIndex1"]["skillRelevant"] is False
        assert all(run["revalidationIndex2"][field] is True for field in (
            "naturalExpression", "plausibleWrongAlternative",
            "closeCompetitor", "skillRelevant",
        ))
    assert fixture["runs"][1]["revalidationIndex1"][
        "wrongByGrammarOnly"
    ] is True
    assert fixture["runs"][1]["revalidationReasonCode"] == "PASS"
    responsibility = fixture["offlineResponsibilityReference"]
    assert responsibility["scheduledNoticeExpression"] == "ご不便をおかけいたします"
    assert responsibility["scheduledNoticeExpressionWellFormed"] is True
    assert responsibility["scheduledNoticeMalformedByGrammar"] is False
    assert responsibility["intrinsicallyMalformedExpressionWellFormed"] is False
    assert responsibility["intrinsicallyMalformedByGrammar"] is True
    assert responsibility["currentBundleSkillContrastSupported"] is False


@pytest.mark.asyncio
async def test_lexical_verifier_contract_separates_expression_bundle_and_context():
    provider = ContextualChoiceV3Provider()

    await ReadingVocabularyGenerationService(provider).generate(_request())

    system_prompt = PRACTICE_CONTEXTUAL_CHOICE_PLAN_VERIFICATION_SYSTEM_PROMPT
    request_prompt = provider.plan_validation_prompts[0]
    assert "No learner-visible context exists at this stage" in system_prompt
    assert "planned service-interruption notice" in system_prompt
    assert "Tense/aspect, polarity, or synonym differences alone" in system_prompt
    assert "no actual context has been generated" in request_prompt
    assert "later context verifier" in request_prompt
    schema = provider.schemas[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME
    ][0]["properties"]["verdicts"]["items"]["properties"]
    assert "targetExpressionWellFormed" in schema
    assert "skillContrastSupported" in schema
    assert "coveredDecisiveDimensions" in schema
    distractor_schema = schema["distractors"]["properties"]["D0"]["properties"]
    assert set((
        "expressionWellFormed", "bundleRelevant", "malformedByGrammar",
        "skillContrastRelevant", "coveredDecisiveDimensions",
    )) <= set(distractor_schema)
    assert "naturalExpression" not in distractor_schema
    assert "wrongByGrammarOnly" not in distractor_schema
    assert distractor_schema["coveredDecisiveDimensions"]["items"]["enum"] == [
        "CONTEXTUAL_MEANING", "SENSE_FIT",
    ]


@pytest.mark.asyncio
async def test_revalidation_rechecks_full_bundle_and_ignores_conflicting_pass_code():
    request = await _candidate_request_at_order(
        4,
        "ご不便をおかけしております",
        [
            "ご不便をおかけしました",
            "ご不便をおかけいたします",
            "ご不便をおかけしません",
        ],
    )
    provider = ContextualChoiceV3Provider(
        distractor_faults={2: ("unnatural", "plausible", "competitor", "skill")},
        revalidation_distractor_faults={
            1: ("unnatural", "plausible", "competitor", "grammar", "skill")
        },
        lexical_repair_mutation="known_good_register",
        plan_reason_code_override="PASS",
    )

    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(request)

    assert caught.value.status_code == 422
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 2
    first, second = (
        verdict["verdicts"][0] for verdict in provider.plan_verdicts
    )
    assert first["distractors"]["D1"]["expressionWellFormed"] is True
    assert second["distractors"]["D1"]["expressionWellFormed"] is False
    assert second["distractors"]["D1"]["malformedByGrammar"] is True
    assert second["distractors"]["D2"]["expressionWellFormed"] is True
    assert second["reasonCode"] == "PASS"
    assert [
        value["expression"]
        for value in provider.plan_validation_payloads[1]["lexicalTargets"]["distractors"].values()
    ] == [
        "ご不便をおかけしました",
        "ご不便をおかけいたします",
        "ご不便をおかけして申し訳ございません",
    ]
    assert provider.operations["CONTEXT"] == 0


@pytest.mark.asyncio
async def test_register_bundle_skill_failure_repairs_distractors_without_target_change():
    request = await _candidate_request_at_order(
        4,
        "ご不便をおかけしております",
        [
            "ご不便をおかけしました",
            "ご不便をおかけいたします",
            "ご迷惑をおかけしております",
        ],
    )
    assert request.vocabulary_plan is not None
    target_before = request.vocabulary_plan.items[3].target_expression
    provider = ContextualChoiceV3Provider(
        plan_quality_failure="skill_bundle",
        plan_quality_order=4,
    )

    response = await ReadingVocabularyGenerationService(provider).generate(request)

    assert response.vocabulary_plan is not None
    assert response.vocabulary_plan.items[3].target_expression == target_before
    repair = provider.plan_generation_payloads[0]
    assert repair["preserveTarget"] is True
    assert repair["repairScope"] == "DISTRACTOR_INDEXES"
    assert repair["invalidDistractorIndexes"] == [0, 1, 2]
    assert provider.plan_verdicts[0]["verdicts"][0][
        "skillContrastSupported"
    ] is False
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 2


@pytest.mark.asyncio
async def test_intrinsically_malformed_expression_still_triggers_lexical_repair():
    request = await _candidate_request_at_order(
        4,
        "ご不便をおかけしております",
        [
            "ご不便をおかけしました",
            "ご不便をおかけにいたします",
            "ご迷惑をおかけしております",
        ],
    )
    provider = ContextualChoiceV3Provider(
        distractor_faults={1: ("unnatural", "grammar")},
    )

    await ReadingVocabularyGenerationService(provider).generate(request)

    repair = provider.plan_generation_payloads[0]
    assert repair["invalidDistractorIndexes"] == [1]
    assert "DISTRACTOR_UNNATURAL" in repair["reasonCodes"]
    assert "DISTRACTOR_GRAMMAR_ONLY" in repair["reasonCodes"]


@pytest.mark.asyncio
async def test_skill_dimension_evidence_is_required_even_when_boolean_is_true():
    provider = ContextualChoiceV3Provider(
        distractor_faults={1: ("dimension",)},
    )

    await ReadingVocabularyGenerationService(provider).generate(_request())

    first_verdict = provider.plan_verdicts[0]["verdicts"][0]["distractors"]["D1"]
    assert first_verdict["skillContrastRelevant"] is True
    assert first_verdict["coveredDecisiveDimensions"] == []
    repair = provider.plan_generation_payloads[1]
    assert repair["invalidDistractorIndexes"] == [1]
    assert repair["reasonCodes"] == ["DISTRACTOR_SKILL_MISMATCH"]


@pytest.mark.asyncio
async def test_context_ambiguity_rejects_even_when_all_rendered_options_are_structural():
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[
            ("ambiguous",), ("ambiguous",), ("ambiguous",),
        ],
    )

    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())

    assert caught.value.status_code == 422
    assert len(provider.context_validation_payloads) == 3
    assert all(
        verdict["verdicts"][0]["structurallyWellFormedKeys"]
        == list(CONTEXTUAL_CHOICE_ANSWER_KEYS)
        for verdict in provider.context_verdicts
    )
    assert all(
        verdict["verdicts"][0]["ambiguous"] is True
        for verdict in provider.context_verdicts
    )


@pytest.mark.asyncio
async def test_new_target_replacement_is_allowed_only_when_structured_target_is_not_viable():
    request = await _candidate_request_at_order(
        2, "承認を得る", ["承認を取る", "承認に得る", "承認が得る"]
    )
    provider = ContextualChoiceV3Provider(
        natural_target=False, target_viable=False,
        distractor_faults={1: ("grammar",)},
    )
    response = await ReadingVocabularyGenerationService(provider).generate(request)
    assert response.vocabulary_plan is not None
    assert provider.plan_generation_payloads[0]["preserveTarget"] is False
    assert provider.plan_generation_payloads[0]["repairScope"] == "FULL_LEXICAL_BUNDLE"
    assert provider.plan_generation_payloads[0]["invalidDistractorIndexes"] == [0, 1, 2]
    assert response.vocabulary_plan.items[1].target_expression != "承認を得る"
    assert response.vocabulary_plan.items[1].canonical_key == (
        response.vocabulary_plan.items[1].target_expression
    )


@pytest.mark.asyncio
async def test_structured_review_distractor_failure_preserves_review_authority():
    reviews = _review_targets()
    provider = ContextualChoiceV3Provider(distractor_faults={
        1: ("grammar", "unnatural"),
    })
    response = await ReadingVocabularyGenerationService(provider).generate(_request(
        review_targets=reviews, review_question_count=1,
    ))
    assert response.vocabulary_plan is not None
    first = response.vocabulary_plan.items[0]
    assert first.target_expression == reviews[0]["expression"]
    assert first.canonical_key == reviews[0]["canonicalKey"]
    assert provider.plan_generation_payloads[1]["invalidDistractorIndexes"] == [1]
    assert provider.plan_generation_payloads[1]["preserveTarget"] is True
    assert response.questions[0].target_expression == reviews[0]["expression"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("order", "skill"),
    [(3, "COLLOCATION"), (4, "REGISTER"), (5, "PRAGMATIC_FIT")],
)
async def test_skill_specific_grammar_only_distractors_never_pass(order, skill):
    request = await _candidate_request_at_order(
        order, f"自然な表現{order}",
        [f"関連表現{order}一", f"不自然な表現{order}二", f"関連表現{order}三"],
    )
    assert request.vocabulary_plan is not None
    assert request.vocabulary_plan.items[order - 1].skill_tag == skill
    provider = ContextualChoiceV3Provider(distractor_faults={1: ("grammar",)})
    response = await ReadingVocabularyGenerationService(provider).generate(request)
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.plan_generation_payloads[0]["reasonCodes"] == [
        "DISTRACTOR_GRAMMAR_ONLY",
    ]
    assert response.vocabulary_plan is not None
    assert response.vocabulary_plan.items[order - 1].target_expression == (
        request.vocabulary_plan.items[order - 1].target_expression
    )


@pytest.mark.asyncio
async def test_context_verifier_receives_all_four_option_substituted_sentences():
    provider = ContextualChoiceV3Provider()
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    visible = provider.context_validation_payloads[0]["question"]
    rendered = visible["renderedOptions"]
    assert [item["key"] for item in rendered] == ["A", "B", "C", "D"]
    assert [item["sentence"] for item in rendered] == [
        response.questions[0].prompt.replace("______", option.text, 1)
        for option in response.questions[0].options
    ]
    assert all("______" not in item["sentence"] for item in rendered)
    assert "correctAnswer" not in json.dumps(visible, ensure_ascii=False)
    assert "canonicalKey" not in json.dumps(visible, ensure_ascii=False)


@pytest.mark.asyncio
async def test_context_generator_receives_server_difficulty_demand_but_verifier_stays_blind():
    provider = ContextualChoiceV3Provider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request(complexity_band=5)
    )

    assert response.vocabulary_plan is not None
    plan_item = response.vocabulary_plan.items[0]
    assert plan_item.anchor_type is not None
    generation = provider.context_generation_payloads[0]
    bundle = generation["fixedLexicalBundle"]
    demand = generation["difficultyDemand"]
    assert bundle["difficulty"] == plan_item.difficulty.value
    assert bundle["complexityBand"] == 5
    assert bundle["scenarioFamily"] == plan_item.scenario_family
    assert bundle["anchorType"] == plan_item.anchor_type.value
    assert bundle["anchorValue"] == plan_item.anchor_value
    assert demand["contextShape"] == "PRECISE_REGISTER_SCOPE_NUANCE_OR_PRAGMATIC_BLANK"
    assert demand["minimumCloseDistractors"] == 2
    assert demand["decisiveDimensions"]
    visible = provider.context_validation_payloads[0]["question"]
    hidden_wire = json.dumps(visible, ensure_ascii=False)
    for hidden in (
        "targetExpression", "canonicalKey", "correctAnswer",
        "complexityBand", "difficultyDemand", "requestedBand",
    ):
        assert hidden not in hidden_wire


@pytest.mark.asyncio
async def test_context_repair_receives_only_latest_candidate_and_structured_evidence():
    class SequencedContextProvider(ContextualChoiceV3Provider):
        def _context(self, payload: dict) -> dict:
            raw = super()._context(payload)
            order = payload["fixedLexicalBundle"]["globalOrder"]
            attempt = self.context_generation_counts_by_order[order]
            raw["contextTemplate"] = (
                f"修正候補文{attempt}では、状況を確認して{{{{TARGET}}}}ことにしました。"
            )
            return raw

    provider = SequencedContextProvider(
        context_failure_sequence=[("structural",), ("ambiguous",), ()],
    )
    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert len(response.questions) == 1
    first_repair, second_repair = provider.context_generation_payloads[1:]
    assert first_repair["previousCandidate"]["contextTemplate"].startswith(
        "修正候補文1"
    )
    assert first_repair["previousCandidate"]["renderedOptions"]
    assert first_repair["failureEvidence"]["structurallyInvalidOptionKeys"] == [
        "A"
    ]
    assert second_repair["previousCandidate"]["contextTemplate"].startswith(
        "修正候補文2"
    )
    assert "修正候補文1" not in json.dumps(second_repair, ensure_ascii=False)
    assert second_repair["failureEvidence"]["ambiguous"] is True
    repair_wire = json.dumps(first_repair, ensure_ascii=False)
    for unrelated in (
        "selectedKeywords", "weakSignals", "recentMistakes", "requestId",
    ):
        assert unrelated not in repair_wire


@pytest.mark.asyncio
async def test_semantically_weaker_options_pass_structural_gate_without_repair():
    request = await _candidate_request_at_order(
        1,
        "追加の検証期間",
        ["追加の開発期間", "追加の監視期間", "追加の準備期間"],
    )
    provider = ContextualChoiceV3Provider(
        context_template_override="安全性を確かめるため、チームは{{TARGET}}を設けた。",
    )
    response = await ReadingVocabularyGenerationService(provider).generate(request)
    assert len(response.questions) == 1
    assert provider.operations["CONTEXT_REPAIR"] == 0
    assert provider.operations["LEXICAL_REPAIR"] == 0
    verdict = provider.context_verdicts[0]["verdicts"][0]
    assert verdict["structurallyWellFormedKeys"] == list(
        CONTEXTUAL_CHOICE_ANSWER_KEYS
    )
    assert verdict["bestAnswerKey"] == "A"
    rendered = provider.context_validation_payloads[0]["question"]["renderedOptions"]
    assert [item["sentence"] for item in rendered] == [
        "安全性を確かめるため、チームは追加の検証期間を設けた。",
        "安全性を確かめるため、チームは追加の開発期間を設けた。",
        "安全性を確かめるため、チームは追加の監視期間を設けた。",
        "安全性を確かめるため、チームは追加の準備期間を設けた。",
    ]
    schema = provider.schemas[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME
    ][0]["properties"]["verdicts"]["items"]["properties"]
    assert "structurallyWellFormedKeys" in schema
    assert "grammaticallyCompatibleKeys" not in schema
    assert "renderedSentenceNaturalKeys" not in schema


@pytest.mark.asyncio
async def test_semantic_mismatch_is_independent_from_structural_contract():
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("answer_mismatch",), ()],
    )
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert len(response.questions) == 1
    assert provider.context_verdicts[0]["verdicts"][0][
        "structurallyWellFormedKeys"
    ] == list(CONTEXTUAL_CHOICE_ANSWER_KEYS)
    assert provider.operations["CONTEXT_REPAIR"] == 1
    assert provider.operations["LEXICAL_REPAIR"] == 0


@pytest.mark.asyncio
async def test_persistent_structural_failure_is_context_local_not_lexical_fallback(caplog):
    caplog.set_level("INFO", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("structural",), ("structural",)],
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    assert cast(dict[str, Any], caught.value.detail)["stage"] == "CONTEXT_VALIDATION"
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 1
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert "phase=context_initial" in caplog.text
    assert "phase=context_structural_repair" in caplog.text
    assert "RENDERED_SENTENCE_STRUCTURALLY_INVALID" in caplog.text


@pytest.mark.asyncio
async def test_lexical_repair_then_structural_to_semantic_transition_recovers(caplog):
    caplog.set_level("INFO", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    provider = ContextualChoiceV3Provider(
        plan_quality_failure="surface",
        context_failure_sequence=[("structural",), ("ambiguous",), ()],
    )
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert len(response.questions) == 1
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 2
    assert len(provider.plan_validation_payloads) == 2
    assert len(provider.context_validation_payloads) == 3
    assert provider.context_generation_payloads[1]["repairClass"] == (
        "STRUCTURAL_CONTEXT"
    )
    assert provider.context_generation_payloads[1]["reasonCodes"] == [
        "RENDERED_SENTENCE_STRUCTURALLY_INVALID"
    ]
    assert provider.context_generation_payloads[1]["repairDirective"] == (
        "Revise the blank frame so all four completed sentences are structurally "
        "well-formed without argument duplication."
    )
    assert provider.context_generation_payloads[2]["repairClass"] == (
        "SEMANTIC_CONTEXT"
    )
    assert provider.context_generation_payloads[2]["reasonCodes"] == ["AMBIGUOUS"]
    assert provider.context_generation_payloads[2]["repairDirective"] == (
        "Make the fixed target the unique best answer; remove visible ambiguity."
    )
    assert "failure_class=STRUCTURAL_CONTEXT" in caplog.text
    assert "failure_class=SEMANTIC_CONTEXT" in caplog.text
    for hidden in ("個別表現", "関連候補", "修正候補", "会議で"):
        assert hidden not in caplog.text
    assert sum(provider.calls.values()) == 11


@pytest.mark.asyncio
async def test_semantic_to_structural_transition_recovers_in_reverse_order():
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("ambiguous",), ("structural",), ()],
    )
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert len(response.questions) == 1
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 2
    assert len(provider.context_validation_payloads) == 3
    assert "unique best answer" in provider.context_generation_payloads[1][
        "repairDirective"
    ]
    assert "structurally well-formed" in provider.context_generation_payloads[2][
        "repairDirective"
    ]
    assert sum(provider.calls.values()) == 9


@pytest.mark.asyncio
async def test_mixed_structural_semantic_codes_use_structural_class_first():
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[
            ("structural", "ambiguous"), ("ambiguous",), (),
        ],
    )
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert len(response.questions) == 1
    assert provider.operations["CONTEXT_REPAIR"] == 2
    assert provider.operations["LEXICAL_REPAIR"] == 0
    first_repair = provider.context_generation_payloads[1]["repairDirective"]
    second_repair = provider.context_generation_payloads[2]["repairDirective"]
    assert "structurally well-formed" in first_repair
    assert "unique best answer" not in first_repair
    assert "unique best answer" in second_repair


@pytest.mark.asyncio
async def test_deterministic_context_then_semantic_transition_uses_two_class_budgets():
    provider = ContextualChoiceV3Provider(
        marker_count=0,
        context_failure_sequence=[("ambiguous",), ()],
    )
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert len(response.questions) == 1
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 2
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert len(provider.context_generation_payloads) == 3
    assert len(provider.context_validation_payloads) == 2
    assert provider.context_generation_payloads[1]["repairClass"] == "CONTRACT_CONTEXT"
    assert provider.context_generation_payloads[1]["reasonCodes"] == [
        "CONTEXT_MARKER_INVALID"
    ]
    assert "exactly one literal {{TARGET}}" in (
        provider.context_generation_payloads[1]["repairDirective"]
    )
    assert "unique best answer" in provider.context_generation_payloads[2][
        "repairDirective"
    ]


@pytest.mark.asyncio
async def test_structural_to_semantic_to_semantic_stops_without_fourth_context():
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[
            ("structural",), ("ambiguous",), ("ambiguous",),
        ],
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())
    assert caught.value.status_code == 422
    assert cast(dict[str, Any], caught.value.detail)["stage"] == "CONTEXT_VALIDATION"
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 2
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert len(provider.context_generation_payloads) == 3
    assert len(provider.context_validation_payloads) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sequence",
    [
        [("structural",), ("ambiguous",), ()],
        [("ambiguous",), ("structural",), ()],
        [("ambiguous",), ("ambiguous",), ()],
    ],
)
async def test_all_context_recovery_paths_are_bounded_to_three_generations(sequence):
    provider = ContextualChoiceV3Provider(context_failure_sequence=sequence)
    await ReadingVocabularyGenerationService(provider).generate(_request())
    assert len(provider.context_generation_payloads) <= 3
    assert provider.operations["LEXICAL_REPAIR"] <= 1


@pytest.mark.asyncio
async def test_real_order_three_structural_then_contract_failure_recovers(caplog):
    baseline = await ReadingVocabularyGenerationService(
        ContextualChoiceV3Provider()
    ).generate(_request(question_count=10))
    assert baseline.vocabulary_plan is not None
    previous = [
        question.model_dump(mode="json", by_alias=True)
        for question in baseline.questions[:2]
    ]
    caplog.set_level("INFO", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=[("structural",), ()],
        context_contract_sequence_by_order={
            3: [None, "missing_marker", None],
        },
    )

    response = await ReadingVocabularyGenerationService(provider).generate(_request(
        request_id="contextual-v3-3-order-3",
        previous_questions=previous,
        vocabulary_plan=baseline.vocabulary_plan.model_dump(mode="json", by_alias=True),
    ))

    assert len(response.questions) == 1
    assert response.questions[0].target_expression == (
        baseline.vocabulary_plan.items[2].target_expression
    )
    assert provider.context_generation_counts_by_order[3] == 3
    assert len(provider.context_validation_payloads) == 2
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 2
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert sum(provider.calls.values()) == 7
    repairs = provider.context_generation_payloads[1:]
    assert [payload["repairClass"] for payload in repairs] == [
        "STRUCTURAL_CONTEXT",
        "CONTRACT_CONTEXT",
    ]
    assert repairs[1]["reasonCodes"] == ["CONTEXT_MARKER_INVALID"]
    assert (
        "attempt=2 phase=context_structural_repair "
        "failure_class=CONTRACT_CONTEXT "
        "reason_codes=['CONTEXT_MARKER_INVALID']"
    ) in caplog.text


@pytest.mark.asyncio
async def test_lexical_repair_then_structural_contract_recovery_mutates_lexicon_once():
    provider = ContextualChoiceV3Provider(
        plan_quality_failure="surface",
        context_failure_sequence=[("structural",), ()],
        context_contract_sequence_by_order={
            1: [None, "missing_marker", None],
        },
    )

    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert len(response.questions) == 1
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 2
    assert provider.context_generation_counts_by_order[1] == 3
    assert len(provider.context_validation_payloads) == 2
    assert sum(provider.calls.values()) == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("contract_sequence", "semantic_sequence", "expected_classes"),
    [
        (["missing_marker", None, None], [("structural",), ()],
         ["CONTRACT_CONTEXT", "STRUCTURAL_CONTEXT"]),
        (["missing_marker", None, None], [("ambiguous",), ()],
         ["CONTRACT_CONTEXT", "SEMANTIC_CONTEXT"]),
        ([None, "missing_marker", None], [("ambiguous",), ()],
         ["SEMANTIC_CONTEXT", "CONTRACT_CONTEXT"]),
    ],
)
async def test_distinct_context_failure_classes_each_receive_one_bounded_repair(
    contract_sequence, semantic_sequence, expected_classes,
):
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=semantic_sequence,
        context_contract_sequence_by_order={1: contract_sequence},
    )

    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert len(response.questions) == 1
    assert provider.context_generation_counts_by_order[1] == 3
    assert [
        payload["repairClass"]
        for payload in provider.context_generation_payloads[1:]
    ] == expected_classes
    assert provider.operations["LEXICAL_REPAIR"] == 0


@pytest.mark.asyncio
async def test_repeated_contract_class_stops_without_blind_third_repair():
    provider = ContextualChoiceV3Provider(
        context_contract_sequence_by_order={
            1: ["missing_marker", "duplicate_marker", None],
        },
    )

    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())

    assert caught.value.status_code == 422
    assert cast(dict[str, Any], caught.value.detail)["stage"] == "CONTEXT_VALIDATION"
    assert provider.context_generation_counts_by_order[1] == 2
    assert provider.operations["CONTEXT_REPAIR"] == 1
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert len(provider.context_validation_payloads) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("contract_sequence", "semantic_sequence"),
    [
        ([None, "missing_marker", None], [("structural",), ("ambiguous",)]),
        (["missing_marker", None, "duplicate_marker"], [("structural",)]),
    ],
)
async def test_three_context_limit_blocks_a_fourth_cross_class_repair(
    contract_sequence, semantic_sequence,
):
    provider = ContextualChoiceV3Provider(
        context_failure_sequence=semantic_sequence,
        context_contract_sequence_by_order={1: contract_sequence},
    )

    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())

    assert caught.value.status_code == 422
    assert provider.context_generation_counts_by_order[1] == 3
    assert provider.operations["CONTEXT_REPAIR"] == 2
    assert provider.operations["LEXICAL_REPAIR"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault", "reason_code"),
    [
        ("response_schema", "CONTEXT_RESPONSE_SCHEMA_INVALID"),
        ("wrong_order", "CONTEXT_ORDER_MISMATCH"),
        ("missing_marker", "CONTEXT_MARKER_INVALID"),
        ("duplicate_marker", "CONTEXT_MARKER_INVALID"),
        ("unknown_marker", "CONTEXT_UNKNOWN_MARKER"),
        ("language", "CONTEXT_LANGUAGE_INVALID"),
        ("target_leak", "CONTEXT_TARGET_LEAK"),
        ("round_trip", "CONTEXT_TEMPLATE_ROUND_TRIP_INVALID"),
        ("explanation", "CONTEXT_EXPLANATION_INVALID"),
    ],
)
async def test_deterministic_context_failures_emit_stable_contract_codes(
    fault, reason_code,
):
    provider = ContextualChoiceV3Provider(
        context_contract_sequence_by_order={1: [fault, None]},
    )

    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert len(response.questions) == 1
    assert provider.context_generation_counts_by_order[1] == 2
    assert len(provider.context_validation_payloads) == 1
    repair = provider.context_generation_payloads[1]
    assert repair["repairClass"] == "CONTRACT_CONTEXT"
    assert repair["reasonCodes"] == [reason_code]
    assert repair["repairDirective"]
    assert "repairReason" not in repair
    assert sum(provider.calls.values()) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["malformed", "coverage", "unknown_key", "unknown_answer"]
)
async def test_verifier_contract_failure_is_not_mutated_as_content(fault, caplog):
    caplog.set_level("WARNING", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    provider = ContextualChoiceV3Provider(context_verifier_fault=fault)

    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())

    detail = cast(dict[str, Any], caught.value.detail)
    assert caught.value.status_code == 502
    assert detail["code"] == "AI_GENERATION_FAILED"
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 0
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert len(provider.context_validation_payloads) == 1
    assert "reason_code=CONTEXT_VERIFIER_CONTRACT_INVALID" in caplog.text


@pytest.mark.asyncio
async def test_application_owned_context_invariant_is_not_sent_to_model_repair(
    monkeypatch, caplog,
):
    caplog.set_level("WARNING", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    provider = ContextualChoiceV3Provider()
    service = ReadingVocabularyGenerationService(provider)

    def reject_internal_authority(*_args, **_kwargs):
        raise ValueError("fixture internal authority detail")

    monkeypatch.setattr(service, "_validate_candidate", reject_internal_authority)
    with pytest.raises(HTTPException) as caught:
        await service.generate(_request())

    assert caught.value.status_code == 502
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 0
    assert len(provider.context_validation_payloads) == 0
    assert "reason_code=CONTEXT_INTERNAL_CONTRACT_INVALID" in caplog.text
    assert "fixture internal authority detail" not in caplog.text


@pytest.mark.asyncio
async def test_rendered_target_argument_duplication_is_visible_and_hard_rejected(caplog):
    caplog.set_level("INFO", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    review = [{
        "canonicalKey": "お客様をご案内する", "expression": "お客様をご案内する",
        "preferredSkill": "REGISTER",
    }]
    provider = ContextualChoiceV3Provider(
        context_template_override=(
            "到着したお客様をロビーから客室まで丁寧に{{TARGET}}。"
        ),
        context_failure_sequence=[("structural",), ("structural",)],
    )
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request(
            review_targets=review, review_question_count=1,
        ))
    assert caught.value.status_code == 422
    assert cast(dict[str, Any], caught.value.detail)["stage"] == "CONTEXT_VALIDATION"
    rendered = provider.context_validation_payloads[0]["question"]["renderedOptions"]
    assert rendered[0]["sentence"] == (
        "到着したお客様をロビーから客室まで丁寧にお客様をご案内する。"
    )
    first_verdict = provider.context_verdicts[0]["verdicts"][0]
    assert "A" not in first_verdict["structurallyWellFormedKeys"]
    assert "RENDERED_SENTENCE_STRUCTURALLY_INVALID" in caplog.text
    assert provider.operations["LEXICAL_REPAIR"] == 0


@pytest.mark.asyncio
async def test_wrong_option_grammar_elimination_rejects_even_if_target_is_compatible():
    provider = ContextualChoiceV3Provider(context_failure_sequence=[
        ("grammar_wrong",), (),
    ])
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert len(response.questions) == 1
    assert provider.operations["CONTEXT_REPAIR"] == 1
    assert provider.operations["LEXICAL_REPAIR"] == 0


@pytest.mark.asyncio
async def test_obviously_truncated_origin_explanation_uses_existing_fallback(caplog):
    caplog.set_level("WARNING", logger=(
        "app.features.language_learning.reading_vocabulary.service"
    ))
    provider = ContextualChoiceV3Provider(origin_explanation_sequence=[
        "따라서 정답 표현인", "따라서 정답 표현인",
        "문맥상 이 표현이 가장 자연스럽습니다.",
    ])
    response = await ReadingVocabularyGenerationService(provider).generate(_request())
    assert response.questions[0].explanation_origin == (
        "문맥상 이 표현이 가장 자연스럽습니다."
    )
    assert provider.calls[
        ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME
    ] == 2
    assert provider.calls[
        ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME
    ] == 1
    assert "reason_code=ORIGIN_EXPLANATION_INCOMPLETE" in caplog.text
    assert "따라서 정답 표현인" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_explanation",
    ["該当なし", "evidenceLearning: 文脈に合います。"],
)
async def test_invalid_learning_explanation_is_repaired_after_answer_without_context_regeneration(
    bad_explanation,
):
    provider = ContextualChoiceV3Provider(context_explanation=bad_explanation)

    response = await ReadingVocabularyGenerationService(provider).generate(_request())

    assert response.vocabulary_plan is not None
    question = response.questions[0]
    assert question.explanation_learning == "正答表現が文脈の条件に最も自然に合います。"
    assert question.target_expression == response.vocabulary_plan.items[0].target_expression
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 0
    assert provider.operations["LEXICAL_REPAIR"] == 0
    assert provider.calls[
        ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME
    ] == 1
    assert provider.origin_payloads[0]["questions"][0][
        "repairExplanationLearning"
    ] is True


@pytest.mark.asyncio
async def test_learning_explanation_repair_uses_existing_two_plus_fallback_budget():
    provider = ContextualChoiceV3Provider(
        context_explanation="該当なし",
        origin_learning_explanation_sequence=[
            "該当なし", "none", "evidenceLearning: 不明",
        ],
    )

    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(_request())

    assert caught.value.status_code == 502
    assert provider.operations["CONTEXT"] == 1
    assert provider.operations["CONTEXT_REPAIR"] == 0
    assert provider.calls[
        ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME
    ] == 2
    assert provider.calls[
        ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME
    ] == 1


@pytest.mark.parametrize(
    ("text", "exposes_metadata"),
    [
        ("제시된 evidenceLearning 필드를 참고했습니다.", True),
        ("targetExpression 값이 정답입니다.", True),
        ("일본어 표현 ‘追加の検証期間’은 이 문맥에서 자연스럽습니다.", False),
    ],
)
def test_origin_metadata_guard_blocks_fields_but_allows_natural_quotes(
    text, exposes_metadata,
):
    assert ReadingVocabularyGenerationService._origin_explanation_exposes_internal_metadata(
        text
    ) is exposes_metadata


@pytest.mark.parametrize(
    ("text", "incomplete"),
    [
        ("따라서 정답 표현인", True),
        ("짧음", True),
        ('{"text":"unfinished', True),
        ("자연스러운 설명이 중간에서...", True),
        ("이 표현은 실제 상황에서 자연스럽게 사용됩니다", False),
    ],
)
def test_contextual_origin_completeness_check_is_limited_to_obvious_cases(
    text, incomplete,
):
    assert ReadingVocabularyGenerationService._contextual_choice_origin_explanation_incomplete(
        text, "ko"
    ) is incomplete


@pytest.mark.asyncio
async def test_answer_rotation_and_full_daily_progressive_generation():
    reviews = _review_targets()
    provider = ContextualChoiceV3Provider()
    service = ReadingVocabularyGenerationService(provider)
    response = await service.generate(
        _request(review_targets=reviews, review_question_count=1)
    )
    assert response.vocabulary_plan is not None
    plan = response.vocabulary_plan
    generated = [response.questions[0]]
    previous = [
        response.questions[0].model_copy(update={"order": 1}).model_dump(
            mode="json", by_alias=True
        )
    ]
    for global_order in range(2, 11):
        current_reviews = [reviews[1], reviews[0]] if global_order == 2 else reviews
        current = await service.generate(
            _request(
                request_id=f"progressive-{global_order}",
                previous_questions=previous,
                vocabulary_plan=plan.model_dump(mode="json", by_alias=True),
                review_targets=current_reviews,
                review_question_count=int(global_order == 2),
            )
        )
        question = current.questions[0]
        generated.append(question)
        previous.append(
            question.model_copy(update={"order": global_order}).model_dump(
                mode="json", by_alias=True
            )
        )
    assert [question.correct_answer[0] for question in generated] == [
        "A", "B", "C", "D", "A", "B", "C", "D", "A", "B"
    ]
    assert provider.operations["PLAN"] == 1
    assert provider.operations["CONTEXT"] == 10
    assert provider.calls[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME
    ] == 10
    assert provider.calls[
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME
    ] == 10
    assert sum(provider.calls.values()) == 41


@pytest.mark.asyncio
async def test_contextual_choice_shadow_is_not_on_the_production_critical_path():
    provider = ContextualChoiceV3Provider()
    response = await ReadingVocabularyGenerationService(
        provider,
        vocabulary_difficulty_shadow_enabled=True,
        vocabulary_difficulty_shadow_sample_percent=100,
    ).generate(_request())
    assert len(response.questions) == 1
    assert provider.calls[VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME] == 0
    assert str(vocabulary_blind_rubric_payload("CONTEXTUAL_CHOICE")["version"]).endswith(
        "shadow-rubric-v1"
    )


@pytest.mark.asyncio
async def test_normal_first_question_and_full_set_call_budget_is_bounded():
    one = ContextualChoiceV3Provider()
    first = await ReadingVocabularyGenerationService(one).generate(_request())
    assert first.vocabulary_plan is not None
    assert one.operations == Counter({"PLAN": 1, "CONTEXT": 1})
    assert sum(one.calls.values()) == 5

    batch = ContextualChoiceV3Provider()
    full = await ReadingVocabularyGenerationService(batch).generate(
        _request(question_count=10)
    )
    assert len(full.questions) == 10
    assert batch.operations["PLAN"] == 1
    assert batch.operations["CONTEXT"] == 10
    assert sum(batch.calls.values()) == 34


def test_plan_contract_rejects_wrong_daily_mix():
    request = _request()
    service = ReadingVocabularyGenerationService(ContextualChoiceV3Provider())
    daily = service._contextual_choice_daily_request(request)
    slots = service._build_slots(daily, {})
    items = [
        {
            "globalOrder": slot.order,
            "reviewTarget": False,
            "targetExpression": f"表現{slot.order}",
            "canonicalKey": f"表現{slot.order}",
            "distractors": [f"候補{slot.order}一", f"候補{slot.order}二", f"候補{slot.order}三"],
            "skillTag": slot.skill_tag,
            "difficulty": "CURRENT",
            "complexityBand": 4,
            "scenarioFamily": slot.scenario_family,
            "anchorType": "SELECTED_KEYWORD",
            "anchorValue": "협업",
        }
        for slot in slots
    ]
    with pytest.raises(ValidationError, match="2/6/2"):
        PersonalizedVocabularyPlan.model_validate({
            "version": CONTEXTUAL_CHOICE_PLAN_VERSION, "items": items,
        })


def test_recipe_v3_keeps_confusion_set_demand_and_shadow_ruler():
    recipe = vocabulary_difficulty_recipe(
        mode="CONTEXTUAL_CHOICE",
        band=5,
        skill_tag="PRAGMATIC_FIT",
        question_type="SINGLE_CHOICE",
    )
    assert recipe.version == "vocabulary-contextual-choice-recipe-v3"
    assert isinstance(recipe.demand, ContextualChoiceDemand)
    assert recipe.demand.minimum_close_distractors == 2
    assert vocabulary_blind_rubric_payload("CONTEXTUAL_CHOICE")["version"] == (
        "vocabulary-contextual-choice-shadow-rubric-v1"
    )
