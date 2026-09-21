"""Provider identity/schema guarantees, not Japanese semantic-quality evidence."""
from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any, cast

import pytest
from fastapi import HTTPException

from app.ai.providers.openai.schema import build_openai_text_config
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
    _CONTEXTUAL_CHOICE_PLAN_GENERATION_SCHEMA,
    _ContextualChoiceLexicalVerifierContractError,
    _contextual_choice_plan_verification_schema,
    _contextual_choice_verifier_coverage_summary,
    _parse_contextual_choice_plan_verdict,
)
from tests.test_language_learning_vocabulary_contextual_choice import (
    ContextualChoiceV3Provider,
    _candidate_request_at_order,
    _request,
    _review_targets,
)


def _valid_response() -> dict[str, Any]:
    return ContextualChoiceV3Provider()._plan_verdicts({
        "currentOrder": 4,
        "structuralDifficultyDemand": {
            "decisiveDimensions": ["REGISTER", "ROLE_FORMALITY_OR_CHANNEL"],
        },
    })


def _mutate(response: dict[str, Any], fault: str) -> None:
    verdict = response["verdicts"][0]
    distractors = verdict["distractors"]
    if fault == "extra_id":
        distractors["D3"] = {**distractors["D2"], "id": "D3"}
    elif fault == "missing_id":
        distractors.pop("D1")
    elif fault == "unknown_id":
        distractors["X1"] = distractors.pop("D1")
    elif fault == "duplicate_id":
        distractors["D2"]["id"] = "D1"
    elif fault == "swapped_id":
        distractors["D0"], distractors["D1"] = distractors["D1"], distractors["D0"]
    elif fault == "missing_binding":
        distractors["D1"].pop("id")
    elif fault == "wrong_order":
        verdict["globalOrder"] += 1
    elif fault == "bool_order":
        verdict["globalOrder"] = True
    elif fault == "duplicate_order":
        response["verdicts"].append(copy.deepcopy(verdict))
    elif fault == "missing_order":
        response["verdicts"].clear()
    elif fault == "target_as_distractor":
        verdict["targetId"] = "D0"
    elif fault == "missing_boolean":
        distractors["D1"].pop("expressionWellFormed")
    elif fault == "coerced_boolean":
        distractors["D1"]["expressionWellFormed"] = "true"
    elif fault == "extra_boolean":
        distractors["D1"]["naturalTarget"] = True
    elif fault == "legacy_four":
        verdict.pop("targetId")
        verdict["distractors"] = [
            {"index": index, **{key: value for key, value in distractors[f"D{index % 3}"].items() if key != "id"}}
            for index in range(4)
        ]
    else:
        raise AssertionError(fault)


_FAILURES = [
    ("extra_id", "DISTRACTOR_COVERAGE_INVALID"),
    ("missing_id", "DISTRACTOR_COVERAGE_INVALID"),
    ("unknown_id", "DISTRACTOR_COVERAGE_INVALID"),
    ("duplicate_id", "DISTRACTOR_ID_MISMATCH"),
    ("swapped_id", "DISTRACTOR_ID_MISMATCH"),
    ("missing_binding", "DISTRACTOR_ID_MISMATCH"),
    ("wrong_order", "ORDER_MISMATCH"),
    ("bool_order", "ORDER_MISMATCH"),
    ("duplicate_order", "ORDER_COVERAGE_INVALID"),
    ("missing_order", "ORDER_COVERAGE_INVALID"),
    ("target_as_distractor", "TARGET_ID_INVALID"),
    ("missing_boolean", "RESPONSE_SCHEMA_INVALID"),
    ("coerced_boolean", "RESPONSE_SCHEMA_INVALID"),
    ("extra_boolean", "RESPONSE_SCHEMA_INVALID"),
    ("legacy_four", "DISTRACTOR_COVERAGE_INVALID"),
]


@pytest.mark.parametrize(("fault", "reason"), _FAILURES)
def test_verifier_contract_rejects_malformed_identity_without_remapping(fault, reason):
    raw = _valid_response()
    _mutate(raw, fault)
    original = copy.deepcopy(raw)
    with pytest.raises(_ContextualChoiceLexicalVerifierContractError) as caught:
        _parse_contextual_choice_plan_verdict(raw, expected_order=4)
    assert caught.value.reason_code == f"LEXICAL_VERIFIER_{reason}"
    assert caught.value.expected_order == 4
    assert raw == original


def test_provider_conversion_preserves_closed_bound_ids_and_single_current_order():
    schema = _contextual_choice_plan_verification_schema(
        ("REGISTER", "ROLE_FORMALITY_OR_CHANNEL"), global_order=4,
    )
    config = build_openai_text_config(
        type_name=ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME,
        schema=schema,
        verbosity="low",
    )
    assert config["format"]["strict"] is True
    converted = config["format"]["schema"]
    assert converted["additionalProperties"] is False
    verdicts = converted["properties"]["verdicts"]
    assert verdicts["minItems"] == verdicts["maxItems"] == 1
    verdict = verdicts["items"]
    assert verdict["additionalProperties"] is False
    assert set(verdict["required"]) == set(verdict["properties"])
    properties = verdict["properties"]
    assert properties["globalOrder"]["enum"] == [4]
    assert properties["targetId"]["enum"] == ["TARGET"]
    distractors = properties["distractors"]
    assert distractors["type"] == "object"
    assert distractors["additionalProperties"] is False
    assert set(distractors["required"]) == set(distractors["properties"]) == {"D0", "D1", "D2"}
    for target_id, bound in distractors["properties"].items():
        assert bound["additionalProperties"] is False
        assert set(bound["required"]) == set(bound["properties"])
        assert bound["properties"]["id"]["enum"] == [target_id]
        assert bound["properties"]["expressionWellFormed"]["type"] == "boolean"
        assert "index" not in bound["properties"]


def test_unordered_object_fields_retain_identity_and_all_semantic_booleans():
    raw = _valid_response()
    original = raw["verdicts"][0]["distractors"]
    original["D1"]["expressionWellFormed"] = False
    raw["verdicts"][0]["distractors"] = {key: original[key] for key in ("D2", "D0", "D1")}
    verdict = _parse_contextual_choice_plan_verdict(raw, expected_order=4)
    assert [item.index for item in verdict.distractors.ordered()] == [0, 1, 2]
    assert verdict.distractors.D1.expressionWellFormed is False
    assert verdict.distractors.D0.expressionWellFormed is True
    assert verdict.reasonCode == "PASS"  # Diagnostic text never overwrites invalid booleans.


@pytest.mark.asyncio
@pytest.mark.parametrize(("fault", "reason"), _FAILURES)
async def test_invalid_verifier_response_does_not_start_repair_or_context(fault, reason):
    class MalformedProvider(ContextualChoiceV3Provider):
        def _plan_verdicts(self, payload):
            raw = super()._plan_verdicts(payload)
            _mutate(raw, fault)
            return raw

    baseline = await ReadingVocabularyGenerationService(ContextualChoiceV3Provider()).generate(
        _request(vocabulary_plan_only=True),
    )
    assert baseline.vocabulary_plan is not None
    request = _request(vocabulary_plan=baseline.vocabulary_plan.model_dump(mode="json", by_alias=True))
    provider = MalformedProvider()
    with pytest.raises(HTTPException) as caught:
        await ReadingVocabularyGenerationService(provider).generate(request)
    assert caught.value.status_code == 502
    detail = cast(dict[str, Any], caught.value.detail)
    assert detail["code"] == "AI_GENERATION_FAILED"
    assert detail["cause"] == "_ContextualChoiceLexicalVerifierContractError"
    assert isinstance(caught.value.__cause__, _ContextualChoiceLexicalVerifierContractError)
    assert caught.value.__cause__.reason_code == f"LEXICAL_VERIFIER_{reason}"
    assert sum(provider.calls.values()) == 1
    assert provider.operations == {}


@pytest.mark.asyncio
async def test_prompt_separates_target_and_identity_bound_distractors(caplog):
    provider = ContextualChoiceV3Provider()
    with caplog.at_level(logging.INFO):
        response = await ReadingVocabularyGenerationService(provider).generate(_request())
    request = provider.plan_validation_payloads[0]
    targets = request["lexicalTargets"]
    assert targets["target"]["id"] == "TARGET"
    assert targets["target"]["expression"] == response.questions[0].target_expression
    assert set(targets["distractors"]) == {"D0", "D1", "D2"}
    assert response.vocabulary_plan is not None
    item = response.vocabulary_plan.items[0]
    for index, expression in enumerate(item.distractors):
        assert targets["distractors"][f"D{index}"] == {"id": f"D{index}", "expression": expression}
        assert expression not in caplog.text
    assert "distractors" not in request["planItem"]
    assert "targetExpression" not in request["planItem"]


def test_coverage_log_summary_never_contains_provider_text():
    raw = _valid_response()
    distractors = raw["verdicts"][0]["distractors"]
    distractors["sensitive provider text"] = {"id": ["raw text"]}
    summary = _contextual_choice_verifier_coverage_summary(raw)
    assert summary["distractorCount"] == 4
    assert summary["distractorIds"] == ["D0", "D1", "D2", "UNEXPECTED_ID"]
    assert summary["boundIds"] == ["D0", "D1", "D2", "INVALID"]
    assert "sensitive" not in str(summary)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_lexical_provider_cancellation_and_timeout_never_begin_repair(cancel, caplog):
    started = asyncio.Event()

    class DelayedProvider(ContextualChoiceV3Provider):
        async def call(self, type_name, data, schema=None):
            if type_name == ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME:
                self.calls[type_name] += 1
                started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            return await super().call(type_name, data, schema)

    provider = DelayedProvider()
    service = ReadingVocabularyGenerationService(provider, timeout_seconds=30 if cancel else 0.01)
    with caplog.at_level(logging.INFO):
        task = asyncio.create_task(service.generate(_request()))
        await started.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert "stage=lexical_validation status=CANCELLED" in caplog.text
        else:
            with pytest.raises(HTTPException) as caught:
                await task
            assert isinstance(caught.value.__cause__, TimeoutError)
            assert "stage=lexical_validation status=FAILED" in caplog.text
    assert provider.operations["PLAN"] == 1
    assert provider.operations["LEXICAL_REPAIR"] == provider.operations["CONTEXT"] == 0
    assert sum(provider.calls.values()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "initial_close", "preserved_close", "required_indexes", "call_count"),
    [
        ("FULL_LEXICAL_BUNDLE", 1, 0, [0, 1], 6),
        ("FULL_DISTRACTOR_SET", 2, 0, [0, 1], 10),
        ("DISTRACTOR_INDEXES", 1, 1, [2], 6),
        ("DISTRACTOR_INDEXES", 2, 2, [], 6),
    ],
)
async def test_repair_close_requirements_count_only_surviving_competitors(
    scope, initial_close, preserved_close, required_indexes, call_count,
):
    """A discarded close verdict cannot fund a replacement bundle's minimum."""
    request = await _candidate_request_at_order(
        4, "状況を確認する", ["状態を確認する", "状況を確認いたします", "状態を見る"],
    )
    assert request.vocabulary_plan is not None
    before = request.vocabulary_plan.model_dump(mode="json", by_alias=True)

    class ScopeProvider(ContextualChoiceV3Provider):
        def _plan_verdicts(self, payload):
            response = super()._plan_verdicts(payload)
            if len(self.plan_validation_payloads) == 1 and scope == "FULL_LEXICAL_BUNDLE":
                response["verdicts"][0]["learningValue"] = False
            return response

    faults = (
        {0: ("skill",), 2: ("plausible", "competitor", "skill")}
        if scope == "FULL_LEXICAL_BUNDLE"
        else {2: ("competitor",)}
        if scope == "FULL_DISTRACTOR_SET"
        else {1: ("competitor",), 2: ("plausible", "competitor", "skill")}
        if initial_close == 1
        else {2: ("plausible", "competitor", "skill")}
    )
    provider = ScopeProvider(
        distractor_faults=faults,
        context_failure_sequence=(
            [("ambiguous",), ("ambiguous",), ()]
            if scope == "FULL_DISTRACTOR_SET" else None
        ),
    )
    result = await ReadingVocabularyGenerationService(provider).generate(request)

    repair = provider.plan_generation_payloads[0]
    assert repair["repairScope"] == scope
    assert repair["requiredCloseDistractors"] == 2
    assert repair["currentValidCloseDistractorCount"] == preserved_close
    assert repair["additionalCloseDistractorsNeeded"] == 2 - preserved_close
    assert [
        int(index) for index, requirements in repair["repairRequirements"].items()
        if requirements["mustBeCloseCompetitor"]
    ] == required_indexes
    assert repair["invalidDistractorIndexes"] == (
        [2] if scope == "DISTRACTOR_INDEXES" else [0, 1, 2]
    )
    assert len(result.questions) == 1
    assert request.vocabulary_plan.model_dump(mode="json", by_alias=True) == before
    assert result.vocabulary_plan is not None
    assert result.vocabulary_plan.items[:3] == request.vocabulary_plan.items[:3]
    assert result.vocabulary_plan.items[4:] == request.vocabulary_plan.items[4:]
    assert provider.operations["LEXICAL_REPAIR"] == 1
    assert len(provider.plan_validation_payloads) == 2
    assert provider.operations["CONTEXT"] + provider.operations["CONTEXT_REPAIR"] <= 3
    assert sum(provider.calls.values()) == call_count


def test_plan_generation_provider_schema_requires_three_distractors_in_every_slot():
    config = build_openai_text_config(
        type_name=ReadingVocabularyGenerationService.TYPE_NAME,
        schema=_CONTEXTUAL_CHOICE_PLAN_GENERATION_SCHEMA,
        verbosity="low",
    )
    # Preserve this shared generation task's existing non-strict provider mode;
    # application validation remains authoritative even if the model ignores it.
    assert config["format"]["strict"] is False
    distractors = config["format"]["schema"]["properties"]["items"]["items"][
        "properties"
    ]["distractors"]
    assert distractors["type"] == "array"
    assert distractors["minItems"] == distractors["maxItems"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent_empty", [False, True])
async def test_empty_review_distractors_still_require_existing_structural_repair(
    persistent_empty,
):
    class EmptyReviewProvider(ContextualChoiceV3Provider):
        def _plan(self, payload):
            response = super()._plan(payload)
            if payload["operation"] == "PLAN" or persistent_empty:
                for item in response["items"]:
                    if item["globalOrder"] == 1:
                        item["distractors"] = []
            return response

    provider = EmptyReviewProvider()
    request = _request(
        question_count=10, review_targets=_review_targets(), review_question_count=2,
        vocabulary_plan_only=True,
    )
    if persistent_empty:
        with pytest.raises(HTTPException) as caught:
            await ReadingVocabularyGenerationService(provider).generate(request)
        assert caught.value.status_code == 422
        assert cast(dict[str, Any], caught.value.detail)["stage"] == "PLAN_STRUCTURE"
    else:
        result = await ReadingVocabularyGenerationService(provider).generate(request)
        assert result.vocabulary_plan is not None
        assert all(len(item.distractors) == 3 for item in result.vocabulary_plan.items)
        assert result.vocabulary_plan.items[0].target_expression == request.review_targets[0].expression
    assert provider.plan_generation_payloads[1]["repairReasons"] == {
        "1": "plan item requires exactly three distractors",
    }
    assert provider.operations["PLAN"] == provider.operations["PLAN_REPAIR"] == 1
    assert sum(provider.calls.values()) == 2
