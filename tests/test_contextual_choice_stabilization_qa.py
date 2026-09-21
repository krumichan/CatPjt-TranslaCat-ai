from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import pytest

from app.ai.ports import StructuredGenerationResult
from app.features.language_learning.reading_vocabulary.contextual_choice_task import (
    CONTEXTUAL_CHOICE_ANSWER_KEYS,
    CONTEXTUAL_CHOICE_PLAN_VERSION,
    contextual_choice_answer_key,
    contextual_choice_scenario_family,
    contextual_choice_skill_for_order,
)
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
    _ContextualChoiceLexicalVerifierContractError,
)
from fastapi import HTTPException
from scripts.run_contextual_choice_stabilization_qa import (
    DEFAULT_FIXTURE,
    QaBudgetExceeded,
    QaCallTimeout,
    QaCaps,
    QaOverallTimeout,
    RecordingProvider,
    _call_summary,
    _diagnostic_outcome,
    _safe_call_diagnostic,
    _write_payload,
    build_order_diagnostic_request,
    compare_results,
    dry_run_manifest,
    load_fixture,
    main,
    run_order_diagnostic_live,
    validate_order_diagnostic_case,
    validate_live_args,
)


class _MetadataProvider:
    def __init__(
        self,
        responses: list[StructuredGenerationResult | BaseException],
    ) -> None:
        self.responses = responses
        self.started = 0

    async def call_with_metadata(
        self,
        *,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> StructuredGenerationResult:
        del type_name, data, schema
        self.started += 1
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _BlockingMetadataProvider:
    def __init__(self) -> None:
        self.started = 0

    async def call_with_metadata(
        self,
        *,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> StructuredGenerationResult:
        del type_name, data, schema
        self.started += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def test_dry_run_manifest_constructs_no_live_work_and_preserves_unavailable_case():
    fixture = load_fixture(DEFAULT_FIXTURE)
    manifest = dry_run_manifest(fixture)

    assert manifest["liveProviderCalls"] == 0
    assert len(manifest["profiles"]) == 3
    assert manifest["historicalCases"][0]["status"] == "UNAVAILABLE"
    assert manifest["existingWorstPathApplicationCallCeiling"]["total"] == 366


def test_live_mode_requires_every_explicit_budget_and_output():
    args = argparse.Namespace(
        variant="candidate",
        output=None,
        max_provider_calls=10,
        max_input_tokens=1000,
        max_output_tokens=1000,
        max_call_seconds=5,
        max_seconds=10,
    )
    with pytest.raises(ValueError, match="output"):
        validate_live_args(args)


def test_comparison_keeps_failed_profile_in_rows():
    def result(variant: str, status: str, completed: int) -> dict:
        return {
            "fixtureVersion": "fixture-v1",
            "variant": variant,
            "runs": [
                {
                    "profileId": "p1",
                    "status": status,
                    "completedQuestionCount": completed,
                    "firstQuestionMs": 10.0 if completed else None,
                    "totalMs": 20.0,
                    "calls": {"count": 2, "inputTokens": 3, "outputTokens": 4},
                }
            ],
        }

    comparison = compare_results(
        result("baseline", "FAILED", 0),
        result("candidate", "COMPLETED", 10),
    )

    assert comparison["allRunsIncluded"] is True
    assert comparison["rows"][0]["baselineStatus"] == "FAILED"
    assert comparison["rows"][0]["candidateCompleted"] == 10


def test_default_cli_is_dry_run(capsys):
    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "DRY_RUN"
    assert payload["liveProviderCalls"] == 0


def test_live_cli_refuses_to_construct_provider_without_all_caps(capsys):
    assert main(["--live", "--variant", "candidate"]) == 2
    assert "requires explicit" in capsys.readouterr().err


def test_private_output_does_not_duplicate_diagnostic_text_to_stdout(tmp_path, capsys):
    from scripts.run_contextual_choice_stabilization_qa import _emit

    payload = {"mode": "ORDER_DIAGNOSTIC_LIVE", "status": "FAILED", "partial": True,
               "diagnosticCapture": {"request": "private learner expression", "response": "private verdict"}}
    output = tmp_path / "private.json"
    _emit(payload, output)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    printed = capsys.readouterr().out
    assert "private learner expression" not in printed
    assert "private verdict" not in printed
    assert json.loads(printed)["diagnosticContent"] == "PRIVATE_ARTIFACT_ONLY"


def test_safe_repair_diagnostic_keeps_codes_and_indexes_but_not_content():
    prompt = (
        "Repair\n<practice-data>\n"
        '{"operation":"LEXICAL_REPAIR","repairScope":"DISTRACTOR_INDEXES",'
        '"reasonCodes":["DISTRACTOR_UNNATURAL"],"invalidDistractorIndexes":[1],'
        '"currentPlanItem":{"globalOrder":6,"targetExpression":"秘密の表現"}}'
        "\n</practice-data>"
    )

    diagnostic = _safe_call_diagnostic(prompt)

    assert diagnostic == {
        "operation": "LEXICAL_REPAIR",
        "globalOrder": 6,
        "task": None,
        "repairClass": None,
        "repairTrigger": None,
        "repairScope": "DISTRACTOR_INDEXES",
        "changedIndexes": [1],
        "failureCodes": ["DISTRACTOR_UNNATURAL"],
    }
    assert "秘密の表現" not in json.dumps(diagnostic, ensure_ascii=False)


@pytest.mark.asyncio
async def test_recorder_counts_failed_call_at_start_and_blocks_after_call_cap():
    upstream = _MetadataProvider([RuntimeError("provider failed")])
    recorder = RecordingProvider(
        upstream=upstream,
        caps=QaCaps(
            provider_calls=1,
            input_tokens=100,
            output_tokens=100,
            call_seconds=5,
            seconds=10,
        ),
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        await recorder.call("TASK", "payload")
    with pytest.raises(QaBudgetExceeded, match="provider call cap reached"):
        await recorder.call("TASK", "must not start")

    assert upstream.started == 1
    assert recorder.calls[0]["attempt"] == 1
    assert recorder.calls[0]["status"] == "FAILED"
    assert recorder.calls[0]["failure"] == {"type": "RuntimeError"}
    summary = _call_summary(recorder.calls)
    assert summary["attemptsStarted"] == 1
    assert summary["failedCalls"] == 1


@pytest.mark.asyncio
async def test_recorder_call_timeout_is_ledgered_and_blocks_followup():
    upstream = _BlockingMetadataProvider()
    recorder = RecordingProvider(
        upstream=upstream,
        caps=QaCaps(
            provider_calls=3,
            input_tokens=100,
            output_tokens=100,
            call_seconds=0.01,
            seconds=10,
        ),
        diagnostic_capture=True,
    )
    prompt = (
        "Verify\n<practice-data>\n"
        '{"operation":"PLAN_VERIFY","planItem":{"globalOrder":4,'
        '"targetExpression":"表現","distractors":["甲","乙","丙"]}}'
        "\n</practice-data>"
    )

    with pytest.raises(QaCallTimeout, match="provider call timed out"):
        await recorder.call("VERIFICATION", prompt)
    with pytest.raises(QaBudgetExceeded, match="provider call timed out"):
        await recorder.call("VERIFICATION", prompt)

    assert upstream.started == 1
    call = recorder.calls[0]
    assert call["status"] == "FAILED"
    assert call["failure"] == {
        "type": "QA_CALL_TIMEOUT",
        "timeoutSeconds": 0.01,
    }
    assert call["latencyMs"] > 0
    assert call["task"] == "VERIFICATION"
    assert call["diagnosticCapture"]["operation"] == "PLAN_VERIFY"


@pytest.mark.asyncio
async def test_recorder_records_in_flight_token_overage_and_starts_no_followup():
    upstream = _MetadataProvider(
        [
            StructuredGenerationResult(
                data={"ok": True},
                input_tokens=11,
                output_tokens=2,
                provider="fake",
                model="fake-model",
            )
        ]
    )
    recorder = RecordingProvider(
        upstream=upstream,
        caps=QaCaps(
            provider_calls=3,
            input_tokens=10,
            output_tokens=100,
            call_seconds=5,
            seconds=10,
        ),
    )

    with pytest.raises(QaBudgetExceeded, match="after the last completed call"):
        await recorder.call("TASK", "payload")
    with pytest.raises(QaBudgetExceeded, match="after the last completed call"):
        await recorder.call("TASK", "must not start")

    assert upstream.started == 1
    assert recorder.calls[0]["status"] == "SUCCEEDED"
    assert recorder.calls[0]["budgetExceededAfterCompletion"].startswith(
        "input token cap exceeded"
    )


@pytest.mark.asyncio
async def test_default_recorder_never_retains_candidate_content():
    upstream = _MetadataProvider(
        [
            StructuredGenerationResult(
                data={"contextTemplate": "秘密の候補"},
                provider="fake",
                model="fake",
            )
        ]
    )
    recorder = RecordingProvider(
        upstream=upstream,
        caps=QaCaps(
            provider_calls=2,
            input_tokens=100,
            output_tokens=100,
            call_seconds=5,
            seconds=10,
        ),
    )
    prompt = (
        "Generate\n<practice-data>\n"
        '{"operation":"CONTEXT","fixedLexicalBundle":{"globalOrder":2,'
        '"targetExpression":"秘密の表現"},"difficultyDemand":{}}'
        "\n</practice-data>"
    )

    await recorder.call("TASK", prompt)

    encoded = json.dumps(_call_summary(recorder.calls), ensure_ascii=False)
    assert "diagnosticCapture" not in encoded
    assert "秘密の候補" not in encoded
    assert "秘密の表現" not in encoded


@pytest.mark.asyncio
async def test_explicit_diagnostic_links_candidate_rendering_and_verifier_evidence():
    rejected_before_verifier = {
        "globalOrder": 2,
        "contextTemplate": "{{TARGET}}が{{TARGET}}。",
        "explanationLearning": "説明",
    }
    repaired_candidate = {
        "globalOrder": 2,
        "contextTemplate": "会議では{{TARGET}}ことが重要だ。",
        "explanationLearning": "修正説明",
    }
    verdict = {
        "verdicts": [
            {
                "order": 2,
                "bestAnswerKey": "B",
                "ambiguous": False,
                "supported": True,
                "skillFit": True,
                "definitionLike": False,
                "structurallyWellFormedKeys": ["A", "B", "D"],
                "reason": "C has a structural mismatch",
            }
        ]
    }
    upstream = _MetadataProvider(
        [
            StructuredGenerationResult(
                data=rejected_before_verifier, provider="fake", model="fake"
            ),
            StructuredGenerationResult(
                data=repaired_candidate, provider="fake", model="fake"
            ),
            StructuredGenerationResult(data=verdict, provider="fake", model="fake"),
        ]
    )
    recorder = RecordingProvider(
        upstream=upstream,
        caps=QaCaps(
            provider_calls=3,
            input_tokens=100,
            output_tokens=100,
            call_seconds=5,
            seconds=10,
        ),
        diagnostic_capture=True,
    )
    repair_prompt = (
        "Repair\n<practice-data>\n"
        '{"operation":"CONTEXT_REPAIR","fixedLexicalBundle":{"globalOrder":2,'
        '"targetExpression":"調整する","distractors":["確認する","共有する",'
        '"延期する"]},"difficultyDemand":{"band":4},'
        '"repairClass":"CONTRACT_CONTEXT","reasonCodes":["CONTEXT_MARKER_INVALID"],'
        '"previousCandidate":{"contextTemplate":"{{TARGET}}が{{TARGET}}。"},'
        '"failureEvidence":{"contractReasonCodes":["CONTEXT_MARKER_INVALID"]}}'
        "\n</practice-data>"
    )
    context_prompt = (
        "Generate\n<practice-data>\n"
        '{"operation":"CONTEXT","fixedLexicalBundle":{"globalOrder":2,'
        '"targetExpression":"調整する","distractors":["確認する","共有する",'
        '"延期する"]},"difficultyDemand":{"band":4}}'
        "\n</practice-data>"
    )
    verifier_prompt = "Verify\n\n" + json.dumps(
        {
            "mode": "CONTEXTUAL_CHOICE",
            "question": {
                "order": 2,
                "prompt": "会議では＿＿ことが重要だ。",
                "renderedOptions": [
                    {"key": "A", "sentence": "会議では確認することが重要だ。"},
                    {"key": "B", "sentence": "会議では調整することが重要だ。"},
                    {"key": "C", "sentence": "会議では共有することが重要だ。"},
                    {"key": "D", "sentence": "会議では延期することが重要だ。"},
                ],
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    await recorder.call("GENERATION", context_prompt)
    await recorder.call("GENERATION", repair_prompt)
    await recorder.call("VERIFICATION", verifier_prompt)
    trace = _call_summary(recorder.calls)["linkedContextDiagnostics"]

    assert len(trace) == 2
    assert trace[0]["contextCandidate"] == rejected_before_verifier
    assert trace[0]["verifierResponse"] is None
    assert trace[1]["lexicalBundle"]["targetExpression"] == "調整する"
    assert trace[1]["previousCandidate"] == {
        "contextTemplate": "{{TARGET}}が{{TARGET}}。"
    }
    assert trace[1]["contextCandidate"]["contextTemplate"].startswith("会議では")
    assert len(trace[1]["renderedOptions"]) == 4
    assert trace[1]["judgmentEvidence"]["structurallyWellFormedKeys"] == [
        "A",
        "B",
        "D",
    ]
    assert trace[1]["judgmentEvidence"]["reason"] == "C has a structural mismatch"


def _accepted_prefix(plan: dict[str, Any], count: int) -> list[dict[str, Any]]:
    questions = []
    for item in plan["items"][:count]:
        order = item["globalOrder"]
        answer_key = contextual_choice_answer_key(order)
        options = list(item["distractors"])
        options.insert(
            CONTEXTUAL_CHOICE_ANSWER_KEYS.index(answer_key), item["targetExpression"]
        )
        questions.append(
            {
                "order": order,
                "questionType": "SINGLE_CHOICE",
                "difficulty": item["difficulty"],
                "complexityBand": item["complexityBand"],
                "prompt": f"文脈{order}の______に入る表現を選んでください。",
                "options": [
                    {"key": key, "text": text}
                    for key, text in zip(
                        CONTEXTUAL_CHOICE_ANSWER_KEYS, options, strict=True
                    )
                ],
                "correctAnswer": [answer_key],
                "skillTag": item["skillTag"],
                "explanationOrigin": "설명",
                "explanationLearning": "説明",
                "targetExpression": item["targetExpression"],
                "canonicalKey": item["canonicalKey"],
                "reviewTarget": item["reviewTarget"],
                "vocabularyCandidates": [],
            }
        )
    return questions


def _order_diagnostic_case(
    *,
    practice_set_id: int = 540001,
    target_order: int = 2,
) -> dict[str, Any]:
    plan_items = []
    difficulties = [
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
    ]
    for order in range(1, 11):
        difficulty = difficulties[order - 1]
        plan_items.append(
            {
                "globalOrder": order,
                "reviewTarget": False,
                "targetExpression": f"表現{order}",
                "canonicalKey": f"表現{order}",
                "distractors": [f"候補{order}A", f"候補{order}B", f"候補{order}C"],
                "skillTag": contextual_choice_skill_for_order(order),
                "difficulty": difficulty,
                "complexityBand": {
                    "EASIER": 3,
                    "CURRENT": 4,
                    "CHALLENGE": 5,
                }[difficulty],
                "scenarioFamily": contextual_choice_scenario_family(order - 1),
                "anchorType": "SELECTED_KEYWORD",
                "anchorValue": "협업",
            }
        )
    plan = {"version": CONTEXTUAL_CHOICE_PLAN_VERSION, "items": plan_items}
    snapshot = {
        "requestId": f"set-{practice_set_id}",
        "domain": "VOCABULARY",
        "mode": "CONTEXTUAL_CHOICE",
        "originLanguage": "ko",
        "learningLanguage": "ja",
        "questionCount": 10,
        "complexityBand": 4,
        "easierCount": 2,
        "currentCount": 6,
        "challengeCount": 2,
        "selectedKeywords": ["협업"],
        "weakSignals": [],
        "recentMistakes": [],
        "reviewTargets": [],
        "reviewQuestionCount": 0,
        "generationDate": "2026-09-17",
        "previousQuestions": [],
        "vocabularyPlan": plan,
        "vocabularyPlanOnly": False,
    }
    return {
        "version": "contextual-choice-order-diagnostic-v1",
        "practiceSetId": practice_set_id,
        "targetOrder": target_order,
        "requestSnapshot": snapshot,
        "acceptedPrefix": _accepted_prefix(plan, target_order - 1),
    }


def test_order_diagnostic_requires_exact_snapshot_prefix_and_order_request():
    case = validate_order_diagnostic_case(_order_diagnostic_case())
    request = build_order_diagnostic_request(case)

    assert case["practiceSetId"] == 540001
    assert case["targetOrder"] == 2
    assert len(case["acceptedPrefix"]) == 1
    assert request.request_id == "qa-set-540001-order-2"
    assert request.question_count == 1
    assert request.easier_count == 1
    assert request.previous_questions[0].order == 1
    assert request.vocabulary_plan is not None
    assert (
        request.vocabulary_plan.version
        == case["requestSnapshot"]["vocabularyPlan"]["version"]
    )


def test_order_diagnostic_rejects_prefix_drift():
    case = _order_diagnostic_case()
    case["acceptedPrefix"] = []

    with pytest.raises(ValueError, match="targetOrder minus one"):
        validate_order_diagnostic_case(case)


def test_order_diagnostic_supports_set_570001_order_4_without_changing_slot_band():
    case = validate_order_diagnostic_case(
        _order_diagnostic_case(practice_set_id=570001, target_order=4)
    )
    request = build_order_diagnostic_request(case)
    assert request.request_id == "qa-set-570001-order-4"
    assert request.question_offset == 3
    assert (request.easier_count, request.current_count, request.challenge_count) == (
        0,
        0,
        1,
    )
    assert request.review_question_count == 0
    assert request.complexity_band == 4
    assert request.vocabulary_plan is not None
    slot = request.vocabulary_plan.items[3]
    assert (slot.skill_tag, slot.difficulty.value, slot.complexity_band) == (
        "REGISTER",
        "CHALLENGE",
        5,
    )
    assert contextual_choice_answer_key(4) == "D"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("practiceSetId", True, "practiceSetId"),
        ("practiceSetId", 0, "practiceSetId"),
        ("targetOrder", True, "targetOrder"),
        ("targetOrder", 11, "within requestSnapshot"),
    ],
)
def test_order_diagnostic_rejects_invalid_id_and_order(field, value, message):
    case = _order_diagnostic_case()
    case[field] = value
    with pytest.raises(ValueError, match=message):
        validate_order_diagnostic_case(case)


def test_order_diagnostic_rejects_noncontiguous_prefix_order():
    case = _order_diagnostic_case(practice_set_id=570001, target_order=4)
    case["acceptedPrefix"][1]["order"] = 3
    with pytest.raises(ValueError, match="contiguous"):
        validate_order_diagnostic_case(case)


def test_order_diagnostic_rejects_plan_prefix_identity_mismatch():
    case = _order_diagnostic_case(practice_set_id=570001, target_order=4)
    case["acceptedPrefix"][2]["options"][0]["text"] = "別の表現"
    with pytest.raises(ValueError, match="accepted prefix identity mismatch"):
        validate_order_diagnostic_case(case)


def test_order_diagnostic_mirrors_be_order_two_review_selection():
    case = _order_diagnostic_case()
    reviews = [
        {
            "canonicalKey": "対象甲",
            "expression": "対象甲",
            "preferredSkill": "MEANING",
        },
        {
            "canonicalKey": "対象乙",
            "expression": "対象乙",
            "preferredSkill": "NUANCE",
        },
    ]
    case["requestSnapshot"]["reviewTargets"] = reviews
    case["requestSnapshot"]["reviewQuestionCount"] = 2
    plan_items = case["requestSnapshot"]["vocabularyPlan"]["items"]
    for index, review in enumerate(reviews):
        plan_items[index].update(
            {
                "reviewTarget": True,
                "targetExpression": review["expression"],
                "canonicalKey": review["canonicalKey"],
                "scenarioFamily": None,
                "anchorType": None,
                "anchorValue": None,
            }
        )
    for new_index, item in enumerate(plan_items[2:]):
        item["scenarioFamily"] = contextual_choice_scenario_family(new_index)
    case["acceptedPrefix"] = _accepted_prefix(
        case["requestSnapshot"]["vocabularyPlan"], 1
    )

    validate_order_diagnostic_case(case)
    request = build_order_diagnostic_request(case)

    assert request.review_question_count == 1
    assert request.review_targets[0].canonical_key == "対象乙"
    assert request.review_targets[1].canonical_key == "対象甲"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("termination", "expected_type", "expected_status"),
    [
        (KeyboardInterrupt(), "KeyboardInterrupt", "INTERRUPTED"),
        (asyncio.CancelledError(), "CancelledError", "INTERRUPTED"),
        (RuntimeError("provider failed"), "RuntimeError", "FAILED"),
    ],
)
async def test_order_diagnostic_flushes_partial_ledger_on_termination(
    tmp_path, monkeypatch, termination, expected_type, expected_status
):
    verifier_response = {
        "verdicts": [
            {
                "order": 4,
                "distractors": [
                    {"index": 0, "naturalExpression": True},
                    {"index": 1, "naturalExpression": True},
                    {"index": 2, "naturalExpression": False},
                ],
                "reason": "third distractor is unsuitable",
            }
        ]
    }
    repair_response = {
        "globalOrder": 4,
        "targetExpression": "表現4",
        "distractorReplacements": [{"index": 2, "text": "修正候補"}],
    }
    upstream = _MetadataProvider(
        [
            StructuredGenerationResult(
                data=verifier_response, provider="fake", model="fake-mini"
            ),
            StructuredGenerationResult(
                data=repair_response, provider="fake", model="fake-luna"
            ),
        ]
    )

    async def interrupt_after_captured_repair(self, request):
        del request
        await self.provider.call(
            self.CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME,
            "Verify\n<practice-data>\n"
            '{"operation":"PLAN_VERIFY","planItem":{"globalOrder":4,'
            '"targetExpression":"表現4","distractors":["候補4A",'
            '"候補4B","候補4C"]}}\n</practice-data>',
        )
        repair_request = {
            "operation": "LEXICAL_REPAIR",
            "currentPlanItem": {
                "globalOrder": 4,
                "targetExpression": "表現4",
                "distractors": ["候補4A", "候補4B", "候補4C"],
            },
            "repairTrigger": "VERIFIER_REJECTED",
            "reasonCodes": ["DISTRACTOR_UNNATURAL"],
            "invalidDistractorIndexes": [2],
            "preserveTarget": True,
            "repairScope": "DISTRACTOR_INDEXES",
            "forbiddenExpressions": ["表現4", "候補4A", "候補4B", "候補4C"],
            "distractorEvidence": {"2": {"naturalExpression": False}},
            "requiredCloseDistractors": 2,
            "currentValidCloseDistractorCount": 1,
            "additionalCloseDistractorsNeeded": 1,
            "repairRequirements": {"2": {"mustBeCloseCompetitor": True}},
            "skillDemand": {
                "skill": "REGISTER",
                "decisiveDimensions": ["REGISTER", "ROLE_FORMALITY_OR_CHANNEL"],
            },
        }
        await self.provider.call(
            self.TYPE_NAME,
            "Repair\n<practice-data>\n"
            + json.dumps(repair_request, ensure_ascii=False, separators=(",", ":"))
            + "\n</practice-data>",
        )
        raise termination

    monkeypatch.setattr(
        ReadingVocabularyGenerationService,
        "generate",
        interrupt_after_captured_repair,
    )
    case = validate_order_diagnostic_case(
        _order_diagnostic_case(practice_set_id=570001, target_order=4)
    )
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "partial.json"

    result = await run_order_diagnostic_live(
        case_path,
        case,
        variant="baseline",
        caps=QaCaps(
            provider_calls=12,
            input_tokens=120_000,
            output_tokens=40_000,
            call_seconds=1,
            seconds=10,
        ),
        partial_output=output,
        provider_factory=lambda: upstream,
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == expected_status
    assert result["partial"] is True
    assert persisted["status"] == expected_status
    assert persisted["error"]["type"] == expected_type
    assert persisted["calls"]["attemptsStarted"] == 2
    ledger = persisted["calls"]["attemptLedger"]
    assert ledger[0]["diagnosticCapture"]["operation"] == "PLAN_VERIFY"
    assert ledger[0]["diagnosticCapture"]["response"] == verifier_response
    assert ledger[1]["diagnosticCapture"]["operation"] == "LEXICAL_REPAIR"
    assert ledger[1]["diagnosticCapture"]["request"][
        "invalidDistractorIndexes"
    ] == [2]
    assert ledger[1]["diagnosticCapture"]["request"][
        "additionalCloseDistractorsNeeded"
    ] == 1
    assert ledger[1]["diagnosticCapture"]["request"][
        "forbiddenExpressions"
    ] == ["表現4", "候補4A", "候補4B", "候補4C"]
    assert ledger[1]["diagnosticCapture"]["request"]["repairRequirements"] == {
        "2": {"mustBeCloseCompetitor": True}
    }
    assert ledger[1]["diagnosticCapture"]["request"]["skillDemand"]["skill"] == (
        "REGISTER"
    )
    assert ledger[1]["diagnosticCapture"]["response"] == repair_response


@pytest.mark.asyncio
async def test_provider_timeout_is_not_misreported_as_qa_deadline():
    provider = _MetadataProvider([TimeoutError("provider deadline")])
    recorder = RecordingProvider(provider, QaCaps(3, 100, 100, 90, 600))
    with pytest.raises(TimeoutError, match="provider deadline") as failure:
        await recorder.call("TASK", "payload")
    assert not isinstance(failure.value, QaCallTimeout)
    assert recorder.calls[0]["failure"] == {"type": "TimeoutError", "source": "PROVIDER"}


@pytest.mark.asyncio
async def test_overall_deadline_bounds_inflight_call_separately_from_per_call_limit():
    provider = _BlockingMetadataProvider()
    recorder = RecordingProvider(provider, QaCaps(3, 100, 100, 90, 0.02))
    with pytest.raises(QaOverallTimeout):
        await recorder.call("TASK", "payload")
    with pytest.raises(QaBudgetExceeded):
        await recorder.call("TASK", "must not start")
    assert provider.started == 1
    assert recorder.calls[0]["failure"]["type"] == "QA_OVERALL_TIMEOUT"


@pytest.mark.asyncio
async def test_checkpoint_retains_completed_response_and_inflight_request_before_cancel(
    tmp_path, monkeypatch,
):
    blocked = asyncio.Event()
    output = tmp_path / "checkpoint.json"

    class Provider(_MetadataProvider):
        async def call_with_metadata(self, **kwargs):
            if self.started == 1:
                self.started += 1
                blocked.set()
                await asyncio.Event().wait()
            return await super().call_with_metadata(**kwargs)

    provider = Provider([StructuredGenerationResult(data={"lexical": "captured"})])

    async def generate(self, request):
        await self.provider.call("VERIFY", "first")
        await self.provider.call("REPAIR", "second")
        pytest.fail("cancelled flow must not resume")

    monkeypatch.setattr(ReadingVocabularyGenerationService, "generate", generate)
    case = validate_order_diagnostic_case(_order_diagnostic_case(target_order=4))
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    task = asyncio.create_task(run_order_diagnostic_live(
        case_path, case, variant="candidate", caps=QaCaps(12, 120000, 40000, 90, 600),
        partial_output=output, provider_factory=lambda: provider,
    ))
    await asyncio.wait_for(blocked.wait(), 2)
    # This file exists before finally, including if an external runner kills us.
    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "RUNNING"
    assert checkpoint["partial"] is True
    ledger = checkpoint["calls"]["attemptLedger"]
    assert ledger[0]["diagnosticCapture"]["response"] == {"lexical": "captured"}
    assert ledger[1]["status"] == "STARTED"
    task.cancel()
    result = await task
    assert result["status"] == "INTERRUPTED"
    assert result["outcomeCategory"] == "INTERRUPTED"
    assert result["calls"]["attemptsStarted"] == 2
    last = result["calls"]["attemptLedger"][-1]
    assert last["failure"]["source"] == "EXTERNAL_OR_UPSTREAM_UNDETERMINED"
    assert last["failure"]["taskCancellationCount"] >= 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "INTERRUPTED"


def test_atomic_checkpoint_failed_replace_preserves_previous_artifact(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    _write_payload({"status": "RUNNING", "calls": [1]}, output)

    def fail_replace(source, destination):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr("scripts.run_contextual_choice_stabilization_qa.os.replace", fail_replace)
    with pytest.raises(OSError):
        _write_payload({"status": "RUNNING", "calls": [1, 2]}, output)
    assert json.loads(output.read_text(encoding="utf-8"))["calls"] == [1]
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_checkpoint_retries_transient_windows_sharing_violation(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    _write_payload({"status": "RUNNING", "calls": [1]}, output)
    from scripts import run_contextual_choice_stabilization_qa as qa

    original = qa.os.replace
    starts = 0

    def sharing_once(source, destination):
        nonlocal starts
        starts += 1
        if starts == 1:
            raise PermissionError(5, "temporary sharing violation")
        return original(source, destination)

    monkeypatch.setattr(qa.os, "replace", sharing_once)
    _write_payload({"status": "INTERRUPTED", "calls": [1, 2]}, output)
    assert starts == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "INTERRUPTED", "calls": [1, 2],
    }


@pytest.mark.asyncio
async def test_late_provider_result_cannot_undo_expired_call_deadline():
    class SuppressesCancellation(_MetadataProvider):
        async def call_with_metadata(self, **kwargs):
            self.started += 1
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return StructuredGenerationResult(data={"late": True})
            raise AssertionError("The private event cannot complete normally")

    provider = SuppressesCancellation([])
    recorder = RecordingProvider(provider, QaCaps(3, 100, 100, 0.01, 1))
    with pytest.raises(QaCallTimeout):
        await recorder.call("TASK", "payload")
    with pytest.raises(QaBudgetExceeded):
        await recorder.call("TASK", "must not start")
    assert provider.started == 1
    assert recorder.calls[0]["status"] == "FAILED"


@pytest.mark.asyncio
async def test_exact_token_cap_blocks_next_start():
    provider = _MetadataProvider([StructuredGenerationResult(data={}, input_tokens=10)])
    recorder = RecordingProvider(provider, QaCaps(3, 10, 100, 1, 2))
    await recorder.call("TASK", "payload")
    with pytest.raises(QaBudgetExceeded, match="token cap reached"):
        await recorder.call("TASK", "must not start")
    assert provider.started == 1


@pytest.mark.asyncio
async def test_checkpoint_exhausting_wall_budget_never_enters_provider():
    provider = _MetadataProvider([])
    recorder = RecordingProvider(provider, QaCaps(3, 100, 100, 90, 600))

    def consume_remaining_budget():
        recorder.started_at -= 601

    recorder.checkpoint = consume_remaining_budget
    with pytest.raises(QaOverallTimeout):
        await recorder.call("TASK", "payload")
    assert provider.started == 0
    assert recorder.calls[0]["failure"]["source"] == "BEFORE_PROVIDER_ENTRY"


def test_diagnostic_separates_contract_failure_from_content_rejection():
    contract = _ContextualChoiceLexicalVerifierContractError(
        "LEXICAL_VERIFIER_DISTRACTOR_COVERAGE_INVALID", expected_order=4,
    )
    outer = HTTPException(status_code=502, detail={"code": "AI_GENERATION_FAILED"})
    outer.__cause__ = contract
    assert _diagnostic_outcome(outer, "FAILED") == "VERIFIER_CONTRACT_FAILURE"
    assert _diagnostic_outcome(HTTPException(status_code=422), "FAILED") == "CONTENT_REJECTED"
    assert _diagnostic_outcome(TimeoutError(), "FAILED") == "INFRA_FAILURE"
    assert _diagnostic_outcome(asyncio.CancelledError(), "INTERRUPTED") == "INTERRUPTED"


@pytest.mark.asyncio
async def test_suppressed_external_cancellation_cannot_continue_diagnostic():
    started = asyncio.Event()

    class SuppressesCancellation(_MetadataProvider):
        async def call_with_metadata(self, **kwargs):
            self.started += 1
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return StructuredGenerationResult(data={"late": True})
            raise AssertionError("The private event cannot complete normally")

    provider = SuppressesCancellation([])
    recorder = RecordingProvider(provider, QaCaps(3, 100, 100, 90, 600))
    task = asyncio.create_task(recorder.call("TASK", "payload"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(QaBudgetExceeded):
        await recorder.call("TASK", "must not start")
    assert provider.started == 1
    assert recorder.calls[0]["status"] == "CANCELLED"
