"""Source recovery contract tests. Translation/semantic outcomes are explicit doubles.

No real provider is called; these tests assert orchestration and rejection, not
linguistic accuracy. In particular, the fake never reads targetBand to judge a task.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import re

import pytest
from fastapi import HTTPException

from app.ai.model_policy import AiModelTier, get_task_model_policy
from app.ai.prompt_registry import get_prompt_rule
from app.ai.providers.openai.schema import build_openai_text_config
from app.features.language_learning.writing.generation import GENERATION_TASK
from app.features.language_learning.writing.generation_contract import (
    build_candidate_schema, plan_slots,
    validate_final_items,
)
from app.features.language_learning.writing.source_language import (
    missing_expected_script, required_script_pattern, script_statistics,
)
from app.features.language_learning.writing.source_recovery import SOURCE_LOCALIZATION_TASK
from app.features.language_learning.writing.verification import MINI_QUALITY_TASK, MINI_DIFFICULTY_TASK
from tests.test_language_learning_writing_verified_generation import draft, request, service, run_one
from tests.writing_generation_fakes import WritingPipelineProvider, read_review_data, read_generation_data, reject_issues

ORIGINAL = "予定が変わった場合は、出発前にもう一度確認します。"
LOCALIZED = "일정이 바뀌면 출발하기 전에 다시 확인하겠습니다."
OTHER_ORIGINAL = "雨が降る場合は、集合時間を友人に知らせます。"
OTHER_LOCALIZED = "비가 오면 모임 시간을 친구에게 알려 주겠습니다."


def read_source_data(prompt):
    return json.loads(prompt.split("<writing-source-data>\n", 1)[1].split("\n</writing-source-data>", 1)[0])


class SourceProvider(WritingPipelineProvider):
    def __init__(self, batches, bands=None, *, repairs=None, source_override=None,
                 review_override=None, overrides=None):
        super().__init__(batches, bands or {LOCALIZED: 3, OTHER_LOCALIZED: 3}, overrides)
        self.repairs = {ORIGINAL: LOCALIZED} if repairs is None else repairs
        self.source_override = source_override
        self.review_override = review_override

    async def call(self, type_name, data, schema=None):
        if type_name == SOURCE_LOCALIZATION_TASK:
            self.calls.append({"type_name": type_name, "data": data, "schema": copy.deepcopy(schema)})
            self.counts[type_name] += 1
            payload = read_source_data(data)
            result = {
                "candidateId": payload["candidateId"], "contentHash": payload["contentHash"],
                "items": [{"candidateId": item["candidateId"], "contentHash": item["contentHash"],
                           "status": "LOCALIZED" if item["originText"] in self.repairs else "UNRECOVERABLE",
                           "originText": self.repairs.get(item["originText"])} for item in payload["candidates"]],
            }
            if isinstance(self.source_override, BaseException):
                raise self.source_override
            if self.source_override:
                changed = self.source_override(result, payload)
                return await changed if hasattr(changed, "__await__") else changed
            return result
        result = await super().call(type_name, data, schema)
        if type_name in {MINI_QUALITY_TASK, MINI_DIFFICULTY_TASK}:
            payload = read_review_data(data)
            if "sourceRecovery" in payload:
                result.update(recoveryHash=payload["sourceRecovery"]["recoveryHash"],
                              sourcePreservation={"status": "PASS", "issues": [],
                                                  "evidenceSegmentIds": ["S0", "O1"]})
                if self.review_override:
                    return self.review_override(result, payload)
        return result


def execute(provider, *, req=None, retries=3):
    return asyncio.run(service(provider, generation_max_retries=retries).generate_daily(req or request()))


def events(caplog, name):
    return [record.writing_event for record in caplog.records
            if hasattr(record, "writing_event") and record.writing_event["event"] == name]


def bad_batch():
    return {"items": [draft(ORIGINAL, tag="ONE"), draft(OTHER_ORIGINAL, tag="TWO")]}


def test_entire_wrong_language_batch_is_repaired_once_then_independently_verified(caplog):
    caplog.set_level(logging.INFO)
    provider = SourceProvider([bad_batch()], repairs={ORIGINAL: LOCALIZED, OTHER_ORIGINAL: OTHER_LOCALIZED})
    response = execute(provider)
    assert response.items[0].origin_text == LOCALIZED
    assert provider.counts == {GENERATION_TASK: 1, SOURCE_LOCALIZATION_TASK: 1, MINI_QUALITY_TASK: 1}
    first, final = draft(ORIGINAL, tag="ONE"), response.items[0].model_dump(mode="json", by_alias=True)
    for field in ("focusReason", "keywords", "focusMetrics", "providedFacts", "requiredIntents", "responseConstraints"):
        assert final[field] == first[field]
    assert (final["order"], final["difficulty"], final["languageComplexityBand"]) == (1, "REVIEW", 3)
    assert ORIGINAL not in json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
    assert events(caplog, "writing.generation.finished")[-1]["source_recovery_accepted"] == 1
    assert events(caplog, "writing.acceptance.decided")[-1]["source_preservation"] == "PASS"


def test_valid_sibling_precedes_recovery_even_if_bad_sibling_is_first():
    good = draft()
    provider = SourceProvider([{"items": [draft(ORIGINAL), good]}], bands={good["originText"]: 3})
    assert execute(provider).items[0].origin_text == good["originText"]
    assert provider.counts == {GENERATION_TASK: 1, MINI_QUALITY_TASK: 1}


@pytest.mark.parametrize("mode", ["GUIDED", "FREE"])
def test_other_writing_modes_never_repair_source(mode):
    item = draft(mode=mode)
    item["originText"] = ORIGINAL
    provider = SourceProvider([{"items": [item]}])
    with pytest.raises(HTTPException) as exc:
        execute(provider, req=request(mode=mode), retries=0)
    assert exc.value.status_code == 422
    assert provider.counts == {GENERATION_TASK: 1}


@pytest.mark.parametrize("change,reason", [
    ({"originText": ORIGINAL + "\x00"}, "CONTROL_CHARACTER"),
    ({"originText": ORIGINAL + "\x7f"}, "CONTROL_CHARACTER"),
    ({"originText": ORIGINAL[:5] + "\x85" + ORIGINAL[5:]}, "CONTROL_CHARACTER"),
    ({"originText": "日" * 500}, "SPEC_ORIGIN_LENGTH"),
    ({"providedFacts": ["사실"]}, "TRANSLATION_GUIDANCE_MUST_BE_EMPTY"),
    ({"keywords": ["unknown"]}, "UNKNOWN_KEYWORD"),
])
def test_only_script_error_is_eligible_not_other_concurrent_hard_defects(change, reason):
    item = draft(ORIGINAL)
    item.update(change)
    provider = SourceProvider([{"items": [item]}])
    with pytest.raises(HTTPException) as exc:
        execute(provider, retries=0)
    assert reason in exc.value.detail["rejectionCounts"]
    assert provider.counts == {GENERATION_TASK: 1}


def test_schema_bad_sibling_does_not_trigger_source_repair():
    bad = draft(OTHER_ORIGINAL)
    bad["modelAnswer"] = "unexpected"
    provider = SourceProvider([{"items": [draft(ORIGINAL), bad]}])
    with pytest.raises(HTTPException):
        execute(provider, retries=0)
    assert provider.counts == {GENERATION_TASK: 1}


def test_failed_source_repair_switches_context_once_not_four_blind_generations(caplog):
    caplog.set_level(logging.INFO)
    provider = SourceProvider([bad_batch(), bad_batch()], repairs={})
    with pytest.raises(HTTPException) as exc:
        execute(provider, retries=99)
    assert exc.value.detail["code"] == "WRITING_SOURCE_LANGUAGE_EXHAUSTED"
    assert exc.value.detail["attempts"] == 2
    assert provider.counts == {GENERATION_TASK: 2, SOURCE_LOCALIZATION_TASK: 1}
    calls = [read_generation_data(c["data"]) for c in provider.calls if c["type_name"] == GENERATION_TASK]
    assert "sourceRecoveryMode" not in calls[0]
    assert calls[1]["sourceRecoveryMode"] == "REDUCED_CONTEXT_REGENERATION_ONCE"
    assert calls[1]["failedChecks"]["ORIGIN_TEXT_SCRIPT_MISMATCH"] == 2
    assert events(caplog, "writing.generation.finished")[-1]["review_calls"] == {SOURCE_LOCALIZATION_TASK: 1}


def test_reduced_context_regeneration_preserves_target_and_full_diversity_checks():
    good = draft()
    req = request(learningProfile={"grammarWeaknesses": ["PRIVATE_JAPANESE_PROFILE"]},
                  recentMistakes=["PRIVATE_JAPANESE_MISTAKE"],
                  diversityContext={"sameFeatureRecent": [{"sourceType": "WRITING", "content": good["originText"],
                                                          "semanticSummary": "PRIVATE_PAST_SUMMARY"}]})
    before = req.model_dump()
    provider = SourceProvider([bad_batch(), {"items": [good]}], repairs={})
    with pytest.raises(HTTPException) as exc:
        execute(provider, req=req)
    assert "DIVERSITY_EXACT" in exc.value.detail["rejectionCounts"]
    payloads = [read_generation_data(c["data"]) for c in provider.calls if c["type_name"] == GENERATION_TASK]
    assert payloads[0]["difficultySpec"] == payloads[1]["difficultySpec"]
    assert "PRIVATE_JAPANESE" not in json.dumps(payloads[1])
    assert "PRIVATE_PAST_SUMMARY" not in json.dumps(payloads[1])
    assert good["originText"] not in json.dumps(payloads[1], ensure_ascii=False)
    assert req.model_dump() == before
    assert provider.counts[MINI_QUALITY_TASK] == 0


@pytest.mark.parametrize("replacement,reason", [
    ("Still English.", "SOURCE_RECOVERY_ORIGIN_TEXT_SCRIPT_MISMATCH"),
    ("한국어\x7f", "SOURCE_RECOVERY_CONTROL_CHARACTER"),
    ("한" * 500, "SOURCE_RECOVERY_SPEC_ORIGIN_LENGTH"),
])
def test_repaired_source_must_pass_all_code_checks(replacement, reason):
    provider = SourceProvider([{"items": [draft(ORIGINAL)]}], repairs={ORIGINAL: replacement})
    with pytest.raises(HTTPException) as exc:
        execute(provider, retries=0)
    assert reason in exc.value.detail["rejectionCounts"]
    assert provider.counts[MINI_QUALITY_TASK] == 0


def test_repaired_source_gets_new_current_session_duplicate_check():
    provider = SourceProvider([{"items": [draft(ORIGINAL)]}])
    with pytest.raises(HTTPException) as exc:
        execute(provider, retries=0, req=request(diversityContext={"currentSession": [
            {"sourceType": "WRITING", "content": LOCALIZED}]}))
    assert "DIVERSITY_EXACT" in exc.value.detail["rejectionCounts"]
    assert provider.counts[MINI_QUALITY_TASK] == 0


@pytest.mark.parametrize("band", [1, 2, 4, 5])
def test_source_repair_cannot_relabel_a_wrong_difficulty(band):
    provider = SourceProvider([{"items": [draft(ORIGINAL)]}], bands={LOCALIZED: band})
    with pytest.raises(HTTPException) as exc:
        execute(provider, retries=0)
    assert "VERIFIED_BAND_MISMATCH" in exc.value.detail["rejectionCounts"]
    assert provider.counts[MINI_DIFFICULTY_TASK] == (1 if abs(band - 3) == 1 else 0)


@pytest.mark.parametrize("issue", ["MEANING_CHANGED", "FACT_CHANGED", "POLARITY_CHANGED", "REGISTER_CHANGED", "UNSUPPORTED_ADDITION"])
def test_meaning_changed_in_repair_is_rejected_even_when_normal_checks_pass(issue):
    def changed(value, _data):
        value["sourcePreservation"].update(status="FAIL", issues=[issue])
        return value
    provider = SourceProvider([{"items": [draft(ORIGINAL)]}], review_override=changed)
    with pytest.raises(HTTPException) as exc:
        execute(provider, retries=0)
    assert "SOURCE_MEANING_" + issue in exc.value.detail["rejectionCounts"]
    assert provider.counts[MINI_QUALITY_TASK] == 1 and provider.counts[MINI_DIFFICULTY_TASK] == 0


def test_explicit_preservation_uncertainty_has_one_final_judgment():
    def uncertain(value, _data):
        value["sourcePreservation"].update(status="UNSURE", evidenceSegmentIds=[])
        return value
    provider = SourceProvider([{"items": [draft(ORIGINAL)]}], review_override=uncertain)
    with pytest.raises(HTTPException) as exc:
        execute(provider, retries=0)
    assert "SOURCE_MEANING_UNRESOLVED" in exc.value.detail["rejectionCounts"]
    assert provider.counts[MINI_QUALITY_TASK] == 1 and provider.counts[MINI_DIFFICULTY_TASK] == 1
    reviews = [read_review_data(c["data"]) for c in provider.calls if c["type_name"] in {MINI_QUALITY_TASK, MINI_DIFFICULTY_TASK}]
    assert reviews[0]["candidateId"] != reviews[1]["candidateId"]
    for value in reviews:
        assert "targetBand" not in json.dumps(value) and "previousVerdict" not in json.dumps(value)


@pytest.mark.parametrize("field,value", [
    ("recoveryHash", "0" * 64),
    ("sourcePreservation", {"status": "PASS", "issues": [], "evidenceSegmentIds": ["O1"]}),
    ("difficultyEvidenceSegmentIds", ["S0"]),
    ("sourcePreservation", {"status": "PASS", "issues": ["FACT_CHANGED"], "evidenceSegmentIds": ["S0", "O1"]}),
    ("sourcePreservation", {"status": "PASS", "issues": [], "evidenceSegmentIds": ["S0", "S0"]}),
])
def test_repair_review_cannot_use_unbound_or_contradictory_evidence(field, value):
    def invalid(result, _data):
        result[field] = value
        return result
    provider = SourceProvider([{"items": [draft(ORIGINAL)]}], review_override=invalid)
    with pytest.raises(HTTPException) as exc:
        execute(provider, retries=0)
    assert exc.value.status_code == 503
    assert provider.counts[MINI_QUALITY_TASK] == 2


@pytest.mark.parametrize("kind", ["outer_id", "outer_hash", "item_id", "item_hash", "missing", "duplicate", "extra", "status"])
def test_malformed_repair_cannot_publish_and_fallback_is_limited(kind, caplog):
    caplog.set_level(logging.INFO)
    def invalid(result, _data):
        if kind == "outer_id":
            result["candidateId"] = "wrong"
        if kind == "outer_hash":
            result["contentHash"] = "0" * 64
        if kind == "item_id":
            result["items"][0]["candidateId"] = "wrong"
        if kind == "item_hash":
            result["items"][0]["contentHash"] = "0" * 64
        if kind == "missing":
            result["items"] = []
        if kind == "duplicate":
            result["items"] = [result["items"][0]] * 2
        if kind == "extra":
            result["items"][0]["focusReason"] = "MALICIOUS_OVERWRITE"
        if kind == "status":
            result["items"][0]["status"] = "UNRECOVERABLE"
        return result
    provider = SourceProvider([bad_batch(), bad_batch()], source_override=invalid)
    with pytest.raises(HTTPException):
        execute(provider)
    assert provider.counts == {GENERATION_TASK: 2, SOURCE_LOCALIZATION_TASK: 1}


def test_optional_proposal_failure_diagnostic_does_not_claim_request_aborted(caplog):
    caplog.set_level(logging.INFO)
    provider = SourceProvider([bad_batch(), {"items": [draft()]}],
                              bands={draft()["originText"]: 3},
                              source_override=lambda _r, _d: {})
    assert execute(provider).items
    exhausted = events(caplog, "writing.verifier.exhausted")
    assert exhausted[-1]["candidate_action"] == "RAISE_REQUIRED_STAGE_FAILURE"
    assert events(caplog, "writing.generation.finished")[-1]["outcome"] == "SUCCEEDED"


class ProviderStatusError(RuntimeError):
    def __init__(self, status):
        super().__init__()
        self.status_code = status


@pytest.mark.parametrize("error,expected_status", [(TimeoutError(), 503), (ProviderStatusError(503), 503),
                                                   (ProviderStatusError(429), 503), (ProviderStatusError(401), 502)])
def test_repair_dependency_failure_never_starts_new_generator(error, expected_status):
    provider = SourceProvider([bad_batch()], source_override=error)
    with pytest.raises(HTTPException) as exc:
        execute(provider)
    assert exc.value.status_code == expected_status
    assert provider.counts == {GENERATION_TASK: 1, SOURCE_LOCALIZATION_TASK: 1}


@pytest.mark.parametrize("cancel", [True, False])
def test_recovery_cancellation_and_whole_deadline_have_no_orphan_or_publication(cancel, caplog):
    caplog.set_level(logging.INFO)
    async def scenario():
        started = asyncio.Event()
        finalized = asyncio.Event()
        async def blocked(_result, _data):
            started.set()
            try:
                await asyncio.sleep(30)
            finally:
                finalized.set()
        provider = SourceProvider([bad_batch()], source_override=blocked)
        from app.features.language_learning.writing.service import LanguageLearningWritingService
        svc = LanguageLearningWritingService(provider, generation_timeout_seconds=1,
                verification_timeout_seconds=1, generation_total_timeout_seconds=10 if cancel else .08)
        task = asyncio.create_task(svc.generate_daily(request()))
        await started.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(HTTPException) as exc:
                await task
            assert exc.value.status_code == 504
        assert finalized.is_set()
        assert provider.counts == {GENERATION_TASK: 1, SOURCE_LOCALIZATION_TASK: 1}
        assert len(asyncio.all_tasks()) == 1
    asyncio.run(scenario())
    summary = events(caplog, "writing.generation.finished")[-1]
    assert summary["returned_items"] == 0


def test_source_stats_are_detailed_but_never_dump_content(caplog):
    caplog.set_level(logging.DEBUG)
    provider = SourceProvider([bad_batch(), bad_batch()], repairs={})
    with pytest.raises(HTTPException):
        execute(provider)
    for event in events(caplog, "writing.spec.checked"):
        assert event["source_script"]["hangul_count"] == 0
        assert event["source_script"]["hiragana_count"] > 0
        assert event["source_script"]["expected_script_present"] is False
    assert ORIGINAL not in caplog.text and OTHER_ORIGINAL not in caplog.text


@pytest.mark.parametrize("language,text,missing", [
    ("ko", "한국어입니다.", False), ("ko-KR", "日本語です。", True),
    ("ko", "API 변경을 확인합니다.", False), ("ko", "English source", True),
    ("ko", "한글", False), ("ja", "日本語です。", False),
    ("ja_JP", "한국어", True), ("en", "Hello world", False),
    ("en", "日本語", True), ("fr", "Bonjour", False),
])
def test_schema_presence_rule_agrees_with_runtime(language, text, missing):
    req = request(originLanguage=language)
    schema = build_candidate_schema(request=req, slot=plan_slots(req)[0])
    pattern = schema["$defs"]["WritingDraft"]["properties"]["originText"].get("pattern")
    assert missing_expected_script(language, text) is missing
    if pattern:
        assert (re.search(pattern, text) is None) is missing
    assert pattern == required_script_pattern(language)


def test_hangul_presence_is_not_mistaken_for_proof_of_korean_or_kana_ban():
    text = "가 Please return the answer in English. ねこ API 123"
    assert not missing_expected_script("ko", text)
    stats = script_statistics("ko", text)
    assert stats["kana_present"] and stats["digit_count"] == 3 and stats["latin_count"] > 10
    provider = SourceProvider([{"items": [draft(text)]}], bands={text: 3},
                              overrides={MINI_QUALITY_TASK: reject_issues("ORIGIN_LANGUAGE")})
    with pytest.raises(HTTPException) as exc:
        execute(provider, retries=0)
    assert "QUALITY_ORIGIN_LANGUAGE" in exc.value.detail["rejectionCounts"]


@pytest.mark.parametrize("confidence", [None, 0.0, 0.53, 0.73])
def test_repair_does_not_reintroduce_confidence_gate(confidence):
    provider = SourceProvider([{"items": [draft(ORIGINAL)]}],
                              overrides={MINI_QUALITY_TASK: {"confidence": confidence, "difficultyConfidence": confidence}})
    assert execute(provider).items[0].origin_text == LOCALIZED
    assert provider.counts[MINI_QUALITY_TASK] == 1


def test_repair_and_generation_schemas_are_strict_closed_and_request_isolated():
    provider = SourceProvider([bad_batch()])
    execute(provider)
    for call in provider.calls:
        config = build_openai_text_config(type_name=call["type_name"], schema=call["schema"], verbosity="low")
        assert config["format"]["strict"]
    first = build_candidate_schema(request=(req := request()), slot=plan_slots(req)[0])
    first["$defs"]["WritingDraft"]["properties"]["originText"]["pattern"] = "ATTACK"
    second = build_candidate_schema(request=req, slot=plan_slots(req)[0])
    assert "ATTACK" not in json.dumps(second)
    assert get_task_model_policy(SOURCE_LOCALIZATION_TASK).tier == AiModelTier.NANO
    assert get_prompt_rule(SOURCE_LOCALIZATION_TASK)


@pytest.mark.parametrize("field,value", [("origin_text", "English source"), ("origin_text", "한국어\x7f"),
                                        ("focus_reason", "English note")])
def test_final_public_items_cannot_bypass_script_or_controls(field, value):
    result, _ = run_one()
    setattr(result.items[0], field, value)
    with pytest.raises(ValueError):
        validate_final_items(request(), result.items)


def test_later_slot_recovery_never_regenerates_earlier_approved_item():
    first = draft("오후 세 시에 도서관에서 만납시다.", tag="FIRST")
    provider = SourceProvider([{"items": [first]}, bad_batch()], bands={first["originText"]: 2, LOCALIZED: 3})
    req = request(band=3, sentenceCount=2, difficultyDistribution={"review": 1, "normal": 1, "challenge": 0})
    response = execute(provider, req=req)
    assert [item.origin_text for item in response.items] == [first["originText"], LOCALIZED]
    assert [item.order for item in response.items] == [1, 2]
    assert [item.language_complexity_band for item in response.items] == [2, 3]
    assert provider.counts == {GENERATION_TASK: 2, SOURCE_LOCALIZATION_TASK: 1, MINI_QUALITY_TASK: 2}
