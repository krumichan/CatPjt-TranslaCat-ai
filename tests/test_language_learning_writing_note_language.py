"""Learner-note protection is preserved without a routine extra reviewer call."""
from __future__ import annotations

import asyncio
import copy
import json

import pytest
from fastapi import HTTPException

from app.features.language_learning.writing.generation import GENERATION_TASK
from app.features.language_learning.writing.generation_contract import (
    deterministic_draft_reason, parse_draft, review_content_hash,
)
from app.features.language_learning.writing.note_localization import NOTE_LOCALIZATION_TASK
from app.features.language_learning.writing.verification import MINI_NOTE_TASK, MINI_QUALITY_TASK, MINI_DIFFICULTY_TASK
from tests.test_language_learning_writing_verified_generation import draft, request, service
from tests.writing_generation_fakes import WritingPipelineProvider, read_review_data, review_text, reject_issues

FOREIGN_NOTE = "Business register and polite request"
KO_NOTE = "조건과 시간 관계를 자연스럽게 표현하는 연습입니다."


def read_note_data(prompt):
    return json.loads(prompt.split("<writing-note-data>\n", 1)[1].split("\n</writing-note-data>", 1)[0])


class NoteProvider(WritingPipelineProvider):
    def __init__(self, batches, bands, overrides=None, *, note=KO_NOTE, proposal_override=None):
        super().__init__(batches, bands, overrides)
        self.note, self.proposal_override = note, proposal_override

    async def call(self, type_name, data, schema=None):
        if type_name != NOTE_LOCALIZATION_TASK:
            return await super().call(type_name, data, schema)
        self.calls.append({"type_name": type_name, "data": data, "schema": copy.deepcopy(schema)})
        self.counts[type_name] += 1
        bound = read_note_data(data)
        value = {"candidateId": bound["candidateId"], "contentHash": bound["contentHash"], "focusReason": self.note}
        if isinstance(self.proposal_override, Exception):
            raise self.proposal_override
        if callable(self.proposal_override):
            return self.proposal_override(value, bound)
        if self.proposal_override:
            value.update(copy.deepcopy(self.proposal_override))
        return value


def rejection(provider, req, reason=None, status=422):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(service(provider, generation_max_retries=0).generate_daily(req))
    assert exc.value.status_code == status
    if reason:
        assert reason in exc.value.detail.get("rejectionCounts", {}) or reason == exc.value.detail.get("failureCode")
    return exc.value


@pytest.mark.parametrize("mode", ["TRANSLATION", "GUIDED", "FREE"])
def test_valid_note_is_audited_in_the_primary_call_only(mode):
    item = draft(mode=mode)
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3})
    result = asyncio.run(service(provider).generate_daily(request(mode=mode)))
    assert result.items[0].focus_reason == item["focusReason"]
    assert provider.counts == {GENERATION_TASK: 1, MINI_QUALITY_TASK: 1}
    data = read_review_data(provider.calls[1]["data"])
    assert review_text(data, "N1") == item["focusReason"]


@pytest.mark.parametrize("mode", ["TRANSLATION", "GUIDED", "FREE"])
def test_note_language_only_error_repairs_without_rewriting_the_task(mode):
    item = draft(mode=mode)
    item["focusReason"] = FOREIGN_NOTE
    snapshot = copy.deepcopy(item)
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3},
                            {MINI_QUALITY_TASK: reject_issues("NOTE_ORIGIN_LANGUAGE")})
    result = asyncio.run(service(provider).generate_daily(request(mode=mode)))
    assert item == snapshot
    final = result.items[0].model_dump(mode="json", by_alias=True)
    for name in ("originText", "providedFacts", "requiredIntents", "responseConstraints", "keywords", "focusMetrics"):
        assert final[name] == item[name]
    assert final["focusReason"] == KO_NOTE
    assert provider.counts == {GENERATION_TASK: 1, MINI_QUALITY_TASK: 1, NOTE_LOCALIZATION_TASK: 1, MINI_NOTE_TASK: 1}


@pytest.mark.parametrize("mode", ["TRANSLATION", "GUIDED", "FREE"])
def test_script_mismatch_cannot_be_published_even_when_reviewer_says_pass(mode):
    item = draft(mode=mode)
    item["focusReason"] = FOREIGN_NOTE
    assert deterministic_draft_reason(request(mode=mode), parse_draft(item)) is None
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3})
    result = asyncio.run(service(provider).generate_daily(request(mode=mode)))
    assert result.items[0].focus_reason == KO_NOTE
    assert provider.counts[NOTE_LOCALIZATION_TASK] == provider.counts[MINI_NOTE_TASK] == 1


@pytest.mark.parametrize("field", ["originText", "providedFacts", "requiredIntents", "responseConstraints"])
def test_wrong_core_script_is_still_rejected_before_review_or_localization(field):
    item = draft(mode="GUIDED")
    item[field] = "English source" if field == "originText" else ["English guidance"]
    provider = NoteProvider([{"items": [item]}], {})
    expected = {"originText": "ORIGIN_TEXT_SCRIPT_MISMATCH", "providedFacts": "GUIDED_PROVIDED_FACTS_SCRIPT_MISMATCH",
                "requiredIntents": "GUIDED_REQUIRED_INTENTS_SCRIPT_MISMATCH", "responseConstraints": "GUIDED_RESPONSE_CONSTRAINTS_SCRIPT_MISMATCH"}
    rejection(provider, request(mode="GUIDED"), expected[field])
    assert provider.counts == {GENERATION_TASK: 1}


@pytest.mark.parametrize("field", ["originText", "providedFacts", "requiredIntents", "responseConstraints", "focusReason"])
def test_control_characters_remain_hard_failures_including_notes(field):
    item = draft(mode="GUIDED")
    if isinstance(item[field], list):
        item[field][0] += "\u0000"
    else:
        item[field] += "\u0000"
    provider = NoteProvider([{"items": [item]}], {})
    rejection(provider, request(mode="GUIDED"), "CONTROL_CHARACTER")
    assert provider.counts == {GENERATION_TASK: 1}


def test_technical_identifier_in_guidance_is_not_itself_a_language_reject():
    item = draft(mode="GUIDED")
    item["providedFacts"] = ["API", "새 버전은 다음 주부터 사용할 수 있습니다."]
    assert deterministic_draft_reason(request(mode="GUIDED"), parse_draft(item)) is None


@pytest.mark.parametrize("issue", ["ANSWER_LEAK", "UNSUPPORTED_FOCUS_REASON", "INTERNAL_CLAIM", "ORIGIN_LANGUAGE", "AMBIGUOUS_TASK"])
def test_real_defect_cannot_be_overridden_by_note_localization(issue):
    item = draft()
    item["focusReason"] = FOREIGN_NOTE
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3},
                            {MINI_QUALITY_TASK: reject_issues("NOTE_ORIGIN_LANGUAGE", issue)})
    rejection(provider, request(), "QUALITY_" + issue)
    assert provider.counts[NOTE_LOCALIZATION_TASK] == 0


def test_wrong_band_cannot_be_rescued_by_note_repair():
    item = draft()
    item["focusReason"] = FOREIGN_NOTE
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 4},
                            {MINI_QUALITY_TASK: reject_issues("NOTE_ORIGIN_LANGUAGE")})
    rejection(provider, request(), "VERIFIED_BAND_MISMATCH")
    assert provider.counts[NOTE_LOCALIZATION_TASK] == 0


@pytest.mark.parametrize("field,value", [
    ("candidateId", "wrong"), ("contentHash", "0" * 64), ("originText", "새 문제"),
    ("providedFacts", []), ("languageComplexityBand", 3), ("order", 1), ("difficulty", "REVIEW"),
    ("focusReason", 5), ("focusReason", None), ("focusReason", ""), ("focusReason", "   "),
])
def test_note_proposal_cannot_change_task_or_binding_or_break_schema(field, value):
    item = draft()
    item["focusReason"] = FOREIGN_NOTE
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3}, proposal_override={field: value})
    rejection(provider, request(), status=503)
    assert provider.counts[NOTE_LOCALIZATION_TASK] == 1 and provider.counts[MINI_NOTE_TASK] == 0


@pytest.mark.parametrize("note,reason", [
    (FOREIGN_NOTE, "NOTE_LOCALIZATION_UNCHANGED"), ("Another English note", "NOTE_LOCALIZATION_SCRIPT_MISMATCH"),
    ("설명\u0000문장", "NOTE_LOCALIZATION_CONTROL_CHARACTER"), ("가" * 401, "SPEC_NOTE_LENGTH"),
])
def test_repaired_note_is_code_checked_before_audit(note, reason):
    item = draft()
    item["focusReason"] = FOREIGN_NOTE
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3}, note=note)
    rejection(provider, request(), reason)
    assert provider.counts[MINI_NOTE_TASK] == 0


@pytest.mark.parametrize("override,unavailable", [
    ({"candidateId": "wrong"}, True), ({"contentHash": "0" * 64}, True),
    ({"confidence": "0.9"}, True), ({"confidence": True}, True), ({"confidence": float("nan")}, True),
    ({"evidenceSegmentIds": []}, True), ({"evidenceSegmentIds": ["O1"]}, True),
    ({"evidenceSegmentIds": ["N1", "O99"]}, True), ({"issues": ["ANSWER_LEAK"]}, True),
    ({"verdict": "UNSURE"}, False),
])
def test_bad_note_audit_never_implies_approval(override, unavailable):
    item = draft()
    item["focusReason"] = FOREIGN_NOTE
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3}, {MINI_NOTE_TASK: override})
    rejection(provider, request(), status=503 if unavailable else 422)
    assert provider.counts[MINI_NOTE_TASK] == (2 if unavailable else 1)
    assert provider.counts[NOTE_LOCALIZATION_TASK] == 1


@pytest.mark.parametrize("issue", ["ORIGIN_LANGUAGE", "ANSWER_LEAK", "UNSUPPORTED_FOCUS_REASON", "INTERNAL_CLAIM"])
def test_revised_note_defect_stops_without_repeated_proposals(issue):
    item = draft()
    item["focusReason"] = FOREIGN_NOTE
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3},
                            {MINI_NOTE_TASK: {"verdict": "REJECT", "issues": [issue]}})
    rejection(provider, request(), "NOTE_" + issue)
    assert provider.counts[NOTE_LOCALIZATION_TASK] == provider.counts[MINI_NOTE_TASK] == 1


def test_localized_note_audit_is_independent_and_bound_to_new_content():
    item = draft()
    item["focusReason"] = FOREIGN_NOTE
    req = request(learningProfile={"grammarWeaknesses": ["PRIVATE_PROFILE"]})
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3})
    result = asyncio.run(service(provider).generate_daily(req))
    primary = read_review_data(provider.calls[1]["data"])
    audit = read_review_data(next(c["data"] for c in provider.calls if c["type_name"] == MINI_NOTE_TASK))
    assert primary["candidateId"] != audit["candidateId"] and primary["contentHash"] != audit["contentHash"]
    assert review_text(audit, "N1") == KO_NOTE
    final_draft = parse_draft({**item, "focusReason": result.items[0].focus_reason})
    assert audit["contentHash"] == review_content_hash(req, final_draft)
    for call in provider.calls[1:]:
        for secret in ("targetBand", "PRIVATE_PROFILE", "previousVerdict"):
            assert secret not in call["data"]
    audit_prompt = next(c["data"] for c in provider.calls if c["type_name"] == MINI_NOTE_TASK)
    assert "rubric" not in audit_prompt and FOREIGN_NOTE not in audit_prompt


def test_old_note_hash_cannot_attest_to_new_note():
    item = draft()
    item["focusReason"] = FOREIGN_NOTE
    old_hash = review_content_hash(request(), parse_draft(item))
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3}, {MINI_NOTE_TASK: {"contentHash": old_hash}})
    rejection(provider, request(), "VERIFIER_CONTENT_HASH_MISMATCH", status=503)


@pytest.mark.parametrize("confidence", [0.0, 0.72, None])
def test_low_note_confidence_is_diagnostic_not_a_retry(confidence):
    item = draft()
    item["focusReason"] = FOREIGN_NOTE
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3}, {MINI_NOTE_TASK: {"confidence": confidence}})
    assert asyncio.run(service(provider).generate_daily(request())).items[0].focus_reason == KO_NOTE
    assert provider.counts[MINI_NOTE_TASK] == 1 and provider.counts[MINI_DIFFICULTY_TASK] == 0


def test_note_verifier_transient_retry_reuses_new_note_without_regenerating():
    item = draft()
    item["focusReason"] = FOREIGN_NOTE
    provider = NoteProvider([{"items": [item]}], {item["originText"]: 3}, {MINI_NOTE_TASK: [TimeoutError(), {}]})
    assert asyncio.run(service(provider).generate_daily(request())).items[0].focus_reason == KO_NOTE
    assert provider.counts == {GENERATION_TASK: 1, MINI_QUALITY_TASK: 1, NOTE_LOCALIZATION_TASK: 1, MINI_NOTE_TASK: 2}
