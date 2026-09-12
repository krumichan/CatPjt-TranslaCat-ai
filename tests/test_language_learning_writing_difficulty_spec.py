"""Only actual surface measurements are hard checks; semantic labels prove nothing."""
from __future__ import annotations

import copy
import unicodedata
from dataclasses import replace

import pytest

from app.features.language_learning.writing.difficulty_spec import (
    WRITING_DIFFICULTY_SPEC_VERSION, build_difficulty_spec, difficulty_spec_reason, measure_draft, surface_units,
)
from app.features.language_learning.writing.generation_contract import (
    build_candidate_schema, parse_draft, plan_slots, validate_final_items,
)
from tests.test_language_learning_writing_verified_generation import draft, request, run_one


@pytest.mark.parametrize("mode", ["TRANSLATION", "GUIDED", "FREE"])
@pytest.mark.parametrize("base", range(1, 6))
@pytest.mark.parametrize("language", ["ko", "ja", "en", "ko-KR", "ja_JP", "fr"])
def test_spec_is_owned_by_server_and_agrees_with_schema(mode, base, language):
    req = request(mode=mode, band=base, difficulty="NORMAL", originLanguage=language)
    slot = plan_slots(req)[0]
    spec = build_difficulty_spec(req, slot)
    assert spec.version == WRITING_DIFFICULTY_SPEC_VERSION
    assert spec.target_band == base and spec.writing_type == mode
    assert spec.guidance_min_entries == (1 if mode == "GUIDED" else 0)
    schema = build_candidate_schema(request=req, slot=slot)
    fields = schema["$defs"]["WritingDraft"]["properties"]
    assert fields["originText"]["maxLength"] == spec.origin_max_characters
    assert fields["focusReason"]["maxLength"] == spec.note_max_characters
    for name in ("providedFacts", "requiredIntents", "responseConstraints"):
        assert fields[name]["minItems"] == spec.guidance_min_entries
        assert fields[name]["maxItems"] == spec.guidance_max_entries
    assert "semanticRecipe" in spec.generation_payload()
    assert "clauseCount" not in spec.generation_payload()["hardConstraints"]


@pytest.mark.parametrize("text,count", [
    ("", 0), ("안녕하세요.", 1), ("질문입니다. 확인해 주세요.", 2),
    ("가격은 3.5달러입니다. 확인해 주세요!", 2), ("質問です。確認してください。", 2),
    ("질문입니다\n확인해 주세요", 2), ("할 수 있나요?!", 1), ("API v1.2를 확인합니다.", 1),
])
def test_surface_measurement_is_defined_without_calling_it_clause_count(text, count):
    assert surface_units(text) == count


def test_surface_measurements_use_actual_text_not_generator_tags():
    item = draft()
    before = measure_draft(parse_draft(item))
    item["diversityMetadata"]["grammarFocusCodes"] = ["LAYERED_DISCOURSE", "ADVANCED"]
    item["diversityMetadata"]["semanticSummary"] = "수준이 매우 높고 의미가 열 개라고 주장합니다."
    assert measure_draft(parse_draft(item)) == before


def test_nfc_measurement_is_normalization_stable():
    item = draft()
    before = measure_draft(parse_draft(item))
    item["originText"] = unicodedata.normalize("NFD", item["originText"])
    assert measure_draft(parse_draft(item)).origin_characters == before.origin_characters


@pytest.mark.parametrize("field,code", [
    ("originText", "SPEC_ORIGIN_LENGTH"), ("focusReason", "SPEC_NOTE_LENGTH"),
    ("providedFacts", "SPEC_GUIDANCE_ENTRY_LENGTH"),
    ("requiredIntents", "SPEC_GUIDANCE_ENTRY_LENGTH"),
    ("responseConstraints", "SPEC_GUIDANCE_ENTRY_LENGTH"),
])
def test_surface_length_boundary_is_inclusive(field, code):
    req = request(mode="GUIDED")
    spec = build_difficulty_spec(req, plan_slots(req)[0])
    item = draft(mode="GUIDED")
    limit = (spec.origin_max_characters if field == "originText" else spec.note_max_characters
             if field == "focusReason" else spec.guidance_max_characters_per_entry)
    item[field] = "가" * limit if isinstance(item[field], str) else ["가" * limit]
    assert difficulty_spec_reason(parse_draft(item), spec) is None
    item[field] = "가" * (limit + 1) if isinstance(item[field], str) else ["가" * (limit + 1)]
    assert difficulty_spec_reason(parse_draft(item), spec) == code


@pytest.mark.parametrize("base", range(1, 6))
def test_high_band_does_not_require_padding_to_pass_code_checks(base):
    req = request(band=base, difficulty="NORMAL")
    spec = build_difficulty_spec(req, plan_slots(req)[0])
    # It passes surface bounds only; an independent semantic review must still
    # reject an actually too-easy prompt for the high target.
    assert difficulty_spec_reason(parse_draft(draft("확인해 주세요.")), spec) is None


@pytest.mark.parametrize("field", ["providedFacts", "requiredIntents", "responseConstraints"])
def test_guidance_cardinality_is_measured_from_arrays_not_metadata(field):
    req = request(mode="GUIDED", band=1, difficulty="NORMAL")
    spec = build_difficulty_spec(req, plan_slots(req)[0])
    item = draft(mode="GUIDED")
    item[field] = [f"{i}번 항목을 확인합니다." for i in range(spec.guidance_max_entries)]
    assert difficulty_spec_reason(parse_draft(item), spec) is None
    item[field].append("다음 항목입니다.")
    assert difficulty_spec_reason(parse_draft(item), spec) == "SPEC_GUIDANCE_COUNT"


def test_total_guidance_budget_and_surface_units_are_checked():
    req = request(mode="GUIDED", band=5, difficulty="NORMAL")
    spec = build_difficulty_spec(req, plan_slots(req)[0])
    item = draft(mode="GUIDED")
    for key in ("providedFacts", "requiredIntents", "responseConstraints"):
        item[key] = [f"{i}" + "가" * 199 for i in range(4)]
    assert difficulty_spec_reason(parse_draft(item), spec) == "SPEC_GUIDANCE_TOTAL_LENGTH"
    item = draft(mode="GUIDED")
    item["originText"] = "확인합니다. " * (spec.origin_max_surface_units + 1)
    assert difficulty_spec_reason(parse_draft(item), spec) == "SPEC_ORIGIN_SURFACE_UNITS"


@pytest.mark.parametrize("mode", ["GUIDED", "FREE"])
def test_instruction_length_is_not_scaled_up_to_fake_production_difficulty(mode):
    specs = [build_difficulty_spec(req := request(mode=mode, band=i, difficulty="NORMAL"), plan_slots(req)[0])
             for i in range(1, 6)]
    assert len({spec.origin_max_characters for spec in specs}) == 1
    assert len({spec.semantic_recipe for spec in specs}) == 5


@pytest.mark.parametrize("band", [0, 6, 3.0, True, None])
def test_bad_internal_spec_band_is_rejected(band):
    req = request()
    slot = replace(plan_slots(req)[0], target_band=band)
    with pytest.raises(ValueError):
        build_difficulty_spec(req, slot)


def test_spec_and_schema_are_request_isolated():
    req = request()
    slot = plan_slots(req)[0]
    schema = build_candidate_schema(request=req, slot=slot)
    saved = copy.deepcopy(schema)
    schema["$defs"]["WritingDraft"]["properties"]["providedFacts"]["maxItems"] = 99
    assert build_candidate_schema(request=req, slot=slot) == saved


def test_final_public_items_are_rechecked_against_server_spec():
    result, _ = run_one()
    result.items[0].origin_text = "가" * 500
    with pytest.raises(ValueError, match="difficulty spec"):
        validate_final_items(request(), result.items)
