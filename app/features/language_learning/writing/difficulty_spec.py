"""Writing reference policy: measurable surface limits != linguistic difficulty.

Only observed string lengths, punctuation-delimited units and array cardinalities are
hard checked here. Intent, clause structure, meaning units, register and lexical
relevance stay semantic questions; generator-authored tags are not evidence.
These limits are versioned editorial bounds, NOT empirically calibrated learner levels.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from app.schemas.language_learning import DailyWritingGenerationRequest, DailyWritingType

if TYPE_CHECKING:
    from app.features.language_learning.writing.generation_contract import WritingDraft, WritingSlot
    from app.schemas.language_learning import DailyWritingItem

WRITING_DIFFICULTY_SPEC_VERSION = "writing-difficulty-spec-v2"

# Shared instructional anchors: generator sees the selected anchor; reviewers see
# ALL anchors and never the selected one. No external qualification is implied.
WRITING_BAND_RUBRIC = {
    1: "FOUNDATION: one direct familiar statement/question; basic words and simple production.",
    2: "BASIC: familiar communication with one simple reason, sequence or ordinary polite request; limited linking.",
    3: ("INTERMEDIATE: a connected practical reason, condition or time relation with explicit, "
        "straightforward scope and suitable everyday/work register. Ordinary politeness alone "
        "does not make this band 4; do not require layered qualifications for this band."),
    4: ("UPPER_INTERMEDIATE: a nuanced condition/concession, indirect request or hedge that needs "
        "register-sensitive phrasing. A longer practical request or several independent steps do "
        "not by themselves reach band 5."),
    5: ("ADVANCED: interacting relations whose scope must be preserved together, a precisely "
        "bounded stance/claim/commitment, and coherent controlled register. For example, "
        "a commitment depends on a condition but is limited by an exception, or evidence "
        "supports only a qualified conclusion that constrains a further action. All needed "
        "facts must be supplied or freely chosen by the learner as the mode allows. Removing "
        "a qualification changes the claim, not just the sentence length. Familiar vocabulary "
        "is sufficient; technical knowledge, verbosity and isolated advanced phrases are not "
        "evidence of band 5. A short message can qualify if these demands are actually present."),
}

# Upper bounds rather than arbitrary minimum lengths: a short nuanced sentence can
# be difficult, and an easy long sentence must not be promoted just for its length.
_CJK_ORIGIN_MAX = (160, 240, 420, 650, 950)
_SURFACE_UNIT_MAX = (2, 3, 4, 5, 6)
_GUIDANCE_ENTRY_MAX = (3, 4, 5, 6, 8)


def language_base(language: str) -> str:
    return language.replace("_", "-").split("-", 1)[0].lower()


def visible_length(text: str) -> int:
    return len(unicodedata.normalize("NFC", text).strip())


def surface_units(text: str) -> int:
    """Deterministic surface measurement, not a clause/intent/sentence parser.

    A decimal point is not a separator. Abbreviations may count as more units;
    the intentionally broad cap is an editorial bound, not a difficulty score.
    """
    return sum(bool(part.strip()) for part in re.split(
        r"[。！？!?]+|(?<!\d)\.(?=\s|$)|\r?\n+", text.strip(),
    ))


@dataclass(frozen=True)
class WritingDifficultySpec:
    version: str
    target_band: int
    writing_type: str
    origin_max_characters: int
    origin_max_surface_units: int
    guidance_min_entries: int
    guidance_max_entries: int
    guidance_max_characters_per_entry: int
    guidance_max_total_characters: int
    note_max_characters: int
    semantic_recipe: str

    def generation_payload(self) -> dict[str, Any]:
        return {
            "policyVersion": self.version,
            "targetBand": self.target_band,
            "writingType": self.writing_type,
            "hardConstraints": {
                "originMaxCharacters": self.origin_max_characters,
                "originMaxSurfaceUnits": self.origin_max_surface_units,
                "guidanceMinEntriesPerGroup": self.guidance_min_entries,
                "guidanceMaxEntriesPerGroup": self.guidance_max_entries,
                "guidanceMaxCharactersPerEntry": self.guidance_max_characters_per_entry,
                "guidanceMaxTotalCharacters": self.guidance_max_total_characters,
                "noteMaxCharacters": self.note_max_characters,
                "characterMeasure": "NFC Unicode code points including internal spaces",
                "surfaceUnitMeasure": "nonempty punctuation/newline-delimited units; NOT clauses",
            },
            "semanticRecipe": self.semantic_recipe,
            "semanticRecipeEnforcement": "independent reviewer, not tag counting or regex",
        }


@dataclass(frozen=True)
class WritingSurfaceMeasurements:
    origin_characters: int
    origin_surface_units: int
    provided_facts: int
    required_intents: int
    response_constraints: int
    guidance_total_characters: int
    note_characters: int

    def log_fields(self) -> dict[str, int]:
        return asdict(self)


def build_difficulty_spec(
    request: DailyWritingGenerationRequest, slot: WritingSlot,
) -> WritingDifficultySpec:
    if type(slot.target_band) is not int or slot.target_band not in WRITING_BAND_RUBRIC:
        raise ValueError("Writing difficulty band must be an integer from 1 to 5")
    if slot.writing_type != request.writing_type:
        raise ValueError("Writing slot and request modes must agree")
    band = slot.target_band
    cjk = language_base(request.origin_language) in {"ko", "ja", "zh"}
    factor = 1 if cjk else 2
    if slot.writing_type == DailyWritingType.TRANSLATION:
        origin_max = _CJK_ORIGIN_MAX[band - 1] * factor
        unit_max = _SURFACE_UNIT_MAX[band - 1]
    else:
        # A FREE/GUIDED instruction's reading length is NOT the learner's required
        # production difficulty. Keep introduction limits independent of the band.
        origin_max, unit_max = 600 * factor, 5
    guided = slot.writing_type == DailyWritingType.GUIDED
    return WritingDifficultySpec(
        version=WRITING_DIFFICULTY_SPEC_VERSION,
        target_band=band,
        writing_type=slot.writing_type.value,
        origin_max_characters=origin_max,
        origin_max_surface_units=unit_max,
        guidance_min_entries=1 if guided else 0,
        guidance_max_entries=_GUIDANCE_ENTRY_MAX[band - 1] if guided else 0,
        guidance_max_characters_per_entry=240 * factor,
        guidance_max_total_characters=1800 * factor,
        note_max_characters=400 * factor,
        semantic_recipe=WRITING_BAND_RUBRIC[band],
    )


def measure_draft(draft: WritingDraft | DailyWritingItem) -> WritingSurfaceMeasurements:
    groups = (draft.provided_facts, draft.required_intents, draft.response_constraints)
    return WritingSurfaceMeasurements(
        origin_characters=visible_length(draft.origin_text),
        origin_surface_units=surface_units(draft.origin_text),
        provided_facts=len(draft.provided_facts), required_intents=len(draft.required_intents),
        response_constraints=len(draft.response_constraints),
        guidance_total_characters=sum(visible_length(text) for group in groups for text in group),
        note_characters=visible_length(draft.focus_reason),
    )


def difficulty_spec_reason(
    draft: WritingDraft | DailyWritingItem, spec: WritingDifficultySpec,
) -> str | None:
    values = measure_draft(draft)
    if values.origin_characters < 1 or values.origin_surface_units < 1:
        return "SPEC_ORIGIN_EMPTY"
    if values.origin_characters > spec.origin_max_characters:
        return "SPEC_ORIGIN_LENGTH"
    if values.origin_surface_units > spec.origin_max_surface_units:
        return "SPEC_ORIGIN_SURFACE_UNITS"
    for group in (draft.provided_facts, draft.required_intents, draft.response_constraints):
        if not spec.guidance_min_entries <= len(group) <= spec.guidance_max_entries:
            return "SPEC_GUIDANCE_COUNT"
        if any(not text.strip() or visible_length(text) > spec.guidance_max_characters_per_entry for text in group):
            return "SPEC_GUIDANCE_ENTRY_LENGTH"
    if values.guidance_total_characters > spec.guidance_max_total_characters:
        return "SPEC_GUIDANCE_TOTAL_LENGTH"
    if not 1 <= values.note_characters <= spec.note_max_characters:
        return "SPEC_NOTE_LENGTH"
    return None
