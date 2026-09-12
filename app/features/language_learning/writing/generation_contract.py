"""Server-owned Writing plans and deterministic content checks.

A generator proposes content, never order/difficulty/complexity. Semantic difficulty
is *not* provable with these checks; only an independently verified draft can be
assembled into the public DailyWritingItem contract by the generation pipeline.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field, StrictBool, StrictStr

from app.features.language_learning.quality import resolve_daily_complexity_band
from app.features.language_learning.writing.difficulty_spec import build_difficulty_spec, difficulty_spec_reason
from app.features.language_learning.writing.source_language import missing_expected_script, required_script_pattern
from app.schemas.language_learning import (
    CamelCaseModel,
    DailyWritingDifficulty,
    DailyWritingGenerationRequest,
    DailyWritingItem,
    DailyWritingType,
    WritingMetric,
)
from app.schemas.language_learning_quality import (
    CommunicativeIntent,
    DiversityMetadata,
    ScenarioCategory,
)

CANDIDATES_PER_SLOT = 2
MAX_GENERATION_ATTEMPTS = 4
WRITING_VERIFICATION_POLICY_VERSION = "writing-difficulty-reference-v4.1-bounded-recovery"

NonBlankText = Annotated[StrictStr, Field(min_length=1, max_length=2000)]
ShortCode = Annotated[StrictStr, Field(min_length=1, max_length=200)]

# Legacy/self-assigned control values are discarded, not trusted or repaired.
# Other unexpected fields (e.g. answer/modelAnswer) remain forbidden.
SERVER_OWNED_FIELDS = frozenset({
    "order", "difficulty", "languageComplexityBand", "language_complexity_band",
    "writingType", "writing_type",
})


class DraftDiversityMetadata(CamelCaseModel):
    scenario_category: ScenarioCategory
    communicative_intent: CommunicativeIntent
    task_archetype: Annotated[StrictStr, Field(min_length=1, max_length=100)]
    grammar_focus_codes: list[ShortCode] = Field(..., max_length=20)
    lexical_focus_codes: list[ShortCode] = Field(..., max_length=20)
    semantic_summary: Annotated[StrictStr, Field(min_length=1, max_length=500)]
    requires_background_knowledge: StrictBool


class WritingDraft(CamelCaseModel):
    origin_text: NonBlankText
    keywords: list[ShortCode] = Field(..., max_length=20)
    focus_metrics: list[WritingMetric] = Field(..., min_length=1, max_length=5)
    focus_reason: Annotated[StrictStr, Field(min_length=1, max_length=1000)]
    provided_facts: list[NonBlankText] = Field(..., max_length=12)
    required_intents: list[NonBlankText] = Field(..., max_length=12)
    response_constraints: list[NonBlankText] = Field(..., max_length=12)
    diversity_metadata: DraftDiversityMetadata

    def task_content(self) -> dict[str, Any]:
        """Allowlist: no generator rationale, level labels, profile, or metadata."""
        return self.model_dump(mode="json", by_alias=True, include={
            "origin_text", "provided_facts", "required_intents", "response_constraints",
        })

    def to_diversity_metadata(self) -> DiversityMetadata:
        return DiversityMetadata.model_validate(
            self.diversity_metadata.model_dump(mode="json", by_alias=True)
        )


class WritingCandidateBatch(CamelCaseModel):
    items: list[WritingDraft] = Field(..., min_length=1, max_length=CANDIDATES_PER_SLOT)


@dataclass(frozen=True)
class WritingSlot:
    order: int
    difficulty: DailyWritingDifficulty
    target_band: int
    writing_type: DailyWritingType


def plan_slots(request: DailyWritingGenerationRequest) -> tuple[WritingSlot, ...]:
    distribution = request.difficulty_distribution
    slots = []
    for difficulty, count in (
        (DailyWritingDifficulty.REVIEW, distribution.review),
        (DailyWritingDifficulty.NORMAL, distribution.normal),
        (DailyWritingDifficulty.CHALLENGE, distribution.challenge),
    ):
        for _ in range(count):
            slots.append(WritingSlot(
                order=len(slots) + 1,
                difficulty=difficulty,
                target_band=resolve_daily_complexity_band(difficulty.value, request.language_complexity),
                writing_type=request.writing_type,
            ))
    if len(slots) != request.sentence_count:
        raise ValueError("Writing slot count must equal the requested sentence count")
    return tuple(slots)


def build_candidate_schema(*, request: DailyWritingGenerationRequest | None = None,
                           slot: WritingSlot | None = None) -> dict[str, Any]:
    # A fresh graph per call. The same spec constrains generation and local checks.
    schema = WritingCandidateBatch.model_json_schema(by_alias=True)
    if request is not None and slot is not None:
        spec = build_difficulty_spec(request, slot)
        fields = schema["$defs"]["WritingDraft"]["properties"]
        fields["originText"]["maxLength"] = spec.origin_max_characters
        if request.writing_type == DailyWritingType.TRANSLATION:
            pattern = required_script_pattern(request.origin_language)
            if pattern is not None:
                fields["originText"]["pattern"] = pattern
        fields["focusReason"]["maxLength"] = spec.note_max_characters
        for name in ("providedFacts", "requiredIntents", "responseConstraints"):
            fields[name]["minItems"] = spec.guidance_min_entries
            fields[name]["maxItems"] = spec.guidance_max_entries
            fields[name]["items"]["maxLength"] = spec.guidance_max_characters_per_entry
    return schema


def parse_draft(raw: dict[str, Any]) -> WritingDraft:
    normalized = {key: copy.deepcopy(value) for key, value in raw.items()
                  if key not in SERVER_OWNED_FIELDS}
    # Canonicalize enum spelling only, never meaning, task text, or an estimated band.
    for key in ("focusMetrics", "focus_metrics"):
        if isinstance(normalized.get(key), list):
            normalized[key] = [_canonical_token(value) for value in normalized[key]]
    metadata = normalized.get("diversityMetadata", normalized.get("diversity_metadata"))
    if isinstance(metadata, dict):
        for key in ("scenarioCategory", "scenario_category", "communicativeIntent", "communicative_intent"):
            if key in metadata:
                metadata[key] = _canonical_token(metadata[key])
    return WritingDraft.model_validate(normalized)


def _canonical_token(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
    return value


def writing_type_contract_reason(
    request: DailyWritingGenerationRequest,
    item: WritingDraft | DailyWritingItem,
) -> str | None:
    groups = (item.provided_facts, item.required_intents, item.response_constraints)
    if any(not isinstance(value, str) or not value.strip() for group in groups for value in group):
        return "GUIDANCE_CONTAINS_BLANK_VALUE"
    mode = request.writing_type.value
    if mode == "GUIDED":
        for label, values in zip(("PROVIDED_FACTS", "REQUIRED_INTENTS", "RESPONSE_CONSTRAINTS"), groups):
            if not values:
                return f"GUIDED_{label}_MISSING"
    elif any(groups):
        return f"{mode}_GUIDANCE_MUST_BE_EMPTY"
    return None


def deterministic_draft_reason(
    request: DailyWritingGenerationRequest,
    draft: WritingDraft,
    *, slot: WritingSlot | None = None,
) -> str | None:
    reason = writing_type_contract_reason(request, draft)
    if reason is not None:
        return reason
    if draft.diversity_metadata.requires_background_knowledge:
        return "BACKGROUND_KNOWLEDGE_REQUIRED"
    if len(set(draft.keywords)) != len(draft.keywords):
        return "DUPLICATE_KEYWORD"
    if not set(draft.keywords).issubset({keyword.key for keyword in request.selected_keywords}):
        return "UNKNOWN_KEYWORD"
    if len(set(draft.focus_metrics)) != len(draft.focus_metrics):
        return "DUPLICATE_FOCUS_METRIC"
    for group in (draft.provided_facts, draft.required_intents, draft.response_constraints):
        if len(set(group)) != len(group):
            return "DUPLICATE_GUIDANCE"
    # Content and learner-visible notes have different recovery boundaries. Never
    # discard an otherwise valid task here just because its note needs localization.
    # Each guidance GROUP keeps the existing script-presence rule (technical names
    # can be members); actual language and mixed content still need the Mini audit.
    fields = {"originText": draft.origin_text}
    if request.writing_type == DailyWritingType.GUIDED:
        fields.update({
            "providedFacts": " ".join(draft.provided_facts),
            "requiredIntents": " ".join(draft.required_intents),
            "responseConstraints": " ".join(draft.response_constraints),
        })
    if any(contains_control_character(text) for text in (*fields.values(), draft.focus_reason)):
        return "CONTROL_CHARACTER"
    if slot is not None:
        reason = difficulty_spec_reason(draft, build_difficulty_spec(request, slot))
        if reason is not None:
            return reason
    for name, text in fields.items():
        if _missing_expected_script(request.origin_language, text):
            return _SCRIPT_REJECTION_CODES[name]
    return None


_SCRIPT_REJECTION_CODES = {
    "originText": "ORIGIN_TEXT_SCRIPT_MISMATCH",
    "providedFacts": "GUIDED_PROVIDED_FACTS_SCRIPT_MISMATCH",
    "requiredIntents": "GUIDED_REQUIRED_INTENTS_SCRIPT_MISMATCH",
    "responseConstraints": "GUIDED_RESPONSE_CONSTRAINTS_SCRIPT_MISMATCH",
}


def script_rejection_field(reason: str) -> str | None:
    """Safe, fixed field names for diagnostics; never log raw generated content."""
    return next((name for name, code in _SCRIPT_REJECTION_CODES.items() if code == reason), None)


def contains_control_character(text: str) -> bool:
    return any(unicodedata.category(char) == "Cc" and char not in "\t\r\n" for char in text)


def note_needs_localization(request: DailyWritingGenerationRequest, draft: WritingDraft) -> bool:
    """An obvious script mismatch needs repair, not publication or a task rewrite.

    This is a necessary-script check for ko/ja/en, NOT a language detector. The
    semantic note audit (the normal combined review, or the repaired-note review)
    remains mandatory even when this function returns False.
    """
    return _missing_expected_script(request.origin_language, draft.focus_reason)


def _missing_expected_script(language: str, text: str) -> bool:
    # Retain the private name for old internal callers/tests; one shared policy.
    return missing_expected_script(language, text)


def review_content_hash(request: DailyWritingGenerationRequest, draft: WritingDraft) -> str:
    material = {
        "originLanguage": request.origin_language,
        "learningLanguage": request.learning_language,
        "writingType": request.writing_type.value,
        "content": draft.task_content(),
        "focusReason": draft.focus_reason,
    }
    return hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_final_items(request: DailyWritingGenerationRequest, items: list[DailyWritingItem]) -> None:
    """Revalidate the public response without relying on model_copy's bypass behavior."""
    if len(items) != request.sentence_count:
        raise ValueError("Writing response item count mismatch")
    expected = plan_slots(request)
    if Counter(item.difficulty for item in items) != Counter(slot.difficulty for slot in expected):
        raise ValueError("Writing response difficulty distribution mismatch")
    for slot, item in zip(expected, items):
        if (item.order, item.difficulty, item.language_complexity_band) != (
            slot.order, slot.difficulty, slot.target_band,
        ):
            raise ValueError("Writing server-owned metadata mismatch")
        if writing_type_contract_reason(request, item) is not None or item.diversity_metadata is None:
            raise ValueError("Writing final content contract mismatch")
        if difficulty_spec_reason(item, build_difficulty_spec(request, slot)) is not None:
            raise ValueError("Writing final difficulty spec mismatch")
        payload = item.model_dump(mode="json", by_alias=True)
        payload["diversityMetadata"] = {
            key: value for key, value in payload["diversityMetadata"].items()
            if key in DraftDiversityMetadata.model_json_schema(by_alias=True)["properties"]
        }
        draft = parse_draft(payload)
        if deterministic_draft_reason(request, draft, slot=slot) is not None:
            raise ValueError("Writing final deterministic contract mismatch")
        if note_needs_localization(request, draft):
            raise ValueError("Writing final note language contract mismatch")
