"""Server-owned editorial recipes for Reading difficulty calibration.

Passage complexity and question-demand complexity are deliberately separate.
V2 gives COMPREHENSION and STRUCTURE an application-owned cognitive blueprint;
surface measurements remain diagnostics and semantic difficulty remains shadow-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


READING_DIFFICULTY_RECIPE_VERSION = "reading-difficulty-recipe-v2"
READING_DIFFICULTY_SHADOW_RUBRIC_VERSION = "reading-difficulty-recipe-v1-shadow"

ReadingRecipeKind = Literal["PASSAGE", "QUESTION_DEMAND"]


class QuestionDemandKind(StrEnum):
    EXPLICIT_LOCAL = "EXPLICIT_LOCAL"
    PARAPHRASED_LOCAL = "PARAPHRASED_LOCAL"
    LOCAL_RELATION = "LOCAL_RELATION"
    CROSS_UNIT_INFERENCE = "CROSS_UNIT_INFERENCE"
    DISCOURSE_STRUCTURE = "DISCOURSE_STRUCTURE"
    WHOLE_TEXT_SYNTHESIS = "WHOLE_TEXT_SYNTHESIS"
    QUALIFIED_MULTI_CUE = "QUALIFIED_MULTI_CUE"


class EvidenceScope(StrEnum):
    LOCAL = "LOCAL"
    ADJACENT_UNITS = "ADJACENT_UNITS"
    CROSS_UNIT = "CROSS_UNIT"
    CROSS_PARAGRAPH = "CROSS_PARAGRAPH"
    WHOLE_TEXT = "WHOLE_TEXT"


@dataclass(frozen=True)
class ReadingQuestionDemand:
    """Application-owned target blueprint, never generator-authored metadata."""

    kind: QuestionDemandKind
    evidence_scope: EvidenceScope
    minimum_distinct_cues: int
    minimum_evidence_units: int
    minimum_evidence_paragraphs: int
    inference_requirement: str
    discourse_requirements: tuple[str, ...]
    distractor_requirements: tuple[str, ...]
    disallowed_shortcuts: tuple[str, ...]

    def generation_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "evidenceScope": self.evidence_scope.value,
            "minimumDistinctCues": self.minimum_distinct_cues,
            "minimumEvidenceUnits": self.minimum_evidence_units,
            "minimumEvidenceParagraphs": self.minimum_evidence_paragraphs,
            "inferenceRequirement": self.inference_requirement,
            "discourseRequirements": list(self.discourse_requirements),
            "distractorRequirements": list(self.distractor_requirements),
            "disallowedShortcuts": list(self.disallowed_shortcuts),
            "authority": "SERVER_SELECTED_REQUIREMENT",
        }


@dataclass(frozen=True)
class ReadingDifficultyRecipe:
    kind: ReadingRecipeKind
    band: int
    anchor: str
    deterministic_dimensions: tuple[str, ...]
    preferred_dimensions: tuple[str, ...]
    semantic_dimensions: tuple[str, ...]
    hard_structural_constraints: tuple[str, ...]
    question_demand: ReadingQuestionDemand | None = None
    enforcement: str = (
        "Preferred dimensions guide generation and shadow measurement only. "
        "They are not empirical hard thresholds."
    )
    version: str = READING_DIFFICULTY_RECIPE_VERSION

    def generation_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "kind": self.kind,
            "band": self.band,
            "anchor": self.anchor,
            "deterministicDimensions": list(self.deterministic_dimensions),
            "preferredDimensions": list(self.preferred_dimensions),
            "semanticDimensions": list(self.semantic_dimensions),
            "hardStructuralConstraints": list(self.hard_structural_constraints),
            "enforcement": self.enforcement,
        }
        if self.question_demand is not None:
            payload["questionDemand"] = self.question_demand.generation_payload()
        return payload


_PASSAGE_ANCHORS = {
    1: "Direct, linear familiar text with one plainly expressed situation or message.",
    2: "Familiar text with a simple sequence, reason, time relation or ordinary request.",
    3: "Coherent connected text using cause, condition or comparison with explicit scope.",
    4: "Connected discourse using concession, indirect intent, stance or register-sensitive relations.",
    5: (
        "Coherent discourse with interacting scope, qualification, stance or reference relations, "
        "without specialist knowledge."
    ),
}

_QUESTION_ANCHORS = {
    1: "Answerable from one explicit local span by direct evidence lookup.",
    2: "Requires a paraphrase or connection between nearby explicit units.",
    3: "Requires a cause, condition, comparison or other bounded relation judgment.",
    4: "Requires integrating multiple units or discourse cues; local lookup alone is insufficient.",
    5: (
        "Requires multiple cues plus at least two interacting constraints such as scope, "
        "qualification, condition or exception."
    ),
}

_V1_QUESTION_ANCHORS = {
    1: "Answerable from one explicit local piece of evidence with direct discrimination.",
    2: "Requires a simple paraphrase or relation between nearby explicit evidence.",
    3: "Requires gist, one bounded inference, or a clear cause/condition/structure relation.",
    4: (
        "Requires cross-unit evidence, discourse-function recognition, or a supported "
        "pragmatic inference."
    ),
    5: (
        "Requires combining multiple supplied linguistic cues while preserving scope, stance "
        "or qualification; ambiguity and outside knowledge do not count as difficulty."
    ),
}

_PASSAGE_DETERMINISTIC = (
    "normalizedCharacterCount",
    "paragraphCount",
    "surfaceUnitCount",
)
_PASSAGE_HARD = (
    "Use only the requested learning language.",
    "Return non-blank passageText bound to the requested passageId.",
)
_QUESTION_DETERMINISTIC = (
    "promptCharacterCount",
    "optionCountAndCharacterLengths",
    "evidenceExactMatchAndOffset",
    "evidenceParagraphAndSurfaceUnitCoverage",
    "evidenceToQuestionSurfaceUnitDistance",
    "skillTagAndPassageBinding",
    "serverQuestionDemandBlueprint",
)
_V1_QUESTION_DETERMINISTIC = (
    "promptCharacterCount",
    "optionCountAndCharacterLengths",
    "evidenceExactMatchAndOffset",
    "evidenceParagraphAndSurfaceUnit",
    "evidenceToQuestionSurfaceUnitDistance",
    "skillTagAndPassageBinding",
)
_QUESTION_HARD = (
    "Reuse the exact server-assigned passage and passageId.",
    "Use the server-required single-choice shape and held answer-key binding.",
    "Keep vocabulary identity fields empty for Reading questions.",
)

_COMPREHENSION_DEMANDS = {
    1: ReadingQuestionDemand(
        QuestionDemandKind.EXPLICIT_LOCAL, EvidenceScope.LOCAL, 1, 1, 1, "NONE", (),
        ("DIRECTLY_SUPPORTED_ALTERNATIVES",), (),
    ),
    2: ReadingQuestionDemand(
        QuestionDemandKind.PARAPHRASED_LOCAL, EvidenceScope.ADJACENT_UNITS, 1, 1, 1,
        "PARAPHRASE_OR_ADJACENT_CONNECTION", ("ADJACENT_INFORMATION_CONNECTION",),
        ("PLAUSIBLE_PARAPHRASE_NEAR_MISS",), ("VERBATIM_ANSWER_COPY",),
    ),
    3: ReadingQuestionDemand(
        QuestionDemandKind.LOCAL_RELATION, EvidenceScope.ADJACENT_UNITS, 2, 2, 1,
        "ONE_BOUNDED_REASONING_STEP", ("CAUSE_OR_CONDITION_OR_COMPARISON",),
        ("RELATION_MISMATCH",), ("SINGLE_SENTENCE_ANSWER_COPY",),
    ),
    4: ReadingQuestionDemand(
        QuestionDemandKind.CROSS_UNIT_INFERENCE, EvidenceScope.CROSS_UNIT, 2, 2, 1,
        "SUPPORTED_CROSS_UNIT_INFERENCE",
        ("CAUSE_EFFECT_OR_CLAIM_EVIDENCE_OR_CONDITION_RESULT",),
        ("PASSAGE_GROUNDED_NEAR_MISS", "SCOPE_OR_RELATION_MISMATCH"),
        ("SINGLE_LOCAL_EXPLICIT_LOOKUP",),
    ),
    5: ReadingQuestionDemand(
        QuestionDemandKind.QUALIFIED_MULTI_CUE, EvidenceScope.CROSS_UNIT, 3, 2, 1,
        "MULTI_CUE_SYNTHESIS",
        ("TWO_OR_MORE_OF_SCOPE_QUALIFICATION_CONDITION_EXCEPTION",),
        ("PARTIAL_TRUTH", "SCOPE_OR_RELATION_MISMATCH"),
        ("SINGLE_LOCAL_EXPLICIT_LOOKUP", "UNQUALIFIED_DETAIL_RETRIEVAL"),
    ),
}

_STRUCTURE_DEMANDS = {
    1: ReadingQuestionDemand(
        QuestionDemandKind.PARAPHRASED_LOCAL, EvidenceScope.LOCAL, 1, 1, 1,
        "PARAGRAPH_GIST_OR_SIMPLE_ROLE", ("ONE_PARAGRAPH_FUNCTION",),
        ("WRONG_PARAGRAPH_ROLE",), ("ISOLATED_FACT_RETRIEVAL",),
    ),
    2: ReadingQuestionDemand(
        QuestionDemandKind.DISCOURSE_STRUCTURE, EvidenceScope.ADJACENT_UNITS, 2, 2, 1,
        "SIMPLE_DEVELOPMENT_ORDER", ("PARAGRAPH_ROLE_OR_SIMPLE_SEQUENCE",),
        ("ROLE_OR_ORDER_MISMATCH",), ("CONTENT_REASON_RETRIEVAL_ONLY",),
    ),
    3: ReadingQuestionDemand(
        QuestionDemandKind.DISCOURSE_STRUCTURE, EvidenceScope.CROSS_PARAGRAPH, 2, 2, 2,
        "ADJACENT_PARAGRAPH_RELATION", ("PROPOSAL_EVIDENCE_RESULT_OR_CONTRAST",),
        ("DISCOURSE_RELATION_MISMATCH",), ("CONTENT_REASON_RETRIEVAL_ONLY",),
    ),
    4: ReadingQuestionDemand(
        QuestionDemandKind.DISCOURSE_STRUCTURE, EvidenceScope.CROSS_PARAGRAPH, 3, 3, 2,
        "MULTI_PARAGRAPH_FUNCTION_RELATION",
        ("PROBLEM_ALTERNATIVE_DECISION_OR_CLAIM_CONCESSION_CONCLUSION",),
        ("PARAGRAPH_FUNCTION_NEAR_MISS", "RELATION_MISMATCH"),
        ("CONTENT_REASON_RETRIEVAL_ONLY", "SINGLE_PARAGRAPH_LOOKUP"),
    ),
    5: ReadingQuestionDemand(
        QuestionDemandKind.WHOLE_TEXT_SYNTHESIS, EvidenceScope.WHOLE_TEXT, 3, 3, 2,
        "WHOLE_ARGUMENT_STRUCTURE",
        ("MULTIPLE_PARAGRAPH_ROLES", "ALTERNATIVE_TO_FINAL_CONCLUSION_FUNCTION"),
        ("PARTIALLY_CORRECT_ARGUMENT_MAP", "FUNCTION_OR_SCOPE_MISMATCH"),
        ("CONTENT_REASON_RETRIEVAL_ONLY", "SINGLE_PARAGRAPH_LOOKUP"),
    ),
}


def passage_difficulty_recipe(band: int, *, mode: str) -> ReadingDifficultyRecipe:
    _validate_band(band)
    preferred = [
        "Use length, paragraphing and surface-unit count only as broad editorial signals.",
        f"Shape discourse so it supports the requested Reading mode {mode}.",
    ]
    semantic = [
        "linguisticAndDiscourseComplexity", "coherenceAndReferenceLoad",
        "registerAndPragmaticNuance", "selfContainedWithoutExternalKnowledge",
    ]
    if mode == "COMPREHENSION" and band >= 4:
        preferred.append(
            "Include a meaningful cross-unit relation such as concession, contrast, condition, "
            "claim/evidence or cause/effect without merely increasing length."
        )
        semantic.append("crossUnitDependencyForComprehension")
    return ReadingDifficultyRecipe(
        kind="PASSAGE", band=band, anchor=_PASSAGE_ANCHORS[band],
        deterministic_dimensions=_PASSAGE_DETERMINISTIC,
        preferred_dimensions=tuple(preferred), semantic_dimensions=tuple(semantic),
        hard_structural_constraints=_PASSAGE_HARD,
    )


def calibrated_reading_skill(*, mode: str, band: int, planned_skill: str) -> str:
    """Remove invalid high-band skill combinations before candidate generation."""

    _validate_band(band)
    if mode == "CONTEXT_INFERENCE":
        return planned_skill
    if mode == "COMPREHENSION" and band >= 4 and planned_skill == "DETAIL":
        return "INFERENCE"
    if mode == "STRUCTURE" and band >= 3:
        return "STRUCTURE"
    return planned_skill


def question_demand_recipe(
    band: int,
    *,
    mode: str,
    skill_tag: str,
) -> ReadingDifficultyRecipe:
    _validate_band(band)
    if mode == "CONTEXT_INFERENCE":
        recipe = ReadingDifficultyRecipe(
            kind="QUESTION_DEMAND",
            band=band,
            anchor=_V1_QUESTION_ANCHORS[band],
            deterministic_dimensions=_V1_QUESTION_DETERMINISTIC,
            preferred_dimensions=(
                "Treat evidence location and distance as observed signals, never as a band by themselves.",
                f"Realize the server-selected skill {skill_tag} without changing the assigned passage.",
            ),
            semantic_dimensions=(
                "evidenceExplicitness",
                "inferenceDemand",
                "discourseRelationDemand",
                "distractorDiscriminationDemand",
                "singleBestSupportedAnswerWithoutAmbiguity",
                "selfContainedWithoutExternalKnowledge",
            ),
            hard_structural_constraints=_QUESTION_HARD,
        )
        validate_question_demand_blueprint(
            mode=mode, band=band, skill_tag=skill_tag, recipe=recipe,
        )
        return recipe
    recipe = ReadingDifficultyRecipe(
        kind="QUESTION_DEMAND", band=band, anchor=_QUESTION_ANCHORS[band],
        deterministic_dimensions=_QUESTION_DETERMINISTIC,
        preferred_dimensions=(
            "Treat measured evidence geometry as an observation, never as a band by itself.",
            f"Realize the server-selected skill {skill_tag} without changing the assigned passage.",
        ),
        semantic_dimensions=(
            "evidenceExplicitness", "inferenceDemand", "discourseRelationDemand",
            "distractorDiscriminationDemand", "singleBestSupportedAnswerWithoutAmbiguity",
            "selfContainedWithoutExternalKnowledge",
        ),
        hard_structural_constraints=_QUESTION_HARD,
        question_demand=_demand_for(mode, band),
        enforcement=(
            "The questionDemand blueprint is server-selected. Surface measurements "
            "are deterministic diagnostics; semantic difficulty remains shadow-only."
        ),
    )
    validate_question_demand_blueprint(
        mode=mode, band=band, skill_tag=skill_tag, recipe=recipe,
    )
    return recipe


def validate_question_demand_blueprint(
    *,
    mode: str,
    band: int,
    skill_tag: str,
    recipe: ReadingDifficultyRecipe,
) -> None:
    """Reject application-owned invalid combinations before a provider call."""

    _validate_band(band)
    if recipe.kind != "QUESTION_DEMAND" or recipe.band != band:
        raise ValueError("Reading question recipe kind/band mismatch")
    if mode == "CONTEXT_INFERENCE":
        if recipe.question_demand is not None:
            raise ValueError("CONTEXT_INFERENCE V1 behavior must not gain a V2 demand override")
        return
    demand = recipe.question_demand
    if demand is None:
        raise ValueError("Reading V2 question demand blueprint is required")
    if demand != _demand_for(mode, band):
        raise ValueError("Reading question demand blueprint does not match server recipe")
    if mode == "COMPREHENSION" and band >= 4 and skill_tag == "DETAIL":
        raise ValueError("High-band COMPREHENSION must not use a low-demand DETAIL blueprint")
    if mode == "STRUCTURE":
        if band >= 3 and skill_tag != "STRUCTURE":
            raise ValueError("High-band STRUCTURE must use the STRUCTURE skill")
        if band >= 4 and demand.evidence_scope not in {
            EvidenceScope.CROSS_PARAGRAPH, EvidenceScope.WHOLE_TEXT,
        }:
            raise ValueError("High-band STRUCTURE requires cross-paragraph structure")


def reading_blind_rubric_payload() -> dict[str, object]:
    """Return the frozen V1 measurement rubric without exposing a requested band."""

    return {
        "version": READING_DIFFICULTY_SHADOW_RUBRIC_VERSION,
        "passageBands": [
            {"band": band, "anchor": _PASSAGE_ANCHORS[band]} for band in range(1, 6)
        ],
        "questionDemandBands": [
            {"band": band, "anchor": _V1_QUESTION_ANCHORS[band]}
            for band in range(1, 6)
        ],
        "classificationRules": [
            "Classify actual visible linguistic/task demand; do not infer the requested band.",
            "Length and evidence distance are supporting observations, not band authorities.",
            "External knowledge, ambiguity and weak distractors never raise the band.",
        ],
    }


def _demand_for(mode: str, band: int) -> ReadingQuestionDemand | None:
    if mode == "COMPREHENSION":
        return _COMPREHENSION_DEMANDS[band]
    if mode == "STRUCTURE":
        return _STRUCTURE_DEMANDS[band]
    if mode == "CONTEXT_INFERENCE":
        return None
    raise ValueError(f"Unsupported Reading mode: {mode}")


def _validate_band(band: int) -> None:
    if type(band) is not int or band not in _PASSAGE_ANCHORS:
        raise ValueError("Reading difficulty band must be an integer from 1 to 5")
