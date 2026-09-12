"""Server-owned editorial recipes for Reading difficulty calibration.

The recipes describe generation intent and measurable signals.  They are not
empirically calibrated thresholds, and none of the preferred dimensions are a
V1 acceptance rule.  Passage complexity and question-demand complexity are
deliberately separate scales even when they use the same 1-5 band numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


READING_DIFFICULTY_RECIPE_VERSION = "reading-difficulty-recipe-v1-shadow"

ReadingRecipeKind = Literal["PASSAGE", "QUESTION_DEMAND"]


@dataclass(frozen=True)
class ReadingDifficultyRecipe:
    kind: ReadingRecipeKind
    band: int
    anchor: str
    deterministic_dimensions: tuple[str, ...]
    preferred_dimensions: tuple[str, ...]
    semantic_dimensions: tuple[str, ...]
    hard_structural_constraints: tuple[str, ...]
    version: str = READING_DIFFICULTY_RECIPE_VERSION

    def generation_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "kind": self.kind,
            "band": self.band,
            "anchor": self.anchor,
            "deterministicDimensions": list(self.deterministic_dimensions),
            "preferredDimensions": list(self.preferred_dimensions),
            "semanticDimensions": list(self.semantic_dimensions),
            "hardStructuralConstraints": list(self.hard_structural_constraints),
            "enforcement": (
                "Preferred dimensions guide generation and shadow measurement only. "
                "They are not empirical hard thresholds."
            ),
        }


_PASSAGE_ANCHORS = {
    1: (
        "Direct, linear familiar text with one plainly expressed situation or message."
    ),
    2: (
        "Familiar text with a simple sequence, reason, time relation or ordinary request."
    ),
    3: (
        "Coherent connected text using cause, condition or comparison with explicit scope."
    ),
    4: (
        "Connected discourse using concession, indirect intent, stance or register-sensitive relations."
    ),
    5: (
        "Coherent discourse with interacting scope, qualification, stance or reference relations, "
        "without specialist knowledge."
    ),
}

_QUESTION_ANCHORS = {
    1: "Answerable from one explicit local piece of evidence with direct discrimination.",
    2: "Requires a simple paraphrase or relation between nearby explicit evidence.",
    3: "Requires gist, one bounded inference, or a clear cause/condition/structure relation.",
    4: (
        "Requires cross-unit evidence, discourse-function recognition, or a supported pragmatic inference."
    ),
    5: (
        "Requires combining multiple supplied linguistic cues while preserving scope, stance or "
        "qualification; ambiguity and outside knowledge do not count as difficulty."
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
    "evidenceParagraphAndSurfaceUnit",
    "evidenceToQuestionSurfaceUnitDistance",
    "skillTagAndPassageBinding",
)
_QUESTION_HARD = (
    "Reuse the exact server-assigned passage and passageId.",
    "Use the server-required single-choice shape and held answer-key binding.",
    "Keep vocabulary identity fields empty for Reading questions.",
)


def passage_difficulty_recipe(band: int, *, mode: str) -> ReadingDifficultyRecipe:
    _validate_band(band)
    return ReadingDifficultyRecipe(
        kind="PASSAGE",
        band=band,
        anchor=_PASSAGE_ANCHORS[band],
        deterministic_dimensions=_PASSAGE_DETERMINISTIC,
        preferred_dimensions=(
            "Use length, paragraphing and surface-unit count only as broad editorial signals.",
            f"Shape discourse so it supports the requested Reading mode {mode}.",
        ),
        semantic_dimensions=(
            "linguisticAndDiscourseComplexity",
            "coherenceAndReferenceLoad",
            "registerAndPragmaticNuance",
            "selfContainedWithoutExternalKnowledge",
        ),
        hard_structural_constraints=_PASSAGE_HARD,
    )


def question_demand_recipe(band: int, *, skill_tag: str) -> ReadingDifficultyRecipe:
    _validate_band(band)
    return ReadingDifficultyRecipe(
        kind="QUESTION_DEMAND",
        band=band,
        anchor=_QUESTION_ANCHORS[band],
        deterministic_dimensions=_QUESTION_DETERMINISTIC,
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


def reading_blind_rubric_payload() -> dict[str, object]:
    """Return all anchors without identifying any candidate's requested band."""

    return {
        "version": READING_DIFFICULTY_RECIPE_VERSION,
        "passageBands": [
            {"band": band, "anchor": _PASSAGE_ANCHORS[band]} for band in range(1, 6)
        ],
        "questionDemandBands": [
            {"band": band, "anchor": _QUESTION_ANCHORS[band]} for band in range(1, 6)
        ],
        "classificationRules": [
            "Classify actual visible linguistic/task demand; do not infer the requested band.",
            "Length and evidence distance are supporting observations, not band authorities.",
            "External knowledge, ambiguity and weak distractors never raise the band.",
        ],
    }


def _validate_band(band: int) -> None:
    if type(band) is not int or band not in _PASSAGE_ANCHORS:
        raise ValueError("Reading difficulty band must be an integer from 1 to 5")
