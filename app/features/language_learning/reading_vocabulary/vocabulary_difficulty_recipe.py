"""Server-owned generation recipes and blind measurement anchors for Vocabulary."""
from __future__ import annotations

from dataclasses import dataclass

from app.features.language_learning.reading_vocabulary.contextual_choice_task import (
    CONTEXTUAL_CHOICE_RECIPE_VERSION,
    CONTEXTUAL_CHOICE_SHADOW_RUBRIC_VERSION,
)
from app.schemas.language_learning_practice import (
    PracticeQuestionType,
    VocabularyMode,
    VocabularySkill,
)


VOCABULARY_DIFFICULTY_RECIPE_VERSION = "vocabulary-difficulty-recipe-v1"
MEANING_RELATION_DIFFICULTY_RECIPE_VERSION = (
    "vocabulary-meaning-relation-recipe-v7"
)
VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION = (
    "vocabulary-difficulty-shadow-rubric-v1"
)


@dataclass(frozen=True)
class MeaningRelationDemand:
    relation_kind: str
    task_shape: str
    decision_basis: tuple[str, ...]
    context_requirement: str
    semantic_distance_class: str
    contrast_dimensions: tuple[str, ...]
    near_neighbor_required: bool
    distractor_semantic_proximity: str
    distractor_requirements: tuple[str, ...]
    disallowed_shortcuts: tuple[str, ...]

    def generation_payload(self) -> dict[str, object]:
        return {
            "kind": "MEANING_RELATION",
            "relationKind": self.relation_kind,
            "taskShape": self.task_shape,
            "decisionBasis": list(self.decision_basis),
            "contextRequirement": self.context_requirement,
            "semanticDistanceClass": self.semantic_distance_class,
            "contrastDimensions": list(self.contrast_dimensions),
            "nearNeighborRequired": self.near_neighbor_required,
            "distractorSemanticProximity": self.distractor_semantic_proximity,
            "distractorRequirements": list(self.distractor_requirements),
            "disallowedShortcuts": list(self.disallowed_shortcuts),
            "authority": "SERVER_SELECTED_REQUIREMENT",
        }


@dataclass(frozen=True)
class UsageDistinctionDemand:
    usage_intent: str
    cue_scope: str
    cue_kinds: tuple[str, ...]
    minimum_relevant_cue_kinds: int
    context_required: bool
    register_dependence: bool
    collocation_dependence: bool
    pragmatic_dependence: bool
    distractor_near_miss_kinds: tuple[str, ...]
    disallowed_shortcuts: tuple[str, ...]

    def generation_payload(self) -> dict[str, object]:
        return {
            "kind": "USAGE_DISTINCTION",
            "usageIntent": self.usage_intent,
            "cueScope": self.cue_scope,
            "cueKinds": list(self.cue_kinds),
            "minimumRelevantCueKinds": self.minimum_relevant_cue_kinds,
            "contextRequired": self.context_required,
            "registerDependence": self.register_dependence,
            "collocationDependence": self.collocation_dependence,
            "pragmaticDependence": self.pragmatic_dependence,
            "distractorNearMissKinds": list(self.distractor_near_miss_kinds),
            "disallowedShortcuts": list(self.disallowed_shortcuts),
            "authority": "SERVER_SELECTED_REQUIREMENT",
        }


@dataclass(frozen=True)
class CompositionDemand:
    subtype: str
    constraint_kinds: tuple[str, ...]
    dependency_span: str
    clause_relation_requirement: str
    required_constraint_count: int
    single_valid_answer_required: bool
    disallowed_shortcuts: tuple[str, ...]

    def generation_payload(self) -> dict[str, object]:
        return {
            "kind": "COMPOSITION",
            "subtype": self.subtype,
            "constraintKinds": list(self.constraint_kinds),
            "dependencySpan": self.dependency_span,
            "clauseRelationRequirement": self.clause_relation_requirement,
            "requiredConstraintCount": self.required_constraint_count,
            "singleValidAnswerRequired": self.single_valid_answer_required,
            "disallowedShortcuts": list(self.disallowed_shortcuts),
            "authority": "SERVER_SELECTED_REQUIREMENT",
        }


@dataclass(frozen=True)
class ContextualChoiceDemand:
    skill: str
    context_shape: str
    decisive_dimensions: tuple[str, ...]
    minimum_close_distractors: int
    maximum_close_distractors: int
    direct_context_cues_allowed: bool
    definition_matching_disallowed: bool
    context_required: bool = True
    single_choice_required: bool = True

    def generation_payload(self) -> dict[str, object]:
        return {
            "kind": "CONTEXTUAL_CHOICE",
            "skill": self.skill,
            "contextShape": self.context_shape,
            "decisiveDimensions": list(self.decisive_dimensions),
            "minimumCloseDistractors": self.minimum_close_distractors,
            "maximumCloseDistractors": self.maximum_close_distractors,
            "directContextCuesAllowed": self.direct_context_cues_allowed,
            "definitionMatchingDisallowed": self.definition_matching_disallowed,
            "contextRequired": self.context_required,
            "singleChoiceRequired": self.single_choice_required,
            "authority": "SERVER_SELECTED_REQUIREMENT",
        }


VocabularyModeDemand = (
    MeaningRelationDemand
    | UsageDistinctionDemand
    | CompositionDemand
    | ContextualChoiceDemand
)


@dataclass(frozen=True)
class VocabularyDifficultyRecipe:
    mode: str
    band: int
    anchor: str
    deterministic_dimensions: tuple[str, ...]
    semantic_dimensions: tuple[str, ...]
    hard_constraints: tuple[str, ...]
    demand: VocabularyModeDemand
    version: str = VOCABULARY_DIFFICULTY_RECIPE_VERSION

    def generation_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "mode": self.mode,
            "band": self.band,
            "anchor": self.anchor,
            "deterministicDimensions": list(self.deterministic_dimensions),
            "semanticDimensions": list(self.semantic_dimensions),
            "hardConstraints": list(self.hard_constraints),
            "modeDemand": self.demand.generation_payload(),
            "enforcement": (
                "Deterministic constraints are application-validated. Semantic difficulty "
                "is independently shadow-observed and never authorizes ambiguity."
            ),
        }


_VOCABULARY_GENERATION_V1_ANCHORS = {
    VocabularyMode.CONTEXTUAL_CHOICE.value: {
        1: (
            "A clear everyday context with direct usable cues and broadly separated "
            "alternatives."
        ),
        2: (
            "A familiar collocation or basic contextual-fit decision with at least one "
            "plausible competitor."
        ),
        3: (
            "At least one close alternative requires reading the context rather than matching "
            "a dictionary meaning."
        ),
        4: (
            "At least two surface-plausible close alternatives are separated by scope, nuance, "
            "collocation, or a visible situation condition."
        ),
        5: (
            "Two or three close alternatives require register, scope, nuance, or pragmatic fit, "
            "while ordinary language knowledge still yields one answer."
        ),
    },
    VocabularyMode.MEANING_RELATION.value: {
        1: "A direct familiar meaning relation with clearly separated alternatives.",
        2: "A basic relation or paraphrase within a familiar semantic group.",
        3: "A distinction among close candidates using at least one semantic feature.",
        4: (
            "A near-neighbor distinction requiring scope, nuance, or usage implication; "
            "distractors remain in the same semantic neighborhood."
        ),
        5: (
            "Several candidates are partly related, but precise scope, nuance, and relation "
            "boundaries still determine one unambiguous best answer."
        ),
    },
    VocabularyMode.USAGE_DISTINCTION.value: {
        1: "One strong local cue makes one familiar expression plainly natural.",
        2: "A basic collocational, grammatical, semantic, or register cue decides usage.",
        3: "The full sentence or one clear social/contextual cue distinguishes close usage.",
        4: (
            "At least two visible context, register, collocation, or pragmatic cue kinds "
            "must be integrated among plausible competitors."
        ),
        5: (
            "Multiple role, register, intent, collocation, or pragmatic constraints jointly "
            "select one answer although several choices are grammatically possible."
        ),
    },
    VocabularyMode.COMPOSITION.value: {
        1: "One direct local form or ordering constraint determines the composition.",
        2: "Two basic grammar, collocation, or word-order constraints determine the result.",
        3: "Clause connection or natural composition requires integrating bounded constraints.",
        4: (
            "Several grammatical and semantic constraints must hold together; locally plausible "
            "alternatives fail in the complete composition."
        ),
        5: (
            "Scope, modification, information structure, or discourse constraints must be "
            "integrated while preserving one intended composition."
        ),
    },
}

_MEANING_RELATION_GENERATION_ANCHORS = {
    1: "A direct familiar relation with one clear answer and widely separated alternatives.",
    2: "A familiar paraphrase or opposition inside a basic semantic group.",
    3: (
        "The visible context selects the relevant sense; a bare dictionary definition or "
        "relation label is not enough by itself."
    ),
    4: (
        "Contextual scope, nuance, implication, or sense determines one answer among "
        "same-neighborhood alternatives."
    ),
    5: (
        "Several alternatives partially fit, but contextual sense and precise semantic "
        "boundaries determine one unambiguous answer."
    ),
}

# Keep the measurement instrument physically separate from generation recipe anchors.
# Generation policy may replace its anchors while this V1 measurement ruler stays frozen.
_VOCABULARY_MEASUREMENT_V1_ANCHORS = {
    VocabularyMode.CONTEXTUAL_CHOICE.value: {
        1: "Direct everyday context cues select one choice among clearly separated alternatives.",
        2: "A familiar collocation or basic contextual-fit cue selects among at least one plausible rival.",
        3: "Visible context is necessary to distinguish at least one semantically close rival.",
        4: (
            "Scope, nuance, collocation, or situation conditions distinguish at least two "
            "surface-plausible close rivals."
        ),
        5: (
            "Register, scope, nuance, or pragmatic fit precisely separates two or three close "
            "rivals without outside knowledge."
        ),
    },
    VocabularyMode.MEANING_RELATION.value: {
        1: "The visible item asks for a direct familiar relation with clearly separated alternatives.",
        2: "The visible item distinguishes a basic relation in a familiar semantic group.",
        3: "Close candidates require at least one visible semantic distinction.",
        4: (
            "Near neighbors require scope, nuance, or usage implication, and the distractors "
            "remain semantically plausible."
        ),
        5: (
            "Partly matching candidates require precise scope, nuance, and relation boundaries "
            "to determine one unambiguous answer."
        ),
    },
    VocabularyMode.USAGE_DISTINCTION.value: {
        1: "One strong local cue plainly selects one familiar expression.",
        2: "A basic collocational, grammatical, semantic, or register cue selects the usage.",
        3: "The full sentence or one clear social/contextual cue distinguishes close usage.",
        4: (
            "At least two visible context, register, collocation, or pragmatic cue kinds "
            "must be integrated among plausible competitors."
        ),
        5: (
            "Multiple role, register, intent, collocation, or pragmatic constraints jointly "
            "select one answer although several choices are grammatically possible."
        ),
    },
    VocabularyMode.COMPOSITION.value: {
        1: "One direct local form or ordering constraint determines the visible composition.",
        2: "Two basic grammar, collocation, or word-order constraints determine the result.",
        3: "Clause connection or natural composition requires integrating bounded constraints.",
        4: (
            "Several grammatical and semantic constraints hold together; locally plausible "
            "alternatives fail in the complete composition."
        ),
        5: (
            "Scope, modification, information structure, or discourse constraints must be "
            "integrated while preserving one intended composition."
        ),
    },
}

_COMMON_DETERMINISTIC = (
    "serverTargetBinding",
    "learningLanguageSurface",
    "questionTypeAndAnswerCardinality",
    "uniqueOptionKeysAndTexts",
    "answerKeysExist",
    "targetExpressionAndCanonicalIdentity",
)
_COMMON_HARD = (
    "Keep application-owned slot metadata and review binding.",
    "Return one unambiguous intended answer without outside knowledge.",
    "Do not use ambiguity or unrelated distractors as difficulty.",
)

_MEANING_RELATION_KIND_BY_SKILL = {
    VocabularySkill.MEANING.value: "DIRECT_MEANING",
    VocabularySkill.SYNONYM.value: "SYNONYM",
    VocabularySkill.ANTONYM.value: "ANTONYM",
    VocabularySkill.DISTINCTION.value: "NEAR_EXPRESSION_DISTINCTION",
}

# This is an operational generation capability policy, not a relabeling rule. The target band
# stays fixed; higher bands use task families that the current provider can realize reliably
# without falling back to direct lexical-pair shortcuts.
_MEANING_RELATION_SKILL_CYCLE_BY_BAND = {
    1: (
        VocabularySkill.MEANING.value,
        VocabularySkill.SYNONYM.value,
        VocabularySkill.ANTONYM.value,
        VocabularySkill.DISTINCTION.value,
    ),
    2: (
        VocabularySkill.MEANING.value,
        VocabularySkill.SYNONYM.value,
        VocabularySkill.ANTONYM.value,
        VocabularySkill.DISTINCTION.value,
    ),
    3: (VocabularySkill.DISTINCTION.value,),
    4: (VocabularySkill.DISTINCTION.value,),
    5: (VocabularySkill.DISTINCTION.value,),
}
_MEANING_TASK_SHAPE_BY_SKILL = {
    VocabularySkill.MEANING.value: {
        1: "DIRECT_FAMILIAR_MEANING",
        2: "FAMILIAR_PARAPHRASE",
        3: "SENSE_IN_CONTEXT_MEANING",
        4: "CONTEXTUAL_SCOPE_OR_NUANCE_MEANING",
        5: "CONTEXTUAL_PRECISE_MEANING_BOUNDARY",
    },
    VocabularySkill.SYNONYM.value: {
        1: "DIRECT_BASIC_SYNONYM",
        2: "FAMILIAR_SYNONYM_OR_PARAPHRASE",
        3: "CONTEXT_SPECIFIC_SENSE_SYNONYM",
        4: "NEAR_SYNONYM_NUANCE_SELECTION",
        5: "CONTEXT_SENSE_PRESERVATION_AMONG_NEAR_SYNONYMS",
    },
    VocabularySkill.ANTONYM.value: {
        1: "DIRECT_LEXICAL_OPPOSITE",
        2: "FAMILIAR_OPPOSITION",
        3: "CONTEXTUAL_SAME_DIMENSION_OPPOSITION",
        4: "SCOPE_SENSITIVE_CONTEXTUAL_OPPOSITION",
        5: "CONTEXT_AND_SCOPE_PRECISE_OPPOSITION",
    },
    VocabularySkill.DISTINCTION.value: {
        1: "BASIC_EXPRESSION_DISTINCTION",
        2: "FAMILIAR_EXPRESSION_CONTRAST",
        3: "ONE_DECISIVE_CONTEXTUAL_CONTRAST",
        4: "CONTEXT_APPLIED_NEAR_EXPRESSION_DISCRIMINATION",
        5: "MULTI_FEATURE_CONTEXTUAL_BOUNDARY_DISCRIMINATION",
    },
}
_MEANING_DECISION_BASIS_BY_SKILL = {
    VocabularySkill.MEANING.value: {
        1: ("CORE_MEANING",),
        2: ("FAMILIAR_PARAPHRASE",),
        3: ("CONTEXTUAL_SENSE", "SEMANTIC_FEATURE"),
        4: ("CONTEXTUAL_SENSE", "SCOPE_OR_NUANCE_OR_IMPLICATION"),
        5: ("CONTEXTUAL_SENSE", "PRECISE_SCOPE", "NUANCE", "BOUNDARY"),
    },
    VocabularySkill.SYNONYM.value: {
        1: ("BASIC_SYNONYMY",),
        2: ("FAMILIAR_PARAPHRASE",),
        3: ("CONTEXT_SPECIFIC_SENSE",),
        4: ("CONTEXT_SPECIFIC_SENSE", "SCOPE_OR_NUANCE_OR_NATURAL_USAGE"),
        5: ("SENSE_PRESERVATION", "SCOPE", "NUANCE", "CONTEXTUAL_FIT"),
    },
    VocabularySkill.ANTONYM.value: {
        1: ("DIRECT_LEXICAL_OPPOSITION",),
        2: ("FAMILIAR_OPPOSITION",),
        3: ("CONTEXTUAL_SENSE", "SAME_SEMANTIC_DIMENSION", "OPPOSITE_DIRECTION"),
        4: ("CONTEXTUAL_SENSE", "SCOPE", "SAME_DIMENSION_OPPOSITION"),
        5: (
            "CONTEXTUAL_SENSE",
            "PRECISE_SCOPE",
            "SAME_DIMENSION_OPPOSITION",
            "OPPOSITION_BOUNDARY",
        ),
    },
    VocabularySkill.DISTINCTION.value: {
        1: ("BASIC_MEANING_DIFFERENCE",),
        2: ("FAMILIAR_EXPRESSION_CONTRAST",),
        3: ("VISIBLE_CONTEXT", "ONE_DECISIVE_CONTRAST"),
        4: ("VISIBLE_CONTEXT", "SCOPE_OR_NUANCE_OR_USAGE_CONTRAST"),
        5: ("VISIBLE_CONTEXT", "SCOPE", "NUANCE", "RELATION_BOUNDARY"),
    },
}
_MEANING_CONTEXT_REQUIREMENT = {
    1: "OPTIONAL",
    2: "OPTIONAL_WHEN_NATURAL",
    3: "SENSE_SELECTING_CONTEXT_REQUIRED",
    4: "DECISION_RELEVANT_CONTEXT_REQUIRED",
    5: "MULTI_FEATURE_DECISION_CONTEXT_REQUIRED",
}
_MEANING_DISTANCE = {
    1: "BROADLY_SEPARATED",
    2: "FAMILIAR_SEMANTIC_GROUP",
    3: "CLOSE_WITH_ONE_DECISIVE_FEATURE",
    4: "NEAR_NEIGHBORS_WITH_NUANCE",
    5: "OVERLAPPING_WITH_PRECISE_BOUNDARIES",
}
_MEANING_CONTRAST = {
    1: ("CORE_MEANING",),
    2: ("BASIC_RELATION_OR_PARAPHRASE",),
    3: ("SEMANTIC_FEATURE",),
    4: ("SCOPE", "NUANCE_OR_USAGE_IMPLICATION"),
    5: ("PRECISE_SCOPE", "NUANCE", "RELATION_BOUNDARY"),
}
_MEANING_DISTRACTOR_PROXIMITY = {
    1: "CLEARLY_SEPARATED",
    2: "SAME_FAMILIAR_GROUP",
    3: "CLOSE_SEMANTIC_NEIGHBORS",
    4: "PLAUSIBLE_NEAR_NEIGHBORS",
    5: "PARTIALLY_MATCHING_NEAR_NEIGHBORS",
}

_USAGE_INTENT_BY_SKILL = {
    VocabularySkill.DISTINCTION.value: "CONTEXTUAL_NEAR_EXPRESSION_CHOICE",
    VocabularySkill.COLLOCATION.value: "COLLOCATION_CHOICE",
    VocabularySkill.REGISTER.value: "REGISTER_CHOICE",
    VocabularySkill.CONTEXT_USAGE.value: "CONTEXTUAL_USAGE_CHOICE",
}
_USAGE_CUE_KINDS = {
    "CONTEXTUAL_NEAR_EXPRESSION_CHOICE": (
        "SEMANTIC_SITUATION",
        "PRAGMATIC_IMPLICATION",
        "LEXICAL_SELECTION_RESTRICTION",
    ),
    "COLLOCATION_CHOICE": (
        "LOCAL_COLLOCATION",
        "GRAMMATICAL_FRAME",
        "SEMANTIC_COMPATIBILITY",
    ),
    "REGISTER_CHOICE": (
        "ROLE_RELATIONSHIP",
        "FORMALITY",
        "COMMUNICATION_CHANNEL",
    ),
    "CONTEXTUAL_USAGE_CHOICE": (
        "SEMANTIC_SITUATION",
        "PRAGMATIC_INTENT",
        "DISCOURSE_CONTEXT",
    ),
}
_USAGE_CUE_SCOPE = {
    1: "STRONG_LOCAL_CUE",
    2: "LOCAL_PHRASE_OR_SENTENCE",
    3: "WHOLE_SENTENCE_OR_EXPLICIT_SOCIAL_CONTEXT",
    4: "MULTIPLE_VISIBLE_CONTEXT_CUES",
    5: "INTEGRATED_ROLE_REGISTER_INTENT_CONTEXT",
}
_USAGE_MINIMUM_CUE_KINDS = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2}

_COMPOSITION_SUBTYPE_BY_SKILL = {
    VocabularySkill.COMPOSITION.value: "EXPRESSION_OR_CLAUSE_COMPLETION",
    VocabularySkill.COLLOCATION.value: "COLLOCATION_ASSEMBLY",
    VocabularySkill.CONTEXT_USAGE.value: "CONTEXTUAL_COMPLETION",
}
_COMPOSITION_CONSTRAINTS = {
    1: ("LOCAL_FORM",),
    2: ("WORD_ORDER", "BASIC_GRAMMAR_OR_COLLOCATION"),
    3: ("GRAMMAR", "SEMANTIC_FIT", "CLAUSE_CONNECTION"),
    4: ("GRAMMAR", "SEMANTIC_FIT", "MODIFICATION_OR_SCOPE"),
    5: ("SCOPE", "MODIFICATION", "INFORMATION_STRUCTURE", "DISCOURSE_RELATION"),
}
_COMPOSITION_DEPENDENCY_SPAN = {
    1: "LOCAL",
    2: "LOCAL_MULTI_PART",
    3: "CLAUSE",
    4: "MULTI_CONSTRAINT_CLAUSE",
    5: "WHOLE_COMPOSITION",
}
_COMPOSITION_REQUIRED_CONSTRAINTS = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}


def usage_intent_for_skill(skill_tag: str) -> str | None:
    return _USAGE_INTENT_BY_SKILL.get(skill_tag)


def meaning_relation_skill_cycle(band: int) -> tuple[str, ...]:
    _validate_band(band)
    return _MEANING_RELATION_SKILL_CYCLE_BY_BAND[band]


def meaning_relation_skill_for_slot(*, band: int, band_occurrence: int) -> str:
    if band_occurrence < 1:
        raise ValueError("Vocabulary band occurrence must be positive")
    cycle = meaning_relation_skill_cycle(band)
    return cycle[(band_occurrence - 1) % len(cycle)]


def vocabulary_difficulty_recipe_version(mode: str) -> str:
    if mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
        return CONTEXTUAL_CHOICE_RECIPE_VERSION
    if mode == VocabularyMode.MEANING_RELATION.value:
        return MEANING_RELATION_DIFFICULTY_RECIPE_VERSION
    if mode in {
        VocabularyMode.USAGE_DISTINCTION.value,
        VocabularyMode.COMPOSITION.value,
    }:
        return VOCABULARY_DIFFICULTY_RECIPE_VERSION
    raise ValueError(f"Unsupported Vocabulary mode: {mode}")


def vocabulary_composition_subtype(question_type: str, skill_tag: str) -> str:
    if question_type == PracticeQuestionType.ORDERING.value:
        return "ORDERING"
    if question_type != PracticeQuestionType.SINGLE_CHOICE.value:
        raise ValueError(f"Unsupported Vocabulary question type: {question_type}")
    try:
        return _COMPOSITION_SUBTYPE_BY_SKILL[skill_tag]
    except KeyError as exc:
        raise ValueError(f"Unsupported COMPOSITION skill: {skill_tag}") from exc


def vocabulary_task_subtype(
    *,
    mode: str,
    skill_tag: str,
    question_type: str,
) -> str:
    if mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
        if skill_tag not in {
            VocabularySkill.MEANING.value,
            VocabularySkill.COLLOCATION.value,
            VocabularySkill.NUANCE.value,
            VocabularySkill.REGISTER.value,
            VocabularySkill.PRAGMATIC_FIT.value,
        }:
            raise ValueError(f"Unsupported CONTEXTUAL_CHOICE skill: {skill_tag}")
        return f"CONTEXTUAL_{skill_tag}_CHOICE"
    if mode == VocabularyMode.MEANING_RELATION.value:
        try:
            return _MEANING_RELATION_KIND_BY_SKILL[skill_tag]
        except KeyError as exc:
            raise ValueError(f"Unsupported MEANING_RELATION skill: {skill_tag}") from exc
    if mode == VocabularyMode.USAGE_DISTINCTION.value:
        usage_intent = usage_intent_for_skill(skill_tag)
        if usage_intent is None:
            raise ValueError(f"Unsupported USAGE_DISTINCTION skill: {skill_tag}")
        return usage_intent
    if mode == VocabularyMode.COMPOSITION.value:
        return vocabulary_composition_subtype(question_type, skill_tag)
    raise ValueError(f"Unsupported Vocabulary mode: {mode}")


def vocabulary_difficulty_recipe(
    *,
    mode: str,
    band: int,
    skill_tag: str,
    question_type: str,
    usage_intent: str | None = None,
) -> VocabularyDifficultyRecipe:
    _validate_band(band)
    if mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
        demand: VocabularyModeDemand = _contextual_choice_demand(band, skill_tag)
        semantic_dimensions = (
            "contextualDecisionDemand",
            "skillSpecificFit",
            "distractorSemanticProximity",
        )
    elif mode == VocabularyMode.MEANING_RELATION.value:
        demand = _meaning_relation_demand(band, skill_tag)
        semantic_dimensions = (
            "semanticDistance",
            "meaningContrastPrecision",
            "distractorSemanticProximity",
        )
    elif mode == VocabularyMode.USAGE_DISTINCTION.value:
        demand = _usage_distinction_demand(band, skill_tag, usage_intent)
        semantic_dimensions = (
            "contextDependence",
            "registerCollocationPragmaticFit",
            "distractorContextualPlausibility",
        )
    elif mode == VocabularyMode.COMPOSITION.value:
        demand = _composition_demand(band, skill_tag, question_type)
        semantic_dimensions = (
            "constraintIntegration",
            "naturalComposition",
            "singleValidArrangementOrCompletion",
        )
    else:
        raise ValueError(f"Unsupported Vocabulary mode: {mode}")

    recipe = VocabularyDifficultyRecipe(
        mode=mode,
        band=band,
        anchor=(
            _MEANING_RELATION_GENERATION_ANCHORS[band]
            if mode == VocabularyMode.MEANING_RELATION.value
            else _VOCABULARY_GENERATION_V1_ANCHORS[mode][band]
        ),
        deterministic_dimensions=_COMMON_DETERMINISTIC,
        semantic_dimensions=semantic_dimensions,
        hard_constraints=_COMMON_HARD,
        demand=demand,
        version=vocabulary_difficulty_recipe_version(mode),
    )
    validate_vocabulary_difficulty_recipe(
        mode=mode,
        band=band,
        skill_tag=skill_tag,
        question_type=question_type,
        usage_intent=usage_intent,
        recipe=recipe,
    )
    return recipe


def validate_vocabulary_difficulty_recipe(
    *,
    mode: str,
    band: int,
    skill_tag: str,
    question_type: str,
    usage_intent: str | None,
    recipe: VocabularyDifficultyRecipe,
) -> None:
    _validate_band(band)
    if recipe.mode != mode or recipe.band != band:
        raise ValueError("Vocabulary difficulty recipe mode/band mismatch")
    if recipe.version != vocabulary_difficulty_recipe_version(mode):
        raise ValueError("Vocabulary difficulty recipe version mismatch")
    if mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
        expected = _contextual_choice_demand(band, skill_tag)
    elif mode == VocabularyMode.MEANING_RELATION.value:
        expected = _meaning_relation_demand(band, skill_tag)
    elif mode == VocabularyMode.USAGE_DISTINCTION.value:
        expected = _usage_distinction_demand(band, skill_tag, usage_intent)
    elif mode == VocabularyMode.COMPOSITION.value:
        expected = _composition_demand(band, skill_tag, question_type)
    else:
        raise ValueError(f"Unsupported Vocabulary mode: {mode}")
    if recipe.demand != expected:
        raise ValueError("Vocabulary difficulty demand does not match server recipe")


def vocabulary_blind_rubric_payload(mode: str) -> dict[str, object]:
    try:
        anchors = _VOCABULARY_MEASUREMENT_V1_ANCHORS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported Vocabulary mode: {mode}") from exc
    return {
        "version": vocabulary_difficulty_shadow_rubric_version(mode),
        "mode": mode,
        "bands": [
            {"band": band, "anchor": anchors[band]}
            for band in range(1, 6)
        ],
        "assessmentBoundary": (
            "Classify cognitive and linguistic demand, not word length, rarity, ambiguity, "
            "outside knowledge, or generator metadata."
        ),
    }


def vocabulary_difficulty_shadow_rubric_version(mode: str) -> str:
    if mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
        return CONTEXTUAL_CHOICE_SHADOW_RUBRIC_VERSION
    if mode in {
        VocabularyMode.MEANING_RELATION.value,
        VocabularyMode.USAGE_DISTINCTION.value,
        VocabularyMode.COMPOSITION.value,
    }:
        return VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION
    raise ValueError(f"Unsupported Vocabulary mode: {mode}")


_CONTEXTUAL_DECISIVE_DIMENSIONS = {
    VocabularySkill.MEANING.value: ("CONTEXTUAL_MEANING", "SENSE_FIT"),
    VocabularySkill.COLLOCATION.value: ("COLLOCATION", "LOCAL_LEXICAL_FRAME"),
    VocabularySkill.NUANCE.value: ("NUANCE", "SCOPE_OR_IMPLICATION"),
    VocabularySkill.REGISTER.value: ("REGISTER", "ROLE_FORMALITY_OR_CHANNEL"),
    VocabularySkill.PRAGMATIC_FIT.value: ("PRAGMATIC_FIT", "SITUATION_OR_INTENT"),
}


def _contextual_choice_demand(band: int, skill_tag: str) -> ContextualChoiceDemand:
    try:
        dimensions = _CONTEXTUAL_DECISIVE_DIMENSIONS[skill_tag]
    except KeyError as exc:
        raise ValueError(f"Unsupported CONTEXTUAL_CHOICE skill: {skill_tag}") from exc
    minimum_close = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2}[band]
    maximum_close = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3}[band]
    context_shape = {
        1: "CLEAR_EVERYDAY_ONE_OR_TWO_SENTENCE_BLANK",
        2: "FAMILIAR_CONTEXTUAL_OR_COLLOCATION_BLANK",
        3: "CONTEXT_REQUIRED_CLOSE_CHOICE_BLANK",
        4: "MULTI_CUE_NEAR_NEIGHBOR_BLANK",
        5: "PRECISE_REGISTER_SCOPE_NUANCE_OR_PRAGMATIC_BLANK",
    }[band]
    return ContextualChoiceDemand(
        skill=skill_tag,
        context_shape=context_shape,
        decisive_dimensions=dimensions,
        minimum_close_distractors=minimum_close,
        maximum_close_distractors=maximum_close,
        direct_context_cues_allowed=band <= 2,
        definition_matching_disallowed=True,
    )


def _meaning_relation_demand(band: int, skill_tag: str) -> MeaningRelationDemand:
    try:
        relation_kind = _MEANING_RELATION_KIND_BY_SKILL[skill_tag]
    except KeyError as exc:
        raise ValueError(f"Unsupported MEANING_RELATION skill: {skill_tag}") from exc
    return MeaningRelationDemand(
        relation_kind=relation_kind,
        task_shape=_MEANING_TASK_SHAPE_BY_SKILL[skill_tag][band],
        decision_basis=_MEANING_DECISION_BASIS_BY_SKILL[skill_tag][band],
        context_requirement=_MEANING_CONTEXT_REQUIREMENT[band],
        semantic_distance_class=_MEANING_DISTANCE[band],
        contrast_dimensions=_MEANING_CONTRAST[band],
        near_neighbor_required=band >= 4,
        distractor_semantic_proximity=_MEANING_DISTRACTOR_PROXIMITY[band],
        distractor_requirements=_meaning_distractor_requirements(band),
        disallowed_shortcuts=_meaning_disallowed_shortcuts(band, skill_tag),
    )


def _meaning_distractor_requirements(band: int) -> tuple[str, ...]:
    if band <= 2:
        return ("ONE_CLEAR_ANSWER", "NATURAL_SAME_LANGUAGE_ALTERNATIVES")
    if band == 3:
        return (
            "SAME_PART_OF_SPEECH_WHEN_APPLICABLE",
            "SEMANTICALLY_RELATED_NEAR_MISSES",
            "FAIL_ONE_DECISIVE_FEATURE",
        )
    return (
        "SAME_PART_OF_SPEECH_WHEN_APPLICABLE",
        "SAME_SEMANTIC_NEIGHBORHOOD",
        "PARTIALLY_PLAUSIBLE_IN_VISIBLE_CONTEXT",
        "FAIL_ONE_OR_MORE_DECISIVE_SEMANTIC_FEATURES",
    )


def _meaning_disallowed_shortcuts(band: int, skill_tag: str) -> tuple[str, ...]:
    shortcuts = [
        "AMBIGUITY_AS_DIFFICULTY",
        "UNRELATED_DISTRACTORS",
        "OUTSIDE_KNOWLEDGE",
    ]
    if band >= 3 and skill_tag == VocabularySkill.DISTINCTION.value:
        shortcuts.extend(
            [
                "TARGET_EXPRESSION_IN_CONTEXT_OR_STEM",
                "TARGET_EXPRESSION_MISSING_FROM_SERVER_CORRECT_OPTION",
            ]
        )
    else:
        shortcuts.append("TARGET_EXPRESSION_IN_ANY_OPTION")
    if band >= 3:
        shortcuts.extend(
            [
                "BARE_DICTIONARY_DEFINITION_AS_COMPLETE_TASK",
                "STEM_DIRECTLY_PARAPHRASES_CORRECT_OPTION",
            ]
        )
    if band >= 4:
        shortcuts.extend(
            [
                "STEM_TEACHES_CANDIDATE_DIFFERENCES_BEFORE_ASKING",
                "CORRECT_LEXICAL_FAMILY_DISCLOSED_IN_STEM",
                "OUT_OF_NEIGHBORHOOD_DISTRACTORS",
            ]
        )
        if skill_tag == VocabularySkill.ANTONYM.value:
            shortcuts.append("DIRECT_LEXICAL_PAIR_AS_SOLE_DECISION_BASIS")
    return tuple(shortcuts)


def _usage_distinction_demand(
    band: int,
    skill_tag: str,
    usage_intent: str | None,
) -> UsageDistinctionDemand:
    expected_intent = usage_intent_for_skill(skill_tag)
    if expected_intent is None or usage_intent not in {None, expected_intent}:
        raise ValueError("Vocabulary usageIntent does not match server skill mapping")
    resolved_intent = expected_intent
    return UsageDistinctionDemand(
        usage_intent=resolved_intent,
        cue_scope=_USAGE_CUE_SCOPE[band],
        cue_kinds=_USAGE_CUE_KINDS[resolved_intent],
        minimum_relevant_cue_kinds=_USAGE_MINIMUM_CUE_KINDS[band],
        context_required=True,
        register_dependence=resolved_intent == "REGISTER_CHOICE",
        collocation_dependence=resolved_intent == "COLLOCATION_CHOICE",
        pragmatic_dependence=(
            band >= 3
            and resolved_intent
            in {
                "CONTEXTUAL_NEAR_EXPRESSION_CHOICE",
                "CONTEXTUAL_USAGE_CHOICE",
            }
        ),
        distractor_near_miss_kinds=(
            "SAME_PART_OF_SPEECH",
            "PLAUSIBLE_OUTSIDE_DECISIVE_CONTEXT",
            "VISIBLE_CUE_MISMATCH",
        ),
        disallowed_shortcuts=(
            "TARGET_IN_STEM",
            "MEANING_RECALL_INSTEAD_OF_USAGE",
            "AMBIGUITY_AS_DIFFICULTY",
        ),
    )


def _composition_demand(
    band: int,
    skill_tag: str,
    question_type: str,
) -> CompositionDemand:
    subtype = vocabulary_composition_subtype(question_type, skill_tag)
    return CompositionDemand(
        subtype=subtype,
        constraint_kinds=_COMPOSITION_CONSTRAINTS[band],
        dependency_span=_COMPOSITION_DEPENDENCY_SPAN[band],
        clause_relation_requirement=(
            "REQUIRED_WHEN_NATURAL_FOR_SUBTYPE" if band >= 3 else "NOT_REQUIRED"
        ),
        required_constraint_count=_COMPOSITION_REQUIRED_CONSTRAINTS[band],
        single_valid_answer_required=True,
        disallowed_shortcuts=(
            "AMBIGUITY_AS_DIFFICULTY",
            "TOKEN_COUNT_AS_DIFFICULTY",
            "MULTIPLE_VALID_ANSWERS",
            "UNNATURAL_FRAGMENTATION",
        ),
    )


def _validate_band(band: int) -> None:
    if type(band) is not int or band not in range(1, 6):
        raise ValueError("Vocabulary difficulty band must be an integer from 1 to 5")
