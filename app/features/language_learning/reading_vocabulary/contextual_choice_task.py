from __future__ import annotations

from app.schemas.language_learning_practice import VocabularySkill


CONTEXTUAL_CHOICE_RECIPE_VERSION = "vocabulary-contextual-choice-recipe-v2"
CONTEXTUAL_CHOICE_SHADOW_RUBRIC_VERSION = (
    "vocabulary-contextual-choice-shadow-rubric-v1"
)
CONTEXTUAL_CHOICE_BLANK = "______"
CONTEXTUAL_CHOICE_CANDIDATES_PER_ROUND = 3
CONTEXTUAL_CHOICE_MAX_ROUNDS = 2
CONTEXTUAL_CHOICE_SKILL_PLAN = (
    VocabularySkill.MEANING.value,
    VocabularySkill.NUANCE.value,
    VocabularySkill.COLLOCATION.value,
    VocabularySkill.REGISTER.value,
    VocabularySkill.PRAGMATIC_FIT.value,
    VocabularySkill.MEANING.value,
    VocabularySkill.COLLOCATION.value,
    VocabularySkill.NUANCE.value,
    VocabularySkill.REGISTER.value,
    VocabularySkill.PRAGMATIC_FIT.value,
)
CONTEXTUAL_CHOICE_ANSWER_KEYS = ("A", "B", "C", "D")
CONTEXTUAL_CHOICE_SCENARIO_FAMILIES = (
    "SYSTEM_OPERATION",
    "CUSTOMER_COMMUNICATION",
    "SCHEDULING_COORDINATION",
    "DECISION_APPROVAL",
    "INCIDENT_RESPONSE",
    "COLLABORATION_HANDOFF",
    "DOCUMENT_REPORTING",
    "PROCESS_IMPROVEMENT",
    "RESOURCE_PLANNING",
    "GENERAL_WORKPLACE",
)


def contextual_choice_skill_for_order(global_order: int) -> str:
    if global_order not in range(1, 11):
        raise ValueError("CONTEXTUAL_CHOICE global order must be from 1 to 10")
    return CONTEXTUAL_CHOICE_SKILL_PLAN[global_order - 1]


def contextual_choice_answer_key(global_order: int) -> str:
    if global_order not in range(1, 11):
        raise ValueError("CONTEXTUAL_CHOICE global order must be from 1 to 10")
    return CONTEXTUAL_CHOICE_ANSWER_KEYS[(global_order - 1) % 4]


def contextual_choice_scenario_family(new_item_index: int) -> str:
    if new_item_index < 0:
        raise ValueError("CONTEXTUAL_CHOICE new item index must not be negative")
    return CONTEXTUAL_CHOICE_SCENARIO_FAMILIES[
        new_item_index % len(CONTEXTUAL_CHOICE_SCENARIO_FAMILIES)
    ]


def render_contextual_choice_prompt(
    complete_sentence: str,
    target_expression: str,
) -> str:
    sentence = complete_sentence.strip()
    target = target_expression.strip()
    if not sentence:
        raise ValueError("CONTEXTUAL_CHOICE requires completeSentence")
    if not target:
        raise ValueError("CONTEXTUAL_CHOICE requires targetExpression")
    if sentence.count(target) != 1:
        raise ValueError(
            "CONTEXTUAL_CHOICE completeSentence must contain targetExpression exactly once"
        )
    prompt = sentence.replace(target, CONTEXTUAL_CHOICE_BLANK, 1)
    if prompt.count(CONTEXTUAL_CHOICE_BLANK) != 1 or target in prompt:
        raise ValueError("CONTEXTUAL_CHOICE deterministic blank assembly failed")
    if prompt.replace(CONTEXTUAL_CHOICE_BLANK, target, 1) != sentence:
        raise ValueError("CONTEXTUAL_CHOICE blank round-trip mismatch")
    return prompt
