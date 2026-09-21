from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, NoReturn, cast

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, ValidationError

from app.ai.ports import TextGenerationProvider
from app.core.config import settings
from app.features.language_learning.reading_vocabulary.contextual_choice_task import (
    CONTEXTUAL_CHOICE_ANSWER_KEYS,
    CONTEXTUAL_CHOICE_BLANK,
    CONTEXTUAL_CHOICE_CANDIDATES_PER_ROUND,
    CONTEXTUAL_CHOICE_MAX_ROUNDS,
    CONTEXTUAL_CHOICE_PLAN_VERSION,
    CONTEXTUAL_CHOICE_RECIPE_VERSION,
    CONTEXTUAL_CHOICE_SKILL_PLAN,
    CONTEXTUAL_CHOICE_TEMPLATE_MARKER,
    contextual_choice_answer_key,
    contextual_choice_required_close_distractors,
    contextual_choice_scenario_family,
    contextual_choice_skill_for_order,
    render_contextual_choice_prompt,
    render_contextual_choice_template,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_adapter import (
    ReadingAcceptanceContext,
    ReadingSemanticQualityPolicy,
    build_reading_passage_difficulty_spec,
    build_reading_question_difficulty_spec,
    normalize_reading_semantic_assessment,
    project_reading_question_validation,
    validate_reading_passage,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_recipe import (
    calibrated_reading_skill,
    question_demand_recipe,
    structure_judgment_contract,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_shadow import (
    ReadingDifficultyShadowCollector,
)
from app.features.language_learning.reading_vocabulary.meaning_relation_task import (
    render_meaning_relation_prompt,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_adapter import (
    VocabularyAcceptanceContext,
    build_vocabulary_difficulty_spec,
    normalize_vocabulary_semantic_assessment,
    project_vocabulary_validation,
    semantic_quality_policy_for_vocabulary_mode,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_recipe import (
    ContextualChoiceDemand,
    MeaningRelationDemand,
    meaning_relation_skill_for_slot,
    usage_intent_for_skill,
    vocabulary_difficulty_recipe,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_shadow import (
    VocabularyDifficultyShadowCollector,
)
from app.features.language_learning.reading_vocabulary.prompts import (
    PRACTICE_GENERATION_PROMPT_VERSION,
    build_contextual_choice_candidate_batch_prompt,
    build_contextual_choice_context_generation_prompt,
    build_contextual_choice_context_verification_prompt,
    build_contextual_choice_plan_generation_prompt,
    build_contextual_choice_plan_verification_prompt,
    build_contextual_choice_lexical_repair_prompt,
    build_contextual_choice_verification_prompt,
    build_origin_explanation_prompt,
    build_practice_generation_prompt,
    build_practice_verification_prompt,
    build_reading_distractor_repair_prompt,
    build_reading_passage_prompt,
    build_usage_prescreen_prompt,
)
from app.schemas.language_learning_practice import (
    PracticeDifficulty,
    PracticeDomain,
    PracticeGeneratedQuestion,
    PracticeGenerationRequest,
    PracticeGenerationResponse,
    PracticeOption,
    PracticeQuestionType,
    PracticeReviewTarget,
    PersonalizedVocabularyPlan,
    ReadingPassageBundle,
    ReadingQuestionPlan,
    ReadingMode,
    ReadingSkill,
    VocabularyMode,
    VocabularyPlanAnchorType,
    VocabularyPlanItem,
    VocabularySkill,
)

logger = logging.getLogger(__name__)

_MAX_CANDIDATE_ATTEMPTS_PER_SLOT = 3
_MAX_READING_GENERATION_ATTEMPTS_PER_SLOT = 8
_MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT = 3
_MAX_DISTRACTOR_REPAIR_ATTEMPTS_PER_SLOT = 1
_MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT = 10
_MAX_USAGE_DISTINCTION_SEMANTIC_ATTEMPTS_PER_SLOT = 5
_MAX_COMPOSITION_GENERATION_ATTEMPTS_PER_SLOT = 8
_MAX_COMPOSITION_SEMANTIC_ATTEMPTS_PER_SLOT = 3
_MAX_PASSAGE_ATTEMPTS = 3
_MAX_ORIGIN_EXPLANATION_ATTEMPTS = 2
_CANDIDATE_BATCH_SIZE = 2
_USAGE_DISTINCTION_BATCH_SIZE = 1
_ORIGIN_EXPLANATION_BATCH_SIZE = 4
_MEANING_RELATION_RESERVE_DISTRACTOR_COUNT = 2
_SINGLE_CHOICE_KEYS = ("A", "B", "C", "D")
_CONTEXTUAL_CHOICE_SKILL_ORDER = tuple(
    dict.fromkeys(CONTEXTUAL_CHOICE_SKILL_PLAN)
)

# ASCII-only tokens that are genuinely used as lexical items inside Japanese/Korean practical
# language. Ordinary English words are intentionally NOT accepted here: selected keywords may
# seed a topic, but they must never silently become the vocabulary answer in another language.
_ASCII_VOCABULARY_ALLOWLIST = {
    "AI", "API", "AWS", "B2B", "B2C", "CD", "CI", "CRM", "CPU", "DB", "DNS",
    "ERP", "GCP", "GPU", "HTTP", "HTTPS", "IP", "IT", "JSON", "KPI", "OKR",
    "PR", "QA", "RAM", "SaaS", "SDK", "SLA", "SQL", "SSH", "SSL", "TCP",
    "TLS", "UDP", "UI", "URI", "URL", "UX", "VPN", "XML",
    "BtoB", "BtoC", "DevOps", "Docker", "Git", "GitHub", "IoT", "Java",
    "JavaScript", "Kubernetes", "Next.js", "NoSQL", "PoC", "Python", "React",
    "RPA", "SRE", "TypeScript",
}
_ASCII_ACRONYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9+./#_-]{1,11}$")

_OPTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {"key": {"type": "STRING"}, "text": {"type": "STRING"}},
    "required": ["key", "text"],
}

# explanationOrigin is intentionally absent. It is generated only after the question has
# passed deterministic + independent semantic verification, so a translation/language
# failure can never invalidate an otherwise-good question candidate.
_PRACTICE_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "order": {"type": "INTEGER"},
                    "questionType": {"type": "STRING", "enum": ["SINGLE_CHOICE", "ORDERING"]},
                    "difficulty": {"type": "STRING", "enum": ["EASIER", "CURRENT", "CHALLENGE"]},
                    "complexityBand": {"type": "INTEGER"},
                    "passageId": {"type": ["STRING", "NULL"]},
                    "passageText": {"type": ["STRING", "NULL"]},
                    "prompt": {"type": "STRING"},
                    "options": {"type": "ARRAY", "items": _OPTION_SCHEMA},
                    "correctAnswer": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "skillTag": {"type": "STRING"},
                    "evidenceText": {"type": ["STRING", "NULL"]},
                    "explanationLearning": {"type": "STRING"},
                    "targetExpression": {"type": ["STRING", "NULL"]},
                    "canonicalKey": {"type": ["STRING", "NULL"]},
                    "reviewTarget": {"type": "BOOLEAN"},
                    "vocabularyCandidates": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": [
                    "order", "questionType", "difficulty", "complexityBand", "passageId",
                    "passageText", "prompt", "options", "correctAnswer", "skillTag",
                    "evidenceText", "explanationLearning", "targetExpression", "canonicalKey",
                    "reviewTarget", "vocabularyCandidates",
                ],
            },
        }
    },
    "required": ["questions"],
}
_B1_COMPREHENSION_CANDIDATE_SCHEMA = copy.deepcopy(_PRACTICE_CANDIDATE_SCHEMA)
_b1_candidate_item = _B1_COMPREHENSION_CANDIDATE_SCHEMA["properties"]["questions"]["items"]
_b1_candidate_item["properties"].update({
    "inferenceClueQuote": {"type": ["STRING", "NULL"]},
    "unstatedInference": {"type": ["STRING", "NULL"]},
})
_b1_candidate_item["required"].extend(["inferenceClueQuote", "unstatedInference"])
_WRONG_OPTION_CANDIDATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {"text": {"type": "STRING"}},
    "required": ["text"],
}

_CONTEXTUAL_CHOICE_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "additionalProperties": False,
                "properties": {
                    "candidateId": {"type": "STRING"},
                    "targetExpression": {"type": "STRING"},
                    "completeSentence": {"type": "STRING"},
                    "distractors": {
                        "type": "ARRAY",
                        "items": _WRONG_OPTION_CANDIDATE_SCHEMA,
                    },
                    "explanationLearning": {"type": "STRING"},
                },
                "required": [
                    "candidateId",
                    "targetExpression",
                    "completeSentence",
                    "distractors",
                    "explanationLearning",
                ],
            },
        }
    },
    "required": ["candidates"],
}

_CONTEXTUAL_CHOICE_PLAN_ITEM_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "additionalProperties": False,
    "properties": {
        "globalOrder": {"type": "INTEGER"},
        "targetExpression": {"type": ["STRING", "NULL"]},
        "distractors": {
            "type": "ARRAY",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "STRING"},
        },
        "anchorType": {
            "type": ["STRING", "NULL"],
            "enum": [
                "SELECTED_KEYWORD",
                "WEAK_SIGNAL",
                "RECENT_MISTAKE",
                "LEARNING_PROFILE",
                None,
            ],
        },
        "anchorValue": {"type": ["STRING", "NULL"]},
    },
    "required": [
        "globalOrder",
        "targetExpression",
        "distractors",
        "anchorType",
        "anchorValue",
    ],
}
_CONTEXTUAL_CHOICE_PLAN_GENERATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": _CONTEXTUAL_CHOICE_PLAN_ITEM_SCHEMA,
        }
    },
    "required": ["items"],
}
def _contextual_choice_lexical_repair_schema(
    *, global_order: int, allowed_indexes: list[int]
) -> dict[str, Any]:
    replacement_count = len(allowed_indexes)
    return {
        "type": "OBJECT",
        "additionalProperties": False,
        "properties": {
            "globalOrder": {"type": "INTEGER", "enum": [global_order]},
            "targetExpression": {"type": ["STRING", "NULL"]},
            "distractorReplacements": {
                "type": "ARRAY",
                "minItems": replacement_count,
                "maxItems": replacement_count,
                "items": {
                    "type": "OBJECT",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "INTEGER", "enum": allowed_indexes},
                        "text": {"type": "STRING", "minLength": 1},
                    },
                    "required": ["index", "text"],
                },
            },
        },
        "required": ["globalOrder", "targetExpression", "distractorReplacements"],
    }
_CONTEXTUAL_CHOICE_PLAN_VERDICT_PROPERTIES: dict[str, Any] = {
    "globalOrder": {"type": "INTEGER"},
    "targetId": {"type": "STRING", "enum": ["TARGET"]},
    "targetExpressionWellFormed": {"type": "BOOLEAN"},
    "learningValue": {"type": "BOOLEAN"},
    "sameSurfaceCategory": {"type": "BOOLEAN"},
    "skillContrastSupported": {"type": "BOOLEAN"},
    "coveredDecisiveDimensions": {
        "type": "ARRAY",
        "items": {"type": "STRING"},
    },
    "definitionOnly": {"type": "BOOLEAN"},
    "lexicalConceptRepeated": {"type": "BOOLEAN"},
    "rareOrTrivia": {"type": "BOOLEAN"},
    "targetViableWithDifferentDistractors": {"type": "BOOLEAN"},
    "distractors": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "additionalProperties": False,
            "properties": {
                "index": {"type": "INTEGER"},
                "expressionWellFormed": {"type": "BOOLEAN"},
                "sameSurfaceCategory": {"type": "BOOLEAN"},
                "bundleRelevant": {"type": "BOOLEAN"},
                "closeCompetitor": {"type": "BOOLEAN"},
                "malformedByGrammar": {"type": "BOOLEAN"},
                "skillContrastRelevant": {"type": "BOOLEAN"},
                "coveredDecisiveDimensions": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
            },
            "required": [
                "index", "expressionWellFormed", "sameSurfaceCategory",
                "bundleRelevant", "closeCompetitor", "malformedByGrammar",
                "skillContrastRelevant", "coveredDecisiveDimensions",
            ],
        },
    },
    "reasonCode": {"type": "STRING", "enum": [
        "PASS", "NATURAL_TARGET", "LEARNING_VALUE",
        "SAME_SURFACE_CATEGORY", "SKILL_FIT", "DEFINITION_ONLY",
        "LEXICAL_REPEAT", "RARE_OR_TRIVIA", "INSUFFICIENT_CLOSE_DISTRACTORS",
        "DISTRACTOR_UNNATURAL", "DISTRACTOR_SURFACE_MISMATCH",
        "DISTRACTOR_GRAMMAR_ONLY", "DISTRACTOR_NOT_PLAUSIBLE",
        "DISTRACTOR_SKILL_MISMATCH",
    ]},
    "reason": {"type": "STRING"},
}


def _contextual_choice_plan_verification_schema(
    decisive_dimensions: tuple[str, ...],
    *,
    global_order: int,
) -> dict[str, Any]:
    properties = copy.deepcopy(_CONTEXTUAL_CHOICE_PLAN_VERDICT_PROPERTIES)
    properties["globalOrder"]["enum"] = [global_order]
    dimension_schema = {
        "type": "STRING",
        "enum": list(decisive_dimensions),
    }
    properties["coveredDecisiveDimensions"]["items"] = dimension_schema
    properties["distractors"]["items"]["properties"][
        "coveredDecisiveDimensions"
    ]["items"] = copy.deepcopy(dimension_schema)
    distractor_schema = properties["distractors"]["items"]
    distractor_schema["properties"].pop("index")
    distractor_schema["required"].remove("index")
    distractor_schema["required"].append("id")
    by_id: dict[str, Any] = {}
    for target_id in ("D0", "D1", "D2"):
        bound_schema = copy.deepcopy(distractor_schema)
        bound_schema["properties"]["id"] = {
            "type": "STRING", "enum": [target_id],
        }
        by_id[target_id] = bound_schema
    properties["distractors"] = {
        "type": "OBJECT",
        "additionalProperties": False,
        "properties": by_id,
        "required": list(by_id),
    }
    return {
        "type": "OBJECT",
        "additionalProperties": False,
        "properties": {
            "verdicts": {
                "type": "ARRAY",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "OBJECT",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": list(properties),
                },
            }
        },
        "required": ["verdicts"],
    }
_CONTEXTUAL_CHOICE_CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "additionalProperties": False,
    "properties": {
        "globalOrder": {"type": "INTEGER"},
        "contextTemplate": {"type": "STRING"},
        "explanationLearning": {"type": "STRING"},
    },
    "required": ["globalOrder", "contextTemplate", "explanationLearning"],
}
_CONTEXTUAL_CHOICE_CONTEXT_VERDICT_PROPERTIES: dict[str, Any] = {
    "order": {"type": "INTEGER"},
    "bestAnswerKey": {"type": "STRING", "enum": ["A", "B", "C", "D"]},
    "ambiguous": {"type": "BOOLEAN"},
    "supported": {"type": "BOOLEAN"},
    "skillFit": {"type": "BOOLEAN"},
    "definitionLike": {"type": "BOOLEAN"},
    "structurallyWellFormedKeys": {
        "type": "ARRAY",
        "items": {"type": "STRING", "enum": ["A", "B", "C", "D"]},
    },
    "reason": {"type": "STRING"},
}
_CONTEXTUAL_CHOICE_CONTEXT_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "additionalProperties": False,
    "properties": {
        "verdicts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "additionalProperties": False,
                "properties": _CONTEXTUAL_CHOICE_CONTEXT_VERDICT_PROPERTIES,
                "required": list(_CONTEXTUAL_CHOICE_CONTEXT_VERDICT_PROPERTIES),
            },
        }
    },
    "required": ["verdicts"],
}

_MEANING_RELATION_CANDIDATE_SCHEMA = copy.deepcopy(_PRACTICE_CANDIDATE_SCHEMA)
_meaning_relation_question_schema = _MEANING_RELATION_CANDIDATE_SCHEMA["properties"][
    "questions"
]["items"]
_meaning_relation_question_schema["properties"]["meaningContext"] = {
    "type": ["STRING", "NULL"]
}
_meaning_relation_question_schema["required"].append("meaningContext")
_meaning_relation_question_schema["properties"]["reserveDistractors"] = {
    "type": "ARRAY",
    "items": _OPTION_SCHEMA,
}
_meaning_relation_question_schema["required"].append("reserveDistractors")

# B3+ DISTINCTION has no model-owned answer. The provider supplies only semantic context,
# target identity, and wrong alternatives; the application inserts targetExpression as the
# correct option after deterministic filtering. Keep the legacy Meaning schema for B1/B2 and
# the rare mixed B2/B3 batch so their established task shape and call count stay unchanged.
_HIGH_BAND_MEANING_RELATION_CANDIDATE_SCHEMA = copy.deepcopy(
    _MEANING_RELATION_CANDIDATE_SCHEMA
)
_high_band_meaning_question_schema = _HIGH_BAND_MEANING_RELATION_CANDIDATE_SCHEMA[
    "properties"
]["questions"]["items"]
for _server_owned_field in ("prompt", "options", "correctAnswer"):
    _high_band_meaning_question_schema["properties"].pop(_server_owned_field)
    _high_band_meaning_question_schema["required"].remove(_server_owned_field)
_high_band_meaning_question_schema["properties"]["meaningContext"] = {
    "type": "STRING"
}
_high_band_meaning_question_schema["properties"]["distractors"] = {
    "type": "ARRAY",
    "items": _WRONG_OPTION_CANDIDATE_SCHEMA,
}
_high_band_meaning_question_schema["required"].append("distractors")
_high_band_meaning_question_schema["properties"]["reserveDistractors"] = {
    "type": "ARRAY",
    "items": _WRONG_OPTION_CANDIDATE_SCHEMA,
}

_READING_PASSAGE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "passageId": {"type": "STRING"},
        "passageText": {"type": "STRING"},
    },
    "required": ["passageId", "passageText"],
}

_B1_COMPREHENSION_PASSAGE_SCHEMA = copy.deepcopy(_READING_PASSAGE_SCHEMA)
_B1_COMPREHENSION_PASSAGE_SCHEMA["properties"].update({
    "inferenceClueQuote": {"type": "STRING"},
    "unstatedInference": {"type": "STRING"},
})
_B1_COMPREHENSION_PASSAGE_SCHEMA["required"].extend(
    ["inferenceClueQuote", "unstatedInference"]
)

_CONTEXT_INFERENCE_PASSAGE_SCHEMA = copy.deepcopy(_READING_PASSAGE_SCHEMA)
_CONTEXT_INFERENCE_PASSAGE_SCHEMA["properties"]["inferencePlans"] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "questionOrder": {"type": "INTEGER"},
            "clueQuote": {"type": "STRING"},
            "unstatedInference": {"type": "STRING"},
        },
        "required": ["questionOrder", "clueQuote", "unstatedInference"],
    },
}
_CONTEXT_INFERENCE_PASSAGE_SCHEMA["required"].append("inferencePlans")

_QUESTION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "globalOrder": {"type": "INTEGER"},
        "skillTag": {"type": "STRING"},
        "difficulty": {"type": "STRING", "enum": [item.value for item in PracticeDifficulty]},
        "complexityBand": {"type": "INTEGER"},
        "clueQuote": {"type": "STRING"},
        "questionFocus": {"type": "STRING"},
        "unstatedInference": {"type": ["STRING", "NULL"]},
    },
    "required": ["globalOrder", "skillTag", "difficulty", "complexityBand",
                 "clueQuote", "questionFocus", "unstatedInference"],
}
_PLANNED_PASSAGE_SCHEMAS: dict[str, dict[str, Any]] = {}
for _plan_schema_name, _base_passage_schema in (
    ("default", _READING_PASSAGE_SCHEMA),
    ("b1", _B1_COMPREHENSION_PASSAGE_SCHEMA),
    ("context", _CONTEXT_INFERENCE_PASSAGE_SCHEMA),
):
    _planned_schema = copy.deepcopy(_base_passage_schema)
    _planned_schema["properties"]["questionPlans"] = {
        "type": "ARRAY", "items": _QUESTION_PLAN_SCHEMA,
    }
    _planned_schema["required"].append("questionPlans")
    _PLANNED_PASSAGE_SCHEMAS[_plan_schema_name] = _planned_schema

_QUALITY_VERDICT_PROPERTIES: dict[str, Any] = {
    "order": {"type": "INTEGER"},
    "bestAnswerKey": {"type": "STRING"},
    "ambiguous": {"type": "BOOLEAN"},
    "supported": {"type": "BOOLEAN"},
    "reason": {"type": "STRING"},
    "modeFit": {"type": "BOOLEAN"},
    "answerLeakage": {"type": "BOOLEAN"},
    "contextDependent": {"type": "BOOLEAN"},
    "distractorsPlausible": {"type": "BOOLEAN"},
}
_QUALITY_VERDICT_REQUIRED = list(_QUALITY_VERDICT_PROPERTIES)

_PRACTICE_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdicts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": _QUALITY_VERDICT_PROPERTIES,
                "required": _QUALITY_VERDICT_REQUIRED,
            },
        },
    },
    "required": ["verdicts"],
}

_CONTEXTUAL_CHOICE_VERDICT_PROPERTIES: dict[str, Any] = {
    "candidateId": {"type": "STRING"},
    "bestAnswerKey": {"type": "STRING", "enum": ["A", "B", "C", "D"]},
    "ambiguous": {"type": "BOOLEAN"},
    "supported": {"type": "BOOLEAN"},
    "skillFit": {"type": "BOOLEAN"},
    "definitionLike": {"type": "BOOLEAN"},
    "lexicalConceptRepeated": {"type": "BOOLEAN"},
    "genuineCompetitorKeys": {
        "type": "ARRAY",
        "items": {"type": "STRING", "enum": ["A", "B", "C", "D"]},
    },
    "reason": {"type": "STRING"},
}
_CONTEXTUAL_CHOICE_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "additionalProperties": False,
    "properties": {
        "verdicts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "additionalProperties": False,
                "properties": _CONTEXTUAL_CHOICE_VERDICT_PROPERTIES,
                "required": list(_CONTEXTUAL_CHOICE_VERDICT_PROPERTIES),
            },
        }
    },
    "required": ["verdicts"],
}

_READING_DISTRACTOR_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "wrongOptions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "key": {"type": "STRING"},
                    "text": {"type": "STRING"},
                },
                "required": ["key", "text"],
            },
        }
    },
    "required": ["wrongOptions"],
}

_USAGE_PRESCREEN_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdicts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "order": {"type": "INTEGER"},
                    "modeFit": {"type": "BOOLEAN"},
                    "answerLeakage": {"type": "BOOLEAN"},
                    "contextDependent": {"type": "BOOLEAN"},
                    "reason": {"type": "STRING"},
                },
                "required": ["order", "modeFit", "answerLeakage", "contextDependent", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}


_ORIGIN_EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "explanations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "order": {"type": "INTEGER"},
                    "text": {"type": "STRING"},
                },
                "required": ["order", "text"],
            },
        }
    },
    "required": ["explanations"],
}

_READING_VERIFICATION_SCHEMA: dict[str, Any] = copy.deepcopy(_PRACTICE_VERIFICATION_SCHEMA)
_READING_VERDICT_PROPERTIES = _READING_VERIFICATION_SCHEMA["properties"]["verdicts"]["items"]["properties"]
_READING_VERDICT_PROPERTIES.update({
    "stemPresuppositionsSupported": {"type": "BOOLEAN"},
    "stemEvidenceSpanIds": {"type": "ARRAY", "items": {"type": "STRING"}},
    "readingOperation": {
        "type": "STRING",
        "enum": ["DIRECT_RETRIEVAL", "INFERENCE", "DISCOURSE_STRUCTURE"],
    },
    "distinctReadingTask": {"type": "BOOLEAN"},
})
_READING_VERIFICATION_SCHEMA["properties"]["verdicts"]["items"]["required"] = list(
    _READING_VERDICT_PROPERTIES
)
_STRUCTURE_VERIFICATION_SCHEMA = copy.deepcopy(_READING_VERIFICATION_SCHEMA)
_STRUCTURE_VERDICT = _STRUCTURE_VERIFICATION_SCHEMA["properties"]["verdicts"]["items"]
_STRUCTURE_VERDICT["properties"]["boundedStructureScope"] = {"type": "BOOLEAN"}
_STRUCTURE_VERDICT["required"].append("boundedStructureScope")
_CONTEXTUAL_CHOICE_ORIGIN_EXPLANATION_SCHEMA = copy.deepcopy(
    _ORIGIN_EXPLANATION_SCHEMA
)
_CONTEXTUAL_CHOICE_ORIGIN_EXPLANATION_SCHEMA["properties"]["explanations"][
    "items"
]["properties"]["explanationLearning"] = {"type": ["STRING", "NULL"]}


class _PracticeVerificationVerdict(BaseModel):
    order: int
    bestAnswerKey: str
    ambiguous: bool
    supported: bool
    reason: str
    modeFit: bool
    answerLeakage: bool
    contextDependent: bool
    distractorsPlausible: bool


class _PracticeVerificationPayload(BaseModel):
    verdicts: list[_PracticeVerificationVerdict]


class _ReadingPracticeVerificationVerdict(_PracticeVerificationVerdict):
    stemPresuppositionsSupported: StrictBool
    stemEvidenceSpanIds: list[str]
    readingOperation: Literal["DIRECT_RETRIEVAL", "INFERENCE", "DISCOURSE_STRUCTURE"]
    distinctReadingTask: StrictBool


class _StructurePracticeVerificationVerdict(_ReadingPracticeVerificationVerdict):
    boundedStructureScope: StrictBool


class _StructurePracticeVerificationPayload(BaseModel):
    verdicts: list[_StructurePracticeVerificationVerdict]


class _ReadingPracticeVerificationPayload(BaseModel):
    verdicts: list[_ReadingPracticeVerificationVerdict]


class _ContextualChoiceCandidatePayload(BaseModel):
    candidates: list[dict[str, Any]]


class _ContextualChoiceVerificationVerdict(BaseModel):
    candidateId: str
    bestAnswerKey: str
    ambiguous: bool
    supported: bool
    skillFit: bool
    definitionLike: bool
    lexicalConceptRepeated: bool
    genuineCompetitorKeys: tuple[str, ...]
    reason: str


class _ContextualChoiceVerificationPayload(BaseModel):
    verdicts: list[_ContextualChoiceVerificationVerdict]


class _ContextualChoicePlanGenerationPayload(BaseModel):
    items: list[dict[str, Any]]


class _ContextualChoiceDistractorVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Literal["D0", "D1", "D2"]
    expressionWellFormed: StrictBool
    sameSurfaceCategory: StrictBool
    bundleRelevant: StrictBool
    closeCompetitor: StrictBool
    malformedByGrammar: StrictBool
    skillContrastRelevant: StrictBool
    coveredDecisiveDimensions: tuple[str, ...]

    @property
    def index(self) -> int:
        return {"D0": 0, "D1": 1, "D2": 2}[self.id]


class _ContextualChoiceDistractorVerdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    D0: _ContextualChoiceDistractorVerdict
    D1: _ContextualChoiceDistractorVerdict
    D2: _ContextualChoiceDistractorVerdict

    def ordered(self) -> tuple[_ContextualChoiceDistractorVerdict, ...]:
        return (self.D0, self.D1, self.D2)


class _ContextualChoicePlanVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    globalOrder: StrictInt
    targetId: Literal["TARGET"]
    targetExpressionWellFormed: StrictBool
    learningValue: StrictBool
    sameSurfaceCategory: StrictBool
    skillContrastSupported: StrictBool
    coveredDecisiveDimensions: tuple[str, ...]
    definitionOnly: StrictBool
    lexicalConceptRepeated: StrictBool
    rareOrTrivia: StrictBool
    targetViableWithDifferentDistractors: StrictBool
    distractors: _ContextualChoiceDistractorVerdicts
    reasonCode: str
    reason: str


class _ContextualChoicePlanVerificationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[_ContextualChoicePlanVerdict]


class _ContextualChoiceLexicalVerifierContractError(RuntimeError):
    """Provider verdict contract failure, never content-quality/recovery evidence."""

    def __init__(self, reason_code: str, *, expected_order: int) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.expected_order = expected_order


def _contextual_choice_verifier_coverage_summary(raw: Any) -> dict[str, object]:
    """Only bounded structural identities/counts, never arbitrary provider text."""
    verdicts = raw.get("verdicts") if isinstance(raw, dict) else None
    if not isinstance(verdicts, list):
        return {"verdictCount": None}
    summary: dict[str, object] = {"verdictCount": len(verdicts)}
    orders: list[int | str] = []
    for verdict in verdicts[:10]:
        order = verdict.get("globalOrder") if isinstance(verdict, dict) else None
        orders.append(order if type(order) is int and 1 <= order <= 10 else "INVALID")
    summary["orders"] = orders
    if len(verdicts) != 1 or not isinstance(verdicts[0], dict):
        return summary
    distractors = verdicts[0].get("distractors")
    summary["distractorCount"] = (
        len(distractors) if isinstance(distractors, (dict, list)) else None
    )
    if isinstance(distractors, dict):
        summary["distractorIds"] = [
            key if key in {"D0", "D1", "D2", "D3"} else "UNEXPECTED_ID"
            for key in list(distractors)[:10]
        ]
        summary["boundIds"] = [
            evidence.get("id")
            if isinstance(evidence, dict) and evidence.get("id") in ("D0", "D1", "D2", "D3")
            else "INVALID"
            for evidence in list(distractors.values())[:10]
        ]
    elif isinstance(distractors, list):
        summary["legacyIndexes"] = [
            evidence.get("index")
            if isinstance(evidence, dict) and type(evidence.get("index")) is int
            and 0 <= evidence["index"] <= 10 else "INVALID"
            for evidence in distractors[:10]
        ]
    return summary


def _parse_contextual_choice_plan_verdict(
    raw: Any, *, expected_order: int,
) -> _ContextualChoicePlanVerdict:
    def fail(reason_code: str) -> NoReturn:
        raise _ContextualChoiceLexicalVerifierContractError(
            reason_code, expected_order=expected_order,
        )

    if not isinstance(raw, dict) or not isinstance(raw.get("verdicts"), list):
        fail("LEXICAL_VERIFIER_RESPONSE_SCHEMA_INVALID")
    verdicts = raw["verdicts"]
    if len(verdicts) != 1:
        fail("LEXICAL_VERIFIER_ORDER_COVERAGE_INVALID")
    verdict = verdicts[0]
    if not isinstance(verdict, dict):
        fail("LEXICAL_VERIFIER_RESPONSE_SCHEMA_INVALID")
    if type(verdict.get("globalOrder")) is not int or verdict["globalOrder"] != expected_order:
        fail("LEXICAL_VERIFIER_ORDER_MISMATCH")
    distractors = verdict.get("distractors")
    if not isinstance(distractors, dict) or set(distractors) != {"D0", "D1", "D2"}:
        fail("LEXICAL_VERIFIER_DISTRACTOR_COVERAGE_INVALID")
    if verdict.get("targetId") != "TARGET":
        fail("LEXICAL_VERIFIER_TARGET_ID_INVALID")
    for target_id in ("D0", "D1", "D2"):
        evidence = distractors[target_id]
        if not isinstance(evidence, dict) or evidence.get("id") != target_id:
            fail("LEXICAL_VERIFIER_DISTRACTOR_ID_MISMATCH")
    try:
        payload = _ContextualChoicePlanVerificationPayload.model_validate(raw)
    except ValidationError as exc:
        raise _ContextualChoiceLexicalVerifierContractError(
            "LEXICAL_VERIFIER_RESPONSE_SCHEMA_INVALID", expected_order=expected_order,
        ) from exc
    return payload.verdicts[0]


class _ContextualChoiceDistractorReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    text: str


class _ContextualChoiceLexicalRepairPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    globalOrder: int
    targetExpression: str | None
    distractorReplacements: list[_ContextualChoiceDistractorReplacement]


_ContextualChoiceLexicalRepairReasonCode = Literal[
    "LEXICAL_REPAIR_RESPONSE_SCHEMA_INVALID",
    "LEXICAL_REPAIR_ORDER_MISMATCH",
    "LEXICAL_REPAIR_SCOPE_MISMATCH",
    "LEXICAL_REPAIR_TARGET_AUTHORITY_VIOLATION",
    "LEXICAL_REPAIR_EMPTY_DISTRACTOR",
    "LEXICAL_REPAIR_UNCHANGED_DISTRACTOR",
    "LEXICAL_REPAIR_DUPLICATE_TARGET",
    "LEXICAL_REPAIR_DUPLICATE_DISTRACTOR",
    "LEXICAL_REPAIR_LANGUAGE_INVALID",
    "LEXICAL_REPAIR_PLAN_AUTHORITY_VIOLATION",
]


class _ContextualChoiceLexicalRepairContractError(ValueError):
    def __init__(
        self,
        reason_code: _ContextualChoiceLexicalRepairReasonCode,
        *,
        index: int | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.index = index


@dataclass(frozen=True)
class _ContextualChoiceDistractorRepairEvidence:
    index: int
    expression_well_formed: bool
    same_surface_category: bool
    bundle_relevant: bool
    close_competitor: bool
    malformed_by_grammar: bool
    skill_contrast_relevant: bool
    covered_decisive_dimensions: tuple[str, ...]

    def prompt_payload(self) -> dict[str, bool | list[str]]:
        return {
            "expressionWellFormed": self.expression_well_formed,
            "sameSurfaceCategory": self.same_surface_category,
            "bundleRelevant": self.bundle_relevant,
            "closeCompetitor": self.close_competitor,
            "malformedByGrammar": self.malformed_by_grammar,
            "skillContrastRelevant": self.skill_contrast_relevant,
            "coveredDecisiveDimensions": list(
                self.covered_decisive_dimensions
            ),
        }


@dataclass(frozen=True)
class _ContextualChoiceLexicalAssessment:
    reason_codes: tuple[str, ...]
    invalid_distractor_indexes: tuple[int, ...]
    preserve_new_target: bool
    distractor_repair_evidence: tuple[
        _ContextualChoiceDistractorRepairEvidence, ...
    ]
    required_close_distractors: int
    current_valid_close_distractor_count: int
    additional_close_distractors_needed: int
    close_required_replacement_indexes: tuple[int, ...]
    skill: str
    decisive_dimensions: tuple[str, ...]


@dataclass(frozen=True)
class _ContextualChoiceContextAssessment:
    reason_codes: tuple[str, ...]
    failure_evidence: dict[str, object]


_ContextFailureClass = Literal[
    "CONTRACT_CONTEXT",
    "STRUCTURAL_CONTEXT",
    "SEMANTIC_CONTEXT",
]
_CONTRACT_CONTEXT_REASON_CODES = frozenset({
    "CONTEXT_RESPONSE_SCHEMA_INVALID",
    "CONTEXT_ORDER_MISMATCH",
    "CONTEXT_MARKER_INVALID",
    "CONTEXT_UNKNOWN_MARKER",
    "CONTEXT_LANGUAGE_INVALID",
    "CONTEXT_TARGET_LEAK",
    "CONTEXT_TEMPLATE_ROUND_TRIP_INVALID",
    "CONTEXT_EXPLANATION_INVALID",
})
_STRUCTURAL_CONTEXT_REASON_CODES = frozenset({
    "RENDERED_SENTENCE_STRUCTURALLY_INVALID",
})
_SEMANTIC_CONTEXT_REASON_CODES = frozenset({
    "AMBIGUOUS",
    "UNSUPPORTED",
    "SKILL_FIT",
    "DEFINITION_LIKE",
    "BEST_ANSWER_MISMATCH",
})


class _ContextualChoiceContextContractError(ValueError):
    def __init__(self, reason_code: str, *, repairable: bool) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.repairable = repairable


class _ContextualChoiceContextVerifierContractError(ValueError):
    reason_code = "CONTEXT_VERIFIER_CONTRACT_INVALID"


class _ContextualChoiceContextPayload(BaseModel):
    globalOrder: int
    contextTemplate: str
    explanationLearning: str


class _ContextualChoiceContextVerdict(BaseModel):
    order: int
    bestAnswerKey: str
    ambiguous: bool
    supported: bool
    skillFit: bool
    definitionLike: bool
    structurallyWellFormedKeys: tuple[str, ...]
    reason: str


class _ContextualChoiceContextVerificationPayload(BaseModel):
    verdicts: list[_ContextualChoiceContextVerdict]


@dataclass
class _ContextualChoiceTelemetry:
    started_at: float
    plan_generation_calls: int = 0
    lexical_validation_calls: int = 0
    plan_repair_calls: int = 0
    lexical_repair_calls: int = 0
    context_generation_calls: int = 0
    context_validation_calls: int = 0
    context_repair_calls: int = 0
    origin_explanation_calls: int = 0
    plan_reused: bool = False

    @property
    def total_ai_calls(self) -> int:
        return (
            self.plan_generation_calls
            + self.lexical_validation_calls
            + self.plan_repair_calls
            + self.lexical_repair_calls
            + self.context_generation_calls
            + self.context_validation_calls
            + self.context_repair_calls
            + self.origin_explanation_calls
        )


class _ContextualChoiceContentRejected(ValueError):
    def __init__(self, message: str, *, stage: str = "UNKNOWN") -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class _ContextualChoiceCandidate:
    candidate_id: str
    question: PracticeGeneratedQuestion


class _ReadingDistractorRepairOption(BaseModel):
    key: str
    text: str


class _ReadingDistractorRepairPayload(BaseModel):
    wrongOptions: list[_ReadingDistractorRepairOption]


@dataclass(frozen=True)
class _SemanticVerificationOutcome:
    failures: dict[int, str]
    verdicts: dict[int, _PracticeVerificationVerdict]


class _UsagePrescreenVerdict(BaseModel):
    order: int
    modeFit: bool = True
    answerLeakage: bool = False
    contextDependent: bool = True
    reason: str = ""


class _UsagePrescreenPayload(BaseModel):
    verdicts: list[_UsagePrescreenVerdict]


class _PracticeCandidatePayload(BaseModel):
    questions: list[dict[str, Any]]


class _ReadingInferencePlan(BaseModel):
    questionOrder: int
    clueQuote: str
    unstatedInference: str


class _ReadingPassagePayload(BaseModel):
    passageId: str
    passageText: str
    inferenceClueQuote: str | None = None
    unstatedInference: str | None = None
    inferencePlans: list[_ReadingInferencePlan] | None = None
    questionPlans: list[ReadingQuestionPlan] | None = None


@dataclass(frozen=True)
class _PlannedPassage:
    text: str
    plans: tuple[ReadingQuestionPlan, ...] = ()


class _OriginExplanationItem(BaseModel):
    order: int
    text: str
    explanationLearning: str | None = None


class _OriginExplanationPayload(BaseModel):
    explanations: list[_OriginExplanationItem]


@dataclass(frozen=True)
class _QuestionSlot:
    order: int
    difficulty: PracticeDifficulty
    complexity_band: int
    skill_tag: str
    question_type: PracticeQuestionType | None = None
    passage_id: str | None = None
    passage_text: str | None = None
    reading_plan: ReadingQuestionPlan | None = None
    same_passage_plans: tuple[ReadingQuestionPlan, ...] = ()
    review_canonical_key: str | None = None
    review_expression: str | None = None
    previous_question_types: tuple[str, ...] = ()
    usage_intent: str | None = None
    reading_mode: str | None = None
    vocabulary_mode: str | None = None
    free_target_focus: str | None = None
    scenario_family: str | None = None
    vocabulary_plan_item: VocabularyPlanItem | None = None

    @property
    def review_target(self) -> bool:
        return self.review_canonical_key is not None

    @property
    def target_as_answer(self) -> bool:
        return (
            self.vocabulary_mode == VocabularyMode.CONTEXTUAL_CHOICE.value
            or (
                self.vocabulary_mode == VocabularyMode.MEANING_RELATION.value
                and self.skill_tag == VocabularySkill.DISTINCTION.value
                and self.complexity_band >= 3
            )
        )

    def prompt_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "order": self.order,
            "difficulty": self.difficulty.value,
            "complexityBand": self.complexity_band,
            "skillTag": self.skill_tag,
            "passageId": self.passage_id,
            "passageText": self.passage_text,
            "reviewTarget": self.review_target,
        }
        if self.question_type is not None:
            payload["questionType"] = self.question_type.value
        if self.passage_id is not None:
            if self.reading_mode is None:
                raise ValueError("Reading slot requires its server-selected mode")
            payload["difficultyRecipe"] = question_demand_recipe(
                self.complexity_band,
                mode=self.reading_mode,
                skill_tag=self.skill_tag,
            ).generation_payload()
            if self.reading_plan is not None:
                payload["currentQuestionPlan"] = self.reading_plan.model_dump(by_alias=True)
                payload["samePassageQuestionPlans"] = [
                    plan.model_dump(by_alias=True) for plan in self.same_passage_plans
                ]
        if self.vocabulary_mode is not None:
            if self.question_type is None:
                raise ValueError("Vocabulary slot requires its server-selected question type")
            payload["difficultyRecipe"] = vocabulary_difficulty_recipe(
                mode=self.vocabulary_mode,
                band=self.complexity_band,
                skill_tag=self.skill_tag,
                question_type=self.question_type.value,
                usage_intent=self.usage_intent,
            ).generation_payload()
            if (
                self.vocabulary_mode == VocabularyMode.MEANING_RELATION.value
                and self.skill_tag == VocabularySkill.DISTINCTION.value
            ):
                payload["reserveDistractorCount"] = (
                    _MEANING_RELATION_RESERVE_DISTRACTOR_COUNT
                )
            if self.target_as_answer:
                payload["answerAuthority"] = "APPLICATION_TARGET_EXPRESSION"
                payload["targetVisibility"] = "FINAL_OPTIONS_ONLY"
                payload["primaryDistractorCount"] = 3
            if self.vocabulary_mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
                payload["answerAuthority"] = "APPLICATION_TARGET_EXPRESSION"
                payload["targetVisibility"] = "FINAL_OPTIONS_ONLY"
                payload["generationShape"] = "COMPLETE_SENTENCE_FIRST"
                payload["candidateCount"] = CONTEXTUAL_CHOICE_CANDIDATES_PER_ROUND
                payload["primaryDistractorCount"] = 3
                if self.scenario_family is not None:
                    payload["scenarioFamily"] = self.scenario_family
                    payload["scenarioAuthority"] = "SERVER_SELECTED_REQUIREMENT"
        if self.free_target_focus is not None:
            payload["freeTargetFocus"] = self.free_target_focus
            payload["focusAuthority"] = "SERVER_SELECTED_HINT"
        if self.usage_intent is not None:
            payload["usageIntent"] = self.usage_intent
            payload["answerVisibilityPolicy"] = "HIDE_TARGET_FROM_STEM"
        if self.review_target:
            payload["boundReviewTarget"] = {
                "canonicalKey": self.review_canonical_key,
                "expression": self.review_expression,
                "previousQuestionTypes": list(self.previous_question_types),
            }
        return payload


class ReadingVocabularyGenerationService:
    TYPE_NAME = "LANGUAGE_LEARNING_READING_VOCABULARY_GENERATION"
    READING_SOL_TYPE_NAME = "LANGUAGE_LEARNING_READING_QUESTION_GENERATION_SOL"

    def _generation_timeout_for(self, request: PracticeGenerationRequest) -> float:
        if request.domain == PracticeDomain.READING and settings.AI_READING_GENERATION_MODEL == "SOL":
            return max(self.timeout_seconds, 80.0)
        return self.timeout_seconds
    DISTRACTOR_REPAIR_TYPE_NAME = "LANGUAGE_LEARNING_READING_DISTRACTOR_REPAIR"
    PASSAGE_TYPE_NAME = "LANGUAGE_LEARNING_READING_PASSAGE_GENERATION"
    PRESCREEN_TYPE_NAME = "LANGUAGE_LEARNING_READING_VOCABULARY_USAGE_PRESCREEN"
    VERIFICATION_TYPE_NAME = "LANGUAGE_LEARNING_READING_VOCABULARY_VERIFICATION"
    CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME = (
        "LANGUAGE_LEARNING_VOCABULARY_CONTEXTUAL_CHOICE_VERIFICATION"
    )
    CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME = (
        "LANGUAGE_LEARNING_VOCABULARY_CONTEXTUAL_CHOICE_PLAN_VERIFICATION"
    )
    ORIGIN_EXPLANATION_TYPE_NAME = "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION"
    ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME = (
        "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION_FALLBACK"
    )

    def __init__(
        self,
        provider: TextGenerationProvider,
        timeout_seconds: float = 45.0,
        *,
        difficulty_shadow_enabled: bool = False,
        difficulty_shadow_sample_percent: float = 0.0,
        difficulty_shadow_timeout_seconds: float = 12.0,
        vocabulary_difficulty_shadow_enabled: bool = False,
        vocabulary_difficulty_shadow_sample_percent: float = 0.0,
        vocabulary_difficulty_shadow_timeout_seconds: float = 12.0,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.difficulty_shadow = ReadingDifficultyShadowCollector(
            provider,
            enabled=difficulty_shadow_enabled,
            sample_percent=difficulty_shadow_sample_percent,
            timeout_seconds=difficulty_shadow_timeout_seconds,
        )
        self.vocabulary_difficulty_shadow = VocabularyDifficultyShadowCollector(
            provider,
            enabled=vocabulary_difficulty_shadow_enabled,
            sample_percent=vocabulary_difficulty_shadow_sample_percent,
            timeout_seconds=vocabulary_difficulty_shadow_timeout_seconds,
        )

    async def generate(self, request: PracticeGenerationRequest) -> PracticeGenerationResponse:
        """Generate with application-owned planning, validation and smallest-unit retries.

        Core questions, semantic verification and origin-language explanations are independent
        stages. A bad explanation therefore repairs only the explanation; an ambiguous question
        regenerates only that slot; successful slots are retained.
        """
        contextual_choice = (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value
        )
        telemetry = (
            _ContextualChoiceTelemetry(started_at=time.perf_counter())
            if contextual_choice
            else None
        )
        try:
            passages = await self._generate_reading_passages(request)
            vocabulary_plan = None
            if contextual_choice:
                assert telemetry is not None
                vocabulary_plan = await self._resolve_contextual_choice_plan(
                    request,
                    telemetry,
                )
                if request.vocabulary_plan_only:
                    response = PracticeGenerationResponse(
                        request_id=request.request_id,
                        prompt_version=CONTEXTUAL_CHOICE_RECIPE_VERSION,
                        domain=request.domain,
                        mode=request.mode,
                        complexity_band=request.complexity_band,
                        questions=[],
                        vocabulary_plan=vocabulary_plan,
                    )
                    logger.info(
                        "CONTEXTUAL_CHOICE V3 plan snapshot completed. request_id=%s "
                        "generated_item_count=0 plan_generation_calls=%d "
                        "lexical_validation_calls=%d plan_repair_calls=%d "
                        "plan_reused=%s total_ai_calls=%d total_generation_latency_ms=%d",
                        request.request_id,
                        telemetry.plan_generation_calls,
                        telemetry.lexical_validation_calls,
                        telemetry.plan_repair_calls,
                        telemetry.plan_reused,
                        telemetry.total_ai_calls,
                        int((time.perf_counter() - telemetry.started_at) * 1000),
                    )
                    return response
                slots = self._contextual_choice_slots_from_plan(
                    request,
                    vocabulary_plan,
                )
            else:
                slots = self._build_slots(request, passages)
            questions = await self._generate_verified_questions(
                request,
                slots,
                contextual_choice_telemetry=telemetry,
                contextual_choice_plan=vocabulary_plan,
            )
            if contextual_choice and vocabulary_plan is not None:
                revised_items = list(vocabulary_plan.items)
                for slot in slots:
                    if slot.vocabulary_plan_item is not None:
                        revised_items[slot.vocabulary_plan_item.global_order - 1] = (
                            slot.vocabulary_plan_item
                        )
                vocabulary_plan = PersonalizedVocabularyPlan(
                    version=vocabulary_plan.version, items=revised_items
                )
                self._validate_contextual_choice_plan_authority(request, vocabulary_plan)
            questions = await self._attach_origin_explanations(
                request,
                questions,
                telemetry=telemetry,
            )
            questions = sorted(questions, key=lambda item: item.order)
            self._validate(request, questions)
            reading_bundle = None
            if request.domain == PracticeDomain.READING and request.question_count in (2, 3):
                passage_id = "p1" if request.question_offset == 0 else "p2"
                planned_passage = passages[passage_id]
                if not planned_passage.plans or len(questions) != len(planned_passage.plans):
                    raise ValueError("Reading passage bundle is incomplete")
                reading_bundle = ReadingPassageBundle(
                    passage_id=passage_id,
                    prompt_version=PRACTICE_GENERATION_PROMPT_VERSION,
                    passage_sha256=hashlib.sha256(planned_passage.text.encode("utf-8")).hexdigest(),
                    question_plans=list(planned_passage.plans),
                    questions=questions,
                )
            response = PracticeGenerationResponse(
                request_id=request.request_id,
                prompt_version=(
                    CONTEXTUAL_CHOICE_RECIPE_VERSION
                    if request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value
                    else PRACTICE_GENERATION_PROMPT_VERSION
                ),
                domain=request.domain,
                mode=request.mode,
                complexity_band=request.complexity_band,
                questions=questions,
                vocabulary_plan=vocabulary_plan,
                reading_bundle=reading_bundle,
            )
            await self.difficulty_shadow.collect_if_selected(request, response.questions)
            if not contextual_choice:
                await self.vocabulary_difficulty_shadow.collect_if_selected(
                    request, response.questions
                )
            if telemetry is not None:
                logger.info(
                    "CONTEXTUAL_CHOICE V3 generation completed. request_id=%s "
                    "generated_item_count=%d plan_generation_calls=%d "
                    "lexical_validation_calls=%d plan_repair_calls=%d "
                    "lexical_repair_calls=%d "
                    "context_generation_calls=%d context_validation_calls=%d "
                    "context_repair_calls=%d plan_reused=%s total_ai_calls=%d "
                    "total_generation_latency_ms=%d",
                    request.request_id,
                    len(response.questions),
                    telemetry.plan_generation_calls,
                    telemetry.lexical_validation_calls,
                    telemetry.plan_repair_calls,
                    telemetry.lexical_repair_calls,
                    telemetry.context_generation_calls,
                    telemetry.context_validation_calls,
                    telemetry.context_repair_calls,
                    telemetry.plan_reused,
                    telemetry.total_ai_calls,
                    int((time.perf_counter() - telemetry.started_at) * 1000),
                )
            return response
        except _ContextualChoiceContentRejected as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": (
                        "AI_SCHEMA_INVALID" if exc.stage == "PLAN_AUTHORITY"
                        else "AI_CONTENT_QUALITY_REJECTED"
                    ),
                    "retryable": False,
                    "message": "Vocabulary lexical plan/context 품질 검증에 실패했습니다.",
                    "cause": type(exc).__name__,
                    "stage": exc.stage,
                },
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(
                "Reading/Vocabulary generation pipeline failed. request_id=%s stage=final type=%s",
                request.request_id,
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "AI_GENERATION_FAILED",
                    "retryable": True,
                    "message": "Reading/Vocabulary 문제 생성에 실패했습니다.",
                    "cause": type(exc).__name__,
                },
            ) from exc

    async def _generate_reading_passages(
        self,
        request: PracticeGenerationRequest,
    ) -> dict[str, _PlannedPassage]:
        if request.domain != PracticeDomain.READING:
            return {}

        passages = {
            question.passage_id: _PlannedPassage(question.passage_text)
            for question in request.previous_questions
            if question.passage_id and question.passage_text
        }
        planned_bundle = request.question_count in (2, 3)
        planned_slots = self._build_slots(request, {}) if planned_bundle else []
        required_passages = {
            1 if index <= 3 else 2
            for index in range(
                request.question_offset + 1,
                request.question_offset + request.question_count + 1,
            )
        }
        for passage_number in sorted(required_passages):
            passage_id = f"p{passage_number}"
            if passage_id in passages:
                self._assert_language_lane(
                    request.learning_language,
                    [passages[passage_id].text],
                    reason="Reading passage is not in learningLanguage",
                )
                continue
            passage_spec = build_reading_passage_difficulty_spec(
                request,
                passage_id=passage_id,
                complexity_band=request.complexity_band,
            )
            last_error: Exception | None = None
            for attempt in range(1, _MAX_PASSAGE_ATTEMPTS + 1):
                try:
                    raw = await asyncio.wait_for(
                        self.provider.call(
                            type_name=self.PASSAGE_TYPE_NAME,
                            data=build_reading_passage_prompt(
                                request,
                                passage_id=passage_id,
                                passage_number=passage_number,
                                planned_slots=([
                                    {
                                        "globalOrder": request.question_offset + slot.order,
                                        "skillTag": slot.skill_tag,
                                        "difficulty": slot.difficulty.value,
                                        "complexityBand": slot.complexity_band,
                                        "questionDemand": question_demand_recipe(
                                            slot.complexity_band, mode=request.mode,
                                            skill_tag=slot.skill_tag,
                                        ).generation_payload(),
                                        **({"structureJudgmentContract": structure_judgment_contract(
                                            band=slot.complexity_band,
                                            position=(request.question_offset + slot.order
                                                      if request.question_offset + slot.order <= 3
                                                      else request.question_offset + slot.order - 3),
                                            count=3 if passage_number == 1 else 2,
                                        )} if request.mode == ReadingMode.STRUCTURE.value
                                            and slot.complexity_band == 5 else {}),
                                    }
                                    for slot in planned_slots if slot.passage_id == passage_id
                                ] if planned_bundle else None),
                            ),
                            schema=(
                                _PLANNED_PASSAGE_SCHEMAS[
                                    "b1" if request.mode == ReadingMode.COMPREHENSION.value
                                    and request.complexity_band == 1 else
                                    "context" if request.mode == ReadingMode.CONTEXT_INFERENCE.value
                                    else "default"
                                ] if planned_bundle else
                                _B1_COMPREHENSION_PASSAGE_SCHEMA
                                if request.mode == ReadingMode.COMPREHENSION.value
                                and request.complexity_band == 1
                                else _CONTEXT_INFERENCE_PASSAGE_SCHEMA
                                if request.mode == ReadingMode.CONTEXT_INFERENCE.value
                                else _READING_PASSAGE_SCHEMA
                            ),
                        ),
                        timeout=self._generation_timeout_for(request),
                    )
                    payload = _ReadingPassagePayload.model_validate(raw)
                    passage_text = payload.passageText.strip()
                    passage_validation = validate_reading_passage(
                        passage_spec,
                        observed_passage_id=payload.passageId,
                        passage_text=passage_text,
                        language_validator=lambda: self._assert_language_lane(
                            request.learning_language,
                            [passage_text],
                            reason="Reading passage is not in learningLanguage",
                        ),
                    )
                    if not passage_validation.passed:
                        raise ValueError(passage_validation.primary_issue)
                    if (request.mode == ReadingMode.COMPREHENSION.value
                            and request.complexity_band == 1):
                        clue = (payload.inferenceClueQuote or "").strip()
                        inference = (payload.unstatedInference or "").strip()
                        if (not clue or not inference or clue not in passage_text
                                or self._normalize_for_leak_check(inference)
                                in self._normalize_for_leak_check(passage_text)):
                            raise ValueError("B1 comprehension passage lacks an explicit implicit-inference plan")
                        self._assert_language_lane(
                            request.learning_language, [inference],
                            reason="Reading unstated inference is not in learningLanguage",
                        )
                    if request.mode == ReadingMode.CONTEXT_INFERENCE.value:
                        plans = payload.inferencePlans or []
                        expected_orders = [1, 2, 3] if passage_number == 1 else [4, 5]
                        if [plan.questionOrder for plan in plans] != expected_orders:
                            raise ValueError("Context inference passage lacks one plan per future slot")
                        clues: set[str] = set()
                        inferences: set[str] = set()
                        normalized_passage = self._normalize_for_leak_check(passage_text)
                        for plan in plans:
                            clue = plan.clueQuote.strip()
                            inference = plan.unstatedInference.strip()
                            normalized_clue = self._normalize_for_leak_check(clue)
                            normalized_inference = self._normalize_for_leak_check(inference)
                            if (not clue or not inference or clue not in passage_text
                                    or normalized_clue in clues or normalized_inference in inferences
                                    or normalized_inference in normalized_passage):
                                raise ValueError("Context inference plans must be distinct and unstated")
                            clues.add(normalized_clue)
                            inferences.add(normalized_inference)
                            self._assert_language_lane(
                                request.learning_language, [inference],
                                reason="Reading unstated inference is not in learningLanguage",
                            )
                    question_plans = tuple(payload.questionPlans or ())
                    if planned_bundle:
                        passage_slots = [slot for slot in planned_slots if slot.passage_id == passage_id]
                        expected_orders = [request.question_offset + slot.order for slot in passage_slots]
                        if [plan.global_order for plan in question_plans] != expected_orders:
                            raise ValueError("Reading plan does not cover exact passage orders")
                        focuses: set[str] = set()
                        for plan, slot in zip(question_plans, passage_slots, strict=True):
                            focus = self._normalize_for_leak_check(plan.question_focus)
                            if (plan.skill_tag != slot.skill_tag
                                    or plan.difficulty != slot.difficulty
                                    or plan.complexity_band != slot.complexity_band
                                    or not plan.clue_quote.strip()
                                    or plan.clue_quote not in passage_text
                                    or not focus or focus in focuses):
                                raise ValueError("Reading passage plan violates server slot binding")
                            focuses.add(focus)
                            if slot.skill_tag in {ReadingSkill.INFERENCE.value,
                                                  ReadingSkill.CONTEXT_INFERENCE.value}:
                                inference = (plan.unstated_inference or "").strip()
                                if (not inference or self._normalize_for_leak_check(inference)
                                        in self._normalize_for_leak_check(passage_text)):
                                    raise ValueError("Reading inference plan lacks an unstated conclusion")
                                self._assert_language_lane(request.learning_language, [inference],
                                    reason="Reading plan inference is not in learningLanguage")
                    passages[passage_id] = _PlannedPassage(passage_text, question_plans)
                    break
                except (ValidationError, ValueError, TimeoutError, asyncio.TimeoutError) as exc:
                    last_error = exc
                    logger.warning(
                        "Reading passage rejected. request_id=%s passage=%s attempt=%d/%d reason=%s",
                        request.request_id,
                        passage_id,
                        attempt,
                        _MAX_PASSAGE_ATTEMPTS,
                        str(exc)[:300],
                    )
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Reading passage provider failure. request_id=%s passage=%s attempt=%d/%d type=%s",
                        request.request_id,
                        passage_id,
                        attempt,
                        _MAX_PASSAGE_ATTEMPTS,
                        type(exc).__name__,
                    )
            if passage_id not in passages:
                raise ValueError(
                    f"reading passage generation exhausted passage={passage_id} "
                    f"cause={type(last_error).__name__ if last_error else 'unknown'}"
                )
        return passages

    def _eligible_review_targets(
        self,
        request: PracticeGenerationRequest,
        *,
        log_rejections: bool = True,
    ) -> list[PracticeReviewTarget]:
        if request.domain != PracticeDomain.VOCABULARY:
            return []
        committed_keys = {question.canonical_key for question in request.previous_questions}
        committed_expressions = {
            self._normalize_for_leak_check(question.target_expression or "")
            for question in request.previous_questions
        }
        eligible = []
        for target in request.review_targets:
            if target.canonical_key in committed_keys or self._normalize_for_leak_check(
                target.expression
            ) in committed_expressions:
                continue
            try:
                self._assert_vocabulary_surface_language(
                    request.learning_language,
                    target.expression,
                    reason=(
                        "review target expression is not a lexical expression in learningLanguage"
                    ),
                )
            except ValueError as exc:
                if log_rejections:
                    logger.warning(
                        "Reading/Vocabulary review target skipped by language lane. "
                        "request_id=%s canonical_key=%s expression=%s reason=%s",
                        request.request_id,
                        target.canonical_key,
                        target.expression[:120],
                        str(exc)[:220],
                    )
                continue
            eligible.append(target)
        return eligible

    def _build_slots(
        self,
        request: PracticeGenerationRequest,
        passages: Mapping[str, str | _PlannedPassage],
    ) -> list[_QuestionSlot]:
        difficulties = self._difficulty_plan(request)
        if request.domain == PracticeDomain.READING:
            if request.reading_slot_targets:
                selected_targets = request.reading_slot_targets[
                    request.question_offset: request.question_offset + request.question_count
                ]
                if (len(selected_targets) != request.question_count
                        or sum(target.difficulty == PracticeDifficulty.EASIER
                               for target in selected_targets) != request.easier_count
                        or sum(target.difficulty == PracticeDifficulty.CURRENT
                               for target in selected_targets) != request.current_count
                        or sum(target.difficulty == PracticeDifficulty.CHALLENGE
                               for target in selected_targets) != request.challenge_count):
                    raise ValueError("Reading slot targets changed difficulty quotas")
            skill_cycle = {
                ReadingMode.COMPREHENSION.value: [
                    ReadingSkill.CONTENT.value,
                    ReadingSkill.DETAIL.value,
                    ReadingSkill.INFERENCE.value,
                    ReadingSkill.DETAIL.value,
                    ReadingSkill.INFERENCE.value,
                ],
                ReadingMode.STRUCTURE.value: [
                    ReadingSkill.GIST.value,
                    ReadingSkill.STRUCTURE.value,
                    ReadingSkill.STRUCTURE.value,
                    ReadingSkill.GIST.value,
                    ReadingSkill.STRUCTURE.value,
                ],
                ReadingMode.CONTEXT_INFERENCE.value: [
                    ReadingSkill.CONTEXT_INFERENCE.value,
                    ReadingSkill.INFERENCE.value,
                    ReadingSkill.CONTEXT_INFERENCE.value,
                    ReadingSkill.INFERENCE.value,
                    ReadingSkill.CONTEXT_INFERENCE.value,
                ],
            }[request.mode]
            slots: list[_QuestionSlot] = []
            for index, difficulty in enumerate(difficulties, 1):
                global_index = request.question_offset + index
                target = (request.reading_slot_targets[global_index - 1]
                          if request.reading_slot_targets else None)
                if target is not None:
                    difficulty = target.difficulty
                band = (target.complexity_band if target is not None
                        else self._band_for(request, difficulty))
                passage_id = "p1" if global_index <= 3 else "p2"
                skill_tag = calibrated_reading_skill(
                    mode=request.mode,
                    band=band,
                    planned_skill=skill_cycle[global_index - 1],
                )
                if target is not None and (target.global_order != global_index
                                           or target.skill_tag != skill_tag
                                           or target.complexity_band != self._band_for(request, difficulty)):
                    raise ValueError("Reading slot target changed server curriculum")
                planned_passage = passages.get(passage_id)
                plans = planned_passage.plans if isinstance(planned_passage, _PlannedPassage) else ()
                slots.append(
                    _QuestionSlot(
                        order=index,
                        difficulty=difficulty,
                        complexity_band=band,
                        skill_tag=skill_tag,
                        passage_id=passage_id,
                        passage_text=(planned_passage.text if isinstance(planned_passage, _PlannedPassage)
                                      else planned_passage),
                        reading_plan=next((plan for plan in plans if plan.global_order == global_index), None),
                        same_passage_plans=plans,
                        reading_mode=request.mode,
                    )
                )
            return slots

        skill_cycle = {
            VocabularyMode.USAGE_DISTINCTION.value: [
                VocabularySkill.DISTINCTION.value,
                VocabularySkill.COLLOCATION.value,
                VocabularySkill.REGISTER.value,
                VocabularySkill.CONTEXT_USAGE.value,
            ],
            VocabularyMode.COMPOSITION.value: [
                VocabularySkill.COMPOSITION.value,
                VocabularySkill.COLLOCATION.value,
                VocabularySkill.COMPOSITION.value,
                VocabularySkill.CONTEXT_USAGE.value,
            ],
        }.get(request.mode)
        ordering_orders = {1, 3, 6, 8} if request.mode == VocabularyMode.COMPOSITION.value else set()
        eligible_review_targets = self._eligible_review_targets(request)
        review_targets = list(eligible_review_targets[: request.review_question_count])
        focus_keywords = self._distinct_focus_keywords(request.selected_keywords)
        free_slot_occurrence = sum(
            not question.review_target for question in request.previous_questions
        )
        meaning_relation_band_occurrences: dict[int, int] = {}
        if request.mode in {
            VocabularyMode.MEANING_RELATION.value,
            VocabularyMode.CONTEXTUAL_CHOICE.value,
        }:
            for question in request.previous_questions:
                band = question.complexity_band
                meaning_relation_band_occurrences[band] = (
                    meaning_relation_band_occurrences.get(band, 0) + 1
                )
        contextual_choice_skill_counts = {
            skill: 0 for skill in _CONTEXTUAL_CHOICE_SKILL_ORDER
        }
        if request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
            for question in request.previous_questions:
                if question.skill_tag in contextual_choice_skill_counts:
                    contextual_choice_skill_counts[question.skill_tag] += 1
        slots: list[_QuestionSlot] = []
        for index, difficulty in enumerate(difficulties, 1):
            global_index = request.question_offset + index
            band = self._band_for(request, difficulty)
            review = review_targets[index - 1] if index <= len(review_targets) else None
            free_target_focus = None
            scenario_family = None
            if (
                request.mode
                in {
                    VocabularyMode.MEANING_RELATION.value,
                    VocabularyMode.CONTEXTUAL_CHOICE.value,
                }
                and review is None
            ):
                if focus_keywords:
                    free_target_focus = focus_keywords[
                        free_slot_occurrence % len(focus_keywords)
                    ]
                if request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
                    scenario_family = contextual_choice_scenario_family(
                        free_slot_occurrence
                    )
                free_slot_occurrence += 1
            if request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
                skill_tag = (
                    review.preferred_skill or VocabularySkill.MEANING.value
                    if review is not None
                    else self._contextual_choice_balanced_skill(
                        global_index,
                        contextual_choice_skill_counts,
                    )
                )
                contextual_choice_skill_counts[skill_tag] += 1
            elif request.mode == VocabularyMode.MEANING_RELATION.value:
                band_occurrence = meaning_relation_band_occurrences.get(band, 0) + 1
                meaning_relation_band_occurrences[band] = band_occurrence
                skill_tag = meaning_relation_skill_for_slot(
                    band=band,
                    band_occurrence=band_occurrence,
                )
            else:
                if skill_cycle is None:
                    raise ValueError(f"Unsupported Vocabulary mode: {request.mode}")
                skill_tag = skill_cycle[(global_index - 1) % len(skill_cycle)]
            slots.append(
                _QuestionSlot(
                    order=index,
                    difficulty=difficulty,
                    complexity_band=band,
                    skill_tag=skill_tag,
                    question_type=(
                        PracticeQuestionType.ORDERING
                        if global_index in ordering_orders
                        else PracticeQuestionType.SINGLE_CHOICE
                    ),
                    review_canonical_key=review.canonical_key if review else None,
                    review_expression=review.expression if review else None,
                    previous_question_types=tuple(
                        item.value for item in (review.previous_question_types if review else [])
                    ),
                    usage_intent=(
                        self._usage_intent_for_skill(skill_tag)
                        if request.mode == VocabularyMode.USAGE_DISTINCTION.value
                        else None
                    ),
                    vocabulary_mode=request.mode,
                    free_target_focus=free_target_focus,
                    scenario_family=scenario_family,
                )
            )
        return slots

    @classmethod
    def _distinct_focus_keywords(cls, selected_keywords: list[str]) -> list[str]:
        focuses: list[str] = []
        seen: set[str] = set()
        for raw_keyword in selected_keywords:
            keyword = raw_keyword.strip()
            normalized = cls._normalize_for_leak_check(keyword)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            focuses.append(keyword)
        return focuses

    @staticmethod
    def _contextual_choice_balanced_skill(
        global_index: int,
        skill_counts: dict[str, int],
    ) -> str:
        minimum = min(skill_counts.values())
        planned = contextual_choice_skill_for_order(global_index)
        start = _CONTEXTUAL_CHOICE_SKILL_ORDER.index(planned)
        tie_break = (
            _CONTEXTUAL_CHOICE_SKILL_ORDER[start:]
            + _CONTEXTUAL_CHOICE_SKILL_ORDER[:start]
        )
        return next(skill for skill in tie_break if skill_counts[skill] == minimum)

    @staticmethod
    def _difficulty_plan(request: PracticeGenerationRequest) -> list[PracticeDifficulty]:
        remaining = {
            PracticeDifficulty.EASIER: request.easier_count,
            PracticeDifficulty.CURRENT: request.current_count,
            PracticeDifficulty.CHALLENGE: request.challenge_count,
        }
        # CURRENT first keeps the set anchored at the learner estimate; easier/challenge are
        # distributed around it instead of relying on the model to invent the mix.
        pattern = [
            PracticeDifficulty.CURRENT,
            PracticeDifficulty.EASIER,
            PracticeDifficulty.CURRENT,
            PracticeDifficulty.CHALLENGE,
        ]
        result: list[PracticeDifficulty] = []
        while len(result) < request.question_count:
            progressed = False
            for difficulty in pattern:
                if remaining[difficulty] <= 0:
                    continue
                result.append(difficulty)
                remaining[difficulty] -= 1
                progressed = True
                if len(result) == request.question_count:
                    break
            if not progressed:
                break
        if len(result) != request.question_count:
            raise ValueError("failed to build deterministic difficulty plan")
        return result

    @staticmethod
    def _band_for(request: PracticeGenerationRequest, difficulty: PracticeDifficulty) -> int:
        if difficulty == PracticeDifficulty.EASIER:
            return max(1, request.complexity_band - 1)
        if difficulty == PracticeDifficulty.CHALLENGE:
            return min(5, request.complexity_band + 1)
        return request.complexity_band

    async def _generate_verified_questions(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
        *,
        contextual_choice_telemetry: _ContextualChoiceTelemetry | None = None,
        contextual_choice_plan: PersonalizedVocabularyPlan | None = None,
    ) -> list[PracticeGeneratedQuestion]:
        if request.domain == PracticeDomain.READING:
            return await self._generate_verified_reading_questions(request, slots)

        if request.domain == PracticeDomain.VOCABULARY:
            if request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
                if contextual_choice_telemetry is None:
                    raise ValueError("CONTEXTUAL_CHOICE requires generation telemetry")
                if contextual_choice_plan is None:
                    raise ValueError("CONTEXTUAL_CHOICE requires a daily plan")
                return await self._generate_verified_contextual_choice_questions(
                    request,
                    slots,
                    contextual_choice_telemetry,
                    contextual_choice_plan,
                )
            if request.mode == VocabularyMode.USAGE_DISTINCTION.value:
                return await self._generate_verified_usage_distinction_questions(request, slots)
            if request.mode == VocabularyMode.COMPOSITION.value:
                return await self._generate_verified_composition_questions(request, slots)

        slot_by_order = {slot.order: slot for slot in slots}
        attempts = {slot.order: 0 for slot in slots}
        accepted: dict[int, PracticeGeneratedQuestion] = {}
        semantically_verified: set[int] = set()
        retry_feedback: dict[int, str] = {}
        retry_target_expressions: dict[int, str] = {}
        retry_excluded_canonical_keys: dict[int, set[str]] = {}
        retry_excluded_target_expressions: dict[int, set[str]] = {}
        attempt_limit = self._candidate_attempt_limit(request)
        batch_size = self._candidate_batch_size(request)

        while len(semantically_verified) < len(slots):
            pending_generation = [
                order
                for order in sorted(slot_by_order)
                if order not in accepted and attempts[order] < attempt_limit
            ]
            if pending_generation:
                for start in range(0, len(pending_generation), batch_size):
                    batch_orders = pending_generation[start : start + batch_size]
                    batch_slots = [slot_by_order[order] for order in batch_orders]
                    for order in batch_orders:
                        attempts[order] += 1
                    candidate_failures = await self._generate_candidate_batch(
                        request,
                        batch_slots,
                        accepted,
                        retry_feedback=retry_feedback,
                        retry_target_expressions=retry_target_expressions,
                        retry_excluded_canonical_keys=retry_excluded_canonical_keys,
                        retry_excluded_target_expressions=(
                            retry_excluded_target_expressions
                        ),
                    )
                    for order, reason in candidate_failures.items():
                        retry_feedback[order] = reason
                    for order in batch_orders:
                        if order in accepted:
                            retry_feedback.pop(order, None)

            exhausted = [
                order
                for order in slot_by_order
                if order not in accepted and attempts[order] >= attempt_limit
            ]
            if exhausted:
                raise ValueError(f"candidate generation exhausted orders={sorted(exhausted)}")

            unverified = [
                accepted[order]
                for order in sorted(accepted)
                if order not in semantically_verified
            ]
            if not unverified:
                continue

            # Nano is deliberately a soft prescreen. It is useful for cheap diagnostics, but it
            # must never be the authority that burns a scarce candidate attempt. Deterministic
            # checks already reject direct leakage and Mini remains the independent semantic gate.
            prescreen_flags = await self._usage_prescreen_failures(request, unverified)
            for question in unverified:
                failure = prescreen_flags.get(question.order)
                if failure is None:
                    continue
                logger.warning(
                    "Reading/Vocabulary Nano prescreen flagged candidate; deferring to Mini. "
                    "request_id=%s order=%d attempt=%d/%d reason=%s",
                    request.request_id,
                    question.order,
                    attempts[question.order],
                    attempt_limit,
                    failure[:300],
                )

            semantic_failures = await self._semantic_failures(request, unverified)
            for question in unverified:
                if question.question_type == PracticeQuestionType.ORDERING:
                    semantically_verified.add(question.order)
                    retry_feedback.pop(question.order, None)
                    continue
                failure = semantic_failures.get(question.order)
                if failure is None:
                    semantically_verified.add(question.order)
                    retry_feedback.pop(question.order, None)
                    retry_target_expressions.pop(question.order, None)
                    retry_excluded_canonical_keys.pop(question.order, None)
                    retry_excluded_target_expressions.pop(question.order, None)
                    continue
                logger.warning(
                    "Reading/Vocabulary semantic candidate rejected. request_id=%s order=%d "
                    "attempt=%d/%d reason=%s",
                    request.request_id,
                    question.order,
                    attempts[question.order],
                    attempt_limit,
                    failure[:300],
                )
                slot = slot_by_order[question.order]
                preserve_target = slot.target_as_answer
                if preserve_target and question.target_expression:
                    retry_target_expressions[question.order] = (
                        question.target_expression
                    )
                else:
                    retry_target_expressions.pop(question.order, None)
                accepted.pop(question.order, None)
                semantically_verified.discard(question.order)
                retry_feedback[question.order] = self._meaning_relation_retry_feedback(
                    failure, slot
                )

            semantic_exhausted = [
                order
                for order in slot_by_order
                if order not in semantically_verified
                and order not in accepted
                and attempts[order] >= attempt_limit
            ]
            if semantic_exhausted:
                raise ValueError(f"semantic verification exhausted orders={sorted(semantic_exhausted)}")

        return [accepted[order] for order in sorted(accepted)]

    async def _resolve_contextual_choice_plan(
        self,
        request: PracticeGenerationRequest,
        telemetry: _ContextualChoiceTelemetry,
    ) -> PersonalizedVocabularyPlan:
        if request.vocabulary_plan is not None:
            try:
                self._validate_contextual_choice_plan_authority(
                    request,
                    request.vocabulary_plan,
                )
            except ValueError as exc:
                raise _ContextualChoiceContentRejected(
                    str(exc), stage="PLAN_AUTHORITY"
                ) from exc
            telemetry.plan_reused = True
            logger.info(
                "CONTEXTUAL_CHOICE V3 persisted plan reused. request_id=%s plan_reused=true",
                request.request_id,
            )
            return request.vocabulary_plan
        if request.previous_questions:
            raise _ContextualChoiceContentRejected(
                "CONTEXTUAL_CHOICE progressive request is missing persisted vocabularyPlan",
                stage="PLAN_AUTHORITY",
            )

        daily_request = self._contextual_choice_daily_request(request)
        daily_slots = self._build_slots(daily_request, {})
        slot_by_order = {slot.order: slot for slot in daily_slots}
        slot_payloads = [
            self._contextual_choice_plan_slot_payload(slot)
            for slot in daily_slots
        ]
        raw = await self._contextual_choice_provider_call(
            request,
            telemetry,
            counter="plan_generation_calls",
            stage="plan_generation",
            type_name=self.TYPE_NAME,
            data=build_contextual_choice_plan_generation_prompt(
                request,
                slots=slot_payloads,
            ),
            schema=_CONTEXTUAL_CHOICE_PLAN_GENERATION_SCHEMA,
        )
        try:
            payload = _ContextualChoicePlanGenerationPayload.model_validate(raw)
        except ValidationError as exc:
            self._log_contextual_choice_lexical_rejection(
                request, 0, "structural_initial", ["FINAL_PLAN_AUTHORITY"]
            )
            raise _ContextualChoiceContentRejected(
                "CONTEXTUAL_CHOICE plan response is structurally invalid",
                stage="PLAN_STRUCTURE",
            ) from exc
        raw_items = list(payload.items)
        if any(type(item.get("globalOrder")) is not int
               or item["globalOrder"] not in range(1, 11) for item in raw_items):
            self._log_contextual_choice_lexical_rejection(
                request, 0, "structural_initial", ["FINAL_PLAN_AUTHORITY"]
            )
            raise _ContextualChoiceContentRejected(
                "CONTEXTUAL_CHOICE plan contains an unknown global order",
                stage="PLAN_STRUCTURE",
            )
        normalized, deterministic_failures = self._normalize_contextual_choice_plan_items(
            request,
            raw_items,
            slot_by_order,
        )
        failures = deterministic_failures

        if failures:
            for order, reason in sorted(failures.items()):
                self._log_contextual_choice_lexical_rejection(
                    request, order, "structural_initial",
                    [self._contextual_choice_structural_reason_code(reason)],
                )
            repair_slots = [
                self._contextual_choice_plan_slot_payload(slot_by_order[order])
                for order in sorted(failures)
                if order in slot_by_order
            ]
            if len(repair_slots) != len(failures):
                raise _ContextualChoiceContentRejected(
                    "CONTEXTUAL_CHOICE plan contains unknown orders",
                    stage="PLAN_STRUCTURE",
                )
            repair_raw = await self._contextual_choice_provider_call(
                request,
                telemetry,
                counter="plan_repair_calls",
                stage="plan_repair",
                type_name=self.TYPE_NAME,
                data=build_contextual_choice_plan_generation_prompt(
                    request,
                    slots=repair_slots,
                    existing_plan_items=[
                        item.model_dump(mode="json", by_alias=True)
                        for order, item in sorted(normalized.items())
                        if order not in failures
                    ],
                    repair_reasons=failures,
                ),
                schema=_CONTEXTUAL_CHOICE_PLAN_GENERATION_SCHEMA,
            )
            try:
                repaired_payload = _ContextualChoicePlanGenerationPayload.model_validate(
                    repair_raw
                )
            except ValidationError as exc:
                raise _ContextualChoiceContentRejected(
                    "CONTEXTUAL_CHOICE structural repair response is invalid",
                    stage="PLAN_STRUCTURE",
                ) from exc
            if {item.get("globalOrder") for item in repaired_payload.items} != set(failures) or len(repaired_payload.items) != len(failures):
                raise _ContextualChoiceContentRejected(
                    "CONTEXTUAL_CHOICE structural repair changed non-invalid plan orders",
                    stage="PLAN_STRUCTURE",
                )
            raw_by_order = {
                item.get("globalOrder"): item
                for item in raw_items
                if isinstance(item.get("globalOrder"), int)
            }
            for item in repaired_payload.items:
                order = item.get("globalOrder")
                if isinstance(order, int):
                    raw_by_order[order] = item
            normalized, repair_deterministic_failures = (
                self._normalize_contextual_choice_plan_items(
                    request,
                    list(raw_by_order.values()),
                    slot_by_order,
                )
            )
            if repair_deterministic_failures:
                for order, reason in sorted(repair_deterministic_failures.items()):
                    self._log_contextual_choice_lexical_rejection(
                        request, order, "structural_repair",
                        [self._contextual_choice_structural_reason_code(reason)],
                    )
                raise _ContextualChoiceContentRejected(
                    "CONTEXTUAL_CHOICE plan repair failed deterministic validation "
                    f"orders={sorted(repair_deterministic_failures)}",
                    stage="PLAN_STRUCTURE",
                )

        if set(normalized) != set(range(1, 11)):
            raise _ContextualChoiceContentRejected(
                "CONTEXTUAL_CHOICE plan must contain global orders 1..10",
                stage="PLAN_STRUCTURE",
            )
        plan = PersonalizedVocabularyPlan(
            version=CONTEXTUAL_CHOICE_PLAN_VERSION,
            items=[normalized[order] for order in range(1, 11)],
        )
        try:
            self._validate_contextual_choice_plan_authority(request, plan)
        except ValueError as exc:
            raise _ContextualChoiceContentRejected(
                str(exc), stage="PLAN_STRUCTURE"
            ) from exc
        return plan

    def _contextual_choice_daily_request(
        self,
        request: PracticeGenerationRequest,
    ) -> PracticeGenerationRequest:
        eligible_reviews = self._eligible_review_targets(request, log_rejections=False)
        review_count = min(len(eligible_reviews), 2)
        return request.model_copy(
            update={
                "question_count": 10,
                "easier_count": 2,
                "current_count": 6,
                "challenge_count": 2,
                "review_question_count": review_count,
                "previous_questions": [],
                "vocabulary_plan": None,
            }
        )

    @staticmethod
    def _contextual_choice_plan_slot_payload(slot: _QuestionSlot) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "globalOrder": slot.order,
            "reviewTarget": slot.review_target,
            "skillTag": slot.skill_tag,
            "difficulty": slot.difficulty.value,
            "complexityBand": slot.complexity_band,
            "scenarioFamily": slot.scenario_family,
        }
        if slot.review_target:
            payload["boundReviewTarget"] = {
                "targetExpression": slot.review_expression,
                "canonicalKey": slot.review_canonical_key,
            }
        return payload

    def _normalize_contextual_choice_plan_items(
        self,
        request: PracticeGenerationRequest,
        raw_items: list[dict[str, Any]],
        slot_by_order: dict[int, _QuestionSlot],
    ) -> tuple[dict[int, VocabularyPlanItem], dict[int, str]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for raw_item in raw_items:
            order = raw_item.get("globalOrder")
            if isinstance(order, int):
                grouped.setdefault(order, []).append(raw_item)

        normalized: dict[int, VocabularyPlanItem] = {}
        failures: dict[int, str] = {}
        review_keys = {
            target.canonical_key for target in request.review_targets
        }
        review_targets = {
            self._normalize_for_leak_check(target.expression)
            for target in request.review_targets
        }
        signal_values = {
            VocabularyPlanAnchorType.SELECTED_KEYWORD: set(request.selected_keywords),
            VocabularyPlanAnchorType.WEAK_SIGNAL: set(request.weak_signals),
            VocabularyPlanAnchorType.RECENT_MISTAKE: set(request.recent_mistakes),
            VocabularyPlanAnchorType.LEARNING_PROFILE: {request.learning_language},
        }
        for order, slot in sorted(slot_by_order.items()):
            candidates = grouped.get(order, [])
            if len(candidates) != 1:
                failures[order] = "plan item coverage must be exactly one"
                continue
            item = candidates[0]
            try:
                if slot.review_target:
                    target_expression = (slot.review_expression or "").strip()
                    canonical_key = (slot.review_canonical_key or "").strip()
                    anchor_type = None
                    anchor_value = None
                else:
                    raw_target = item.get("targetExpression")
                    if not isinstance(raw_target, str) or not raw_target.strip():
                        raise ValueError("new plan item requires targetExpression")
                    target_expression = raw_target.strip()
                    canonical_key = self._canonical_key_for_expression(
                        target_expression
                    )
                    anchor_type = VocabularyPlanAnchorType(item.get("anchorType"))
                    raw_anchor = item.get("anchorValue")
                    if not isinstance(raw_anchor, str) or not raw_anchor.strip():
                        raise ValueError("new plan item requires anchorValue")
                    anchor_value = raw_anchor.strip()
                    if anchor_value not in signal_values[anchor_type]:
                        raise ValueError(
                            "new plan anchorValue is not present in the request signal"
                        )
                    if (
                        anchor_type == VocabularyPlanAnchorType.LEARNING_PROFILE
                        and any(
                            (
                                request.selected_keywords,
                                request.weak_signals,
                                request.recent_mistakes,
                            )
                        )
                    ):
                        raise ValueError(
                            "LEARNING_PROFILE anchor is only allowed without learner signals"
                        )
                    if (
                        canonical_key in review_keys
                        or self._normalize_for_leak_check(target_expression)
                        in review_targets
                    ):
                        raise ValueError("new plan target reuses a review identity")

                self._assert_vocabulary_surface_language(
                    request.learning_language,
                    target_expression,
                    reason="CONTEXTUAL_CHOICE plan target is not in learningLanguage",
                )
                raw_distractors = item.get("distractors")
                if not isinstance(raw_distractors, list) or len(raw_distractors) != 3:
                    raise ValueError("plan item requires exactly three distractors")
                distractors: list[str] = []
                seen = {self._normalize_for_leak_check(target_expression)}
                for raw_distractor in raw_distractors:
                    if not isinstance(raw_distractor, str) or not raw_distractor.strip():
                        raise ValueError("plan distractor must be a non-empty string")
                    distractor = raw_distractor.strip()
                    self._assert_vocabulary_surface_language(
                        request.learning_language,
                        distractor,
                        reason="CONTEXTUAL_CHOICE plan distractor is not in learningLanguage",
                    )
                    normalized_distractor = self._normalize_for_leak_check(distractor)
                    if normalized_distractor in seen:
                        raise ValueError(
                            "plan distractors must be distinct from target and each other"
                        )
                    seen.add(normalized_distractor)
                    distractors.append(distractor)
                normalized[order] = VocabularyPlanItem(
                    global_order=order,
                    review_target=slot.review_target,
                    target_expression=target_expression,
                    canonical_key=canonical_key,
                    distractors=distractors,
                    skill_tag=cast(
                        Literal[
                            "MEANING",
                            "COLLOCATION",
                            "NUANCE",
                            "REGISTER",
                            "PRAGMATIC_FIT",
                        ],
                        slot.skill_tag,
                    ),
                    difficulty=slot.difficulty,
                    complexity_band=slot.complexity_band,
                    scenario_family=slot.scenario_family,
                    anchor_type=anchor_type,
                    anchor_value=anchor_value,
                )
            except (TypeError, ValueError, ValidationError) as exc:
                failures[order] = str(exc)[:300]

        target_orders: dict[str, list[int]] = {}
        canonical_orders: dict[str, list[int]] = {}
        for order, item in normalized.items():
            target_orders.setdefault(
                self._normalize_for_leak_check(item.target_expression), []
            ).append(order)
            canonical_orders.setdefault(item.canonical_key, []).append(order)
        for duplicate_orders in [*target_orders.values(), *canonical_orders.values()]:
            if len(duplicate_orders) > 1:
                for order in duplicate_orders:
                    failures[order] = "plan target/canonical duplicate"
                    normalized.pop(order, None)
        if request.selected_keywords and not any(
            item.anchor_type == VocabularyPlanAnchorType.SELECTED_KEYWORD
            for item in normalized.values()
            if not item.review_target
        ):
            first_new_order = next(
                (
                    order
                    for order, slot in sorted(slot_by_order.items())
                    if not slot.review_target
                ),
                None,
            )
            if first_new_order is not None:
                failures[first_new_order] = (
                    "at least one New plan item must be anchored to selectedKeywords"
                )
                normalized.pop(first_new_order, None)
        return normalized, failures

    @staticmethod
    def _contextual_choice_structural_reason_code(reason: str) -> str:
        lowered = reason.lower()
        if "review" in lowered:
            return "REVIEW_AUTHORITY"
        if "anchor" in lowered or "selectedkeyword" in lowered:
            return "ANCHOR_AUTHORITY"
        if "duplicate" in lowered or "distinct" in lowered:
            return "STRUCTURAL_DUPLICATE"
        if "distractor" in lowered:
            return "STRUCTURAL_DISTRACTOR"
        if "target" in lowered or "canonical" in lowered:
            return "STRUCTURAL_TARGET"
        return "FINAL_PLAN_AUTHORITY"

    @staticmethod
    def _log_contextual_choice_lexical_rejection(
        request: PracticeGenerationRequest,
        order: int,
        phase: str,
        reason_codes: list[str],
        invalid_distractor_indexes: list[int] | None = None,
    ) -> None:
        logger.info(
            "CONTEXTUAL_CHOICE V3.3 lexical validation rejected. "
            "request_id=%s global_order=%d phase=%s reason_codes=%s "
            "invalid_distractor_indexes=%s",
            request.request_id, order, phase, sorted(set(reason_codes)),
            sorted(set(invalid_distractor_indexes or [])),
        )

    async def _verify_contextual_choice_plan_item(
        self,
        request: PracticeGenerationRequest,
        telemetry: _ContextualChoiceTelemetry,
        item: VocabularyPlanItem,
        accepted: dict[int, PracticeGeneratedQuestion],
    ) -> _ContextualChoiceLexicalAssessment:
        demand = vocabulary_difficulty_recipe(
            mode=VocabularyMode.CONTEXTUAL_CHOICE.value,
            band=item.complexity_band,
            skill_tag=item.skill_tag,
            question_type=PracticeQuestionType.SINGLE_CHOICE.value,
        ).demand
        if not isinstance(demand, ContextualChoiceDemand):
            raise ValueError("CONTEXTUAL_CHOICE requires contextual-choice demand")
        previous = [*request.previous_questions, *accepted.values()]
        raw = await self._contextual_choice_provider_call(
            request,
            telemetry,
            counter="lexical_validation_calls",
            stage="lexical_validation",
            type_name=self.CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME,
            data=build_contextual_choice_plan_verification_prompt(
                request,
                plan_item=item.model_dump(mode="json", by_alias=True),
                structural_demand=demand.generation_payload(),
                previous_lexical_identities=[
                    {
                        "canonicalKey": question.canonical_key,
                        "targetExpression": question.target_expression,
                        "skillTag": question.skill_tag,
                    }
                    for question in previous
                ],
            ),
            schema=_contextual_choice_plan_verification_schema(
                demand.decisive_dimensions, global_order=item.global_order,
            ),
        )
        try:
            verdict = _parse_contextual_choice_plan_verdict(
                raw, expected_order=item.global_order,
            )
        except _ContextualChoiceLexicalVerifierContractError as exc:
            logger.warning(
                "CONTEXTUAL_CHOICE lexical verifier contract failed. "
                "request_id=%s reason_code=%s expected_order=%d expected_ids=%s "
                "received_coverage=%s",
                request.request_id, exc.reason_code, item.global_order,
                ["TARGET", "D0", "D1", "D2"],
                _contextual_choice_verifier_coverage_summary(raw),
            )
            raise
        distractors = verdict.distractors.ordered()
        # The verifier's reasonCode/reason are diagnostic only. All rejection and
        # repair authority below is derived from the structured boolean verdicts.
        codes: list[str] = []
        invalid_indexes: set[int] = set()
        if not verdict.targetExpressionWellFormed:
            codes.append("NATURAL_TARGET")
        if not verdict.learningValue:
            codes.append("LEARNING_VALUE")
        if not verdict.sameSurfaceCategory:
            codes.append("SAME_SURFACE_CATEGORY")
        required_decisive_dimensions = set(demand.decisive_dimensions)
        covered_decisive_dimensions = set(verdict.coveredDecisiveDimensions)
        bundle_dimension_match = bool(
            covered_decisive_dimensions & required_decisive_dimensions
        ) and covered_decisive_dimensions <= required_decisive_dimensions
        if not verdict.skillContrastSupported or not bundle_dimension_match:
            codes.append("SKILL_FIT")
        if verdict.definitionOnly:
            codes.append("DEFINITION_ONLY")
        if verdict.lexicalConceptRepeated and not item.review_target:
            codes.append("LEXICAL_REPEAT")
        if verdict.rareOrTrivia:
            codes.append("RARE_OR_TRIVIA")
        required_close_count = contextual_choice_required_close_distractors(
            demand.minimum_close_distractors,
            review_target=item.review_target,
        )
        repair_evidence = tuple(
            _ContextualChoiceDistractorRepairEvidence(
                index=distractor.index,
                expression_well_formed=distractor.expressionWellFormed,
                same_surface_category=distractor.sameSurfaceCategory,
                bundle_relevant=distractor.bundleRelevant,
                close_competitor=distractor.closeCompetitor,
                malformed_by_grammar=distractor.malformedByGrammar,
                skill_contrast_relevant=distractor.skillContrastRelevant,
                covered_decisive_dimensions=(
                    distractor.coveredDecisiveDimensions
                ),
            )
            for distractor in sorted(
                distractors,
                key=lambda candidate: candidate.index,
            )
        )
        close_competitor_indexes: set[int] = set()
        for distractor in distractors:
            index = distractor.index
            base_valid = True
            if not distractor.expressionWellFormed:
                codes.append("DISTRACTOR_UNNATURAL")
                invalid_indexes.add(index)
                base_valid = False
            if not distractor.sameSurfaceCategory:
                codes.append("DISTRACTOR_SURFACE_MISMATCH")
                invalid_indexes.add(index)
                base_valid = False
            if distractor.malformedByGrammar:
                codes.append("DISTRACTOR_GRAMMAR_ONLY")
                invalid_indexes.add(index)
                base_valid = False
            if not distractor.bundleRelevant:
                codes.append("DISTRACTOR_NOT_PLAUSIBLE")
                invalid_indexes.add(index)
                base_valid = False
            distractor_dimensions = set(
                distractor.coveredDecisiveDimensions
            )
            distractor_dimension_match = bool(
                distractor_dimensions & required_decisive_dimensions
            ) and distractor_dimensions <= required_decisive_dimensions
            if (
                not distractor.skillContrastRelevant
                or not distractor_dimension_match
            ):
                codes.append("DISTRACTOR_SKILL_MISMATCH")
                invalid_indexes.add(index)
                base_valid = False
            if base_valid and distractor.closeCompetitor:
                close_competitor_indexes.add(index)
        if len(close_competitor_indexes) < required_close_count:
            codes.append("INSUFFICIENT_CLOSE_DISTRACTORS")
            projected_close_count = len(close_competitor_indexes) + len(
                invalid_indexes
            )
            additional_indexes_needed = max(
                0,
                required_close_count - projected_close_count,
            )
            nonclose_valid_indexes = [
                distractor.index
                for distractor in distractors
                if distractor.index not in invalid_indexes
                and distractor.index not in close_competitor_indexes
            ]
            invalid_indexes.update(
                nonclose_valid_indexes[:additional_indexes_needed]
            )
        additional_close_needed = max(
            0,
            required_close_count - len(close_competitor_indexes),
        )
        close_required_replacement_indexes = tuple(
            sorted(invalid_indexes)[:additional_close_needed]
        )
        return _ContextualChoiceLexicalAssessment(
            reason_codes=tuple(dict.fromkeys(codes)),
            invalid_distractor_indexes=tuple(sorted(invalid_indexes)),
            preserve_new_target=(
                verdict.targetExpressionWellFormed and verdict.learningValue
                and verdict.targetViableWithDifferentDistractors
            ),
            distractor_repair_evidence=repair_evidence,
            required_close_distractors=required_close_count,
            current_valid_close_distractor_count=len(close_competitor_indexes),
            additional_close_distractors_needed=additional_close_needed,
            close_required_replacement_indexes=close_required_replacement_indexes,
            skill=demand.skill,
            decisive_dimensions=demand.decisive_dimensions,
        )

    def _validate_contextual_choice_plan_authority(
        self,
        request: PracticeGenerationRequest,
        plan: PersonalizedVocabularyPlan,
    ) -> None:
        if plan.version != CONTEXTUAL_CHOICE_PLAN_VERSION:
            raise ValueError("CONTEXTUAL_CHOICE vocabularyPlan version mismatch")
        review_by_key = {
            target.canonical_key: target for target in request.review_targets
        }
        review_orders = [
            item.global_order for item in plan.items if item.review_target
        ]
        expected_review_count = (
            len(review_orders)
            if request.previous_questions
            else min(
                len(self._eligible_review_targets(request, log_rejections=False)),
                2,
            )
        )
        if review_orders != list(range(1, expected_review_count + 1)):
            raise ValueError("vocabularyPlan review slot authority mismatch")
        new_index = 0
        seen_targets: set[str] = set()
        seen_canonical: set[str] = set()
        for item in plan.items:
            expected_band = self._band_for(request, item.difficulty)
            if item.complexity_band != expected_band:
                raise ValueError("vocabularyPlan complexityBand authority mismatch")
            if item.canonical_key != self._canonical_key_for_expression(
                item.target_expression
            ) and not item.review_target:
                raise ValueError("vocabularyPlan canonicalKey normalization mismatch")
            normalized_target = self._normalize_for_leak_check(item.target_expression)
            if normalized_target in seen_targets or item.canonical_key in seen_canonical:
                raise ValueError("vocabularyPlan target/canonical duplicate")
            seen_targets.add(normalized_target)
            seen_canonical.add(item.canonical_key)
            if item.review_target:
                review = review_by_key.get(item.canonical_key)
                if review is None or review.expression != item.target_expression:
                    raise ValueError("vocabularyPlan review identity mismatch")
                if item.skill_tag != (review.preferred_skill or VocabularySkill.MEANING.value):
                    raise ValueError("vocabularyPlan review preferredSkill mismatch")
            else:
                expected_scenario = contextual_choice_scenario_family(new_index)
                new_index += 1
                if item.scenario_family != expected_scenario:
                    raise ValueError("vocabularyPlan scenarioFamily authority mismatch")
                if item.anchor_type is None or item.anchor_value is None:
                    raise ValueError("vocabularyPlan new item requires anchor metadata")
                signals = {
                    VocabularyPlanAnchorType.SELECTED_KEYWORD: request.selected_keywords,
                    VocabularyPlanAnchorType.WEAK_SIGNAL: request.weak_signals,
                    VocabularyPlanAnchorType.RECENT_MISTAKE: request.recent_mistakes,
                    VocabularyPlanAnchorType.LEARNING_PROFILE: [
                        request.learning_language
                    ],
                }
                if item.anchor_value not in signals[item.anchor_type]:
                    raise ValueError("vocabularyPlan anchor membership mismatch")
                if (
                    item.anchor_type == VocabularyPlanAnchorType.LEARNING_PROFILE
                    and any(
                        (
                            request.selected_keywords,
                            request.weak_signals,
                            request.recent_mistakes,
                        )
                    )
                ):
                    raise ValueError("vocabularyPlan profile fallback is not allowed")
            self._validate_contextual_choice_plan_distractors(request, item)
        for question in request.previous_questions:
            item = plan.items[question.order - 1]
            correct_key = contextual_choice_answer_key(question.order)
            expected_options = list(item.distractors)
            expected_options.insert(
                CONTEXTUAL_CHOICE_ANSWER_KEYS.index(correct_key),
                item.target_expression,
            )
            if (
                question.target_expression != item.target_expression
                or question.canonical_key != item.canonical_key
                or question.skill_tag != item.skill_tag
                or question.difficulty != item.difficulty
                or question.complexity_band != item.complexity_band
                or question.review_target != item.review_target
                or question.correct_answer != [correct_key]
                or [(option.key, option.text) for option in question.options]
                != list(zip(CONTEXTUAL_CHOICE_ANSWER_KEYS, expected_options, strict=True))
            ):
                raise ValueError("vocabularyPlan accepted prefix identity mismatch")

    def _validate_contextual_choice_plan_distractors(
        self,
        request: PracticeGenerationRequest,
        item: VocabularyPlanItem,
    ) -> None:
        self._assert_vocabulary_surface_language(
            request.learning_language,
            item.target_expression,
            reason="CONTEXTUAL_CHOICE plan target is not in learningLanguage",
        )
        seen = {self._normalize_for_leak_check(item.target_expression)}
        for distractor in item.distractors:
            self._assert_vocabulary_surface_language(
                request.learning_language,
                distractor,
                reason="CONTEXTUAL_CHOICE plan distractor is not in learningLanguage",
            )
            normalized = self._normalize_for_leak_check(distractor)
            if not normalized or normalized in seen:
                raise ValueError("vocabularyPlan distractor identity mismatch")
            seen.add(normalized)

    def _contextual_choice_slots_from_plan(
        self,
        request: PracticeGenerationRequest,
        plan: PersonalizedVocabularyPlan,
    ) -> list[_QuestionSlot]:
        by_order = {item.global_order: item for item in plan.items}
        review_by_key = {
            target.canonical_key: target for target in request.review_targets
        }
        slots: list[_QuestionSlot] = []
        for local_order in range(1, request.question_count + 1):
            global_order = request.question_offset + local_order
            item = by_order[global_order]
            review = review_by_key.get(item.canonical_key) if item.review_target else None
            slots.append(
                _QuestionSlot(
                    order=local_order,
                    difficulty=item.difficulty,
                    complexity_band=item.complexity_band,
                    skill_tag=item.skill_tag,
                    question_type=PracticeQuestionType.SINGLE_CHOICE,
                    review_canonical_key=item.canonical_key if item.review_target else None,
                    review_expression=item.target_expression if item.review_target else None,
                    previous_question_types=tuple(
                        question_type.value
                        for question_type in (review.previous_question_types if review else [])
                    ),
                    vocabulary_mode=VocabularyMode.CONTEXTUAL_CHOICE.value,
                    scenario_family=item.scenario_family,
                    vocabulary_plan_item=item,
                )
            )
        return slots

    async def _contextual_choice_provider_call(
        self,
        request: PracticeGenerationRequest,
        telemetry: _ContextualChoiceTelemetry,
        *,
        counter: str,
        stage: str,
        type_name: str,
        data: str,
        schema: dict[str, Any],
    ) -> Any:
        setattr(telemetry, counter, getattr(telemetry, counter) + 1)
        started = time.perf_counter()
        outcome = "FAILED"
        try:
            raw = await asyncio.wait_for(
                self.provider.call(type_name=type_name, data=data, schema=schema),
                timeout=self.timeout_seconds,
            )
            outcome = "SUCCEEDED"
            return raw
        except asyncio.CancelledError:
            outcome = "CANCELLED"
            raise
        finally:
            logger.info(
                "CONTEXTUAL_CHOICE V3 stage finished. request_id=%s stage=%s "
                "status=%s latency_ms=%d",
                request.request_id,
                stage,
                outcome,
                int((time.perf_counter() - started) * 1000),
            )

    async def _repair_contextual_choice_lexical_item(
        self,
        request: PracticeGenerationRequest,
        telemetry: _ContextualChoiceTelemetry,
        plan: PersonalizedVocabularyPlan,
        item: VocabularyPlanItem,
        lexical_assessment: _ContextualChoiceLexicalAssessment,
        reason_codes: list[str],
        *,
        invalid_distractor_indexes: list[int] | None = None,
        preserve_target: bool = False,
        repair_trigger: str = "LEXICAL_VALIDATION",
    ) -> VocabularyPlanItem:
        preserve_target = preserve_target or item.review_target
        requested_indexes = sorted(set(invalid_distractor_indexes or []))
        if any(index not in range(3) for index in requested_indexes):
            raise ValueError("lexical repair index is outside the current distractor bundle")
        if preserve_target and requested_indexes:
            repair_scope = "DISTRACTOR_INDEXES"
            allowed_indexes = requested_indexes
        elif preserve_target:
            repair_scope = "FULL_DISTRACTOR_SET"
            allowed_indexes = [0, 1, 2]
        else:
            repair_scope = "FULL_LEXICAL_BUNDLE"
            allowed_indexes = [0, 1, 2]
        evidence_by_index = {
            evidence.index: evidence
            for evidence in lexical_assessment.distractor_repair_evidence
        }
        required_dimensions = set(lexical_assessment.decisive_dimensions)
        # Only unchanged, still-valid competitors survive the effective repair
        # scope. A full bundle/set replacement cannot reuse their old close count.
        preserved_close_count = sum(
            1
            for index, evidence in evidence_by_index.items()
            if preserve_target
            and index not in allowed_indexes
            and evidence.expression_well_formed
            and evidence.same_surface_category
            and evidence.bundle_relevant
            and not evidence.malformed_by_grammar
            and evidence.skill_contrast_relevant
            and evidence.close_competitor
            and bool(set(evidence.covered_decisive_dimensions) & required_dimensions)
            and set(evidence.covered_decisive_dimensions) <= required_dimensions
        )
        additional_close_needed = max(
            0,
            lexical_assessment.required_close_distractors - preserved_close_count,
        )
        close_required_indexes = set(allowed_indexes[:additional_close_needed])
        forbidden_expressions = list(
            dict.fromkeys([item.target_expression, *item.distractors])
        )
        repair_evidence: dict[str, object] = {
            "forbiddenExpressions": forbidden_expressions,
            "distractorEvidence": {
                str(index): evidence_by_index[index].prompt_payload()
                for index in allowed_indexes
            },
            "requiredCloseDistractors": (
                lexical_assessment.required_close_distractors
            ),
            "currentValidCloseDistractorCount": preserved_close_count,
            "additionalCloseDistractorsNeeded": additional_close_needed,
            "repairRequirements": {
                str(index): {
                    "mustBeNonEmpty": True,
                    "mustUseLearningLanguage": True,
                    "mustDifferFromPrevious": True,
                    "mustDifferFromTarget": True,
                    "mustDifferFromPreservedDistractors": True,
                    "mustBeExpressionWellFormed": True,
                    "mustBeSameSurfaceCategory": True,
                    "mustBeBundleRelevant": True,
                    "mustBeSkillContrastRelevant": True,
                    "mustBeMalformedByGrammar": False,
                    "mustBeCloseCompetitor": index
                    in close_required_indexes,
                }
                for index in allowed_indexes
            },
            "skillDemand": {
                "skill": lexical_assessment.skill,
                "decisiveDimensions": list(
                    lexical_assessment.decisive_dimensions
                ),
            },
            "authority": "APPLICATION_DERIVED_FROM_STRUCTURED_VERDICT",
        }
        raw = await self._contextual_choice_provider_call(
            request,
            telemetry,
            counter="lexical_repair_calls",
            stage="lexical_repair",
            type_name=self.TYPE_NAME,
            data=build_contextual_choice_lexical_repair_prompt(
                request,
                plan_item=item.model_dump(mode="json", by_alias=True),
                reason_codes=reason_codes,
                invalid_distractor_indexes=allowed_indexes,
                preserve_target=preserve_target,
                repair_scope=repair_scope,
                repair_trigger=repair_trigger,
                repair_evidence=repair_evidence,
                other_canonical_keys=[
                    other.canonical_key for other in plan.items
                    if other.global_order != item.global_order
                ],
            ),
            schema=_contextual_choice_lexical_repair_schema(
                global_order=item.global_order,
                allowed_indexes=allowed_indexes,
            ),
        )
        try:
            try:
                payload = _ContextualChoiceLexicalRepairPayload.model_validate(raw)
            except (TypeError, ValidationError) as exc:
                raise _ContextualChoiceLexicalRepairContractError(
                    "LEXICAL_REPAIR_RESPONSE_SCHEMA_INVALID"
                ) from exc
            if payload.globalOrder != item.global_order:
                raise _ContextualChoiceLexicalRepairContractError(
                    "LEXICAL_REPAIR_ORDER_MISMATCH"
                )
            raw_target = payload.targetExpression
            if preserve_target:
                if raw_target is not None:
                    raise _ContextualChoiceLexicalRepairContractError(
                        "LEXICAL_REPAIR_TARGET_AUTHORITY_VIOLATION"
                    )
                target = item.target_expression
                canonical = item.canonical_key
            else:
                if not isinstance(raw_target, str) or not raw_target.strip():
                    raise _ContextualChoiceLexicalRepairContractError(
                        "LEXICAL_REPAIR_TARGET_AUTHORITY_VIOLATION"
                    )
                target = raw_target.strip()
                if self._normalize_contextual_choice_repair_identity(
                    target
                ) == self._normalize_contextual_choice_repair_identity(
                    item.target_expression
                ):
                    raise _ContextualChoiceLexicalRepairContractError(
                        "LEXICAL_REPAIR_TARGET_AUTHORITY_VIOLATION"
                    )
                try:
                    self._assert_vocabulary_surface_language(
                        request.learning_language,
                        target,
                        reason="lexical repair target is not in learningLanguage",
                    )
                except ValueError as exc:
                    raise _ContextualChoiceLexicalRepairContractError(
                        "LEXICAL_REPAIR_LANGUAGE_INVALID"
                    ) from exc
                canonical = self._canonical_key_for_expression(target)
            replacement_indexes = [
                replacement.index for replacement in payload.distractorReplacements
            ]
            if (
                len(replacement_indexes) != len(set(replacement_indexes))
                or sorted(replacement_indexes) != allowed_indexes
            ):
                raise _ContextualChoiceLexicalRepairContractError(
                    "LEXICAL_REPAIR_SCOPE_MISMATCH"
                )
            distractors = list(item.distractors)
            normalized_target = self._normalize_contextual_choice_repair_identity(
                target
            )
            previous_distractor_identities = {
                index: self._normalize_contextual_choice_repair_identity(distractor)
                for index, distractor in enumerate(item.distractors)
            }
            replacement_identities: set[str] = set()
            for replacement in payload.distractorReplacements:
                replacement_text = replacement.text.strip()
                if not replacement_text:
                    raise _ContextualChoiceLexicalRepairContractError(
                        "LEXICAL_REPAIR_EMPTY_DISTRACTOR",
                        index=replacement.index,
                    )
                replacement_identity = (
                    self._normalize_contextual_choice_repair_identity(replacement_text)
                )
                if replacement_identity == previous_distractor_identities[
                    replacement.index
                ]:
                    raise _ContextualChoiceLexicalRepairContractError(
                        "LEXICAL_REPAIR_UNCHANGED_DISTRACTOR",
                        index=replacement.index,
                    )
                if replacement_identity == normalized_target:
                    raise _ContextualChoiceLexicalRepairContractError(
                        "LEXICAL_REPAIR_DUPLICATE_TARGET",
                        index=replacement.index,
                    )
                if (
                    replacement_identity in {
                        identity
                        for index, identity in previous_distractor_identities.items()
                        if index != replacement.index
                    }
                    or replacement_identity in replacement_identities
                ):
                    raise _ContextualChoiceLexicalRepairContractError(
                        "LEXICAL_REPAIR_DUPLICATE_DISTRACTOR",
                        index=replacement.index,
                    )
                try:
                    self._assert_vocabulary_surface_language(
                        request.learning_language,
                        replacement_text,
                        reason="lexical repair distractor is not in learningLanguage",
                    )
                except ValueError as exc:
                    raise _ContextualChoiceLexicalRepairContractError(
                        "LEXICAL_REPAIR_LANGUAGE_INVALID",
                        index=replacement.index,
                    ) from exc
                replacement_identities.add(replacement_identity)
                distractors[replacement.index] = replacement_text
            try:
                revised = VocabularyPlanItem.model_validate({
                    **item.model_dump(mode="json", by_alias=True),
                    "targetExpression": target,
                    "canonicalKey": canonical,
                    "distractors": distractors,
                })
            except ValidationError as exc:
                raise _ContextualChoiceLexicalRepairContractError(
                    "LEXICAL_REPAIR_PLAN_AUTHORITY_VIOLATION"
                ) from exc
            revised_items = list(plan.items)
            revised_items[item.global_order - 1] = revised
            try:
                self._validate_contextual_choice_plan_authority(
                    request,
                    PersonalizedVocabularyPlan(version=plan.version, items=revised_items),
                )
            except (TypeError, ValueError, ValidationError) as exc:
                raise _ContextualChoiceLexicalRepairContractError(
                    "LEXICAL_REPAIR_PLAN_AUTHORITY_VIOLATION"
                ) from exc
            logger.info(
                "CONTEXTUAL_CHOICE V3.3 lexical repair applied. "
                "request_id=%s global_order=%d repair_scope=%s "
                "changed_distractor_indexes=%s target_changed=%s",
                request.request_id,
                item.global_order,
                repair_scope,
                allowed_indexes,
                target != item.target_expression,
            )
            return revised
        except _ContextualChoiceLexicalRepairContractError as exc:
            self._log_contextual_choice_lexical_rejection(
                request, item.global_order, "repair",
                [exc.reason_code],
                [] if exc.index is None else [exc.index],
            )
            raise _ContextualChoiceContentRejected(
                f"CONTEXTUAL_CHOICE lexical repair failed structural validation order={item.global_order}",
                stage="LEXICAL_VALIDATION",
            ) from exc

    async def _generate_verified_contextual_choice_questions(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
        telemetry: _ContextualChoiceTelemetry,
        plan: PersonalizedVocabularyPlan,
    ) -> list[PracticeGeneratedQuestion]:
        accepted: dict[int, PracticeGeneratedQuestion] = {}
        current_items = list(plan.items)
        for slot in sorted(slots, key=lambda item: item.order):
            question, validated_item = await self._generate_verified_contextual_choice_v3_slot(
                request,
                slot,
                accepted,
                telemetry,
                PersonalizedVocabularyPlan(version=plan.version, items=current_items),
            )
            accepted[slot.order] = question
            current_items[validated_item.global_order - 1] = validated_item
            slots[slot.order - 1] = replace(slot, vocabulary_plan_item=validated_item)
        return [accepted[order] for order in sorted(accepted)]

    async def _generate_verified_contextual_choice_v3_slot(
        self,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        accepted: dict[int, PracticeGeneratedQuestion],
        telemetry: _ContextualChoiceTelemetry,
        plan: PersonalizedVocabularyPlan,
    ) -> tuple[PracticeGeneratedQuestion, VocabularyPlanItem]:
        plan_item = slot.vocabulary_plan_item
        if plan_item is None:
            raise ValueError("CONTEXTUAL_CHOICE V3 slot requires a persisted plan item")
        lexical_assessment = await self._verify_contextual_choice_plan_item(
            request, telemetry, plan_item, accepted
        )
        lexical_repair_used = False
        if lexical_assessment.reason_codes:
            self._log_contextual_choice_lexical_rejection(
                request, plan_item.global_order, "initial",
                list(lexical_assessment.reason_codes),
                list(lexical_assessment.invalid_distractor_indexes),
            )
            plan_item = await self._repair_contextual_choice_lexical_item(
                request, telemetry, plan, plan_item,
                lexical_assessment,
                list(lexical_assessment.reason_codes),
                invalid_distractor_indexes=list(
                    lexical_assessment.invalid_distractor_indexes
                ),
                preserve_target=lexical_assessment.preserve_new_target,
            )
            lexical_repair_used = True
            lexical_assessment = await self._verify_contextual_choice_plan_item(
                request, telemetry, plan_item, accepted
            )
            if lexical_assessment.reason_codes:
                self._log_contextual_choice_lexical_rejection(
                    request, plan_item.global_order, "repair",
                    list(lexical_assessment.reason_codes),
                    list(lexical_assessment.invalid_distractor_indexes),
                )
                raise _ContextualChoiceContentRejected(
                    f"CONTEXTUAL_CHOICE lexical repair exhausted order={plan_item.global_order}",
                    stage="LEXICAL_VALIDATION",
                )
        slot = replace(slot, vocabulary_plan_item=plan_item)
        demand = vocabulary_difficulty_recipe(
            mode=VocabularyMode.CONTEXTUAL_CHOICE.value,
            band=plan_item.complexity_band,
            skill_tag=plan_item.skill_tag,
            question_type=PracticeQuestionType.SINGLE_CHOICE.value,
        ).demand
        if not isinstance(demand, ContextualChoiceDemand):
            raise ValueError("CONTEXTUAL_CHOICE requires contextual-choice demand")
        next_repair_class: _ContextFailureClass | None = None
        next_repair_codes: list[str] = []
        next_repair_directive: str | None = None
        next_previous_candidate: dict[str, object] | None = None
        next_failure_evidence: dict[str, object] | None = None
        next_phase = "context_initial"
        repaired_classes: set[_ContextFailureClass] = set()
        context_generation_count = 0
        initial_reason_codes: list[str] = []
        while context_generation_count < 3:
            context_generation_count += 1
            attempt = context_generation_count
            phase = next_phase
            raw = await self._contextual_choice_provider_call(
                request,
                telemetry,
                counter=(
                    "context_generation_calls"
                    if phase in {"context_initial", "context_lexical_fallback"}
                    else "context_repair_calls"
                ),
                stage=("context_generation" if phase == "context_initial" else phase),
                type_name=self.TYPE_NAME,
                data=build_contextual_choice_context_generation_prompt(
                    request,
                    plan_item={
                        "globalOrder": plan_item.global_order,
                        "reviewTarget": plan_item.review_target,
                        "targetExpression": plan_item.target_expression,
                        "distractors": plan_item.distractors,
                        "skillTag": plan_item.skill_tag,
                        "difficulty": plan_item.difficulty.value,
                        "complexityBand": plan_item.complexity_band,
                        "scenarioFamily": plan_item.scenario_family,
                        "anchorType": (
                            plan_item.anchor_type.value
                            if plan_item.anchor_type is not None else None
                        ),
                        "anchorValue": plan_item.anchor_value,
                    },
                    difficulty_demand=demand.generation_payload(),
                    repair_class=next_repair_class,
                    reason_codes=next_repair_codes,
                    repair_directive=next_repair_directive,
                    previous_candidate=next_previous_candidate,
                    failure_evidence=next_failure_evidence,
                ),
                schema=_CONTEXTUAL_CHOICE_CONTEXT_SCHEMA,
            )
            previous_candidate = self._contextual_choice_context_candidate_snapshot(raw)
            failure_evidence: dict[str, object]
            try:
                payload = _ContextualChoiceContextPayload.model_validate(raw)
            except ValidationError:
                reason_codes = ["CONTEXT_RESPONSE_SCHEMA_INVALID"]
                failure_evidence = {"contractReasonCodes": reason_codes}
            else:
                try:
                    question = self._normalize_contextual_choice_v3_context(
                        request,
                        slot,
                        plan_item,
                        payload,
                        accepted,
                    )
                except _ContextualChoiceContextContractError as exc:
                    if not exc.repairable:
                        logger.warning(
                            "CONTEXTUAL_CHOICE V3.3.3 internal context contract failed. "
                            "request_id=%s global_order=%d attempt=%d phase=%s "
                            "reason_code=%s",
                            request.request_id,
                            plan_item.global_order,
                            attempt,
                            phase,
                            exc.reason_code,
                        )
                        raise
                    reason_codes = [exc.reason_code]
                    failure_evidence = {"contractReasonCodes": reason_codes}
                else:
                    previous_candidate = self._contextual_choice_context_candidate_snapshot(
                        raw, question=question
                    )
                    try:
                        assessment = await self._contextual_choice_context_failure(
                            request,
                            slot,
                            question,
                            telemetry,
                        )
                    except _ContextualChoiceContextVerifierContractError as exc:
                        logger.warning(
                            "CONTEXTUAL_CHOICE V3.3.3 verifier contract failed. "
                            "request_id=%s global_order=%d attempt=%d phase=%s "
                            "reason_code=%s",
                            request.request_id,
                            plan_item.global_order,
                            attempt,
                            phase,
                            exc.reason_code,
                        )
                        raise
                    reason_codes = list(assessment.reason_codes)
                    failure_evidence = assessment.failure_evidence
                    if not reason_codes:
                        return question, plan_item
            if attempt == 1:
                initial_reason_codes = reason_codes
            failure_class = self._contextual_choice_failure_class(reason_codes)
            logger.info(
                "CONTEXTUAL_CHOICE V3.3.3 context validation rejected. "
                "request_id=%s global_order=%d attempt=%d phase=%s "
                "failure_class=%s reason_codes=%s",
                request.request_id,
                plan_item.global_order,
                attempt,
                phase,
                failure_class,
                reason_codes,
            )
            if attempt >= 3:
                break
            if failure_class not in repaired_classes:
                repaired_classes.add(failure_class)
                class_reason_codes = self._contextual_choice_reason_codes_for_class(
                    reason_codes, failure_class
                )
                next_repair_class = failure_class
                next_repair_codes = class_reason_codes
                next_repair_directive = self._contextual_choice_context_repair_reason(
                    class_reason_codes
                )
                next_previous_candidate = previous_candidate
                next_failure_evidence = failure_evidence
                next_phase = {
                    "CONTRACT_CONTEXT": "context_contract_repair",
                    "STRUCTURAL_CONTEXT": "context_structural_repair",
                    "SEMANTIC_CONTEXT": "context_semantic_repair",
                }[failure_class]
                continue
            if (
                not lexical_repair_used
                and self._contextual_choice_bundle_suspect(
                    initial_reason_codes, reason_codes
                )
            ):
                plan_item = await self._repair_contextual_choice_lexical_item(
                    request, telemetry, plan, plan_item, lexical_assessment,
                    reason_codes,
                    preserve_target=lexical_assessment.preserve_new_target,
                    repair_trigger="CONTEXT_VALIDATION",
                )
                lexical_repair_used = True
                lexical_assessment = await self._verify_contextual_choice_plan_item(
                    request, telemetry, plan_item, accepted
                )
                if lexical_assessment.reason_codes:
                    self._log_contextual_choice_lexical_rejection(
                        request, plan_item.global_order, "context_fallback",
                        list(lexical_assessment.reason_codes),
                        list(lexical_assessment.invalid_distractor_indexes),
                    )
                    raise _ContextualChoiceContentRejected(
                        f"CONTEXTUAL_CHOICE context-triggered lexical repair failed order={plan_item.global_order}",
                        stage="LEXICAL_VALIDATION",
                    )
                slot = replace(slot, vocabulary_plan_item=plan_item)
                next_repair_class = None
                next_repair_codes = []
                next_repair_directive = None
                next_previous_candidate = None
                next_failure_evidence = None
                next_phase = "context_lexical_fallback"
                continue
            break
        raise _ContextualChoiceContentRejected(
            f"CONTEXTUAL_CHOICE context repair exhausted order={plan_item.global_order}",
            stage="CONTEXT_VALIDATION",
        )

    def _normalize_contextual_choice_v3_context(
        self,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        plan_item: VocabularyPlanItem,
        payload: _ContextualChoiceContextPayload,
        accepted: dict[int, PracticeGeneratedQuestion],
    ) -> PracticeGeneratedQuestion:
        if payload.globalOrder != plan_item.global_order:
            raise _ContextualChoiceContextContractError(
                "CONTEXT_ORDER_MISMATCH", repairable=True
            )
        template = payload.contextTemplate.strip()
        unknown_markers = {
            marker
            for marker in re.findall(r"\{\{[^{}]*\}\}", template)
            if marker != CONTEXTUAL_CHOICE_TEMPLATE_MARKER
        }
        if unknown_markers:
            raise _ContextualChoiceContextContractError(
                "CONTEXT_UNKNOWN_MARKER", repairable=True
            )
        if template.count(CONTEXTUAL_CHOICE_TEMPLATE_MARKER) != 1:
            raise _ContextualChoiceContextContractError(
                "CONTEXT_MARKER_INVALID", repairable=True
            )
        normalized_target = self._normalize_for_leak_check(
            plan_item.target_expression
        )
        context_without_marker = template.replace(
            CONTEXTUAL_CHOICE_TEMPLATE_MARKER, "", 1
        )
        if (
            normalized_target
            and normalized_target
            in self._normalize_for_leak_check(context_without_marker)
        ):
            raise _ContextualChoiceContextContractError(
                "CONTEXT_TARGET_LEAK", repairable=True
            )
        explanation = payload.explanationLearning.strip()
        if not explanation or len(explanation) > 3000:
            raise _ContextualChoiceContextContractError(
                "CONTEXT_EXPLANATION_INVALID", repairable=True
            )
        try:
            complete_sentence, learner_prompt = render_contextual_choice_template(
                template,
                plan_item.target_expression,
            )
        except ValueError as exc:
            raise _ContextualChoiceContextContractError(
                "CONTEXT_TEMPLATE_ROUND_TRIP_INVALID", repairable=True
            ) from exc
        if len(learner_prompt) > 3000:
            raise _ContextualChoiceContextContractError(
                "CONTEXT_TEMPLATE_ROUND_TRIP_INVALID", repairable=True
            )
        try:
            self._assert_language_lane(
                request.learning_language,
                [complete_sentence, explanation],
                reason="CONTEXTUAL_CHOICE context is not in learningLanguage",
            )
        except ValueError as exc:
            raise _ContextualChoiceContextContractError(
                "CONTEXT_LANGUAGE_INVALID", repairable=True
            ) from exc
        correct_key = contextual_choice_answer_key(plan_item.global_order)
        correct_position = CONTEXTUAL_CHOICE_ANSWER_KEYS.index(correct_key)
        option_texts = list(plan_item.distractors)
        option_texts.insert(correct_position, plan_item.target_expression)
        try:
            question = PracticeGeneratedQuestion(
                order=slot.order,
                question_type=PracticeQuestionType.SINGLE_CHOICE,
                difficulty=plan_item.difficulty,
                complexity_band=plan_item.complexity_band,
                passage_id=None,
                passage_text=None,
                prompt=learner_prompt,
                options=[
                    PracticeOption(key=key, text=text)
                    for key, text in zip(
                        CONTEXTUAL_CHOICE_ANSWER_KEYS,
                        option_texts,
                        strict=True,
                    )
                ],
                correct_answer=[correct_key],
                skill_tag=plan_item.skill_tag,
                evidence_text=None,
                explanation_origin="PENDING",
                explanation_learning=explanation,
                target_expression=plan_item.target_expression,
                canonical_key=plan_item.canonical_key,
                review_target=plan_item.review_target,
                vocabulary_candidates=[],
            )
            self._validate_candidate(request, question, slot, accepted)
        except (ValidationError, ValueError, TypeError) as exc:
            raise _ContextualChoiceContextContractError(
                "CONTEXT_INTERNAL_CONTRACT_INVALID", repairable=False
            ) from exc
        return question

    @staticmethod
    def _contextual_choice_context_candidate_snapshot(
        raw: Any,
        *,
        question: PracticeGeneratedQuestion | None = None,
    ) -> dict[str, object]:
        """Keep only the immediately previous, bounded learner-visible candidate."""
        snapshot: dict[str, object] = {}
        if isinstance(raw, dict):
            order = raw.get("globalOrder")
            if isinstance(order, int):
                snapshot["globalOrder"] = order
            for source_key, output_key in (
                ("contextTemplate", "contextTemplate"),
                ("explanationLearning", "explanationLearning"),
            ):
                value = raw.get(source_key)
                if isinstance(value, str):
                    snapshot[output_key] = value[:3000]
        if question is not None:
            snapshot["renderedOptions"] = [
                {
                    "key": option.key,
                    "sentence": question.prompt.replace(
                        CONTEXTUAL_CHOICE_BLANK, option.text, 1
                    )[:3000],
                }
                for option in question.options
            ]
        return snapshot

    async def _contextual_choice_context_failure(
        self,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        question: PracticeGeneratedQuestion,
        telemetry: _ContextualChoiceTelemetry,
    ) -> _ContextualChoiceContextAssessment:
        plan_item = slot.vocabulary_plan_item
        if plan_item is None:
            raise ValueError("CONTEXTUAL_CHOICE V3 slot requires a plan item")
        visible = {
            "order": plan_item.global_order,
            "prompt": question.prompt,
            "options": [
                option.model_dump(mode="json", by_alias=True)
                for option in question.options
            ],
            "renderedOptions": [
                {
                    "key": option.key,
                    "sentence": question.prompt.replace(
                        CONTEXTUAL_CHOICE_BLANK, option.text, 1
                    ),
                }
                for option in question.options
            ],
            "skillTag": question.skill_tag,
            "reviewTarget": question.review_target,
        }
        raw = await self._contextual_choice_provider_call(
            request,
            telemetry,
            counter="context_validation_calls",
            stage="context_validation",
            type_name=self.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME,
            data=build_contextual_choice_context_verification_prompt(
                request,
                question=visible,
            ),
            schema=_CONTEXTUAL_CHOICE_CONTEXT_VERIFICATION_SCHEMA,
        )
        try:
            payload = _ContextualChoiceContextVerificationPayload.model_validate(raw)
        except ValidationError as exc:
            raise _ContextualChoiceContextVerifierContractError from exc
        if (
            len(payload.verdicts) != 1
            or payload.verdicts[0].order != plan_item.global_order
        ):
            raise _ContextualChoiceContextVerifierContractError
        verdict = payload.verdicts[0]
        structural_keys = tuple(dict.fromkeys(verdict.structurallyWellFormedKeys))
        if (
            verdict.bestAnswerKey not in CONTEXTUAL_CHOICE_ANSWER_KEYS
            or len(structural_keys) != len(verdict.structurallyWellFormedKeys)
            or not set(structural_keys).issubset(CONTEXTUAL_CHOICE_ANSWER_KEYS)
        ):
            raise _ContextualChoiceContextVerifierContractError
        expected_key = question.correct_answer[0]
        reason_codes: list[str] = []
        if verdict.ambiguous:
            reason_codes.append("AMBIGUOUS")
        if not verdict.supported:
            reason_codes.append("UNSUPPORTED")
        if not verdict.skillFit:
            reason_codes.append("SKILL_FIT")
        if verdict.definitionLike:
            reason_codes.append("DEFINITION_LIKE")
        if verdict.bestAnswerKey != expected_key:
            reason_codes.append("BEST_ANSWER_MISMATCH")
        if set(structural_keys) != set(CONTEXTUAL_CHOICE_ANSWER_KEYS):
            reason_codes.append("RENDERED_SENTENCE_STRUCTURALLY_INVALID")
        return _ContextualChoiceContextAssessment(
            reason_codes=tuple(reason_codes),
            failure_evidence={
                "ambiguous": verdict.ambiguous,
                "supported": verdict.supported,
                "skillFit": verdict.skillFit,
                "definitionLike": verdict.definitionLike,
                "bestAnswerKey": verdict.bestAnswerKey,
                "structurallyInvalidOptionKeys": [
                    key for key in CONTEXTUAL_CHOICE_ANSWER_KEYS
                    if key not in structural_keys
                ],
            },
        )

    @staticmethod
    def _contextual_choice_context_repair_reason(reason_codes: list[str]) -> str:
        instructions = {
            "CONTEXT_RESPONSE_SCHEMA_INVALID": (
                "Return only globalOrder, contextTemplate, and explanationLearning with the required types."
            ),
            "CONTEXT_ORDER_MISMATCH": (
                "Preserve the supplied globalOrder exactly."
            ),
            "CONTEXT_MARKER_INVALID": (
                "Return exactly one literal {{TARGET}} insertion marker."
            ),
            "CONTEXT_UNKNOWN_MARKER": (
                "Use no {{...}} marker other than the single literal {{TARGET}} marker."
            ),
            "CONTEXT_LANGUAGE_INVALID": (
                "Write contextTemplate and explanationLearning in the supplied learningLanguage."
            ),
            "CONTEXT_TARGET_LEAK": (
                "Do not copy the target literal outside the single {{TARGET}} marker."
            ),
            "CONTEXT_TEMPLATE_ROUND_TRIP_INVALID": (
                "Return a simple grammatical template with exactly one insertion site and no literal blank."
            ),
            "CONTEXT_EXPLANATION_INVALID": (
                "Return a non-empty concise explanationLearning in the learning language."
            ),
            "AMBIGUOUS": "Make the fixed target the unique best answer; remove visible ambiguity.",
            "UNSUPPORTED": "Strengthen visible support for the fixed target without changing the lexical bundle.",
            "SKILL_FIT": "Revise the context to test the assigned skill with the fixed bundle.",
            "DEFINITION_LIKE": "Replace the definition-like context with a natural usage decision.",
            "BEST_ANSWER_MISMATCH": "Revise context only so the fixed target is the unique best answer.",
            "RENDERED_SENTENCE_STRUCTURALLY_INVALID": (
                "Revise the blank frame so all four completed sentences are structurally well-formed "
                "without argument duplication."
            ),
        }
        return " ".join(instructions[code] for code in reason_codes)

    @staticmethod
    def _contextual_choice_failure_class(
        reason_codes: list[str],
    ) -> _ContextFailureClass:
        codes = set(reason_codes)
        if codes & _CONTRACT_CONTEXT_REASON_CODES:
            return "CONTRACT_CONTEXT"
        if codes & _STRUCTURAL_CONTEXT_REASON_CODES:
            return "STRUCTURAL_CONTEXT"
        if codes and codes.issubset(_SEMANTIC_CONTEXT_REASON_CODES):
            return "SEMANTIC_CONTEXT"
        raise _ContextualChoiceContextContractError(
            "CONTEXT_INTERNAL_CONTRACT_INVALID", repairable=False
        )

    @staticmethod
    def _contextual_choice_reason_codes_for_class(
        reason_codes: list[str], failure_class: _ContextFailureClass
    ) -> list[str]:
        if failure_class == "CONTRACT_CONTEXT":
            return [
                code for code in reason_codes
                if code in _CONTRACT_CONTEXT_REASON_CODES
            ]
        if failure_class == "STRUCTURAL_CONTEXT":
            return [
                code for code in reason_codes
                if code in _STRUCTURAL_CONTEXT_REASON_CODES
            ]
        return [
            code for code in reason_codes
            if code in _SEMANTIC_CONTEXT_REASON_CODES
        ]

    @staticmethod
    def _contextual_choice_bundle_suspect(
        initial_reason_codes: list[str], repair_reason_codes: list[str]
    ) -> bool:
        last = set(repair_reason_codes)
        if not last or last & {"UNSUPPORTED", "DEFINITION_LIKE"}:
            return False
        if last & {"AMBIGUOUS", "BEST_ANSWER_MISMATCH"}:
            return True
        return "SKILL_FIT" in initial_reason_codes and "SKILL_FIT" in last

    async def _generate_verified_contextual_choice_slot(
        self,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        accepted: dict[int, PracticeGeneratedQuestion],
    ) -> PracticeGeneratedQuestion:
        prior_questions = [
            *request.previous_questions,
            *(accepted[order] for order in sorted(accepted)),
        ]
        eligible_reviews = self._eligible_review_targets(request, log_rejections=False)
        excluded_canonical_keys = {
            question.canonical_key
            for question in prior_questions
            if question.canonical_key
        }
        excluded_target_expressions = {
            question.target_expression.strip()
            for question in prior_questions
            if question.target_expression and question.target_expression.strip()
        }
        if not slot.review_target:
            excluded_canonical_keys.update(
                target.canonical_key for target in eligible_reviews
            )
            excluded_target_expressions.update(
                target.expression.strip()
                for target in eligible_reviews
                if target.expression.strip()
            )

        last_reasons: list[str] = []
        for round_number in range(1, CONTEXTUAL_CHOICE_MAX_ROUNDS + 1):
            candidates, attempted_targets, attempted_keys = (
                await self._generate_contextual_choice_candidate_round(
                    request,
                    slot,
                    accepted,
                    round_number=round_number,
                    excluded_canonical_keys=excluded_canonical_keys,
                    excluded_target_expressions=excluded_target_expressions,
                )
            )
            if not slot.review_target:
                excluded_target_expressions.update(attempted_targets)
                excluded_canonical_keys.update(attempted_keys)
            if not candidates:
                last_reasons.append("no candidates survived deterministic validation")
                continue

            verified, reasons = await self._verify_contextual_choice_candidates(
                request,
                slot,
                candidates,
                prior_questions=prior_questions,
            )
            last_reasons.extend(reasons)
            if verified:
                selected, competitor_count = min(
                    verified,
                    key=lambda item: (
                        -item[1],
                        item[0].candidate_id,
                    ),
                )
                logger.info(
                    "CONTEXTUAL_CHOICE V2 candidate selected. request_id=%s order=%d "
                    "round=%d candidate_id=%s scenario_family=%s genuine_competitors=%d",
                    request.request_id,
                    slot.order,
                    round_number,
                    selected.candidate_id[:80],
                    slot.scenario_family,
                    competitor_count,
                )
                return selected.question

        logger.warning(
            "CONTEXTUAL_CHOICE V2 selection exhausted. request_id=%s order=%d rounds=%d reasons=%s",
            request.request_id,
            slot.order,
            CONTEXTUAL_CHOICE_MAX_ROUNDS,
            last_reasons[-6:],
        )
        raise ValueError(
            f"CONTEXTUAL_CHOICE candidate selection exhausted order={slot.order}"
        )

    async def _generate_contextual_choice_candidate_round(
        self,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        accepted: dict[int, PracticeGeneratedQuestion],
        *,
        round_number: int,
        excluded_canonical_keys: set[str],
        excluded_target_expressions: set[str],
    ) -> tuple[list[_ContextualChoiceCandidate], set[str], set[str]]:
        prior_questions = [
            *request.previous_questions,
            *(accepted[order] for order in sorted(accepted)),
        ]
        try:
            raw = await asyncio.wait_for(
                self.provider.call(
                    type_name=self.TYPE_NAME,
                    data=build_contextual_choice_candidate_batch_prompt(
                        request,
                        slot=slot.prompt_payload(),
                        round_number=round_number,
                        previous_questions=prior_questions,
                        excluded_canonical_keys=sorted(excluded_canonical_keys),
                        excluded_target_expressions=sorted(
                            excluded_target_expressions
                        ),
                    ),
                    schema=_CONTEXTUAL_CHOICE_CANDIDATE_SCHEMA,
                ),
                timeout=self.timeout_seconds,
            )
            payload = _ContextualChoiceCandidatePayload.model_validate(raw)
            if len(payload.candidates) != CONTEXTUAL_CHOICE_CANDIDATES_PER_ROUND:
                raise ValueError(
                    "CONTEXTUAL_CHOICE generation round must return exactly three candidates"
                )
        except Exception as exc:
            logger.warning(
                "CONTEXTUAL_CHOICE V2 generation round failed. request_id=%s order=%d "
                "round=%d/%d type=%s",
                request.request_id,
                slot.order,
                round_number,
                CONTEXTUAL_CHOICE_MAX_ROUNDS,
                type(exc).__name__,
            )
            return [], set(), set()

        candidate_ids = [
            item.get("candidateId", "").strip()
            if isinstance(item.get("candidateId"), str)
            else ""
            for item in payload.candidates
        ]
        targets = [
            item.get("targetExpression", "").strip()
            if isinstance(item.get("targetExpression"), str)
            else ""
            for item in payload.candidates
        ]
        canonical_keys = [
            self._canonical_key_for_expression(target) if target else ""
            for target in targets
        ]
        complete_sentences = [
            item.get("completeSentence", "").strip()
            if isinstance(item.get("completeSentence"), str)
            else ""
            for item in payload.candidates
        ]
        attempted_targets = {target for target in targets if target}
        attempted_keys = {key for key in canonical_keys if key}
        duplicate_ids = {
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id and candidate_ids.count(candidate_id) > 1
        }
        duplicate_targets = {
            self._normalize_for_leak_check(target)
            for target in targets
            if target
            and sum(
                self._normalize_for_leak_check(other) == self._normalize_for_leak_check(target)
                for other in targets
            )
            > 1
        }
        duplicate_keys = {
            key for key in canonical_keys if key and canonical_keys.count(key) > 1
        }
        duplicate_review_sentences = {
            self._normalize_for_leak_check(sentence)
            for sentence in complete_sentences
            if slot.review_target
            and sentence
            and sum(
                self._normalize_for_leak_check(other)
                == self._normalize_for_leak_check(sentence)
                for other in complete_sentences
            )
            > 1
        }
        previous_complete_sentences = {
            self._normalize_for_leak_check(
                question.prompt.replace(
                    CONTEXTUAL_CHOICE_BLANK,
                    question.target_expression,
                    1,
                )
            )
            for question in prior_questions
            if question.prompt.count(CONTEXTUAL_CHOICE_BLANK) == 1
            and question.target_expression
        }
        normalized_excluded_targets = {
            self._normalize_for_leak_check(target)
            for target in excluded_target_expressions
            if target.strip()
        }

        survivors: list[_ContextualChoiceCandidate] = []
        for item, candidate_id, target, canonical_key, complete_sentence in zip(
            payload.candidates,
            candidate_ids,
            targets,
            canonical_keys,
            complete_sentences,
            strict=True,
        ):
            try:
                if not candidate_id:
                    raise ValueError("CONTEXTUAL_CHOICE candidateId must not be empty")
                if candidate_id in duplicate_ids:
                    raise ValueError("CONTEXTUAL_CHOICE candidateId must be unique")
                if slot.review_target:
                    normalized_sentence = self._normalize_for_leak_check(
                        complete_sentence
                    )
                    if normalized_sentence in duplicate_review_sentences:
                        raise ValueError(
                            "CONTEXTUAL_CHOICE review candidate sentences must be distinct"
                        )
                    if normalized_sentence in previous_complete_sentences:
                        raise ValueError(
                            "CONTEXTUAL_CHOICE review candidate copied a previous sentence"
                        )
                else:
                    normalized_target = self._normalize_for_leak_check(target)
                    if normalized_target in duplicate_targets:
                        raise ValueError(
                            "CONTEXTUAL_CHOICE batch targetExpression must be unique"
                        )
                    if canonical_key in duplicate_keys:
                        raise ValueError(
                            "CONTEXTUAL_CHOICE batch canonicalKey must be unique"
                        )
                    if normalized_target in normalized_excluded_targets:
                        raise ValueError(
                            "CONTEXTUAL_CHOICE targetExpression repeats an accepted or excluded target"
                        )
                    if canonical_key in excluded_canonical_keys:
                        raise ValueError(
                            "CONTEXTUAL_CHOICE canonicalKey repeats an accepted or excluded target"
                        )
                question = self._normalize_contextual_choice_v2_candidate(
                    request,
                    slot,
                    item,
                    accepted,
                )
                survivors.append(
                    _ContextualChoiceCandidate(
                        candidate_id=candidate_id,
                        question=question,
                    )
                )
            except (ValidationError, ValueError) as exc:
                logger.info(
                    "CONTEXTUAL_CHOICE V2 deterministic candidate rejected. "
                    "request_id=%s order=%d round=%d candidate_id=%s reason=%s",
                    request.request_id,
                    slot.order,
                    round_number,
                    candidate_id[:80],
                    str(exc)[:300],
                )
        return survivors, attempted_targets, attempted_keys

    async def _verify_contextual_choice_candidates(
        self,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        candidates: list[_ContextualChoiceCandidate],
        *,
        prior_questions: list[PracticeGeneratedQuestion],
    ) -> tuple[list[tuple[_ContextualChoiceCandidate, int]], list[str]]:
        verification_input = [
            {
                "candidateId": candidate.candidate_id,
                "prompt": candidate.question.prompt,
                "options": [
                    option.model_dump(by_alias=True)
                    for option in candidate.question.options
                ],
                "skillTag": candidate.question.skill_tag,
                "reviewTarget": candidate.question.review_target,
            }
            for candidate in candidates
        ]
        try:
            raw = await asyncio.wait_for(
                self.provider.call(
                    type_name=self.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME,
                    data=build_contextual_choice_verification_prompt(
                        request,
                        candidates=verification_input,
                        previous_questions=prior_questions,
                    ),
                    schema=_CONTEXTUAL_CHOICE_VERIFICATION_SCHEMA,
                ),
                timeout=self.timeout_seconds,
            )
            payload = _ContextualChoiceVerificationPayload.model_validate(raw)
            verdict_by_id = {
                verdict.candidateId: verdict for verdict in payload.verdicts
            }
            expected_ids = {candidate.candidate_id for candidate in candidates}
            if (
                set(verdict_by_id) != expected_ids
                or len(payload.verdicts) != len(expected_ids)
            ):
                raise ValueError("CONTEXTUAL_CHOICE verifier coverage mismatch")
        except Exception as exc:
            logger.warning(
                "CONTEXTUAL_CHOICE V2 verifier round failed. request_id=%s order=%d type=%s",
                request.request_id,
                slot.order,
                type(exc).__name__,
            )
            return [], [f"verifier failure: {type(exc).__name__}"]

        demand = vocabulary_difficulty_recipe(
            mode=request.mode,
            band=slot.complexity_band,
            skill_tag=slot.skill_tag,
            question_type=PracticeQuestionType.SINGLE_CHOICE.value,
        ).demand
        if not isinstance(demand, ContextualChoiceDemand):
            raise ValueError("CONTEXTUAL_CHOICE requires contextual-choice demand")
        minimum_competitors = contextual_choice_required_close_distractors(
            demand.minimum_close_distractors,
            review_target=slot.review_target,
        )
        accepted: list[tuple[_ContextualChoiceCandidate, int]] = []
        reasons: list[str] = []
        for candidate in candidates:
            verdict = verdict_by_id[candidate.candidate_id]
            expected_key = candidate.question.correct_answer[0]
            option_keys = {option.key for option in candidate.question.options}
            genuine_keys = tuple(dict.fromkeys(verdict.genuineCompetitorKeys))
            if not set(genuine_keys).issubset(option_keys - {expected_key}):
                reasons.append("genuine competitor keys include a non-wrong option")
                continue
            assessment = normalize_vocabulary_semantic_assessment(
                best_answer_key=verdict.bestAnswerKey,
                ambiguous=verdict.ambiguous,
                supported=verdict.supported,
                mode_fit=verdict.skillFit,
                answer_leakage=False,
                context_dependent=True,
                distractors_plausible=True,
                definition_like=verdict.definitionLike,
                lexical_concept_repeated=(
                    verdict.lexicalConceptRepeated and not slot.review_target
                ),
                genuine_competitor_keys=genuine_keys,
            )
            decision = semantic_quality_policy_for_vocabulary_mode(
                request.mode
            ).decide(
                assessment=assessment,
                context=VocabularyAcceptanceContext(
                    expected_answer_key=expected_key,
                    minimum_close_distractors=minimum_competitors,
                    review_target=slot.review_target,
                ),
            )
            if decision.action == "ACCEPT":
                accepted.append((candidate, len(genuine_keys)))
            else:
                reasons.append(decision.reason)
                logger.info(
                    "CONTEXTUAL_CHOICE V2 semantic candidate rejected. request_id=%s "
                    "order=%d candidate_id=%s reason=%s",
                    request.request_id,
                    slot.order,
                    candidate.candidate_id[:80],
                    decision.reason[:300],
                )
        return accepted, reasons

    async def _generate_verified_reading_questions(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
    ) -> list[PracticeGeneratedQuestion]:
        """Generate Reading with independent structural and semantic retry budgets.

        Reading candidate generation can fail for deterministic/schema/language reasons while an
        otherwise-good question can fail Mini only for semantic quality. Those are different
        failure classes and must not consume the same scarce retry counter. Keep successful slots,
        regenerate only failed slots, and preserve the existing small generation batches.
        """
        slot_by_order = {slot.order: slot for slot in slots}
        generation_attempts = {slot.order: 0 for slot in slots}
        semantic_attempts = {slot.order: 0 for slot in slots}
        repair_attempts = {slot.order: 0 for slot in slots}
        accepted: dict[int, PracticeGeneratedQuestion] = {}
        semantically_verified: set[int] = set()
        pending_repaired: set[int] = set()
        retry_feedback: dict[int, str] = {}
        repair_calls = 0
        repair_accepted = 0
        repair_rejected = 0

        while len(semantically_verified) < len(slots):
            pending_generation = [
                order
                for order in sorted(slot_by_order)
                if order not in accepted
                and order not in semantically_verified
                and generation_attempts[order] < _MAX_READING_GENERATION_ATTEMPTS_PER_SLOT
                and semantic_attempts[order] < _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT
            ]
            for start in range(0, len(pending_generation), _CANDIDATE_BATCH_SIZE):
                batch_orders = pending_generation[start : start + _CANDIDATE_BATCH_SIZE]
                batch_slots = [slot_by_order[order] for order in batch_orders]
                for order in batch_orders:
                    generation_attempts[order] += 1
                candidate_failures = await self._generate_candidate_batch(
                    request,
                    batch_slots,
                    accepted,
                    retry_feedback=(retry_feedback or None),
                )
                for order in batch_orders:
                    if order in accepted:
                        retry_feedback.pop(order, None)
                        continue
                    reason = candidate_failures.get(
                        order, "candidate generation did not yield an accepted item"
                    )
                    retry_feedback[order] = self._reading_retry_feedback(reason)

            generation_exhausted = [
                order
                for order in sorted(slot_by_order)
                if order not in accepted
                and order not in semantically_verified
                and generation_attempts[order] >= _MAX_READING_GENERATION_ATTEMPTS_PER_SLOT
            ]
            if generation_exhausted:
                raise ValueError(
                    "reading candidate generation exhausted orders="
                    f"{generation_exhausted}"
                )

            unverified = [
                accepted[order]
                for order in sorted(accepted)
                if order not in semantically_verified
            ]
            if not unverified:
                continue

            for question in unverified:
                semantic_attempts[question.order] += 1
            semantic_outcome = await self._semantic_verification_outcome(
                request,
                unverified,
                previous_reading_questions=[
                    accepted[order] for order in sorted(semantically_verified)
                ],
            )

            semantic_exhausted: list[int] = []
            for question in unverified:
                failure = semantic_outcome.failures.get(question.order)
                if failure is None:
                    semantically_verified.add(question.order)
                    retry_feedback.pop(question.order, None)
                    if question.order in pending_repaired:
                        pending_repaired.remove(question.order)
                        repair_accepted += 1
                        logger.info(
                            "Reading distractor repair accepted. request_id=%s order=%d "
                            "generation_attempt=%d/%d semantic_attempt=%d/%d "
                            "repair_attempt=%d/%d",
                            request.request_id,
                            question.order,
                            generation_attempts[question.order],
                            _MAX_READING_GENERATION_ATTEMPTS_PER_SLOT,
                            semantic_attempts[question.order],
                            _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT,
                            repair_attempts[question.order],
                            _MAX_DISTRACTOR_REPAIR_ATTEMPTS_PER_SLOT,
                        )
                    continue

                logger.warning(
                    "Reading/Vocabulary semantic candidate rejected. request_id=%s order=%d "
                    "generation_attempt=%d/%d semantic_attempt=%d/%d reason=%s",
                    request.request_id,
                    question.order,
                    generation_attempts[question.order],
                    _MAX_READING_GENERATION_ATTEMPTS_PER_SLOT,
                    semantic_attempts[question.order],
                    _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT,
                    failure[:300],
                )
                if question.order in pending_repaired:
                    pending_repaired.remove(question.order)
                    repair_rejected += 1
                    logger.warning(
                        "Reading distractor repair rejected. request_id=%s order=%d "
                        "generation_attempt=%d/%d semantic_attempt=%d/%d "
                        "repair_attempt=%d/%d reason=%s",
                        request.request_id,
                        question.order,
                        generation_attempts[question.order],
                        _MAX_READING_GENERATION_ATTEMPTS_PER_SLOT,
                        semantic_attempts[question.order],
                        _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT,
                        repair_attempts[question.order],
                        _MAX_DISTRACTOR_REPAIR_ATTEMPTS_PER_SLOT,
                        failure[:300],
                    )

                verdict = semantic_outcome.verdicts.get(question.order)
                expected_key = question.correct_answer[0]
                if (
                    repair_attempts[question.order]
                    < _MAX_DISTRACTOR_REPAIR_ATTEMPTS_PER_SLOT
                    and semantic_attempts[question.order]
                    < _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT
                    and failure == "distractors are too weak or unrelated"
                    and verdict is not None
                    and self._is_reading_distractor_only_failure(
                        verdict,
                        expected_answer_key=expected_key,
                    )
                ):
                    repair_attempts[question.order] += 1
                    repair_calls += 1
                    logger.info(
                        "Reading distractor repair attempted. request_id=%s order=%d "
                        "generation_attempt=%d/%d semantic_attempt=%d/%d "
                        "repair_attempt=%d/%d",
                        request.request_id,
                        question.order,
                        generation_attempts[question.order],
                        _MAX_READING_GENERATION_ATTEMPTS_PER_SLOT,
                        semantic_attempts[question.order],
                        _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT,
                        repair_attempts[question.order],
                        _MAX_DISTRACTOR_REPAIR_ATTEMPTS_PER_SLOT,
                    )
                    repaired = await self._repair_reading_distractors(
                        request,
                        question,
                        slot_by_order[question.order],
                        accepted,
                    )
                    if repaired is not None:
                        accepted[question.order] = repaired
                        pending_repaired.add(question.order)
                        retry_feedback.pop(question.order, None)
                        continue
                    repair_rejected += 1
                    logger.warning(
                        "Reading distractor repair fallback to full regeneration. "
                        "request_id=%s order=%d generation_attempt=%d/%d "
                        "semantic_attempt=%d/%d repair_attempt=%d/%d",
                        request.request_id,
                        question.order,
                        generation_attempts[question.order],
                        _MAX_READING_GENERATION_ATTEMPTS_PER_SLOT,
                        semantic_attempts[question.order],
                        _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT,
                        repair_attempts[question.order],
                        _MAX_DISTRACTOR_REPAIR_ATTEMPTS_PER_SLOT,
                    )
                accepted.pop(question.order, None)
                retry_feedback[question.order] = self._reading_retry_feedback(failure)
                if semantic_attempts[question.order] >= _MAX_READING_SEMANTIC_ATTEMPTS_PER_SLOT:
                    semantic_exhausted.append(question.order)

            if semantic_exhausted:
                raise ValueError(
                    "reading semantic verification exhausted orders="
                    f"{sorted(semantic_exhausted)}"
                )

        if repair_calls:
            logger.info(
                "Reading distractor repair summary. request_id=%s "
                "distractorRepairCalls=%d distractorRepairAccepted=%d "
                "distractorRepairRejected=%d",
                request.request_id,
                repair_calls,
                repair_accepted,
                repair_rejected,
            )
        return [accepted[order] for order in sorted(accepted)]

    async def _generate_verified_usage_distinction_questions(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
    ) -> list[PracticeGeneratedQuestion]:
        """Run the strict usage pipeline end-to-end one slot at a time.

        Generation uses one-item-per-Luna-call and previously accumulated all ten
        candidates before asking Mini to judge them together. In production that caused
        synchronized batch rejections and made retry feedback much less targeted. Keep
        already-approved slots, and finish deterministic -> Nano -> Mini for one slot
        before moving to the next.
        """
        accepted: dict[int, PracticeGeneratedQuestion] = {}

        for slot in slots:
            retry_feedback: str | None = None
            retry_target_expression: str | None = None
            generation_attempts = 0
            semantic_attempts = 0
            last_stage = "candidate"

            while (
                generation_attempts < _MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT
                and semantic_attempts < _MAX_USAGE_DISTINCTION_SEMANTIC_ATTEMPTS_PER_SLOT
            ):
                generation_attempts += 1
                candidate_failures = await self._generate_candidate_batch(
                    request,
                    [slot],
                    accepted,
                    retry_feedback=(
                        {slot.order: retry_feedback} if retry_feedback else None
                    ),
                    retry_target_expressions=(
                        {slot.order: retry_target_expression}
                        if retry_target_expression
                        else None
                    ),
                )
                question = accepted.get(slot.order)
                if question is None:
                    retry_feedback = candidate_failures.get(
                        slot.order, "candidate generation did not yield an accepted item"
                    )
                    last_stage = "candidate"
                    continue

                prescreen_flags = await self._usage_prescreen_failures(request, [question])
                prescreen_failure = prescreen_flags.get(slot.order)
                if prescreen_failure is not None:
                    logger.warning(
                        "Reading/Vocabulary Nano prescreen flagged candidate; deferring to Mini. "
                        "request_id=%s order=%d generation_attempt=%d/%d "
                        "semantic_attempt=%d/%d reason=%s",
                        request.request_id,
                        slot.order,
                        generation_attempts,
                        _MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT,
                        semantic_attempts + 1,
                        _MAX_USAGE_DISTINCTION_SEMANTIC_ATTEMPTS_PER_SLOT,
                        prescreen_failure[:300],
                    )

                semantic_attempts += 1
                semantic_failures = await self._semantic_failures(request, [question])
                semantic_failure = semantic_failures.get(slot.order)
                if semantic_failure is None:
                    retry_feedback = None
                    retry_target_expression = None
                    break

                logger.warning(
                    "Reading/Vocabulary semantic candidate rejected. request_id=%s order=%d "
                    "generation_attempt=%d/%d semantic_attempt=%d/%d reason=%s",
                    request.request_id,
                    slot.order,
                    generation_attempts,
                    _MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT,
                    semantic_attempts,
                    _MAX_USAGE_DISTINCTION_SEMANTIC_ATTEMPTS_PER_SLOT,
                    semantic_failure[:300],
                )
                accepted.pop(slot.order, None)
                retry_target_expression = question.target_expression
                retry_feedback = self._usage_retry_feedback(semantic_failure)
                last_stage = "semantic"

            if slot.order not in accepted:
                raise ValueError(
                    f"{last_stage} verification exhausted orders={[slot.order]} "
                    f"generationAttempts={generation_attempts}/"
                    f"{_MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT} "
                    f"semanticAttempts={semantic_attempts}/"
                    f"{_MAX_USAGE_DISTINCTION_SEMANTIC_ATTEMPTS_PER_SLOT}"
                )

        return [accepted[order] for order in sorted(accepted)]

    async def _generate_verified_composition_questions(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
    ) -> list[PracticeGeneratedQuestion]:
        """Generate COMPOSITION slots independently with separate structural/semantic budgets.

        COMPOSITION has strict application-owned ORDERING shapes. A malformed Luna payload must
        not consume the same three-attempt budget used for semantic quality, otherwise one slot can
        exhaust before Mini ever gets a fair chance to judge it. ORDERING items need structural
        validation only; SINGLE_CHOICE items still pass the independent Mini verifier.
        """
        accepted: dict[int, PracticeGeneratedQuestion] = {}

        for slot in slots:
            retry_feedback: str | None = None
            retry_target_expression: str | None = None
            generation_attempts = 0
            semantic_attempts = 0
            last_stage = "candidate"

            while (
                generation_attempts < _MAX_COMPOSITION_GENERATION_ATTEMPTS_PER_SLOT
                and semantic_attempts < _MAX_COMPOSITION_SEMANTIC_ATTEMPTS_PER_SLOT
            ):
                generation_attempts += 1
                candidate_failures = await self._generate_candidate_batch(
                    request,
                    [slot],
                    accepted,
                    retry_feedback=({slot.order: retry_feedback} if retry_feedback else None),
                    retry_target_expressions=(
                        {slot.order: retry_target_expression}
                        if retry_target_expression
                        else None
                    ),
                )
                question = accepted.get(slot.order)
                if question is None:
                    retry_feedback = self._composition_retry_feedback(
                        candidate_failures.get(
                            slot.order, "candidate generation did not yield an accepted item"
                        ),
                        slot,
                    )
                    last_stage = "candidate"
                    continue

                if question.question_type == PracticeQuestionType.ORDERING:
                    retry_feedback = None
                    retry_target_expression = None
                    break

                semantic_attempts += 1
                semantic_failures = await self._semantic_failures(request, [question])
                semantic_failure = semantic_failures.get(slot.order)
                if semantic_failure is None:
                    retry_feedback = None
                    retry_target_expression = None
                    break

                logger.warning(
                    "Reading/Vocabulary semantic candidate rejected. request_id=%s order=%d "
                    "generation_attempt=%d/%d semantic_attempt=%d/%d reason=%s",
                    request.request_id,
                    slot.order,
                    generation_attempts,
                    _MAX_COMPOSITION_GENERATION_ATTEMPTS_PER_SLOT,
                    semantic_attempts,
                    _MAX_COMPOSITION_SEMANTIC_ATTEMPTS_PER_SLOT,
                    semantic_failure[:300],
                )
                accepted.pop(slot.order, None)
                retry_target_expression = question.target_expression
                retry_feedback = self._composition_retry_feedback(semantic_failure, slot)
                last_stage = "semantic"

            if slot.order not in accepted:
                raise ValueError(
                    f"{last_stage} verification exhausted orders={[slot.order]} "
                    f"generationAttempts={generation_attempts}/"
                    f"{_MAX_COMPOSITION_GENERATION_ATTEMPTS_PER_SLOT} "
                    f"semanticAttempts={semantic_attempts}/"
                    f"{_MAX_COMPOSITION_SEMANTIC_ATTEMPTS_PER_SLOT}"
                )

        return [accepted[order] for order in sorted(accepted)]

    @staticmethod
    def _candidate_attempt_limit(request: PracticeGenerationRequest) -> int:
        if (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.USAGE_DISTINCTION.value
        ):
            return _MAX_USAGE_DISTINCTION_GENERATION_ATTEMPTS_PER_SLOT
        return _MAX_CANDIDATE_ATTEMPTS_PER_SLOT

    @staticmethod
    def _candidate_batch_size(request: PracticeGenerationRequest) -> int:
        if (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.USAGE_DISTINCTION.value
        ):
            # Contextual usage questions are the strictest mode. Generate one slot per call so a
            # difficult slot cannot contaminate the rest of the batch and retries stay minimal.
            return _USAGE_DISTINCTION_BATCH_SIZE
        return _CANDIDATE_BATCH_SIZE

    @staticmethod
    def _usage_intent_for_skill(skill_tag: str) -> str | None:
        return usage_intent_for_skill(skill_tag)

    async def _generate_candidate_batch(
        self,
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot],
        accepted: dict[int, PracticeGeneratedQuestion],
        *,
        retry_feedback: dict[int, str] | None = None,
        retry_target_expressions: dict[int, str] | None = None,
        retry_excluded_canonical_keys: dict[int, set[str]] | None = None,
        retry_excluded_target_expressions: dict[int, set[str]] | None = None,
    ) -> dict[int, str]:
        expected_orders = {slot.order for slot in slots}
        accepted_keys = {
            question.canonical_key
            for question in [*request.previous_questions, *accepted.values()]
            if question.canonical_key
        }
        accepted_target_expressions = {
            question.target_expression.strip()
            for question in [*request.previous_questions, *accepted.values()]
            if question.target_expression and question.target_expression.strip()
        }
        review_targets = self._eligible_review_targets(request, log_rejections=False)
        free_slot_excluded_keys = sorted(
            accepted_keys | {target.canonical_key for target in review_targets}
        )
        free_slot_excluded_target_expressions = sorted(
            accepted_target_expressions
            | {target.expression.strip() for target in review_targets if target.expression.strip()}
        )
        if request.mode == VocabularyMode.MEANING_RELATION.value:
            accepted_keys_payload = sorted(accepted_keys)
            accepted_targets_payload = sorted(accepted_target_expressions)
        else:
            # Preserve the existing exclusion contract for modes not covered by this hardening.
            accepted_keys_payload = free_slot_excluded_keys
            accepted_targets_payload = free_slot_excluded_target_expressions
        slot_payloads: list[dict[str, Any]] = []
        for slot in slots:
            payload = slot.prompt_payload()
            if (
                request.domain == PracticeDomain.READING
                and request.mode in {ReadingMode.STRUCTURE.value, ReadingMode.CONTEXT_INFERENCE.value}
            ):
                global_order = request.question_offset + slot.order
                payload["samePassageTaskPosition"] = (
                    global_order if global_order <= 3 else global_order - 3
                )
                payload["samePassageTaskCount"] = 3 if global_order <= 3 else 2
                if request.mode == ReadingMode.STRUCTURE.value and slot.complexity_band == 5:
                    payload["structureJudgmentContract"] = structure_judgment_contract(
                        band=5,
                        position=payload["samePassageTaskPosition"],
                        count=payload["samePassageTaskCount"],
                    )
                    payload["reservedFutureQuestionFocuses"] = [
                        plan.question_focus for plan in slot.same_passage_plans
                        if plan.global_order > global_order
                    ]
            if (
                request.domain == PracticeDomain.VOCABULARY
                and request.mode
                in {
                    VocabularyMode.MEANING_RELATION.value,
                }
                and not slot.review_target
            ):
                payload["excludedCanonicalKeys"] = sorted(
                    set(free_slot_excluded_keys)
                    | (
                        retry_excluded_canonical_keys.get(slot.order, set())
                        if retry_excluded_canonical_keys
                        else set()
                    )
                )
                payload["excludedTargetExpressions"] = sorted(
                    set(free_slot_excluded_target_expressions)
                    | (
                        retry_excluded_target_expressions.get(slot.order, set())
                        if retry_excluded_target_expressions
                        else set()
                    )
                )
            if retry_feedback and retry_feedback.get(slot.order):
                payload["retryFeedback"] = retry_feedback[slot.order][:600]
            if retry_target_expressions and retry_target_expressions.get(slot.order):
                payload["retryTargetExpression"] = retry_target_expressions[slot.order]
                payload["preserveTargetOnRetry"] = True
            slot_payloads.append(payload)

        failures: dict[int, str] = {}
        try:
            raw = await asyncio.wait_for(
                self.provider.call(
                    type_name=(
                        self.READING_SOL_TYPE_NAME
                        if request.domain == PracticeDomain.READING
                        and settings.AI_READING_GENERATION_MODEL == "SOL"
                        else self.TYPE_NAME
                    ),
                    data=build_practice_generation_prompt(
                        request,
                        slot_payloads,
                        excluded_canonical_keys=accepted_keys_payload,
                        excluded_target_expressions=accepted_targets_payload,
                        previous_questions=(
                            [*request.previous_questions, *accepted.values()]
                            if request.domain == PracticeDomain.READING else None
                        ),
                    ),
                    schema=self._candidate_schema_for(request, slots),
                ),
                timeout=self._generation_timeout_for(request),
            )
            payload = _PracticeCandidatePayload.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "Reading/Vocabulary candidate batch provider/schema failure. request_id=%s orders=%s type=%s",
                request.request_id,
                sorted(expected_orders),
                type(exc).__name__,
            )
            return {order: f"provider/schema failure: {type(exc).__name__}" for order in expected_orders}

        raw_by_order: dict[int, dict[str, Any]] = {}
        duplicate_orders: set[int] = set()
        for item in payload.questions:
            order = item.get("order")
            if not isinstance(order, int) or order not in expected_orders:
                continue
            if order in raw_by_order:
                duplicate_orders.add(order)
            raw_by_order[order] = item
        for order in duplicate_orders:
            raw_by_order.pop(order, None)

        slot_by_order = {slot.order: slot for slot in slots}
        for order in sorted(expected_orders):
            item = raw_by_order.get(order)
            if item is None:
                logger.warning(
                    "Reading/Vocabulary candidate missing/duplicate. request_id=%s order=%d",
                    request.request_id,
                    order,
                )
                failures[order] = "candidate missing or duplicate in provider response"
                continue
            try:
                normalized_item = item
                if request.domain == PracticeDomain.READING:
                    reading_slot = slot_by_order[order]
                    if (request.mode == ReadingMode.COMPREHENSION.value
                            and request.complexity_band == 1
                            and reading_slot.skill_tag == ReadingSkill.INFERENCE.value):
                        self._validate_b1_inference_candidate_evidence(
                            request, reading_slot, item,
                        )
                    normalized_item = {
                        # Inference evidence is an internal generator contract. It is checked
                        # above, then excluded from the public question DTO for every B1 slot.
                        **{key: value for key, value in item.items()
                           if key not in {"inferenceClueQuote", "unstatedInference"}},
                        # Difficulty labels are application-owned plan metadata. They do not prove
                        # that the generated question achieved the intended cognitive demand.
                        "difficulty": reading_slot.difficulty.value,
                        "complexityBand": reading_slot.complexity_band,
                        "skillTag": reading_slot.skill_tag,
                        "vocabularyCandidates": self._sanitize_reading_vocabulary_candidates(
                            item.get("vocabularyCandidates", []),
                            reading_slot.passage_text or "",
                        ),
                    }
                elif request.domain == PracticeDomain.VOCABULARY:
                    vocabulary_slot = slot_by_order[order]
                    normalized_item = {
                        **item,
                        # Target labels and subtype binding come from the application slot. A
                        # generator self-report cannot prove or relabel achieved difficulty.
                        "difficulty": vocabulary_slot.difficulty.value,
                        "complexityBand": vocabulary_slot.complexity_band,
                        "skillTag": vocabulary_slot.skill_tag,
                        "questionType": (
                            vocabulary_slot.question_type.value
                            if vocabulary_slot.question_type is not None
                            else item.get("questionType")
                        ),
                        "passageId": None,
                        "passageText": None,
                        "evidenceText": None,
                        "vocabularyCandidates": [],
                    }
                    normalized_item = self._normalize_meaning_relation_candidate(
                        request,
                        vocabulary_slot,
                        normalized_item,
                    )
                # Placeholder is internal only and always overwritten after semantic acceptance.
                question = PracticeGeneratedQuestion.model_validate(
                    {**normalized_item, "explanationOrigin": "PENDING"}
                )
                self._normalize_reading_candidate_metadata(
                    request,
                    question,
                    slot_by_order[order],
                )
                self._normalize_usage_distinction_candidate_metadata(
                    request,
                    question,
                    slot_by_order[order],
                )
                self._normalize_composition_candidate_metadata(
                    request,
                    question,
                    slot_by_order[order],
                )
                expected_retry_target = (
                    retry_target_expressions.get(order)
                    if retry_target_expressions
                    else None
                )
                if (
                    expected_retry_target
                    and question.target_expression != expected_retry_target
                ):
                    raise ValueError("semantic retry changed targetExpression")
                if request.domain == PracticeDomain.READING:
                    slot = slot_by_order[order]
                    assert slot.passage_id is not None
                    assert slot.passage_text is not None
                    question_spec = build_reading_question_difficulty_spec(
                        mode=request.mode,
                        difficulty=slot.difficulty.value,
                        complexity_band=slot.complexity_band,
                        skill_tag=slot.skill_tag,
                        passage_id=slot.passage_id,
                    )
                    validation = project_reading_question_validation(
                        question_spec,
                        lambda: self._validate_candidate(
                            request,
                            question,
                            slot,
                            accepted,
                        ),
                        question=question,
                        passage_text=slot.passage_text,
                    )
                    if not validation.passed:
                        raise ValueError(validation.primary_issue)
                else:
                    slot = slot_by_order[order]
                    vocabulary_spec = build_vocabulary_difficulty_spec(
                        mode=request.mode,
                        difficulty=slot.difficulty.value,
                        complexity_band=slot.complexity_band,
                        skill_tag=slot.skill_tag,
                        question_type=question.question_type.value,
                        target_expression=question.target_expression or "",
                        usage_intent=slot.usage_intent,
                    )
                    validation = project_vocabulary_validation(
                        vocabulary_spec,
                        lambda: self._validate_candidate(
                            request,
                            question,
                            slot,
                            accepted,
                        ),
                    )
                    if not validation.passed:
                        raise ValueError(validation.primary_issue)
                accepted[order] = question
            except (ValidationError, ValueError) as exc:
                reason = str(exc)
                failures[order] = reason
                logger.warning(
                    "Reading/Vocabulary candidate rejected. request_id=%s order=%d reason=%s",
                    request.request_id,
                    order,
                    reason[:300],
                )
        return failures

    def _validate_b1_inference_candidate_evidence(
        self,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        item: dict[str, Any],
    ) -> None:
        clue = item.get("inferenceClueQuote")
        inference = item.get("unstatedInference")
        passage = slot.passage_text or ""
        if (not isinstance(clue, str) or not clue.strip()
                or clue.strip() not in passage
                or not isinstance(inference, str) or not inference.strip()
                or self._normalize_for_leak_check(inference)
                in self._normalize_for_leak_check(passage)):
            raise ValueError("B1 inference candidate lacks an unstated passage-grounded conclusion")
        if slot.reading_plan is not None and (
            clue.strip() != slot.reading_plan.clue_quote.strip()
            or self._normalize_for_leak_check(inference)
            != self._normalize_for_leak_check(slot.reading_plan.unstated_inference or "")
        ):
            raise ValueError("B1 inference candidate changed its private passage plan")
        self._assert_language_lane(
            request.learning_language, [inference],
            reason="B1 candidate inference is not in learningLanguage",
        )

    @staticmethod
    def _candidate_schema_for(
        request: PracticeGenerationRequest,
        slots: list[_QuestionSlot] | None = None,
    ) -> dict[str, Any]:
        if (request.domain == PracticeDomain.READING
                and request.mode == ReadingMode.COMPREHENSION.value
                and request.complexity_band == 1):
            return _B1_COMPREHENSION_CANDIDATE_SCHEMA
        if (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value
        ):
            return _CONTEXTUAL_CHOICE_CANDIDATE_SCHEMA
        if (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.MEANING_RELATION.value
        ):
            if slots and all(slot.target_as_answer for slot in slots):
                return _HIGH_BAND_MEANING_RELATION_CANDIDATE_SCHEMA
            return _MEANING_RELATION_CANDIDATE_SCHEMA
        return _PRACTICE_CANDIDATE_SCHEMA

    def _normalize_contextual_choice_v2_candidate(
        self,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        item: dict[str, Any],
        accepted: dict[int, PracticeGeneratedQuestion],
    ) -> PracticeGeneratedQuestion:
        target_value = item.get("targetExpression")
        if not isinstance(target_value, str) or not target_value.strip():
            raise ValueError("CONTEXTUAL_CHOICE requires targetExpression")
        target_expression = target_value.strip()
        if slot.review_target:
            if target_expression != slot.review_expression:
                raise ValueError(
                    "review slot targetExpression does not match bound review target"
                )
            if slot.review_canonical_key is None:
                raise ValueError("review slot requires a bound canonicalKey")
            canonical_key = slot.review_canonical_key
        else:
            canonical_key = self._canonical_key_for_expression(target_expression)

        self._assert_vocabulary_surface_language(
            request.learning_language,
            target_expression,
            reason="CONTEXTUAL_CHOICE targetExpression is not in learningLanguage",
        )
        complete_sentence = item.get("completeSentence")
        if not isinstance(complete_sentence, str):
            raise ValueError("CONTEXTUAL_CHOICE requires completeSentence")
        complete_sentence = complete_sentence.strip()
        self._assert_language_lane(
            request.learning_language,
            [complete_sentence],
            reason="CONTEXTUAL_CHOICE completeSentence is not in learningLanguage",
        )
        prompt = render_contextual_choice_prompt(
            complete_sentence,
            target_expression,
        )

        raw_primary = item.get("distractors")
        if not isinstance(raw_primary, list) or len(raw_primary) != 3:
            raise ValueError("CONTEXTUAL_CHOICE requires exactly three distractors")
        selected: list[PracticeOption] = []
        normalized_target = self._normalize_for_leak_check(target_expression)
        seen_texts = {normalized_target}
        for candidate_index, raw_option in enumerate(raw_primary, start=1):
            option = self._wrong_option_candidate(
                raw_option,
                fallback_key=f"candidate-{candidate_index}",
            )
            self._assert_vocabulary_surface_language(
                request.learning_language,
                option.text,
                reason="CONTEXTUAL_CHOICE distractor is not in learningLanguage",
            )
            normalized_text = self._normalize_for_leak_check(option.text)
            if not normalized_text or normalized_text in seen_texts:
                raise ValueError(
                    "CONTEXTUAL_CHOICE distractors must be distinct from target and each other"
                )
            seen_texts.add(normalized_text)
            selected.append(option)

        global_order = request.question_offset + slot.order
        correct_key = contextual_choice_answer_key(global_order)
        correct_position = CONTEXTUAL_CHOICE_ANSWER_KEYS.index(correct_key)
        final_options = list(selected)
        final_options.insert(
            correct_position,
            PracticeOption(key="server-correct", text=target_expression),
        )
        normalized_item = {
            "order": slot.order,
            "questionType": PracticeQuestionType.SINGLE_CHOICE.value,
            "difficulty": slot.difficulty.value,
            "complexityBand": slot.complexity_band,
            "passageId": None,
            "passageText": None,
            "prompt": prompt,
            "options": [
                {"key": key, "text": option.text}
                for key, option in zip(
                    CONTEXTUAL_CHOICE_ANSWER_KEYS,
                    final_options,
                    strict=True,
                )
            ],
            "correctAnswer": [correct_key],
            "skillTag": slot.skill_tag,
            "evidenceText": None,
            "explanationLearning": item.get("explanationLearning"),
            "targetExpression": target_expression,
            "canonicalKey": canonical_key,
            "reviewTarget": slot.review_target,
            "vocabularyCandidates": [],
        }
        question = PracticeGeneratedQuestion.model_validate(
            {**normalized_item, "explanationOrigin": "PENDING"}
        )
        vocabulary_spec = build_vocabulary_difficulty_spec(
            mode=request.mode,
            difficulty=slot.difficulty.value,
            complexity_band=slot.complexity_band,
            skill_tag=slot.skill_tag,
            question_type=question.question_type.value,
            target_expression=question.target_expression or "",
            usage_intent=slot.usage_intent,
        )
        validation = project_vocabulary_validation(
            vocabulary_spec,
            lambda: self._validate_candidate(
                request,
                question,
                slot,
                accepted,
            ),
        )
        if not validation.passed:
            raise ValueError(validation.primary_issue)
        return question

    @classmethod
    def _normalize_meaning_relation_candidate(
        cls,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        if request.mode != VocabularyMode.MEANING_RELATION.value:
            return item

        normalized = {
            key: value
            for key, value in item.items()
            if key not in {"meaningContext", "distractors", "reserveDistractors"}
        }
        target_as_answer = slot.target_as_answer
        context_required = cls._meaning_relation_context_required(
            complexity_band=slot.complexity_band,
            skill_tag=slot.skill_tag,
            question_type=slot.question_type,
        )
        if context_required:
            meaning_context = item.get("meaningContext")
            if not isinstance(meaning_context, str):
                raise ValueError("MEANING_RELATION B3+ requires string meaningContext")
            target_expression = item.get("targetExpression")
            if not isinstance(target_expression, str):
                raise ValueError("MEANING_RELATION task shell requires targetExpression")
            normalized_target = cls._normalize_for_leak_check(target_expression)
            if (
                target_as_answer
                and normalized_target
                and normalized_target
                in cls._normalize_for_leak_check(meaning_context)
            ):
                raise ValueError(
                    "MEANING_RELATION target-as-answer context leaks targetExpression"
                )
            normalized["prompt"] = render_meaning_relation_prompt(
                learning_language=request.learning_language,
                skill_tag=slot.skill_tag,
                target_expression=target_expression,
                meaning_context=meaning_context,
                target_as_answer=target_as_answer,
            )
        if slot.skill_tag == VocabularySkill.DISTINCTION.value:
            if target_as_answer:
                normalized = cls._assemble_high_band_meaning_relation_options(
                    request=request,
                    slot=slot,
                    item=item,
                    normalized=normalized,
                )
            else:
                normalized = cls._assemble_meaning_relation_distinction_options(
                    request=request,
                    slot=slot,
                    item=item,
                    normalized=normalized,
                )
        return normalized

    @classmethod
    def _assemble_high_band_meaning_relation_options(
        cls,
        *,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        item: dict[str, Any],
        normalized: dict[str, Any],
    ) -> dict[str, Any]:
        generated_target = item.get("targetExpression")
        if not isinstance(generated_target, str) or not generated_target.strip():
            raise ValueError("MEANING_RELATION task shell requires targetExpression")
        target_expression = generated_target.strip()
        normalized["targetExpression"] = target_expression
        if slot.review_target:
            if target_expression != slot.review_expression:
                raise ValueError(
                    "review slot targetExpression does not match bound review target"
                )
            if item.get("canonicalKey") != slot.review_canonical_key:
                raise ValueError(
                    "review slot canonicalKey does not match bound review target"
                )
            assert slot.review_expression is not None
            target_expression = slot.review_expression
            normalized["targetExpression"] = target_expression
            normalized["canonicalKey"] = slot.review_canonical_key
        cls._assert_vocabulary_surface_language(
            request.learning_language,
            target_expression,
            reason="MEANING_RELATION targetExpression is not in learningLanguage",
        )

        # Dedicated B3+ batches use `distractors`. A mixed B2/B3 batch keeps the legacy
        # response schema so call-count parity is preserved; for that compatibility shape all
        # `options` are interpreted as wrong-only candidates and `correctAnswer` is ignored.
        raw_primary = item.get("distractors")
        if raw_primary is None:
            raw_primary = item.get("options")
        raw_reserves = item.get("reserveDistractors", [])
        if not isinstance(raw_primary, list) or not isinstance(raw_reserves, list):
            raise ValueError("MEANING_RELATION distractor candidates must be arrays")

        normalized_target = cls._normalize_for_leak_check(target_expression)
        selected: list[tuple[PracticeOption, bool]] = []
        seen_texts = {normalized_target}
        filtered_target_identity = 0
        filtered_duplicate = 0
        filtered_language_or_schema = 0
        candidates = [
            (raw_option, False) for raw_option in raw_primary
        ] + [(raw_option, True) for raw_option in raw_reserves]
        for candidate_index, (raw_option, is_reserve) in enumerate(candidates, start=1):
            try:
                option = cls._wrong_option_candidate(
                    raw_option,
                    fallback_key=f"candidate-{candidate_index}",
                )
                cls._assert_vocabulary_surface_language(
                    request.learning_language,
                    option.text,
                    reason="MEANING_RELATION distractor candidate is not in learningLanguage",
                )
            except (ValidationError, ValueError):
                filtered_language_or_schema += 1
                continue
            normalized_text = cls._normalize_for_leak_check(option.text)
            if normalized_text == normalized_target:
                filtered_target_identity += 1
                continue
            if not normalized_text or normalized_text in seen_texts:
                filtered_duplicate += 1
                continue
            seen_texts.add(normalized_text)
            selected.append((option, is_reserve))
            if len(selected) == 3:
                break

        if filtered_target_identity:
            logger.info(
                "MEANING_RELATION target identity distractor filtered. "
                "request_id=%s order=%d target_identity_role=distractor count=%d",
                request.request_id,
                slot.order,
                filtered_target_identity,
            )
        if len(selected) < 3:
            raise ValueError(
                "MEANING_RELATION requires three valid distinct distractors after salvage"
            )

        global_order = request.question_offset + slot.order
        correct_position = (global_order - 1) % len(_SINGLE_CHOICE_KEYS)
        final_options = [option for option, _ in selected]
        final_options.insert(
            correct_position,
            PracticeOption(key="server-correct", text=target_expression),
        )
        normalized["options"] = [
            {"key": key, "text": option.text}
            for key, option in zip(_SINGLE_CHOICE_KEYS, final_options, strict=True)
        ]
        normalized["correctAnswer"] = [_SINGLE_CHOICE_KEYS[correct_position]]
        if (
            filtered_target_identity
            or filtered_duplicate
            or filtered_language_or_schema
            or any(is_reserve for _, is_reserve in selected)
        ):
            logger.info(
                "MEANING_RELATION distractor salvage assembled. request_id=%s order=%d "
                "answer_authority=application_target_expression "
                "filtered_target_identity=%d filtered_duplicate=%d "
                "filtered_language_or_schema=%d reserve_used=%d",
                request.request_id,
                slot.order,
                filtered_target_identity,
                filtered_duplicate,
                filtered_language_or_schema,
                sum(is_reserve for _, is_reserve in selected),
            )
        return normalized

    @staticmethod
    def _wrong_option_candidate(
        raw_option: Any,
        *,
        fallback_key: str,
    ) -> PracticeOption:
        if not isinstance(raw_option, dict):
            raise ValueError("MEANING_RELATION wrong candidate must be an object")
        return PracticeOption.model_validate(
            {
                "key": raw_option.get("key") or fallback_key,
                "text": raw_option.get("text"),
            }
        )

    @classmethod
    def _assemble_meaning_relation_distinction_options(
        cls,
        *,
        request: PracticeGenerationRequest,
        slot: _QuestionSlot,
        item: dict[str, Any],
        normalized: dict[str, Any],
    ) -> dict[str, Any]:
        raw_options = item.get("options")
        raw_reserves = item.get("reserveDistractors", [])
        correct_answer = item.get("correctAnswer")
        if not isinstance(raw_options, list) or not isinstance(raw_reserves, list):
            raise ValueError("MEANING_RELATION distractor candidates must be arrays")
        if (
            not isinstance(correct_answer, list)
            or len(correct_answer) != 1
            or not isinstance(correct_answer[0], str)
        ):
            raise ValueError("MEANING_RELATION requires one correct candidate key")

        correct_key = correct_answer[0]
        correct_indexes = [
            index
            for index, raw_option in enumerate(raw_options)
            if isinstance(raw_option, dict) and raw_option.get("key") == correct_key
        ]
        if len(correct_indexes) != 1:
            raise ValueError("MEANING_RELATION correct candidate key is missing or duplicated")
        correct_index = correct_indexes[0]
        try:
            correct_option = PracticeOption.model_validate(raw_options[correct_index])
            cls._assert_vocabulary_surface_language(
                request.learning_language,
                correct_option.text,
                reason="MEANING_RELATION correct candidate is not in learningLanguage",
            )
        except (ValidationError, ValueError) as exc:
            raise ValueError("MEANING_RELATION correct candidate is structurally invalid") from exc

        normalized_target = cls._normalize_for_leak_check(
            str(item.get("targetExpression") or "")
        )
        normalized_correct = cls._normalize_for_leak_check(correct_option.text)
        if normalized_target and normalized_correct == normalized_target:
            logger.warning(
                "MEANING_RELATION target identity cannot be salvaged. "
                "request_id=%s order=%d target_identity_role=correct",
                request.request_id,
                slot.order,
            )
            raise ValueError("MEANING_RELATION option must not repeat targetExpression")

        candidates = [
            (raw_option, False)
            for index, raw_option in enumerate(raw_options)
            if index != correct_index
        ] + [(raw_option, True) for raw_option in raw_reserves]
        selected: list[tuple[PracticeOption, bool]] = []
        seen_texts = {normalized_correct}
        filtered_target_identity = 0
        filtered_duplicate = 0
        filtered_language_or_schema = 0
        for raw_option, is_reserve in candidates:
            try:
                option = PracticeOption.model_validate(raw_option)
                cls._assert_vocabulary_surface_language(
                    request.learning_language,
                    option.text,
                    reason="MEANING_RELATION distractor candidate is not in learningLanguage",
                )
            except (ValidationError, ValueError):
                filtered_language_or_schema += 1
                continue
            normalized_text = cls._normalize_for_leak_check(option.text)
            if normalized_target and normalized_text == normalized_target:
                filtered_target_identity += 1
                continue
            if not normalized_text or normalized_text in seen_texts:
                filtered_duplicate += 1
                continue
            seen_texts.add(normalized_text)
            selected.append((option, is_reserve))
            if len(selected) == 3:
                break

        if filtered_target_identity:
            logger.info(
                "MEANING_RELATION target identity distractor filtered. "
                "request_id=%s order=%d target_identity_role=distractor count=%d",
                request.request_id,
                slot.order,
                filtered_target_identity,
            )
        if len(selected) < 3:
            raise ValueError(
                "MEANING_RELATION requires three valid distinct distractors after salvage"
            )

        global_order = request.question_offset + slot.order
        correct_position = (global_order - 1) % len(_SINGLE_CHOICE_KEYS)
        selected_options = [option for option, _ in selected]
        selected_options.insert(correct_position, correct_option)
        assembled = [
            {"key": key, "text": option.text}
            for key, option in zip(_SINGLE_CHOICE_KEYS, selected_options, strict=True)
        ]
        normalized["options"] = assembled
        normalized["correctAnswer"] = [_SINGLE_CHOICE_KEYS[correct_position]]
        if (
            filtered_target_identity
            or filtered_duplicate
            or filtered_language_or_schema
            or any(is_reserve for _, is_reserve in selected)
        ):
            logger.info(
                "MEANING_RELATION distractor salvage assembled. request_id=%s order=%d "
                "filtered_target_identity=%d filtered_duplicate=%d "
                "filtered_language_or_schema=%d reserve_used=%d",
                request.request_id,
                slot.order,
                filtered_target_identity,
                filtered_duplicate,
                filtered_language_or_schema,
                sum(is_reserve for _, is_reserve in selected),
            )
        return normalized

    @staticmethod
    def _meaning_relation_context_required(
        *,
        complexity_band: int,
        skill_tag: str,
        question_type: PracticeQuestionType | None,
    ) -> bool:
        if question_type is None:
            raise ValueError("MEANING_RELATION slot requires questionType")
        demand = vocabulary_difficulty_recipe(
            mode=VocabularyMode.MEANING_RELATION.value,
            band=complexity_band,
            skill_tag=skill_tag,
            question_type=question_type.value,
        ).demand
        if not isinstance(demand, MeaningRelationDemand):
            raise ValueError("MEANING_RELATION slot requires meaning-relation demand")
        return demand.context_requirement.endswith("_REQUIRED")

    def _normalize_reading_candidate_metadata(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
        slot: _QuestionSlot,
    ) -> None:
        if request.domain != PracticeDomain.READING:
            return

        # vocabularyCandidates is optional enrichment, never part of the question's correctness.
        # Keep only useful exact surface forms from the assigned passage instead of discarding an
        # otherwise-valid Reading question because Luna returned an inflected/paraphrased form.
        passage_text = slot.passage_text or question.passage_text or ""
        question.vocabulary_candidates = self._sanitize_reading_vocabulary_candidates(
            question.vocabulary_candidates,
            passage_text,
        )
        question.review_target = False

    @staticmethod
    def _sanitize_reading_vocabulary_candidates(
        values: Any,
        passage_text: str,
    ) -> list[str]:
        if not isinstance(values, list):
            return []

        sanitized: list[str] = []
        seen: set[str] = set()
        for raw_value in values:
            if not isinstance(raw_value, str):
                continue
            value = raw_value.strip()
            if not value or value in seen or value not in passage_text:
                continue
            seen.add(value)
            sanitized.append(value)
            if len(sanitized) == 3:
                break
        return sanitized

    def _normalize_usage_distinction_candidate_metadata(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
        slot: _QuestionSlot,
    ) -> None:
        if not (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.USAGE_DISTINCTION.value
        ):
            return

        # These fields are application-owned plan metadata, not learner-authored semantics.
        # Normalize harmless model drift instead of wasting a scarce semantic retry.
        question.difficulty = slot.difficulty
        question.complexity_band = slot.complexity_band
        question.skill_tag = slot.skill_tag
        if slot.question_type is not None:
            question.question_type = slot.question_type
        question.passage_id = None
        question.passage_text = None
        question.evidence_text = None
        question.vocabulary_candidates = []

        target = (question.target_expression or "").strip()
        if slot.review_target:
            question.review_target = True
            # Only canonical metadata is repairable. A different target changes the learning item
            # and must still be rejected by deterministic validation below.
            if target and target == (slot.review_expression or ""):
                question.canonical_key = slot.review_canonical_key
        else:
            question.review_target = False
            if target:
                # New mastery identity is application-owned and derived from the exact expression.
                # Do not let an LLM hallucinated/reused key poison uniqueness or review binding.
                question.canonical_key = self._canonical_key_for_expression(target)

        # targetExpression is the trained answer by contract. If it appears exactly once, bind
        # correctAnswer to that option deterministically; Mini still verifies whether the context
        # actually makes that target the best answer.
        normalized_target = self._normalize_for_leak_check(target)
        target_option_keys = [
            option.key
            for option in question.options
            if self._normalize_for_leak_check(option.text) == normalized_target
        ]
        if normalized_target and len(target_option_keys) == 1:
            question.correct_answer = [target_option_keys[0]]

    def _normalize_composition_candidate_metadata(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
        slot: _QuestionSlot,
    ) -> None:
        if not (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.COMPOSITION.value
        ):
            return

        # Difficulty/skill/question type/review flag are deterministic plan metadata. Repair model
        # drift instead of burning a generation attempt on fields the application already owns.
        question.difficulty = slot.difficulty
        question.complexity_band = slot.complexity_band
        question.skill_tag = slot.skill_tag
        if slot.question_type is not None:
            question.question_type = slot.question_type
        question.passage_id = None
        question.passage_text = None
        question.evidence_text = None
        question.vocabulary_candidates = []
        question.review_target = slot.review_target

        target = (question.target_expression or "").strip()
        if slot.review_target:
            # Review identity may never be invented. Only repair canonical metadata after the model
            # has actually preserved the bound expression; a different target still fails below.
            if target and target == (slot.review_expression or ""):
                question.canonical_key = slot.review_canonical_key
            return

        # For NEW ORDERING, the trained expression is the assembled result. If Luna omitted only
        # targetExpression/canonicalKey but returned a valid permutation, reconstruct the mastery
        # identity deterministically instead of discarding an otherwise-good ordering task.
        if not target and question.question_type == PracticeQuestionType.ORDERING:
            target = self._assembled_ordering_expression(question) or ""
            if target:
                question.target_expression = target

        if target:
            question.canonical_key = self._canonical_key_for_expression(target)

    @staticmethod
    def _assembled_ordering_expression(question: PracticeGeneratedQuestion) -> str | None:
        option_by_key = {option.key: option.text for option in question.options}
        option_keys = list(option_by_key)
        if (
            len(option_keys) != len(set(option_keys))
            or len(question.correct_answer) != len(option_keys)
            or set(question.correct_answer) != set(option_keys)
        ):
            return None
        return "".join(option_by_key[key] for key in question.correct_answer).strip() or None

    @staticmethod
    def _reading_retry_feedback(reason: str) -> str:
        if reason in {
            "question stem presupposition is not supported by passage",
            "stem evidence quote is missing from passage",
        }:
            return (
                "REPAIR_READING_STEM_GROUNDING: keep the exact passage and fixed skill. "
                "Remove or correct every stem premise not entailed by the passage; a locally "
                "supported answer is insufficient. Cite exact passage text for the revised stem."
            )
        if reason in {
            "question repeats a previous reading judgment",
            "reading candidate repeats a normalized prompt on the same passage",
        }:
            return (
                "REPAIR_READING_DUPLICATE: keep the exact passage and fixed skill, but ask a "
                "materially different reading judgment from previousQuestions, not a paraphrase "
                "of the same discourse relation or answer."
            )
        if reason == "inference question only requires direct retrieval":
            return (
                "REPAIR_READING_INFERENCE: keep the passage and fixed inference skill. "
                "Require a supported inference from the visible context rather than copying "
                "a fact explicitly stated in one sentence."
            )
        if reason == "structure question does not require discourse reasoning":
            return (
                "REPAIR_READING_STRUCTURE: keep the passage and fixed STRUCTURE skill. "
                "Ask about the role or relationship of text units, not merely the cause of an event."
            )
        if reason == "ambiguous single-choice item":
            return (
                "REPAIR_READING_AMBIGUITY: keep the exact assigned passage. Rewrite only the "
                "question/options so one answer is uniquely supported by a visible passage cue; "
                "replace any rival that could also be accepted."
            )
        if reason == "distractors are too weak or unrelated":
            return (
                "REPAIR_READING_DISTRACTORS: keep the exact assigned passage and intended skill. "
                "Use plausible passage-related distractors that require careful reading to reject, "
                "while preserving exactly one clearly supported answer."
            )
        if reason == "answer is not sufficiently supported":
            return (
                "REPAIR_READING_SUPPORT: keep the exact assigned passage. Rebuild the prompt and "
                "options around evidence that is explicitly present or safely inferable from that passage."
            )
        if reason.startswith("answer mismatch"):
            return (
                "REPAIR_READING_ANSWER: keep the exact assigned passage. Rebuild the question/options "
                "so the declared answer is the single best answer under the passage evidence."
            )
        if reason == "question does not fit requested mode/skill":
            return (
                "REPAIR_READING_MODE: keep the exact assigned passage and fixed skillTag. Rebuild the "
                "question to test that reading skill rather than vocabulary recall or outside knowledge."
            )
        if reason == "semantic verifier detected answer leakage":
            return (
                "REPAIR_READING_LEAKAGE: keep the exact assigned passage but remove wording that "
                "directly gives away the answer; preserve the same reading skill."
            )
        return (
            "REPAIR_READING_CANDIDATE: keep the exact assigned passageId/passageText and fixed slot "
            f"metadata, then repair only this question: {reason}. vocabularyCandidates are optional; "
            "return [] unless each value is copied exactly from passageText."
        )

    @staticmethod
    def _composition_retry_feedback(reason: str, slot: _QuestionSlot) -> str:
        if reason == "ordering answer must use each option exactly once":
            return (
                "REPAIR_ORDERING_KEYS: keep this fixed ORDERING slot. Return 3-8 unique chunks "
                "and correctAnswer containing every option key exactly once, with no duplicate, "
                "missing, or extra key."
            )
        if reason == "ordering answer length mismatch":
            return (
                "REPAIR_ORDERING_LENGTH: keep this fixed ORDERING slot. correctAnswer must have "
                "exactly the same number of keys as options and use every chunk once."
            )
        if reason == "vocabulary question requires targetExpression/canonicalKey":
            if slot.question_type == PracticeQuestionType.ORDERING:
                return (
                    "REPAIR_ORDERING_TARGET: targetExpression is mandatory. For a new ORDERING "
                    "slot, use the exact assembled expression; for a review slot, use the exact "
                    "bound review expression/canonicalKey."
                )
            return "REPAIR_TARGET: return the exact vocabulary targetExpression for this slot."
        if reason == "ambiguous single-choice item":
            return (
                "REPAIR_AMBIGUITY: keep the same targetExpression and rebuild the completion/"
                "collocation context so exactly one option is clearly best. Replace any tied rival."
            )
        if reason == "distractors are too weak or unrelated":
            return (
                "REPAIR_DISTRACTORS: keep the same targetExpression. Use plausible same-neighborhood "
                "completion/collocation rivals that are wrong for a visible lexical or grammatical cue."
            )
        if reason.startswith("answer mismatch"):
            return (
                "REPAIR_ANSWER_MISMATCH: keep the same targetExpression and rebuild the completion "
                "context/options until the target-aligned answer is the single best choice."
            )
        if "duplicate" in reason or "reused a review" in reason:
            return (
                "REPAIR_DUPLICATE_TARGET: generate a genuinely different targetExpression for this "
                "new slot; do not reuse any accepted or review expression."
            )
        return f"REPAIR_COMPOSITION: fix this slot without changing its planned type/skill: {reason}"

    @staticmethod
    def _canonical_key_for_expression(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).strip().casefold()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized[:200]

    @staticmethod
    def _usage_retry_feedback(reason: str) -> str:
        if reason == "ambiguous single-choice item":
            return (
                "REPAIR_AMBIGUITY: keep the same targetExpression. Rewrite the learner-visible "
                "context so one explicit semantic/register/collocation cue makes the target clearly "
                "better than every rival, and replace any tied rival option."
            )
        if reason == "distractors are too weak or unrelated":
            return (
                "REPAIR_DISTRACTORS: keep the same targetExpression. Replace all weak/unrelated "
                "distractors with same-part-of-speech competitors from the same usage neighborhood; "
                "each should be plausible in isolation but clearly lose on a visible context cue."
            )
        if reason.startswith("answer mismatch"):
            return (
                "REPAIR_ANSWER_MISMATCH: keep the same targetExpression. The independent verifier "
                "preferred another option, so strengthen the decisive context and/or replace that "
                "rival until the target is unmistakably the single best answer."
            )
        if reason == "USAGE_DISTINCTION does not require context":
            return (
                "REPAIR_CONTEXT: keep the same targetExpression. Add a learner-visible cue required "
                "to choose among plausible alternatives. For collocation the local lexical frame is "
                "enough; for register show relationship/formality/channel; otherwise add semantic or "
                "pragmatic situation cues."
            )
        if reason == "semantic verifier detected answer leakage":
            return (
                "REPAIR_LEAKAGE: keep the same targetExpression but remove direct quotation, "
                "definition, synonym/paraphrase hints, or wording that gives the answer away."
            )
        if reason == "question does not fit requested mode/skill":
            return (
                "REPAIR_MODE_FIT: keep the same targetExpression and rebuild the item around the "
                "supplied usageIntent/skillTag rather than meaning recall or generic synonym matching."
            )
        return f"REPAIR_SEMANTIC_QUALITY: keep the same targetExpression and fix: {reason}"

    def _validate_candidate(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
        slot: _QuestionSlot,
        accepted: dict[int, PracticeGeneratedQuestion],
    ) -> None:
        if question.order != slot.order:
            raise ValueError("candidate order mismatch")
        if question.difficulty != slot.difficulty:
            raise ValueError("candidate difficulty mismatch")
        if question.complexity_band != slot.complexity_band:
            raise ValueError("candidate complexityBand mismatch")
        if question.skill_tag != slot.skill_tag:
            raise ValueError("candidate skillTag mismatch")
        if slot.question_type is not None and question.question_type != slot.question_type:
            raise ValueError("candidate questionType mismatch")

        self._validate_question_shape(question)
        self._validate_learning_language(request, question)

        if request.domain == PracticeDomain.READING:
            if question.passage_id != slot.passage_id or question.passage_text != slot.passage_text:
                raise ValueError("reading candidate must reuse the exact planned passage")
            if question.target_expression or question.canonical_key:
                raise ValueError("reading question must not become vocabulary item")
            normalized_candidates = [value.strip() for value in question.vocabulary_candidates]
            if len(normalized_candidates) != len(set(normalized_candidates)):
                raise ValueError("reading vocabularyCandidates must be unique")
            if any(not value for value in normalized_candidates):
                raise ValueError("reading vocabularyCandidates must not contain blanks")
            if any(value not in (question.passage_text or "") for value in normalized_candidates):
                raise ValueError("reading vocabularyCandidates must use passage surface forms")
            normalized_prompt = self._normalize_for_leak_check(
                unicodedata.normalize("NFKC", question.prompt)
            )
            if any(
                previous.passage_id == question.passage_id
                and self._normalize_for_leak_check(
                    unicodedata.normalize("NFKC", previous.prompt)
                ) == normalized_prompt
                for previous in [
                    *request.previous_questions,
                    *(item for order, item in accepted.items() if order != question.order),
                ]
            ):
                raise ValueError("reading candidate repeats a normalized prompt on the same passage")
            return

        if question.passage_id is not None or question.passage_text is not None:
            raise ValueError("vocabulary candidate must not include reading passage")
        if not question.target_expression or not question.canonical_key:
            raise ValueError("vocabulary question requires targetExpression/canonicalKey")
        self._validate_vocabulary_surface_language(request, question)
        if question.vocabulary_candidates:
            raise ValueError("vocabulary question must not return reading vocabularyCandidates")

        accepted_keys = {
            item.canonical_key for item in accepted.values() if item.canonical_key and item.order != question.order
        } | {item.canonical_key for item in request.previous_questions if item.canonical_key}
        if question.canonical_key in accepted_keys:
            raise ValueError("vocabulary canonicalKey duplicates an accepted slot")

        normalized_target = self._normalize_for_leak_check(question.target_expression)
        accepted_targets = {
            self._normalize_for_leak_check(item.target_expression or "")
            for item in accepted.values()
            if item.order != question.order and item.target_expression
        } | {
            self._normalize_for_leak_check(item.target_expression)
            for item in request.previous_questions
            if item.target_expression
        }
        if normalized_target and normalized_target in accepted_targets:
            raise ValueError("vocabulary targetExpression duplicates an accepted slot")

        eligible_review_targets = self._eligible_review_targets(request, log_rejections=False)
        review_target_expressions = {
            self._normalize_for_leak_check(target.expression)
            for target in eligible_review_targets
            if target.expression.strip()
        }
        if (
            not slot.review_target
            and normalized_target
            and normalized_target in review_target_expressions
        ):
            raise ValueError("new vocabulary slot reused a review targetExpression")

        if slot.review_target:
            if not question.review_target:
                raise ValueError("planned review slot must set reviewTarget=true")
            if question.canonical_key != slot.review_canonical_key:
                raise ValueError("review slot canonicalKey does not match bound review target")
            if question.target_expression != slot.review_expression:
                raise ValueError("review slot targetExpression does not match bound review target")
        else:
            if question.review_target:
                raise ValueError("new vocabulary slot must set reviewTarget=false")
            review_keys = {target.canonical_key for target in request.review_targets}
            if question.canonical_key in review_keys:
                raise ValueError("new vocabulary slot reused a review canonicalKey")

        if request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
            self._validate_contextual_choice_candidate(request, question, slot)
        elif request.mode == VocabularyMode.USAGE_DISTINCTION.value:
            self._validate_usage_distinction_candidate(question)
        elif request.mode == VocabularyMode.MEANING_RELATION.value:
            self._validate_meaning_relation_candidate(question)

    def _validate_vocabulary_surface_language(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
    ) -> None:
        assert question.target_expression is not None
        self._assert_vocabulary_surface_language(
            request.learning_language,
            question.target_expression,
            reason=(
                f"vocabulary targetExpression order={question.order} is not a lexical "
                "expression in learningLanguage"
            ),
        )
        for option in question.options:
            self._assert_vocabulary_surface_language(
                request.learning_language,
                option.text,
                reason=(
                    f"vocabulary option order={question.order} key={option.key} is not a lexical "
                    "expression in learningLanguage"
                ),
            )

    @classmethod
    def _assert_vocabulary_surface_language(
        cls,
        language_code: str,
        value: str,
        *,
        reason: str,
    ) -> None:
        text = value.strip()
        if not text:
            raise ValueError(reason)
        language = cls._base_language(language_code)
        has_hangul = bool(re.search(r"[\uac00-\ud7a3]", text))
        has_kana = bool(re.search(r"[\u3040-\u30ff]", text))
        has_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
        has_ascii = bool(re.search(r"[A-Za-z]", text))

        if language == "ja":
            if has_hangul:
                raise ValueError(reason)
            if has_kana or has_cjk:
                return
            if has_ascii and cls._is_allowed_ascii_vocabulary_token(text):
                return
            raise ValueError(reason)
        if language == "ko":
            if has_kana:
                raise ValueError(reason)
            if has_hangul:
                return
            if has_ascii and cls._is_allowed_ascii_vocabulary_token(text):
                return
            raise ValueError(reason)
        if language == "en":
            if has_hangul or has_kana or has_cjk or not has_ascii:
                raise ValueError(reason)
            return
        # Same-script/unknown languages cannot be proven from Unicode ranges. Keep the existing
        # generic language-lane validation as the fallback rather than inventing a false rule.

    @staticmethod
    def _is_allowed_ascii_vocabulary_token(value: str) -> bool:
        compact = value.strip()
        if compact in _ASCII_VOCABULARY_ALLOWLIST:
            return True
        return bool(_ASCII_ACRONYM_RE.fullmatch(compact))

    @staticmethod
    def _base_language(language_code: str) -> str:
        return language_code.strip().lower().split("-")[0].split("_")[0]

    @classmethod
    def _validate_contextual_choice_candidate(
        cls,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
        slot: _QuestionSlot,
    ) -> None:
        if question.question_type != PracticeQuestionType.SINGLE_CHOICE:
            raise ValueError("CONTEXTUAL_CHOICE must use single-choice questions")
        if tuple(option.key for option in question.options) != CONTEXTUAL_CHOICE_ANSWER_KEYS:
            raise ValueError("CONTEXTUAL_CHOICE options must use A/B/C/D in order")
        if question.prompt.count(CONTEXTUAL_CHOICE_BLANK) != 1:
            raise ValueError("CONTEXTUAL_CHOICE must contain exactly one blank")

        normalized_target = cls._normalize_for_leak_check(
            question.target_expression or ""
        )
        if not normalized_target:
            raise ValueError("CONTEXTUAL_CHOICE requires targetExpression")
        if normalized_target in cls._normalize_for_leak_check(question.prompt):
            raise ValueError("CONTEXTUAL_CHOICE prompt leaks targetExpression")
        target_options = [
            option
            for option in question.options
            if cls._normalize_for_leak_check(option.text) == normalized_target
        ]
        if len(target_options) != 1:
            raise ValueError(
                "CONTEXTUAL_CHOICE targetExpression must appear in exactly one option"
            )
        expected_key = contextual_choice_answer_key(
            request.question_offset + slot.order
        )
        if question.correct_answer != [expected_key]:
            raise ValueError("CONTEXTUAL_CHOICE correct answer rotation mismatch")
        if target_options[0].key != expected_key:
            raise ValueError(
                "CONTEXTUAL_CHOICE targetExpression must be the application-owned answer"
            )

    @classmethod
    def _validate_meaning_relation_candidate(
        cls,
        question: PracticeGeneratedQuestion,
    ) -> None:
        if question.question_type != PracticeQuestionType.SINGLE_CHOICE:
            raise ValueError("MEANING_RELATION must use single-choice questions")
        option_by_key = {option.key: option.text for option in question.options}
        correct_text = option_by_key.get(question.correct_answer[0], "")
        normalized_correct = cls._normalize_for_leak_check(correct_text)
        normalized_target = cls._normalize_for_leak_check(
            question.target_expression or ""
        )
        target_as_answer = (
            question.complexity_band >= 3
            and question.skill_tag == VocabularySkill.DISTINCTION.value
        )
        if target_as_answer:
            target_option_count = sum(
                cls._normalize_for_leak_check(option_text) == normalized_target
                for option_text in option_by_key.values()
            )
            if not normalized_target or target_option_count != 1:
                raise ValueError(
                    "MEANING_RELATION target-as-answer must appear in exactly one option"
                )
            if normalized_correct != normalized_target:
                raise ValueError(
                    "MEANING_RELATION target-as-answer must be the correct option"
                )
            stem = cls._normalize_for_leak_check(question.prompt)
            if normalized_target in stem:
                raise ValueError(
                    "MEANING_RELATION target-as-answer stem leaks targetExpression"
                )
            return

        if normalized_target and any(
            cls._normalize_for_leak_check(option_text) == normalized_target
            for option_text in option_by_key.values()
        ):
            raise ValueError(
                "MEANING_RELATION option must not repeat targetExpression"
            )
        stem = cls._normalize_for_leak_check(question.prompt)
        if len(normalized_correct) >= 4 and normalized_correct in stem:
            raise ValueError("MEANING_RELATION stem leaks correct option text")

    @classmethod
    def _validate_usage_distinction_candidate(cls, question: PracticeGeneratedQuestion) -> None:
        if question.question_type != PracticeQuestionType.SINGLE_CHOICE:
            raise ValueError("USAGE_DISTINCTION must use single-choice questions")
        target = cls._normalize_for_leak_check(question.target_expression or "")
        stem = cls._normalize_for_leak_check(question.prompt)
        if target and target in stem:
            raise ValueError("USAGE_DISTINCTION stem leaks targetExpression")

        option_by_key = {option.key: option.text for option in question.options}
        correct_text = option_by_key.get(question.correct_answer[0], "")
        normalized_correct = cls._normalize_for_leak_check(correct_text)
        if target != normalized_correct:
            raise ValueError("USAGE_DISTINCTION correct option must equal targetExpression")
        if len(normalized_correct) >= 4 and normalized_correct in stem:
            raise ValueError("USAGE_DISTINCTION stem leaks correct option text")

        semantic_relation_patterns = (
            r"同じ意味",
            r"最も近い意味",
            r"近い意味",
            r"言い換え",
            r"意味(?:は|として)",
            r"같은\s*의미",
            r"가장\s*가까운\s*의미",
            r"(?:뜻|의미)(?:은|는)",
            r"same\s+meaning",
            r"closest\s+meaning",
            r"synonym",
        )
        if any(re.search(pattern, question.prompt, re.IGNORECASE) for pattern in semantic_relation_patterns):
            raise ValueError("USAGE_DISTINCTION became a meaning/synonym question")

    @staticmethod
    def _normalize_for_leak_check(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7a3]+", "", value).casefold()

    @classmethod
    def _normalize_contextual_choice_repair_identity(cls, value: str) -> str:
        return cls._normalize_for_leak_check(unicodedata.normalize("NFKC", value))

    @staticmethod
    def _validate_question_shape(question: PracticeGeneratedQuestion) -> None:
        option_keys = [option.key for option in question.options]
        option_texts = [option.text.strip().casefold() for option in question.options]
        if len(option_keys) != len(set(option_keys)):
            raise ValueError("duplicate option keys")
        if len(option_texts) != len(set(option_texts)):
            raise ValueError("duplicate option texts")
        if question.question_type == PracticeQuestionType.SINGLE_CHOICE:
            if len(question.options) != 4 or len(question.correct_answer) != 1:
                raise ValueError("single choice must have 4 options and one answer")
            if question.correct_answer[0] not in option_keys:
                raise ValueError("single choice answer key missing from options")
            return
        if len(question.options) < 3 or len(question.options) > 8:
            raise ValueError("ordering must have 3-8 chunks")
        if len(question.correct_answer) != len(option_keys):
            raise ValueError("ordering answer length mismatch")
        if set(question.correct_answer) != set(option_keys):
            raise ValueError("ordering answer must use each option exactly once")

    def _validate_learning_language(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
    ) -> None:
        values = [
            question.passage_text,
            question.prompt,
            question.target_expression,
            question.evidence_text,
            question.explanation_learning,
            *[option.text for option in question.options],
        ]
        self._assert_language_lane(
            request.learning_language,
            [value for value in values if value and value.strip()],
            reason=f"question order={question.order} learner content is not in learningLanguage",
        )

    async def _usage_prescreen_failures(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
    ) -> dict[int, str]:
        if not (
            request.domain == PracticeDomain.VOCABULARY
            and request.mode == VocabularyMode.USAGE_DISTINCTION.value
        ):
            return {}
        single_choice = [
            question for question in questions
            if question.question_type == PracticeQuestionType.SINGLE_CHOICE
        ]
        if not single_choice:
            return {}
        payload_questions = [
            {
                "order": question.order,
                "prompt": question.prompt,
                "options": [option.model_dump(by_alias=True) for option in question.options],
                "skillTag": question.skill_tag,
                **(
                    {"usageIntent": self._usage_intent_for_skill(question.skill_tag)}
                    if request.mode == VocabularyMode.USAGE_DISTINCTION.value
                    else {}
                ),
            }
            for question in single_choice
        ]
        try:
            raw = await asyncio.wait_for(
                self.provider.call(
                    type_name=self.PRESCREEN_TYPE_NAME,
                    data=build_usage_prescreen_prompt(request, payload_questions),
                    schema=_USAGE_PRESCREEN_SCHEMA,
                ),
                timeout=self.timeout_seconds,
            )
            payload = _UsagePrescreenPayload.model_validate(raw)
            verdict_by_order = {verdict.order: verdict for verdict in payload.verdicts}
            expected_orders = {question.order for question in single_choice}
            if set(verdict_by_order) != expected_orders:
                raise ValueError("usage prescreen verdict coverage mismatch")
        except Exception as exc:
            # Nano is a cheap quality gate, not an availability dependency. If it is unavailable,
            # fail open to the Mini verifier rather than failing the whole daily set.
            logger.warning(
                "Reading/Vocabulary Nano prescreen unavailable. request_id=%s type=%s reason=%s",
                request.request_id,
                type(exc).__name__,
                str(exc)[:300],
            )
            return {}

        failures: dict[int, str] = {}
        for order, verdict in verdict_by_order.items():
            reasons: list[str] = []
            if not verdict.modeFit:
                reasons.append("not a usage-distinction task")
            if verdict.answerLeakage:
                reasons.append("answer/target leakage")
            if not verdict.contextDependent:
                reasons.append("context is not required")
            if reasons:
                failures[order] = "; ".join(reasons) + f" ({verdict.reason})"
        return failures

    @staticmethod
    def _is_reading_distractor_only_failure(
        verdict: _PracticeVerificationVerdict,
        *,
        expected_answer_key: str,
    ) -> bool:
        return (
            (
                not isinstance(verdict, _ReadingPracticeVerificationVerdict)
                or (
                    verdict.stemPresuppositionsSupported
                    and verdict.distinctReadingTask
                    and bool(verdict.stemEvidenceSpanIds)
                )
            )
            and verdict.bestAnswerKey == expected_answer_key
            and not verdict.ambiguous
            and verdict.supported
            and verdict.modeFit
            and not verdict.answerLeakage
            and not verdict.distractorsPlausible
        )

    @staticmethod
    def _meaning_relation_retry_feedback(reason: str, slot: _QuestionSlot) -> str:
        if not slot.target_as_answer:
            return reason
        return (
            "REPAIR_TARGET_AS_ANSWER: preserve the exact retryTargetExpression; it is the "
            "application-owned correct answer. Revise meaningContext and wrong distractors so "
            "that target alone is the unique best answer, and never expose it in meaningContext. "
            f"Quality failure: {reason}"
        )

    async def _repair_reading_distractors(
        self,
        request: PracticeGenerationRequest,
        question: PracticeGeneratedQuestion,
        slot: _QuestionSlot,
        accepted: dict[int, PracticeGeneratedQuestion],
    ) -> PracticeGeneratedQuestion | None:
        try:
            if (
                question.question_type != PracticeQuestionType.SINGLE_CHOICE
                or len(question.correct_answer) != 1
                or slot.passage_id is None
                or slot.passage_text is None
                or slot.reading_mode is None
            ):
                raise ValueError("distractor repair requires a passage-bound single-choice slot")
            correct_key = question.correct_answer[0]
            option_by_key = {option.key: option for option in question.options}
            correct_option = option_by_key.get(correct_key)
            if correct_option is None:
                raise ValueError("distractor repair correct key is missing from options")
            wrong_options = [
                option for option in question.options if option.key != correct_key
            ]
            raw = await asyncio.wait_for(
                self.provider.call(
                    type_name=self.DISTRACTOR_REPAIR_TYPE_NAME,
                    data=build_reading_distractor_repair_prompt(
                        request,
                        passage_id=slot.passage_id,
                        passage_text=slot.passage_text,
                        prompt=question.prompt,
                        correct_option=correct_option.model_dump(by_alias=True),
                        wrong_options=[
                            option.model_dump(by_alias=True) for option in wrong_options
                        ],
                        evidence_text=question.evidence_text,
                        skill_tag=slot.skill_tag,
                        difficulty=slot.difficulty.value,
                        complexity_band=slot.complexity_band,
                        question_type=question.question_type.value,
                        difficulty_recipe=question_demand_recipe(
                            slot.complexity_band,
                            mode=slot.reading_mode,
                            skill_tag=slot.skill_tag,
                        ).generation_payload(),
                    ),
                    schema=_READING_DISTRACTOR_REPAIR_SCHEMA,
                ),
                timeout=self._generation_timeout_for(request),
            )
            repair = _ReadingDistractorRepairPayload.model_validate(raw)
            repaired = self._reassemble_reading_distractors(question, repair)
            self._validate_candidate(request, repaired, slot, accepted)
            return repaired
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Reading distractor repair provider/schema/validation failure. "
                "request_id=%s order=%d type=%s",
                request.request_id,
                question.order,
                type(exc).__name__,
            )
            return None

    @classmethod
    def _reassemble_reading_distractors(
        cls,
        question: PracticeGeneratedQuestion,
        repair: _ReadingDistractorRepairPayload,
    ) -> PracticeGeneratedQuestion:
        if question.question_type != PracticeQuestionType.SINGLE_CHOICE:
            raise ValueError("distractor repair requires a single-choice question")
        if len(question.correct_answer) != 1:
            raise ValueError("distractor repair requires exactly one correct answer")

        correct_key = question.correct_answer[0]
        option_by_key = {option.key: option for option in question.options}
        correct_option = option_by_key.get(correct_key)
        if correct_option is None:
            raise ValueError("distractor repair correct key is missing from options")
        expected_wrong_keys = set(option_by_key) - {correct_key}
        returned_keys = [option.key for option in repair.wrongOptions]
        if (
            len(returned_keys) != len(expected_wrong_keys)
            or len(returned_keys) != len(set(returned_keys))
            or set(returned_keys) != expected_wrong_keys
        ):
            raise ValueError("distractor repair wrong option keys/cardinality mismatch")

        replacement_by_key = {
            option.key: option.text.strip() for option in repair.wrongOptions
        }
        if any(not text for text in replacement_by_key.values()):
            raise ValueError("distractor repair returned a blank wrong option")
        normalized_wrong = [
            cls._normalize_repair_option_text(text)
            for text in replacement_by_key.values()
        ]
        if len(normalized_wrong) != len(set(normalized_wrong)):
            raise ValueError("distractor repair returned duplicate wrong options")
        if cls._normalize_repair_option_text(correct_option.text) in normalized_wrong:
            raise ValueError("distractor repair copied the correct answer into a wrong option")

        repaired_options = [
            option
            if option.key == correct_key
            else option.model_copy(update={"text": replacement_by_key[option.key]})
            for option in question.options
        ]
        return question.model_copy(update={"options": repaired_options})

    @staticmethod
    def _normalize_repair_option_text(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()

    async def _semantic_verification_outcome(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
        *,
        previous_reading_questions: list[PracticeGeneratedQuestion] | None = None,
    ) -> _SemanticVerificationOutcome:
        single_choice = [question for question in questions if question.question_type == PracticeQuestionType.SINGLE_CHOICE]
        if not single_choice:
            return _SemanticVerificationOutcome(failures={}, verdicts={})
        def reading_demand(question: PracticeGeneratedQuestion) -> dict[str, object] | None:
            demand = question_demand_recipe(
                question.complexity_band,
                mode=request.mode,
                skill_tag=question.skill_tag,
            ).question_demand
            return demand.generation_payload() if demand is not None else None

        verification_input = [
            {
                "order": question.order,
                **({"passageId": question.passage_id} if request.domain == PracticeDomain.READING else {}),
                "passageText": question.passage_text,
                "prompt": question.prompt,
                "options": [option.model_dump(by_alias=True) for option in question.options],
                "skillTag": question.skill_tag,
                **({"questionDemand": reading_demand(question)} if request.domain == PracticeDomain.READING else {}),
                **({"structureJudgmentContract": structure_judgment_contract(
                    band=5,
                    position=(request.question_offset + question.order
                              if request.question_offset + question.order <= 3
                              else request.question_offset + question.order - 3),
                    count=3 if request.question_offset + question.order <= 3 else 2,
                )} if request.domain == PracticeDomain.READING
                    and request.mode == ReadingMode.STRUCTURE.value
                    and question.complexity_band == 5 else {}),
                **({
                    "samePassageTaskPosition": (
                        request.question_offset + question.order
                        if request.question_offset + question.order <= 3
                        else request.question_offset + question.order - 3
                    ),
                    "samePassageTaskCount": (
                        3 if request.question_offset + question.order <= 3 else 2
                    ),
                } if request.domain == PracticeDomain.READING
                    and request.mode == ReadingMode.STRUCTURE.value else {}),
                **(
                    {
                        "reviewTarget": question.review_target,
                    }
                    if request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value
                    else {}
                ),
                **(
                    {"usageIntent": self._usage_intent_for_skill(question.skill_tag)}
                    if request.mode == VocabularyMode.USAGE_DISTINCTION.value
                    else {}
                ),
            }
            for question in single_choice
        ]
        expected_by_order = {
            question.order: question.correct_answer[0] for question in single_choice
        }
        question_by_order = {question.order: question for question in single_choice}
        evidence_spans: list[dict[str, str]] = []
        span_by_id: dict[str, tuple[str, str]] = {}
        if request.domain == PracticeDomain.READING:
            passage_by_id: dict[str, str] = {}
            for question in single_choice:
                passage_id = question.passage_id or ""
                passage_text = question.passage_text or ""
                if (not passage_id or not passage_text
                        or (passage_id in passage_by_id
                            and passage_by_id[passage_id] != passage_text)):
                    raise ValueError("Reading verifier passage binding is invalid")
                passage_by_id[passage_id] = passage_text
            for passage_id, passage_text in passage_by_id.items():
                span_number = 0
                for match in re.finditer(r"[^。！？.!?\r\n]+[。！？.!?]?", passage_text):
                    exact_text = match.group().strip()
                    if not exact_text:
                        continue
                    span_number += 1
                    span_id = f"{passage_id}:s{span_number}"
                    evidence_spans.append({
                        "id": span_id, "passageId": passage_id, "text": exact_text,
                    })
                    span_by_id[span_id] = (passage_id, exact_text)
            if not evidence_spans:
                raise ValueError("Reading verifier passage lacks evidence spans")
        verification_schema = (
            _STRUCTURE_VERIFICATION_SCHEMA
            if request.domain == PracticeDomain.READING
            and request.mode == ReadingMode.STRUCTURE.value
            else _READING_VERIFICATION_SCHEMA
            if request.domain == PracticeDomain.READING
            else _PRACTICE_VERIFICATION_SCHEMA
        )
        if request.domain == PracticeDomain.READING:
            verification_schema = copy.deepcopy(verification_schema)
            verification_schema["properties"]["verdicts"]["items"]["properties"][
                "stemEvidenceSpanIds"
            ]["items"]["enum"] = list(span_by_id)

        last_error: Exception | None = None
        for verification_attempt in (1, 2, 3):
            try:
                raw = await asyncio.wait_for(
                    self.provider.call(
                        type_name=self.VERIFICATION_TYPE_NAME,
                        data=build_practice_verification_prompt(
                            request,
                            verification_input,
                            previous_reading_questions=[
                                *request.previous_questions,
                                *(previous_reading_questions or []),
                            ] if request.domain == PracticeDomain.READING else None,
                            reading_evidence_spans=evidence_spans,
                        ),
                        schema=verification_schema,
                    ),
                    timeout=self.timeout_seconds,
                )
                if request.domain == PracticeDomain.READING:
                    verdicts = cast(
                        list[_PracticeVerificationVerdict],
                        (_StructurePracticeVerificationPayload.model_validate(raw).verdicts
                         if request.mode == ReadingMode.STRUCTURE.value
                         else _ReadingPracticeVerificationPayload.model_validate(raw).verdicts),
                    )
                else:
                    verdicts = _PracticeVerificationPayload.model_validate(raw).verdicts
                verdict_by_order = {verdict.order: verdict for verdict in verdicts}
                if len(verdicts) != len(expected_by_order) or set(verdict_by_order) != set(expected_by_order):
                    raise ValueError("semantic verifier verdict coverage mismatch")
                if request.domain == PracticeDomain.READING:
                    for order, verdict in verdict_by_order.items():
                        assert isinstance(verdict, _ReadingPracticeVerificationVerdict)
                        selected = verdict.stemEvidenceSpanIds
                        passage_id = question_by_order[order].passage_id
                        if (not selected or len(selected) != len(set(selected))
                                or any(span_by_id.get(span_id, (None,))[0] != passage_id
                                       for span_id in selected)):
                            # A verifier binding failure is not a question-quality rejection.
                            raise ValueError("semantic verifier evidence span binding mismatch")
                failures: dict[int, str] = {}
                for order, expected_key in expected_by_order.items():
                    verdict = verdict_by_order[order]
                    if request.domain == PracticeDomain.READING:
                        assert isinstance(verdict, _ReadingPracticeVerificationVerdict)
                        assessment = normalize_reading_semantic_assessment(
                            best_answer_key=verdict.bestAnswerKey,
                            ambiguous=verdict.ambiguous,
                            supported=verdict.supported,
                            mode_fit=verdict.modeFit,
                            answer_leakage=verdict.answerLeakage,
                            distractors_plausible=verdict.distractorsPlausible,
                            stem_presuppositions_supported=verdict.stemPresuppositionsSupported,
                            distinct_reading_task=verdict.distinctReadingTask,
                            reading_operation=verdict.readingOperation,
                        )
                        decision = ReadingSemanticQualityPolicy().decide(
                            assessment=assessment,
                            context=ReadingAcceptanceContext(
                                expected_answer_key=expected_key,
                                skill_tag=question_by_order[order].skill_tag,
                                mode=request.mode,
                                same_passage_task_position=(
                                    request.question_offset + order
                                    if request.question_offset + order <= 3
                                    else request.question_offset + order - 3
                                ) if request.mode == ReadingMode.STRUCTURE.value else None,
                            ),
                        )
                        if decision.action != "ACCEPT":
                            failures[order] = decision.reason
                        elif (request.mode == ReadingMode.STRUCTURE.value
                              and isinstance(verdict, _StructurePracticeVerificationVerdict)
                              and request.question_offset + order not in {3, 5}
                              and not verdict.boundedStructureScope):
                            failures[order] = "structure question consumes future passage tasks"
                    else:
                        if (
                            request.mode == VocabularyMode.MEANING_RELATION.value
                            and not verdict.contextDependent
                        ):
                            logger.info(
                                "Vocabulary MEANING_RELATION context diagnostic. "
                                "request_id=%s order=%d context_dependent=false",
                                request.request_id,
                                order,
                            )
                        assessment = normalize_vocabulary_semantic_assessment(
                            best_answer_key=verdict.bestAnswerKey,
                            ambiguous=verdict.ambiguous,
                            supported=verdict.supported,
                            mode_fit=verdict.modeFit,
                            answer_leakage=verdict.answerLeakage,
                            context_dependent=verdict.contextDependent,
                            distractors_plausible=verdict.distractorsPlausible,
                        )
                        decision = semantic_quality_policy_for_vocabulary_mode(
                            request.mode
                        ).decide(
                            assessment=assessment,
                            context=VocabularyAcceptanceContext(
                                expected_answer_key=expected_key,
                            ),
                        )
                        if decision.action != "ACCEPT":
                            failures[order] = decision.reason
                return _SemanticVerificationOutcome(
                    failures=failures,
                    verdicts=verdict_by_order,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Reading/Vocabulary semantic verifier failure. request_id=%s attempt=%d/3 type=%s",
                    request.request_id,
                    verification_attempt,
                    type(exc).__name__,
                )
        raise ValueError(
            "semantic verifier unavailable after provider retries: "
            f"{type(last_error).__name__ if last_error else 'unknown'}"
        )

    async def _semantic_failures(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
    ) -> dict[int, str]:
        return (
            await self._semantic_verification_outcome(request, questions)
        ).failures

    async def _attach_origin_explanations(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
        *,
        telemetry: _ContextualChoiceTelemetry | None = None,
    ) -> list[PracticeGeneratedQuestion]:
        by_order = {question.order: question for question in questions}
        explanations: dict[int, str] = {}
        learning_explanations: dict[int, str] = {}
        learning_repairs_required = {
            question.order
            for question in questions
            if (
                request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value
                and self._contextual_choice_explanation_learning_requires_repair(
                    question.explanation_learning
                )
            )
        }
        attempts = {question.order: 0 for question in questions}

        def complete(order: int) -> bool:
            return (
                order in explanations
                and (
                    order not in learning_repairs_required
                    or order in learning_explanations
                )
            )

        while not all(complete(order) for order in by_order):
            pending = [
                order
                for order in sorted(by_order)
                if not complete(order)
                and attempts[order] < _MAX_ORIGIN_EXPLANATION_ATTEMPTS
            ]
            if not pending:
                break
            for start in range(0, len(pending), _ORIGIN_EXPLANATION_BATCH_SIZE):
                batch_orders = pending[start : start + _ORIGIN_EXPLANATION_BATCH_SIZE]
                for order in batch_orders:
                    attempts[order] += 1
                await self._generate_origin_explanation_batch(
                    request,
                    [by_order[order] for order in batch_orders],
                    explanations,
                    learning_explanations,
                    learning_repairs_required,
                    type_name=self.ORIGIN_EXPLANATION_TYPE_NAME,
                    telemetry=telemetry,
                )

        remaining = [order for order in sorted(by_order) if not complete(order)]
        if remaining:
            logger.warning(
                "Reading/Vocabulary Nano explanation exhausted; using Mini fallback. request_id=%s orders=%s",
                request.request_id,
                remaining,
            )
            for start in range(0, len(remaining), _ORIGIN_EXPLANATION_BATCH_SIZE):
                batch_orders = remaining[start : start + _ORIGIN_EXPLANATION_BATCH_SIZE]
                await self._generate_origin_explanation_batch(
                    request,
                    [by_order[order] for order in batch_orders],
                    explanations,
                    learning_explanations,
                    learning_repairs_required,
                    type_name=self.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME,
                    telemetry=telemetry,
                )

        remaining = [order for order in sorted(by_order) if not complete(order)]
        if remaining:
            raise ValueError(f"origin explanation generation exhausted orders={remaining}")

        return [
            question.model_copy(update={
                "explanation_origin": explanations[question.order],
                "explanation_learning": learning_explanations.get(
                    question.order, question.explanation_learning
                ),
            })
            for question in questions
        ]

    async def _generate_origin_explanation_batch(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
        explanations: dict[int, str],
        learning_explanations: dict[int, str],
        learning_repairs_required: set[int],
        *,
        type_name: str,
        telemetry: _ContextualChoiceTelemetry | None = None,
    ) -> None:
        expected_orders = {question.order for question in questions}
        payload_questions = []
        for question in questions:
            correct_text = self._correct_answer_text(question)
            payload_question: dict[str, Any] = {
                "order": question.order,
                "passageText": question.passage_text,
                "prompt": question.prompt,
                "targetExpression": question.target_expression,
                "correctAnswerText": correct_text,
                "evidenceText": question.evidence_text,
                "explanationLearning": question.explanation_learning,
            }
            if request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
                payload_question["repairExplanationLearning"] = (
                    question.order in learning_repairs_required
                )
            if (
                request.domain == PracticeDomain.VOCABULARY
                and request.mode == VocabularyMode.MEANING_RELATION.value
            ):
                payload_question["immutableTaskFact"] = {
                    "authority": "APPLICATION_SELECTED_IMMUTABLE",
                    "mode": VocabularyMode.MEANING_RELATION.value,
                    "relationKind": question.skill_tag,
                }
            payload_questions.append(payload_question)
        try:
            if telemetry is not None:
                telemetry.origin_explanation_calls += 1
            raw = await asyncio.wait_for(
                self.provider.call(
                    type_name=type_name,
                    data=build_origin_explanation_prompt(request, payload_questions),
                    schema=(
                        _CONTEXTUAL_CHOICE_ORIGIN_EXPLANATION_SCHEMA
                        if request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value
                        else _ORIGIN_EXPLANATION_SCHEMA
                    ),
                ),
                timeout=self.timeout_seconds,
            )
            payload = _OriginExplanationPayload.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "Reading/Vocabulary origin explanation provider/schema failure. request_id=%s "
                "modelTask=%s orders=%s type=%s",
                request.request_id,
                type_name,
                sorted(expected_orders),
                type(exc).__name__,
            )
            return

        seen: set[int] = set()
        for item in payload.explanations:
            if item.order not in expected_orders or item.order in seen:
                continue
            seen.add(item.order)
            text = item.text.strip()
            if not text:
                continue
            repaired_learning: str | None = None
            try:
                if self._origin_explanation_exposes_internal_metadata(text):
                    raise ValueError(
                        "origin explanation contains JSON/tool/internal metadata artifacts"
                    )
                if (
                    request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value
                    and self._contextual_choice_origin_explanation_incomplete(
                        text, request.origin_language
                    )
                ):
                    raise ValueError("ORIGIN_EXPLANATION_INCOMPLETE")
                # Validate each explanation independently. One good Korean explanation can no
                # longer hide nine Japanese explanations in a combined-language check.
                self._assert_language_lane(
                    request.origin_language,
                    [text],
                    reason=f"origin explanation order={item.order} is not in originLanguage",
                    allow_mixed_scripts=True,
                )
                if item.order in learning_repairs_required:
                    repaired_learning = (item.explanationLearning or "").strip()
                    if self._contextual_choice_explanation_learning_requires_repair(
                        repaired_learning
                    ):
                        raise ValueError("LEARNING_EXPLANATION_INVALID")
                    self._assert_language_lane(
                        request.learning_language,
                        [repaired_learning],
                        reason=(
                            f"learning explanation order={item.order} is not in "
                            "learningLanguage"
                        ),
                        allow_mixed_scripts=True,
                    )
            except ValueError as exc:
                logger.warning(
                    "Reading/Vocabulary origin explanation rejected. request_id=%s order=%d "
                    "modelTask=%s reason_code=%s",
                    request.request_id,
                    item.order,
                    type_name,
                    (
                        "ORIGIN_EXPLANATION_INCOMPLETE"
                        if str(exc) == "ORIGIN_EXPLANATION_INCOMPLETE"
                        else "LEARNING_EXPLANATION_INVALID"
                        if str(exc) == "LEARNING_EXPLANATION_INVALID"
                        else "ORIGIN_EXPLANATION_INVALID"
                    ),
                )
                continue
            explanations[item.order] = text
            if repaired_learning is not None:
                learning_explanations[item.order] = repaired_learning

    @classmethod
    def _contextual_choice_explanation_learning_requires_repair(
        cls, text: str
    ) -> bool:
        stripped = unicodedata.normalize("NFKC", text).strip()
        folded = stripped.casefold().rstrip(".!?。！？")
        placeholders = {
            "該当なし",
            "該当無し",
            "なし",
            "不明",
            "n/a",
            "na",
            "none",
            "not applicable",
            "해당 없음",
            "없음",
            "모름",
        }
        return (
            not stripped
            or folded in placeholders
            or cls._origin_explanation_exposes_internal_metadata(stripped)
        )

    @staticmethod
    def _contextual_choice_origin_explanation_incomplete(
        text: str, origin_language: str
    ) -> bool:
        stripped = text.strip()
        if (
            len(stripped) < 12
            or stripped.endswith(("...", "…", ":", ",", "、", "-", "—", "(", "["))
            or stripped.startswith(("{", "[{", '["', "```"))
        ):
            return True
        if origin_language.lower() in {"ko", "kr"}:
            return stripped.split()[-1] in {
                "인", "은", "는", "이", "가", "을", "를", "의", "와", "과",
                "에게", "따라서", "또는", "및",
            }
        return False

    @staticmethod
    def _origin_explanation_exposes_internal_metadata(text: str) -> bool:
        stripped = text.strip()
        folded = text.casefold()
        if any(
            token.casefold() in folded
            for token in (
                "immutableTaskFact",
                "APPLICATION_SELECTED_IMMUTABLE",
                "evidenceLearning",
                "explanationLearning",
                "evidenceText",
                "correctAnswerText",
                "targetExpression",
                "```json",
                "tool_call",
                "function_call",
            )
        ):
            return True
        if re.match(
            r"^(?:json|tool|assistant|system|developer|analysis|metadata)\s*:",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True
        if stripped.startswith(("{", "[")):
            try:
                return isinstance(json.loads(stripped), (dict, list))
            except json.JSONDecodeError:
                pass
        return False

    @staticmethod
    def _correct_answer_text(question: PracticeGeneratedQuestion) -> str:
        option_by_key = {option.key: option.text for option in question.options}
        if question.question_type == PracticeQuestionType.SINGLE_CHOICE:
            return option_by_key.get(question.correct_answer[0], "")
        return " ".join(option_by_key.get(key, "") for key in question.correct_answer).strip()

    def _validate(
        self,
        request: PracticeGenerationRequest,
        questions: list[PracticeGeneratedQuestion],
    ) -> None:
        if len(questions) != request.question_count:
            raise ValueError("question count mismatch")
        if [q.order for q in questions] != list(range(1, request.question_count + 1)):
            raise ValueError("question order must be contiguous")

        allowed_skills = (
            {item.value for item in ReadingSkill}
            if request.domain == PracticeDomain.READING
            else {item.value for item in VocabularySkill}
        )
        expected_bands = sorted(
            [max(1, request.complexity_band - 1)] * request.easier_count
            + [request.complexity_band] * request.current_count
            + [min(5, request.complexity_band + 1)] * request.challenge_count
        )
        actual_bands = sorted(q.complexity_band for q in questions)
        if actual_bands != expected_bands:
            raise ValueError(f"complexity mix mismatch: expected={expected_bands} actual={actual_bands}")

        expected_difficulty_counts = {
            PracticeDifficulty.EASIER: request.easier_count,
            PracticeDifficulty.CURRENT: request.current_count,
            PracticeDifficulty.CHALLENGE: request.challenge_count,
        }
        actual_difficulty_counts = {
            difficulty: sum(1 for q in questions if q.difficulty == difficulty)
            for difficulty in expected_difficulty_counts
        }
        if actual_difficulty_counts != expected_difficulty_counts:
            raise ValueError("difficulty label mix mismatch")

        for question in questions:
            if question.skill_tag not in allowed_skills:
                raise ValueError(f"invalid skillTag: {question.skill_tag}")
            self._validate_question_shape(question)
            self._validate_learning_language(request, question)
            self._assert_language_lane(
                request.origin_language,
                [question.explanation_origin],
                reason=f"origin explanation order={question.order} is not in originLanguage",
                allow_mixed_scripts=True,
            )

            if request.domain == PracticeDomain.READING:
                if not question.passage_id or not question.passage_text:
                    raise ValueError("reading question requires passageId/passageText")
                if question.target_expression or question.canonical_key:
                    raise ValueError("reading question must not become vocabulary item")
            else:
                if question.passage_id is not None or question.passage_text is not None:
                    raise ValueError("vocabulary question must not include passage")
                if not question.target_expression or not question.canonical_key:
                    raise ValueError("vocabulary question requires targetExpression/canonicalKey")
                self._validate_vocabulary_surface_language(request, question)

        if request.domain == PracticeDomain.READING:
            passage_text_by_id: dict[str, str] = {}
            for question in [*request.previous_questions, *questions]:
                assert question.passage_id is not None
                assert question.passage_text is not None
                previous = passage_text_by_id.setdefault(question.passage_id, question.passage_text)
                if previous != question.passage_text:
                    raise ValueError("same passageId must reuse identical passageText")
            expected_passages = (
                {"p1", "p2"}
                if request.question_offset + request.question_count > 3
                else {"p1"}
            )
            if set(passage_text_by_id) != expected_passages:
                raise ValueError("reading set must use planned passages")
        else:
            canonical_keys = [
                q.canonical_key for q in [*request.previous_questions, *questions]
            ]
            if len(canonical_keys) != len(set(canonical_keys)):
                raise ValueError("vocabulary set must use unique canonicalKey values")
            review_count = sum(1 for q in questions if q.review_target)
            eligible_reviews = self._eligible_review_targets(request, log_rejections=False)
            expected_review_count = min(request.review_question_count, len(eligible_reviews))
            if review_count != expected_review_count:
                raise ValueError(
                    f"review mix mismatch expected={expected_review_count} actual={review_count}"
                )
            review_keys = {target.canonical_key for target in eligible_reviews}
            for question in questions:
                if question.review_target and question.canonical_key not in review_keys:
                    raise ValueError("review question canonicalKey is not in reviewTargets")
            if request.mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
                if request.vocabulary_plan is not None:
                    expected_skill_by_order = {
                        local_order: request.vocabulary_plan.items[
                            request.question_offset + local_order - 1
                        ].skill_tag
                        for local_order in range(1, request.question_count + 1)
                    }
                else:
                    expected_skill_by_order = {
                        slot.order: slot.skill_tag
                        for slot in self._build_slots(request, {})
                    }
                for question in questions:
                    if question.question_type != PracticeQuestionType.SINGLE_CHOICE:
                        raise ValueError(
                            "CONTEXTUAL_CHOICE must not generate ordering questions"
                        )
                    if question.skill_tag != expected_skill_by_order[question.order]:
                        raise ValueError(
                            "CONTEXTUAL_CHOICE skill does not match the daily plan"
                        )
            if request.mode == VocabularyMode.COMPOSITION.value:
                ordering_count = sum(
                    1 for q in questions if q.question_type == PracticeQuestionType.ORDERING
                )
                expected_ordering_count = sum(
                    index in {1, 3, 6, 8}
                    for index in range(
                        request.question_offset + 1,
                        request.question_offset + request.question_count + 1,
                    )
                )
                if ordering_count < expected_ordering_count:
                    raise ValueError("composition requires its planned ordering questions")

    @staticmethod
    def _assert_language_lane(
        language_code: str,
        texts: list[str],
        *,
        reason: str,
        allow_mixed_scripts: bool = False,
    ) -> None:
        language = ReadingVocabularyGenerationService._base_language(language_code)
        normalized = "\n".join(text for text in texts if text and text.strip())
        if not normalized:
            raise ValueError(reason)

        has_hangul = bool(re.search(r"[\uac00-\ud7a3]", normalized))
        has_kana = bool(re.search(r"[\u3040-\u30ff]", normalized))
        has_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", normalized))
        has_ascii = bool(re.search(r"[A-Za-z]", normalized))

        mismatch = False
        missing_expected_script = False
        if language == "ja":
            mismatch = has_hangul
            missing_expected_script = not (has_kana or has_cjk)
        elif language == "ko":
            mismatch = has_kana
            missing_expected_script = not has_hangul
        elif language == "en":
            mismatch = has_hangul or has_kana or has_cjk
            missing_expected_script = not has_ascii
        else:
            # Unknown/same-script languages cannot be proven safely with Unicode ranges.
            # Keep the content non-empty and rely on the generation/explanation stage contract.
            return

        if (mismatch and not allow_mixed_scripts) or missing_expected_script:
            raise ValueError(reason)
