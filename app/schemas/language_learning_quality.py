from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class LanguageComplexityBand(str, Enum):
    FOUNDATION = "FOUNDATION"
    BASIC = "BASIC"
    INTERMEDIATE = "INTERMEDIATE"
    UPPER_INTERMEDIATE = "UPPER_INTERMEDIATE"
    ADVANCED = "ADVANCED"

    @property
    def level(self) -> int:
        return {
            LanguageComplexityBand.FOUNDATION: 1,
            LanguageComplexityBand.BASIC: 2,
            LanguageComplexityBand.INTERMEDIATE: 3,
            LanguageComplexityBand.UPPER_INTERMEDIATE: 4,
            LanguageComplexityBand.ADVANCED: 5,
        }[self]

    @classmethod
    def from_level(cls, value: int) -> "LanguageComplexityBand":
        clamped = max(1, min(5, value))
        return {
            1: cls.FOUNDATION,
            2: cls.BASIC,
            3: cls.INTERMEDIATE,
            4: cls.UPPER_INTERMEDIATE,
            5: cls.ADVANCED,
        }[clamped]


class ScenarioCategory(str, Enum):
    DAILY_LIFE = "DAILY_LIFE"
    WORK = "WORK"
    TRAVEL = "TRAVEL"
    SHOPPING = "SHOPPING"
    FOOD = "FOOD"
    SERVICE = "SERVICE"
    LEARNING = "LEARNING"
    HOBBY = "HOBBY"
    DIGITAL_LIFE = "DIGITAL_LIFE"
    SOCIAL = "SOCIAL"
    SCHEDULE = "SCHEDULE"
    HEALTH_GENERAL = "HEALTH_GENERAL"


class CommunicativeIntent(str, Enum):
    DESCRIBE = "DESCRIBE"
    REQUEST = "REQUEST"
    CONFIRM = "CONFIRM"
    REPORT = "REPORT"
    SUGGEST = "SUGGEST"
    DECLINE = "DECLINE"
    APOLOGIZE = "APOLOGIZE"
    COMPARE = "COMPARE"
    EXPLAIN_REASON = "EXPLAIN_REASON"
    ASK_INFORMATION = "ASK_INFORMATION"
    GIVE_INSTRUCTION = "GIVE_INSTRUCTION"
    EXPRESS_PREFERENCE = "EXPRESS_PREFERENCE"
    SUMMARIZE = "SUMMARIZE"


class GenerationSourceType(str, Enum):
    LEVEL_TEST = "LEVEL_TEST"
    WRITING = "WRITING"
    LISTENING = "LISTENING"


class LanguageComplexityContext(CamelCaseModel):
    base_level_score: float | None = Field(default=None, ge=0, le=100)
    base_complexity_band: int = Field(default=3, ge=1, le=5)
    target_complexity_band: int | None = Field(default=None, ge=1, le=5)
    policy_version: str = Field(default="language-complexity-v1", min_length=1, max_length=100)


class DiversityMetadata(CamelCaseModel):
    scenario_category: ScenarioCategory
    communicative_intent: CommunicativeIntent
    task_archetype: str = Field(..., min_length=1, max_length=100)
    grammar_focus_codes: list[str] = Field(default_factory=list, max_length=20)
    lexical_focus_codes: list[str] = Field(default_factory=list, max_length=20)
    semantic_summary: str = Field(..., min_length=1, max_length=500)
    requires_background_knowledge: bool = False
    content_hash: str | None = Field(default=None, max_length=128)
    similarity_key: str | None = Field(default=None, max_length=128)


class DiversityHistoryEntry(CamelCaseModel):
    source_type: GenerationSourceType
    content: str = Field(..., min_length=1, max_length=4000)
    content_hash: str | None = Field(default=None, max_length=128)
    scenario_category: ScenarioCategory | None = None
    communicative_intent: CommunicativeIntent | None = None
    task_archetype: str | None = Field(default=None, max_length=100)
    grammar_focus_codes: list[str] = Field(default_factory=list, max_length=20)
    semantic_summary: str | None = Field(default=None, max_length=500)
    age_days: int | None = Field(default=None, ge=0, le=3650)


class DiversityContext(CamelCaseModel):
    current_session: list[DiversityHistoryEntry] = Field(default_factory=list, max_length=40)
    same_feature_recent: list[DiversityHistoryEntry] = Field(default_factory=list, max_length=80)
    cross_feature_recent: list[DiversityHistoryEntry] = Field(default_factory=list, max_length=40)
    exact_content_hashes_90d: list[str] = Field(
        default_factory=list,
        max_length=200,
        alias="exactContentHashes90d",
    )


class DiversitySummary(CamelCaseModel):
    policy_version: str = "language-learning-diversity-v1"
    candidate_count: int = Field(default=0, ge=0)
    accepted_count: int = Field(default=0, ge=0)
    rejected_exact: int = Field(default=0, ge=0)
    rejected_similarity: int = Field(default=0, ge=0)
    rejected_structural: int = Field(default=0, ge=0)
    rejected_background_knowledge: int = Field(default=0, ge=0)
    fallback_used: bool = False
