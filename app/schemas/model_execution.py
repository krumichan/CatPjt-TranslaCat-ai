from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelTier(str, Enum):
    NANO = "NANO"
    LUNA = "LUNA"
    MINI = "MINI"
    SOL = "SOL"


class ExecutionMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ModelExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    trace_id: str = Field(alias="traceId", pattern=r"^[A-Za-z0-9._:-]{1,100}$")
    instructions: str = Field(min_length=1, max_length=100_000)
    messages: list[ExecutionMessage] = Field(min_length=1, max_length=30)
    tier: ModelTier
    reasoning_effort: Literal["none", "low", "high"] = Field(alias="reasoningEffort")
    verbosity: Literal["low", "medium", "high"]
    max_output_tokens: int = Field(alias="maxOutputTokens", ge=1, le=8192)
    remaining_milliseconds: int = Field(alias="remainingMilliseconds", ge=1, le=300_000)
    max_provider_calls: Literal[1] = Field(alias="maxProviderCalls")
    response_schema: dict[str, Any] | None = Field(default=None, alias="responseSchema")
    schema_name: str | None = Field(default=None, alias="schemaName", max_length=64)
    strict: bool = False
    task_name: str | None = Field(default=None, alias="taskName", max_length=100)

    @model_validator(mode="after")
    def validate_profile(self) -> ModelExecutionRequest:
        expected = {
            ModelTier.NANO: "low",
            ModelTier.LUNA: "none",
            ModelTier.MINI: "low",
            ModelTier.SOL: "high",
        }
        if self.reasoning_effort != expected[self.tier]:
            raise ValueError("Unsupported execution profile")
        if (self.response_schema is None) != (self.schema_name is None):
            raise ValueError("responseSchema and schemaName must be supplied together")
        if self.strict and self.response_schema is None:
            raise ValueError("strict output requires responseSchema")
        return self


class ModelExecutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    output: Any
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    provider: str
    model: str
    provider_calls: Literal[1] = Field(default=1, alias="providerCalls")
