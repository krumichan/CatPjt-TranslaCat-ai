from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ai.providers.openai.execution_profiles import EXECUTION_PROFILES
from app.schemas.model_execution import ExecutionMessage, ModelExecutionResponse


class ModelProfileExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    trace_id: str = Field(alias="traceId", pattern=r"^[A-Za-z0-9._:-]{1,100}$")
    instructions: str = Field(min_length=1, max_length=100_000)
    messages: list[ExecutionMessage] = Field(min_length=1, max_length=30)
    profile_id: str = Field(alias="profileId", max_length=64)
    max_output_tokens: int = Field(alias="maxOutputTokens", ge=1, le=32768)
    remaining_milliseconds: int = Field(alias="remainingMilliseconds", ge=1, le=300_000)
    max_provider_calls: Literal[1] = Field(alias="maxProviderCalls")
    response_schema: dict[str, Any] | None = Field(default=None, alias="responseSchema")
    schema_name: str | None = Field(default=None, alias="schemaName", min_length=1, max_length=64)
    strict: bool = False
    task_name: str | None = Field(default=None, alias="taskName", max_length=100)

    @model_validator(mode="after")
    def validate_profile(self) -> ModelProfileExecutionRequest:
        profile = EXECUTION_PROFILES.get(self.profile_id)
        if profile is None or self.max_output_tokens > profile.max_output_tokens:
            raise ValueError("Unsupported execution profile")
        if (self.response_schema is None) != (self.schema_name is None):
            raise ValueError("Schema and name must be supplied together")
        if self.strict and self.response_schema is None:
            raise ValueError("Strict output requires schema")
        return self


class ModelProfileExecutionResponse(ModelExecutionResponse):
    profile_id: str = Field(alias="profileId")
    profile_version: str = Field(alias="profileVersion")
    reasoning_effort: str = Field(alias="reasoningEffort")
    service_tier: str = Field(alias="serviceTier")
    cached_input_tokens: int = Field(alias="cachedInputTokens", ge=0)
    reasoning_tokens: int = Field(alias="reasoningTokens", ge=0)
    metadata: dict[str, Any]
