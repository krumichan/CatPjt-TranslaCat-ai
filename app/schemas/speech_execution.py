from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SpeechSynthesisCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(alias="requestId", pattern=r"^[A-Za-z0-9._:-]{1,100}$")
    text: str = Field(min_length=1, max_length=4096)
    voice: str = Field(min_length=1, max_length=80)
    language: str = Field(min_length=2, max_length=30)
    speed: Literal["NORMAL", "SLOW"]
    remaining_milliseconds: int = Field(alias="remainingMilliseconds", ge=1, le=300_000)
    max_provider_calls: Literal[1] = Field(alias="maxProviderCalls")


class SpeechSynthesisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    audio_base64: str = Field(alias="audioBase64")
    content_type: str = Field(alias="contentType")
    duration_seconds: float | None = Field(alias="durationSeconds", ge=0, allow_inf_nan=False)
    provider: str
    model: str
    provider_calls: Literal[1] = Field(default=1, alias="providerCalls")
