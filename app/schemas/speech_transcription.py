from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AudioDecodeCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    request_id: str = Field(alias="requestId", pattern=r"^[A-Za-z0-9._:-]{1,100}$")
    audio_base64: str = Field(alias="audioBase64", min_length=1, max_length=14_000_000)
    remaining_milliseconds: int = Field(alias="remainingMilliseconds", ge=1, le=300_000)


class AudioDecodeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    audio_base64: str = Field(alias="audioBase64")
    duration_seconds: float = Field(alias="durationSeconds", ge=0)
    source_format: str = Field(alias="sourceFormat")
    rms: float
    peak: float
    silence_ratio: float = Field(alias="silenceRatio")
    sample_rate: int = Field(alias="sampleRate")
    channels: int


class SpeechTranscriptionCommand(AudioDecodeCommand):
    runtime: Literal["shared", "accurate"]
    language: str | None = Field(default=None, min_length=2, max_length=30)
    initial_prompt: str | None = Field(default=None, alias="initialPrompt", max_length=12000)
    beam_size: int = Field(alias="beamSize", ge=1, le=5)
    vad_filter: bool = Field(alias="vadFilter")
    min_silence_duration_ms: int = Field(alias="minSilenceDurationMs", ge=0, le=10000)
    condition_on_previous_text: bool = Field(alias="conditionOnPreviousText")
    max_provider_calls: Literal[1] = Field(alias="maxProviderCalls")


class TranscriptionSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    start_seconds: float = Field(alias="startSeconds")
    end_seconds: float = Field(alias="endSeconds")
    text: str
    avg_logprob: float = Field(alias="avgLogprob")
    no_speech_probability: float | None = Field(alias="noSpeechProbability")


class SpeechTranscriptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    text: str
    language: str | None
    language_probability: float | None = Field(alias="languageProbability")
    duration_seconds: float | None = Field(alias="durationSeconds")
    segments: list[TranscriptionSegment]
    provider: str
    model: str
    model_version: str | None = Field(alias="modelVersion")
    provider_calls: Literal[1] = Field(default=1, alias="providerCalls")
