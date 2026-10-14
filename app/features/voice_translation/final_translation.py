from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.ai.model_policy import get_model_name_for_task
from app.ai.ports import StructuredTextGenerationProvider, VoiceTranslationGenerationResult
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.prompts import build_voice_translation_prompt
from app.schemas.voice_translation import VoiceErrorCode, VoiceStage


class _FinalTranslationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    translated_text: str = Field(alias="translatedText", min_length=1, max_length=10_000)


class OpenAIVoiceFinalTranslationProvider:
    """확정 원문을 번역하되 후리가나 생성을 실시간 응답의 필수 작업에서 제외한다."""

    generates_source_reading_tokens = False

    def __init__(self, provider: StructuredTextGenerationProvider) -> None:
        self._provider = provider

    @property
    def ready(self) -> bool:
        return bool(getattr(self._provider, "ready", True))

    def model_name_for(self, type_name: str) -> str:
        del type_name
        return get_model_name_for_task("VOICE_TRANSLATION_TEXT")

    async def translate_voice_utterance(
        self,
        *,
        source_text: str,
        source_language: str,
        target_language: str,
        previous_source_text: str | None = None,
    ) -> VoiceTranslationGenerationResult:
        # 미리보기 결과를 확정으로 승격하지 않고 확정 원문과 직전 문맥을 다시 번역한다.
        result = await self._provider.call_with_metadata(
            type_name="VOICE_TRANSLATION_TEXT",
            data=build_voice_translation_prompt(
                source_text=source_text,
                source_language=source_language,
                target_language=target_language,
                previous_source_text=previous_source_text,
            ),
            schema=_FinalTranslationPayload.model_json_schema(by_alias=True),
        )

        # 완료 여부와 출력 구조를 모두 검사한다. 잘린 번역을 성공으로 저장하지 않는다.
        if result.status != "completed" or result.incomplete_reason is not None:
            raise self._invalid_result()
        try:
            payload = _FinalTranslationPayload.model_validate(result.data)
        except ValidationError as exc:
            raise self._invalid_result() from exc

        return VoiceTranslationGenerationResult(
            translated_text=payload.translated_text,
            source_reading_tokens=[],
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            provider=result.provider,
            model=result.model,
        )

    @staticmethod
    def _invalid_result() -> VoicePipelineException:
        return VoicePipelineException(
            code=VoiceErrorCode.TRANSLATION_FAILED,
            stage=VoiceStage.TRANSLATION,
            message="확정 번역 응답을 확인하지 못했습니다.",
            retryable=False,
        )
