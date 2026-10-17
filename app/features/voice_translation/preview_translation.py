from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.ai.model_policy import get_model_name_for_task
from app.ai.ports import StructuredTextGenerationProvider, VoiceTranslationGenerationResult
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.prompts import build_voice_translation_prompt
from app.schemas.voice_translation import VoiceErrorCode, VoiceStage


class _PreviewTranslationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    translated_text: str = Field(alias="translatedText", min_length=1, max_length=10_000)


class OpenAIVoicePreviewTranslationProvider:
    """임시 번역만 생성하며 확정 번역의 읽기 토큰 생성과 예산을 분리한다."""

    generates_source_reading_tokens = False

    def __init__(self, provider: StructuredTextGenerationProvider) -> None:
        self._provider = provider

    @property
    def ready(self) -> bool:
        return bool(getattr(self._provider, "ready", True))

    def model_name_for(self, type_name: str) -> str:
        # VoiceTranslationService가 요청하는 최종 작업명과 관계없이 임시 작업 정책을 보고한다.
        del type_name
        return get_model_name_for_task("VOICE_TRANSLATION_PREVIEW")

    async def translate_voice_utterance(
        self,
        *,
        source_text: str,
        source_language: str,
        target_language: str,
        previous_source_text: str | None = None,
    ) -> VoiceTranslationGenerationResult:
        # 원문/문맥은 JSON 데이터로만 전달하고 한 번의 호출로 텍스트 번역만 요청한다.
        result = await self._provider.call_with_metadata(
            type_name="VOICE_TRANSLATION_PREVIEW",
            data=build_voice_translation_prompt(
                source_text=source_text,
                source_language=source_language,
                target_language=target_language,
                previous_source_text=previous_source_text,
            ),
            schema=_PreviewTranslationPayload.model_json_schema(by_alias=True),
        )

        # 잘린 응답이나 형식 실패를 임시 번역으로 노출하지 않는다. 응답 내용은 오류에 넣지 않는다.
        if result.status != "completed" or result.incomplete_reason is not None:
            raise self._invalid_result()
        try:
            payload = _PreviewTranslationPayload.model_validate(result.data)
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
            message="임시 번역 응답을 확인하지 못했습니다.",
            retryable=False,
        )
