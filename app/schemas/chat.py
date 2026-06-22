from pydantic import BaseModel, Field


class ChatTranslationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="번역 대상 채팅 메시지 원문",
    )
    target_language_code: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="번역 대상 언어 코드. 예: ko, ja, en",
    )
    source_language_code: str | None = Field(
        None,
        max_length=10,
        description="원문 언어 코드. 미지정 시 AI가 문맥으로 판단",
    )


class ChatTranslationResponse(BaseModel):
    translated_text: str = Field(
        ...,
        description="번역된 채팅 메시지",
    )