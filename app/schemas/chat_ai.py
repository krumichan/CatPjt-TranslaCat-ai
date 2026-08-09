from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from app.core.config import settings


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ChatAiTriggerType(str, Enum):
    MENTION = "MENTION"
    CONVERSATION = "CONVERSATION"
    REVIVAL = "REVIVAL"


class ChatAiRoomType(str, Enum):
    GROUP = "GROUP"
    OPEN = "OPEN"


class ChatAiContextSenderType(str, Enum):
    USER = "USER"
    AI = "AI"
    SYSTEM = "SYSTEM"


class ChatAiRoom(CamelCaseModel):
    room_id: int = Field(..., gt=0)
    room_type: ChatAiRoomType
    name: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)


class ChatAiMember(CamelCaseModel):
    ai_member_id: int = Field(..., gt=0)
    nickname: str = Field(..., min_length=1, max_length=100)
    bio: str | None = Field(default=None, max_length=1000)
    persona_prompt: str | None = Field(default=None, max_length=5000)
    original_language_code: str = Field(..., min_length=2, max_length=10)


class ChatAiTriggerMessage(CamelCaseModel):
    message_id: int = Field(..., gt=0)
    sender_id: str = Field(..., min_length=1, max_length=100)
    sender_name: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=5000)
    created_at: datetime


class ChatAiContextMessage(CamelCaseModel):
    message_id: int = Field(..., gt=0)
    sender_type: ChatAiContextSenderType
    sender_id: str | None = Field(default=None, max_length=100)
    sender_name: str | None = Field(default=None, max_length=100)
    content: str = Field(..., min_length=1, max_length=5000)
    created_at: datetime


class ChatAiReplyRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    trigger_type: ChatAiTriggerType
    room: ChatAiRoom
    ai_member: ChatAiMember
    trigger_message: ChatAiTriggerMessage | None = None
    context_messages: list[ChatAiContextMessage] = Field(
        default_factory=list,
        max_length=settings.AI_CHAT_CONTEXT_HARD_MAX_MESSAGES,
    )
    context_max_messages: int = Field(
        default=settings.AI_CHAT_CONTEXT_DEFAULT_MAX_MESSAGES,
        ge=1,
        le=settings.AI_CHAT_CONTEXT_HARD_MAX_MESSAGES,
        description=(
            "Backend의 현재 AI 시스템 설정에 따른 Context 최대 메시지 수. "
            "미지정 시 Phase 2 초기값을 사용한다."
        ),
    )
    context_max_characters: int = Field(
        default=settings.AI_CHAT_CONTEXT_DEFAULT_MAX_CHARACTERS,
        ge=1,
        le=settings.AI_CHAT_CONTEXT_HARD_MAX_CHARACTERS,
        description=(
            "Backend의 현재 AI 시스템 설정에 따른 Context 전체 최대 문자 수. "
            "미지정 시 Phase 2 초기값을 사용한다."
        ),
    )
    reply_max_characters: int = Field(
        default=settings.AI_CHAT_REPLY_MAX_CHARACTERS,
        ge=1,
        le=settings.AI_CHAT_REPLY_HARD_MAX_CHARACTERS,
        description=(
            "Backend의 현재 AI 시스템 설정에 따른 응답 최대 문자 수. "
            "미지정 시 AI Server 기본값을 사용한다."
        ),
    )

    @model_validator(mode="after")
    def validate_trigger_and_context(self) -> "ChatAiReplyRequest":
        if self.trigger_type == ChatAiTriggerType.REVIVAL:
            if self.trigger_message is not None:
                raise ValueError("REVIVAL 요청의 triggerMessage는 null이어야 합니다.")
        elif self.trigger_message is None:
            raise ValueError(
                f"{self.trigger_type.value} 요청에는 triggerMessage가 필요합니다."
            )

        if len(self.context_messages) > self.context_max_messages:
            raise ValueError(
                "contextMessages 개수는 현재 요청의 contextMaxMessages "
                f"({self.context_max_messages})를 초과할 수 없습니다."
            )

        total_characters = sum(
            len(message.content)
            for message in self.context_messages
        )
        if total_characters > self.context_max_characters:
            raise ValueError(
                "contextMessages의 전체 content 길이는 현재 요청의 "
                f"contextMaxCharacters ({self.context_max_characters})를 "
                "초과할 수 없습니다."
            )

        return self


class ChatAiReplyResponse(CamelCaseModel):
    request_id: str
    should_respond: bool
    reply: str | None = None
    language_code: str | None = None
