from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import HTTPException

from app.ai.ports import TextGenerationProvider
from app.core.config import settings
from app.features.chat_ai_reply.prompts import build_chat_ai_reply_prompt
from app.schemas.chat_ai import ChatAiReplyRequest, ChatAiReplyResponse

logger = logging.getLogger(__name__)

_AI_CHAT_REPLY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "shouldRespond": {"type": "BOOLEAN"},
        "reply": {"type": "STRING", "nullable": True},
        "languageCode": {"type": "STRING", "nullable": True},
    },
    "required": [
        "shouldRespond",
        "reply",
        "languageCode",
    ],
}


class ChatAiReplyService:
    def __init__(
        self,
        provider: TextGenerationProvider,
        timeout_seconds: float | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.AI_CHAT_REPLY_TIMEOUT_SECONDS
        )

    async def generate_reply(
        self,
        request: ChatAiReplyRequest,
    ) -> ChatAiReplyResponse:
        prompt = build_chat_ai_reply_prompt(request)

        try:
            result = await asyncio.wait_for(
                self.provider.call(
                    type_name="AI_CHAT_REPLY",
                    data=prompt,
                    schema=_AI_CHAT_REPLY_SCHEMA,
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            logger.warning(
                "AI chat reply timed out. request_id=%s trigger=%s",
                request.request_id,
                request.trigger_type.value,
            )
            raise HTTPException(
                status_code=504,
                detail="AI 채팅 응답 생성 시간이 초과되었습니다.",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                "AI chat reply generation failed. request_id=%s trigger=%s",
                request.request_id,
                request.trigger_type.value,
            )
            raise HTTPException(
                status_code=502,
                detail="AI 채팅 응답 생성에 실패했습니다.",
            ) from exc

        response = self._normalize_result(
            request=request,
            result=result,
        )

        logger.info(
            "AI chat reply completed. request_id=%s trigger=%s should_respond=%s",
            request.request_id,
            request.trigger_type.value,
            response.should_respond,
        )
        return response

    def _normalize_result(
        self,
        request: ChatAiReplyRequest,
        result: Any,
    ) -> ChatAiReplyResponse:
        if not isinstance(result, dict):
            raise HTTPException(
                status_code=502,
                detail="AI 채팅 응답 형식이 올바르지 않습니다.",
            )

        should_respond = result.get("shouldRespond")
        if not isinstance(should_respond, bool):
            raise HTTPException(
                status_code=502,
                detail="AI 채팅 응답의 shouldRespond 값이 올바르지 않습니다.",
            )

        if not should_respond:
            return ChatAiReplyResponse(
                request_id=request.request_id,
                should_respond=False,
                reply=None,
                language_code=None,
            )

        reply = result.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            raise HTTPException(
                status_code=502,
                detail="AI 채팅 응답 내용이 비어 있습니다.",
            )

        normalized_reply = reply.strip()
        if len(normalized_reply) > request.reply_max_characters:
            logger.warning(
                "AI chat reply exceeded character limit and was truncated. "
                "request_id=%s actual=%s limit=%s",
                request.request_id,
                len(normalized_reply),
                request.reply_max_characters,
            )
            normalized_reply = normalized_reply[: request.reply_max_characters].rstrip()

        if not normalized_reply:
            raise HTTPException(
                status_code=502,
                detail="AI 채팅 응답 내용이 비어 있습니다.",
            )

        returned_language_code = result.get("languageCode")
        expected_language_code = request.ai_member.original_language_code
        if not isinstance(returned_language_code, str) or not returned_language_code.strip():
            raise HTTPException(
                status_code=502,
                detail="AI 채팅 응답의 languageCode가 비어 있습니다.",
            )

        normalized_language_code = returned_language_code.strip()
        if normalized_language_code.lower() != expected_language_code.lower():
            logger.warning(
                "AI chat reply languageCode mismatch. request_id=%s expected=%s actual=%s",
                request.request_id,
                expected_language_code,
                normalized_language_code,
            )
            raise HTTPException(
                status_code=502,
                detail="AI 채팅 응답 언어가 AI 멤버의 원문 언어와 일치하지 않습니다.",
            )

        return ChatAiReplyResponse(
            request_id=request.request_id,
            should_respond=True,
            reply=normalized_reply,
            language_code=expected_language_code,
        )
