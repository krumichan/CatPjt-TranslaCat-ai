import asyncio
import unittest
from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import ValidationError

from app.features.chat_ai_reply.prompts import build_chat_ai_reply_prompt
from app.features.chat_ai_reply.service import ChatAiReplyService
from app.schemas.chat_ai import ChatAiReplyRequest


class FakeProvider:
    def __init__(self, result=None, error: Exception | None = None, delay: float = 0) -> None:
        self.result = result
        self.error = error
        self.delay = delay
        self.calls: list[dict] = []

    async def call(self, type_name: str, data: str, schema: dict | None = None):
        self.calls.append({"type_name": type_name, "data": data, "schema": schema})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result

    async def call_with_image(self, *args, **kwargs):
        raise NotImplementedError


def build_request(trigger_type: str = "MENTION", **overrides) -> ChatAiReplyRequest:
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "requestId": "req-1",
        "triggerType": trigger_type,
        "room": {
            "roomId": 123,
            "roomType": "OPEN",
            "name": "일본 생활 이야기",
            "description": "일본 생활에 대해 이야기하는 방",
        },
        "aiMember": {
            "aiMemberId": 10,
            "nickname": "미야",
            "bio": "도쿄에서 사는 일본인 친구",
            "personaPrompt": "밝고 친근한 채팅체를 사용한다.",
            "originalLanguageCode": "ja",
        },
        "triggerMessage": {
            "messageId": 1004,
            "senderId": "member-22",
            "senderName": "철수",
            "content": "@미야 일본에서 어디가 제일 좋아?",
            "createdAt": now,
        },
        "contextMessages": [
            {
                "messageId": 1001,
                "senderType": "USER",
                "senderId": "member-13",
                "senderName": "영희",
                "content": "이번에 일본 여행 가려고",
                "createdAt": now,
            }
        ],
    }
    if trigger_type == "REVIVAL":
        data["triggerMessage"] = None
    data.update(overrides)
    return ChatAiReplyRequest.model_validate(data)


class ChatAiReplySchemaTest(unittest.TestCase):
    def test_revival_requires_null_trigger_message(self):
        request = build_request("MENTION")
        data = request.model_dump(mode="json", by_alias=True)
        data["triggerType"] = "REVIVAL"

        with self.assertRaises(ValidationError):
            ChatAiReplyRequest.model_validate(data)

    def test_mention_requires_trigger_message(self):
        data = build_request("MENTION").model_dump(mode="json", by_alias=True)
        data["triggerMessage"] = None

        with self.assertRaises(ValidationError):
            ChatAiReplyRequest.model_validate(data)

    def test_direct_room_is_not_supported(self):
        data = build_request().model_dump(mode="json", by_alias=True)
        data["room"]["roomType"] = "DIRECT"

        with self.assertRaises(ValidationError):
            ChatAiReplyRequest.model_validate(data)

    def test_context_uses_request_level_system_setting_limit(self):
        data = build_request().model_dump(mode="json", by_alias=True)
        data["contextMaxMessages"] = 1
        data["contextMessages"] = data["contextMessages"] * 2
        data["contextMessages"][1]["messageId"] = 1002

        with self.assertRaises(ValidationError):
            ChatAiReplyRequest.model_validate(data)


class ChatAiReplyPromptTest(unittest.TestCase):
    def test_conversation_prompt_contains_decision_instruction_and_language(self):
        request = build_request("CONVERSATION")
        prompt = build_chat_ai_reply_prompt(request)

        self.assertIn("shouldRespond=false", prompt)
        self.assertIn("ja", prompt)
        self.assertIn("<chat-data>", prompt)
        self.assertIn("CONVERSATION", prompt)

    def test_revival_prompt_does_not_require_trigger_message(self):
        request = build_request("REVIVAL")
        prompt = build_chat_ai_reply_prompt(request)

        self.assertIn("inactive for a long time", prompt)
        self.assertIn('"triggerMessage":null', prompt)


class ChatAiReplyServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_mention_returns_original_language_reply(self):
        provider = FakeProvider(
            result={
                "shouldRespond": True,
                "reply": "京都が好きかな！",
                "languageCode": "ja",
            }
        )
        service = ChatAiReplyService(provider=provider, timeout_seconds=1)

        response = await service.generate_reply(build_request("MENTION"))

        self.assertTrue(response.should_respond)
        self.assertEqual(response.reply, "京都が好きかな！")
        self.assertEqual(response.language_code, "ja")
        self.assertEqual(provider.calls[0]["type_name"], "AI_CHAT_REPLY")
        self.assertIsNotNone(provider.calls[0]["schema"])

    async def test_conversation_should_respond_false_clears_reply(self):
        provider = FakeProvider(
            result={
                "shouldRespond": False,
                "reply": "should be discarded",
                "languageCode": "ja",
            }
        )
        service = ChatAiReplyService(provider=provider, timeout_seconds=1)

        response = await service.generate_reply(build_request("CONVERSATION"))

        self.assertFalse(response.should_respond)
        self.assertIsNone(response.reply)
        self.assertIsNone(response.language_code)

    async def test_reply_is_hard_limited_by_request(self):
        provider = FakeProvider(
            result={
                "shouldRespond": True,
                "reply": "1234567890",
                "languageCode": "ja",
            }
        )
        request = build_request("MENTION", replyMaxCharacters=5)
        service = ChatAiReplyService(provider=provider, timeout_seconds=1)

        response = await service.generate_reply(request)

        self.assertEqual(response.reply, "12345")

    async def test_provider_failure_is_502(self):
        provider = FakeProvider(error=RuntimeError("provider down"))
        service = ChatAiReplyService(provider=provider, timeout_seconds=1)

        with self.assertRaises(HTTPException) as raised:
            await service.generate_reply(build_request("MENTION"))

        self.assertEqual(raised.exception.status_code, 502)

    async def test_timeout_is_504(self):
        provider = FakeProvider(
            result={
                "shouldRespond": False,
                "reply": None,
                "languageCode": None,
            },
            delay=0.05,
        )
        service = ChatAiReplyService(provider=provider, timeout_seconds=0.001)

        with self.assertRaises(HTTPException) as raised:
            await service.generate_reply(build_request("CONVERSATION"))

        self.assertEqual(raised.exception.status_code, 504)

    async def test_invalid_provider_shape_is_502(self):
        provider = FakeProvider(result="not-an-object")
        service = ChatAiReplyService(provider=provider, timeout_seconds=1)

        with self.assertRaises(HTTPException) as raised:
            await service.generate_reply(build_request("MENTION"))

        self.assertEqual(raised.exception.status_code, 502)

    async def test_language_code_mismatch_is_502(self):
        provider = FakeProvider(
            result={
                "shouldRespond": True,
                "reply": "Hello!",
                "languageCode": "en",
            }
        )
        service = ChatAiReplyService(provider=provider, timeout_seconds=1)

        with self.assertRaises(HTTPException) as raised:
            await service.generate_reply(build_request("MENTION"))

        self.assertEqual(raised.exception.status_code, 502)


if __name__ == "__main__":
    unittest.main()
