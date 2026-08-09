import importlib
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.schemas.chat_ai import ChatAiReplyResponse


class FakeReplyService:
    async def generate_reply(self, request):
        return ChatAiReplyResponse(
            request_id=request.request_id,
            should_respond=True,
            reply="京都が好きかな！",
            language_code=request.ai_member.original_language_code,
        )


class ChatAiReplyApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_dependencies = types.ModuleType("app.api.dependencies")
        cls.fake_service = FakeReplyService()
        fake_dependencies.get_chat_ai_reply_service = lambda: cls.fake_service
        fake_dependencies.get_chat_translation_service = lambda: None

        cls.original_dependencies = sys.modules.get("app.api.dependencies")
        cls.original_v1_package = sys.modules.get("app.api.v1")
        cls.original_chat_module = sys.modules.pop("app.api.v1.chat", None)

        fake_v1_package = types.ModuleType("app.api.v1")
        fake_v1_package.__path__ = [
            str(Path(__file__).resolve().parents[1] / "app" / "api" / "v1")
        ]

        sys.modules["app.api.dependencies"] = fake_dependencies
        sys.modules["app.api.v1"] = fake_v1_package

        chat_api = importlib.import_module("app.api.v1.chat")
        cls.chat_api = chat_api

        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("app.api.v1.chat", None)

        if cls.original_chat_module is not None:
            sys.modules["app.api.v1.chat"] = cls.original_chat_module

        if cls.original_v1_package is None:
            sys.modules.pop("app.api.v1", None)
        else:
            sys.modules["app.api.v1"] = cls.original_v1_package

        if cls.original_dependencies is None:
            sys.modules.pop("app.api.dependencies", None)
        else:
            sys.modules["app.api.dependencies"] = cls.original_dependencies

    def _payload(self):
        now = datetime.now(timezone.utc).isoformat()
        return {
            "requestId": "req-api-1",
            "triggerType": "MENTION",
            "room": {
                "roomId": 123,
                "roomType": "GROUP",
                "name": "친구방",
                "description": None,
            },
            "aiMember": {
                "aiMemberId": 10,
                "nickname": "미야",
                "bio": "친근한 AI",
                "personaPrompt": "밝고 자연스럽게 대화한다.",
                "originalLanguageCode": "ja",
            },
            "triggerMessage": {
                "messageId": 1004,
                "senderId": "member-22",
                "senderName": "철수",
                "content": "@미야 안녕?",
                "createdAt": now,
            },
            "contextMessages": [],
        }

    def test_reply_endpoint_uses_camel_case_contract(self):
        response = self.client.post(
            "/api/v1/chat/ai/reply",
            json=self._payload(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "requestId": "req-api-1",
                "shouldRespond": True,
                "reply": "京都が好きかな！",
                "languageCode": "ja",
            },
        )

    def test_invalid_revival_trigger_message_returns_422(self):
        payload = self._payload()
        payload["triggerType"] = "REVIVAL"

        response = self.client.post(
            "/api/v1/chat/ai/reply",
            json=payload,
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
