"""명시적 로컬 결합 검사 표식에만 응답하는 합성 모델 출력."""

import os
import re
from typing import Any

from app.ai.ports import StructuredGenerationResult


def _is_recorded_chat_translation(kwargs: dict[str, Any]) -> bool:
    # FE fresh-start 검사가 실제로 만든 원문과 합성 AI 응답만 번역 fixture로 받는다.
    # CHAT의 원문 태그 안을 검사하므로 일반 메시지 속 부분 일치는 허용하지 않는다.
    messages = kwargs.get("messages", [])
    if kwargs.get("schema_name_value") is not None or len(messages) != 1:
        return False
    message = messages[0]
    content = message.get("content", "")
    if message.get("role") != "user" or not content.startswith(
        "You are a chat message translator."
    ):
        return False
    return (
        re.search(
            r"\n<message>\r?\n(Fresh-(?:first|second|unread)-[0-9]+|Local synthetic reply)"
            r"\r?\n</message>\s*\Z",
            content,
        )
        is not None
    )


def execute_local_smoke_fixture(kwargs: dict[str, Any]) -> StructuredGenerationResult | None:
    # 실제 사용자 입력은 합성 정상 응답으로 덮지 않는다.
    messages = kwargs.get("messages", [])
    explicit_smoke = any("TRANSLACAT_LOCAL_SMOKE" in item.get("content", "") for item in messages)
    if not explicit_smoke and not _is_recorded_chat_translation(kwargs):
        return None
    if os.environ.get("TRANSLACAT_TEST_MODEL_EXECUTION") != "1":
        raise RuntimeError("Local smoke fixture requires explicit test marker")

    # 업무 상태와 prompt 구성은 호출 서비스가 담당하고 모델 출력만 고정한다.
    schema = kwargs.get("schema_name_value")
    if schema == "AI_CHAT_REPLY":
        output = {"shouldRespond": True, "reply": "Local synthetic reply", "languageCode": "en"}
    elif schema is None:
        output = "Local synthetic translation"
    else:
        raise ValueError("Unsupported local smoke schema")
    return StructuredGenerationResult(output, 7, 2, "test-provider", "synthetic-local-smoke")
