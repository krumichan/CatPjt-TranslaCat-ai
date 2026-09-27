"""Loopback-only CHAT → real FastAPI → OpenAIService → synthetic SDK transport fixture."""

import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main() -> None:
    # 합성 키와 격리 작업 디렉터리만 받아 기존 개발 설정·실제 Provider 호출을 차단한다.
    if (
        os.environ.get("CHAT_AUTH_CONTRACT_TEST") != "true"
        or Path(".env").exists()
        or "AI_SETTINGS_ENV_FILE" in os.environ
    ):
        raise RuntimeError("An isolated synthetic test directory is required")

    import uvicorn

    from app.ai.providers.openai.client import OpenAIService
    from app.api.dependencies import get_model_execution_provider
    from app.core.chat_auth import validate_chat_auth_configuration
    from app.core.config import settings

    with patch("app.core.config_logger.setup_logging"):
        from app.main import app

    validate_chat_auth_configuration(settings)
    if settings.CHAT_AUTH_ENVIRONMENT != "Development":
        raise RuntimeError("The loopback fixture is Development only")

    calls = []

    class SyntheticThrottlingError(RuntimeError):
        status_code = 429

        def __init__(self, seconds: int):
            super().__init__("synthetic-provider-throttle")
            self.retry_after_seconds = seconds

    class SyntheticResponses:
        def create(self, **kwargs):
            raise RuntimeError("Synchronous Provider call is forbidden")

        async def create_async(self, **kwargs):
            # 실제 OpenAIService.execute_explicit을 통과한 SDK 인자를 검증한다.
            calls.append(kwargs)
            if kwargs["model"] != settings.OPENAI_MODEL_LUNA:
                raise AssertionError("Unexpected model alias")
            if kwargs["reasoning"] != {"effort": "none"} or kwargs["store"] is not False:
                raise AssertionError("Unexpected generic execution profile")

            # 기술 오류 경로는 실제 FastAPI 예외 분류를 통과시킨다.
            content = " ".join(message["content"] for message in kwargs["input"])
            for seconds in (600, 86_400):
                if f"synthetic-retry-after-{seconds}" in content:
                    raise SyntheticThrottlingError(seconds)

            structured = "format" in kwargs["text"]
            if structured:
                fmt = kwargs["text"]["format"]
                if fmt["strict"] is not False or fmt["name"] != "AI_CHAT_REPLY":
                    raise AssertionError("Chat reply schema was changed")
                output = json.dumps(
                    {"shouldRespond": True, "reply": "合成応答", "languageCode": "ja"}
                    if "synthetic-runtime-save" in content
                    else {"shouldRespond": False, "reply": "discard", "languageCode": "ko"}
                )
            else:
                output = '```json\n{"translatedText":"synthetic translation"}\n```'

            return SimpleNamespace(
                output_text=output,
                status="completed",
                output=[],
                model=kwargs["model"],
                usage=SimpleNamespace(input_tokens=10, output_tokens=3),
            )

    class SyntheticSdk:
        def __init__(self):
            self.responses = self

        def with_options(self, **kwargs):
            if not 0 < kwargs["timeout"] <= 300:
                raise AssertionError("Unbounded Provider timeout")
            return self

        async def create(self, **kwargs):
            return await SyntheticResponses().create_async(**kwargs)

    provider = OpenAIService()
    provider._client = SyntheticSdk()
    app.dependency_overrides[get_model_execution_provider] = lambda: provider

    @app.get("/_synthetic/chat-execution/counts")
    def read_counts():
        return {"model": len(calls), "structured": sum("format" in c["text"] for c in calls)}

    # 운영 router/middleware와 실제 Provider adapter를 loopback socket으로 연결한다.
    container_mode = os.environ.get("CHAT_AUTH_CONTRACT_CONTAINER") == "true"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("0.0.0.0" if container_mode else "127.0.0.1", 8000 if container_mode else 0))
        listener.listen(64)
        origin = (
            os.environ["CHAT_AUTH_CONTRACT_ORIGIN"]
            if container_mode
            else f"http://127.0.0.1:{listener.getsockname()[1]}"
        )
        manifest = Path(os.environ["CHAT_AUTH_CONTRACT_MANIFEST"])
        manifest.write_text(
            json.dumps(
                {
                    "origin": origin,
                    "pid": os.getpid(),
                    "runId": os.environ["CHAT_AUTH_CONTRACT_RUN_ID"],
                }
            ),
            encoding="utf-8",
        )
        uvicorn.Server(
            uvicorn.Config(app, lifespan="off", log_level="warning", access_log=False)
        ).run(sockets=[listener])


if __name__ == "__main__":
    main()
