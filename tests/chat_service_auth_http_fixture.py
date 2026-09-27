import json
import os
import socket
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    # 명시 실행 harness만 이 fixture를 시작한다. 개인 설정과 실제 Provider는 사용하지 않는다.
    if os.environ.get("CHAT_AUTH_CONTRACT_TEST") != "true" or Path(".env").exists():
        raise RuntimeError("An isolated synthetic test directory is required")

    import uvicorn

    from app.api.dependencies import (
        get_chat_ai_reply_service,
        get_chat_translation_service,
        get_model_execution_provider,
    )
    from app.core.chat_auth import validate_chat_auth_configuration
    from app.core.config import settings
    from app.schemas.chat_ai import ChatAiReplyResponse

    with patch("app.core.config_logger.setup_logging"):
        from app.main import app

    validate_chat_auth_configuration(settings)
    if settings.CHAT_AUTH_ENVIRONMENT != "Development":
        raise RuntimeError("The loopback fixture is Development only")
    counts = {"translation": 0, "reply": 0, "model": 0}

    class SyntheticChatService:
        async def translate(self, **kwargs):
            counts["translation"] += 1
            return "synthetic translation"

        async def generate_reply(self, request):
            counts["reply"] += 1
            return ChatAiReplyResponse(
                request_id=request.request_id,
                should_respond=False,
                reply=None,
                language_code=None,
            )

        async def execute_explicit(self, **kwargs):
            counts["model"] += 1
            raise AssertionError("This fixture must never execute a model task")

    service = SyntheticChatService()
    app.dependency_overrides[get_chat_translation_service] = lambda: service
    app.dependency_overrides[get_chat_ai_reply_service] = lambda: service
    app.dependency_overrides[get_model_execution_provider] = lambda: service

    @app.get("/_synthetic/chat-auth/counts")
    def read_counts():
        return counts

    # 운영 인증 middleware/router/schema를 그대로 띄우고 서비스만 대체한다. lifespan warm-up은 끈다.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(64)
        origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
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
        server = uvicorn.Server(
            uvicorn.Config(app, lifespan="off", log_level="warning", access_log=False)
        )
        server.run(sockets=[listener])


if __name__ == "__main__":
    main()
