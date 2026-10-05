"""명시적 로컬 합성 서버. 운영 앱에서 import하지 않는다."""

import asyncio
import json
import os

if os.environ.get("TRANSLACAT_TEST_MODEL_EXECUTION") != "1":
    raise RuntimeError("Local test model execution server requires explicit test marker")

from fastapi.responses import JSONResponse  # noqa: E402

from app.ai.ports import StructuredGenerationResult  # noqa: E402
from app.api.dependencies import (  # noqa: E402
    get_model_execution_provider,
    get_speaking_speech_runtime,
    get_speech_execution_provider,
    get_speech_runtime,
)
from app.api.internal.speech_transcription import get_speech_evidence_runtime  # noqa: E402
from app.core.config import settings  # noqa: E402
from scripts.leveltest_test_provider import execute_leveltest_fixture  # noqa: E402
from scripts.listening_test_provider import execute_listening_fixture  # noqa: E402
from scripts.local_smoke_test_provider import execute_local_smoke_fixture  # noqa: E402
from scripts.model_execution_replay import execute_replay  # noqa: E402
from scripts.practice_test_provider import execute_practice_fixture  # noqa: E402
from scripts.speaking_conversation_test_provider import (  # noqa: E402
    execute_speaking_conversation_fixture,
)
from scripts.speaking_evaluation_test_provider import (  # noqa: E402
    execute_speaking_evaluation_fixture,
)
from scripts.speaking_runtime_test_provider import (  # noqa: E402
    execute_speaking_runtime_fixture,
)
from scripts.synthetic_paths import synthetic_runtime_directory  # noqa: E402
from scripts.synthetic_speech_provider import SyntheticSpeechProvider  # noqa: E402
from scripts.writing_cutover_provider import execute_writing_fixture  # noqa: E402

# 합성 서버는 호스트 .env의 음성/OCR 예열 설정을 상속하지 않는다.
settings.AI_VOICE_WARM_UP = False
settings.OCR_WARM_UP = False

from app.main import app  # noqa: E402


class SyntheticProvider:
    calls = 0

    @classmethod
    def record_calls(cls):
        # 프로세스가 바뀐 뒤에도 이전 계수로 차이를 계산하지 않도록 시작 시 0과 PID를 게시한다.
        stats_path = synthetic_runtime_directory() / "synthetic_execution_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        pending = stats_path.with_suffix(f".{os.getpid()}.tmp")
        pending.write_text(
            json.dumps({"processId": os.getpid(), "modelCalls": cls.calls, "paidCalls": 0}),
            encoding="utf-8",
        )
        pending.replace(stats_path)

    async def execute_explicit(self, **kwargs):
        # 합성 호출 수만 남긴다. 요청 원문·자격증명·응답 내용은 기록하지 않는다.
        type(self).calls += 1
        self.record_calls()

        # 결합 검사의 명시적 표식만 고정 출력으로 받으며 실제 업무 판단은 호출 서비스에 둔다.
        local_smoke = execute_local_smoke_fixture(kwargs)
        if local_smoke is not None:
            return local_smoke

        # 검증된 원본 호출을 먼저 대조하며 알 수 없는 입력을 실 Provider로 넘기지 않는다.
        replay = execute_replay(kwargs)
        if replay is not None:
            return replay
        leveltest = execute_leveltest_fixture(kwargs)
        if leveltest is not None:
            return leveltest
        listening = await execute_listening_fixture(kwargs)
        if listening is not None:
            return listening
        writing = await execute_writing_fixture(kwargs)
        if writing is not None:
            return writing
        practice = await execute_practice_fixture(kwargs)
        if practice is not None:
            return practice
        speaking = execute_speaking_conversation_fixture(kwargs)
        if speaking is not None:
            return speaking
        evaluation = execute_speaking_evaluation_fixture(kwargs)
        if evaluation is not None:
            return evaluation
        speaking_runtime = await execute_speaking_runtime_fixture(kwargs)
        if speaking_runtime is not None:
            return speaking_runtime

        # deadline 검사는 HTTP 취소 경로를 그대로 지나도록 합성 호출만 지연한다.
        if kwargs.get("instructions") == "Synthetic instruction" and kwargs.get("messages") == [
            {"role": "user", "content": "Synthetic timeout"}
        ]:
            await asyncio.sleep(2)
            return StructuredGenerationResult(
                {"result": "synthetic"}, 7, 2, "test-provider", "synthetic-model"
            )
        raise ValueError("No validated synthetic execution fixture")


# 기존 fixture 계약은 유지하고 명시한 로컬 launcher의 공통 개발 키만 선택적으로 받는다.
settings.SERVER_API_KEY = os.environ.get(
    "TRANSLACAT_TEST_SERVER_API_KEY", "synthetic-local-model-key"
)
if not settings.SERVER_API_KEY:
    raise RuntimeError("Synthetic local execution requires a server API key")
SyntheticProvider.record_calls()
app.dependency_overrides[get_model_execution_provider] = lambda: SyntheticProvider()
speech_fixture = SyntheticSpeechProvider()
for dependency in (
    get_speech_execution_provider,
    get_speech_runtime,
    get_speaking_speech_runtime,
    get_speech_evidence_runtime,
):
    app.dependency_overrides[dependency] = lambda: speech_fixture


@app.middleware("http")
async def restrict_synthetic_execution(request, call_next):
    # 일반 업무 endpoint를 실수로 호출해도 호스트의 실 Provider로 우회하지 못하게 한다.
    allowed = {
        "/",
        "/internal/v1/model/execute",
        "/internal/v1/speech/synthesize",
        "/internal/v1/speech/normalize",
        "/internal/v1/speech/transcribe",
        "/internal/v1/speech/evidence",
        "/api/v1/account-book/receipts/runtime-identity",
    }
    if request.url.path not in allowed:
        return JSONResponse(status_code=404, content={"code": "SYNTHETIC_ENDPOINT_DISABLED"})
    return await call_next(request)
