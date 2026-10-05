"""합성 실행의 제어 파일과 로그를 선택한 로컬 작업 디렉터리에 모은다."""

import os
from pathlib import Path


def synthetic_runtime_directory() -> Path:
    configured = os.environ.get("TRANSLACAT_TEST_RUNTIME_DIRECTORY")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            raise ValueError("Synthetic runtime directory must be absolute")
        return path
    return Path(__file__).resolve().parents[2] / ".codex-workspace/verification/ai/runtime"
