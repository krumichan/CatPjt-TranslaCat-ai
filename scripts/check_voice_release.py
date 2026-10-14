"""Verify the approved production Voice model without logging credentials or content."""

from __future__ import annotations

import json
import os
import sys
from urllib.request import Request, urlopen


def verify_voice_release() -> None:
    # 비밀 값은 인증 헤더로만 사용한다. 설정 누락은 실제 배포 성공으로 취급하지 않는다.
    if os.environ.get("AI_VOICE_STT_PROVIDER") != "openai":
        raise RuntimeError("Production Voice provider must be openai")
    if os.environ.get("AI_VOICE_REALTIME_PREVIEW_ENABLED") != "true":
        raise RuntimeError("Production Voice realtime preview must be enabled")
    key = os.environ.get("SERVER_API_KEY", "").strip()
    if not key or not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("Production Voice credentials are not configured")
    request = Request(
        "http://127.0.0.1:8000/internal/v1/voice/readiness",
        headers={"X-API-KEY": key},
    )
    with urlopen(request, timeout=3) as response:
        state = json.load(response)

    # 공개 HTTP health나 환경 문자열 대신 실행 중 서비스의 선택 모델을 검증한다.
    if not isinstance(state, dict) or (
        state.get("ready") is not True
        or state.get("acceptingStreams") is not True
        or state.get("sttModel") != "whisper-1@openai"
    ):
        raise RuntimeError("Production Voice readiness/model verification failed")


if __name__ == "__main__":
    try:
        verify_voice_release()
    except Exception as error:
        print(f"Voice release verification failed ({type(error).__name__})", file=sys.stderr)
        raise SystemExit(1) from None
    print("Voice release verified: whisper-1@openai")
