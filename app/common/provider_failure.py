"""원문 없이 Provider 예외의 기술적 출처만 전달한다."""

import httpx
from openai import APIConnectionError, APITimeoutError


# 예외를 재시도·안전 정책이 해석할 수 있는 제한된 필드로 변환한다.
def provider_failure_origin(error: Exception) -> dict[str, str | int]:
    # 예외 타입과 HTTP 응답을 기준으로 기술적 출처를 기록한다.
    details = _exception_origin(error)

    # SDK 메시지는 분류에만 쓰고, 응답에는 고정된 신호만 담는다.
    message = f"{type(error).__name__} {error}".lower()
    if any(
        marker in message
        for marker in (
            "rate limit",
            "ratelimit",
            "resource_exhausted",
            "resource exhausted",
            "too many requests",
        )
    ):
        details["failureSignal"] = "RATE_LIMIT"
    elif "deadline exceeded" in message:
        details["failureSignal"] = "DEADLINE"
    elif any(marker in message for marker in ("safety", "blocked", "prohibited content", "unsafe")):
        details["failureSignal"] = "SAFETY"

    return details


# SDK, 전송 계층, HTTP 상태, 입력 오류 순으로 출처를 판정한다.
def _exception_origin(error: Exception) -> dict[str, str | int]:
    # SDK timeout은 일반 deadline 초과와 구분해야 호출자가 기존 정책을 유지할 수 있다.
    if isinstance(error, APITimeoutError):
        return {"failureKind": "SDK_TIMEOUT"}
    if isinstance(error, APIConnectionError):
        return {"failureKind": "SDK_CONNECTION"}
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return {"failureKind": "TIMEOUT"}
    if isinstance(error, (ConnectionError, httpx.TransportError)):
        return {"failureKind": "CONNECTION"}

    # 오류 문장 대신 안전한 HTTP 상태와 값 오류 여부만 남긴다.
    values = [getattr(error, key, None) for key in ("status_code", "status", "code")]
    values.append(getattr(getattr(error, "response", None), "status_code", None))
    for value in values:
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool) and 400 <= value <= 599:
            return {"failureKind": "HTTP_STATUS", "providerStatus": value}
    if isinstance(error, ValueError):
        return {"failureKind": "VALUE_ERROR"}
    return {}
