"""원문 없이 Provider 예외의 기술적 출처만 전달한다."""
import httpx
from openai import APIConnectionError, APITimeoutError


def provider_failure_origin(error: Exception) -> dict[str, str | int]:
    # SDK가 상태 없이 주는 표식은 고정 enum으로만 전달하며 원문은 경계를 넘기지 않는다.
    details = _exception_origin(error)
    message = f"{type(error).__name__} {error}".lower()
    if any(marker in message for marker in (
        "rate limit", "ratelimit", "resource_exhausted", "resource exhausted", "too many requests",
    )):
        details["failureSignal"] = "RATE_LIMIT"
    elif "deadline exceeded" in message:
        details["failureSignal"] = "DEADLINE"
    elif any(marker in message for marker in ("safety", "blocked", "prohibited content", "unsafe")):
        details["failureSignal"] = "SAFETY"
    return details


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
