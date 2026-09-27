from fastapi import HTTPException

from app.common.provider_failure import provider_failure_origin
from app.common.provider_retry import retry_after_details


def provider_failure(error: Exception) -> HTTPException:
    origin = provider_failure_origin(error)
    # SDK마다 다른 상태 표현을 읽되 원문과 request 본문은 응답에 넣지 않는다.
    status = None
    for attribute in ("status_code", "status", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            status = value
            break
        if isinstance(value, str) and value.isdigit():
            status = int(value)
            break
    if status is None:
        value = getattr(getattr(error, "response", None), "status_code", None)
        status = value if isinstance(value, int) else None
    message = f"{type(error).__name__} {error}".lower()

    # 기존 Provider 기술 실패 분류만 보존한다. 학습 재시도와 평가 불가 여부는 LL에서 결정한다.
    if status == 429 or any(
        marker in message
        for marker in (
            "rate limit",
            "ratelimit",
            "resource_exhausted",
            "resource exhausted",
            "too many requests",
        )
    ):
        return HTTPException(
            429,
            detail={
                "code": "PROVIDER_RATE_LIMITED",
                "retryable": True,
                **retry_after_details(error),
                **origin,
            },
        )
    if isinstance(error, TimeoutError) or status in {408, 504} or "deadline exceeded" in message:
        return HTTPException(504, detail={"code": "PROVIDER_TIMEOUT", "retryable": True, **origin})
    if status in {400, 401, 403} or isinstance(error, ValueError):
        return HTTPException(
            400, detail={"code": "STT_EXECUTION_FAILED", "retryable": False, **origin},
        )
    if any(marker in message for marker in ("safety", "blocked", "prohibited content", "unsafe")):
        return HTTPException(422, detail={"code": "PROVIDER_REFUSAL", "retryable": False, **origin})
    return HTTPException(503, detail={"code": "STT_EXECUTION_FAILED", "retryable": True, **origin})
