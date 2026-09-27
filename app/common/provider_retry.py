import math


def retry_after_details(error: Exception) -> dict[str, int]:
    # 기존 typed cooldown과 SDK의 기술 header만 읽고 오류 메시지의 숫자는 해석하지 않는다.
    value = getattr(error, "retry_after_seconds", None)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        headers = getattr(getattr(error, "response", None), "headers", None)
        if headers is None:
            headers = getattr(error, "headers", None)
        try:
            milliseconds = headers.get("retry-after-ms") if headers is not None else None
            seconds = headers.get("retry-after") if headers is not None else None
            if milliseconds is not None:
                value = float(milliseconds) / 1000
            elif seconds is not None:
                value = float(seconds)
            else:
                return {}
        except (TypeError, ValueError, AttributeError, OverflowError):
            return {}
    if not math.isfinite(value) or value <= 0 or value > 2**31 - 1:
        return {}

    # 기존 Listening worker의 하루 상한과 올림을 HTTP 경계에서도 유지한다.
    return {"retryAfterSeconds": min(math.ceil(value), 86_400)}
