"""OpenAI 오류를 사용자 데이터 없이 진단하기 위한 안전한 메타데이터 추출기."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+\-/=]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"data:[^\s]+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9+/=_-]{80,}\b"),
)
_LONG_SINGLE_QUOTED = re.compile(r"'[^'\r\n]{161,}'")
_LONG_DOUBLE_QUOTED = re.compile(r'"[^"\r\n]{161,}"')


def schema_fingerprint(schema: dict | None) -> str | None:
    """Schema 원문 대신 재현 가능한 짧은 hash만 로그에 남긴다."""
    if schema is None:
        return None
    try:
        payload = json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
    except Exception:
        return "unavailable"
    return hashlib.sha256(payload).hexdigest()[:24]


def safe_openai_error_metadata(exc: Exception, *, schema: dict | None) -> dict[str, Any]:
    """Provider body 전체를 노출하지 않고 오류 분류에 필요한 필드만 반환한다."""
    response = getattr(exc, "response", None)
    status_code = _first_int(
        getattr(exc, "status_code", None),
        getattr(response, "status_code", None),
    )
    body = getattr(exc, "body", None)
    error = _error_mapping(body)

    code = _safe_scalar(error.get("code") if error else None) or _safe_scalar(
        getattr(exc, "code", None)
    )
    param = _safe_scalar(error.get("param") if error else None) or _safe_scalar(
        getattr(exc, "param", None)
    )
    provider_type = _safe_scalar(error.get("type") if error else None) or _safe_scalar(
        getattr(exc, "type", None)
    )
    request_id = _safe_request_id(
        getattr(exc, "request_id", None)
        or _header_value(response, "x-request-id")
        or _header_value(response, "request-id")
    )
    message = _safe_message_hint(
        error.get("message") if error else getattr(exc, "message", None)
    )

    return {
        "status_code": status_code,
        "provider_code": code,
        "provider_type": provider_type,
        "provider_param": param,
        "provider_request_id": request_id,
        "provider_message_hint": message,
        "structured": schema is not None,
        "schema_sha256": schema_fingerprint(schema),
    }


def _error_mapping(body: Any) -> Mapping[str, Any] | None:
    if not isinstance(body, Mapping):
        return None
    nested = body.get("error")
    if isinstance(nested, Mapping):
        return nested
    return body


def _first_int(*values: Any) -> int | None:
    for value in values:
        if type(value) is int and 100 <= value <= 599:
            return value
    return None


def _safe_scalar(value: Any, *, max_length: int = 160) -> str | None:
    if not isinstance(value, (str, int, float, bool)):
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("<redacted>", text)
    return text[:max_length]


def _safe_request_id(value: Any) -> str | None:
    text = _safe_scalar(value, max_length=160)
    if text is None:
        return None
    return re.sub(r"[^A-Za-z0-9._:-]", "?", text)


def _safe_message_hint(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("<redacted>", text)
    text = _LONG_SINGLE_QUOTED.sub("'<redacted-long-value>'", text)
    text = _LONG_DOUBLE_QUOTED.sub('"<redacted-long-value>"', text)
    return text[:700]


def _header_value(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    try:
        value = getter(name)
    except Exception:
        return None
    return value if isinstance(value, str) else None
