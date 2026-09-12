"""Safe structured diagnostic primitives with caller-owned event namespaces."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Iterable


def safe_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value):
        return value
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def safe_token_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def emit_structured_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    schema_version: int = 1,
    record_attribute: str | None = None,
    identifier_fields: Iterable[str] = (),
    **fields: Any,
) -> None:
    values = dict(fields)
    for name in identifier_fields:
        value = values.get(name)
        if value is not None:
            values[name] = safe_identifier(str(value))
    payload = {"schema_version": schema_version, "event": event, **values}
    extra = {record_attribute: payload} if record_attribute is not None else None
    logger.log(
        level,
        "%s",
        json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        extra=extra,
    )
