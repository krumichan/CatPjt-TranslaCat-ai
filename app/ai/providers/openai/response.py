"""OpenAI response decoding with safe reason codes; no SDK import or raw-body logs."""
from __future__ import annotations

import json
from typing import Any


class OpenAIProviderResponseError(RuntimeError):
    status_code = 502

    def __init__(self, message: str, *, reason_code: str = "RESPONSE_INCOMPLETE", retryable: bool = True) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.retryable = retryable


def decode_response(response: Any, *, structured: bool) -> Any:
    status = getattr(response, "status", "completed") or "completed"
    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None)
    if isinstance(incomplete, dict):
        reason = incomplete.get("reason")
    if status != "completed":
        if reason == "max_output_tokens":
            # Repeating the same token budget is not a remedy. Do not multiply calls.
            raise OpenAIProviderResponseError("OpenAI output exceeded its token limit",
                                              reason_code="OUTPUT_TOKEN_LIMIT", retryable=False)
        if reason == "content_filter":
            raise OpenAIProviderResponseError("OpenAI declined the request",
                                              reason_code="REFUSAL", retryable=False)
        raise OpenAIProviderResponseError("OpenAI response did not complete")
    for output in getattr(response, "output", None) or []:
        content = (output.get("content") or []) if isinstance(output, dict) else getattr(output, "content", None) or []
        for item in content:
            kind = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if kind == "refusal":
                raise OpenAIProviderResponseError("OpenAI declined the request",
                                                  reason_code="REFUSAL", retryable=False)
    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or not text.strip():
        raise OpenAIProviderResponseError("OpenAI response has no output text", reason_code="EMPTY_OUTPUT")
    if not structured:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenAIProviderResponseError("OpenAI structured response is not valid JSON",
                                          reason_code="JSON_INVALID") from exc
