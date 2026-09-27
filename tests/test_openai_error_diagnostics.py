from __future__ import annotations

from app.ai.providers.openai.error_diagnostics import (
    safe_openai_error_metadata,
    schema_fingerprint,
)


class _Response:
    status_code = 400
    headers = {"x-request-id": "req_test-123"}


class _BadRequest(Exception):
    status_code = 400
    response = _Response()
    body = {
        "message": (
            "Invalid schema for response_format 'writing': "
            "In context=('properties', 'items'), 'additionalProperties' is required."
        ),
        "type": "invalid_request_error",
        "param": "text.format.schema",
        "code": "invalid_json_schema",
    }


def test_safe_openai_error_metadata_keeps_diagnostic_fields_without_raw_body() -> None:
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array"}},
        "required": ["items"],
        "additionalProperties": False,
    }
    metadata = safe_openai_error_metadata(_BadRequest(), schema=schema)

    assert metadata == {
        "status_code": 400,
        "provider_code": "invalid_json_schema",
        "provider_type": "invalid_request_error",
        "provider_param": "text.format.schema",
        "provider_request_id": "req_test-123",
        "provider_message_hint": (
            "Invalid schema for response_format 'writing': "
            "In context=('properties', 'items'), 'additionalProperties' is required."
        ),
        "structured": True,
        "schema_sha256": schema_fingerprint(schema),
    }


def test_safe_openai_error_metadata_redacts_secret_like_values() -> None:
    exc = _BadRequest()
    exc.body = {
        "message": "Bearer abc.def.ghi sk-secret123 data:image/png;base64,AAAA verylong=" + "A" * 100,
        "type": "invalid_request_error",
        "param": "input",
        "code": "bad_request",
    }

    metadata = safe_openai_error_metadata(exc, schema={"type": "object"})
    hint = metadata["provider_message_hint"]

    assert hint is not None
    assert "abc.def.ghi" not in hint
    assert "sk-secret123" not in hint
    assert "data:image" not in hint
    assert "A" * 80 not in hint
    assert "<redacted>" in hint


def test_schema_fingerprint_is_stable_and_does_not_expose_schema() -> None:
    left = {"b": 2, "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": 2}

    fingerprint = schema_fingerprint(left)
    assert fingerprint == schema_fingerprint(right)
    assert fingerprint is not None
    assert len(fingerprint) == 24
    assert "x" not in fingerprint


def test_unstructured_error_has_no_schema_fingerprint() -> None:
    metadata = safe_openai_error_metadata(RuntimeError("boom"), schema=None)
    assert metadata["structured"] is False
    assert metadata["schema_sha256"] is None
    assert metadata["provider_message_hint"] is None
