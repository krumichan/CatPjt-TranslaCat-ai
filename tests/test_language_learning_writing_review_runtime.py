"""Review diagnostics and retry contracts. No network/model-quality assertions."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import pytest

from app.ai.ports import StructuredGenerationResult
from app.ai.providers.openai.response import OpenAIProviderResponseError, decode_response
from app.ai.providers.openai.schema import build_openai_text_config
from app.features.language_learning.writing.generation_contract import parse_draft, review_content_hash
from app.features.language_learning.writing.note_localization import LocalizedWritingNote, NOTE_LOCALIZATION_TASK
from app.features.language_learning.writing.review_diagnostics import (
    WritingProviderConfigurationError, WritingVerificationUnavailableError,
    classify_exception, retry_after_seconds,
)
from app.features.language_learning.writing.review_runtime import ReviewContext, ReviewRuntime
from app.features.language_learning.writing.verification import (
    TaskReview, NoteReview,
    MINI_QUALITY_TASK, MINI_NOTE_TASK, MINI_DIFFICULTY_TASK,
)
from app.features.language_learning.writing.verification_prompts import build_writing_review_prompt
from tests.test_language_learning_writing_verified_generation import draft, request
from tests.writing_generation_fakes import WritingPipelineProvider, reject_issues, unsure_review
from app.features.language_learning.writing.assessment_contract import binding_failure, build_review_schema


def setup_runtime(overrides=None, **options):
    req = request()
    item = parse_draft(draft())
    context = ReviewContext(req.request_id, "candidate-A", review_content_hash(req, item), 1, 3)
    prompt = build_writing_review_prompt(req, item, candidate_id=context.candidate_id, content_hash=context.content_hash)
    provider = WritingPipelineProvider([], {item.origin_text: 3}, {MINI_QUALITY_TASK: overrides} if overrides is not None else {})
    runtime = ReviewRuntime(provider, timeout_seconds=options.pop("timeout_seconds", 0.2),
                            retry_base_seconds=options.pop("retry_base_seconds", 0), **options)
    kwargs = dict(task=MINI_QUALITY_TASK, prompt=prompt, model=TaskReview, context=context,
                  inspect=lambda result: binding_failure(result, item, context.candidate_id, context.content_hash),
                  schema=build_review_schema(TaskReview, item))
    return runtime, provider, kwargs


def events(caplog, event="writing.verifier.finished"):
    return [record.writing_event for record in caplog.records
            if hasattr(record, "writing_event") and record.writing_event["event"] == event]


@pytest.mark.parametrize("override,code", [
    ({"confidence": float("nan")}, "VERIFIER_INVALID_CONFIDENCE"),
    ({"confidence": float("inf")}, "VERIFIER_INVALID_CONFIDENCE"),
    ({"confidence": -0.1}, "VERIFIER_INVALID_CONFIDENCE"),
    ({"difficultyConfidence": "0.9"}, "VERIFIER_INVALID_CONFIDENCE"),
    ({"confidence": True}, "VERIFIER_INVALID_CONFIDENCE"),
    ({"estimatedBand": "3"}, "VERIFIER_INVALID_BAND"),
    ({"estimatedBand": 0}, "VERIFIER_INVALID_BAND"),
    ({"estimatedBand": None}, "VERIFIER_INCONSISTENT_RESULT"),
    ({"verdict": "ACCEPT"}, "VERIFIER_INVALID_VERDICT"),
    ({"candidateId": "somebody-else"}, "VERIFIER_IDENTITY_MISMATCH"),
    ({"contentHash": "0" * 64}, "VERIFIER_CONTENT_HASH_MISMATCH"),
    ({"contentHash": "broken"}, "VERIFIER_CONTENT_HASH_MISMATCH"),
    ({"difficultyEvidenceSegmentIds": []}, "VERIFIER_INCONSISTENT_RESULT"),
    ({"issues": ["ANSWER_LEAK"]}, "VERIFIER_INCONSISTENT_RESULT"),
    ({"private-extra": "secret"}, "VERIFIER_SCHEMA_INVALID"),
    ({"difficultyEvidenceSegmentIds": ["N1"]}, "VERIFIER_EVIDENCE_SCOPE_MISMATCH"),
    ({"difficultyEvidenceSegmentIds": ["O99"]}, "VERIFIER_EVIDENCE_SEGMENT_INVALID"),
])
def test_protocol_failure_has_exact_code_and_retries_same_candidate_once(override, code, caplog):
    runtime, provider, kwargs = setup_runtime(copy.deepcopy(override))
    with caplog.at_level(logging.INFO), pytest.raises(WritingVerificationUnavailableError) as exc:
        asyncio.run(runtime.review(**kwargs))
    assert exc.value.failure.code == code
    assert exc.value.attempts == 2
    assert len(provider.calls) == 2
    assert provider.calls[0]["data"] == provider.calls[1]["data"]
    assert provider.calls[0]["schema"] == provider.calls[1]["schema"]
    logs = events(caplog)
    assert [e["failure_code"] for e in logs] == [code, code]
    assert [e["retry_scheduled"] for e in logs] == [True, False]
    assert all(e["failure_category"] == "PROTOCOL" for e in logs)
    assert all(e["duration_ms"] >= 0 for e in logs)
    assert runtime.metrics.calls[MINI_QUALITY_TASK] == 2
    assert runtime.metrics.retries[MINI_QUALITY_TASK] == 1
    assert runtime.metrics.active == 0


@pytest.mark.parametrize("override", [unsure_review(), {"confidence": 0.0}, {"difficultyConfidence": 0.7}])
def test_valid_uncertainty_and_low_confidence_are_not_provider_failures_or_retries(override, caplog):
    runtime, provider, kwargs = setup_runtime(override)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(runtime.review(**kwargs))
    assert result is not None
    assert len(provider.calls) == 1
    assert events(caplog)[0]["failure_code"] is None
    assert events(caplog)[0]["confidence_used_for_acceptance"] is False


def test_firm_reject_is_a_valid_result_never_retried(caplog):
    runtime, provider, kwargs = setup_runtime(reject_issues("ANSWER_LEAK"))
    with caplog.at_level(logging.INFO):
        result = asyncio.run(runtime.review(**kwargs))
    assert result.verdict == "REJECT"
    assert len(provider.calls) == 1
    assert events(caplog)[0]["verdict"] == "REJECT"
    assert events(caplog)[0]["failure_code"] is None


@pytest.mark.parametrize("bad,code", [(None, "VERIFIER_EMPTY_RESPONSE"), ({}, "VERIFIER_EMPTY_RESPONSE"),
                                      ("", "VERIFIER_EMPTY_RESPONSE"), ("{}", "VERIFIER_RESPONSE_TYPE_INVALID"),
                                      ([], "VERIFIER_RESPONSE_TYPE_INVALID"), (True, "VERIFIER_RESPONSE_TYPE_INVALID")])
def test_raw_output_shape_is_not_silently_coerced(bad, code):
    runtime, provider, kwargs = setup_runtime(lambda *_: copy.deepcopy(bad))
    with pytest.raises(WritingVerificationUnavailableError) as exc:
        asyncio.run(runtime.review(**kwargs))
    assert exc.value.failure.code == code
    assert len(provider.calls) == 2


class StatusError(Exception):
    def __init__(self, status, headers=None):
        super().__init__("NEVER_LOG_BODY secret-token=abc")
        self.status_code = status
        self.headers = headers


@pytest.mark.parametrize("exc,code", [
    (TimeoutError("secret timeout body"), "VERIFIER_TIMEOUT"),
    (ConnectionError("secret socket"), "VERIFIER_CONNECTION_ERROR"),
    (httpx.ReadTimeout("secret"), "VERIFIER_TIMEOUT"),
    (httpx.ConnectError("secret"), "VERIFIER_CONNECTION_ERROR"),
    (StatusError(429), "VERIFIER_RATE_LIMITED"),
    (StatusError(503), "VERIFIER_PROVIDER_ERROR"),
    (StatusError(500), "VERIFIER_PROVIDER_ERROR"),
    (StatusError(408), "VERIFIER_PROVIDER_ERROR"),
    (ValueError("secret output"), "VERIFIER_INVALID_RESULT"),
    (json.JSONDecodeError("secret", "secret document", 0), "VERIFIER_JSON_INVALID"),
])
def test_transient_exception_retries_only_same_stage_then_recovers(exc, code, caplog):
    runtime, provider, kwargs = setup_runtime([exc, {}])
    with caplog.at_level(logging.INFO):
        result = asyncio.run(runtime.review(**kwargs))
    assert result.verdict == "PASS"
    assert len(provider.calls) == 2
    assert events(caplog)[0]["failure_code"] == code
    assert events(caplog)[1]["failure_code"] is None
    assert "secret" not in caplog.text
    assert "NEVER_LOG_BODY" not in caplog.text


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 413, 422])
def test_configuration_error_aborts_without_model_retry(status, caplog):
    runtime, provider, kwargs = setup_runtime(StatusError(status))
    with caplog.at_level(logging.INFO), pytest.raises(WritingProviderConfigurationError):
        asyncio.run(runtime.review(**kwargs))
    assert len(provider.calls) == 1
    assert events(caplog)[0]["failure_code"] == "VERIFIER_PROVIDER_CONFIGURATION"
    assert events(caplog)[0]["status_code"] == status
    assert "secret" not in caplog.text


def test_unknown_programming_error_is_not_hidden_as_candidate_failure(caplog):
    runtime, provider, kwargs = setup_runtime(KeyError("secret programmer error"))
    with caplog.at_level(logging.INFO), pytest.raises(KeyError):
        asyncio.run(runtime.review(**kwargs))
    assert len(provider.calls) == 1
    assert events(caplog)[0]["failure_code"] == "VERIFIER_UNEXPECTED_ERROR"
    assert "secret programmer" not in caplog.text


def test_schema_error_locations_never_expose_extra_key_or_input(caplog):
    secret = "SECRET_EMAIL_user@example.com\nPROMPT_BODY"
    runtime, _, kwargs = setup_runtime({secret: {"nested": secret}, "confidence": secret})
    with caplog.at_level(logging.INFO), pytest.raises(WritingVerificationUnavailableError):
        asyncio.run(runtime.review(**kwargs))
    assert secret not in caplog.text
    assert "SECRET_EMAIL" not in caplog.text
    assert "<extra>" in caplog.text
    for record in caplog.records:
        if hasattr(record, "writing_event"):
            assert json.loads(record.getMessage()) == record.writing_event
            assert "\n" not in record.getMessage()
            assert "input" not in {key for error in record.writing_event.get("schema_errors", []) for key in error}


@pytest.mark.parametrize("headers,value", [
    ({"retry-after": "0.02"}, 0.02), ({"retry-after-ms": "20"}, 0.02),
    ({"retry-after": "nan"}, None), ({"retry-after-ms": "inf"}, None),
    ({"retry-after": "-2"}, 0), ({"retry-after": "wrong"}, None),
])
def test_retry_after_is_bounded_finite_and_understands_milliseconds(headers, value):
    assert retry_after_seconds(StatusError(429, headers)) == value


def test_http_date_retry_after_is_respected():
    date = datetime.now(timezone.utc) + timedelta(seconds=60)
    delay = retry_after_seconds(StatusError(429, {"retry-after": format_datetime(date, usegmt=True)}))
    assert 58 < delay <= 60


def test_retry_after_too_long_aborts_instead_of_hammering_provider(caplog):
    runtime, provider, kwargs = setup_runtime(StatusError(429, {"retry-after": "120"}))
    with caplog.at_level(logging.INFO), pytest.raises(WritingVerificationUnavailableError) as exc:
        asyncio.run(runtime.review(**kwargs))
    assert exc.value.attempts == 1
    assert len(provider.calls) == 1
    assert events(caplog)[0]["retry_stop_reason"] == "RETRY_AFTER_TOO_LONG"


def test_remaining_request_budget_prevents_retry_wait(caplog):
    runtime, provider, kwargs = setup_runtime(StatusError(503), deadline=time.monotonic() + 0.15,
                                              retry_base_seconds=0.25)
    with caplog.at_level(logging.INFO), pytest.raises(WritingVerificationUnavailableError):
        asyncio.run(runtime.review(**kwargs))
    assert len(provider.calls) == 1
    assert events(caplog)[0]["retry_stop_reason"] == "REQUEST_BUDGET"


def test_expired_request_never_starts_a_call():
    runtime, provider, kwargs = setup_runtime(deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        asyncio.run(runtime.review(**kwargs))
    assert not provider.calls


def test_retry_after_short_delay_is_not_shortened():
    runtime, provider, kwargs = setup_runtime([StatusError(429, {"retry-after-ms": "20"}), {}])
    start = time.monotonic()
    assert asyncio.run(runtime.review(**kwargs)).verdict == "PASS"
    assert time.monotonic() - start >= 0.019
    assert runtime.metrics.backoff_ms >= 19
    assert len(provider.calls) == 2


def test_request_bound_schema_echoes_identity_without_exposing_target():
    runtime, provider, kwargs = setup_runtime()
    asyncio.run(runtime.review(**kwargs))
    schema = provider.calls[0]["schema"]
    assert schema["properties"]["candidateId"]["enum"] == [kwargs["context"].candidate_id]
    assert schema["properties"]["contentHash"]["enum"] == [kwargs["context"].content_hash]
    assert "enum" not in TaskReview.model_json_schema(by_alias=True)["properties"]["candidateId"]
    assert "enum" not in schema["properties"]["estimatedBand"]
    assert "targetBand" not in json.dumps(schema)


def test_schema_is_copied_for_each_retry_even_if_provider_mutates_it():
    runtime, provider, kwargs = setup_runtime([ValueError(), {}])
    original = provider.call
    async def mutate(*args, **kw):
        result = None
        try:
            result = await original(*args, **kw)
        finally:
            kw["schema"]["properties"]["candidateId"]["enum"] = ["corrupted"]
        return result
    provider.call = mutate
    result = asyncio.run(runtime.review(**kwargs))
    assert result.verdict == "PASS"
    assert provider.calls[0]["schema"] == provider.calls[1]["schema"]


def test_metadata_is_taken_from_actual_success_not_guessed_from_pool(caplog):
    runtime, provider, kwargs = setup_runtime()
    original = provider.call
    class MetadataProvider:
        provider_name = "pool"
        async def call(self, **kw):
            raise AssertionError("Do not call twice")
        async def call_with_metadata(self, **kw):
            return StructuredGenerationResult(await original(**kw), 32, 45, "fallback-provider", "actual-model")
    runtime.provider = MetadataProvider()
    with caplog.at_level(logging.INFO):
        asyncio.run(runtime.review(**kwargs))
    result = events(caplog)[0]
    assert (result["provider_route"], result["provider"], result["model"]) == ("pool", "fallback-provider", "actual-model")
    assert (result["input_tokens"], result["output_tokens"]) == (32, 45)


@pytest.mark.parametrize("task,model", [
    (MINI_QUALITY_TASK, TaskReview),
    (MINI_DIFFICULTY_TASK, TaskReview), (MINI_NOTE_TASK, NoteReview),
    (NOTE_LOCALIZATION_TASK, LocalizedWritingNote),
])
def test_only_closed_review_schemas_enable_openai_strict_mode(task, model):
    config = build_openai_text_config(type_name=task, schema=model.model_json_schema(by_alias=True), verbosity="low")
    assert config["format"]["strict"] is True
    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)
    check(config["format"]["schema"])


def test_review_schema_is_not_silently_downgraded_to_nonstrict():
    with pytest.raises(ValueError):
        build_openai_text_config(type_name=MINI_QUALITY_TASK, schema={"type": "object", "properties": {}}, verbosity="low")


@pytest.mark.parametrize("response,reason,retryable", [
    (SimpleNamespace(status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens")), "OUTPUT_TOKEN_LIMIT", False),
    (SimpleNamespace(status="incomplete", incomplete_details={"reason": "content_filter"}), "REFUSAL", False),
    (SimpleNamespace(status="incomplete"), "RESPONSE_INCOMPLETE", True),
    (SimpleNamespace(status="completed", output=[{"content": [{"type": "refusal", "refusal": "private"}]}]), "REFUSAL", False),
    (SimpleNamespace(status="completed", output_text=""), "EMPTY_OUTPUT", True),
    (SimpleNamespace(status="completed", output_text="private invalid JSON"), "JSON_INVALID", True),
])
def test_response_decoder_exposes_safe_failure_codes(response, reason, retryable):
    with pytest.raises(OpenAIProviderResponseError) as exc:
        decode_response(response, structured=True)
    assert exc.value.reason_code == reason
    assert exc.value.retryable == retryable
    assert "private" not in str(exc.value)
    assert classify_exception(exc.value).code.startswith("VERIFIER_")


@pytest.mark.parametrize("structured,expected", [(True, {"ok": True}), (False, '{"ok":true}')])
def test_response_decoder_does_not_change_successful_payloads(structured, expected):
    assert decode_response(SimpleNamespace(status="completed", output_text='{"ok":true}'), structured=structured) == expected


@pytest.mark.parametrize("exc", [OpenAIProviderResponseError("limited", reason_code="OUTPUT_TOKEN_LIMIT", retryable=False),
                                  OpenAIProviderResponseError("declined", reason_code="REFUSAL", retryable=False)])
def test_nonretryable_provider_output_does_not_repeat_or_fail_over(exc):
    from app.ai.provider_pool import AiProviderPool
    assert AiProviderPool._is_transient_provider_error(exc) is False
    runtime, provider, kwargs = setup_runtime(exc)
    with pytest.raises(WritingVerificationUnavailableError) as raised:
        asyncio.run(runtime.review(**kwargs))
    assert raised.value.attempts == 1
    assert raised.value.failure.retryable is False
    assert len(provider.calls) == 1


def test_semantic_diagnostics_include_validated_confidence_without_evidence_text(caplog):
    runtime, _, kwargs = setup_runtime({"confidence": 0.6, "difficultyConfidence": 0.7})
    with caplog.at_level(logging.INFO):
        assert asyncio.run(runtime.review(**kwargs)).verdict == "PASS"
    log = events(caplog)[0]
    assert log["confidence"] == 0.6 and log["difficulty_confidence"] == 0.7
    assert log["estimated_band"] == 3 and log["verdict"] == "PASS"
    assert "quote" not in log and "evidence" not in log


@pytest.mark.parametrize("bad_count", [float("nan"), float("inf"), -1, True, "5"])
def test_invalid_token_metadata_cannot_break_diagnostics_or_approval(bad_count, caplog):
    runtime, provider, kwargs = setup_runtime()
    async def metadata_call(type_name, data, schema):
        raw = await provider.call(type_name, data, schema)
        return StructuredGenerationResult(data=raw, provider="fake", model="fake-model",
                                          input_tokens=bad_count, output_tokens=bad_count)
    provider.call_with_metadata = metadata_call
    with caplog.at_level(logging.INFO):
        assert asyncio.run(runtime.review(**kwargs)).verdict == "PASS"
    assert events(caplog)[0]["input_tokens"] is None
    assert events(caplog)[0]["output_tokens"] is None


def test_sdk_specific_error_shapes_are_classified_without_importing_sdk():
    class APIError(Exception):
        code = 503
    class APITimeoutError(Exception):
        pass
    class ReadTimeout(Exception):
        pass
    assert classify_exception(APIError()).code == "VERIFIER_PROVIDER_ERROR"
    assert classify_exception(APITimeoutError()).code == "VERIFIER_TIMEOUT"
    assert classify_exception(ReadTimeout()).code == "VERIFIER_TIMEOUT"
