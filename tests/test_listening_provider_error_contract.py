import math

import pytest

from app.ai.providers.gemini.client import GeminiTtsQuotaCooldownError
from app.features.language_learning.listening.provider_error import map_provider_exception
from app.schemas.language_learning_listening import ListeningErrorCode, ListeningStage


def _mapped(error: Exception):
    return map_provider_exception(
        error,
        stage=ListeningStage.TTS,
        fallback_code=ListeningErrorCode.TTS_FAILED,
        fallback_message="Reference TTS failed.",
    )


def test_typed_quota_cooldown_preserves_code_and_durable_retry_delay():
    error = GeminiTtsQuotaCooldownError(36660.8)
    # Message is not a policy input; use an opaque value to prove typed handling.
    error.args = ("opaque provider diagnostic",)
    result = _mapped(error)
    assert result.code == ListeningErrorCode.PROVIDER_RATE_LIMITED
    assert result.retryable is True
    assert result.details == {"retryAfterSeconds": 36661}
    assert result.to_schema().model_dump(by_alias=True)["details"] == result.details


@pytest.mark.parametrize("delay", [None, True, -1, 0, math.inf, math.nan, 2**63, "36661s"])
def test_invalid_cooldown_metadata_does_not_escape_to_wire(delay):
    class TypedRateLimit(Exception):
        status_code = 429
        retry_after_seconds = delay

    result = _mapped(TypedRateLimit("opaque"))
    assert result.code == ListeningErrorCode.PROVIDER_RATE_LIMITED
    assert result.retryable is True
    assert result.details == {}


@pytest.mark.parametrize(
    ("status", "code"),
    [(408, "PROVIDER_TIMEOUT"), (504, "PROVIDER_TIMEOUT"), (503, "PROVIDER_UNAVAILABLE")],
)
def test_existing_typed_infrastructure_classification_is_preserved(status, code):
    class TypedFailure(Exception):
        status_code = status

    result = _mapped(TypedFailure("opaque"))
    assert result.code.value == code
    assert result.retryable is True
