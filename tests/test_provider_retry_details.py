from types import SimpleNamespace

import pytest

from app.common.provider_retry import retry_after_details


@pytest.mark.parametrize(
    "attribute,expected",
    [
        (1.25, 2),
        (100_000, 86_400),
        (True, None),
        (0, None),
        (-1, None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_retry_metadata_accepts_only_finite_typed_cooldown(attribute, expected):
    # 준비
    failure = RuntimeError("secret 900 seconds")
    failure.retry_after_seconds = attribute

    # 실행
    result = retry_after_details(failure)

    # 검증
    assert result == ({} if expected is None else {"retryAfterSeconds": expected})


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"retry-after-ms": "1250"}, 2),
        ({"retry-after": "2.2"}, 3),
        ({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}, None),
        ({"retry-after": "nan"}, None),
    ],
)
def test_sdk_numeric_headers_are_safely_bounded(headers, expected):
    # 준비
    failure = RuntimeError("private provider diagnostic")
    failure.response = SimpleNamespace(headers=headers)

    # 실행
    result = retry_after_details(failure)

    # 검증
    assert result == ({} if expected is None else {"retryAfterSeconds": expected})
