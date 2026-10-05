import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.internal.model_profile_execution import get_profile_executor
from app.core.config import settings


def command(**overrides):
    result = {
        "traceId": "child-1",
        "instructions": "Synthetic instruction",
        "messages": [{"role": "user", "content": "Synthetic source"}],
        "profileId": "luna-6-none-v1",
        "maxOutputTokens": 32768,
        "remainingMilliseconds": 30000,
        "maxProviderCalls": 1,
    }
    result.update(overrides)
    return result


def profile_scope():
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/internal/v1/model/execute-profile",
        "raw_path": b"/internal/v1/model/execute-profile",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"x-api-key", b"synthetic-novel-key"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }


@pytest.fixture
def harness(monkeypatch):
    # 준비
    with patch("app.core.config_logger.setup_logging"):
        from app.main import app
    monkeypatch.setattr(settings, "SERVER_API_KEY", "synthetic-legacy-key")
    monkeypatch.setattr(settings, "NOVEL_AUTH_ENVIRONMENT", "Development")
    monkeypatch.setattr(settings, "NOVEL_SERVER_API_KEY", SecretStr("synthetic-novel-key"))
    executor = AsyncMock()
    executor.execute.return_value = {
        "output": "value",
        "inputTokens": 10,
        "outputTokens": 10,
        "provider": "openai",
        "model": "gpt-6-luna",
        "providerCalls": 1,
        "profileId": "luna-6-none-v1",
        "profileVersion": "2026-10-04.1",
        "reasoningEffort": "none",
        "serviceTier": "default",
        "cachedInputTokens": 0,
        "reasoningTokens": 0,
        "metadata": {"providerFirstTokenMs": None},
    }
    app.dependency_overrides[get_profile_executor] = lambda: executor
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield app, client, executor
    finally:
        client.close()
        app.dependency_overrides.pop(get_profile_executor, None)


@pytest.mark.parametrize("key", ["synthetic-legacy-key", "incorrect"])
def test_profiles_reject_non_novel_keys_before_provider(harness, key):
    # 실행
    _, client, executor = harness
    response = client.post(
        "/internal/v1/model/execute-profile", json=command(), headers={"X-API-KEY": key}
    )

    # 검증
    assert response.status_code == 401
    executor.execute.assert_not_awaited()


@pytest.mark.parametrize(
    "overrides",
    [
        {"profileId": "arbitrary"},
        {"model": "gpt-6.1-sol"},
        {"reasoningEffort": "high"},
        {"maxOutputTokens": 32769},
        {"maxProviderCalls": 2},
    ],
)
def test_profiles_fail_closed_on_unallowlisted_configuration(harness, overrides):
    # 실행
    _, client, executor = harness
    response = client.post(
        "/internal/v1/model/execute-profile",
        json=command(**overrides),
        headers={"X-API-KEY": "synthetic-novel-key"},
    )

    # 검증
    assert response.status_code == 422
    executor.execute.assert_not_awaited()
    assert "Synthetic source" not in response.text


def test_profile_parent_trace_and_null_ttft_preserved(harness):
    # 실행
    _, client, executor = harness
    trace = "133094fa-0bea-4723-8c09-f69ad1cb6a86"
    response = client.post(
        "/internal/v1/model/execute-profile",
        json=command(),
        headers={"X-API-KEY": "synthetic-novel-key", "X-Novel-Trace-Id": trace},
    )

    # 검증
    assert response.status_code == 200
    assert executor.execute.call_args.kwargs["parent_trace_id"] == trace
    assert response.json()["metadata"]["providerFirstTokenMs"] is None


def test_profiles_are_available_without_a_service_enable_flag(harness):
    # 준비
    _, client, executor = harness

    # 실행 / 검증
    response = client.post(
        "/internal/v1/model/execute-profile",
        json=command(),
        headers={"X-API-KEY": "synthetic-novel-key"},
    )
    assert response.status_code == 200
    executor.execute.assert_awaited_once()
    assert "NOVEL_MODEL_PROFILES_ENABLED" not in type(settings).model_fields


def test_stream_profile_emits_fields_before_verified_result_and_requires_novel_key(harness):
    _, client, executor = harness
    body = command(
        responseSchema={"type": "object", "properties": {"s0": {"type": "string"}},
                        "required": ["s0"], "additionalProperties": False},
        schemaName="novel_test", strict=True,
    )
    async def stream(**_kwargs):
        yield {"type": "field", "key": "s0", "value": "완성 문장", "providerElapsedMs": 5}
        yield {"type": "result", "result": {**executor.execute.return_value,
                "output": {"s0": "완성 문장"}}}
    executor.stream = stream
    denied = client.post("/internal/v1/model/execute-profile-stream", json=body,
                         headers={"X-API-KEY": "synthetic-legacy-key"})
    assert denied.status_code == 401
    accepted = client.post("/internal/v1/model/execute-profile-stream", json=body,
                           headers={"X-API-KEY": "synthetic-novel-key"})
    assert accepted.status_code == 200
    events = [json.loads(line.removeprefix("data: ")) for line in accepted.text.splitlines()
              if line.startswith("data: ")]
    assert [event["type"] for event in events] == ["field", "result"]
    assert events[0]["value"] == events[1]["result"]["output"]["s0"]
    executor.execute.assert_not_awaited()


def test_stream_profile_late_error_keeps_partial_result_distinct(harness):
    _, client, executor = harness
    async def stream(**_kwargs):
        yield {"type": "field", "key": "s0", "value": "보임"}
        raise RuntimeError("private provider body")
    executor.stream = stream
    response = client.post("/internal/v1/model/execute-profile-stream", json=command(),
                           headers={"X-API-KEY": "synthetic-novel-key"})
    events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines()
              if line.startswith("data: ")]
    assert [event["type"] for event in events] == ["field", "error"]
    assert events[-1]["code"] == "PROGRESSIVE_EXECUTION_FAILED"
    assert "private provider body" not in response.text


@pytest.mark.asyncio
async def test_profile_queue_is_bounded_and_cancelled_waiter_never_calls_provider(
    harness, monkeypatch
):
    # 준비
    app, _, executor = harness
    monkeypatch.setattr(settings, "NOVEL_MODEL_PROFILE_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(settings, "NOVEL_MODEL_PROFILE_MAX_QUEUE", 1)
    entered, release = asyncio.Event(), asyncio.Event()
    result = executor.execute.return_value

    async def blocked(**kwargs):
        entered.set()
        await release.wait()
        return result

    executor.execute.side_effect = blocked
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:

        async def call(ms):
            return await client.post(
                "/internal/v1/model/execute-profile",
                json=command(remainingMilliseconds=ms),
                headers={"X-API-KEY": "synthetic-novel-key"},
            )

        # 실행
        first = asyncio.create_task(call(5000))
        await entered.wait()
        second = asyncio.create_task(call(50))
        await asyncio.sleep(0.01)
        third = await call(5000)
        timed_out = await second
        release.set()
        completed = await first

    # 검증
    assert third.status_code == 429
    assert timed_out.status_code == 504
    assert completed.status_code == 200
    assert executor.execute.await_count == 1


@pytest.mark.asyncio
async def test_actual_asgi_disconnect_in_queue_never_enters_provider(harness, monkeypatch):
    # 준비: 첫 요청이 슬롯을 점유하고 두 번째 요청에 실제 ASGI disconnect를 전달한다.
    app, _, executor = harness
    monkeypatch.setattr(settings, "NOVEL_MODEL_PROFILE_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(settings, "NOVEL_MODEL_PROFILE_MAX_QUEUE", 2)
    entered, release = asyncio.Event(), asyncio.Event()
    result = executor.execute.return_value

    async def blocked(**kwargs):
        entered.set()
        await release.wait()
        return result

    executor.execute.side_effect = blocked
    receive_queue = asyncio.Queue()
    await receive_queue.put(
        {"type": "http.request", "body": json.dumps(command()).encode(), "more_body": False}
    )
    scope = profile_scope()
    sent = []

    async def send(message):
        sent.append(message)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = asyncio.create_task(
            client.post(
                "/internal/v1/model/execute-profile",
                json=command(),
                headers={"X-API-KEY": "synthetic-novel-key"},
            )
        )
        await entered.wait()
        second = asyncio.create_task(app(scope, receive_queue.get, send))
        await asyncio.sleep(0.03)

        # 실행: 슬롯은 여전히 점유된 상태에서 연결만 종료한다.
        await receive_queue.put({"type": "http.disconnect"})
        await asyncio.sleep(0.03)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=2)

    # 검증: 이미 끊긴 대기 요청은 이후 슬롯이 나도 유료 실행을 시작하지 않는다.
    assert executor.execute.await_count == 1


@pytest.mark.asyncio
async def test_profile_deadline_includes_body_read_time(harness):
    # 준비: 본문 수신만으로 호출자가 제시한 deadline을 소진한다.
    app, _, executor = harness
    receives = asyncio.Queue()
    sent = []

    async def send(message):
        sent.append(message)

    request = asyncio.create_task(app(profile_scope(), receives.get, send))
    await asyncio.sleep(0.06)

    # 실행
    await receives.put(
        {
            "type": "http.request",
            "body": json.dumps(command(remainingMilliseconds=20)).encode(),
            "more_body": False,
        }
    )
    await asyncio.wait_for(request, timeout=2)

    # 검증
    assert next(item["status"] for item in sent if item["type"] == "http.response.start") == 504
    executor.execute.assert_not_awaited()
