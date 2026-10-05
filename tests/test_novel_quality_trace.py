import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

from scripts import novel_quality_server as server


def application(handler):
    app = FastAPI()
    app.add_middleware(server.LegacyChildTraceMiddleware)

    # BaseHTTPMiddleware의 실제 자식 task 경계를 포함해서 ContextVar 공유를 확인한다.
    @app.middleware("http")
    async def ordinary_middleware(request, call_next):
        return await call_next(request)

    app.add_api_route("/internal/v1/model/execute", handler, methods=["POST"])
    return app


@pytest.mark.asyncio
async def test_interleaved_streaming_requests_preserve_body_and_isolate_child_trace():
    # 준비: 서로 다른 청크의 본문이 병행해서 도착한다.
    async def handler(request: Request):
        body = await request.body()
        await asyncio.sleep(0)
        return {
            "sha": hashlib.sha256(body).hexdigest(),
            "child": (server.LEGACY_CHILD_TRACE.get() or {}).get("childTraceId"),
        }

    async def chunks(body):
        for index in range(0, len(body), 7):
            yield body[index : index + 7]
            await asyncio.sleep(0)

    bodies = [
        json.dumps({"traceId": f"job-{i}:chunk:attempt", "instructions": "safe synthetic"}).encode()
        for i in range(2)
    ]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application(handler)), base_url="http://local"
    ) as client:
        # 실행 / 검증: 본문을 재작성하지 않고 각 request의 자식 ID만 관찰한다.
        results = await asyncio.gather(
            *(client.post("/internal/v1/model/execute", content=chunks(body)) for body in bodies)
        )
    assert [row.json()["child"] for row in results] == [f"job-{i}:chunk:attempt" for i in range(2)]
    assert [row.json()["sha"] for row in results] == [
        hashlib.sha256(body).hexdigest() for body in bodies
    ]
    assert server.LEGACY_CHILD_TRACE.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b'{"traceId":"bad key"}',
        b'{"traceId":"accepted","data":"' + b"x" * 500000 + b'"}',
    ],
    ids=["invalid-json", "invalid-trace", "too-large"],
)
async def test_invalid_or_oversized_trace_observation_is_not_claimed(body):
    async def handler(request: Request):
        actual = await request.body()
        return {
            "bytes": len(actual),
            "child": (server.LEGACY_CHILD_TRACE.get() or {}).get("childTraceId"),
        }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application(handler)), base_url="http://local"
    ) as client:
        result = await client.post("/internal/v1/model/execute", content=body)
    assert result.json() == {"bytes": len(body), "child": None}


@pytest.mark.asyncio
async def test_budgeted_legacy_records_observed_child_without_changing_provider_kwargs(
    monkeypatch, tmp_path
):
    # 준비: 외부 provider와 영속 장부 대신 작은 stub을 사용한다.
    recorded = []
    kwargs_seen = []

    class Ledger:
        path = tmp_path / "ledger.json"

        def reserve(self, task, model, amount, *, metadata):
            recorded.append(metadata)
            return 1

        def finish(self, *args, **kwargs):
            pass

        def annotate(self, *args, **kwargs):
            pass

    class Provider:
        async def execute_explicit(self, **kwargs):
            kwargs_seen.append(kwargs)
            return SimpleNamespace(input_tokens=5, output_tokens=5)

    @asynccontextmanager
    async def slot(_path):
        yield

    monkeypatch.setattr(server, "provider_slot", slot)
    upstream = server.BudgetedLegacyExecutor(Provider(), Ledger(), phase="final")

    async def handler(request: Request):
        from app.ai.model_policy import AiModelTier

        await request.body()
        await upstream.execute_explicit(tier=AiModelTier.LUNA, max_output_tokens=32)
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application(handler)), base_url="http://local"
    ) as client:
        # 실행 / 검증: child ID는 QA metadata만 보강한다.
        response = await client.post("/internal/v1/model/execute", json={"traceId": "job:0:1"})
    assert response.status_code == 200
    assert recorded[0]["childTraceId"] == "job:0:1"
    assert recorded[0]["phase"] == "final"
    assert "traceId" not in kwargs_seen[0]
