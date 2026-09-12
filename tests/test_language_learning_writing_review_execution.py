"""Reference execution contracts: cancellation, retries, API errors and isolation."""
from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi import HTTPException

from app.features.language_learning.writing.generation import GENERATION_TASK
from app.features.language_learning.writing.service import LanguageLearningWritingService
from app.features.language_learning.writing.verification import MINI_QUALITY_TASK, MINI_DIFFICULTY_TASK, MINI_NOTE_TASK
from app.features.language_learning.writing.note_localization import NOTE_LOCALIZATION_TASK
from tests.test_language_learning_writing_verified_generation import draft, request, service
from tests.test_language_learning_writing_note_language import NoteProvider, FOREIGN_NOTE
from tests.test_language_learning_writing_review_runtime import events, StatusError
from tests.writing_generation_fakes import WritingPipelineProvider, unsure_review, read_review_data, review_text


@pytest.mark.parametrize("stage", [GENERATION_TASK, MINI_QUALITY_TASK, MINI_DIFFICULTY_TASK, NOTE_LOCALIZATION_TASK, MINI_NOTE_TASK])
@pytest.mark.parametrize("deadline", [False, True])
def test_cancel_or_deadline_leaves_no_inflight_task_or_fallback(stage, deadline, caplog):
    async def scenario():
        item = draft()
        if stage in {NOTE_LOCALIZATION_TASK, MINI_NOTE_TASK}:
            item["focusReason"] = FOREIGN_NOTE
        entered, cleaned = asyncio.Event(), asyncio.Event()
        started = []
        class WaitingProvider(NoteProvider):
            async def call(self, type_name, data, schema=None):
                started.append(type_name)
                if type_name == stage:
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cleaned.set()
                return await super().call(type_name, data, schema)
        provider = WaitingProvider([{"items": [item]}], {item["originText"]: 3},
                                   {MINI_QUALITY_TASK: unsure_review()} if stage == MINI_DIFFICULTY_TASK else {})
        svc = LanguageLearningWritingService(provider, generation_timeout_seconds=2,
                                             verification_timeout_seconds=2,
                                             generation_total_timeout_seconds=0.08 if deadline else 10)
        task = asyncio.create_task(svc.generate_daily(request()))
        await asyncio.wait_for(entered.wait(), 0.5)
        if deadline:
            with pytest.raises(HTTPException) as exc:
                await task
            assert exc.value.status_code == 504
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert cleaned.is_set() and started[-1] == stage
        current = asyncio.current_task()
        assert not [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    summary = events(caplog, "writing.generation.finished")[-1]
    assert summary["returned_items"] == 0
    assert summary["outcome"] == ("FAILED" if deadline else "CANCELLED")


@pytest.mark.parametrize("failure", [TimeoutError(), StatusError(503), ValueError("private bad json")])
def test_primary_transport_or_protocol_retry_preserves_candidate_and_does_not_regenerate(failure, caplog):
    item = draft()
    provider = WritingPipelineProvider([{"items": [item]}], {item["originText"]: 3},
                                      {MINI_QUALITY_TASK: [failure, {}]})
    with caplog.at_level(logging.INFO):
        result = asyncio.run(service(provider, generation_max_retries=3).generate_daily(request()))
    assert len(result.items) == 1
    assert provider.counts == {GENERATION_TASK: 1, MINI_QUALITY_TASK: 2}
    review_calls = [c for c in provider.calls if c["type_name"] == MINI_QUALITY_TASK]
    assert review_calls[0]["data"] == review_calls[1]["data"]
    assert "private" not in caplog.text


@pytest.mark.parametrize("failure", [TimeoutError(), StatusError(503), ValueError("private output")])
def test_reviewer_outage_does_not_burn_new_generator_rounds(failure):
    item = draft()
    provider = WritingPipelineProvider([{"items": [item]}] * 4, {item["originText"]: 3}, {MINI_QUALITY_TASK: failure})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(service(provider, generation_max_retries=3).generate_daily(request()))
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "WRITING_VERIFICATION_UNAVAILABLE"
    assert provider.counts == {GENERATION_TASK: 1, MINI_QUALITY_TASK: 2}


@pytest.mark.parametrize("stage", [GENERATION_TASK, MINI_QUALITY_TASK, MINI_DIFFICULTY_TASK, NOTE_LOCALIZATION_TASK, MINI_NOTE_TASK])
def test_configuration_error_aborts_current_request_without_regeneration(stage):
    item = draft()
    if stage in {NOTE_LOCALIZATION_TASK, MINI_NOTE_TASK}:
        item["focusReason"] = FOREIGN_NOTE
    overrides = {stage: StatusError(401)}
    if stage == MINI_DIFFICULTY_TASK:
        overrides[MINI_QUALITY_TASK] = unsure_review()
    provider = NoteProvider([StatusError(401)] if stage == GENERATION_TASK else [{"items": [item]}],
                            {item["originText"]: 3}, overrides,
                            proposal_override=StatusError(401) if stage == NOTE_LOCALIZATION_TASK else None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(service(provider).generate_daily(request()))
    assert exc.value.status_code == 502
    assert provider.counts[GENERATION_TASK] == provider.counts[stage] == 1


def test_actual_per_stage_timeout_is_bounded_and_never_a_pass():
    async def scenario():
        item = draft()
        started = 0
        cleaned = 0
        class TimeoutProvider(WritingPipelineProvider):
            async def call(self, type_name, data, schema=None):
                nonlocal started, cleaned
                if type_name == MINI_QUALITY_TASK:
                    started += 1
                    try:
                        await asyncio.sleep(1)
                    finally:
                        cleaned += 1
                return await super().call(type_name, data, schema)
        provider = TimeoutProvider([{"items": [item]}], {item["originText"]: 3})
        svc = LanguageLearningWritingService(provider, verification_timeout_seconds=0.01, generation_total_timeout_seconds=2)
        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(svc.generate_daily(request()), 1)
        assert exc.value.status_code == 503 and exc.value.detail["failureCode"] == "VERIFIER_TIMEOUT"
        assert started == cleaned == 2
        assert provider.counts[GENERATION_TASK] == 1
    asyncio.run(scenario())


def test_concurrent_requests_share_no_targets_approvals_or_progress():
    async def scenario():
        a, b = draft("문을 닫아 주세요.", tag="A"), draft()
        class SharedProvider(WritingPipelineProvider):
            async def call(self, type_name, data, schema=None):
                if type_name != GENERATION_TASK:
                    await asyncio.sleep(0.002)
                return await super().call(type_name, data, schema)
        provider = SharedProvider([{"items": [a]}, {"items": [b]}], {a["originText"]: 2, b["originText"]: 3})
        svc = service(provider)
        first, second = await asyncio.gather(svc.generate_daily(request(request_id="A", band=3)),
                                            svc.generate_daily(request(request_id="B", band=4)))
        assert [(r.request_id, r.items[0].language_complexity_band) for r in (first, second)] == [("A", 2), ("B", 3)]
        assert first.items[0].origin_text == a["originText"] and second.items[0].origin_text == b["originText"]
        calls = [read_review_data(c["data"]) for c in provider.calls if c["type_name"] == MINI_QUALITY_TASK]
        assert len({c["candidateId"] for c in calls}) == 2
        assert {review_text(c) for c in calls} == {a["originText"], b["originText"]}
    asyncio.run(scenario())


def test_success_summary_records_surface_policy_and_one_review_without_raw_text(caplog):
    item = draft()
    provider = WritingPipelineProvider([{"items": [item]}], {item["originText"]: 3})
    with caplog.at_level(logging.INFO):
        asyncio.run(service(provider).generate_daily(request()))
    summary = events(caplog, "writing.generation.finished")[0]
    assert summary["policy"] == "writing-difficulty-reference-v4.1-bounded-recovery"
    assert summary["difficulty_spec_version"] == "writing-difficulty-spec-v2"
    assert summary["returned_items"] == 1 and summary["outcome"] == "SUCCEEDED"
    assert summary["review_calls"] == {MINI_QUALITY_TASK: 1}
    assert summary["review_retries"] == {} and summary["peak_inflight_reviews"] == 1
    surface = events(caplog, "writing.spec.checked")[0]
    assert surface["measurements"]["origin_characters"] > 0 and surface["outcome"] == "PASS"
    assert item["originText"] not in caplog.text and item["focusReason"] not in caplog.text


@pytest.mark.parametrize("kind,status,code", [
    ("protocol", 503, "WRITING_VERIFICATION_UNAVAILABLE"),
    ("configuration", 502, "WRITING_PROVIDER_CONFIGURATION_ERROR"),
    ("deadline", 504, "WRITING_GENERATION_DEADLINE_EXCEEDED"),
])
def test_actual_route_serializes_safe_failure_without_candidate(monkeypatch, kind, status, code):
    import importlib.util
    import sys
    import types
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    item = draft()
    class Provider(WritingPipelineProvider):
        async def call(self, type_name, data, schema=None):
            if type_name == MINI_QUALITY_TASK and kind == "deadline":
                await asyncio.sleep(1)
            return await super().call(type_name, data, schema)
    override = StatusError(401) if kind == "configuration" else {"confidence": "PRIVATE_BAD_VALUE"}
    provider = Provider([{"items": [item]}], {item["originText"]: 3}, {MINI_QUALITY_TASK: override})
    svc = LanguageLearningWritingService(provider, generation_total_timeout_seconds=0.04 if kind == "deadline" else 2)
    dependencies = types.ModuleType("app.api.dependencies")
    dependencies.get_language_learning_writing_service = lambda: svc
    monkeypatch.setitem(sys.modules, "app.api.dependencies", dependencies)
    filename = Path(__file__).parents[1] / "app/api/v1/language_learning.py"
    spec = importlib.util.spec_from_file_location("writing_reference_error_route", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/v1")
    with TestClient(app) as client:
        response = client.post("/api/v1/language-learning/writing/daily/generate",
                               json=request().model_dump(mode="json", by_alias=True))
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "items" not in response.json()["detail"]
    assert item["originText"] not in response.text and "PRIVATE_BAD_VALUE" not in response.text
