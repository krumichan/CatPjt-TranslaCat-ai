"""V4.1 regression: model answers are scripted, never live quality/calibration claims.

These tests check call bounds, independent input contracts and failure handling.
Assigned bands deliberately do not claim that the example sentences have that level.
"""
from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import replace

import pytest
from fastapi import HTTPException

from app.features.language_learning.writing.assessment_contract import TaskReview, assess_acceptance
from app.features.language_learning.writing.difficulty_planning import (
    DifficultyRecoveryState, production_blueprint,
)
from app.features.language_learning.writing.difficulty_spec import WRITING_BAND_RUBRIC
from app.features.language_learning.writing.generation import GENERATION_TASK
from app.features.language_learning.writing.generation_contract import plan_slots, parse_draft
from app.features.language_learning.writing.verification import (
    MINI_DIFFICULTY_TASK, MINI_QUALITY_TASK, WritingCandidateVerifier,
)
from tests.test_language_learning_writing_verified_generation import draft, request, service
from tests.test_language_learning_writing_review_runtime import events, StatusError
from tests.writing_generation_fakes import (
    WritingPipelineProvider, read_generation_data, read_review_data, task_review, reject_issues, unsure_review,
)


def candidates(n=8):
    return [draft(text, tag=f"FIXTURE_{index}") for index, text in enumerate([
        "도서관에서 빌린 책을 이번 주 안에 반납해 주세요.",
        "비가 그치면 공원을 산책하고 벤치에서 쉬겠습니다.",
        "식당의 예약 시간을 바꾸려면 담당자에게 먼저 연락하세요.",
        "열차가 지연되어 도착 시간이 달라질 수 있습니다.",
        "직접 만든 빵을 내일 아침 이웃과 나누려고 합니다.",
        "회의 자료는 아직 확인 중이니 조금만 기다려 주세요.",
        "농장에서 돌아오는 길에 작은 가게를 발견했습니다.",
        "운동을 마친 후에는 따뜻한 물을 마시고 쉬세요.",
    ][:n])]


def batches(items):
    return [{"items": items[index:index + 2]} for index in range(0, len(items), 2)]


def execute(provider, req=None, *, retries=3, **options):
    return asyncio.run(service(provider, generation_max_retries=retries, **options).generate_daily(
        req or request(band=4, difficulty="CHALLENGE")))


@pytest.mark.parametrize("target", range(1, 6))
@pytest.mark.parametrize("observed", range(1, 6))
def test_exact_adjacent_and_far_acceptance_matrix(target, observed):
    slot = plan_slots(request(band=target, difficulty="NORMAL"))[0]
    data = {"candidateId": "test", "contentHash": "a" * 64, "writingType": "TRANSLATION"}
    review = TaskReview.model_validate(task_review(data, observed))
    first = assess_acceptance(review, slot)
    expected = "ACCEPT" if target == observed else "ADJUDICATE" if abs(target - observed) == 1 else "REJECT"
    assert first.action == expected
    # No second adjudication and no automatic promotion/demotion after budget use.
    final = assess_acceptance(review, slot, adjudicated=True)
    spent = assess_acceptance(review, slot, allow_adjacent_recheck=False)
    assert final.action == spent.action == ("ACCEPT" if target == observed else "REJECT")


@pytest.mark.parametrize("issue", ["ORIGIN_LANGUAGE", "ANSWER_LEAK", "TASK_TYPE", "UNNATURAL_LANGUAGE",
                                    "AMBIGUOUS_TASK", "NOTE_ORIGIN_LANGUAGE", "UNSUPPORTED_FOCUS_REASON"])
def test_adjacent_gap_is_not_permission_to_revote_quality_failure(issue):
    item = draft()
    provider = WritingPipelineProvider([{"items": [item]}], {item["originText"]: 4},
                                      {MINI_QUALITY_TASK: reject_issues(issue)})
    with pytest.raises(HTTPException):
        execute(provider, retries=0)
    assert provider.counts[MINI_QUALITY_TASK] == 1
    assert provider.counts[MINI_DIFFICULTY_TASK] == 0


@pytest.mark.parametrize("last", [4, 5])
def test_one_adjacent_recheck_must_independently_pass_exact_target(last, caplog):
    item = draft()
    provider = WritingPipelineProvider([{"items": [item]}], {item["originText"]: 4},
                                      {MINI_DIFFICULTY_TASK: {"estimatedBand": last, "confidence": 0.0}})
    caplog.set_level(logging.INFO)
    if last == 5:
        result = execute(provider)
        assert result.items[0].language_complexity_band == 5
    else:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(service(provider, generation_max_retries=0).generate_daily(
                request(band=4, difficulty="CHALLENGE")))
        assert exc.value.detail["rejectionCounts"] == {"VERIFIED_BAND_MISMATCH": 1}
    assert provider.counts == {GENERATION_TASK: 1, MINI_QUALITY_TASK: 1, MINI_DIFFICULTY_TASK: 1}
    logs = events(caplog, "writing.adjudication.requested")
    assert len(logs) == 1 and logs[0]["reason"] == "ADJACENT_BAND_RECHECK"
    assert logs[0]["adjacent_rechecks_used"] == logs[0]["adjacent_recheck_limit"] == 1


@pytest.mark.parametrize("override", [unsure_review(), unsure_review(borderline=(4, 5)),
                                     reject_issues("ANSWER_LEAK"), {"estimatedBand": 3}])
def test_unresolved_or_failed_final_review_never_uses_first_pass_as_fallback(override):
    item = draft()
    provider = WritingPipelineProvider([{"items": [item]}], {item["originText"]: 4},
                                      {MINI_DIFFICULTY_TASK: override})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(service(provider, generation_max_retries=0).generate_daily(request(band=4, difficulty="CHALLENGE")))
    assert exc.value.status_code == 422 and "items" not in exc.value.detail
    assert provider.counts[MINI_DIFFICULTY_TASK] == 1


def test_seven_wrong_band_pattern_stops_after_one_recovery_round_not_four(caplog):
    items = candidates()
    provider = WritingPipelineProvider(batches(items), {item["originText"]: 4 for item in items})
    caplog.set_level(logging.INFO)
    with pytest.raises(HTTPException) as exc:
        execute(provider)
    assert exc.value.status_code == 422
    assert exc.value.detail["targetBand"] == 5
    assert exc.value.detail["observedBandCounts"] == {"4": 4}
    assert exc.value.detail["difficultyRecoveryUsed"] is True
    assert exc.value.detail["attempts"] == 2
    # Two batches, four distinct semantic checks, only one extra boundary review.
    assert provider.counts == {GENERATION_TASK: 2, MINI_QUALITY_TASK: 4, MINI_DIFFICULTY_TASK: 1}
    assert len(provider.batches) == 2  # Do not spend the remaining two rounds.
    summary = events(caplog, "writing.generation.finished")[-1]
    assert summary["difficulty_recovery_rounds"] == summary["adjacent_band_rechecks"] == 1
    assert summary["difficulty_mismatches"] == {"5->4": 4}
    assert not summary["review_failures"] and not summary["review_retries"]
    assert all(item["originText"] not in caplog.text for item in items)


@pytest.mark.parametrize("target,observed,direction", [
    (5, 4, "INCREASE_PRODUCTION_DEMAND"), (3, 4, "DECREASE_PRODUCTION_DEMAND"),
])
def test_targeted_round_changes_blueprint_not_goal_and_preserves_diversity(target, observed, direction):
    items = candidates(4)
    req = request(band=target, difficulty="NORMAL", learningProfile={"grammarWeaknesses": ["PRIVATE_PROFILE"]},
                  recentMistakes=["PRIVATE_MISTAKE"], recentlyLearnedExpressions=["PRIVATE_EXPRESSION"],
                  diversityContext={"currentSession": [{"sourceType": "WRITING", "content": "独立した別の話題です。"}]})
    provider = WritingPipelineProvider(batches(items), {item["originText"]: observed for item in items[:2]} |
                                      {item["originText"]: target for item in items[2:]})
    result = execute(provider, req)
    assert result.items[0].origin_text == items[2]["originText"]
    assert result.items[0].language_complexity_band == target
    generations = [read_generation_data(c["data"]) for c in provider.calls if c["type_name"] == GENERATION_TASK]
    first, retry = generations
    assert "difficultyRecovery" not in first
    assert retry["difficultyRecovery"]["direction"] == direction
    assert retry["difficultyRecovery"]["observedBandCounts"] == {str(observed): 2}
    assert first["productionBlueprint"]["blueprintId"] != retry["productionBlueprint"]["blueprintId"]
    assert first["generationPlan"]["targetBand"] == retry["generationPlan"]["targetBand"] == target
    for name in ("learningProfile", "recentMistakes", "recentlyLearnedExpressions", "recentEvaluationSummary"):
        assert name not in retry
    assert first["diversityContext"] == retry["diversityContext"]
    for call in provider.calls:
        if call["type_name"] in {MINI_QUALITY_TASK, MINI_DIFFICULTY_TASK}:
            for field in ("targetBand", "productionBlueprint", "difficultyRecovery", "PRIVATE_PROFILE", "PRIVATE_MISTAKE"):
                assert field not in call["data"]
            assert str(call["schema"]).find("productionBlueprint") == -1


def test_fresh_boundary_review_never_sees_previous_verdict_or_changes_task():
    item = draft()
    provider = WritingPipelineProvider([{"items": [item]}], {item["originText"]: 4},
                                      {MINI_DIFFICULTY_TASK: {"estimatedBand": 5}})
    execute(provider)
    reviewed = [c for c in provider.calls if c["type_name"] != GENERATION_TASK]
    a, b = [read_review_data(c["data"]) for c in reviewed]
    assert a["candidateId"] != b["candidateId"]
    assert a["contentHash"] == b["contentHash"] and a["segments"] == b["segments"]
    assert "previousVerdict" not in reviewed[1]["data"]
    # Full all-band rubric is identical for both; it is not restricted to 4 vs 5.
    assert reviewed[0]["data"].split("<writing-review-data>")[0] == reviewed[1]["data"].split("<writing-review-data>")[0]
    assert all(str(band) in reviewed[1]["data"] for band in range(1, 6))


@pytest.mark.parametrize("failure,expected", [(TimeoutError(), 503), (StatusError(401), 502),
                                            ({"candidateId": "replayed-primary"}, 503)])
def test_adjacent_recheck_failure_is_not_retried_or_reclassified_as_bad_candidate(failure, expected):
    item = draft()
    provider = WritingPipelineProvider([{"items": [item]}] * 4, {item["originText"]: 4},
                                      {MINI_DIFFICULTY_TASK: failure})
    with pytest.raises(HTTPException) as exc:
        execute(provider)
    assert exc.value.status_code == expected
    assert provider.counts == {GENERATION_TASK: 1, MINI_QUALITY_TASK: 1, MINI_DIFFICULTY_TASK: 1}


def test_same_text_with_different_note_cannot_get_another_adjacent_vote():
    a = draft()
    b = copy.deepcopy(a)
    b["focusReason"] = "별도의 설명이 붙었습니다."
    provider = WritingPipelineProvider([{"items": [a]}, {"items": [b]}], {a["originText"]: 4})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(service(provider, generation_max_retries=1).generate_daily(request(band=4, difficulty="CHALLENGE")))
    assert exc.value.detail["rejectionCounts"]["REPEATED_REJECTED_TASK"] == 1
    assert provider.counts[MINI_QUALITY_TASK] == provider.counts[MINI_DIFFICULTY_TASK] == 1


@pytest.mark.parametrize("base,difficulty,target", [(5, "CHALLENGE", 5), (5, "NORMAL", 5),
                                                  (1, "REVIEW", 1), (4, "CHALLENGE", 5)])
@pytest.mark.parametrize("mode", ["TRANSLATION", "GUIDED", "FREE"])
def test_cap_and_normal_call_count_are_unchanged(base, difficulty, target, mode):
    req = request(band=base, difficulty=difficulty, mode=mode)
    item = draft(mode=mode)
    provider = WritingPipelineProvider([{"items": [item]}], {item["originText"]: target})
    result = execute(provider, req)
    assert result.items[0].language_complexity_band == target
    assert provider.counts == {GENERATION_TASK: 1, MINI_QUALITY_TASK: 1}
    data = read_generation_data(provider.calls[0]["data"])
    assert f".B{target}." in data["productionBlueprint"]["blueprintId"]
    assert data["difficultySpec"]["targetBand"] == target
    assert "B6" not in str(data["productionBlueprint"])


def test_same_base5_normal_and_challenge_do_not_invent_different_band6_blueprints():
    normal = plan_slots(request(band=5, difficulty="NORMAL"))[0]
    challenge = plan_slots(request(band=5, difficulty="CHALLENGE"))[0]
    assert production_blueprint(normal, 1) == production_blueprint(challenge, 1)


def test_recovery_state_ignores_confidence_unknown_quality_and_opposing_directions():
    state = DifficultyRecoveryState(3)
    assert not state.record(estimated_band=4, difficulty_status="BORDERLINE", reason="VERIFIED_BAND_MISMATCH", generation_attempt=1)
    assert not state.record(estimated_band=4, difficulty_status="ASSESSED", reason="QUALITY_ORIGIN_LANGUAGE", generation_attempt=1)
    assert not state.record(estimated_band=True, difficulty_status="ASSESSED", reason="VERIFIED_BAND_MISMATCH", generation_attempt=1)
    for band in (4, 4, 2):
        state.record(estimated_band=band, difficulty_status="ASSESSED", reason="VERIFIED_BAND_MISMATCH", generation_attempt=1)
    assert state.select_after_round(1, 4) is False and state.payload() is None


def test_recovery_state_selects_once_and_never_expands_caller_budget():
    state = DifficultyRecoveryState(5)
    for _ in range(2):
        state.record(estimated_band=4, difficulty_status="ASSESSED", reason="VERIFIED_BAND_MISMATCH", generation_attempt=1)
    assert not state.select_after_round(1, 1)
    assert state.select_after_round(1, 4)
    assert state.start_attempt == 2
    assert not state.select_after_round(2, 4)
    value = state.payload()
    value["observedBandCounts"]["4"] = 999
    assert state.payload()["observedBandCounts"] == {"4": 2}


@pytest.mark.parametrize("bad", [0, 6, True, "5", 5.0])
def test_no_invalid_or_virtual_band_enters_blueprint_or_recovery(bad):
    with pytest.raises(ValueError):
        DifficultyRecoveryState(bad)
    slot = replace(plan_slots(request())[0], target_band=bad)
    with pytest.raises(ValueError):
        production_blueprint(slot, 1)


def test_blueprint_payloads_cannot_mutate_another_request_or_the_rubric():
    slot = plan_slots(request(band=4, difficulty="CHALLENGE"))[0]
    a = production_blueprint(slot, 1)
    a["semanticRequirements"][0]["demand"] = "PRIVATE_OVERRIDE"
    assert "PRIVATE_OVERRIDE" not in str(production_blueprint(slot, 1))
    assert "PRIVATE_OVERRIDE" not in str(WRITING_BAND_RUBRIC)


def test_adjacent_allowance_is_per_slot_not_per_candidate_or_service_instance():
    async def scenario():
        items = candidates(2)
        provider = WritingPipelineProvider([], {item["originText"]: 4 for item in items})
        verifier = WritingCandidateVerifier(provider, 1)
        req = request(band=4, difficulty="CHALLENGE")
        slot = plan_slots(req)[0]
        # Two siblings in one slot consume only one allowance. A different slot
        # gets its own one-shot allowance; all of these fixtures remain rejected.
        for index, item in enumerate(items):
            result = await verifier.verify(req, parse_draft(item), slot, f"c{index}")
            assert not result.accepted
        await verifier.verify(req, parse_draft(items[0]), replace(slot, order=2), "c2")
        assert verifier.adjacent_rechecks == {1: 1, 2: 1}
        assert provider.counts[MINI_DIFFICULTY_TASK] == 2
    asyncio.run(scenario())


def test_two_requests_do_not_share_recovery_allowance():
    async def scenario():
        item = draft()
        provider = WritingPipelineProvider([{"items": [item]}] * 2, {item["originText"]: 4},
                                          {MINI_DIFFICULTY_TASK: {"estimatedBand": 5}})
        svc = service(provider)
        result = await asyncio.gather(svc.generate_daily(request(request_id="A", band=4, difficulty="CHALLENGE")),
                                      svc.generate_daily(request(request_id="B", band=4, difficulty="CHALLENGE")))
        assert len(result) == 2 and provider.counts[MINI_DIFFICULTY_TASK] == 2
        calls = [read_review_data(c["data"])["candidateId"] for c in provider.calls if c["type_name"] == MINI_DIFFICULTY_TASK]
        assert len(set(calls)) == 2
    asyncio.run(scenario())


def test_cancel_during_adjacent_recheck_leaves_no_task_or_published_item():
    async def scenario():
        item = draft()
        entered, cleaned = asyncio.Event(), asyncio.Event()
        class Provider(WritingPipelineProvider):
            async def call(self, type_name, data, schema=None):
                if type_name == MINI_DIFFICULTY_TASK:
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cleaned.set()
                return await super().call(type_name, data, schema)
        provider = Provider([{"items": [item]}], {item["originText"]: 4})
        task = asyncio.create_task(service(provider).generate_daily(request(band=4, difficulty="CHALLENGE")))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        assert not [other for other in asyncio.all_tasks() if other is not asyncio.current_task() and not other.done()]
    asyncio.run(scenario())


def test_source_recovery_cannot_extend_the_final_difficulty_recovery_round():
    from tests.test_language_learning_writing_source_recovery import SourceProvider, bad_batch
    from app.features.language_learning.writing.source_recovery import SOURCE_LOCALIZATION_TASK
    items = candidates(2)
    provider = SourceProvider([{"items": items}, bad_batch(), bad_batch()],
                              bands={item["originText"]: 4 for item in items}, repairs={})
    with pytest.raises(HTTPException) as exc:
        execute(provider)
    assert exc.value.status_code == 422 and exc.value.detail["attempts"] == 2
    assert provider.counts[GENERATION_TASK] == 2
    assert provider.counts[SOURCE_LOCALIZATION_TASK] == 1
    assert len(provider.batches) == 1


def test_repaired_sources_can_select_directional_recovery_without_repeating_source_repair():
    from tests.test_language_learning_writing_source_recovery import (
        SourceProvider, bad_batch, ORIGINAL, LOCALIZED, OTHER_ORIGINAL, OTHER_LOCALIZED,
    )
    from app.features.language_learning.writing.source_recovery import SOURCE_LOCALIZATION_TASK
    good = candidates(1)[0]
    provider = SourceProvider([bad_batch(), {"items": [good]}],
                              bands={LOCALIZED: 4, OTHER_LOCALIZED: 4, good["originText"]: 5},
                              repairs={ORIGINAL: LOCALIZED, OTHER_ORIGINAL: OTHER_LOCALIZED})
    result = execute(provider)
    assert result.items[0].origin_text == good["originText"]
    assert provider.counts == {GENERATION_TASK: 2, SOURCE_LOCALIZATION_TASK: 1,
                              MINI_QUALITY_TASK: 3, MINI_DIFFICULTY_TASK: 1}
    plans = [read_generation_data(c["data"]) for c in provider.calls if c["type_name"] == GENERATION_TASK]
    assert plans[1]["sourceRecoveryMode"] == "REDUCED_CONTEXT_REGENERATION_ONCE"
    assert plans[1]["difficultyRecovery"]["direction"] == "INCREASE_PRODUCTION_DEMAND"


@pytest.mark.parametrize("success", [True, False])
def test_real_router_serializes_targeted_recovery_without_unapproved_item(monkeypatch, success):
    import importlib.util
    import sys
    import types
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    items = candidates(4)
    bands = {item["originText"]: 4 for item in items}
    if success:
        bands[items[2]["originText"]] = 5
    provider = WritingPipelineProvider(batches(items), bands)
    dependencies = types.ModuleType("app.api.dependencies")
    dependencies.get_language_learning_writing_service = lambda: service(provider)
    monkeypatch.setitem(sys.modules, "app.api.dependencies", dependencies)
    filename = Path(__file__).resolve().parents[1] / "app/api/v1/language_learning.py"
    spec = importlib.util.spec_from_file_location("writing_difficulty_recovery_route", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/v1")
    with TestClient(app) as client:
        response = client.post("/api/v1/language-learning/writing/daily/generate", json=request(
            band=4, difficulty="CHALLENGE").model_dump(mode="json", by_alias=True))
    assert response.status_code == (200 if success else 422)
    data = response.json()
    if success:
        assert data["promptVersion"] == "writing-generation-difficulty-recovery-v1"
        assert len(data["items"]) == 1
        assert data["items"][0]["languageComplexityBand"] == 5
    else:
        assert data["detail"]["difficultyRecoveryUsed"] is True
        assert data["detail"]["observedBandCounts"] == {"4": 4}
        assert "items" not in data
    assert provider.counts[GENERATION_TASK] == 2


def test_overall_deadline_still_cancels_targeted_generator():
    from app.features.language_learning.writing.service import LanguageLearningWritingService
    async def scenario():
        items = candidates(4)
        entered, cleaned = asyncio.Event(), asyncio.Event()
        class Provider(WritingPipelineProvider):
            async def call(self, type_name, data, schema=None):
                if type_name == GENERATION_TASK and self.counts[GENERATION_TASK] == 1:
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cleaned.set()
                return await super().call(type_name, data, schema)
        provider = Provider(batches(items), {item["originText"]: 4 for item in items})
        svc = LanguageLearningWritingService(provider, generation_timeout_seconds=5,
                                             verification_timeout_seconds=5,
                                             generation_total_timeout_seconds=0.1)
        with pytest.raises(HTTPException) as exc:
            await svc.generate_daily(request(band=4, difficulty="CHALLENGE"))
        assert exc.value.status_code == 504
        assert entered.is_set() and cleaned.is_set()
        assert not [other for other in asyncio.all_tasks() if other is not asyncio.current_task() and not other.done()]
    asyncio.run(scenario())
