import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.novel_quality_verification import QualityLedger
from scripts.qa_campaign_budget import CampaignBudgetExceeded, Reservation, atomic_json


def setup_guard(tmp_path, *, cap=1.0, starts=2, protected=0.0):
    # 준비: 이전 단계의 확정 장부와 별도 불변 admission을 연결한다.
    ledger_path = tmp_path / "ledger.json"
    ledger = QualityLedger(ledger_path)
    attempt = ledger.reserve("PRIOR", "gpt-6-luna", Reservation(usd=1), metadata={"phase": "final"})
    ledger.finish(attempt, status="COMPLETED", elapsed=0, accounted=Reservation(usd=1))
    prior = ledger.snapshot()["calls"]
    path = tmp_path / "guard.json"
    atomic_json(
        path,
        {
            "stageId": "fresh-browser-80",
            "campaignId": ledger.campaign_id,
            "startAttempts": 1,
            "startUsd": 1.0,
            "priorCallsSha256": hashlib.sha256(
                json.dumps(prior, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "maxStarts": starts,
            "maxUsd": cap,
            "protectedRemainingUsd": protected,
            "allowedTasks": ["NOVEL_PROFILE", "NOVEL_LEGACY"],
            "allowedPhases": ["final"],
        },
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ledger_path, path, digest


def open_guard(binding):
    ledger_path, path, digest = binding
    return QualityLedger(ledger_path, stage_budget_path=path, stage_budget_sha256=digest)


def reserve(ledger, usd=0.6, task="NOVEL_PROFILE", phase="final"):
    return ledger.reserve(task, "gpt-6.1-sol", Reservation(usd=usd), metadata={"phase": phase})


def test_stage_concurrent_reservations_are_atomic(tmp_path):
    binding = setup_guard(tmp_path)

    def try_reserve(_):
        try:
            return reserve(open_guard(binding))
        except CampaignBudgetExceeded:
            return None

    # 실행 / 검증: 두 인스턴스가 경쟁해도 하나만 시작한다.
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(try_reserve, range(2)))
    assert sum(result is not None for result in results) == 1
    assert open_guard(binding).snapshot()["totalsIncludingReserved"]["usd"] == pytest.approx(1.6)


def test_stage_failed_unknown_and_restart_keep_full_hold(tmp_path):
    binding = setup_guard(tmp_path)
    ledger = open_guard(binding)
    attempt = reserve(ledger)
    ledger.finish(attempt, status="FAILED", elapsed=0, failure="TimeoutError")

    # 실행 / 검증: 재시도도 새 호출이며 실패의 미관측 예약을 지우지 않는다.
    resumed = open_guard(binding)
    with pytest.raises(CampaignBudgetExceeded, match="STAGE_USD_LIMIT"):
        reserve(resumed)
    assert len(resumed.snapshot()["calls"]) == 2


def test_stage_settled_retry_and_exact_cap_then_stop(tmp_path):
    binding = setup_guard(tmp_path)
    ledger = open_guard(binding)
    first = reserve(ledger, 0.8)
    ledger.finish(first, status="FAILED", elapsed=0, accounted=Reservation(usd=0.4))
    reserve(open_guard(binding), 0.6)
    with pytest.raises(CampaignBudgetExceeded, match="STAGE_START_LIMIT"):
        reserve(open_guard(binding), 0.0)
    assert len(ledger.snapshot()["calls"]) == 3


@pytest.mark.parametrize("operation", ["missing", "changed"])
def test_stage_guard_file_missing_or_changed_fails_closed(tmp_path, operation):
    binding = setup_guard(tmp_path)
    ledger = open_guard(binding)
    if operation == "missing":
        binding[1].unlink()
    else:
        binding[1].write_text("{}", encoding="utf-8")
    with pytest.raises(CampaignBudgetExceeded, match="STAGE_ADMISSION"):
        reserve(ledger)
    assert len(ledger.snapshot()["calls"]) == 1


def test_stage_foreign_call_after_start_is_rejected(tmp_path):
    binding = setup_guard(tmp_path)
    QualityLedger(binding[0]).reserve(
        "OTHER", "gpt-6-luna", Reservation(usd=0.1), metadata={"phase": "final"}
    )
    with pytest.raises(CampaignBudgetExceeded, match="STAGE_FOREIGN_CALL"):
        open_guard(binding)


@pytest.mark.parametrize("task,phase", [("OPENAI_TTS", "final"), ("NOVEL_PROFILE", "explore")])
def test_stage_scope_is_explicit(tmp_path, task, phase):
    ledger = open_guard(setup_guard(tmp_path))
    with pytest.raises(CampaignBudgetExceeded, match="STAGE_SCOPE"):
        reserve(ledger, task=task, phase=phase)
    assert len(ledger.snapshot()["calls"]) == 1


def test_stage_protected_other_budget_survives(tmp_path):
    ledger = open_guard(setup_guard(tmp_path, cap=10, protected=48.5))
    with pytest.raises(CampaignBudgetExceeded, match="STAGE_OTHER_RESERVE"):
        reserve(ledger, 0.6)
    assert ledger.snapshot()["stoppedReason"] is None


def test_stage_prefix_change_and_active_prior_are_rejected(tmp_path):
    binding = setup_guard(tmp_path)
    state = json.loads(binding[0].read_text())
    state["calls"][0]["status"] = "STARTED"
    atomic_json(binding[0], state)
    with pytest.raises(CampaignBudgetExceeded, match="STAGE_PRIOR"):
        open_guard(binding)


def test_stage_requires_both_path_and_hash(tmp_path):
    with pytest.raises(ValueError, match="Both"):
        QualityLedger(tmp_path / "ledger.json", stage_budget_path=tmp_path / "guard.json")


def speech_guard(tmp_path, *, tasks=None, phases=None):
    binding = setup_guard(tmp_path, cap=5, starts=10)
    guard = json.loads(binding[1].read_text())
    guard["stageId"] = "final-speech-only"
    guard["allowedTasks"] = ["OPENAI_TTS"] if tasks is None else tasks
    guard["allowedPhases"] = ["tts"] if phases is None else phases
    atomic_json(binding[1], guard)
    return binding[0], binding[1], hashlib.sha256(binding[1].read_bytes()).hexdigest()


def test_speech_only_stage_limits_concurrent_attempts_to_ten(tmp_path):
    # 준비: 음성만 허용하는 10회/5달러 예약을 여러 인스턴스가 공유한다.
    binding = speech_guard(tmp_path)

    def attempt(_):
        try:
            return reserve(open_guard(binding), 0.5, "OPENAI_TTS", "tts")
        except CampaignBudgetExceeded:
            return None

    # 실행 / 검증: 11번째 요청은 외부 호출 전에 거절한다.
    with ThreadPoolExecutor(max_workers=11) as pool:
        results = list(pool.map(attempt, range(11)))
    assert sum(result is not None for result in results) == 10
    state = open_guard(binding).snapshot()
    assert state["totalsIncludingReserved"]["usd"] == pytest.approx(6)
    assert state["stoppedReason"] is None


@pytest.mark.parametrize("task,phase", [("NOVEL_PROFILE", "final"), ("OPENAI_TTS", "final")])
def test_speech_stage_cannot_admit_translation_or_wrong_phase(tmp_path, task, phase):
    # 준비 / 실행 / 검증: 음성 단계가 기존 번역 허용 범위까지 넓히지 않는다.
    ledger = open_guard(speech_guard(tmp_path))
    with pytest.raises(CampaignBudgetExceeded, match="STAGE_SCOPE"):
        reserve(ledger, 0.5, task, phase)
    assert len(ledger.snapshot()["calls"]) == 1


@pytest.mark.parametrize("tasks,phases", [
    (["NOVEL_PROFILE", "OPENAI_TTS"], ["tts"]),
    (["OPENAI_TTS"], ["final", "tts"]),
    (["NOVEL_PROFILE"], ["final"]),
])
def test_stage_rejects_unapproved_scope_combinations(tmp_path, tasks, phases):
    # 준비 / 실행 / 검증: 승인된 정확한 조합 밖의 admission은 거절한다.
    with pytest.raises(CampaignBudgetExceeded, match="STAGE_ADMISSION_INVALID"):
        open_guard(speech_guard(tmp_path, tasks=tasks, phases=phases))


def test_recovery_and_speech_share_one_atomic_cap_and_preserve_unknown_usage(tmp_path):
    # 준비: 실제 읽기 중 명시 복구도 같은 추가 예산 안에서만 허용한다.
    binding = setup_guard(tmp_path, cap=1, starts=3)
    guard = json.loads(binding[1].read_text())
    guard["allowedPhases"] = ["final", "tts"]
    guard["allowedTasks"] = ["NOVEL_PROFILE", "NOVEL_LEGACY", "OPENAI_TTS"]
    atomic_json(binding[1], guard)
    binding = binding[0], binding[1], hashlib.sha256(binding[1].read_bytes()).hexdigest()
    ledger = open_guard(binding)

    # 실행: 사용량 미관측 음성의 예약은 복구 요청에도 합산된다.
    speech = reserve(ledger, 0.5, "OPENAI_TTS", "tts")
    ledger.finish(speech, status="COMPLETED", elapsed=0)
    reserve(open_guard(binding), 0.5)

    # 검증: 합계 상한과 task/phase 경계는 혼합 단계에서도 유지된다.
    with pytest.raises(CampaignBudgetExceeded, match="STAGE_USD_LIMIT"):
        reserve(open_guard(binding), 0.01)
    for task, phase in [("OPENAI_TTS", "final"), ("NOVEL_PROFILE", "tts")]:
        with pytest.raises(CampaignBudgetExceeded, match="STAGE_SCOPE"):
            reserve(open_guard(binding), 0, task, phase)
    assert open_guard(binding).snapshot()["totalsIncludingReserved"]["usd"] == pytest.approx(2)
