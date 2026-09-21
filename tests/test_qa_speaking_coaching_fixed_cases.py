from scripts.qa_speaking_coaching_fixed_cases import FIXED_CASES, _request


def test_six_cases_are_fixed_across_required_quality_classes() -> None:
    assert len(FIXED_CASES) == 6
    assert [case["class"] for case in FIXED_CASES].count("SUFFICIENT_NORMAL") == 2
    assert [case["class"] for case in FIXED_CASES].count("UNDERSTANDABLE_ERROR") == 2
    assert [case["class"] for case in FIXED_CASES].count("INSUFFICIENT_EVIDENCE") == 2


def test_fixed_requests_pin_coaching_policy_and_source_identity() -> None:
    requests = [_request(case) for case in FIXED_CASES]
    assert all(item["resultKind"] == "SESSION_COACHING" for item in requests)
    assert all(item["resultPolicyVersion"] == "free-session-coaching-v1" for item in requests)
    assert all(len(item["sourceSnapshotHash"]) == 64 for item in requests)
    assert len({item["sourceSnapshotHash"] for item in requests}) == 6
