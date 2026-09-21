"""Offline guards for the precommitted Reading A/B QA corpus."""

from datetime import date

from app.schemas.language_learning_practice import PracticeGenerationRequest
from scripts.qa_reading_model_comparison import CASES, _manifest, _requests
from scripts.qa_reading_selected_holdout import CASES as HOLDOUT_CASES, manifest as holdout_manifest


def test_fixed_six_cases_two_arms_and_stable_date(tmp_path):
    manifest = _manifest(tmp_path, learning_date="2026-09-20")
    assert len(manifest["cases"]) == 6
    assert len(manifest["runOrder"]) == 12
    assert manifest["arms"] == {"A": "LUNA", "B": "SOL"}
    assert manifest["learningDate"] == "2026-09-20"
    assert manifest == _manifest(tmp_path, learning_date="2026-09-20")
    assert {case["band"] for case in manifest["cases"]} == {1, 3, 5}
    assert all(len([entry for entry in manifest["runOrder"]
                    if entry["caseId"] == case["id"]]) == 2 for case in CASES)


def test_p1_p2_requests_preserve_be_difficulty_rotation_without_provider():
    for case in CASES:
        first, second = _requests(case, date(2026, 9, 20).isoformat())
        assert (first["questionCount"], second["questionCount"]) == (3, 2)
        assert (first["easierCount"], first["currentCount"], first["challengeCount"]) == (1, 2, 0)
        assert (second["easierCount"], second["currentCount"], second["challengeCount"]) == (0, 1, 1)
        assert first["selectedKeywords"] == second["selectedKeywords"] == case["keywords"]
        assert first["weakSignals"] == second["weakSignals"] == case["weak"]
        PracticeGenerationRequest.model_validate({**first, "requestId": "fixed-p1"})


def test_selected_sol_holdouts_are_unseen_and_keep_all_five_bands(tmp_path):
    (tmp_path / "reading-comparison-summary.json").write_text("{}", encoding="utf-8")
    manifest = holdout_manifest(tmp_path, learning_date="2026-09-21")
    assert manifest == holdout_manifest(tmp_path, learning_date="2026-09-21")
    assert manifest["model"] == "SOL"
    assert [case["band"] for case in HOLDOUT_CASES] == [1, 2, 3, 4, 5]
    assert {case["id"] for case in HOLDOUT_CASES}.isdisjoint({case["id"] for case in CASES})
    for case in HOLDOUT_CASES:
        first, second = _requests(case, manifest["learningDate"])
        assert (first["questionCount"], second["questionCount"]) == (3, 2)
        assert (first["easierCount"], first["currentCount"], first["challengeCount"]) == (1, 2, 0)
        assert (second["easierCount"], second["currentCount"], second["challengeCount"]) == (0, 1, 1)
