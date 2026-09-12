from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.features.language_learning.difficulty.contracts import (
    DifficultyAcceptanceDecision,
    DifficultySpec,
    DifficultyTarget,
    DifficultyValidationResult,
    SemanticAssessment,
)


class ExampleSpec:
    def __init__(self, target):
        self.target = target
        self.version = "example-v1"

def test_target_is_immutable_namespaced_and_does_not_impose_a_band_range():
    target = DifficultyTarget("listening", "acoustic-load", "FAST_NOISY", "audio-v1")
    assert target.value == "FAST_NOISY"
    assert target != DifficultyTarget("writing", "acoustic-load", "FAST_NOISY", "audio-v1")
    with pytest.raises(FrozenInstanceError):
        target.value = "SLOW"  # type: ignore[misc]


def test_spec_is_a_protocol_not_a_common_writing_dto():
    spec = ExampleSpec(DifficultyTarget("example", "custom", 999, "target-v1"))
    assert isinstance(spec, DifficultySpec)
    assert not hasattr(spec, "generation_payload")


def test_validation_result_invariants_and_opaque_issue_codes():
    accepted = DifficultyValidationResult.accept(measurements={"serviceMetric": 2})
    rejected = DifficultyValidationResult.reject("SERVICE_PRIVATE_CODE")
    assert accepted.passed and accepted.primary_issue is None
    assert rejected.issues == ("SERVICE_PRIVATE_CODE",)
    with pytest.raises(ValueError):
        DifficultyValidationResult(True, ("BAD",))
    with pytest.raises(ValueError):
        DifficultyValidationResult(False)


@pytest.mark.parametrize("confidence", [None, 0.0, 0.5, 1.0])
def test_semantic_assessment_keeps_confidence_diagnostic(confidence):
    assessment = SemanticAssessment(
        status="PASS",
        difficulty_status="SERVICE_ASSESSED",
        observed_target="opaque",
        issue_codes=("SERVICE_NOTE",),
        confidence=confidence,
    )
    assert assessment.status == "PASS" and assessment.confidence == confidence


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), True])
def test_semantic_assessment_rejects_invalid_diagnostic_confidence(confidence):
    with pytest.raises(ValueError):
        SemanticAssessment(
            status="PASS", difficulty_status="ASSESSED", confidence=confidence,
        )


def test_acceptance_decision_keeps_service_actions_and_reasons_opaque():
    decision = DifficultyAcceptanceDecision("SERVICE_REPAIR", "PRIVATE_REASON")
    assert (decision.action, decision.reason) == ("SERVICE_REPAIR", "PRIVATE_REASON")


def test_common_difficulty_has_no_writing_imports():
    root = Path(__file__).parents[1] / "app/features/language_learning/difficulty"
    forbidden = "app.features.language_learning.writing"
    imports = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
    assert not [name for name in imports if name == forbidden or name.startswith(forbidden + ".")]


def test_common_runtime_does_not_own_writing_diagnostic_projection():
    root = Path(__file__).parents[1] / "app/features/language_learning/difficulty"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    forbidden = {
        "review_calls", "review_elapsed_ms", "review_failures", "review_retries",
        "review_backoff_ms", "peak_inflight_reviews", "estimated_band",
        "confidence_used_for_acceptance", "INVALID_BAND",
        "verdict",
    }
    assert not {value for value in forbidden if value in source}


def test_common_source_does_not_own_writing_identity_wire_fields():
    root = Path(__file__).parents[1] / "app/features/language_learning/difficulty"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert '"candidateId"' not in source
    assert '"contentHash"' not in source


def test_common_orchestrator_does_not_interpret_service_action_literals():
    root = Path(__file__).parents[1] / "app/features/language_learning/difficulty"
    source = (root / "orchestration.py").read_text(encoding="utf-8")
    assert '"ADJUDICATE"' not in source
    assert 'DifficultyAcceptanceDecision("REJECT"' not in source
