"""Bounded, offline-by-default CONTEXTUAL_CHOICE baseline/candidate QA.

Dry-run never constructs a provider. A live run requires an explicit variant plus
call, input-token, output-token, and wall-clock caps. Run baseline and candidate
from separate source trees, then compare their immutable JSON result files.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import unicodedata
from typing import Any, Callable, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.features.language_learning.reading_vocabulary.contextual_choice_task import (  # noqa: E402
    CONTEXTUAL_CHOICE_RECIPE_VERSION,
    contextual_choice_answer_key,
)
from app.features.language_learning.reading_vocabulary.service import (  # noqa: E402
    ReadingVocabularyGenerationService,
)
from app.ai.model_policy import (  # noqa: E402
    get_model_name_for_task,
    get_task_model_policy,
)
from app.schemas.language_learning_practice import PracticeGenerationRequest  # noqa: E402


DEFAULT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "contextual-choice-stabilization-qa-v1.json"
)
DAILY_DIFFICULTIES = (
    "CURRENT",
    "EASIER",
    "CURRENT",
    "CHALLENGE",
    "CURRENT",
    "EASIER",
    "CURRENT",
    "CHALLENGE",
    "CURRENT",
    "CURRENT",
)
SUGGESTED_LIVE_CAPS = {
    "maxProviderCalls": 366,
    "maxInputTokens": 1_500_000,
    "maxOutputTokens": 500_000,
    "maxCallSeconds": 90,
    "maxSeconds": 3600,
}
SUGGESTED_ORDER_DIAGNOSTIC_CAPS = {
    "maxProviderCalls": 12,
    "maxInputTokens": 120_000,
    "maxOutputTokens": 40_000,
    "maxCallSeconds": 90,
    "maxSeconds": 600,
}


class QaBudgetExceeded(RuntimeError):
    pass


class QaCallTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class QaCaps:
    provider_calls: int
    input_tokens: int
    output_tokens: int
    call_seconds: float
    seconds: float


@dataclass
class RecordingProvider:
    upstream: Any
    caps: QaCaps
    diagnostic_capture: bool = False
    started_at: float = field(default_factory=time.perf_counter)
    calls: list[dict[str, Any]] = field(default_factory=list)
    budget_exhausted_reason: str | None = None

    def _check_before_call(self) -> None:
        if self.budget_exhausted_reason is not None:
            raise QaBudgetExceeded(self.budget_exhausted_reason)
        if len(self.calls) >= self.caps.provider_calls:
            self.budget_exhausted_reason = "provider call cap reached"
            raise QaBudgetExceeded(self.budget_exhausted_reason)
        if time.perf_counter() - self.started_at >= self.caps.seconds:
            self.budget_exhausted_reason = "wall-clock cap reached"
            raise QaBudgetExceeded(self.budget_exhausted_reason)

    def _start_call(self, type_name: str, data: str) -> dict[str, Any]:
        self._check_before_call()
        record: dict[str, Any] = {
            "attempt": len(self.calls) + 1,
            "status": "STARTED",
            "task": type_name,
            "provider": None,
            "model": None,
            "latencyMs": 0.0,
            "inputTokens": 0,
            "outputTokens": 0,
            "providerInternalRetries": "NOT_EXPOSED_BY_PROVIDER_INTERFACE",
            "diagnostic": _safe_call_diagnostic(data, type_name=type_name),
        }
        if self.diagnostic_capture:
            record["diagnosticCapture"] = _diagnostic_request_capture(type_name, data)
        # Append before entering the provider so exceptions and cancellation remain
        # visible and count against the application-level call budget.
        self.calls.append(record)
        return record

    def _finish_timing(self, record: dict[str, Any], started: float) -> None:
        record["latencyMs"] = round((time.perf_counter() - started) * 1000, 3)

    def _check_after_call(self, record: dict[str, Any]) -> None:
        reason = None
        if sum(item["inputTokens"] for item in self.calls) > self.caps.input_tokens:
            reason = "input token cap exceeded after the last completed call"
        elif sum(item["outputTokens"] for item in self.calls) > self.caps.output_tokens:
            reason = "output token cap exceeded after the last completed call"
        elif time.perf_counter() - self.started_at >= self.caps.seconds:
            reason = "wall-clock cap exceeded during the last completed call"
        if reason is None:
            return
        self.budget_exhausted_reason = reason
        record["budgetExceededAfterCompletion"] = reason
        raise QaBudgetExceeded(reason)

    async def call(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> Any:
        record = self._start_call(type_name, data)
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.upstream.call_with_metadata(
                    type_name=type_name,
                    data=data,
                    schema=schema,
                ),
                timeout=self.caps.call_seconds,
            )
        except TimeoutError as exc:
            record["status"] = "FAILED"
            record["failure"] = {
                "type": "QA_CALL_TIMEOUT",
                "timeoutSeconds": self.caps.call_seconds,
            }
            self._finish_timing(record, started)
            self.budget_exhausted_reason = (
                f"provider call timed out after {self.caps.call_seconds:g} seconds"
            )
            raise QaCallTimeout(self.budget_exhausted_reason) from exc
        except BaseException as exc:
            record["status"] = (
                "CANCELLED" if isinstance(exc, asyncio.CancelledError) else "FAILED"
            )
            record["failure"] = {"type": type(exc).__name__}
            self._finish_timing(record, started)
            raise
        record.update(
            {
                "status": "SUCCEEDED",
                "provider": str(getattr(result, "provider", "unknown")),
                "model": str(getattr(result, "model", "unknown")),
                "inputTokens": int(getattr(result, "input_tokens", 0) or 0),
                "outputTokens": int(getattr(result, "output_tokens", 0) or 0),
            }
        )
        self._finish_timing(record, started)
        if self.diagnostic_capture:
            record["diagnosticCapture"]["response"] = _json_copy(result.data)
        self._check_after_call(record)
        return result.data

    async def call_with_image(self, *args: Any, **kwargs: Any) -> Any:
        type_name = str(kwargs.get("type_name", args[0] if args else "IMAGE_CALL"))
        prompt = str(kwargs.get("prompt", args[1] if len(args) > 1 else ""))
        record = self._start_call(type_name, prompt)
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.upstream.call_with_image(*args, **kwargs),
                timeout=self.caps.call_seconds,
            )
        except TimeoutError as exc:
            record["status"] = "FAILED"
            record["failure"] = {
                "type": "QA_CALL_TIMEOUT",
                "timeoutSeconds": self.caps.call_seconds,
            }
            self._finish_timing(record, started)
            self.budget_exhausted_reason = (
                f"provider call timed out after {self.caps.call_seconds:g} seconds"
            )
            raise QaCallTimeout(self.budget_exhausted_reason) from exc
        except BaseException as exc:
            record["status"] = (
                "CANCELLED" if isinstance(exc, asyncio.CancelledError) else "FAILED"
            )
            record["failure"] = {"type": type(exc).__name__}
            self._finish_timing(record, started)
            raise
        record["status"] = "SUCCEEDED"
        self._finish_timing(record, started)
        if self.diagnostic_capture:
            record["diagnosticCapture"]["response"] = _json_copy(result)
        self._check_after_call(record)
        return result


def load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 3:
        raise ValueError("QA fixture must contain exactly three fixed fresh profiles")
    ids = [profile.get("profileId") for profile in profiles]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Every QA profile requires profileId")
    if len(ids) != len(set(ids)):
        raise ValueError("QA profileId values must be unique")
    return payload


def load_order_diagnostic_case(path: Path) -> dict[str, Any]:
    """Load an exact persisted snapshot and the accepted prefix before a target order."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_order_diagnostic_case(payload)


def validate_order_diagnostic_case(payload: dict[str, Any]) -> dict[str, Any]:
    practice_set_id = payload.get("practiceSetId")
    target_order = payload.get("targetOrder")
    if type(practice_set_id) is not int or practice_set_id <= 0:
        raise ValueError("practiceSetId must be a positive integer, not bool")
    if type(target_order) is not int:
        raise ValueError("targetOrder must be an integer, not bool")
    snapshot = payload.get("requestSnapshot")
    accepted_prefix = payload.get("acceptedPrefix")
    if not isinstance(snapshot, dict):
        raise ValueError(
            "diagnostic case requires the persisted requestSnapshot object"
        )
    snapshot_request = PracticeGenerationRequest.model_validate(snapshot)
    if not isinstance(accepted_prefix, list):
        raise ValueError("diagnostic case requires an acceptedPrefix array")
    if target_order not in range(1, snapshot_request.question_count + 1):
        raise ValueError("targetOrder must be within requestSnapshot.questionCount")
    if len(accepted_prefix) != target_order - 1:
        raise ValueError("acceptedPrefix length must equal targetOrder minus one")
    prefix_orders = [
        item.get("order") if isinstance(item, dict) else None
        for item in accepted_prefix
    ]
    if prefix_orders != list(range(1, target_order)):
        raise ValueError("acceptedPrefix orders must be contiguous 1..targetOrder-1")
    if (
        snapshot_request.domain.value != "VOCABULARY"
        or snapshot_request.mode != "CONTEXTUAL_CHOICE"
        or snapshot_request.question_count != 10
        or snapshot_request.previous_questions
        or snapshot_request.vocabulary_plan_only
        or snapshot_request.vocabulary_plan is None
    ):
        raise ValueError(
            "requestSnapshot must be the persisted ten-item request with its plan"
        )
    build_order_diagnostic_request(payload)
    return payload


def _be_progressive_difficulty_at(
    snapshot: PracticeGenerationRequest,
    order: int,
) -> str:
    """Mirror BE PracticePersistenceService.difficultyAt without changing BE."""
    remaining = [
        snapshot.easier_count,
        snapshot.current_count,
        snapshot.challenge_count,
    ]
    position = 0
    for _ in range(snapshot.question_count):
        progressed = False
        for difficulty in (1, 0, 1, 2):
            if remaining[difficulty] <= 0:
                continue
            remaining[difficulty] -= 1
            progressed = True
            position += 1
            if position == order:
                return ("EASIER", "CURRENT", "CHALLENGE")[difficulty]
        if not progressed:
            break
    raise ValueError(
        "requestSnapshot difficulty distribution does not cover targetOrder"
    )


def build_order_diagnostic_request(
    case: dict[str, Any],
) -> PracticeGenerationRequest:
    """Reproduce BE itemRequest(snapshot, targetOrder, prefix, freshToken)."""
    order = cast(int, case["targetOrder"])
    snapshot = PracticeGenerationRequest.model_validate(case["requestSnapshot"])
    accepted_prefix = copy.deepcopy(case["acceptedPrefix"])
    difficulty = _be_progressive_difficulty_at(snapshot, order)
    reviews = [
        item.model_dump(mode="json", by_alias=True) for item in snapshot.review_targets
    ]
    review_index = order - 1
    review = review_index < snapshot.review_question_count and review_index < len(
        reviews
    )
    if review:
        selected = reviews.pop(review_index)
        reviews.insert(0, selected)
    payload = snapshot.model_dump(mode="json", by_alias=True)
    payload.update(
        {
            # Keep this stable across baseline/candidate to compare identical model input.
            # It is diagnostic-only and never persisted back to the PracticeSet.
            "requestId": f"qa-set-{case['practiceSetId']}-order-{order}",
            "questionCount": 1,
            "easierCount": int(difficulty == "EASIER"),
            "currentCount": int(difficulty == "CURRENT"),
            "challengeCount": int(difficulty == "CHALLENGE"),
            "reviewTargets": reviews,
            "reviewQuestionCount": int(review),
            "previousQuestions": accepted_prefix,
            "vocabularyPlanOnly": False,
        }
    )
    request = PracticeGenerationRequest.model_validate(payload)
    if request.question_offset != order - 1 or request.vocabulary_plan is None:
        raise ValueError("diagnostic order request must preserve prefix and plan")
    # Reuse the production deterministic authority/binding validator before any
    # provider is constructed. This makes a dry-run reject plan/prefix drift.
    validator = ReadingVocabularyGenerationService(cast(Any, None))
    validate_authority = getattr(
        validator, "_validate_contextual_choice_plan_authority"
    )
    validate_authority(request, request.vocabulary_plan)
    return request


def order_diagnostic_manifest(path: Path, case: dict[str, Any]) -> dict[str, Any]:
    request = build_order_diagnostic_request(case)
    order = cast(int, case["targetOrder"])
    assert request.vocabulary_plan is not None
    slot = request.vocabulary_plan.items[order - 1]
    return {
        "mode": "ORDER_DIAGNOSTIC_DRY_RUN",
        "liveProviderCalls": 0,
        "practiceSetId": case["practiceSetId"],
        "targetOrder": case["targetOrder"],
        "caseSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "acceptedPrefixCount": len(case["acceptedPrefix"]),
        "planVersion": case["requestSnapshot"]["vocabularyPlan"].get("version"),
        "providerConstructed": False,
        "sourceGitBlobSha1": source_git_blob_fingerprint(),
        "effectiveRequest": {
            "questionCount": request.question_count,
            "questionOffset": request.question_offset,
            "easierCount": request.easier_count,
            "currentCount": request.current_count,
            "challengeCount": request.challenge_count,
            "reviewQuestionCount": request.review_question_count,
            "setComplexityBand": request.complexity_band,
            "slotComplexityBand": slot.complexity_band,
            "slotDifficulty": slot.difficulty.value,
            "slotSkillTag": slot.skill_tag,
            "serverCorrectAnswerKey": contextual_choice_answer_key(order),
            "persistedPlanReused": True,
        },
        "suggestedExplicitLiveCaps": SUGGESTED_ORDER_DIAGNOSTIC_CAPS,
        "executionBoundary": (
            f"A live run mirrors BE itemRequest for order {order} from the exact persisted "
            "snapshot and accepted prefix, then invokes service.generate exactly once. "
            f"It does not generate a plan or replay orders 1..{order - 1}."
        ),
        "diagnosticCaptureRequiredForLive": True,
    }


def source_fingerprint() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in _source_paths()
    }


def source_git_blob_fingerprint() -> dict[str, str]:
    result = {}
    for path in _source_paths():
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        completed = subprocess.run(  # noqa: S603 - fixed executable and local paths
            ["git", "hash-object", f"--path={relative}", str(path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result[relative] = completed.stdout.strip()
    return result


def _source_paths() -> list[Path]:
    return [
        ROOT / "app/features/language_learning/reading_vocabulary/service.py",
        ROOT / "app/features/language_learning/reading_vocabulary/prompts.py",
        ROOT
        / "app/features/language_learning/reading_vocabulary/contextual_choice_task.py",
        ROOT / "app/schemas/language_learning_practice.py",
    ]


def _live_task_model_policies(upstream: Any) -> dict[str, dict[str, Any]]:
    tasks = (
        ReadingVocabularyGenerationService.TYPE_NAME,
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_PLAN_VERIFICATION_TYPE_NAME,
        ReadingVocabularyGenerationService.CONTEXTUAL_CHOICE_VERIFICATION_TYPE_NAME,
        ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME,
        ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME,
    )
    provider_name_for = getattr(upstream, "provider_name_for", None)
    result = {}
    for task in tasks:
        policy = get_task_model_policy(task)
        provider_name = (
            str(provider_name_for(task)) if callable(provider_name_for) else "unknown"
        )
        result[task] = {
            "configuredProvider": provider_name,
            "tier": policy.tier.value,
            "model": get_model_name_for_task(task),
            "reasoningEffort": policy.reasoning_effort,
            "maxOutputTokens": policy.max_output_tokens,
            "verbosity": policy.verbosity,
        }
    return result


def dry_run_manifest(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "DRY_RUN",
        "fixtureVersion": fixture["version"],
        "liveProviderCalls": 0,
        "recipeVersion": CONTEXTUAL_CHOICE_RECIPE_VERSION,
        "profiles": [profile["profileId"] for profile in fixture["profiles"]],
        "historicalCases": fixture.get("historicalCases", []),
        "normalApplicationCallEstimate": {
            "perProfile": 41,
            "total": 123,
            "basis": "1 plan call plus 10 items x lexical/context/verification/explanation",
        },
        "existingWorstPathApplicationCallCeiling": {
            "perProfile": 122,
            "total": 366,
            "providerInternalRetriesIncluded": False,
        },
        "suggestedExplicitLiveCaps": SUGGESTED_LIVE_CAPS,
        "comparisonProtocol": (
            "Run once from the preserved baseline tree with --variant baseline and once "
            "from the candidate tree with --variant candidate, using this same fixture "
            "and equal caps; then use --compare. Failed runs remain in the denominator."
        ),
        "qualityBoundary": (
            "Production verifier acceptance is not independent linguistic proof. The result "
            "retains every generated question for blinded human/independent review."
        ),
    }


def validate_live_args(args: argparse.Namespace) -> QaCaps:
    required = {
        "variant": args.variant,
        "output": args.output,
        "max-provider-calls": args.max_provider_calls,
        "max-input-tokens": args.max_input_tokens,
        "max-output-tokens": args.max_output_tokens,
        "max-call-seconds": args.max_call_seconds,
        "max-seconds": args.max_seconds,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("live mode requires explicit " + ", ".join(missing))
    numeric = (
        args.max_provider_calls,
        args.max_input_tokens,
        args.max_output_tokens,
        args.max_call_seconds,
        args.max_seconds,
    )
    if any(value <= 0 for value in numeric):
        raise ValueError("all live caps must be positive")
    return QaCaps(
        provider_calls=args.max_provider_calls,
        input_tokens=args.max_input_tokens,
        output_tokens=args.max_output_tokens,
        call_seconds=args.max_call_seconds,
        seconds=args.max_seconds,
    )


def _request_payload(
    fixture: dict[str, Any],
    profile: dict[str, Any],
    *,
    request_id: str,
    previous_questions: list[dict[str, Any]],
    vocabulary_plan: dict[str, Any] | None,
    vocabulary_plan_only: bool,
    review_targets: list[dict[str, Any]],
    review_question_count: int,
) -> dict[str, Any]:
    global_order = len(previous_questions) + 1
    difficulty = DAILY_DIFFICULTIES[min(global_order - 1, 9)]
    question_count = 10 if vocabulary_plan_only else 1
    return {
        "requestId": request_id,
        "domain": "VOCABULARY",
        "mode": "CONTEXTUAL_CHOICE",
        "originLanguage": fixture["originLanguage"],
        "learningLanguage": fixture["learningLanguage"],
        "questionCount": question_count,
        "complexityBand": profile["complexityBand"],
        "easierCount": 2 if vocabulary_plan_only else int(difficulty == "EASIER"),
        "currentCount": 6 if vocabulary_plan_only else int(difficulty == "CURRENT"),
        "challengeCount": 2 if vocabulary_plan_only else int(difficulty == "CHALLENGE"),
        "selectedKeywords": profile["selectedKeywords"],
        "weakSignals": profile["weakSignals"],
        "recentMistakes": profile["recentMistakes"],
        "reviewTargets": review_targets,
        "reviewQuestionCount": review_question_count,
        "generationDate": fixture["generationDate"],
        "previousQuestions": previous_questions,
        "vocabularyPlan": vocabulary_plan,
        "vocabularyPlanOnly": vocabulary_plan_only,
    }


def _review_request_for_order(
    profile: dict[str, Any],
    global_order: int,
) -> tuple[list[dict[str, Any]], int]:
    reviews = list(profile["reviewTargets"])
    if global_order <= len(reviews):
        selected = reviews[global_order - 1]
        return [selected, *[item for item in reviews if item is not selected]], 1
    return reviews, 0


def _normalized(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value).casefold()
        if not character.isspace()
    )


def _json_copy(value: Any) -> Any:
    """Detach provider data while keeping the diagnostic artifact JSON-only."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _prompt_payload(data: str) -> dict[str, Any] | None:
    payload: object | None = None
    for tag in ("practice-data", "plan-data"):
        opening = f"<{tag}>\n"
        closing = f"\n</{tag}>"
        if opening in data and closing in data:
            try:
                payload = json.loads(data.split(opening, 1)[1].split(closing, 1)[0])
            except json.JSONDecodeError:
                return None
            break
    if payload is None and "\n\n{" in data:
        try:
            payload = json.loads("{" + data.rsplit("\n\n{", 1)[1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _safe_call_diagnostic(
    data: str,
    *,
    type_name: str | None = None,
) -> dict[str, Any] | None:
    """Extract only IDs/codes/indexes, never candidate or user text."""
    payload = _prompt_payload(data)
    if not isinstance(payload, dict):
        return None
    operation = payload.get("operation")
    if operation not in {
        "PLAN",
        "PLAN_REPAIR",
        "LEXICAL_REPAIR",
        "CONTEXT",
        "CONTEXT_REPAIR",
    }:
        if "planItem" in payload:
            operation = "PLAN_VERIFY"
        elif "question" in payload:
            operation = "CONTEXT_VERIFY"
        else:
            return None
    bundle = payload.get("fixedLexicalBundle")
    current = payload.get("currentPlanItem")
    plan_item = payload.get("planItem")
    question = payload.get("question")
    order = None
    if isinstance(bundle, dict):
        order = bundle.get("globalOrder")
    elif isinstance(current, dict):
        order = current.get("globalOrder")
    elif isinstance(plan_item, dict):
        order = plan_item.get("globalOrder")
    elif isinstance(question, dict):
        order = question.get("order")
    return {
        "operation": operation,
        "globalOrder": order if isinstance(order, int) else None,
        "task": type_name,
        "repairClass": payload.get("repairClass"),
        "repairTrigger": payload.get("repairTrigger"),
        "repairScope": payload.get("repairScope"),
        "changedIndexes": payload.get("invalidDistractorIndexes", []),
        "failureCodes": payload.get("reasonCodes", []),
    }


def _diagnostic_request_capture(type_name: str, data: str) -> dict[str, Any]:
    """Capture allowlisted model content only for explicit diagnostic executions."""
    payload = _prompt_payload(data)
    if payload is None:
        return {"task": type_name, "operation": "UNCLASSIFIED"}
    safe = _safe_call_diagnostic(data, type_name=type_name) or {}
    operation = safe.get("operation", "UNCLASSIFIED")
    capture: dict[str, Any] = {
        "task": type_name,
        "operation": operation,
        "globalOrder": safe.get("globalOrder"),
    }
    if operation in {"CONTEXT", "CONTEXT_REPAIR"}:
        capture["request"] = {
            "fixedLexicalBundle": _json_copy(payload.get("fixedLexicalBundle")),
            "difficultyDemand": _json_copy(payload.get("difficultyDemand")),
            "repairClass": payload.get("repairClass"),
            "reasonCodes": _json_copy(payload.get("reasonCodes", [])),
            "previousCandidate": _json_copy(payload.get("previousCandidate", {})),
            "failureEvidence": _json_copy(payload.get("failureEvidence", {})),
        }
    elif operation == "CONTEXT_VERIFY":
        capture["request"] = {
            "question": _json_copy(payload.get("question")),
        }
    elif operation == "PLAN_VERIFY":
        capture["request"] = {
            "planItem": _json_copy(payload.get("planItem")),
            "structuralDifficultyDemand": _json_copy(
                payload.get("structuralDifficultyDemand")
            ),
            "previousLexicalIdentities": _json_copy(
                payload.get("previousLexicalIdentities", [])
            ),
        }
    elif operation == "LEXICAL_REPAIR":
        capture["request"] = {
            "currentPlanItem": _json_copy(payload.get("currentPlanItem")),
            "repairTrigger": payload.get("repairTrigger"),
            "reasonCodes": _json_copy(payload.get("reasonCodes", [])),
            "invalidDistractorIndexes": _json_copy(
                payload.get("invalidDistractorIndexes", [])
            ),
            "preserveTarget": payload.get("preserveTarget"),
            "repairScope": payload.get("repairScope"),
            "forbiddenExpressions": _json_copy(
                payload.get("forbiddenExpressions", [])
            ),
            "distractorEvidence": _json_copy(
                payload.get("distractorEvidence", {})
            ),
            "requiredCloseDistractors": payload.get("requiredCloseDistractors"),
            "currentValidCloseDistractorCount": payload.get(
                "currentValidCloseDistractorCount"
            ),
            "additionalCloseDistractorsNeeded": payload.get(
                "additionalCloseDistractorsNeeded"
            ),
            "repairRequirements": _json_copy(
                payload.get("repairRequirements", {})
            ),
            "skillDemand": _json_copy(payload.get("skillDemand", {})),
        }
    elif operation in {"PLAN", "PLAN_REPAIR"}:
        capture["request"] = {
            "operation": operation,
            "requestId": payload.get("requestId"),
            "serverSlots": _json_copy(payload.get("serverSlots", [])),
            "existingPlanItems": _json_copy(payload.get("existingPlanItems", [])),
            "repairReasons": _json_copy(payload.get("repairReasons", {})),
        }
    return capture


def _linked_context_diagnostics(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join each context candidate to the verifier evidence that immediately judged it."""
    pending: dict[int, list[dict[str, Any]]] = {}
    linked: list[dict[str, Any]] = []
    context_counts: Counter[int] = Counter()
    for call in calls:
        capture = call.get("diagnosticCapture")
        if not isinstance(capture, dict):
            continue
        operation = capture.get("operation")
        order = capture.get("globalOrder")
        if not isinstance(order, int):
            continue
        if operation in {"CONTEXT", "CONTEXT_REPAIR"}:
            context_counts[order] += 1
            request = capture.get("request", {})
            response = capture.get("response")
            item = {
                "globalOrder": order,
                "contextAttempt": context_counts[order],
                "generationCallAttempt": call["attempt"],
                "generationStatus": call["status"],
                "operation": operation,
                "lexicalBundle": _json_copy(request.get("fixedLexicalBundle")),
                "difficultyDemand": _json_copy(request.get("difficultyDemand")),
                "previousCandidate": _json_copy(request.get("previousCandidate", {})),
                "priorFailureEvidence": _json_copy(request.get("failureEvidence", {})),
                "contextCandidate": _json_copy(response),
                "renderedOptions": None,
                "verifierResponse": None,
                "judgmentEvidence": None,
            }
            linked.append(item)
            pending.setdefault(order, []).append(item)
        elif operation == "CONTEXT_VERIFY" and pending.get(order):
            # A deterministically invalid context never reaches the verifier. The
            # verifier therefore judges the most recent generated candidate, not
            # necessarily the oldest unpaired candidate for this order.
            item = pending[order].pop()
            request = capture.get("request", {})
            question = request.get("question")
            if isinstance(question, dict):
                item["renderedOptions"] = _json_copy(
                    question.get("renderedOptions", [])
                )
                item["learnerVisiblePrompt"] = question.get("prompt")
            response = capture.get("response")
            item["verifierCallAttempt"] = call["attempt"]
            item["verifierStatus"] = call["status"]
            item["verifierResponse"] = _json_copy(response)
            if isinstance(response, dict):
                verdicts = response.get("verdicts")
                if isinstance(verdicts, list) and verdicts:
                    verdict = verdicts[0]
                    if isinstance(verdict, dict):
                        item["judgmentEvidence"] = {
                            "structurallyWellFormedKeys": _json_copy(
                                verdict.get("structurallyWellFormedKeys", [])
                            ),
                            "bestAnswerKey": verdict.get("bestAnswerKey"),
                            "ambiguous": verdict.get("ambiguous"),
                            "supported": verdict.get("supported"),
                            "skillFit": verdict.get("skillFit"),
                            "definitionLike": verdict.get("definitionLike"),
                            "reason": verdict.get("reason"),
                        }
    return linked


def _call_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    repairs = [
        item["diagnostic"]
        for item in calls
        if item["diagnostic"]
        and item["diagnostic"].get("operation")
        in {
            "PLAN_REPAIR",
            "LEXICAL_REPAIR",
            "CONTEXT_REPAIR",
        }
    ]
    summary = {
        "count": len(calls),
        "attemptsStarted": len(calls),
        "successfulCalls": sum(item["status"] == "SUCCEEDED" for item in calls),
        "failedCalls": sum(item["status"] == "FAILED" for item in calls),
        "cancelledCalls": sum(item["status"] == "CANCELLED" for item in calls),
        "inputTokens": sum(item["inputTokens"] for item in calls),
        "outputTokens": sum(item["outputTokens"] for item in calls),
        "latencyMs": round(sum(item["latencyMs"] for item in calls), 3),
        "byTask": dict(Counter(item["task"] for item in calls)),
        "repairs": repairs,
        "providerInternalRetries": "NOT_EXPOSED_BY_PROVIDER_INTERFACE",
        "budgetAccountingBoundary": (
            "Caps count calls entering this QA wrapper. Provider SDK retries/failover inside "
            "one upstream call are not exposed and cannot be independently counted here."
        ),
        "inFlightTokenCapBoundary": (
            "Token usage is known only after a provider call returns. The final in-flight call "
            "can cross a token cap; it is recorded and all subsequent starts are blocked."
        ),
    }
    if any("diagnosticCapture" in item for item in calls):
        summary["attemptLedger"] = copy.deepcopy(calls)
        summary["lexicalRepairDiagnostics"] = [
            {
                "callAttempt": item["attempt"],
                "status": item["status"],
                "globalOrder": item["diagnosticCapture"].get("globalOrder"),
                "request": copy.deepcopy(item["diagnosticCapture"].get("request", {})),
                "response": copy.deepcopy(item["diagnosticCapture"].get("response")),
            }
            for item in calls
            if item.get("diagnosticCapture", {}).get("operation") == "LEXICAL_REPAIR"
        ]
        summary["linkedContextDiagnostics"] = _linked_context_diagnostics(calls)
    return summary


async def run_profile(
    fixture: dict[str, Any],
    profile: dict[str, Any],
    service: ReadingVocabularyGenerationService,
    recorder: RecordingProvider,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_question_ms: float | None = None
    start_call = len(recorder.calls)
    questions: list[dict[str, Any]] = []
    previous: list[dict[str, Any]] = []
    plan: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    try:
        plan_request = PracticeGenerationRequest.model_validate(
            _request_payload(
                fixture,
                profile,
                request_id=f"qa-{profile['profileId']}-plan",
                previous_questions=[],
                vocabulary_plan=None,
                vocabulary_plan_only=True,
                review_targets=list(profile["reviewTargets"]),
                review_question_count=len(profile["reviewTargets"]),
            )
        )
        snapshot = await service.generate(plan_request)
        if snapshot.vocabulary_plan is None:
            raise RuntimeError("plan-only response omitted vocabularyPlan")
        plan = snapshot.vocabulary_plan.model_dump(mode="json", by_alias=True)
        for global_order in range(1, 11):
            reviews, review_count = _review_request_for_order(profile, global_order)
            request = PracticeGenerationRequest.model_validate(
                _request_payload(
                    fixture,
                    profile,
                    request_id=f"qa-{profile['profileId']}-item-{global_order}",
                    previous_questions=previous,
                    vocabulary_plan=plan,
                    vocabulary_plan_only=False,
                    review_targets=reviews,
                    review_question_count=review_count,
                )
            )
            response = await service.generate(request)
            if response.vocabulary_plan is None or len(response.questions) != 1:
                raise RuntimeError("progressive response contract mismatch")
            plan = response.vocabulary_plan.model_dump(mode="json", by_alias=True)
            question = response.questions[0].model_copy(update={"order": global_order})
            question_wire = question.model_dump(mode="json", by_alias=True)
            questions.append(question_wire)
            previous.append(question_wire)
            if first_question_ms is None:
                first_question_ms = round((time.perf_counter() - started) * 1000, 3)
    except Exception as exc:  # Preserve every failed run in comparison output.
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}

    profile_calls = recorder.calls[start_call:]
    targets = [
        item.get("targetExpression", "") for item in (plan or {}).get("items", [])
    ]
    normalized = [_normalized(value) for value in targets if value]
    signals = {
        *profile["selectedKeywords"],
        *profile["weakSignals"],
        *profile["recentMistakes"],
        fixture["learningLanguage"],
    }
    new_items = [
        item for item in (plan or {}).get("items", []) if not item["reviewTarget"]
    ]
    return {
        "profileId": profile["profileId"],
        "status": "COMPLETED" if len(questions) == 10 and error is None else "FAILED",
        "error": error,
        "plannedQuestionCount": 10,
        "completedQuestionCount": len(questions),
        "firstQuestionMs": first_question_ms,
        "totalMs": round((time.perf_counter() - started) * 1000, 3),
        "calls": _call_summary(profile_calls),
        "planChecks": {
            "reviewCount": sum(
                bool(item["reviewTarget"]) for item in (plan or {}).get("items", [])
            ),
            "difficultyCounts": dict(
                Counter(item["difficulty"] for item in (plan or {}).get("items", []))
            ),
            "skillCounts": dict(
                Counter(item["skillTag"] for item in (plan or {}).get("items", []))
            ),
            "duplicateNormalizedTargets": len(normalized) - len(set(normalized)),
            "newAnchorValuesBoundToProfile": all(
                item.get("anchorValue") in signals for item in new_items
            ),
        },
        "questionsForBlindedHumanReview": [
            {
                "order": question["order"],
                "prompt": question["prompt"],
                "options": question["options"],
                "skillTag": question["skillTag"],
            }
            for question in questions
        ],
        "explanationsForPostAnswerReview": [
            {
                "order": question["order"],
                "explanationLearning": question["explanationLearning"],
                "explanationOrigin": question["explanationOrigin"],
            }
            for question in questions
        ],
        "postReviewAnswerMetadata": [
            {
                "order": question["order"],
                "correctAnswer": question["correctAnswer"],
                "targetExpression": question["targetExpression"],
                "difficulty": question["difficulty"],
                "complexityBand": question["complexityBand"],
            }
            for question in questions
        ],
        "independentContentReview": {
            "status": "NOT_RUN",
            "note": (
                "Runtime verifier acceptance is retained but is not independent quality proof. "
                "Review question content without requested band/answer metadata."
            ),
        },
    }


async def run_live(
    fixture: dict[str, Any],
    *,
    variant: str,
    caps: QaCaps,
    diagnostic_capture: bool = False,
) -> dict[str, Any]:
    # Keep provider SDK imports out of dry-run and comparison-only execution.
    from app.ai.provider_factory import create_text_generation_provider

    upstream = create_text_generation_provider()
    recorder = RecordingProvider(
        upstream=upstream,
        caps=caps,
        diagnostic_capture=diagnostic_capture,
    )
    service = ReadingVocabularyGenerationService(recorder)
    runs = []
    try:
        for profile in fixture["profiles"]:
            runs.append(await run_profile(fixture, profile, service, recorder))
    finally:
        shutdown = getattr(upstream, "shutdown", None)
        if shutdown is not None:
            await shutdown()
    return {
        "mode": "LIVE",
        "variant": variant,
        "createdAt": datetime.now(UTC).isoformat(),
        "fixtureVersion": fixture["version"],
        "recipeVersion": CONTEXTUAL_CHOICE_RECIPE_VERSION,
        "sourceSha256": source_fingerprint(),
        "caps": {
            "providerCalls": caps.provider_calls,
            "inputTokens": caps.input_tokens,
            "outputTokens": caps.output_tokens,
            "callSeconds": caps.call_seconds,
            "seconds": caps.seconds,
        },
        "diagnosticCapture": diagnostic_capture,
        "historicalCases": fixture.get("historicalCases", []),
        "runs": runs,
        "totals": _call_summary(recorder.calls),
        "disclaimer": (
            "Three fresh profiles are smoke evidence, not a success-rate or SLA guarantee. "
            "Failures are retained in the denominator."
        ),
    }


async def run_order_diagnostic_live(
    case_path: Path,
    case: dict[str, Any],
    *,
    variant: str,
    caps: QaCaps,
    partial_output: Path | None = None,
    provider_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run only the target order; never reconstruct its prefix or vocabulary plan."""
    if provider_factory is None:
        from app.ai.provider_factory import create_text_generation_provider

        provider_factory = create_text_generation_provider
    upstream = provider_factory()
    recorder = RecordingProvider(
        upstream=upstream,
        caps=caps,
        diagnostic_capture=True,
    )
    service = ReadingVocabularyGenerationService(recorder)
    started = time.perf_counter()
    response_payload: dict[str, Any] | None = None
    caught: BaseException | None = None
    request = build_order_diagnostic_request(case)
    try:
        async with asyncio.timeout(caps.seconds):
            response = await service.generate(request)
        response_payload = response.model_dump(mode="json", by_alias=True)
    except BaseException as exc:  # Preserve diagnostics even for cancellation/SIGINT.
        caught = exc
        if partial_output is not None:
            _write_payload(
                {
                    "mode": "ORDER_DIAGNOSTIC_LIVE",
                    "variant": variant,
                    "createdAt": datetime.now(UTC).isoformat(),
                    "practiceSetId": case["practiceSetId"],
                    "targetOrder": case["targetOrder"],
                    "status": (
                        "INTERRUPTED"
                        if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
                        else "FAILED"
                    ),
                    "partial": True,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc)[:500],
                    },
                    "diagnosticCapture": True,
                    "calls": _call_summary(recorder.calls),
                },
                partial_output,
            )
    finally:
        shutdown = getattr(upstream, "shutdown", None)
        if shutdown is not None:
            try:
                await shutdown()
            except BaseException as exc:
                if caught is None:
                    caught = exc
    interrupted = isinstance(caught, (KeyboardInterrupt, asyncio.CancelledError))
    status = (
        "COMPLETED"
        if caught is None and response_payload is not None
        else "INTERRUPTED"
        if interrupted
        else "FAILED"
    )
    error = (
        None
        if caught is None
        else {"type": type(caught).__name__, "message": str(caught)[:500]}
    )
    result = {
        "mode": "ORDER_DIAGNOSTIC_LIVE",
        "variant": variant,
        "createdAt": datetime.now(UTC).isoformat(),
        "practiceSetId": case["practiceSetId"],
        "targetOrder": case["targetOrder"],
        "caseSha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "recipeVersion": CONTEXTUAL_CHOICE_RECIPE_VERSION,
        "sourceSha256": source_fingerprint(),
        "sourceGitBlobSha1": source_git_blob_fingerprint(),
        "variantSemantics": "label-only; runs the current checkout without source switching",
        "providerRuntimeType": type(upstream).__name__,
        "taskModelPolicies": _live_task_model_policies(upstream),
        "status": status,
        "partial": status != "COMPLETED",
        "error": error,
        "totalMs": round((time.perf_counter() - started) * 1000, 3),
        "caps": {
            "providerCalls": caps.provider_calls,
            "inputTokens": caps.input_tokens,
            "outputTokens": caps.output_tokens,
            "callSeconds": caps.call_seconds,
            "seconds": caps.seconds,
        },
        "diagnosticCapture": True,
        "replayInput": {
            "requestSnapshot": copy.deepcopy(case["requestSnapshot"]),
            "acceptedPrefix": copy.deepcopy(case["acceptedPrefix"]),
            "effectiveOrderRequest": request.model_dump(mode="json", by_alias=True),
        },
        "response": response_payload,
        "calls": _call_summary(recorder.calls),
        "interpretationBoundary": (
            "This artifact links generated candidates and verifier judgments. It does "
            "not by itself decide whether Japanese structure or the verifier was wrong."
        ),
    }
    if status != "COMPLETED" and partial_output is not None:
        _write_payload(result, partial_output)
    return result


def compare_results(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    if (
        baseline.get("mode") == "ORDER_DIAGNOSTIC_LIVE"
        or candidate.get("mode") == "ORDER_DIAGNOSTIC_LIVE"
    ):
        if baseline.get("caseSha256") != candidate.get("caseSha256"):
            raise ValueError("baseline and candidate diagnostic cases differ")
        return {
            "mode": "ORDER_DIAGNOSTIC_COMPARISON",
            "practiceSetId": baseline.get("practiceSetId"),
            "targetOrder": baseline.get("targetOrder"),
            "caseSha256": baseline.get("caseSha256"),
            "baselineVariant": baseline.get("variant"),
            "candidateVariant": candidate.get("variant"),
            "baselineStatus": baseline.get("status"),
            "candidateStatus": candidate.get("status"),
            "providerCallDelta": (
                candidate["calls"]["attemptsStarted"]
                - baseline["calls"]["attemptsStarted"]
            ),
            "inputTokenDelta": (
                candidate["calls"]["inputTokens"] - baseline["calls"]["inputTokens"]
            ),
            "outputTokenDelta": (
                candidate["calls"]["outputTokens"] - baseline["calls"]["outputTokens"]
            ),
            "reviewProtocol": (
                "Compare calls.linkedContextDiagnostics side by side: lexicalBundle, "
                "contextCandidate.contextTemplate, renderedOptions, "
                "judgmentEvidence.structurallyWellFormedKeys, and reason. Do not alter "
                "production acceptance until independent language review establishes "
                "sentence error versus verifier error."
            ),
        }
    if baseline.get("fixtureVersion") != candidate.get("fixtureVersion"):
        raise ValueError("baseline and candidate fixture versions differ")
    baseline_runs = {item["profileId"]: item for item in baseline.get("runs", [])}
    candidate_runs = {item["profileId"]: item for item in candidate.get("runs", [])}
    if set(baseline_runs) != set(candidate_runs):
        raise ValueError("baseline and candidate profile sets differ")
    rows = []
    for profile_id in sorted(baseline_runs):
        before, after = baseline_runs[profile_id], candidate_runs[profile_id]
        rows.append(
            {
                "profileId": profile_id,
                "baselineStatus": before["status"],
                "candidateStatus": after["status"],
                "baselineCompleted": before["completedQuestionCount"],
                "candidateCompleted": after["completedQuestionCount"],
                "providerCallDelta": after["calls"]["count"] - before["calls"]["count"],
                "inputTokenDelta": after["calls"]["inputTokens"]
                - before["calls"]["inputTokens"],
                "outputTokenDelta": after["calls"]["outputTokens"]
                - before["calls"]["outputTokens"],
                "firstQuestionMsDelta": (
                    None
                    if before["firstQuestionMs"] is None
                    or after["firstQuestionMs"] is None
                    else round(after["firstQuestionMs"] - before["firstQuestionMs"], 3)
                ),
                "totalMsDelta": round(after["totalMs"] - before["totalMs"], 3),
            }
        )
    return {
        "fixtureVersion": baseline["fixtureVersion"],
        "baselineVariant": baseline.get("variant"),
        "candidateVariant": candidate.get("variant"),
        "allRunsIncluded": True,
        "rows": rows,
        "qualityComparison": (
            "Read questionsForBlindedHumanReview from both inputs. Runtime PASS alone must "
            "not be reported as independent Japanese quality validation."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument(
        "--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"), type=Path
    )
    parser.add_argument(
        "--diagnostic-capture",
        action="store_true",
        help=(
            "Explicitly retain candidate/verifier content in a development QA artifact. "
            "Never use this for production INFO logging."
        ),
    )
    parser.add_argument(
        "--order-diagnostic-case",
        type=Path,
        help=(
            "Exact persisted snapshot plus accepted-prefix JSON. In live mode, mirrors "
            "BE itemRequest and runs only targetOrder; requires --diagnostic-capture."
        ),
    )
    parser.add_argument("--max-provider-calls", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--max-call-seconds",
        type=float,
        help="QA-wrapper timeout for each application-level provider call",
    )
    parser.add_argument("--max-seconds", type=float)
    return parser


def _write_payload(payload: dict[str, Any], output: Path) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    return encoded


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    encoded = (
        _write_payload(payload, output)
        if output is not None
        else json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(encoded, end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.live and args.compare:
        parser.error("--live and --compare are mutually exclusive")
    try:
        if args.compare:
            baseline = json.loads(args.compare[0].read_text(encoding="utf-8"))
            candidate = json.loads(args.compare[1].read_text(encoding="utf-8"))
            _emit(compare_results(baseline, candidate), args.output)
            return 0
        if args.order_diagnostic_case is not None:
            case = load_order_diagnostic_case(args.order_diagnostic_case)
            if not args.live:
                _emit(
                    order_diagnostic_manifest(args.order_diagnostic_case, case),
                    args.output,
                )
                return 0
            if not args.diagnostic_capture:
                raise ValueError(
                    "order diagnostic live mode requires explicit --diagnostic-capture"
                )
            caps = validate_live_args(args)
            result = asyncio.run(
                run_order_diagnostic_live(
                    args.order_diagnostic_case,
                    case,
                    variant=args.variant,
                    caps=caps,
                    partial_output=args.output,
                )
            )
            _emit(result, args.output)
            return 0 if result["status"] == "COMPLETED" else 2
        fixture = load_fixture(args.fixture)
        if not args.live:
            _emit(dry_run_manifest(fixture), args.output)
            return 0
        caps = validate_live_args(args)
        result = asyncio.run(
            run_live(
                fixture,
                variant=args.variant,
                caps=caps,
                diagnostic_capture=args.diagnostic_capture,
            )
        )
        _emit(result, args.output)
        return 0 if all(run["status"] == "COMPLETED" for run in result["runs"]) else 2
    except (OSError, ValueError, QaBudgetExceeded) as exc:
        print(f"QA configuration/run failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
