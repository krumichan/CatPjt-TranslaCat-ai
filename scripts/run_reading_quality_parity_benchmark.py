from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.model_policy import get_task_model_policy  # noqa: E402
from app.ai.providers.openai.client import OpenAIService  # noqa: E402
from app.ai.providers.openai.response import decode_response  # noqa: E402
from app.ai.providers.openai.schema import build_openai_text_config  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.features.language_learning.reading_vocabulary.prompts import (  # noqa: E402
    PRACTICE_VERIFICATION_SYSTEM_PROMPT,
    build_practice_verification_prompt,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_adapter import (  # noqa: E402
    ReadingAcceptanceContext,
    ReadingSemanticQualityPolicy,
    normalize_reading_semantic_assessment,
    normalize_reading_semantic_difficulty_assessment,
    reading_passage_segments,
)
from app.features.language_learning.reading_vocabulary.service import (  # noqa: E402
    ReadingVocabularyGenerationService,
    _PracticeVerificationPayload,
    _READING_PRACTICE_VERIFICATION_SCHEMA,
)
from app.schemas.language_learning_practice import PracticeGenerationRequest  # noqa: E402


DEFAULT_CORPUS = PROJECT_ROOT / "tests" / "fixtures" / "reading-quality-parity-v1.json"
TASK_NAME = ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME
QUALITY_FIELDS = (
    "bestAnswerKey",
    "ambiguous",
    "supported",
    "modeFit",
    "answerLeakage",
    "distractorsPlausible",
)

# Snapshot of the quality-only verifier instructions at repository HEAD immediately
# before Reading Difficulty Calibration V1. Keep this benchmark-only baseline frozen.
QUALITY_ONLY_SYSTEM_PROMPT = r"""
You are an independent semantic quality verifier for TranslaCat Reading and Vocabulary practice.

The generator's expected answer keys and hidden vocabulary targetExpression are deliberately absent. Judge each SINGLE_CHOICE item independently from learner-visible content.
For every supplied question:
- decide the single best option using only supplied passage/context and ordinary language knowledge;
- ambiguous=true if two or more options could reasonably be accepted or wording is underspecified;
- supported=false if there is not enough evidence to answer reliably;
- bestAnswerKey must be one supplied option key;
- distractorsPlausible=false when the wrong options are obviously unrelated or mechanically easy to eliminate;
- modeFit=true only when the question genuinely matches the requested mode/skill;
- answerLeakage=true when the stem reveals the answer or repeats the target in a way that makes selection trivial;
- for USAGE_DISTINCTION, evaluate contextDependent according to usageIntent: COLLOCATION_CHOICE may rely on the local
  phrase/sentence environment, REGISTER_CHOICE on explicit social/register cues, and contextual intents on semantic
  or pragmatic situation cues. Do not require a separate narrative when the supplied local context already decides use;
- actively search for a rival option that could tie the apparent best answer, but do not reject merely because another
  option is grammatical if the requested collocation/register/usage cue makes one option materially better;
- never reconstruct a hidden generator answer key.
ORDERING questions are omitted and structurally validated by the application.
Return exactly one verdict for every supplied item and no extras.
""".strip()

QUALITY_ONLY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdicts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "order": {"type": "INTEGER"},
                    "bestAnswerKey": {"type": "STRING"},
                    "ambiguous": {"type": "BOOLEAN"},
                    "supported": {"type": "BOOLEAN"},
                    "reason": {"type": "STRING"},
                    "modeFit": {"type": "BOOLEAN"},
                    "answerLeakage": {"type": "BOOLEAN"},
                    "contextDependent": {"type": "BOOLEAN"},
                    "distractorsPlausible": {"type": "BOOLEAN"},
                },
                "required": [
                    "order",
                    "bestAnswerKey",
                    "ambiguous",
                    "supported",
                    "reason",
                    "modeFit",
                    "answerLeakage",
                    "contextDependent",
                    "distractorsPlausible",
                ],
            },
        }
    },
    "required": ["verdicts"],
}


class CorpusOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    text: str


class CorpusCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    case_id: str = Field(alias="caseId")
    intended_quality_condition: str = Field(alias="intendedQualityCondition")
    difficulty: str
    skill_tag: str = Field(alias="skillTag")
    prompt: str
    options: list[CorpusOption]
    expected_answer_key: str = Field(alias="expectedAnswerKey")

    @model_validator(mode="after")
    def validate_choice_contract(self) -> "CorpusCandidate":
        keys = [option.key for option in self.options]
        if len(keys) != 4 or len(set(keys)) != 4:
            raise ValueError("each benchmark candidate must have four unique option keys")
        if self.expected_answer_key not in keys:
            raise ValueError("expectedAnswerKey must reference a supplied option")
        if self.difficulty not in {"EASIER", "CURRENT", "CHALLENGE"}:
            raise ValueError("unsupported difficulty label")
        return self


class CorpusBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    batch_id: str = Field(alias="batchId")
    requested_band: int = Field(alias="requestedBand", ge=1, le=5)
    mode: str
    passage_id: str = Field(alias="passageId")
    passage_text: str = Field(alias="passageText")
    candidates: list[CorpusCandidate]


class BenchmarkCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version: str
    origin_language: str = Field(alias="originLanguage")
    learning_language: str = Field(alias="learningLanguage")
    generation_date: date = Field(alias="generationDate")
    batches: list[CorpusBatch]

    @model_validator(mode="after")
    def validate_coverage(self) -> "BenchmarkCorpus":
        bands = {batch.requested_band for batch in self.batches}
        candidates = [candidate for batch in self.batches for candidate in batch.candidates]
        if bands != {1, 2, 3, 4, 5}:
            raise ValueError("corpus must cover requested bands 1 through 5")
        if {candidate.difficulty for candidate in candidates} != {
            "EASIER",
            "CURRENT",
            "CHALLENGE",
        }:
            raise ValueError("corpus must cover all relative difficulty labels")
        required_skills = {"DETAIL", "GIST", "INFERENCE"}
        if not required_skills.issubset({candidate.skill_tag for candidate in candidates}):
            raise ValueError("corpus must cover DETAIL, GIST and INFERENCE")
        required_conditions = {
            "GOOD",
            "AMBIGUOUS",
            "WEAK_DISTRACTORS",
            "UNSUPPORTED_ANSWER",
            "ANSWER_LEAKAGE",
            "ANSWER_KEY_MISMATCH",
        }
        if not required_conditions.issubset(
            {candidate.intended_quality_condition for candidate in candidates}
        ):
            raise ValueError("corpus is missing a required quality condition")
        return self


def load_corpus(path: Path) -> BenchmarkCorpus:
    return BenchmarkCorpus.model_validate_json(path.read_text(encoding="utf-8"))


def quality_only_prompt(
    corpus: BenchmarkCorpus,
    batch: CorpusBatch,
    questions: list[dict[str, Any]],
) -> str:
    payload = {
        "domain": "READING",
        "mode": batch.mode,
        "originLanguage": corpus.origin_language,
        "learningLanguage": corpus.learning_language,
        "questions": questions,
    }
    return (
        "Independently verify semantic uniqueness/support. Expected answer keys are not included.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_batch_inputs(
    corpus: BenchmarkCorpus,
    batch: CorpusBatch,
) -> tuple[PracticeGenerationRequest, list[dict[str, Any]], list[dict[str, Any]]]:
    passage_segments = [
        {"id": segment["id"], "ordinal": index}
        for index, segment in enumerate(
            reading_passage_segments(batch.passage_id, batch.passage_text),
            1,
        )
    ]
    quality_questions: list[dict[str, Any]] = []
    combined_questions: list[dict[str, Any]] = []
    for order, candidate in enumerate(batch.candidates, 1):
        base = {
            "order": order,
            "passageText": batch.passage_text,
            "prompt": candidate.prompt,
            "options": [option.model_dump() for option in candidate.options],
            "skillTag": candidate.skill_tag,
        }
        quality_questions.append(base)
        combined_questions.append(
            {
                **base,
                "passageId": batch.passage_id,
                "passageSegments": passage_segments,
                "questionSegments": [
                    {"id": f"question:{order}:prompt", "field": "prompt"},
                    *[
                        {
                            "id": f"question:{order}:option:{option.key}",
                            "field": "option",
                            "optionKey": option.key,
                        }
                        for option in candidate.options
                    ],
                ],
            }
        )

    request = PracticeGenerationRequest.model_validate(
        {
            "requestId": f"offline-quality-parity-{batch.batch_id}",
            "domain": "READING",
            "mode": batch.mode,
            "originLanguage": corpus.origin_language,
            "learningLanguage": corpus.learning_language,
            # The public request contract permits Reading sets of 1 or 5. The verifier
            # prompt only consumes mode/language fields, so use a valid five-item shell
            # while the offline benchmark intentionally sends three candidates per band.
            "questionCount": 5,
            "complexityBand": batch.requested_band,
            "easierCount": 1,
            "currentCount": 3,
            "challengeCount": 1,
            "generationDate": corpus.generation_date.isoformat(),
        }
    )
    return request, quality_questions, combined_questions


async def call_mini(
    service: OpenAIService,
    *,
    instructions: str,
    prompt: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    policy = get_task_model_policy(TASK_NAME)
    started = time.perf_counter()
    create_response = cast(Any, service.client.responses.create)
    response = await create_response(
        model=settings.OPENAI_MODEL_MINI,
        instructions=instructions,
        input=prompt,
        reasoning={"effort": policy.reasoning_effort},
        max_output_tokens=policy.max_output_tokens,
        text=build_openai_text_config(
            type_name=TASK_NAME,
            schema=schema,
            verbosity=policy.verbosity,
        ),
        store=False,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    usage = getattr(response, "usage", None)
    return {
        "data": decode_response(response, structured=True),
        "latencyMs": latency_ms,
        "inputTokens": int(getattr(usage, "input_tokens", 0) or 0),
        "outputTokens": int(getattr(usage, "output_tokens", 0) or 0),
        "model": str(getattr(response, "model", None) or settings.OPENAI_MODEL_MINI),
    }


def quality_record(verdict: Any, expected_answer_key: str) -> dict[str, Any]:
    assessment = normalize_reading_semantic_assessment(
        best_answer_key=verdict.bestAnswerKey,
        ambiguous=verdict.ambiguous,
        supported=verdict.supported,
        mode_fit=verdict.modeFit,
        answer_leakage=verdict.answerLeakage,
        distractors_plausible=verdict.distractorsPlausible,
    )
    decision = ReadingSemanticQualityPolicy().decide(
        assessment=assessment,
        context=ReadingAcceptanceContext(expected_answer_key=expected_answer_key),
    )
    return {
        "bestAnswerKey": verdict.bestAnswerKey,
        "ambiguous": verdict.ambiguous,
        "supported": verdict.supported,
        "modeFit": verdict.modeFit,
        "answerLeakage": verdict.answerLeakage,
        "distractorsPlausible": verdict.distractorsPlausible,
        "qualityAction": decision.action,
    }


def difficulty_record(raw: object) -> dict[str, Any]:
    assessment = normalize_reading_semantic_difficulty_assessment(raw)
    return {
        "difficultyStatus": assessment.difficulty_status,
        "observedBand": assessment.observed_target,
        "alternativeBand": assessment.alternative_target,
        "difficultyConfidence": assessment.difficulty_confidence,
    }


def _verdicts_by_order(raw: object, expected_orders: set[int]) -> dict[int, Any]:
    payload = _PracticeVerificationPayload.model_validate(raw)
    result = {verdict.order: verdict for verdict in payload.verdicts}
    if set(result) != expected_orders:
        raise ValueError("benchmark verifier verdict coverage mismatch")
    return result


def _passage_difficulty(raw: object, passage_id: str) -> dict[str, Any]:
    payload = _PracticeVerificationPayload.model_validate(raw)
    assessments = payload.passageDifficultyAssessments
    if isinstance(assessments, list):
        for item in assessments:
            if isinstance(item, dict) and item.get("passageId") == passage_id:
                return difficulty_record(item)
    return difficulty_record(None)


def _observed_bucket(record: dict[str, Any]) -> str:
    status = record["difficultyStatus"]
    observed = record["observedBand"]
    alternative = record["alternativeBand"]
    if status == "BORDERLINE" and observed is not None and alternative is not None:
        return f"{observed}|{alternative}"
    if status == "ASSESSED" and observed is not None:
        return str(observed)
    return status


def _condition_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["latencyMs"]) for item in values]
    input_tokens = [int(item["inputTokens"]) for item in values]
    output_tokens = [int(item["outputTokens"]) for item in values]
    return {
        "callCount": len(values),
        "latencyMs": {
            "mean": round(statistics.fmean(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "inputTokens": {
            "total": sum(input_tokens),
            "meanPerCall": round(statistics.fmean(input_tokens), 3),
        },
        "outputTokens": {
            "total": sum(output_tokens),
            "meanPerCall": round(statistics.fmean(output_tokens), 3),
        },
    }


def summarize(
    comparisons: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    passage_observations: list[dict[str, Any]],
) -> dict[str, Any]:
    agreement_fields = (*QUALITY_FIELDS, "qualityAction")
    rates = {
        field: round(
            sum(item["agreement"][field] for item in comparisons) / len(comparisons),
            4,
        )
        for field in agreement_fields
    }
    action_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    by_condition: dict[str, list[bool]] = defaultdict(list)
    question_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    difficulty_statuses: Counter[str] = Counter()
    confidences: list[float] = []
    for item in comparisons:
        quality_only = item["qualityOnly"]["qualityAction"]
        combined = item["combined"]["qualityAction"]
        action_confusion[quality_only][combined] += 1
        by_condition[item["intendedQualityCondition"]].append(
            item["agreement"]["qualityAction"]
        )
        difficulty = item["combinedDifficulty"]
        question_matrix[str(item["requestedBand"])][
            _observed_bucket(difficulty)
        ] += 1
        difficulty_statuses[difficulty["difficultyStatus"]] += 1
        if difficulty["difficultyConfidence"] is not None:
            confidences.append(float(difficulty["difficultyConfidence"]))

    passage_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for item in passage_observations:
        passage_matrix[str(item["requestedBand"])][
            _observed_bucket(item["difficulty"])
        ] += 1

    minimum_field_agreement = min(rates[field] for field in QUALITY_FIELDS)
    action_agreement = rates["qualityAction"]
    if action_agreement >= 0.90 and minimum_field_agreement >= 0.90:
        verdict = "QUALITY-PARITY-PASS"
    elif action_agreement >= 0.80 and minimum_field_agreement >= 0.80:
        verdict = "QUALITY-PARITY-WITH-CONDITIONS"
    else:
        verdict = "QUALITY-PARITY-FAIL"

    calls_by_condition = {
        condition: _condition_summary(
            [item for item in calls if item["condition"] == condition]
        )
        for condition in ("QUALITY_ONLY", "QUALITY_AND_DIFFICULTY")
    }
    return {
        "comparisonCount": len(comparisons),
        "agreementRates": rates,
        "qualityActionConfusion": {
            source: dict(targets) for source, targets in action_confusion.items()
        },
        "qualityActionAgreementByIntendedCondition": {
            condition: round(sum(values) / len(values), 4)
            for condition, values in sorted(by_condition.items())
        },
        "calls": calls_by_condition,
        "combinedDifficultyObservations": {
            "goldLabelDisclaimer": (
                "Model classifications are observations only and are not difficulty gold labels."
            ),
            "questionStatusCounts": dict(difficulty_statuses),
            "questionConfidenceMean": (
                round(statistics.fmean(confidences), 4) if confidences else None
            ),
            "questionRequestedVsObserved": {
                requested: dict(observed)
                for requested, observed in sorted(question_matrix.items())
            },
            "passageRequestedVsObserved": {
                requested: dict(observed)
                for requested, observed in sorted(passage_matrix.items())
            },
        },
        "thresholds": {
            "pass": {
                "qualityActionAgreementAtLeast": 0.90,
                "everyQualityFieldAgreementAtLeast": 0.90,
            },
            "conditional": {
                "qualityActionAgreementAtLeast": 0.80,
                "everyQualityFieldAgreementAtLeast": 0.80,
            },
        },
        "verdict": verdict,
    }


async def run_benchmark(corpus: BenchmarkCorpus, *, rounds: int) -> dict[str, Any]:
    if not settings.OPENAI_API_KEY.strip():
        raise RuntimeError("OPENAI_API_KEY is required for the live offline benchmark")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    service = OpenAIService()
    calls: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    passage_observations: list[dict[str, Any]] = []
    try:
        for round_number in range(1, rounds + 1):
            for batch in corpus.batches:
                request, quality_questions, combined_questions = build_batch_inputs(
                    corpus,
                    batch,
                )
                prompts = {
                    "QUALITY_ONLY": quality_only_prompt(
                        corpus,
                        batch,
                        quality_questions,
                    ),
                    "QUALITY_AND_DIFFICULTY": build_practice_verification_prompt(
                        request,
                        combined_questions,
                    ),
                }
                instructions = {
                    "QUALITY_ONLY": QUALITY_ONLY_SYSTEM_PROMPT,
                    "QUALITY_AND_DIFFICULTY": PRACTICE_VERIFICATION_SYSTEM_PROMPT,
                }
                schemas = {
                    "QUALITY_ONLY": QUALITY_ONLY_SCHEMA,
                    "QUALITY_AND_DIFFICULTY": _READING_PRACTICE_VERIFICATION_SCHEMA,
                }
                results: dict[str, dict[str, Any]] = {}
                for condition in ("QUALITY_ONLY", "QUALITY_AND_DIFFICULTY"):
                    print(
                        f"round={round_number}/{rounds} batch={batch.batch_id} "
                        f"condition={condition}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result = await call_mini(
                        service,
                        instructions=instructions[condition],
                        prompt=prompts[condition],
                        schema=schemas[condition],
                    )
                    results[condition] = result
                    calls.append(
                        {
                            "round": round_number,
                            "batchId": batch.batch_id,
                            "condition": condition,
                            "latencyMs": result["latencyMs"],
                            "inputTokens": result["inputTokens"],
                            "outputTokens": result["outputTokens"],
                        }
                    )

                expected_orders = set(range(1, len(batch.candidates) + 1))
                quality_only = _verdicts_by_order(
                    results["QUALITY_ONLY"]["data"],
                    expected_orders,
                )
                combined = _verdicts_by_order(
                    results["QUALITY_AND_DIFFICULTY"]["data"],
                    expected_orders,
                )
                passage_observations.append(
                    {
                        "round": round_number,
                        "batchId": batch.batch_id,
                        "requestedBand": batch.requested_band,
                        "difficulty": _passage_difficulty(
                            results["QUALITY_AND_DIFFICULTY"]["data"],
                            batch.passage_id,
                        ),
                    }
                )
                for order, candidate in enumerate(batch.candidates, 1):
                    quality_only_record = quality_record(
                        quality_only[order],
                        candidate.expected_answer_key,
                    )
                    combined_record = quality_record(
                        combined[order],
                        candidate.expected_answer_key,
                    )
                    comparisons.append(
                        {
                            "round": round_number,
                            "batchId": batch.batch_id,
                            "caseId": candidate.case_id,
                            "intendedQualityCondition": (
                                candidate.intended_quality_condition
                            ),
                            "requestedBand": batch.requested_band,
                            "relativeDifficulty": candidate.difficulty,
                            "skillTag": candidate.skill_tag,
                            "qualityOnly": quality_only_record,
                            "combined": combined_record,
                            "agreement": {
                                field: quality_only_record[field]
                                == combined_record[field]
                                for field in (*QUALITY_FIELDS, "qualityAction")
                            },
                            "combinedDifficulty": difficulty_record(
                                combined[order].difficulty
                            ),
                        }
                    )
    finally:
        await service.shutdown()

    summary = summarize(comparisons, calls, passage_observations)
    return {
        "benchmarkVersion": "reading-quality-parity-v1",
        "corpusVersion": corpus.version,
        "generatedAt": datetime.now(UTC).isoformat(),
        "provider": "openai",
        "model": settings.OPENAI_MODEL_MINI,
        "rounds": rounds,
        "candidateCount": sum(len(batch.candidates) for batch in corpus.batches),
        "purpose": (
            "Measure whether adding the Reading difficulty task perturbs the existing "
            "semantic-quality gate. Difficulty observations are never acceptance labels."
        ),
        "summary": summary,
        "calls": calls,
        "passageDifficultyObservations": passage_observations,
        "comparisons": comparisons,
        "verdict": summary["verdict"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the live Mini Reading quality-only versus quality+difficulty offline "
            "parity benchmark. This does not call the production generation service."
        )
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is not None and args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )
    result = asyncio.run(run_benchmark(load_corpus(args.corpus), rounds=args.rounds))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"result={args.output}")
        print(f"verdict={result['verdict']}")
    return 1 if result["verdict"] == "QUALITY-PARITY-FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
