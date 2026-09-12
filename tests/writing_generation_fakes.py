"""Explicit model doubles. Bands are indexed by content, NEVER by server targets.

The fake is not a semantic oracle. Every band is supplied by a test; unknown task
text or an unexpected Nano call fails loudly. No production data is replayed as a
linguistic-quality assertion.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

from app.features.language_learning.writing.generation import GENERATION_TASK
from app.features.language_learning.writing.assessment_contract import ISSUE_CRITERION
from app.features.language_learning.writing.verification import MINI_DIFFICULTY_TASK, MINI_QUALITY_TASK, MINI_NOTE_TASK

NANO_DIFFICULTY_TASK = "LANGUAGE_LEARNING_WRITING_DIFFICULTY_PRESCREEN"  # forbidden route sentinel


def read_review_data(prompt: str) -> dict:
    return json.loads(prompt.split("<writing-review-data>\n", 1)[1].split("\n</writing-review-data>", 1)[0])


def review_text(data: dict, segment_id="O1") -> str:
    return next(segment["text"] for segment in data["segments"] if segment["id"] == segment_id)


def read_generation_data(prompt: str) -> dict:
    return json.loads(prompt.split("<learning-data>\n", 1)[1].split("\n</learning-data>", 1)[0])


def content_only(item: dict) -> dict:
    data = copy.deepcopy(item)
    for key in ("order", "difficulty", "languageComplexityBand", "language_complexity_band", "writingType", "writing_type"):
        data.pop(key, None)
    for key in ("providedFacts", "requiredIntents", "responseConstraints"):
        data.setdefault(key, [])
    return data


def task_review(data: dict, band: int) -> dict:
    return {
        "candidateId": data["candidateId"], "contentHash": data["contentHash"],
        "verdict": "PASS", "observedWritingType": data["writingType"],
        "confidence": 0.73, "difficultyConfidence": 0.7,  # deliberate v3 failure values
        "difficultyStatus": "ASSESSED", "estimatedBand": band, "alternativeBand": None,
        "difficultyEvidenceSegmentIds": ["O1"], "issues": [],
        "checks": [
            {"criterion": name, "status": "PASS", "evidenceSegmentIds": ids}
            for name, ids in (("ORIGIN_LANGUAGE", ["O1"]), ("TASK_VALIDITY", ["O1"]),
                              ("ANSWER_LEAK", ["O1", "N1"]), ("NATURALNESS", ["O1"]),
                              ("NOTE_QUALITY", ["N1"]))
        ],
    }


def reject_issues(*issues: str):
    """Construct a VALID explicit rejection, not a contradictory PASS fixture."""
    def override(value, _data):
        value["verdict"], value["issues"] = "REJECT", list(issues)
        failed = {ISSUE_CRITERION[issue] for issue in issues}
        for check in value["checks"]:
            if check["criterion"] in failed:
                check["status"] = "FAIL"
        return value
    return override


def unsure_review(*, criterion=None, borderline=None):
    def override(value, _data):
        value["verdict"] = "UNSURE"
        if criterion:
            next(check for check in value["checks"] if check["criterion"] == criterion)["status"] = "UNSURE"
        elif borderline:
            value.update(difficultyStatus="BORDERLINE", estimatedBand=borderline[0], alternativeBand=borderline[1])
        else:
            value.update(difficultyStatus="UNSURE", estimatedBand=None, alternativeBand=None,
                         difficultyEvidenceSegmentIds=[])
        return value
    return override


class WritingPipelineProvider:
    def __init__(self, batches: list, bands: dict[str, int], overrides: dict | None = None) -> None:
        self.batches = [value if isinstance(value, BaseException) or callable(value) else copy.deepcopy(value) for value in batches]
        self.bands = bands.copy()
        self.overrides = dict(overrides or {})
        self.calls: list[dict] = []
        self.counts: Counter[str] = Counter()

    async def call(self, type_name: str, data: str, schema: dict | None = None):
        self.calls.append({"type_name": type_name, "data": data, "schema": copy.deepcopy(schema)})
        self.counts[type_name] += 1
        if type_name == GENERATION_TASK:
            if not self.batches:
                raise AssertionError("No explicit generator response configured")
            value = self.batches.pop(0)
            if isinstance(value, BaseException):
                raise value
            if callable(value):
                value = value(data, schema)
            return copy.deepcopy(value)
        review_data = read_review_data(data)
        text = review_text(review_data)
        if text not in self.bands:
            raise AssertionError(f"No explicit independent review fixture for {text!r}")
        if type_name in {MINI_QUALITY_TASK, MINI_DIFFICULTY_TASK}:
            value = task_review(review_data, self.bands[text])
        elif type_name == MINI_NOTE_TASK:
            value = {"candidateId": review_data["candidateId"], "contentHash": review_data["contentHash"],
                     "verdict": "PASS", "confidence": 0.2, "issues": [], "evidenceSegmentIds": ["N1"]}
        else:
            raise AssertionError(f"Unexpected model task: {type_name}")
        override = self.overrides.get(type_name)
        if isinstance(override, list):
            if not override:
                raise AssertionError("Review override queue exhausted")
            override = override.pop(0)
        if isinstance(override, BaseException):
            raise override
        if callable(override):
            return override(value, review_data)
        if override is not None:
            value.update(copy.deepcopy(override))
        return value

    async def call_with_image(self, *args, **kwargs):
        raise AssertionError("Writing never calls vision")
