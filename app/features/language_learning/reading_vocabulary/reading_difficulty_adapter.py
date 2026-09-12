from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass

from app.features.language_learning.difficulty.contracts import (
    DifficultyAcceptanceDecision,
    DifficultyTarget,
    DifficultyValidationResult,
    SemanticAssessment,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_spec import (
    READING_DIFFICULTY_SPEC_VERSION,
    ReadingPassageDifficultySpec,
    ReadingPassageDifficultyTargetValue,
    ReadingQuestionDifficultySpec,
    ReadingQuestionDifficultyTargetValue,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_recipe import (
    passage_difficulty_recipe,
    question_demand_recipe,
)
from app.schemas.language_learning_practice import (
    PracticeGeneratedQuestion,
    PracticeGenerationRequest,
)


_SURFACE_BOUNDARY_RE = re.compile(
    r"[。！？!?]+|(?<!\d)\.(?=\s|$)|\r?\n+"
)


def build_reading_passage_difficulty_spec(
    request: PracticeGenerationRequest,
    *,
    passage_id: str,
    complexity_band: int,
) -> ReadingPassageDifficultySpec:
    return ReadingPassageDifficultySpec(
        target=DifficultyTarget(
            service="reading",
            scale="reading-passage-generation",
            value=ReadingPassageDifficultyTargetValue(
                complexity_band=complexity_band,
                mode=request.mode,
            ),
            policy_version=READING_DIFFICULTY_SPEC_VERSION,
        ),
        passage_id=passage_id,
        recipe=passage_difficulty_recipe(complexity_band, mode=request.mode),
    )


def build_reading_question_difficulty_spec(
    *,
    difficulty: str,
    complexity_band: int,
    skill_tag: str,
    passage_id: str,
) -> ReadingQuestionDifficultySpec:
    return ReadingQuestionDifficultySpec(
        target=DifficultyTarget(
            service="reading",
            scale="reading-question-generation",
            value=ReadingQuestionDifficultyTargetValue(
                difficulty=difficulty,
                complexity_band=complexity_band,
                skill_tag=skill_tag,
            ),
            policy_version=READING_DIFFICULTY_SPEC_VERSION,
        ),
        passage_id=passage_id,
        recipe=question_demand_recipe(complexity_band, skill_tag=skill_tag),
    )


@dataclass(frozen=True)
class ReadingPassageMeasurements:
    normalized_character_count: int
    paragraph_count: int
    surface_unit_count: int

    def log_fields(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReadingQuestionMeasurements:
    """Stable surface observations, not calibrated difficulty decisions.

    ``evidence_to_question_surface_unit_distance`` counts passage surface units
    after the matched evidence, because the learner-visible question follows the
    passage.  V1 records this value but does not map it to a band.
    """

    prompt_character_count: int
    option_count: int
    option_character_lengths: tuple[int, ...]
    evidence_supplied: bool
    evidence_exact_match: bool
    evidence_offset: int | None
    evidence_paragraph: int | None
    evidence_surface_unit: int | None
    evidence_to_question_surface_unit_distance: int | None
    skill_tag: str
    passage_id: str
    passage_binding_matches: bool

    def log_fields(self) -> dict[str, object]:
        return asdict(self)


def normalized_character_count(value: str) -> int:
    return len(unicodedata.normalize("NFC", value).strip())


def measure_reading_passage(passage_text: str) -> ReadingPassageMeasurements:
    text = unicodedata.normalize("NFC", passage_text).strip()
    paragraphs = [part for part in re.split(r"\r?\n\s*\r?\n", text) if part.strip()]
    return ReadingPassageMeasurements(
        normalized_character_count=len(text),
        paragraph_count=len(paragraphs),
        surface_unit_count=len(_surface_unit_spans(text)),
    )


def measure_reading_question(
    question: PracticeGeneratedQuestion,
    *,
    passage_id: str,
    passage_text: str,
) -> ReadingQuestionMeasurements:
    passage = unicodedata.normalize("NFC", passage_text).strip()
    evidence = unicodedata.normalize("NFC", question.evidence_text or "").strip()
    evidence_offset = passage.find(evidence) if evidence else -1
    exact_match = evidence_offset >= 0
    spans = _surface_unit_spans(passage)
    evidence_unit: int | None = None
    evidence_end_unit: int | None = None
    if exact_match:
        for index, (start, end, _) in enumerate(spans, 1):
            if start <= evidence_offset < end:
                evidence_unit = index
            if start < evidence_offset + len(evidence) <= end:
                evidence_end_unit = index
                break
    evidence_paragraph = None
    if exact_match:
        evidence_paragraph = len(
            re.findall(r"\r?\n\s*\r?\n", passage[:evidence_offset])
        ) + 1
    return ReadingQuestionMeasurements(
        prompt_character_count=normalized_character_count(question.prompt),
        option_count=len(question.options),
        option_character_lengths=tuple(
            normalized_character_count(option.text) for option in question.options
        ),
        evidence_supplied=bool(evidence),
        evidence_exact_match=exact_match,
        evidence_offset=evidence_offset if exact_match else None,
        evidence_paragraph=evidence_paragraph,
        evidence_surface_unit=evidence_unit,
        evidence_to_question_surface_unit_distance=(
            len(spans) - evidence_end_unit if evidence_end_unit is not None else None
        ),
        skill_tag=question.skill_tag,
        passage_id=passage_id,
        passage_binding_matches=(
            question.passage_id == passage_id and question.passage_text == passage_text
        ),
    )


def reading_passage_segments(passage_id: str, passage_text: str) -> list[dict[str, str]]:
    text = unicodedata.normalize("NFC", passage_text).strip()
    return [
        {"id": f"passage:{passage_id}:unit:{index}", "text": value}
        for index, (_, _, value) in enumerate(_surface_unit_spans(text), 1)
    ]


def reading_question_segment_ids(question: PracticeGeneratedQuestion) -> set[str]:
    return {
        f"question:{question.order}:prompt",
        *(
            f"question:{question.order}:option:{option.key}"
            for option in question.options
        ),
    }


def _surface_unit_spans(value: str) -> list[tuple[int, int, str]]:
    text = value.strip()
    if not text:
        return []
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in _SURFACE_BOUNDARY_RE.finditer(text):
        segment = text[start : match.end()].strip()
        if segment:
            spans.append((start, match.end(), segment))
        start = match.end()
    tail = text[start:].strip()
    if tail:
        spans.append((start, len(text), tail))
    return spans


def validate_reading_passage(
    spec: ReadingPassageDifficultySpec,
    *,
    observed_passage_id: str,
    passage_text: str,
    language_validator: Callable[[], None],
) -> DifficultyValidationResult:
    if observed_passage_id != spec.passage_id:
        return DifficultyValidationResult.reject("reading passageId mismatch")
    if not passage_text:
        return DifficultyValidationResult.reject("reading passageText is blank")
    try:
        language_validator()
    except ValueError as exc:
        return DifficultyValidationResult.reject(str(exc))
    return DifficultyValidationResult.accept(
        measurements={
            "complexityBand": spec.target.value.complexity_band,
            "passageId": spec.passage_id,
            "recipeVersion": spec.recipe.version,
            **measure_reading_passage(passage_text).log_fields(),
        }
    )


def project_reading_question_validation(
    spec: ReadingQuestionDifficultySpec,
    validator: Callable[[], None],
    *,
    question: PracticeGeneratedQuestion | None = None,
    passage_text: str | None = None,
) -> DifficultyValidationResult:
    try:
        validator()
    except ValueError as exc:
        return DifficultyValidationResult.reject(str(exc))
    measurements: dict[str, object] = {
        "complexityBand": spec.target.value.complexity_band,
        "skillTag": spec.target.value.skill_tag,
        "recipeVersion": spec.recipe.version,
    }
    if question is not None and passage_text is not None:
        measurements.update(
            measure_reading_question(
                question,
                passage_id=spec.passage_id,
                passage_text=passage_text,
            ).log_fields()
        )
    return DifficultyValidationResult.accept(measurements=measurements)


@dataclass(frozen=True)
class ReadingSemanticAssessment(SemanticAssessment[ReadingQuestionDifficultyTargetValue]):
    best_answer_key: str = ""
    ambiguous: bool = False
    supported: bool = True
    mode_fit: bool = True
    answer_leakage: bool = False
    distractors_plausible: bool = True


def normalize_reading_semantic_assessment(
    *,
    best_answer_key: str,
    ambiguous: bool,
    supported: bool,
    mode_fit: bool,
    answer_leakage: bool,
    distractors_plausible: bool,
) -> ReadingSemanticAssessment:
    rejected = (
        ambiguous
        or not supported
        or not mode_fit
        or answer_leakage
        or not distractors_plausible
    )
    return ReadingSemanticAssessment(
        status="REJECT" if rejected else "PASS",
        difficulty_status="NOT_ASSESSED",
        best_answer_key=best_answer_key,
        ambiguous=ambiguous,
        supported=supported,
        mode_fit=mode_fit,
        answer_leakage=answer_leakage,
        distractors_plausible=distractors_plausible,
    )


@dataclass(frozen=True)
class ReadingAcceptanceContext:
    expected_answer_key: str


@dataclass(frozen=True)
class ReadingSemanticQualityPolicy:
    def decide(
        self,
        *,
        assessment: ReadingSemanticAssessment,
        context: ReadingAcceptanceContext,
    ) -> DifficultyAcceptanceDecision:
        reason: str | None = None
        if assessment.ambiguous:
            reason = "ambiguous single-choice item"
        elif not assessment.supported:
            reason = "answer is not sufficiently supported"
        elif not assessment.mode_fit:
            reason = "question does not fit requested mode/skill"
        elif assessment.answer_leakage:
            reason = "semantic verifier detected answer leakage"
        elif not assessment.distractors_plausible:
            reason = "distractors are too weak or unrelated"
        elif assessment.best_answer_key != context.expected_answer_key:
            reason = (
                f"answer mismatch expected={context.expected_answer_key} "
                f"verifier={assessment.best_answer_key}"
            )
        if reason is not None:
            return DifficultyAcceptanceDecision("REJECT", reason)
        return DifficultyAcceptanceDecision("ACCEPT", "semantic verifier accepted candidate")


@dataclass(frozen=True, kw_only=True)
class ReadingSemanticDifficultyAssessment(SemanticAssessment[int]):
    """A blind Reading difficulty classification used for V1 shadow comparison."""


def normalize_reading_semantic_difficulty_assessment(
    raw: object,
    *,
    allowed_evidence_refs: set[str] | None = None,
) -> ReadingSemanticDifficultyAssessment:
    if not isinstance(raw, dict):
        return ReadingSemanticDifficultyAssessment(
            status="UNSURE",
            difficulty_status="NOT_ASSESSED",
        )

    status = str(raw.get("difficultyStatus", "UNSURE")).upper()
    observed = _difficulty_band(raw.get("observedBand"))
    alternative = _difficulty_band(raw.get("alternativeBand"))
    valid = (
        (status == "ASSESSED" and observed is not None and alternative is None)
        or (
            status == "BORDERLINE"
            and observed is not None
            and alternative is not None
            and abs(observed - alternative) == 1
        )
        or (status in {"UNSURE", "NOT_ASSESSED"} and observed is None and alternative is None)
    )
    issue_codes = _opaque_codes(raw.get("issueCodes"))
    if not valid:
        status = "UNSURE"
        observed = alternative = None
        issue_codes = (*issue_codes, "INVALID_DIFFICULTY_ASSESSMENT")
    evidence_refs = _evidence_refs(
        raw.get("evidenceSegmentIds"),
        allowed=allowed_evidence_refs,
    )
    confidence = raw.get("difficultyConfidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        confidence = None
    return ReadingSemanticDifficultyAssessment(
        status="UNSURE" if status in {"UNSURE", "NOT_ASSESSED"} else "PASS",
        difficulty_status=status,
        observed_target=observed,
        alternative_target=alternative,
        issue_codes=tuple(dict.fromkeys(issue_codes)),
        difficulty_confidence=confidence,
        evidence_refs=evidence_refs,
    )


@dataclass(frozen=True)
class ReadingSemanticDifficultyPolicy:
    """Compare blind classifications without granting V1 rejection authority."""

    def decide(
        self,
        *,
        requested_band: int,
        assessment: ReadingSemanticDifficultyAssessment,
    ) -> DifficultyAcceptanceDecision:
        if assessment.difficulty_status == "ASSESSED":
            if assessment.observed_target == requested_band:
                return DifficultyAcceptanceDecision("SHADOW_MATCH", "observed band matches target")
            return DifficultyAcceptanceDecision(
                "SHADOW_MISMATCH",
                "observed band does not match target",
            )
        if assessment.difficulty_status == "BORDERLINE":
            observed = {assessment.observed_target, assessment.alternative_target}
            if requested_band in observed:
                return DifficultyAcceptanceDecision(
                    "SHADOW_BORDERLINE_MATCH",
                    "target is included in adjacent observed bands",
                )
            return DifficultyAcceptanceDecision(
                "SHADOW_MISMATCH",
                "adjacent observed bands do not include target",
            )
        return DifficultyAcceptanceDecision(
            "SHADOW_UNRESOLVED",
            "difficulty was not independently assessed",
        )


def _difficulty_band(value: object) -> int | None:
    if type(value) is int and 1 <= value <= 5:
        return value
    return None


def _opaque_codes(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        item
        for item in value[:12]
        if isinstance(item, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", item)
    )


def _evidence_refs(value: object, *, allowed: set[str] | None) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    refs = tuple(
        item
        for item in value[:20]
        if isinstance(item, str)
        and re.fullmatch(r"(?:passage:[^:]+:unit:\d+|question:\d+:(?:prompt|option:[^:]+))", item)
        and (allowed is None or item in allowed)
    )
    return tuple(dict.fromkeys(refs))
