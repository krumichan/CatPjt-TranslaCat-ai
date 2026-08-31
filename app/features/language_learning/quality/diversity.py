from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from app.schemas.language_learning_quality import (
    DiversityContext,
    DiversityHistoryEntry,
    DiversityMetadata,
)


@dataclass(frozen=True)
class DiversityCandidate:
    content: str
    metadata: DiversityMetadata


@dataclass(frozen=True)
class DiversityDecision:
    accepted: bool
    reason: str | None
    metadata: DiversityMetadata


@dataclass
class DiversityValidationStats:
    candidate_count: int = 0
    accepted_count: int = 0
    rejected_exact: int = 0
    rejected_similarity: int = 0
    rejected_structural: int = 0
    rejected_background_knowledge: int = 0

    def record(self, decision: DiversityDecision) -> None:
        self.candidate_count += 1
        if decision.accepted:
            self.accepted_count += 1
            return
        if decision.reason == "EXACT":
            self.rejected_exact += 1
        elif decision.reason == "SIMILARITY":
            self.rejected_similarity += 1
        elif decision.reason == "STRUCTURAL":
            self.rejected_structural += 1
        elif decision.reason == "BACKGROUND_KNOWLEDGE":
            self.rejected_background_knowledge += 1


_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_generation_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _PUNCTUATION_RE.sub(" ", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _character_ngrams(text: str, n: int = 3) -> Counter[str]:
    compact = normalize_generation_text(text).replace(" ", "")
    if not compact:
        return Counter()
    if len(compact) < n:
        return Counter({compact: 1})
    return Counter(compact[index : index + n] for index in range(len(compact) - n + 1))


def character_ngram_cosine(a: str, b: str) -> float:
    left = _character_ngrams(a)
    right = _character_ngrams(b)
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _hash(text: str) -> str:
    return hashlib.sha256(normalize_generation_text(text).encode("utf-8")).hexdigest()


def _similarity_key(content: str, summary: str) -> str:
    material = f"{normalize_generation_text(summary)}|{normalize_generation_text(content)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _signature(metadata: DiversityMetadata) -> tuple[str, str, str]:
    return (
        metadata.scenario_category.value,
        metadata.communicative_intent.value,
        metadata.task_archetype.strip().upper(),
    )


def _entry_signature(entry: DiversityHistoryEntry) -> tuple[str, str, str] | None:
    if (
        entry.scenario_category is None
        or entry.communicative_intent is None
        or not entry.task_archetype
    ):
        return None
    return (
        entry.scenario_category.value,
        entry.communicative_intent.value,
        entry.task_archetype.strip().upper(),
    )


class DiversityValidator:
    SAME_BATCH_THRESHOLD = 0.82
    SAME_FEATURE_THRESHOLD = 0.88
    CROSS_FEATURE_THRESHOLD = 0.92
    ABSOLUTE_MAX_THRESHOLD = 0.95

    def __init__(
        self,
        context: DiversityContext | None = None,
        *,
        relaxed_history: bool = False,
    ) -> None:
        self.context = context or DiversityContext()
        self.same_feature_threshold = self.SAME_FEATURE_THRESHOLD + (
            0.03 if relaxed_history else 0.0
        )
        self.cross_feature_threshold = min(
            self.ABSOLUTE_MAX_THRESHOLD,
            self.CROSS_FEATURE_THRESHOLD + (0.03 if relaxed_history else 0.0),
        )
        self._exact_hashes = set(self.context.exact_content_hashes_90d)
        for entry in (
            self.context.current_session
            + self.context.same_feature_recent
            + self.context.cross_feature_recent
        ):
            if entry.content_hash:
                self._exact_hashes.add(entry.content_hash)
            else:
                self._exact_hashes.add(_hash(entry.content))

    def validate(
        self,
        candidate: DiversityCandidate,
        accepted: list[DiversityCandidate],
    ) -> DiversityDecision:
        content_hash = _hash(candidate.content)
        finalized_metadata = candidate.metadata.model_copy(
            update={
                "content_hash": content_hash,
                "similarity_key": _similarity_key(
                    candidate.content,
                    candidate.metadata.semantic_summary,
                ),
            }
        )
        candidate = DiversityCandidate(candidate.content, finalized_metadata)

        if finalized_metadata.requires_background_knowledge:
            return DiversityDecision(False, "BACKGROUND_KNOWLEDGE", finalized_metadata)

        if content_hash in self._exact_hashes:
            return DiversityDecision(False, "EXACT", finalized_metadata)

        for current in accepted:
            if _signature(current.metadata) == _signature(finalized_metadata):
                return DiversityDecision(False, "STRUCTURAL", finalized_metadata)
            if (
                character_ngram_cosine(candidate.content, current.content)
                >= self.SAME_BATCH_THRESHOLD
            ):
                return DiversityDecision(False, "SIMILARITY", finalized_metadata)

        for entry in self.context.current_session:
            signature = _entry_signature(entry)
            if signature is not None and signature == _signature(finalized_metadata):
                return DiversityDecision(False, "STRUCTURAL", finalized_metadata)
            if character_ngram_cosine(candidate.content, entry.content) >= self.SAME_BATCH_THRESHOLD:
                return DiversityDecision(False, "SIMILARITY", finalized_metadata)

        for entry in self.context.same_feature_recent:
            if character_ngram_cosine(candidate.content, entry.content) >= self.same_feature_threshold:
                return DiversityDecision(False, "SIMILARITY", finalized_metadata)
            signature = _entry_signature(entry)
            if (
                entry.age_days is not None
                and entry.age_days <= 7
                and signature is not None
                and signature == _signature(finalized_metadata)
                and set(entry.grammar_focus_codes) & set(finalized_metadata.grammar_focus_codes)
            ):
                return DiversityDecision(False, "STRUCTURAL", finalized_metadata)

        for entry in self.context.cross_feature_recent:
            if character_ngram_cosine(candidate.content, entry.content) >= self.cross_feature_threshold:
                return DiversityDecision(False, "SIMILARITY", finalized_metadata)

        return DiversityDecision(True, None, finalized_metadata)
