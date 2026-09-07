from __future__ import annotations

from app.schemas.language_learning_quality import LanguageComplexityContext

LANGUAGE_COMPLEXITY_POLICY_VERSION = "language-complexity"
CONTENT_DIVERSITY_POLICY_VERSION = "language-learning-diversity"


def _base_band(context: LanguageComplexityContext | None) -> int:
    if context is None:
        return 3
    return context.base_complexity_band


def resolve_daily_complexity_band(
    difficulty: str,
    context: LanguageComplexityContext | None,
) -> int:
    # Daily Writing can contain REVIEW/NORMAL/CHALLENGE in one batch.
    # Therefore each item must be resolved from the learner's base band rather
    # than applying one request-wide target band to every difficulty bucket.
    base = _base_band(context)
    offsets = {
        "REVIEW": -1,
        "EASY": -1,
        "NORMAL": 0,
        "MY_LEVEL": 0,
        "CHALLENGE": 1,
    }
    return max(1, min(5, base + offsets.get(difficulty, 0)))


def resolve_listening_complexity_band(
    difficulty: str,
    context: LanguageComplexityContext | None,
) -> int:
    # A Listening set has one difficulty, so BE may provide the already-resolved
    # target band explicitly. Fall back to the common base-band mapping when it
    # does not.
    if context is not None and context.target_complexity_band is not None:
        return context.target_complexity_band
    return resolve_daily_complexity_band(difficulty, context)
