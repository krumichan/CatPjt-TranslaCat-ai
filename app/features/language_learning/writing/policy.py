from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

EVALUATION_RUBRIC_VERSION = "writing-evaluation-rubric"
SCORING_POLICY_VERSION = "writing-scoring-policy"

SCORING_WEIGHTS: dict[str, Decimal] = {
    "meaning": Decimal("0.30"),
    "grammar": Decimal("0.25"),
    "vocabulary": Decimal("0.15"),
    "naturalness": Decimal("0.20"),
    "expression": Decimal("0.10"),
}


def calculate_overall_score(
    *,
    meaning: float,
    grammar: float,
    vocabulary: float,
    naturalness: float,
    expression: float,
) -> int:
    weighted = (
        Decimal(str(meaning)) * SCORING_WEIGHTS["meaning"]
        + Decimal(str(grammar)) * SCORING_WEIGHTS["grammar"]
        + Decimal(str(vocabulary)) * SCORING_WEIGHTS["vocabulary"]
        + Decimal(str(naturalness)) * SCORING_WEIGHTS["naturalness"]
        + Decimal(str(expression)) * SCORING_WEIGHTS["expression"]
    )
    overall = int(weighted.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    if meaning <= 29:
        return min(overall, 49)
    if meaning <= 49:
        return min(overall, 69)
    return min(overall, 100)
