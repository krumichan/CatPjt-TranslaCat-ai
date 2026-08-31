from __future__ import annotations

from collections import defaultdict

from app.schemas.language_learning_level_test_benchmark import (
    LevelTestBenchmarkResult,
    LevelTestBenchmarkSample,
    LevelTestBenchmarkSegment,
)


class LevelTestEvaluationBenchmarkService:
    def evaluate(
        self,
        samples: list[LevelTestBenchmarkSample],
        *,
        threshold_points: float = 10,
        required_agreement_rate: float = 0.70,
    ) -> LevelTestBenchmarkResult:
        if not samples:
            raise ValueError("Level Test Benchmark Sample이 필요합니다.")

        matched: list[bool] = []
        segmented: dict[str, list[bool]] = defaultdict(list)
        for sample in samples:
            human_average = sum(score.score for score in sample.human_scores) / len(
                sample.human_scores
            )
            is_match = abs(sample.ai_score - human_average) <= threshold_points
            matched.append(is_match)
            segmented[f"domain:{sample.domain.value}"].append(is_match)
            segmented[f"item:{sample.item_type.value}"].append(is_match)
            segmented[f"language:{sample.language}"].append(is_match)

        segments = [
            self._segment(name, values, required_agreement_rate)
            for name, values in sorted(segmented.items())
        ]
        matched_count = sum(matched)
        agreement_rate = matched_count / len(samples)
        return LevelTestBenchmarkResult(
            sample_count=len(samples),
            matched_count=matched_count,
            agreement_rate=round(agreement_rate, 4),
            threshold_points=threshold_points,
            required_agreement_rate=required_agreement_rate,
            segments=segments,
            passed=(
                agreement_rate >= required_agreement_rate
                and all(segment.passed for segment in segments)
            ),
        )

    @staticmethod
    def _segment(
        name: str,
        matches: list[bool],
        required_agreement_rate: float,
    ) -> LevelTestBenchmarkSegment:
        matched_count = sum(matches)
        rate = matched_count / len(matches)
        return LevelTestBenchmarkSegment(
            segment=name,
            sample_count=len(matches),
            matched_count=matched_count,
            agreement_rate=round(rate, 4),
            passed=rate >= required_agreement_rate,
        )
