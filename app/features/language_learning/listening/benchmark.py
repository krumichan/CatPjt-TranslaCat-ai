from __future__ import annotations

from collections import defaultdict

from app.schemas.language_learning_listening import (
    ListeningBenchmarkResult,
    ListeningBenchmarkSample,
    ListeningBenchmarkSegment,
)


class ListeningEvaluationBenchmarkService:
    def evaluate(
        self,
        samples: list[ListeningBenchmarkSample],
        *,
        threshold_points: float = 10,
        required_agreement_rate: float = 0.70,
    ) -> ListeningBenchmarkResult:
        if not samples:
            raise ValueError("Listening Benchmark Sample이 필요합니다.")

        matched_by_sample: dict[str, bool] = {}
        segmented: dict[str, list[bool]] = defaultdict(list)
        for sample in samples:
            human_average = sum(score.score for score in sample.human_scores) / len(
                sample.human_scores
            )
            matched = abs(sample.ai_score - human_average) <= threshold_points
            matched_by_sample[sample.sample_id] = matched
            segmented[f"language:{sample.language}"].append(matched)
            segmented[f"task:{sample.task_type.value}"].append(matched)
            segmented[
                f"language-task:{sample.language}:{sample.task_type.value}"
            ].append(matched)

        segments = [
            self._segment(name, matches, required_agreement_rate)
            for name, matches in sorted(segmented.items())
        ]
        matched_count = sum(matched_by_sample.values())
        agreement_rate = matched_count / len(samples)
        return ListeningBenchmarkResult(
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
    ) -> ListeningBenchmarkSegment:
        matched_count = sum(matches)
        rate = matched_count / len(matches)
        return ListeningBenchmarkSegment(
            segment=name,
            sample_count=len(matches),
            matched_count=matched_count,
            agreement_rate=round(rate, 4),
            passed=rate >= required_agreement_rate,
        )
