from __future__ import annotations

from statistics import mean

from app.schemas.language_learning_speaking import BenchmarkResult, BenchmarkSample


class SpeakingEvaluationBenchmarkService:
    def evaluate(
        self,
        samples: list[BenchmarkSample],
        *,
        tolerance_points: float = 10.0,
        required_agreement_rate: float = 0.70,
    ) -> BenchmarkResult:
        total = 0
        matched = 0

        for sample in samples:
            ai_scores = {item.metric_type: item.score for item in sample.ai_scores}
            human_maps = [
                {item.metric_type: item.score for item in evaluator.scores}
                for evaluator in sample.human_evaluators
            ]
            metric_types = set(ai_scores)
            if any(set(scores) != metric_types for scores in human_maps):
                raise ValueError(
                    f"Human evaluator metric set이 일치하지 않습니다: {sample.sample_id}"
                )

            for metric_type, ai_score in ai_scores.items():
                human_average = mean(scores[metric_type] for scores in human_maps)
                total += 1
                if abs(ai_score - human_average) <= tolerance_points:
                    matched += 1

        agreement = matched / total if total else 0.0
        return BenchmarkResult(
            total_metrics=total,
            matched_metrics=matched,
            agreement_rate=round(agreement, 4),
            threshold_points=tolerance_points,
            required_agreement_rate=required_agreement_rate,
            passed=agreement >= required_agreement_rate,
        )
