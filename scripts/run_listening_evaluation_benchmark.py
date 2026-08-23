from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.features.language_learning.listening.benchmark import (  # noqa: E402
    ListeningEvaluationBenchmarkService,
)
from app.schemas.language_learning_listening import (  # noqa: E402
    ListeningBenchmarkSample,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate AI Listening scores against two or more human evaluators "
            "for every language and task segment."
        ),
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--threshold-points", type=float, default=10)
    parser.add_argument("--required-agreement-rate", type=float, default=0.70)
    args = parser.parse_args()

    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    raw_samples = raw["samples"] if isinstance(raw, dict) else raw
    samples = [ListeningBenchmarkSample.model_validate(item) for item in raw_samples]
    result = ListeningEvaluationBenchmarkService().evaluate(
        samples,
        threshold_points=args.threshold_points,
        required_agreement_rate=args.required_agreement_rate,
    )
    print(result.model_dump_json(by_alias=True, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
