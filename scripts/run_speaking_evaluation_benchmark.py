from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.features.language_learning.speaking.benchmark import (
    SpeakingEvaluationBenchmarkService,
)
from app.schemas.language_learning_speaking import BenchmarkSample


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare AI speaking scores against 2+ human evaluators.",
    )
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    samples = [BenchmarkSample.model_validate(item) for item in raw]
    result = SpeakingEvaluationBenchmarkService().evaluate(samples)
    print(result.model_dump_json(by_alias=True, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
