from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.evaluate_receipt_images import _equal_decimal, _equal_text


def test_receipt_evaluation_runner_is_directly_executable() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_receipt_images.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--max-images" in result.stdout
    assert "--concurrency" in result.stdout


def test_evaluation_comparisons_reject_missing_values_and_compare_decimal_numerically() -> None:
    assert _equal_decimal("12.34", "12.3400")
    assert not _equal_decimal(None, None)
    assert not _equal_decimal("12.34", None)
    assert _equal_text("JPY", "JPY")
    assert not _equal_text(None, None)
