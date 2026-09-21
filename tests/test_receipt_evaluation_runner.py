from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
