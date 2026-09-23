from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.receipt import ReceiptRuntimeIdentity


_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_FILES = (
    "app/api/v1/receipt.py",
    "app/ai/model_policy.py",
    "app/core/config.py",
    "app/main.py",
    "app/features/receipt/currencies.py",
    "app/features/receipt/parser.py",
    "app/features/receipt/prompts.py",
    "app/features/receipt/runtime_identity.py",
    "app/features/receipt/service.py",
    "app/features/receipt/validation.py",
    "app/features/receipt/vision_scheduler.py",
    "app/schemas/receipt.py",
    "app/services/ocr_service.py",
)
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,99}")
_PROVIDER_LOCK = threading.Lock()
_PROVIDER_CALL_COUNT = 0


def receipt_source_fingerprint(root: Path = _ROOT) -> str:
    digest = hashlib.sha256()
    for relative in sorted(_SOURCE_FILES):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    git_marker = root / ".git"
    try:
        git_dir = git_marker
        if git_marker.is_file():
            marker = git_marker.read_text("utf-8").strip()
            if not marker.startswith("gitdir:"):
                return None
            git_dir = (root / marker.split(":", 1)[1].strip()).resolve()
        head = (git_dir / "HEAD").read_text("utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            return (git_dir / ref).read_text("utf-8").strip()
        return head or None
    except OSError:
        return None


_SOURCE_FINGERPRINT = receipt_source_fingerprint()
_EXPECTED_FINGERPRINT = os.environ.get(
    "RECEIPT_RUNTIME_EXPECTED_SOURCE_FINGERPRINT", ""
).strip()
if _EXPECTED_FINGERPRINT and _EXPECTED_FINGERPRINT != _SOURCE_FINGERPRINT:
    raise RuntimeError(
        "Receipt runtime source fingerprint differs from the launcher expectation."
    )

_REQUESTED_RUN_ID = os.environ.get("RECEIPT_RUNTIME_RUN_ID", "").strip()
_RUN_ID = (
    _REQUESTED_RUN_ID
    if _RUN_ID_PATTERN.fullmatch(_REQUESTED_RUN_ID)
    else f"receipt-{uuid.uuid4()}"
)
_STARTED_AT = datetime.now(timezone.utc)
_COMMAND_FINGERPRINT = hashlib.sha256(
    json.dumps([sys.executable, *sys.argv], ensure_ascii=False).encode("utf-8")
).hexdigest()
_GIT_HEAD = os.environ.get("RECEIPT_RUNTIME_GIT_HEAD", "").strip() or _git_head(_ROOT)


def record_receipt_provider_call() -> None:
    global _PROVIDER_CALL_COUNT
    with _PROVIDER_LOCK:
        _PROVIDER_CALL_COUNT += 1


def get_receipt_runtime_identity() -> ReceiptRuntimeIdentity:
    with _PROVIDER_LOCK:
        calls = _PROVIDER_CALL_COUNT
    return ReceiptRuntimeIdentity(
        run_id=_RUN_ID,
        source_fingerprint=_SOURCE_FINGERPRINT,
        started_at=_STARTED_AT,
        process_id=os.getpid(),
        working_directory=str(Path.cwd().resolve()),
        command_fingerprint=_COMMAND_FINGERPRINT,
        git_head=_GIT_HEAD,
        provider_call_count=calls,
    )


if __name__ == "__main__":
    print(_SOURCE_FINGERPRINT)
