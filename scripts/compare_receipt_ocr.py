"""Bounded receipt OCR comparison without changing the application provider.

Ground truth is read only after OCR completes and is never included in a provider
request. Google Cloud Vision execution requires both --execute and
--allow-paid-calls so a dry run cannot create billable traffic accidentally.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import unicodedata
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.services.ocr_service import OCRService


MAX_COMPARISON_CALLS = 12
GOOGLE_VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+", "", normalized)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text("utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_COMPARISON_CALLS:
        raise ValueError(f"manifest must contain 1..{MAX_COMPARISON_CALLS} samples")
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("file"), str):
            raise ValueError("every sample needs a file")
        source = (path.parent / sample["file"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        sample["resolvedFile"] = str(source)
    return samples


async def _paddle_text(source: Path) -> str:
    content_type = "image/png" if source.suffix.lower() == ".png" else "image/jpeg"
    with source.open("rb") as stream:
        upload = UploadFile(
            filename=source.name,
            file=stream,
            headers=Headers({"content-type": content_type}),
        )
        return await OCRService().extract_text_from_upload(upload, "japan")


def _google_text(source: Path, api_key: str) -> str:
    request_body = {
        "requests": [{
            "image": {"content": base64.b64encode(source.read_bytes()).decode("ascii")},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }],
    }
    request = urllib.request.Request(
        GOOGLE_VISION_ENDPOINT,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        payload = json.load(response)
    first = (payload.get("responses") or [{}])[0]
    if first.get("error"):
        raise RuntimeError(str(first["error"]))
    return str((first.get("fullTextAnnotation") or {}).get("text") or "")


def _score(sample: dict[str, Any], text: str) -> dict[str, Any]:
    observed = _normalize(text)
    tokens = [str(token) for token in sample.get("expectedTokens", [])]
    matches = [{"token": token, "matched": _normalize(token) in observed} for token in tokens]
    return {
        "id": sample.get("id"),
        "role": sample.get("role"),
        "file": sample["resolvedFile"],
        "expectedTokenCount": len(tokens),
        "matchedTokenCount": sum(1 for item in matches if item["matched"]),
        "matches": matches,
        "text": text,
    }


async def _run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    samples = _load_manifest(manifest_path)
    result: dict[str, Any] = {
        "createdAt": datetime.now(UTC).isoformat(),
        "provider": args.provider,
        "sampleCount": len(samples),
        "maximumCalls": MAX_COMPARISON_CALLS,
        "calls": 0,
        "estimatedUsd": 0,
        "status": "PREPARED_NOT_RUN",
        "results": [],
    }
    if not args.execute:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
        return 0
    if args.provider == "google-vision" and not args.allow_paid_calls:
        result["status"] = "PAID_CALL_AUTHORIZATION_REQUIRED"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
        return 2
    api_key = os.getenv("GOOGLE_CLOUD_VISION_API_KEY", "")
    if args.provider == "google-vision" and not api_key:
        result["status"] = "CREDENTIALS_MISSING"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
        return 2
    for sample in samples:
        source = Path(sample["resolvedFile"])
        text = (
            await _paddle_text(source)
            if args.provider == "paddle"
            else await asyncio.to_thread(_google_text, source, api_key)
        )
        result["calls"] += 1
        result["results"].append(_score(sample, text))
    result["status"] = "COMPLETED"
    # First 1,000 Cloud Vision units/month are currently listed as free, but the
    # run reports the conservative post-free-tier unit price for transparency.
    if args.provider == "google-vision":
        result["estimatedUsd"] = round(result["calls"] * 1.50 / 1000, 6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=("paddle", "google-vision"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-paid-calls", action="store_true")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
