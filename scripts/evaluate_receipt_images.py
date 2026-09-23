"""Run a bounded live vision evaluation against an independently reviewed manifest.

The runner deliberately calls the configured runtime provider/model directly with the
production receipt prompt and schema so evaluation output can retain bounding boxes for
physical one-to-one matching. It never reads or prints API keys.
"""

from __future__ import annotations

# ruff: noqa: E402 -- allow this checked-in script to run directly from scripts/.

import argparse
import asyncio
import hashlib
import json
import mimetypes
import sys
import time
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.model_policy import get_model_name_for_task
from app.api.dependencies import get_ai_provider
from app.core.config import settings
from app.features.receipt.prompts import build_receipt_vision_prompt
from app.features.receipt.service import _RECEIPT_ANALYSIS_SCHEMA
from app.features.receipt.validation import validate_response
from app.schemas.receipt import ReceiptAnalysisMode, ReceiptAnalysisOptions


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("core", "holdout", "all"), default="all")
    parser.add_argument("--max-images", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.max_images <= 60:
        parser.error("--max-images must be between 1 and 60")
    if not 1 <= args.concurrency <= 2:
        parser.error("--concurrency must be 1 or 2")
    return args


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        result = tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return None
    left, top, right, bottom = result
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        return None
    return left, top, right, bottom


def _iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _canonical(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (Decimal, date)):
        return str(value)
    return str(value).strip()


def _equal_text(expected: Any, actual: Any) -> bool:
    """Missing values never count as a successful extraction."""
    return expected is not None and actual is not None and _canonical(expected) == _canonical(actual)


def _equal_decimal(expected: Any, actual: Any) -> bool:
    """Compare monetary values numerically while rejecting null==null."""
    if expected is None or actual is None:
        return False
    try:
        return Decimal(str(expected)) == Decimal(str(actual))
    except Exception:
        return False


def _match(
    truth: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    pairs: list[tuple[float, int, int]] = []
    for truth_index, expected in enumerate(truth):
        expected_box = _box(expected.get("boundingBox"))
        for prediction_index, actual in enumerate(predictions):
            actual_box = _box(actual.get("boundingBox"))
            if expected_box and actual_box:
                score = _iou(expected_box, actual_box)
            else:
                compared = (
                    ("storeName", "store_name"),
                    ("bookAmount", "book_amount"),
                    ("currency", "detected_currency_code"),
                    ("transactionDate", "transaction_date"),
                )
                scored = [
                    _equal_text(expected.get(expected_key), actual.get(actual_key))
                    for expected_key, actual_key in compared
                    if expected.get(expected_key) is not None
                ]
                score = sum(scored) / len(scored) if scored else 0
            if score > 0:
                pairs.append((score, truth_index, prediction_index))
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, truth_index, prediction_index in sorted(pairs, reverse=True):
        if truth_index in matched_truth or prediction_index in matched_predictions:
            continue
        if score < 0.10:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, round(score, 6)))
    return (
        matches,
        sorted(set(range(len(truth))) - matched_truth),
        sorted(set(range(len(predictions))) - matched_predictions),
    )


def _validate_manifest(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Manifest must contain at least one image")
    for image in images:
        required = {
            "imageId", "path", "sha256", "classification", "split", "source",
            "consentApproved", "deidentified", "independentlyReviewed", "receipts",
        }
        missing = sorted(required - set(image))
        if missing:
            raise ValueError(f"{image.get('imageId', '<unknown>')}: missing {missing}")
        if image["classification"] == "real" and not (
            image["consentApproved"]
            and image["deidentified"]
            and image["independentlyReviewed"]
        ):
            raise ValueError(f"{image['imageId']}: real image is not approved/reviewed")
        path = (root / image["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"{image['imageId']}: image file does not exist")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest.lower() != str(image["sha256"]).lower():
            raise ValueError(f"{image['imageId']}: SHA-256 mismatch")
        receipts = image["receipts"]
        if not isinstance(receipts, list):
            raise ValueError(f"{image['imageId']}: receipts must be an array")
        if len(receipts) != image.get("expectedReceiptCount"):
            raise ValueError(f"{image['imageId']}: expectedReceiptCount mismatch")
        if len(receipts) > 1 and any(_box(item.get("boundingBox")) is None for item in receipts):
            raise ValueError(f"{image['imageId']}: multi-receipt truth needs bounding boxes")
        image["_resolvedPath"] = path
    return images


async def _evaluate(image: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    path: Path = image["_resolvedPath"]
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    options = ReceiptAnalysisOptions(analysis_mode=ReceiptAnalysisMode.VISION_ONLY)
    provider = get_ai_provider()
    started = time.perf_counter()
    async with semaphore:
        raw = await provider.call_with_image(
            type_name="RECEIPT_ANALYSIS",
            prompt=build_receipt_vision_prompt(options),
            image_bytes=path.read_bytes(),
            mime_type=mime_type,
            schema=_RECEIPT_ANALYSIS_SCHEMA,
        )
    latency_ms = round((time.perf_counter() - started) * 1000)
    validated = validate_response(raw, options, ocr_engine="vision", used_ai=True)
    raw_items = raw.get("receipts", []) if isinstance(raw, dict) else []
    predictions: list[dict[str, Any]] = []
    validated_by_id = {item.receipt_id: item for item in validated.receipts}
    for raw_index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            continue
        item = validated_by_id.get(f"receipt-{raw_index + 1}")
        if item is None:
            continue
        payload = item.model_dump(mode="json")
        payload["boundingBox"] = raw_item.get("bounding_box")
        predictions.append(payload)

    truth = image["receipts"]
    matches, missing, extras = _match(truth, predictions)
    comparisons: list[dict[str, Any]] = []
    false_ready = 0
    for truth_index, prediction_index, score in matches:
        expected = truth[truth_index]
        actual = predictions[prediction_index]
        expected_purchase_total = (
            expected["purchaseTotal"]
            if "purchaseTotal" in expected
            else expected.get("originalAmount")
        )
        expected_book_amount = (
            expected["bookAmount"]
            if "bookAmount" in expected
            else expected.get("originalAmount")
        )
        fields = {
            "purchaseTotal": _equal_decimal(
                expected_purchase_total, actual.get("purchase_total")
            ),
            "bookAmount": _equal_decimal(
                expected_book_amount, actual.get("book_amount")
            ),
            "currency": _equal_text(
                expected.get("currency"), actual.get("detected_currency_code")
            ),
            "date": _equal_text(
                expected.get("transactionDate"), actual.get("transaction_date")
            ),
        }
        critical_ok = all(fields.values())
        if actual.get("status") == "READY" and not critical_ok:
            false_ready += 1
        comparisons.append(
            {
                "truthId": expected.get("truthId"),
                "receiptId": actual.get("receipt_id"),
                "matchScore": score,
                "fieldMatches": fields,
                "jointCriticalMatch": critical_ok,
                "status": actual.get("status"),
                "warnings": actual.get("warnings", []),
            }
        )
    return {
        "imageId": image["imageId"],
        "sha256": image["sha256"],
        "classification": image["classification"],
        "split": image["split"],
        "expectedReceiptCount": len(truth),
        "actualReceiptCount": len(predictions),
        "matches": comparisons,
        "missingTruthIds": [truth[index].get("truthId") for index in missing],
        "extraReceiptIds": [predictions[index].get("receipt_id") for index in extras],
        "falseReady": false_ready,
        "needsReview": sum(item.get("status") == "NEEDS_REVIEW" for item in predictions),
        "latencyMs": latency_ms,
        "providerCalls": 1,
        "actual": predictions,
    }


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    real = [item for item in results if item["classification"] == "real"]
    expected = sum(item["expectedReceiptCount"] for item in real)
    actual = sum(item["actualReceiptCount"] for item in real)
    matched = sum(len(item["matches"]) for item in real)
    field_rows = [match for item in real for match in item["matches"]]
    return {
        "realImages": len(real),
        "expectedReceipts": expected,
        "actualReceipts": actual,
        "detectionPrecision": {"numerator": matched, "denominator": actual},
        "detectionRecall": {"numerator": matched, "denominator": expected},
        "exactCountImages": {
            "numerator": sum(item["expectedReceiptCount"] == item["actualReceiptCount"] for item in real),
            "denominator": len(real),
        },
        "purchaseTotalAccuracy": {
            "numerator": sum(row["fieldMatches"]["purchaseTotal"] for row in field_rows),
            "denominator": len(field_rows),
        },
        "bookAmountAccuracy": {
            "numerator": sum(row["fieldMatches"]["bookAmount"] for row in field_rows),
            "denominator": len(field_rows),
        },
        "currencyAccuracy": {
            "numerator": sum(row["fieldMatches"]["currency"] for row in field_rows),
            "denominator": len(field_rows),
        },
        "dateAccuracy": {
            "numerator": sum(row["fieldMatches"]["date"] for row in field_rows),
            "denominator": len(field_rows),
        },
        "jointCriticalAccuracy": {
            "numerator": sum(row["jointCriticalMatch"] for row in field_rows),
            "denominator": len(field_rows),
        },
        "falseReady": sum(item["falseReady"] for item in real),
        "needsReview": sum(item["needsReview"] for item in real),
        "providerCalls": sum(item["providerCalls"] for item in results),
    }


async def _main() -> None:
    args = _arguments()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = _validate_manifest(manifest, manifest_path.parent)
    selected = [
        image for image in images if args.split == "all" or image["split"] == args.split
    ][: args.max_images]
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(_evaluate(image, semaphore) for image in selected))
    output = {
        "schemaVersion": 1,
        "runtime": {
            "provider": settings.AI_TEXT_PROVIDER,
            "model": get_model_name_for_task("RECEIPT_ANALYSIS"),
            "task": "RECEIPT_ANALYSIS",
            "mode": "VISION_ONLY",
            "concurrency": args.concurrency,
            "maxImages": args.max_images,
        },
        "metrics": _metrics(results),
        "images": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runtime": output["runtime"], "metrics": output["metrics"]}, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
