import asyncio
import hashlib
import io
import json
import logging
import os
import re
import time
import unicodedata
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps

from app.ai.model_policy import get_task_model_policy
from app.ai.ports import (
    StructuredGenerationResult,
    StructuredImageGenerationProvider,
    TextGenerationProvider,
)
from app.core.config import settings
from app.features.receipt.parser import extract_receipt_candidates
from app.features.receipt.prompts import (
    build_receipt_identity_recovery_prompt,
    build_receipt_text_analysis_prompt,
    build_receipt_vision_prompt,
)
from app.features.receipt.runtime_identity import (
    get_receipt_runtime_identity,
    record_receipt_provider_call,
)
from app.features.receipt.validation import validate_response
from app.features.receipt.vision_scheduler import (
    ReceiptVisionCallStart,
    ReceiptVisionDeadlineExceeded,
    ReceiptVisionQueueFullError,
    get_receipt_vision_scheduler,
)
from app.schemas.receipt import (
    ReceiptAnalysisMode,
    ReceiptAnalysisOptions,
    ReceiptAnalysisResponse,
    ReceiptAmountReviewStatus,
    ReceiptStatus,
)
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ReceiptRecoveryPlan:
    identity_needed: bool
    financial_needed: bool

    @property
    def kind(self) -> str:
        return (
            "INCOMPLETE_REGION_RECOVERY"
            if self.financial_needed
            else "IDENTITY_REGION_RECOVERY"
        )


@dataclass
class _ReceiptRequestContext:
    started_at: float
    deadline: float
    provider_attempts: list[dict[str, Any]]


@dataclass(frozen=True)
class _PreparedRecovery:
    index: int
    plan: _ReceiptRecoveryPlan
    source_box: tuple[float, float, float, float]
    image_bytes: bytes
    prompt: str
    schema: dict[str, Any]

_NULLABLE_TEXT = {"type": "STRING", "nullable": True}
_ITEM_PROPERTIES: dict[str, Any] = {
    name: dict(_NULLABLE_TEXT)
    for name in (
        "title",
        "store_name",
        "branch_name",
        "merchant_evidence",
        "branch_evidence",
        "purchase_total",
        "original_amount",
        "cash_tendered",
        "change",
        "detected_currency_code",
        "currency_evidence",
        "transaction_date",
        "transaction_time",
        "source_date",
        "source_time",
        "date_order",
        "date_order_evidence",
        "category_name",
        "category_reason",
        "memo",
        "detected_language",
    )
}
_ITEM_PROPERTIES.update(
    {
        "payment_breakdown": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "payment_type": {
                        "type": "STRING",
                        "enum": [
                            "LOYALTY_POINTS",
                            "CASH",
                            "CREDIT_CARD",
                            "DEBIT_CARD",
                            "ELECTRONIC_MONEY",
                            "GIFT_CARD",
                            "VOUCHER",
                            "OTHER_PAID",
                            "UNKNOWN",
                        ],
                    },
                    "amount": dict(_NULLABLE_TEXT),
                    "evidence": dict(_NULLABLE_TEXT),
                    "duplicate_group": dict(_NULLABLE_TEXT),
                },
                "required": [
                    "payment_type",
                    "amount",
                    "evidence",
                    "duplicate_group",
                ],
            },
        },
        "confidence": {"type": "NUMBER", "nullable": True},
        "currency_confidence": {"type": "NUMBER", "nullable": True},
        "bounding_box": {
            "type": "ARRAY",
            "items": {"type": "NUMBER"},
            "nullable": True,
        },
        "status": {"type": "STRING", "enum": ["READY", "NEEDS_REVIEW", "UNREADABLE"]},
        "category_source": {
            "type": "STRING",
            "enum": ["EXISTING", "DEFAULT", "NEW", "FALLBACK"],
        },
    }
)
_RECEIPT_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "receipts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": _ITEM_PROPERTIES,
                "required": list(_ITEM_PROPERTIES),
            },
        },
    },
    "required": ["receipts"],
}
_IDENTITY_RECOVERY_PROPERTIES = {
    name: dict(_NULLABLE_TEXT)
    for name in ("title", "store_name", "branch_name", "merchant_evidence", "branch_evidence")
}
_IDENTITY_RECOVERY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "receipts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": _IDENTITY_RECOVERY_PROPERTIES,
                "required": list(_IDENTITY_RECOVERY_PROPERTIES),
            },
        }
    },
    "required": ["receipts"],
}


class ReceiptAnalysisService:
    def __init__(
        self, ocr_service: OCRService, ai_provider: TextGenerationProvider
    ) -> None:
        self.ocr_service = ocr_service
        self.ai_provider = ai_provider

    async def analyze(
        self,
        file: UploadFile,
        options: ReceiptAnalysisOptions | None = None,
        trace_id: str | None = None,
    ) -> ReceiptAnalysisResponse:
        started = time.perf_counter()
        request_context = _ReceiptRequestContext(
            started_at=started,
            deadline=started + settings.RECEIPT_ANALYSIS_TOTAL_TIMEOUT_SECONDS,
            provider_attempts=[],
        )
        trace_id = self._trace_id(trace_id)
        options = options or ReceiptAnalysisOptions()
        self.ocr_service._validate_file(file)
        # Enforce the same upload policy for vision and OCR; limit the read as well.
        image_bytes = await file.read(settings.OCR_MAX_FILE_SIZE + 1)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty receipt image")
        if len(image_bytes) > settings.OCR_MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Receipt image is too large")
        await file.seek(0)
        mode = self._resolve_analysis_mode(options)
        fallback: list[str] = []
        response: ReceiptAnalysisResponse | None = None
        if mode in (ReceiptAnalysisMode.VISION_FIRST, ReceiptAnalysisMode.VISION_ONLY):
            prompt = build_receipt_vision_prompt(options)
            provider_result: StructuredGenerationResult | None = None
            try:
                provider_result = await self._call_vision_provider(
                    prompt=prompt,
                    image_bytes=image_bytes,
                    mime_type=file.content_type or "image/jpeg",
                    request_context=request_context,
                    attempt_kind="WHOLE_IMAGE",
                )
                result = provider_result.data
                result, recovered_indexes = (
                    await self._recover_incomplete_regions(
                        result=result,
                        image_bytes=image_bytes,
                        prompt=prompt,
                        request_context=request_context,
                    )
                )
                response = validate_response(
                    result, options, ocr_engine="vision", used_ai=True
                )
                if recovered_indexes:
                    response.warnings.append("VISION_REGION_RECOVERY_USED")
                    for index in recovered_indexes:
                        if index < len(response.receipts):
                            response.receipts[index].warnings.append(
                                "REGION_RECOVERY_USED"
                            )
                self._write_vision_trace(
                    trace_id=trace_id,
                    image_bytes=image_bytes,
                    mime_type=file.content_type or "image/jpeg",
                    prompt=prompt,
                    request_context=request_context,
                    validated=response,
                )
                # Keep partial and low-confidence objects. A second whole-image pass
                # must never replace good receipts or merge them with an unreadable one.
                if not response.receipts and mode == ReceiptAnalysisMode.VISION_FIRST:
                    fallback.append("VISION_NO_RECEIPTS_OCR_FALLBACK")
                    response = None
            except Exception as exc:
                self._write_vision_trace(
                    trace_id=trace_id,
                    image_bytes=image_bytes,
                    mime_type=file.content_type or "image/jpeg",
                    prompt=prompt,
                    request_context=request_context,
                    validated=None,
                    error_type=type(exc).__name__,
                )
                logger.warning(
                    "Receipt vision failed. error_type=%s", type(exc).__name__
                )
                fallback.append("VISION_UNAVAILABLE")
                if mode == ReceiptAnalysisMode.VISION_ONLY:
                    response = ReceiptAnalysisResponse(
                        ocr_engine="vision", warnings=["ANALYSIS_UNAVAILABLE"]
                    )
        if response is None:
            response = await self._analyze_ocr(
                file, options, use_ai=mode != ReceiptAnalysisMode.OCR_ONLY
            )
            if fallback:
                fallback.append("OCR_FALLBACK_USED")
        response.warnings = list(dict.fromkeys([*fallback, *response.warnings]))
        response.analysis_trace_id = trace_id
        response.runtime_identity = get_receipt_runtime_identity()
        logger.info(
            "Receipt analysis mode=%s count=%d ready=%d review=%d unreadable=%d currencies=%s engine=%s fallback=%s latency_ms=%d",
            mode.value,
            response.receipt_count,
            sum(item.status == ReceiptStatus.READY for item in response.receipts),
            sum(
                item.status == ReceiptStatus.NEEDS_REVIEW for item in response.receipts
            ),
            sum(item.status == ReceiptStatus.UNREADABLE for item in response.receipts),
            sorted(
                {
                    item.detected_currency_code
                    for item in response.receipts
                    if item.detected_currency_code
                }
            ),
            response.ocr_engine,
            bool(fallback),
            round((time.perf_counter() - started) * 1000),
        )
        return response

    async def _call_vision_provider(
        self, *, prompt: str, image_bytes: bytes, mime_type: str,
        schema: dict[str, Any] = _RECEIPT_ANALYSIS_SCHEMA,
        request_context: _ReceiptRequestContext,
        attempt_kind: str,
        source_index: int | None = None,
        source_box: tuple[float, float, float, float] | None = None,
    ) -> StructuredGenerationResult:
        scheduler = get_receipt_vision_scheduler()

        async def invoke(slot: ReceiptVisionCallStart) -> StructuredGenerationResult:
            attempt = self._started_provider_attempt(
                kind=attempt_kind,
                image_bytes=image_bytes,
                mime_type=mime_type,
                prompt=prompt,
                request_context=request_context,
                slot=slot,
                source_index=source_index,
                source_box=source_box,
            )
            request_context.provider_attempts.append(attempt)
            record_receipt_provider_call()
            call_started = time.perf_counter()
            try:
                metadata_method = getattr(
                    type(self.ai_provider), "call_with_image_with_metadata", None
                )
                if callable(metadata_method):
                    provider = cast(StructuredImageGenerationProvider, self.ai_provider)
                    result = await provider.call_with_image_with_metadata(
                        type_name="RECEIPT_ANALYSIS",
                        prompt=prompt,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        schema=schema,
                    )
                else:
                    data = await self.ai_provider.call_with_image(
                        type_name="RECEIPT_ANALYSIS",
                        prompt=prompt,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        schema=schema,
                    )
                    result = StructuredGenerationResult(data=data)
                self._complete_provider_attempt(attempt, result)
                return result
            except BaseException as exc:
                attempt.update(
                    {
                        "status": "failed",
                        "errorType": type(exc).__name__,
                    }
                )
                raise
            finally:
                attempt["serverCallLatencyMs"] = max(
                    0, round((time.perf_counter() - call_started) * 1000)
                )

        return cast(
            StructuredGenerationResult,
            await scheduler.run(
                kind="WHOLE_IMAGE" if attempt_kind == "WHOLE_IMAGE" else "RECOVERY",
                deadline=request_context.deadline,
                invoke=invoke,
            ),
        )

    async def _recover_incomplete_regions(
        self,
        *,
        result: Any,
        image_bytes: bytes,
        prompt: str,
        request_context: _ReceiptRequestContext,
    ) -> tuple[Any, list[int]]:
        raw_items = result.get("receipts") if isinstance(result, dict) else None
        if not isinstance(raw_items, list) or not raw_items:
            return result, []
        maximum = settings.RECEIPT_VISION_RECOVERY_MAX_CROPS
        if maximum == 0:
            return result, []

        merged_result = deepcopy(result)
        merged_items = merged_result["receipts"]
        recovered_indexes: list[int] = []
        remaining = request_context.deadline - time.perf_counter()
        if remaining < settings.RECEIPT_VISION_RECOVERY_MIN_REMAINING_SECONDS:
            for item in merged_items:
                if self._recovery_plan(item) is not None and isinstance(item, dict):
                    self._append_warning(item, "REGION_RECOVERY_SKIPPED_TIME_BUDGET")
            return merged_result, []

        try:
            prepared = await asyncio.to_thread(
                self._prepare_recovery_regions,
                merged_items,
                image_bytes,
                prompt,
                maximum,
            )
        except Exception:
            for item in merged_items:
                if self._recovery_plan(item) is not None and isinstance(item, dict):
                    self._append_warning(item, "REGION_RECOVERY_SOURCE_UNAVAILABLE")
            return merged_result, []

        async def recover(region: _PreparedRecovery) -> StructuredGenerationResult:
            return await self._call_vision_provider(
                prompt=region.prompt,
                image_bytes=region.image_bytes,
                mime_type="image/png",
                schema=region.schema,
                request_context=request_context,
                attempt_kind=region.plan.kind,
                source_index=region.index,
                source_box=region.source_box,
            )

        outcomes = await asyncio.gather(
            *(recover(region) for region in prepared), return_exceptions=True
        )
        for region, outcome in zip(prepared, outcomes, strict=True):
            target = merged_items[region.index]
            if isinstance(outcome, BaseException):
                if isinstance(target, dict):
                    warning = (
                        "REGION_RECOVERY_SKIPPED_QUEUE_LIMIT"
                        if isinstance(outcome, ReceiptVisionQueueFullError)
                        else "REGION_RECOVERY_TIMEOUT"
                        if isinstance(outcome, ReceiptVisionDeadlineExceeded)
                        else "REGION_RECOVERY_FAILED"
                    )
                    self._append_warning(target, warning)
                    target["status"] = "NEEDS_REVIEW"
                continue
            recovery_result = outcome
            recovered = (
                recovery_result.data.get("receipts")
                if isinstance(recovery_result.data, dict)
                else None
            )
            if not isinstance(recovered, list) or len(recovered) != 1:
                if isinstance(target, dict):
                    self._append_warning(target, "REGION_RECOVERY_INVALID_RESPONSE")
                    target["status"] = "NEEDS_REVIEW"
                continue
            candidate = recovered[0]
            if not isinstance(candidate, dict):
                if isinstance(target, dict):
                    self._append_warning(target, "REGION_RECOVERY_INVALID_RESPONSE")
                    target["status"] = "NEEDS_REVIEW"
                continue
            merged = self._merge_missing_receipt_fields(
                target,
                candidate,
                same_receipt_region=True,
                merge_identity=region.plan.identity_needed,
                merge_financial=region.plan.financial_needed,
                source_box=region.source_box,
            )
            if region.plan.identity_needed and isinstance(target, dict):
                # A second model reading of the crop is useful review evidence, but
                # agreement between two model responses is not pixel-level proof.
                # Keep every identity recovery reviewable even when the identity
                # strings happen to be identical or their evidence is inconsistent.
                if target.get("identity_verification") != "SOURCE_REGION_CONFLICT":
                    target["identity_verification"] = "SOURCE_REGION_MODEL_REREAD"
                target["status"] = "NEEDS_REVIEW"
                self._append_warning(target, "IDENTITY_SOURCE_REVIEW_REQUIRED")
                if target.get("identity_source_box") is None:
                    target["identity_source_box"] = list(region.source_box)
            if merged or region.plan.identity_needed:
                recovered_indexes.append(region.index)
        for item in merged_items:
            self._annotate_identity_verification(item)
        return merged_result, recovered_indexes

    def _prepare_recovery_regions(
        self,
        raw_items: list[Any],
        image_bytes: bytes,
        prompt: str,
        maximum: int,
    ) -> list[_PreparedRecovery]:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            source = ImageOps.exif_transpose(opened)
            if source.width * source.height > settings.RECEIPT_VISION_MAX_IMAGE_PIXELS:
                raise ValueError("Receipt image exceeds the decoded pixel limit")
            source = source.convert("RGB")

        facts_recovery_prompt = (
            prompt
            + "\nThis image is an automatically selected high-resolution region of one "
            "physical receipt whose whole-image candidate was incomplete. Extract only "
            "the receipt visible in this crop. Do not infer facts outside the crop."
        )
        prepared: list[_PreparedRecovery] = []
        for index, raw in enumerate(raw_items):
            plan = self._recovery_plan(raw)
            if plan is None:
                continue
            if len(prepared) >= maximum:
                if isinstance(raw, dict):
                    self._append_warning(raw, "REGION_RECOVERY_SKIPPED_LIMIT")
                continue
            box = self._normalized_box(raw.get("bounding_box"))
            if box is None:
                if isinstance(raw, dict):
                    self._append_warning(raw, "SOURCE_REGION_MISSING_OR_INVALID")
                continue
            identity_recovery = not plan.financial_needed and plan.identity_needed
            crop_box = self._identity_region_box(box) if identity_recovery else box
            crop = self._crop_region(source, crop_box)
            if crop is None:
                if isinstance(raw, dict):
                    self._append_warning(raw, "SOURCE_REGION_TOO_SMALL")
                continue
            if identity_recovery and crop.width < 1400:
                scale = min(2.5, 1400 / crop.width)
                crop = crop.resize(
                    (round(crop.width * scale), round(crop.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG", optimize=True)
            prepared.append(
                _PreparedRecovery(
                    index=index,
                    plan=plan,
                    source_box=crop_box,
                    image_bytes=buffer.getvalue(),
                    prompt=(
                        build_receipt_identity_recovery_prompt(raw)
                        if identity_recovery
                        else facts_recovery_prompt
                    ),
                    schema=(
                        _IDENTITY_RECOVERY_SCHEMA
                        if identity_recovery
                        else _RECEIPT_ANALYSIS_SCHEMA
                    ),
                )
            )
        return prepared

    @staticmethod
    def _needs_region_recovery(raw: object) -> bool:
        return ReceiptAnalysisService._recovery_plan(raw) is not None

    @staticmethod
    def _recovery_plan(raw: object) -> _ReceiptRecoveryPlan | None:
        if not isinstance(raw, dict):
            return None
        missing_amount = raw.get("purchase_total") is None and raw.get(
            "original_amount"
        ) is None
        financial_needed = (
            missing_amount
            or raw.get("detected_currency_code") is None
            or raw.get("transaction_date") is None
        )
        identity_needed = ReceiptAnalysisService._needs_identity_region_recovery(raw)
        if not financial_needed and not identity_needed:
            return None
        return _ReceiptRecoveryPlan(
            identity_needed=identity_needed,
            financial_needed=financial_needed,
        )

    @staticmethod
    def _needs_identity_region_recovery(raw: object) -> bool:
        if not isinstance(raw, dict) or not (raw.get("title") or raw.get("store_name")):
            return False
        if any(ReceiptAnalysisService._is_incomplete_identity(raw.get(field))
               for field in ("title", "store_name", "branch_name")):
            return True
        if not ReceiptAnalysisService._identity_text_is_self_consistent(raw):
            return True
        box = ReceiptAnalysisService._normalized_box(raw.get("bounding_box"))
        area = (box[2] - box[0]) * (box[3] - box[1]) if box else 1.0
        return bool(raw.get("branch_name")) and area <= 0.55

    @staticmethod
    def _is_incomplete_identity(value: object) -> bool:
        return isinstance(value, str) and ("..." in value or "…" in value)

    @staticmethod
    def _normalized_box(value: object) -> tuple[float, float, float, float] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        if any(
            not isinstance(part, (int, float))
            or isinstance(part, bool)
            or not 0 <= part <= 1
            for part in value
        ):
            return None
        left, top, right, bottom = (float(part) for part in value)
        return (left, top, right, bottom) if left < right and top < bottom else None

    @staticmethod
    def _crop_region(
        source: Image.Image, box: tuple[float, float, float, float]
    ) -> Image.Image | None:
        margin_x = max(8, round(source.width * 0.015))
        margin_top = max(8, round(source.height * 0.015))
        # Payment allocation, loyalty points, change and card detail are commonly
        # printed at the bottom edge, so preserve more source pixels below the box.
        margin_bottom = max(16, round(source.height * 0.04))
        left = max(0, int(box[0] * source.width) - margin_x)
        top = max(0, int(box[1] * source.height) - margin_top)
        right = min(source.width, int(box[2] * source.width) + margin_x)
        bottom = min(source.height, int(box[3] * source.height) + margin_bottom)
        if right - left < 96 or bottom - top < 96:
            return None
        return source.crop((left, top, right, bottom))

    @staticmethod
    def _identity_region_box(
        box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        left, top, right, bottom = box
        # Merchant logos and printed branch names belong to the receipt header. A
        # bounded header crop avoids treating lower head-office/legal addresses as
        # the branch while preserving original pixels and aspect ratio.
        header_bottom = min(bottom, top + (bottom - top) * 0.28)
        return left, top, right, header_bottom

    @staticmethod
    def _merge_missing_receipt_fields(
        target: object,
        recovery: dict[str, Any],
        *,
        same_receipt_region: bool,
        merge_identity: bool,
        merge_financial: bool,
        source_box: tuple[float, float, float, float] | None = None,
    ) -> bool:
        if not isinstance(target, dict):
            return False
        changed = False
        identity_self_consistent = ReceiptAnalysisService._identity_text_is_self_consistent(
            recovery
        )
        target_identity_self_consistent = (
            ReceiptAnalysisService._identity_text_is_self_consistent(target)
        )
        identity_differs = any(
            recovery.get(field) is not None and recovery.get(field) != target.get(field)
            for field in ("title", "store_name", "branch_name")
        )
        identity_conflict = merge_identity and identity_differs and (
            not same_receipt_region
            or target_identity_self_consistent
            or not identity_self_consistent
        )
        if identity_conflict:
            changed = ReceiptAnalysisService._mark_identity_conflict(target) or changed
        elif merge_identity and same_receipt_region and identity_self_consistent:
            identity_fields = (
                ("store_name", recovery.get("store_name")),
                ("merchant_evidence", recovery.get("merchant_evidence")),
                ("branch_name", recovery.get("branch_name")),
                ("branch_evidence", recovery.get("branch_evidence")),
                ("title", recovery.get("title")),
            )
            for field, value in identity_fields:
                if value is not None and target.get(field) != value:
                    target[field] = value
                    changed = True
            if target.get("identity_verification") != "SOURCE_REGION_MODEL_REREAD":
                target["identity_verification"] = "SOURCE_REGION_MODEL_REREAD"
                changed = True
            target["status"] = "NEEDS_REVIEW"
            ReceiptAnalysisService._append_warning(
                target, "IDENTITY_SOURCE_REVIEW_REQUIRED"
            )
            if source_box is not None:
                target["identity_source_box"] = list(source_box)

        # Identity disagreement must not suppress otherwise safe completion of
        # missing finance fields. Finance can only cross the merge boundary when
        # the crop was derived from this candidate's source receipt region.
        if merge_financial and same_receipt_region:
            financial_changed = False
            for field in (
                "purchase_total",
                "original_amount",
                "cash_tendered",
                "change",
                "detected_currency_code",
                "currency_evidence",
                "currency_confidence",
                "transaction_date",
                "transaction_time",
                "source_date",
                "source_time",
                "date_order",
                "date_order_evidence",
                "memo",
                "detected_language",
            ):
                if target.get(field) is None and recovery.get(field) is not None:
                    target[field] = recovery[field]
                    changed = True
                    financial_changed = True
            if not target.get("payment_breakdown") and recovery.get("payment_breakdown"):
                target["payment_breakdown"] = recovery["payment_breakdown"]
                changed = True
                financial_changed = True
            if financial_changed:
                target["financial_recovery_provenance"] = "SAME_RECEIPT_SOURCE_REGION"
                if source_box is not None:
                    target["financial_source_box"] = list(source_box)
        return changed

    @staticmethod
    def _append_warning(target: dict[str, Any], warning: str) -> None:
        warnings = target.setdefault("warnings", [])
        if isinstance(warnings, list) and warning not in warnings:
            warnings.append(warning)

    @staticmethod
    def _mark_identity_conflict(target: dict[str, Any]) -> bool:
        changed = False
        if target.get("identity_verification") != "SOURCE_REGION_CONFLICT":
            target["identity_verification"] = "SOURCE_REGION_CONFLICT"
            changed = True
        if target.get("status") != "NEEDS_REVIEW":
            target["status"] = "NEEDS_REVIEW"
            changed = True
        warnings = target.setdefault("warnings", [])
        if isinstance(warnings, list) and "IDENTITY_REGION_CONFLICT" not in warnings:
            warnings.append("IDENTITY_REGION_CONFLICT")
            changed = True
        return changed

    @staticmethod
    def _identity_text(value: object) -> str:
        if not isinstance(value, str):
            return ""
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return re.sub(r"[\s\-‐‑‒–—―・･·,，.。:：/／()（）\[\]【】]+", "", normalized)

    @staticmethod
    def _identity_text_is_self_consistent(raw: object) -> bool:
        if not isinstance(raw, dict):
            return False
        store = ReceiptAnalysisService._identity_text(raw.get("store_name"))
        title = ReceiptAnalysisService._identity_text(raw.get("title"))
        merchant_evidence = ReceiptAnalysisService._identity_text(raw.get("merchant_evidence"))
        branch = ReceiptAnalysisService._identity_text(raw.get("branch_name"))
        branch_evidence = ReceiptAnalysisService._identity_text(raw.get("branch_evidence"))
        if not store or store not in merchant_evidence or store not in title:
            return False
        return not branch or (branch in branch_evidence and branch in title)

    @staticmethod
    def _annotate_identity_verification(raw: object) -> None:
        if not isinstance(raw, dict) or raw.get("identity_verification"):
            return
        if ReceiptAnalysisService._identity_text_is_self_consistent(raw):
            # These strings are all emitted by one model response. Agreement is
            # useful diagnostics, but it is not independent verification against
            # the pixels.
            raw["identity_verification"] = "MODEL_TEXT_SELF_CONSISTENT"
            raw["identity_source_box"] = raw.get("bounding_box")
        elif raw.get("title") or raw.get("store_name"):
            raw["identity_verification"] = "UNVERIFIED"
            raw["status"] = "NEEDS_REVIEW"

    @staticmethod
    def _trace_id(value: str | None) -> str:
        if value and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}", value):
            return value
        return str(uuid.uuid4())

    @staticmethod
    def _image_metadata(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "bytes": len(image_bytes),
            "mime": mime_type,
            "width": None,
            "height": None,
            "format": None,
            "exifOrientation": None,
        }
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                metadata.update(
                    width=image.width,
                    height=image.height,
                    format=image.format,
                    exifOrientation=image.getexif().get(274),
                )
        except Exception:
            metadata["decodeStatus"] = "UNAVAILABLE"
        return metadata

    @staticmethod
    def _hash_json(value: Any) -> str:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _write_vision_trace(
        self,
        *,
        trace_id: str,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        request_context: _ReceiptRequestContext,
        validated: ReceiptAnalysisResponse | None,
        error_type: str | None = None,
    ) -> None:
        trace_root = os.environ.get("RECEIPT_DEBUG_TRACE_DIR", "").strip()
        if not trace_root:
            return
        policy = get_task_model_policy("RECEIPT_ANALYSIS")
        scheduler = get_receipt_vision_scheduler().snapshot
        payload = {
            "traceId": trace_id,
            "input": self._image_metadata(image_bytes, mime_type),
            "request": {
                "type": "RECEIPT_ANALYSIS",
                "imageDetail": "high",
                "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "schemaSha256": self._hash_json(_RECEIPT_ANALYSIS_SCHEMA),
                "reasoningEffort": policy.reasoning_effort,
                "maxOutputTokens": policy.max_output_tokens,
            },
            "providerAttempts": request_context.provider_attempts,
            "providerCallCount": len(request_context.provider_attempts),
            "timing": {
                "totalElapsedMs": max(
                    0, round((time.perf_counter() - request_context.started_at) * 1000)
                ),
                "deadlineMs": round(
                    (request_context.deadline - request_context.started_at) * 1000
                ),
            },
            "scheduler": {
                "active": scheduler.active,
                "activeRecoveries": scheduler.active_recoveries,
                "pending": scheduler.pending,
                "maxObservedActive": scheduler.max_observed_active,
                "maxObservedRecoveries": scheduler.max_observed_recoveries,
                "processScoped": True,
            },
            "runtimeIdentity": get_receipt_runtime_identity().model_dump(mode="json"),
            "validation": {
                "response": (
                    validated.model_dump(mode="json") if validated is not None else None
                ),
                "candidateCount": validated.receipt_count if validated else None,
            },
            "errorType": error_type,
        }
        try:
            directory = Path(trace_root).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{trace_id}.json"
            temporary = directory / f".{trace_id}.{uuid.uuid4().hex}.tmp"
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, target)
        except Exception:
            logger.warning("Receipt debug trace write failed. trace_id=%s", trace_id)

    def _started_provider_attempt(
        self,
        *,
        kind: str,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        request_context: _ReceiptRequestContext,
        slot: ReceiptVisionCallStart,
        source_index: int | None = None,
        source_box: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "sourceIndex": source_index,
            "sourceBox": source_box,
            "input": self._image_metadata(image_bytes, mime_type),
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "queueWaitMs": slot.queue_wait_ms,
            "inFlightAtStart": slot.in_flight_at_start,
            "recoveryInFlightAtStart": slot.recovery_in_flight_at_start,
            "startedOffsetMs": max(
                0, round((slot.started_at - request_context.started_at) * 1000)
            ),
            "status": "started",
        }

    @staticmethod
    def _complete_provider_attempt(
        attempt: dict[str, Any], result: StructuredGenerationResult
    ) -> None:
        raw = result.data
        attempt.update(
            {
                "name": result.provider,
                "model": result.model,
                "status": result.status,
                "incompleteReason": result.incomplete_reason,
                "latencyMs": result.latency_ms,
                "inputTokens": result.input_tokens,
                "outputTokens": result.output_tokens,
                "structuredResponse": raw,
                "candidateCount": (
                    len(raw.get("receipts", []))
                    if isinstance(raw, dict)
                    and isinstance(raw.get("receipts"), list)
                    else None
                ),
            }
        )

    @staticmethod
    def _resolve_analysis_mode(options: ReceiptAnalysisOptions) -> ReceiptAnalysisMode:
        mode_value = options.analysis_mode or settings.RECEIPT_ANALYSIS_MODE
        try:
            return ReceiptAnalysisMode(
                str(
                    mode_value.value
                    if isinstance(mode_value, ReceiptAnalysisMode)
                    else mode_value
                ).upper()
            )
        except ValueError:
            # A deployment typo must not silently enable the legacy OCR fallback.
            logger.warning("Invalid receipt analysis mode; using VISION_ONLY")
            return ReceiptAnalysisMode.VISION_ONLY

    async def _analyze_ocr(
        self, file: UploadFile, options: ReceiptAnalysisOptions, *, use_ai: bool
    ) -> ReceiptAnalysisResponse:
        try:
            document = await self.ocr_service.extract_document_from_upload(
                file=file,
                ocr_language=options.ocr_language or settings.OCR_LANGUAGE,
            )
        except HTTPException as exc:
            if exc.status_code in (400, 413):
                raise
            logger.warning("Receipt OCR failed. error_type=%s", type(exc).__name__)
            return ReceiptAnalysisResponse(
                ocr_engine="paddleocr", warnings=["OCR_UNAVAILABLE"]
            )
        except Exception as exc:
            logger.warning("Receipt OCR failed. error_type=%s", type(exc).__name__)
            return ReceiptAnalysisResponse(
                ocr_engine="paddleocr", warnings=["OCR_UNAVAILABLE"]
            )
        if not document.lines:
            return ReceiptAnalysisResponse(
                ocr_engine="paddleocr", warnings=["OCR_NO_TEXT"]
            )
        candidates = extract_receipt_candidates(document.text)
        warnings = ["OCR_LANGUAGE_HINT_LIMITATION"]
        if use_ai:
            try:
                record_receipt_provider_call()
                result = await self.ai_provider.call(
                    type_name="RECEIPT_ANALYSIS",
                    data=build_receipt_text_analysis_prompt(
                        document.text,
                        candidates,
                        options,
                        spatial_lines=[line.as_payload() for line in document.lines],
                    ),
                    schema=_RECEIPT_ANALYSIS_SCHEMA,
                )
                response = validate_response(
                    result, options, ocr_engine="paddleocr", used_ai=True
                )
                if response.receipts:
                    if not all(line.bounding_box for line in document.lines):
                        for item in response.receipts:
                            item.book_amount = None
                            item.original_amount = None
                            item.review_status = ReceiptAmountReviewStatus.NEEDS_REVIEW
                            item.status = ReceiptStatus.NEEDS_REVIEW
                            item.warnings.append("OCR_RECEIPT_BOUNDARIES_UNVERIFIED")
                    response.warnings.extend(warnings)
                    return response
            except Exception as exc:
                logger.warning(
                    "Receipt OCR structuring failed. error_type=%s", type(exc).__name__
                )
            warnings.append("OCR_AI_UNAVAILABLE_RULE_FALLBACK")
        # No local image-segmentation heuristic claims to establish physical receipt
        # boundaries. This fallback is an explicitly unconfirmed manual-review draft.
        response = validate_response(
            {"receipts": [candidates]}, options, ocr_engine="paddleocr", used_ai=False
        )
        item = response.receipts[0]
        item.status = ReceiptStatus.NEEDS_REVIEW
        item.warnings.append("OCR_RECEIPT_BOUNDARIES_UNVERIFIED")
        if candidates["total_line_count"] != 1:
            item.book_amount = None
            item.original_amount = None
            item.review_status = ReceiptAmountReviewStatus.NEEDS_REVIEW
            item.warnings.append("OCR_MULTIPLE_OR_UNKNOWN_TOTALS")
        response.warnings.extend(warnings)
        return response
