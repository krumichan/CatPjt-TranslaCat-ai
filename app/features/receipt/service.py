import logging
import re
import time
from typing import Any

from fastapi import HTTPException, UploadFile

from app.ai.ports import TextGenerationProvider
from app.core.config import settings
from app.features.receipt.parser import extract_receipt_candidates
from app.features.receipt.prompts import (
    build_receipt_text_analysis_prompt,
    build_receipt_vision_prompt,
)
from app.schemas.receipt import (
    ReceiptAnalysisMode,
    ReceiptAnalysisOptions,
    ReceiptAnalysisResponse,
)
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)

_SUSPICIOUS_OCR_PATTERNS = (
    r"[水氷米]{4,}",
    r"[_]{2,}",
    r"[^\w\sぁ-んァ-ン一-龥가-힣¥￥()/:%.,\-]{6,}",
)
_MEANINGFUL_MEMO_CHAR_PATTERN = r"[A-Za-z0-9가-힣ぁ-んァ-ン一-龥]"
_DEFAULT_STOP_KEYWORDS = (
    "クレジット売上票",
    "カード会社",
    "ブランド名",
    "会員番号",
    "承認番号",
    "ATC",
    "お客様控え",
    "登録番号",
    "株式会社",
)
_DEFAULT_IMPORTANT_KEYWORDS = (
    "合計",
    "税込合計",
    "総合計",
    "お買上計",
    "小計",
    "計",
    "TOTAL",
    "Total",
    "total",
    "円",
    "￥",
    "¥",
    "イートイン",
    "テイクアウト",
)
_DEFAULT_EXCLUDE_ITEM_KEYWORDS = (
    "ください",
    "キャンペーン",
    "おためし",
    "お知らせ",
    "ご利用",
    "ありがとう",
)

_RECEIPT_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING", "nullable": True},
        "store_name": {"type": "STRING", "nullable": True},
        "amount": {"type": "INTEGER", "nullable": True},
        "transaction_date": {"type": "STRING", "nullable": True},
        "category_name": {"type": "STRING", "nullable": True},
        "memo": {"type": "STRING", "nullable": True},
        "confidence": {"type": "NUMBER", "nullable": True},
    },
    "required": [
        "title",
        "store_name",
        "amount",
        "transaction_date",
        "category_name",
        "memo",
        "confidence",
    ],
}


class ReceiptAnalysisService:
    def __init__(
        self,
        ocr_service: OCRService,
        ai_provider: TextGenerationProvider,
    ) -> None:
        self.ocr_service = ocr_service
        self.ai_provider = ai_provider

    async def analyze(
        self,
        file: UploadFile,
        options: ReceiptAnalysisOptions | None = None,
    ) -> ReceiptAnalysisResponse:
        analysis_options = options or ReceiptAnalysisOptions()
        analysis_mode = self._resolve_analysis_mode(analysis_options)

        if analysis_mode == ReceiptAnalysisMode.VISION_ONLY:
            return await self._analyze_with_vision_only(file, analysis_options)
        if analysis_mode == ReceiptAnalysisMode.VISION_FIRST:
            return await self._analyze_with_vision_first(file, analysis_options)
        if analysis_mode == ReceiptAnalysisMode.OCR_ONLY:
            return await self._analyze_with_ocr_only(file, analysis_options)

        return await self._analyze_with_ocr_and_ai(file, analysis_options)

    def _resolve_analysis_mode(self, options: ReceiptAnalysisOptions) -> ReceiptAnalysisMode:
        mode_value = options.analysis_mode or getattr(
            settings,
            "RECEIPT_ANALYSIS_MODE",
            ReceiptAnalysisMode.OCR_WITH_AI.value,
        )

        if isinstance(mode_value, ReceiptAnalysisMode):
            return mode_value

        try:
            return ReceiptAnalysisMode(str(mode_value).strip().upper())
        except ValueError:
            logger.warning("지원하지 않는 영수증 분석 모드입니다. OCR_WITH_AI로 처리합니다. mode=%s", mode_value)
            return ReceiptAnalysisMode.OCR_WITH_AI

    async def _analyze_with_vision_only(
        self,
        file: UploadFile,
        options: ReceiptAnalysisOptions,
    ) -> ReceiptAnalysisResponse:
        total_start = time.perf_counter()
        image_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"

        ai_start = time.perf_counter()
        ai_result = await self._analyze_image_with_ai(
            image_bytes=image_bytes,
            mime_type=mime_type,
            options=options,
        )
        ai_elapsed = time.perf_counter() - ai_start
        total_elapsed = time.perf_counter() - total_start

        logger.info("Receipt analysis completed with VISION_ONLY. total=%.2fs, ai=%.2fs", total_elapsed, ai_elapsed)

        return self._build_response(
            result=ai_result,
            raw_text=None,
            used_ai=True,
            candidates={},
            ocr_engine="gemini_vision",
        )

    async def _analyze_with_vision_first(
        self,
        file: UploadFile,
        options: ReceiptAnalysisOptions,
    ) -> ReceiptAnalysisResponse:
        total_start = time.perf_counter()
        image_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"

        try:
            ai_start = time.perf_counter()
            ai_result = await self._analyze_image_with_ai(
                image_bytes=image_bytes,
                mime_type=mime_type,
                options=options,
            )
            ai_elapsed = time.perf_counter() - ai_start
            response = self._build_response(
                result=ai_result,
                raw_text=None,
                used_ai=True,
                candidates={},
                ocr_engine="gemini_vision",
            )

            threshold = getattr(settings, "GEMINI_VISION_CONFIDENCE_THRESHOLD", 0.75)
            if (response.confidence or 0) >= threshold:
                total_elapsed = time.perf_counter() - total_start
                logger.info(
                    "Receipt analysis completed with VISION_FIRST. total=%.2fs, ai=%.2fs, confidence=%.2f",
                    total_elapsed,
                    ai_elapsed,
                    response.confidence or 0,
                )
                return response

            logger.warning("Gemini Vision confidence is low. fallback to OCR_WITH_AI. confidence=%s", response.confidence)
        except Exception as exc:
            logger.warning("Gemini Vision 분석 실패. OCR_WITH_AI로 fallback 합니다: %s", exc)

        await file.seek(0)
        return await self._analyze_with_ocr_and_ai(file, options)

    async def _analyze_with_ocr_only(
        self,
        file: UploadFile,
        options: ReceiptAnalysisOptions,
    ) -> ReceiptAnalysisResponse:
        total_start = time.perf_counter()
        ocr_language = options.ocr_language or settings.OCR_LANGUAGE

        ocr_start = time.perf_counter()
        raw_text = await self.ocr_service.extract_text_from_upload(
            file=file,
            ocr_language=ocr_language,
        )
        ocr_elapsed = time.perf_counter() - ocr_start

        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="영수증에서 텍스트를 추출하지 못했습니다.")

        parse_start = time.perf_counter()
        candidates = extract_receipt_candidates(raw_text)
        parse_elapsed = time.perf_counter() - parse_start
        total_elapsed = time.perf_counter() - total_start

        logger.info(
            "Receipt analysis completed with OCR_ONLY. total=%.2fs, ocr=%.2fs, parse=%.2fs",
            total_elapsed,
            ocr_elapsed,
            parse_elapsed,
        )

        return self._build_rule_based_response(raw_text, candidates)

    async def _analyze_with_ocr_and_ai(
        self,
        file: UploadFile,
        options: ReceiptAnalysisOptions,
    ) -> ReceiptAnalysisResponse:
        total_start = time.perf_counter()
        ocr_language = options.ocr_language or settings.OCR_LANGUAGE

        ocr_start = time.perf_counter()
        raw_text = await self.ocr_service.extract_text_from_upload(
            file=file,
            ocr_language=ocr_language,
        )
        ocr_elapsed = time.perf_counter() - ocr_start

        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="영수증에서 텍스트를 추출하지 못했습니다.")

        parse_start = time.perf_counter()
        candidates = extract_receipt_candidates(raw_text)
        parse_elapsed = time.perf_counter() - parse_start

        try:
            ai_start = time.perf_counter()
            ai_result = await self._analyze_with_ai(
                raw_text=raw_text,
                candidates=candidates,
                options=options,
            )
            ai_elapsed = time.perf_counter() - ai_start
            total_elapsed = time.perf_counter() - total_start

            logger.info(
                "Receipt analysis completed with OCR_WITH_AI. total=%.2fs, ocr=%.2fs, parse=%.2fs, ai=%.2fs",
                total_elapsed,
                ocr_elapsed,
                parse_elapsed,
                ai_elapsed,
            )

            return self._build_response(
                result=ai_result,
                raw_text=raw_text,
                used_ai=True,
                candidates=candidates,
                ocr_engine="paddleocr",
            )
        except Exception as exc:
            total_elapsed = time.perf_counter() - total_start
            logger.warning("영수증 AI 구조화에 실패하여 룰 기반 후보값으로 응답합니다: %s", exc)
            logger.info(
                "Receipt analysis completed with OCR fallback. total=%.2fs, ocr=%.2fs, parse=%.2fs",
                total_elapsed,
                ocr_elapsed,
                parse_elapsed,
            )
            return self._build_rule_based_response(raw_text, candidates)

    def _compact_raw_text_for_ai(
        self,
        raw_text: str,
        options: ReceiptAnalysisOptions,
    ) -> str:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return raw_text

        stop_keywords = tuple(options.stop_keywords) if options.stop_keywords is not None else _DEFAULT_STOP_KEYWORDS
        important_keywords = (
            tuple(options.important_keywords)
            if options.important_keywords is not None
            else _DEFAULT_IMPORTANT_KEYWORDS
        )

        compact_lines: list[str] = []

        for index, line in enumerate(lines):
            if any(keyword in line for keyword in stop_keywords):
                break

            digit_count = len(re.findall(r"\d", line))
            if digit_count >= 10 and not self._looks_like_date_line(line):
                continue

            if index < 8:
                compact_lines.append(line)
                continue
            if self._looks_like_date_line(line):
                compact_lines.append(line)
                continue
            if any(keyword in line for keyword in important_keywords):
                compact_lines.append(line)
                continue
            if self._looks_like_amount_line(line):
                compact_lines.append(line)
                continue
            if self._looks_like_item_line(line, options):
                compact_lines.append(line)
                continue

            if len(compact_lines) >= 35:
                break

        return "\n".join(dict.fromkeys(compact_lines))

    async def _analyze_image_with_ai(
        self,
        image_bytes: bytes,
        mime_type: str,
        options: ReceiptAnalysisOptions,
    ) -> dict[str, Any]:
        prompt = build_receipt_vision_prompt(options)
        result = await self.ai_provider.call_with_image(
            type_name="RECEIPT_ANALYSIS",
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            schema=_RECEIPT_ANALYSIS_SCHEMA,
        )

        if not isinstance(result, dict):
            raise ValueError("Gemini Vision 응답이 객체 형식이 아닙니다.")

        return result

    async def _analyze_with_ai(
        self,
        raw_text: str,
        candidates: dict[str, Any],
        options: ReceiptAnalysisOptions,
    ) -> dict[str, Any]:
        compact_raw_text = self._compact_raw_text_for_ai(
            raw_text=raw_text,
            options=options,
        )
        payload = build_receipt_text_analysis_prompt(
            raw_text=compact_raw_text,
            candidates=candidates,
            options=options,
        )
        result = await self.ai_provider.call(
            type_name="RECEIPT_ANALYSIS",
            data=payload,
            schema=_RECEIPT_ANALYSIS_SCHEMA,
        )

        if not isinstance(result, dict):
            raise ValueError("Gemini 응답이 객체 형식이 아닙니다.")

        return result

    def _looks_like_date_line(self, line: str) -> bool:
        return re.search(r"(20\d{2})[./\-年]?\d{1,2}[./\-月]?\d{1,2}", line) is not None

    def _looks_like_amount_line(self, line: str) -> bool:
        normalized = line.replace(",", "").strip()
        if re.fullmatch(r"[¥￥]?\s*\d{2,7}", normalized):
            return True
        if re.search(r"[¥￥]\s*\d{2,7}", normalized):
            return True
        return False

    def _looks_like_item_line(self, line: str, options: ReceiptAnalysisOptions) -> bool:
        if len(line) < 2 or len(line) > 24:
            return False

        digit_count = len(re.findall(r"\d", line))
        if digit_count >= 6:
            return False

        exclude_item_keywords = (
            tuple(options.exclude_item_keywords)
            if options.exclude_item_keywords is not None
            else _DEFAULT_EXCLUDE_ITEM_KEYWORDS
        )
        if any(keyword in line for keyword in exclude_item_keywords):
            return False

        meaningful_chars = re.findall(_MEANINGFUL_MEMO_CHAR_PATTERN, line)
        meaningful_ratio = len(meaningful_chars) / max(len(line), 1)
        return meaningful_ratio >= 0.6

    def _build_response(
        self,
        result: dict[str, Any],
        raw_text: str | None,
        used_ai: bool,
        candidates: dict[str, Any] | None = None,
        ocr_engine: str = "paddleocr",
    ) -> ReceiptAnalysisResponse:
        title = self._to_optional_str(result.get("title"))
        store_name = self._to_optional_str(result.get("store_name"))
        amount = self._to_optional_int(result.get("amount"))
        transaction_date = self._to_optional_str(result.get("transaction_date"))
        category_name = self._to_optional_str(result.get("category_name"))
        memo = self._sanitize_memo(self._to_optional_str(result.get("memo")))
        ai_confidence = self._to_optional_float(result.get("confidence"))

        confidence = self._adjust_confidence(
            ai_confidence=ai_confidence,
            title=title,
            store_name=store_name,
            amount=amount,
            transaction_date=transaction_date,
            category_name=category_name,
            memo=memo,
            raw_text=raw_text or "",
            candidates=candidates or {},
        )

        return ReceiptAnalysisResponse(
            title=title,
            store_name=store_name,
            amount=amount,
            transaction_date=transaction_date,
            category_name=category_name,
            memo=memo,
            confidence=confidence,
            raw_text=raw_text,
            ocr_engine=ocr_engine,
            used_ai=used_ai,
        )

    def _adjust_confidence(
        self,
        ai_confidence: float | None,
        title: str | None,
        store_name: str | None,
        amount: int | None,
        transaction_date: str | None,
        category_name: str | None,
        memo: str | None,
        raw_text: str,
        candidates: dict[str, Any],
    ) -> float:
        confidence = ai_confidence if ai_confidence is not None else 0.75
        confidence = max(0.0, min(confidence, 1.0))

        has_store = bool(store_name or title)
        has_amount = amount is not None
        has_date = transaction_date is not None
        has_category = category_name is not None

        if not has_amount:
            confidence = min(confidence, 0.65)
        if not has_date:
            confidence = min(confidence, 0.75)
        if not has_store:
            confidence = min(confidence, 0.75)
        if not has_category:
            confidence = min(confidence, 0.85)
        if raw_text and self._looks_noisy_ocr_text(raw_text):
            confidence = min(confidence, 0.85)
        if memo is None:
            confidence = min(confidence, 0.85)

        amount_candidates = candidates.get("amountCandidates") or []
        if amount is not None and amount_candidates and amount not in amount_candidates:
            confidence = min(confidence, 0.75)

        if has_store and has_amount and has_date:
            confidence = max(confidence, 0.78)

        return round(confidence, 2)

    def _sanitize_memo(self, memo: str | None) -> str | None:
        if memo is None:
            return None

        text = memo.strip()
        if len(text) <= 1:
            return None

        has_hangul = re.search(r"[가-힣]", text) is not None
        has_japanese = re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text) is not None
        if has_hangul and has_japanese:
            return None

        meaningful_chars = re.findall(_MEANINGFUL_MEMO_CHAR_PATTERN, text)
        meaningful_ratio = len(meaningful_chars) / max(len(text), 1)
        if meaningful_ratio < 0.6:
            return None

        return text

    def _looks_noisy_ocr_text(self, raw_text: str) -> bool:
        if not raw_text.strip():
            return True

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return True

        joined_text = "\n".join(lines)
        suspicious_count = 0
        for pattern in _SUSPICIOUS_OCR_PATTERNS:
            suspicious_count += len(re.findall(pattern, joined_text))

        short_lines = [line for line in lines if len(line) <= 2]
        short_line_ratio = len(short_lines) / max(len(lines), 1)

        return suspicious_count >= 2 or short_line_ratio >= 0.35

    def _build_rule_based_response(
        self,
        raw_text: str,
        candidates: dict[str, Any],
    ) -> ReceiptAnalysisResponse:
        amount_candidates = candidates.get("amountCandidates") or []
        date_candidates = candidates.get("dateCandidates") or []
        store_name_candidates = candidates.get("storeNameCandidates") or []

        store_name = store_name_candidates[0] if store_name_candidates else None
        amount = amount_candidates[0] if amount_candidates else None
        transaction_date = date_candidates[0] if date_candidates else None

        return ReceiptAnalysisResponse(
            title=store_name,
            store_name=store_name,
            amount=amount,
            transaction_date=transaction_date,
            category_name=None,
            memo=None,
            confidence=0.45 if amount or transaction_date or store_name else 0.2,
            raw_text=raw_text,
            ocr_engine="paddleocr",
            used_ai=False,
        )

    def _to_optional_str(self, value: Any) -> str | None:
        if value is None:
            return None

        text = str(value).strip()
        return text or None

    def _to_optional_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _to_optional_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None

        return max(0.0, min(confidence, 1.0))
