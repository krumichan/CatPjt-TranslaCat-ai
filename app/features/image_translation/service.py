from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from anyio.to_thread import run_sync

from app.features.image_translation.ocr import (
    ImageInputError,
    ImageOcr,
    _Region,
    decode_image,
    recognize_page,
)
from app.schemas.image_translation import (
    ImageLanguage,
    ImagePageState,
    ImagePageTranslation,
    ImageTranslationSegment,
)

logger = logging.getLogger(__name__)

_PAGE_CONCURRENCY = 4
_TRANSLATION_CHUNK = 20


class TextTranslation(Protocol):
    """同じ順序で受け取ったテキストの翻訳を返す既存の翻訳契約。"""

    async def translate_batch(self, texts: list[str], type_name: str) -> list[str]:
        """入力と同数の翻訳を順番どおり返す。"""
        ...


class ImageOcrUnavailable(RuntimeError):
    """画像 OCR が実行できず、今回のページ結果を確定できなかった。"""


class ImageTranslationService:
    """OCR 領域と翻訳を結び付け、回やジョブの状態を持たずにページ処理を行う。"""

    def __init__(self, ocr: ImageOcr, translation: TextTranslation) -> None:
        self._ocr = ocr
        self._translation = translation
        self._page_slots = asyncio.Semaphore(_PAGE_CONCURRENCY)

    async def translate_page(
        self,
        contents: bytes,
        content_type: str,
        source_language: ImageLanguage,
        target_language: ImageLanguage,
    ) -> ImagePageTranslation:
        """静止画像を処理し、失敗時も確定済みの領域を返す。"""
        if source_language == target_language:
            raise ImageInputError(400, "INVALID_LANGUAGE_PAIR")

        # CPU の同時使用を制限し、デコード後の全画像は OCR が終わるまでだけ保持する。
        async with self._page_slots:
            image = await run_sync(decode_image, contents, content_type)
            with image:
                width, height = image.size
                try:
                    regions, geometry_needs_review = await recognize_page(
                        image, source_language, self._ocr
                    )
                except Exception as exc:
                    logger.warning("Page OCR failed. error_type=%s", type(exc).__name__)
                    raise ImageOcrUnavailable from exc

            # 対応関係を保持したまま、翻訳済み領域は後続の失敗時にも返す。
            translations = await self._translate_confirmed_regions(regions, target_language)

        # OCR 検出の完了、確認が必要な領域、翻訳失敗を異なる状態として返す。
        segments = [
            ImageTranslationSegment(
                segment_id=f"segment-{index}",
                reading_order=index,
                source_text=region.text,
                translated_text=translations.get(index),
                polygon=region.polygon,
                confidence=None,
                writing_mode=region.writing_mode,
                needs_review=region.needs_review or index not in translations,
            )
            for index, region in enumerate(regions)
        ]
        ocr_needs_review = geometry_needs_review or any(region.needs_review for region in regions)
        translation_failed = any(
            not region.needs_review and index not in translations
            for index, region in enumerate(regions)
        )
        state: ImagePageState = "READY" if segments else "NO_TEXT"
        if ocr_needs_review or translation_failed:
            state = "FAILED"
        result = ImagePageTranslation(
            width_px=width,
            height_px=height,
            ocr_contract_version=1,
            segments=segments,
            state=state,
        )
        if state == "FAILED":
            result.error_code = (
                "TRANSLATION_TEMPORARY" if translation_failed else "OCR_UNCERTAIN"
            )
            result.retryable = translation_failed
        return result

    # アイテムの位置を維持し、失敗した単位だけを未翻訳のまま残す。
    async def _translate_confirmed_regions(
        self, regions: list[_Region], target_language: ImageLanguage
    ) -> dict[int, str]:
        eligible = [
            (index, region.text)
            for index, region in enumerate(regions)
            if not region.needs_review
        ]
        translations: dict[int, str] = {}
        for start in range(0, len(eligible), _TRANSLATION_CHUNK):
            chunk = eligible[start : start + _TRANSLATION_CHUNK]
            texts = [text for _, text in chunk]

            # Provider の失敗はページ再試行に委ね、ここでは追加呼び出しを行わない。
            try:
                values = await self._translation.translate_batch(
                    texts, type_name=f"IMAGE_{target_language}"
                )
                if len(values) != len(texts) or any(
                    not isinstance(value, str) or not value.strip() for value in values
                ):
                    raise ValueError("Invalid image translation batch")
            except Exception as exc:
                logger.warning("Page translation batch failed. error_type=%s", type(exc).__name__)
                continue

            translations.update(
                (index, value) for (index, _), value in zip(chunk, values, strict=True)
            )
        return translations
