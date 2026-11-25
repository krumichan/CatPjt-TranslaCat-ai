from __future__ import annotations

import io
from collections import deque
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api.internal.image_translation import get_image_translation_service, router
from app.features.image_translation.service import ImageTranslationService
from app.services.ocr_service import OCRDocument, OCRLine, OCRService


class _Ocr:
    def __init__(self, *documents: OCRDocument | Exception) -> None:
        self.documents = deque(documents)
        self.calls: list[tuple[tuple[int, int], str, str]] = []

    # タイルサイズとモデル指定を記録し、順番どおりに固定結果を返す。
    async def extract_document_from_image(
        self, image: Image.Image, *, ocr_language: str, ocr_version: str
    ) -> OCRDocument:
        self.calls.append((image.size, ocr_language, ocr_version))
        document = self.documents.popleft()
        if isinstance(document, Exception):
            raise document
        return document


class _Translation:
    def __init__(self, fail_at: int = 0) -> None:
        self.fail_at = fail_at
        self.calls: list[tuple[list[str], str]] = []

    # 翻訳順を記録し、指定された一回だけを失敗させる。
    async def translate_batch(self, texts: list[str], type_name: str) -> list[str]:
        self.calls.append((texts, type_name))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("temporary provider failure")
        return [f"Translated {text}" for text in texts]


# 実体のある小さな静止画像を作り、必要なら EXIF 回転を指定する。
def _image(width: int, height: int, *, orientation: int | None = None) -> bytes:
    with Image.new("RGB", (width, height), "white") as image, io.BytesIO() as encoded:
        if orientation is None:
            image.save(encoded, format="PNG")
        else:
            metadata = Image.Exif()
            metadata[274] = orientation
            image.save(encoded, format="JPEG", exif=metadata)
        return encoded.getvalue()


# 並べ替えも確認できるよう、左下から始まる文字領域を作る。
def _line(text: str, left: float, top: float, right: float, bottom: float) -> OCRLine:
    return OCRLine(text, [[left, bottom], [right, bottom], [right, top], [left, top]], 0.97)


# FastAPI の multipart・JSON 契約は実際の Router を通過させる。
def _api(service: ImageTranslationService) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/internal/v1")
    app.dependency_overrides[get_image_translation_service] = lambda: service
    return TestClient(app)


# EXIF 補正後のサイズ、順番、座標の始点と nullable の wire 契約を確認する。
def test_image_page_route_returns_oriented_coordinates_and_camel_case() -> None:
    # 準備: 90° 回転する画像と横並びの日本語領域を用意する。
    ocr = _Ocr(OCRDocument([
        _line("左の台詞", 10, 10, 35, 40),
        _line("右の台詞", 60, 11, 75, 47),
    ]))
    translation = _Translation()
    client = _api(ImageTranslationService(ocr, translation))
    data = {"sourceLanguage": "ja", "targetLanguage": "en"}

    # 実行: multipart のページ処理を呼ぶ。
    response = client.post(
        "/internal/v1/image-translation/page",
        data=data,
        files={"file": ("rotated.jpg", _image(120, 90, orientation=6), "image/jpeg")},
    )

    # 検証: 正規化された寸法と右からの読順が API の各領域に反映される。
    assert response.status_code == 200
    result = response.json()
    assert (result["state"], result["widthPx"], result["heightPx"]) == ("READY", 90, 120)
    assert result["ocrContractVersion"] == 1
    assert ocr.calls == [((90, 120), "japan", "PP-OCRv5")]
    assert translation.calls == [(["右の台詞", "左の台詞"], "IMAGE_en")]
    assert result["segments"][0] == {
        "segmentId": "segment-0",
        "readingOrder": 0,
        "sourceText": "右の台詞",
        "translatedText": "Translated 右の台詞",
        "polygon": [[60.0, 11.0], [75.0, 11.0], [75.0, 47.0], [60.0, 47.0]],
        "confidence": None,
        "writingMode": "vertical-rtl",
        "needsReview": False,
    }


# 隣接タイルの分割された認識だけを統合し、別の位置で繰り返す台詞を保持する。
@pytest.mark.asyncio
async def test_long_page_merges_seam_and_keeps_distant_repeated_text() -> None:
    # 準備: 2048px の境界に全文と切断された断片を配置する。
    ocr = _Ocr(
        OCRDocument([_line("また会おう", 10, 1990, 90, 2048)]),
        OCRDocument([
            _line("また会おうね", 10, 198, 90, 270),
            _line("また会おうね", 10, 480, 90, 540),
        ]),
    )
    translation = _Translation()
    service = ImageTranslationService(ocr, translation)

    # 実行: 高さが一タイルを超える一枚を処理する。
    result = await service.translate_page(_image(120, 2500), "image/png", "ja", "en")

    # 検証: 結果座標はタイル相対ではなく元ページの空間に戻る。
    assert result.state == "READY"
    assert ocr.calls == [((120, 2048), "japan", "PP-OCRv5"), ((120, 708), "japan", "PP-OCRv5")]
    assert [region.polygon[0][1] for region in result.segments] == [1990, 2272]
    assert [region.source_text for region in result.segments] == ["また会おうね", "また会おうね"]


# 認識できた部分を維持し、座標の欠損を空ページと区別する。
@pytest.mark.asyncio
async def test_page_with_invalid_polygon_retains_confirmed_region() -> None:
    # 準備: 同じ OCR 結果に有効な領域と失われた領域を含める。
    ocr = _Ocr(OCRDocument([_line("Some text", 5, 10, 50, 25), OCRLine("Broken")]))
    service = ImageTranslationService(ocr, _Translation())

    # 実行: 通常の一枚を処理する。
    result = await service.translate_page(_image(120, 120), "image/png", "en", "ja")

    # 検証: 翻訳済みの領域は返すが、ページは完了扱いにしない。
    assert result.state == "FAILED"
    assert result.error_code == "OCR_UNCERTAIN"
    assert result.retryable is False
    assert [segment.translated_text for segment in result.segments] == ["Translated Some text"]


# Provider の一部失敗で先に成功した翻訳や領域順序が失われないことを確認する。
@pytest.mark.asyncio
async def test_translation_failure_preserves_earlier_batch() -> None:
    # 準備: 21 領域を作り、二つ目の翻訳単位で失敗させる。
    ocr = _Ocr(OCRDocument([
        _line(f"Line {index}", 8, index * 13, 60, index * 13 + 5)
        for index in range(21)
    ]))
    translation = _Translation(fail_at=2)
    service = ImageTranslationService(ocr, translation)

    # 実行: Provider を実際に呼ばずにページ処理の失敗分岐を通る。
    result = await service.translate_page(_image(120, 300), "image/png", "en", "ja")

    # 検証: 再呼び出しなしで既に成功した領域を表示できる。
    assert result.state == "FAILED"
    assert result.error_code == "TRANSLATION_TEMPORARY"
    assert result.retryable is True
    assert [len(texts) for texts, _ in translation.calls] == [20, 1]
    assert result.segments[19].translated_text == "Translated Line 19"
    assert result.segments[20].translated_text is None
    assert result.segments[20].needs_review


# 空の認識、MIME 不一致、OCR 実行不可を別の契約として確認する。
def test_route_distinguishes_empty_page_invalid_input_and_ocr_outage() -> None:
    # 準備: OCR が一度空を返したあとにエンジン障害を起こす。
    ocr = _Ocr(OCRDocument(), RuntimeError("engine unavailable"))
    translation = _Translation()
    client = _api(ImageTranslationService(ocr, translation))
    url = "/internal/v1/image-translation/page"
    data = {"sourceLanguage": "ja", "targetLanguage": "en"}
    image = _image(48, 48)

    # 実行: 画像を正常送信し、形式不一致とエンジン障害も順に送信する。
    empty = client.post(url, data=data, files={"file": ("page.png", image, "image/png")})
    invalid = client.post(url, data=data, files={"file": ("page.jpg", image, "image/jpeg")})
    unavailable = client.post(url, data=data, files={"file": ("page.png", image, "image/png")})

    # 検証: 形式不一致は OCR に渡らず、空のページに翻訳依頼は発生しない。
    assert empty.status_code == 200
    assert empty.json()["state"] == "NO_TEXT"
    assert empty.json()["segments"] == []
    assert invalid.status_code == 415
    assert invalid.json()["detail"] == {"code": "INVALID_IMAGE", "retryable": False}
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"] == {"code": "OCR_UNAVAILABLE", "retryable": True}
    assert not translation.calls


# 領収書のリサイズ設定を越えるタイルでも、OCR に渡す寸法と原点を変えない。
@pytest.mark.asyncio
async def test_shared_ocr_image_adapter_does_not_resize(monkeypatch: pytest.MonkeyPatch) -> None:
    # 準備: エンジン生成を置き換え、ファイル化された実画素を実行時に確認する。
    service = OCRService()
    file_paths: list[Path] = []
    calls: list[tuple[str, str | None]] = []

    async def get_model(language: str, *, ocr_version: str | None = None) -> object:
        calls.append((language, ocr_version))
        return object()

    def extract(_: Any, path: str) -> OCRDocument:
        file_paths.append(Path(path))
        with Image.open(path) as image:
            assert (image.size, image.format) == ((2600, 30), "PNG")
        return OCRDocument([_line("text", 2400, 4, 2499, 16)])

    monkeypatch.setattr(service, "_get_ocr", get_model)
    monkeypatch.setattr(service, "_extract_document_from_path", extract)

    # 実行: 座標を保持する画像アダプターを呼ぶ。
    with Image.new("RGB", (2600, 30)) as image:
        result = await service.extract_document_from_image(
            image, ocr_language="en", ocr_version="PP-OCRv5"
        )

    # 検証: OCR の返す座標は元の画素と同じで、一時ファイルは処理後に消える。
    assert calls == [("en", "PP-OCRv5")]
    assert result.lines[0].bounding_box == [[2400, 16], [2499, 16], [2499, 4], [2400, 4]]
    assert len(file_paths) == 1 and not file_paths[0].exists()
