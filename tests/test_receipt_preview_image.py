import asyncio
import io
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile
from PIL import Image
from starlette.datastructures import Headers

from app.core.config import settings
from app.features.receipt.service import ReceiptAnalysisService
from app.schemas.receipt import ReceiptAnalysisMode, ReceiptAnalysisOptions
from app.services.ocr_service import OCRService


def encoded_image(size, *, orientation=1, mode="RGB", format="JPEG"):
    output = io.BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    Image.new(mode, size, "white").save(output, format=format, exif=exif)
    return output.getvalue()


def test_first_pass_is_bounded_and_financial_crop_keeps_original_pixels():
    # 준비: 큰 원본의 금액 누락만 보완하는 실제 서비스 경로.
    original = encoded_image((3000, 4000))
    item = {
        "title": "Example",
        "store_name": "Example",
        "merchant_evidence": "Example",
        "purchase_total": None,
        "original_amount": None,
        "detected_currency_code": "USD",
        "transaction_date": "2026-09-15",
        "bounding_box": [0.1, 0.1, 0.5, 0.9],
    }
    provider = AsyncMock()
    provider.call_with_image.side_effect = [
        {"receipts": [item]},
        {"receipts": [{**item, "purchase_total": "12.34"}]},
    ]
    uploaded = UploadFile(
        file=io.BytesIO(original),
        filename="receipt.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )

    # 실행: 전체 판독과 선택 복구를 실행한다. 실제 외부 호출은 없다.
    result = asyncio.run(
        ReceiptAnalysisService(OCRService(), provider).analyze(
            uploaded, ReceiptAnalysisOptions(analysis_mode=ReceiptAnalysisMode.VISION_ONLY)
        )
    )

    # 검증: 별도 영역 검출 없이 2회 호출하고 원본과 금융정보를 보존한다.
    assert provider.call_with_image.await_count == 2
    first, recovery = [call.kwargs for call in provider.call_with_image.call_args_list]
    with Image.open(io.BytesIO(first["image_bytes"])) as preview:
        assert preview.size == (1800, 2400)
        assert preview.format == "JPEG"
    with Image.open(io.BytesIO(recovery["image_bytes"])) as crop:
        assert crop.size == (1290, 3420)
    assert first["mime_type"] == recovery["mime_type"] == "image/jpeg"
    assert uploaded.file.getvalue() == original
    assert result.receipts[0].book_amount == Decimal("12.34")


@pytest.mark.parametrize(
    "orientation,expected", [(6, (1200, 2400)), (8, (1200, 2400)), (3, (2400, 1200))]
)
def test_preview_normalizes_rotation_before_resizing(orientation, expected):
    # 준비 / 실행: EXIF 좌표계를 먼저 바로잡는다.
    original = encoded_image((4000, 2000), orientation=orientation)
    data, mime = ReceiptAnalysisService._prepare_whole_image(original, "image/jpeg")

    # 검증: 복구에 쓰는 원본의 EXIF 보정과 같은 좌표계를 유지한다.
    with Image.open(io.BytesIO(data)) as preview:
        assert preview.size == expected
        assert preview.getexif().get(274, 1) == 1
        assert mime == "image/jpeg"


def test_small_preview_keeps_exact_original_without_upscaling():
    # 준비 / 실행
    original = encoded_image((100, 200))
    result = ReceiptAnalysisService._prepare_whole_image(original, "image/jpeg")

    # 검증
    assert result == (original, "image/jpeg")


def test_preview_checks_decoded_limit_before_provider(monkeypatch):
    # 준비: 크기 한도를 넘는 입력은 외부로 전송하지 않는다.
    monkeypatch.setattr(settings, "RECEIPT_VISION_MAX_IMAGE_PIXELS", 100)
    provider = AsyncMock()
    image = UploadFile(
        file=io.BytesIO(encoded_image((20, 20))),
        filename="large.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )

    # 실행 / 검증
    result = asyncio.run(
        ReceiptAnalysisService(OCRService(), provider).analyze(
            image, ReceiptAnalysisOptions(analysis_mode=ReceiptAnalysisMode.VISION_ONLY)
        )
    )
    provider.call_with_image.assert_not_awaited()
    assert "ANALYSIS_UNAVAILABLE" in result.warnings


def test_preview_composites_transparency_on_white():
    # 준비 / 실행: 투명 배경을 검정으로 바꾸어 글자를 잃지 않는다.
    output = io.BytesIO()
    Image.new("RGBA", (3000, 3000), (0, 0, 0, 0)).save(output, format="PNG")
    data, mime = ReceiptAnalysisService._prepare_whole_image(output.getvalue(), "image/png")

    # 검증
    with Image.open(io.BytesIO(data)) as preview:
        assert preview.getpixel((0, 0)) == (255, 255, 255)
        assert mime == "image/jpeg"
