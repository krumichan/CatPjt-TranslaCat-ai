"""정상 AI 라우팅·권한·OCR 좌표 및 번역 ID 연결을 합성 입력으로 검증한다."""

import io
import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from app.ai.ports import StructuredGenerationResult
from app.api.v1.camera_translation import get_camera_model_provider, get_camera_ocr_service
from app.core.config import settings
from app.main import app
from app.services.ocr_service import OCRDocument, OCRLine

_KEY = "synthetic-camera-translation-key"
_HEADERS = {"X-API-KEY": _KEY}
_OCR_PATH = "/api/v1/camera-translation/ocr"
_TRANSLATION_PATH = "/api/v1/camera-translation/segments"
_IMAGE_ID = "c138cdbe-f01d-402a-9ffe-c6e9757af9e9"
_REQUEST_ID = "c9975729-f9f2-4388-8ab4-11ae1e8eb4f5"
_INPUT_SIZE = (40, 60)


@pytest.fixture
def camera_client(monkeypatch: pytest.MonkeyPatch):
    """실제 라우팅과 공통 API 키 미들웨어를 통과하고 두 모델 의존성만 바꾼다."""
    # 공통 미들웨어가 검사하는 서비스별 인증 상태를 합성 개발 설정으로 격리한다.
    monkeypatch.setattr(settings, "SERVER_API_KEY", _KEY)
    monkeypatch.setattr(settings, "NOVEL_AUTH_ENVIRONMENT", "Development")
    monkeypatch.setattr(settings, "NOVEL_SERVER_API_KEY", SecretStr("synthetic-camera-novel-key"))
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", "legacy")

    ocr = AsyncMock()
    provider = AsyncMock()
    provider.ready = True
    app.dependency_overrides[get_camera_ocr_service] = lambda: ocr
    app.dependency_overrides[get_camera_model_provider] = lambda: provider
    client = TestClient(app)
    try:
        yield client, ocr, provider
    finally:
        client.close()
        app.dependency_overrides.pop(get_camera_ocr_service, None)
        app.dependency_overrides.pop(get_camera_model_provider, None)


# 방향 보정 전 좌표를 재현할 JPEG를 메모리에서 작성한다.
def _jpeg(orientation: int = 1, size: tuple[int, int] = _INPUT_SIZE) -> bytes:
    with Image.new("RGB", size, color=(70, 110, 155)) as image:
        with io.BytesIO() as buffer:
            exif = Image.Exif()
            exif[274] = orientation
            image.save(buffer, format="JPEG", exif=exif)
            return buffer.getvalue()


# 클라이언트가 제출하는 카메라 번역의 JSON 필드 이름으로 요청을 구성한다.
def _translation_data() -> dict:
    return {
        "requestId": _REQUEST_ID,
        "mode": "PREVIEW",
        "sourceLanguage": "ja",
        "targetLanguage": "ko",
        "segments": [
            {"id": "top", "text": "本日のおすすめ", "context": "昼食"},
            {"id": "middle", "text": "本日のおすすめ"},
            {"id": "bottom", "text": "価格"},
        ],
    }


def test_camera_ocr_preserves_reflected_exif_and_common_polygon(camera_client) -> None:
    """EXIF 7 회전·반전과 불규칙 입력 꼭짓점을 공통 형식으로 반환한다."""
    client, ocr, _ = camera_client
    ocr.extract_document_from_image.return_value = OCRDocument(
        [OCRLine("本日のおすすめ", [[26, 15], [3, 15], [3, 7], [26, 7]], 0.99)]
    )

    response = client.post(
        _OCR_PATH,
        data={"imageId": _IMAGE_ID, "sourceLanguage": "ja"},
        files={"image": ("selected-crop.jpg", _jpeg(orientation=7), "image/jpeg")},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["imageId"] == _IMAGE_ID
    assert response.json()["schemaVersion"] == 1
    assert response.json()["coordinateSpace"] == "image-pixels-top-left"
    assert response.json()["image"] == {
        "widthPx": 60,
        "heightPx": 40,
        "inputToImage": [0, -1, 60, -1, 0, 40],
    }
    assert response.json()["regions"] == [
        {
            "segmentId": "line-0",
            "readingOrder": 0,
            "polygon": [[3, 7], [26, 7], [26, 15], [3, 15]],
            "sourceText": "本日のおすすめ",
            "writingMode": "unknown",
            "confidence": None,
        }
    ]
    ocr.extract_document_from_image.assert_awaited_once()
    assert ocr.extract_document_from_image.call_args.args[0].size == (60, 40)
    assert ocr.extract_document_from_image.call_args.kwargs["ocr_language"] == "japan"


@pytest.mark.parametrize(
    ("document", "status"),
    [
        (OCRDocument(), "NO_TEXT"),
        (OCRDocument([OCRLine("주의", None)]), "UNCERTAIN"),
    ],
)
def test_camera_ocr_keeps_empty_detection_distinct_from_bad_geometry(
    camera_client, document: OCRDocument, status: str
) -> None:
    """검출 결과 없음과 감지했지만 위치를 확인할 수 없는 상태를 구분한다."""
    client, ocr, _ = camera_client
    ocr.extract_document_from_image.return_value = document

    response = client.post(
        _OCR_PATH,
        data={"imageId": _IMAGE_ID, "sourceLanguage": "ko"},
        files={"image": ("selected-crop.jpg", _jpeg(), "image/jpeg")},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["status"] == status
    assert response.json()["regions"] == []


@pytest.mark.parametrize(
    ("size", "content_type", "status"),
    [
        (_INPUT_SIZE, "image/png", 415),
        ((3_073, 1), "image/jpeg", 413),
    ],
)
def test_camera_ocr_rejects_mime_mismatch_and_oversize_before_recognition(
    camera_client, size: tuple[int, int], content_type: str, status: int
) -> None:
    """입력 형식·해상도 위반은 모델을 사용하지 않고 안전한 코드로 반환한다."""
    client, ocr, _ = camera_client

    response = client.post(
        _OCR_PATH,
        data={"imageId": _IMAGE_ID, "sourceLanguage": "en"},
        files={"image": ("selected-crop", _jpeg(size=size), content_type)},
        headers=_HEADERS,
    )

    assert response.status_code == status
    assert response.json() == {"detail": {"code": "OCR_DECODE_FAILED"}}
    ocr.extract_document_from_image.assert_not_awaited()


def test_camera_segments_keep_input_order_and_discard_duplicate_model_ids(camera_client) -> None:
    """모델 출력의 순서나 같은 원문 때문에 영역의 번역을 섞지 않는다."""
    client, _, provider = camera_client
    provider.execute_explicit.return_value = StructuredGenerationResult(
        data={
            "segments": [
                {"id": "middle", "translatedText": "둘째 번역"},
                {"id": "unexpected", "translatedText": "다른 이미지"},
                {"id": "top", "translatedText": "오늘의 추천"},
                {"id": "middle", "translatedText": "서로 다른 둘째 번역"},
            ]
        }
    )

    response = client.post(_TRANSLATION_PATH, json=_translation_data(), headers=_HEADERS)

    assert response.status_code == 200
    result = response.json()
    assert result["requestId"] == _REQUEST_ID
    assert [item["id"] for item in result["segments"]] == ["top", "middle", "bottom"]
    assert [item["translatedText"] for item in result["segments"]] == ["오늘의 추천", None, None]
    assert [item["errorCode"] for item in result["segments"]] == [
        None,
        "INVALID_MODEL_OUTPUT",
        "INVALID_MODEL_OUTPUT",
    ]
    provider.execute_explicit.assert_awaited_once()
    execute_options = provider.execute_explicit.call_args.kwargs
    assert execute_options["remaining_milliseconds"] == 4_500
    assert json.loads(execute_options["messages"][0]["content"]) == {
        "sourceLanguage": "Japanese",
        "targetLanguage": "Korean",
        "segments": _translation_data()["segments"],
    }


def test_camera_routes_keep_auth_and_validation_before_inference(camera_client) -> None:
    """공용 인증과 필드 검사가 실패하면 모델을 호출하거나 입력을 반사하지 않는다."""
    client, ocr, provider = camera_client
    data = _translation_data()
    data["targetLanguage"] = "en"

    unauthorized = client.post(_TRANSLATION_PATH, json=_translation_data())
    invalid = client.post(_TRANSLATION_PATH, json=data, headers=_HEADERS)

    assert unauthorized.status_code == 401
    assert invalid.status_code == 422
    assert invalid.json() == {"detail": {"code": "INVALID_INPUT"}}
    ocr.extract_document_from_image.assert_not_awaited()
    provider.execute_explicit.assert_not_awaited()


def test_camera_translation_deadline_returns_safe_error_without_retry(camera_client) -> None:
    """모델 대기 제한을 HTTP 504로 변환하며 다른 모델 요청을 만들지 않는다."""
    client, _, provider = camera_client
    provider.execute_explicit.side_effect = TimeoutError("sensitive provider detail")

    response = client.post(_TRANSLATION_PATH, json=_translation_data(), headers=_HEADERS)

    assert response.status_code == 504
    assert response.json() == {"detail": {"code": "DEADLINE_EXCEEDED"}}
    provider.execute_explicit.assert_awaited_once()
