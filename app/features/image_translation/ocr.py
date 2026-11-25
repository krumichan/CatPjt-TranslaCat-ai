from __future__ import annotations

import io
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Protocol

from PIL import Image, ImageOps

from app.schemas.image_translation import ImageLanguage, Point, WritingMode
from app.services.ocr_service import OCRDocument, OCRLine

MAX_IMAGE_BYTES = 15 * 1024 * 1024
_MAX_REGIONS = 300
_MAX_TILE_CANDIDATES = _MAX_REGIONS * 4
_MAX_WIDTH = 4096
_MAX_HEIGHT = 24_000
_MAX_PIXELS = 20_000_000
_TILE_HEIGHT = 2048
_TILE_OVERLAP = 256
_EDGE_MARGIN = 4
_OCR_VERSION = "PP-OCRv5"
_OCR_LANGUAGES: dict[ImageLanguage, str] = {"ja": "japan", "ko": "korean", "en": "en"}
_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class ImageInputError(ValueError):
    """入力画像を処理できないときに HTTP 境界へ安全な失敗情報を渡す。"""

    def __init__(self, status_code: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code


class ImageOcr(Protocol):
    """渡されたピクセルの原点を保ったまま OCR を実行する。"""

    async def extract_document_from_image(
        self,
        image: Image.Image,
        *,
        ocr_language: str,
        ocr_version: str,
    ) -> OCRDocument:
        """画像を読み取り、同じ画像空間のテキスト領域を返す。"""
        ...


@dataclass(frozen=True)
class _Region:
    text: str
    normalized_text: str
    polygon: list[Point]
    bounds: tuple[float, float, float, float]
    writing_mode: WritingMode
    tile_top: int
    clipped: bool
    needs_review: bool


def decode_image(contents: bytes, content_type: str) -> Image.Image:
    """静止画像のサイズと形式を検査し、EXIF 補正後の RGB 画像を返す。"""
    if not contents:
        raise ImageInputError(400, "INVALID_IMAGE")
    if len(contents) > MAX_IMAGE_BYTES:
        raise ImageInputError(413, "IMAGE_LIMIT_EXCEEDED")
    if content_type not in _FORMATS.values():
        raise ImageInputError(415, "INVALID_IMAGE")

    try:
        with Image.open(io.BytesIO(contents)) as source:
            # ファイルの宣言と実体を照合し、デコード前に展開サイズを制限する。
            if _FORMATS.get(source.format or "") != content_type or getattr(
                source, "is_animated", False
            ):
                raise ImageInputError(415, "INVALID_IMAGE")
            width, height = source.size
            if source.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
            if width > _MAX_WIDTH or height > _MAX_HEIGHT or width * height > _MAX_PIXELS:
                raise ImageInputError(413, "IMAGE_LIMIT_EXCEEDED")

            # ブラウザーと同じ回転後の座標を使い、透明部分だけを OCR 用に白く合成する。
            with ImageOps.exif_transpose(source) as oriented:
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    with oriented.convert("RGBA") as rgba:
                        background = Image.new("RGB", rgba.size, "white")
                        with rgba.getchannel("A") as alpha:
                            background.paste(rgba, mask=alpha)
                        return background
                return oriented.convert("RGB")
    except ImageInputError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageInputError(413, "IMAGE_LIMIT_EXCEEDED") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise ImageInputError(400, "INVALID_IMAGE") from exc


async def recognize_page(
    image: Image.Image, language: ImageLanguage, ocr: ImageOcr
) -> tuple[list[_Region], bool]:
    """縦長ページを重複付きで認識し、ページ座標と確認要否を返す。"""
    width, height = image.size
    candidates: list[_Region] = []
    needs_review = False
    top = 0

    # 隣接タイルの重複部分を残して、境界で分かれた認識を照合する。
    while top < height:
        bottom = min(top + _TILE_HEIGHT, height)
        with image.crop((0, top, width, bottom)) as tile:
            document = await ocr.extract_document_from_image(
                tile, ocr_language=_OCR_LANGUAGES[language], ocr_version=_OCR_VERSION
            )
            for line in document.lines:
                if len(candidates) >= _MAX_TILE_CANDIDATES:
                    needs_review = True
                    break
                candidate = _project_region(line, language, width, height, top, bottom)
                if candidate is None:
                    needs_review = True
                else:
                    candidates.append(candidate)
        if bottom == height or len(candidates) >= _MAX_TILE_CANDIDATES:
            needs_review |= bottom < height
            break
        top = bottom - _TILE_OVERLAP

    # 同じページ位置の候補を統合してから、確定した順序で ID を割り当てられるようにする。
    ordered = _reading_order(_merge_regions(candidates), language)
    return ordered[:_MAX_REGIONS], needs_review or len(ordered) > _MAX_REGIONS


# 四角形をタイルからページへ移し、欠損した多角形と境界での切断を判定する。
def _project_region(
    line: OCRLine, language: ImageLanguage, width: int, height: int, top: int, bottom: int
) -> _Region | None:
    text = line.text.strip()
    if not text or line.bounding_box is None or len(line.bounding_box) != 4:
        return None

    # 左上原点の時計回りに整列し、契約に従い最上・最左の点から始める。
    if any(
        len(point) != 2 or not all(math.isfinite(value) for value in point)
        for point in line.bounding_box
    ):
        return None
    points = [
        (min(max(x, 0.0), float(width)), min(max(y + top, float(top)), float(bottom)))
        for x, y in line.bounding_box
    ]
    x_center = sum(x for x, _ in points) / 4
    y_center = sum(y for _, y in points) / 4
    points.sort(key=lambda point: math.atan2(point[1] - y_center, point[0] - x_center))
    first = min(range(4), key=lambda index: (points[index][1], points[index][0]))
    polygon = points[first:] + points[:first]
    if not _is_convex(polygon):
        return None

    # タイルの内部境界に触れた認識だけを切断候補にし、画像端の文字を保持する。
    left = min(x for x, _ in polygon)
    upper = min(y for _, y in polygon)
    right = max(x for x, _ in polygon)
    lower = max(y for _, y in polygon)
    clipped = (top > 0 and upper <= top + _EDGE_MARGIN) or (
        bottom < height and lower >= bottom - _EDGE_MARGIN
    )
    writing_mode: WritingMode = "horizontal-ltr"
    if lower - upper > 1.6 * (right - left) and len(text) > 1:
        writing_mode = "vertical-rtl" if language == "ja" else "unknown"
    normalized = unicodedata.normalize("NFKC", "".join(text.split()))
    return _Region(
        text, normalized, polygon, (left, upper, right, lower), writing_mode, top, clipped, clipped
    )


# 時計回りの四角形が退化せず凸であるかを判定する。
def _is_convex(points: list[Point]) -> bool:
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % 4]
        x3, y3 = points[(index + 2) % 4]
        if (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2) <= 0:
            return False
    return True


# 位置が重なる候補だけを統合し、別位置で反復する同一の台詞を残す。
def _merge_regions(candidates: list[_Region]) -> list[_Region]:
    # 切れていない候補を先に採用し、離れた位置の照合を避けるために縦帯で索引付けする。
    merged: list[_Region] = []
    vertical_index: defaultdict[int, set[int]] = defaultdict(set)
    for candidate in sorted(candidates, key=lambda region: region.clipped):
        _, top, _, bottom = candidate.bounds
        bands = range(int(top // _TILE_OVERLAP), int(bottom // _TILE_OVERLAP) + 1)
        nearby = set().union(*(vertical_index[band] for band in bands))
        match = next(
            (index for index in sorted(nearby) if _same_region(merged[index], candidate)), None
        )
        if match is None:
            index = len(merged)
            merged.append(candidate)
            for band in bands:
                vertical_index[band].add(index)
        elif not _compatible_text(merged[match], candidate):
            merged[match] = replace(merged[match], needs_review=True)
    return merged


# 文字列が似ていても、実際に重なる位置でなければ同じ領域と見なさない。
def _same_region(first: _Region, second: _Region) -> bool:
    if first.tile_top == second.tile_top:
        return False
    a_left, a_top, a_right, a_bottom = first.bounds
    b_left, b_top, b_right, b_bottom = second.bounds
    overlap_width = min(a_right, b_right) - max(a_left, b_left)
    overlap_height = min(a_bottom, b_bottom) - max(a_top, b_top)
    if overlap_width <= 0 or overlap_height <= 0:
        return False

    # 切断候補は包含を、異なる認識文は高い重なり率を要求する。
    overlap = overlap_width * overlap_height
    first_area = (a_right - a_left) * (a_bottom - a_top)
    second_area = (b_right - b_left) * (b_bottom - b_top)
    if _compatible_text(first, second):
        return overlap / min(first_area, second_area) >= 0.70
    return overlap / (first_area + second_area - overlap) >= 0.75


# OCR タイルによる部分認識を全文と照合する。
def _compatible_text(first: _Region, second: _Region) -> bool:
    return (
        first.normalized_text == second.normalized_text
        or (first.clipped and first.normalized_text in second.normalized_text)
        or (second.clipped and second.normalized_text in first.normalized_text)
    )


# 上下の重なりで行を作り、日本語ページでは同一行を右から読む。
def _reading_order(regions: list[_Region], language: ImageLanguage) -> list[_Region]:
    ordered: list[_Region] = []
    row: list[_Region] = []
    for region in sorted(regions, key=lambda item: (item.bounds[1], item.bounds[0])):
        if row:
            anchor = row[0]
            overlap = min(region.bounds[3], anchor.bounds[3]) - region.bounds[1]
            short_height = min(
                region.bounds[3] - region.bounds[1], anchor.bounds[3] - anchor.bounds[1]
            )
            if overlap < short_height * 0.35:
                ordered.extend(_sort_row(row, language))
                row = []
        row.append(region)
    ordered.extend(_sort_row(row, language))
    return ordered


# 同じ行にある領域を言語のページ既定方向で並べる。
def _sort_row(regions: list[_Region], language: ImageLanguage) -> list[_Region]:
    if language == "ja":
        return sorted(regions, key=lambda region: (-region.bounds[2], region.bounds[1]))
    return sorted(regions, key=lambda region: (region.bounds[0], region.bounds[1]))
