import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.api.dependencies import get_receipt_analysis_service
from app.features.receipt.service import ReceiptAnalysisService
from app.schemas.receipt import ReceiptAnalysisOptions, ReceiptAnalysisResponse

router = APIRouter(
    prefix="/account-book/receipts",
    tags=["Receipt Analysis"],
)


@router.post("/analyze", response_model=ReceiptAnalysisResponse)
async def analyze_receipt(
    file: UploadFile = File(...),
    options: str | None = Form(None),
    service: ReceiptAnalysisService = Depends(get_receipt_analysis_service),
) -> ReceiptAnalysisResponse:
    parsed_options = _parse_options(options)
    return await service.analyze(file=file, options=parsed_options)


def _parse_options(options: str | None) -> ReceiptAnalysisOptions:
    if not options:
        return ReceiptAnalysisOptions()

    try:
        data = json.loads(options)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="options는 JSON 문자열이어야 합니다.",
        ) from exc

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail="options는 JSON object 형식이어야 합니다.",
        )

    try:
        return ReceiptAnalysisOptions.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=exc.errors(),
        ) from exc
