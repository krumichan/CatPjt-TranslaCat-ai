import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.api.dependencies import get_receipt_analysis_service
from app.features.receipt.service import ReceiptAnalysisService
from app.features.receipt.runtime_identity import get_receipt_runtime_identity
from app.schemas.receipt import ReceiptAnalysisOptions, ReceiptAnalysisResponse, ReceiptRuntimeIdentity

router = APIRouter(
    prefix="/account-book/receipts",
    tags=["Receipt Analysis"],
)


@router.get("/runtime-identity", response_model=ReceiptRuntimeIdentity)
async def receipt_runtime_identity() -> ReceiptRuntimeIdentity:
    return get_receipt_runtime_identity()


@router.post("/analyze", response_model=ReceiptAnalysisResponse)
async def analyze_receipt(
    file: UploadFile = File(...),
    options: str | None = Form(None),
    trace_id: str | None = Form(None),
    service: ReceiptAnalysisService = Depends(get_receipt_analysis_service),
) -> ReceiptAnalysisResponse:
    parsed_options = _parse_options(options)
    return await service.analyze(
        file=file,
        options=parsed_options,
        trace_id=trace_id,
    )


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
