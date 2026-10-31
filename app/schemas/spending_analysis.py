from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AnalysisCoverage(AnalysisModel):
    analyzedTransactionCount: int = Field(ge=0, le=600)
    totalTransactionCount: int = Field(ge=0)
    isPartial: bool
    reason: str | None = Field(max_length=1000)


class PreviousAnalysisPeriod(AnalysisModel):
    periodStart: date
    periodEndExclusive: date
    totalExpense: str = Field(pattern=r"^\d{1,40}(\.\d{1,12})?$", max_length=54)
    transactionCount: int = Field(ge=0)
    isComparable: bool


class AnalysisTransaction(AnalysisModel):
    refId: str = Field(pattern=r"^tx-[1-9]\d{0,8}$")
    date: date
    title: str = Field(min_length=1, max_length=1000)
    storeName: str | None = Field(max_length=1000)
    category: str | None = Field(max_length=1000)
    memo: str | None = Field(max_length=20000)
    amount: str = Field(pattern=r"^\d{1,40}(\.\d{1,12})?$", max_length=54)
    sourceType: str | None = Field(default=None, max_length=80)

    @field_validator("amount")
    @classmethod
    def positive_expense(cls, value: str) -> str:
        if Decimal(value) <= 0:
            raise ValueError("Expense must be positive")
        return value


class SpendingAnalysisRequest(AnalysisModel):
    periodType: Literal["MONTHLY", "YEARLY"]
    year: int = Field(ge=2000, le=9998)
    month: int | None = Field(ge=1, le=12)
    locale: Literal["ko", "ja", "en"]
    currencyCode: str = Field(pattern=r"^[A-Z]{3}$")
    decimalPlaces: int = Field(ge=0, le=12)
    periodStart: date
    periodEndExclusive: date
    totalExpense: str = Field(pattern=r"^\d{1,40}(\.\d{1,12})?$", max_length=54)
    transactionCount: int = Field(ge=0)
    coverage: AnalysisCoverage
    previousPeriod: PreviousAnalysisPeriod | None
    comment: str | None = Field(max_length=2000)
    transactions: list[AnalysisTransaction] = Field(max_length=600)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        # 기간·샘플 범위·금액의 정합성은 모델 호출 전에 서버에서 확인한다.
        start = date(self.year, self.month or 1, 1)
        if self.periodType == "MONTHLY":
            if self.month is None:
                raise ValueError("Monthly analysis requires month")
            end = date(self.year + 1, 1, 1) if self.month == 12 else date(self.year, self.month + 1, 1)
        else:
            if self.month is not None:
                raise ValueError("Yearly analysis must omit month")
            end = date(self.year + 1, 1, 1)
        if self.periodStart != start or self.periodEndExclusive != end:
            raise ValueError("Snapshot dates must match the requested calendar period")

        refs = [item.refId for item in self.transactions]
        if len(set(refs)) != len(refs):
            raise ValueError("Transaction references must be unique")
        if any(not start <= item.date < end for item in self.transactions):
            raise ValueError("Evidence transaction lies outside the selected period")
        coverage = self.coverage
        if (
            coverage.analyzedTransactionCount != len(refs)
            or coverage.totalTransactionCount != self.transactionCount
            or len(refs) > self.transactionCount
            or coverage.isPartial != (len(refs) < self.transactionCount)
        ):
            raise ValueError("Analysis coverage does not match supplied transactions")
        if coverage.isPartial and not coverage.reason:
            raise ValueError("Partial analysis must explain its coverage")
        with localcontext() as context:
            context.prec = 80
            supplied = sum((Decimal(item.amount) for item in self.transactions), Decimal(0))
            total = Decimal(self.totalExpense)
        if supplied > total or (not coverage.isPartial and supplied != total):
            raise ValueError("Supplied expense amounts do not match the snapshot total")
        if self.transactionCount > 0 and not refs:
            raise ValueError("Nonempty expense periods require some evidence")
        if self.previousPeriod and self.previousPeriod.periodEndExclusive > start:
            raise ValueError("Previous period must precede the selected period")
        return self


class SpendingAnalysisFinding(AnalysisModel):
    title: str = Field(min_length=3, max_length=160)
    reason: str = Field(min_length=40, max_length=2000)
    action: str = Field(min_length=10, max_length=1200)
    contextQuestion: str | None = Field(max_length=500)
    uncertainty: str | None = Field(max_length=500)
    evidenceTransactionRefs: list[str] = Field(min_length=1, max_length=20)


class SpendingAnalysisResponse(AnalysisModel):
    summary: str = Field(min_length=10, max_length=2000)
    findings: list[SpendingAnalysisFinding] = Field(max_length=8)


class SpendingAnalysisCandidate(SpendingAnalysisFinding):
    reviewDisposition: Literal["REVIEW", "EXPLAINED_BY_COMMENT", "NECESSARY", "UNSUPPORTED"] = Field(
        description="REVIEW only for a concrete unresolved concern. A gift or necessary overtime meal explained by the comment is EXPLAINED_BY_COMMENT. Routine essential care/rent is NECESSARY. Mere generic checks are UNSUPPORTED."
    )
    reviewPattern: Literal["REPEAT_PURCHASE", "REPEATED_MEALS", "DUPLICATE_CHARGE", "ITEM_REVIEW"] = Field(
        description="Use DUPLICATE_CHARGE only for matching amount, date, title in at least two source transactions. ITEM_REVIEW still requires a specific supported concern, not a largest-category or routine check."
    )


class SpendingAnalysisDraft(AnalysisModel):
    summary: str = Field(min_length=10, max_length=2000)
    findings: list[SpendingAnalysisCandidate] = Field(max_length=8)
