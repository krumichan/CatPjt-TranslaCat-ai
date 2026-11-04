from __future__ import annotations

from calendar import monthrange
from datetime import date
from decimal import Decimal, localcontext
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AnalysisCoverage(AnalysisModel):
    analyzedTransactionCount: int = Field(ge=0, le=6000)
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
    periodType: Literal["MONTHLY", "QUARTERLY", "HALF_YEARLY", "YEARLY", "CUSTOM"]
    year: int = Field(ge=2000, le=9998)
    month: int | None = Field(ge=1, le=12)
    quarter: int | None = Field(default=None, ge=1, le=4)
    half: int | None = Field(default=None, ge=1, le=2)
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
    transactions: list[AnalysisTransaction] = Field(max_length=6000)
    historicalTransactions: list[AnalysisTransaction] = Field(default_factory=list, max_length=6000)
    previousAdvice: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        # 기간·샘플 범위·금액의 정합성은 모델 호출 전에 서버에서 확인한다.
        start = date(self.year, self.month or 1, 1)
        if self.periodType == "MONTHLY":
            if self.month is None:
                raise ValueError("Monthly analysis requires month")
            end = date(self.year + 1, 1, 1) if self.month == 12 else date(self.year, self.month + 1, 1)
        elif self.periodType == "YEARLY":
            if self.month is not None:
                raise ValueError("Yearly analysis must omit month")
            end = date(self.year + 1, 1, 1)
        elif self.periodType in ("QUARTERLY", "HALF_YEARLY"):
            index = self.quarter if self.periodType == "QUARTERLY" else self.half
            size = 3 if self.periodType == "QUARTERLY" else 6
            if index is None or self.month is not None:
                raise ValueError("Calendar grouping requires its quarter or half")
            start = date(self.year, (index - 1) * size + 1, 1)
            following = start.month + size
            end = date(self.year + (following - 1) // 12, (following - 1) % 12 + 1, 1)
        else:
            start, end = self.periodStart, self.periodEndExclusive
            next_month = start.month % 12 + 1
            next_year = start.year + (start.month == 12)
            minimum_end = date(next_year, next_month, min(start.day, monthrange(next_year, next_month)[1]))
            if self.year != start.year or self.month is not None or end < minimum_end:
                raise ValueError("Custom period must include at least one calendar month")
        if (self.periodType != "QUARTERLY" and self.quarter is not None) or (self.periodType != "HALF_YEARLY" and self.half is not None):
            raise ValueError("Unexpected calendar grouping")
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
        history_refs = [item.refId for item in self.historicalTransactions]
        if len(history_refs) != len(set(history_refs)) or set(history_refs).intersection(refs):
            raise ValueError("Historical and current references must be distinct")
        if any(item.date >= start for item in self.historicalTransactions):
            raise ValueError("Historical context must precede the selected period")
        if any(len(advice) > 4000 for advice in self.previousAdvice):
            raise ValueError("Previous advice exceeds its context budget")
        return self


class SpendingAnalysisFinding(AnalysisModel):
    title: str = Field(min_length=3, max_length=160)
    reason: str = Field(min_length=40, max_length=2000)
    action: str = Field(min_length=10, max_length=1200)
    contextQuestion: str | None = Field(max_length=500)
    uncertainty: str | None = Field(max_length=500)
    evidenceTransactionRefs: list[str] = Field(min_length=1, max_length=6000)
    opinion: str | None = Field(default=None, max_length=1200)
    historicalTransactionRefs: list[str] = Field(default_factory=list, max_length=6000)
    savings: "SavingsScenario | None" = None
    savingsTransactionRefs: list[str] = Field(default_factory=list, max_length=6000)


class SpendingAnalysisResponse(AnalysisModel):
    summary: str = Field(min_length=10, max_length=2000)
    findings: list[SpendingAnalysisFinding] = Field(max_length=8)
    policyVersion: str = "legacy-v2"
    savings: "SavingsScenario | None" = None


class SavingsScenario(AnalysisModel):
    basisAmount: str = "0"
    lower: str
    estimate: str
    upper: str
    grossReduction: str
    replacementCost: str
    reductionLowBps: int
    reductionHighBps: int
    replacementLowBps: int
    replacementHighBps: int
    kind: Literal["SMALLER_SELECTION", "DRINK_REPLACEMENT", "SKIP_EXTRA_ITEM", "COMBINED"]
    assumption: str


class CoachingFinding(AnalysisModel):
    title: str = Field(min_length=3, max_length=160)
    observation: str = Field(min_length=40, max_length=1400)
    opinion: str = Field(min_length=20, max_length=1200)
    action: str = Field(min_length=20, max_length=1200)
    assumption: str = Field(min_length=10, max_length=500)
    evidenceRefs: list[str] = Field(min_length=1, max_length=20)
    targetRefs: list[str] = Field(max_length=20)
    historicalRefs: list[str] = Field(max_length=30)
    estimateKind: Literal["NONE", "SMALLER_SELECTION", "DRINK_REPLACEMENT", "SKIP_EXTRA_ITEM"]


class CoachingDraft(AnalysisModel):
    policyVersion: Literal["purchase-context-v4"]
    summary: str = Field(min_length=50, max_length=2000)
    findings: list[CoachingFinding] = Field(max_length=2)


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
