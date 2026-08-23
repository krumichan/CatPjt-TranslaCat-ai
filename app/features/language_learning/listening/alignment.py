from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.schemas.language_learning_listening import AlignmentEntry, AlignmentStatus


@dataclass(frozen=True)
class AlignmentSummary:
    entries: list[AlignmentEntry]
    reference_count: int
    answer_count: int
    recognized_count: int
    omission_count: int
    addition_count: int
    substitution_count: int
    order_count: int

    @property
    def recognition_ratio(self) -> float:
        denominator = max(self.reference_count, self.answer_count, 1)
        return self.recognized_count / denominator

    @property
    def reference_coverage(self) -> float:
        return self.recognized_count / max(self.reference_count, 1)


def align_tokens(
    reference: list[str],
    answer: list[str],
    *,
    accepted_tokens: set[str] | None = None,
) -> AlignmentSummary:
    accepted_tokens = accepted_tokens or set()
    rows = len(reference) + 1
    columns = len(answer) + 1
    costs = [[0] * columns for _ in range(rows)]
    operations = [[""] * columns for _ in range(rows)]

    for row in range(1, rows):
        costs[row][0] = row
        operations[row][0] = "OMISSION"
    for column in range(1, columns):
        costs[0][column] = column
        operations[0][column] = "ADDITION"

    for row in range(1, rows):
        for column in range(1, columns):
            is_match = reference[row - 1] == answer[column - 1]
            candidates = [
                (
                    costs[row - 1][column - 1] + (0 if is_match else 1),
                    "MATCH" if is_match else "SUBSTITUTION",
                ),
                (costs[row - 1][column] + 1, "OMISSION"),
                (costs[row][column - 1] + 1, "ADDITION"),
            ]
            costs[row][column], operations[row][column] = min(
                candidates,
                key=lambda item: (
                    item[0],
                    {"MATCH": 0, "SUBSTITUTION": 1, "OMISSION": 2, "ADDITION": 3}[
                        item[1]
                    ],
                ),
            )

    entries: list[AlignmentEntry] = []
    row = len(reference)
    column = len(answer)
    while row > 0 or column > 0:
        operation = operations[row][column]
        if operation in {"MATCH", "SUBSTITUTION"}:
            source = reference[row - 1]
            recognized = answer[column - 1]
            status = (
                AlignmentStatus.ACCEPTED_VARIANT
                if operation == "MATCH" and source in accepted_tokens
                else AlignmentStatus(operation)
            )
            entries.append(
                AlignmentEntry(
                    source=source,
                    answer=recognized,
                    status=status,
                    source_index=row - 1,
                    answer_index=column - 1,
                )
            )
            row -= 1
            column -= 1
        elif operation == "OMISSION":
            entries.append(
                AlignmentEntry(
                    source=reference[row - 1],
                    answer=None,
                    status=AlignmentStatus.OMISSION,
                    source_index=row - 1,
                )
            )
            row -= 1
        else:
            entries.append(
                AlignmentEntry(
                    source=None,
                    answer=answer[column - 1],
                    status=AlignmentStatus.ADDITION,
                    answer_index=column - 1,
                )
            )
            column -= 1
    entries.reverse()

    if reference != answer and Counter(reference) == Counter(answer):
        entries = [
            entry.model_copy(update={"status": AlignmentStatus.ORDER})
            if entry.status
            not in {AlignmentStatus.MATCH, AlignmentStatus.ACCEPTED_VARIANT}
            else entry
            for entry in entries
        ]

    recognized_statuses = {
        AlignmentStatus.MATCH,
        AlignmentStatus.ACCEPTED_VARIANT,
        AlignmentStatus.ORDER,
    }
    return AlignmentSummary(
        entries=entries,
        reference_count=len(reference),
        answer_count=len(answer),
        recognized_count=sum(entry.status in recognized_statuses for entry in entries),
        omission_count=sum(
            entry.status == AlignmentStatus.OMISSION for entry in entries
        ),
        addition_count=sum(
            entry.status == AlignmentStatus.ADDITION for entry in entries
        ),
        substitution_count=sum(
            entry.status == AlignmentStatus.SUBSTITUTION for entry in entries
        ),
        order_count=sum(entry.status == AlignmentStatus.ORDER for entry in entries),
    )


def attach_segment_timestamps(
    entries: list[AlignmentEntry],
    *,
    segment_ranges: list[tuple[int, int]],
) -> list[AlignmentEntry]:
    if not segment_ranges:
        return entries
    answer_entries = [entry for entry in entries if entry.answer_index is not None]
    if not answer_entries:
        return entries
    start_ms = min(start for start, _ in segment_ranges)
    end_ms = max(end for _, end in segment_ranges)
    span = max(end_ms - start_ms, 1)
    denominator = max(len(answer_entries), 1)
    timestamp_by_answer_index: dict[int, tuple[int, int]] = {}
    for position, entry in enumerate(answer_entries):
        if entry.answer_index is None:
            continue
        timestamp_by_answer_index[entry.answer_index] = (
            start_ms + int(span * position / denominator),
            start_ms + int(span * (position + 1) / denominator),
        )
    return [
        entry.model_copy(
            update={
                "start_ms": timestamp_by_answer_index[entry.answer_index][0],
                "end_ms": timestamp_by_answer_index[entry.answer_index][1],
            }
        )
        if entry.answer_index in timestamp_by_answer_index
        else entry
        for entry in entries
    ]
