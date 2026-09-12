"""Server-owned evidence IDs. Membership proves a reference, NOT its truth.

Each string field/list entry is an immutable segment. No model-generated quote,
character offset, tokenizer or sentence reconstruction is needed for binding.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.features.language_learning.writing.generation_contract import WritingDraft


@dataclass(frozen=True)
class WritingSegment:
    id: str
    field: str
    text: str

    def payload(self) -> dict[str, str]:
        return {"id": self.id, "field": self.field, "text": self.text}


def writing_segments(draft: WritingDraft) -> tuple[WritingSegment, ...]:
    segments = [WritingSegment("O1", "originText", draft.origin_text)]
    for prefix, name, entries in (
        ("F", "providedFacts", draft.provided_facts),
        ("I", "requiredIntents", draft.required_intents),
        ("C", "responseConstraints", draft.response_constraints),
    ):
        segments.extend(WritingSegment(f"{prefix}{index}", name, text)
                        for index, text in enumerate(entries, 1))
    segments.append(WritingSegment("N1", "focusReason", draft.focus_reason))
    return tuple(segments)
