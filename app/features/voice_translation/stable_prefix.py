from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StableTranscriptUpdate:
    text: str
    stable_prefix: str
    changed: bool
    revision: int


class StablePrefixAssembler:
    """Tracks the common prefix of consecutive rolling-window STT results."""

    def __init__(self) -> None:
        self._previous = ""
        self._stable_prefix = ""
        self._revision = 0

    @property
    def stable_prefix(self) -> str:
        return self._stable_prefix

    @property
    def revision(self) -> int:
        return self._revision

    def update(self, text: str) -> StableTranscriptUpdate:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return StableTranscriptUpdate(
                text="",
                stable_prefix=self._stable_prefix,
                changed=False,
                revision=self._revision,
            )

        if self._previous:
            common = _longest_common_prefix(self._previous, normalized)
            if len(common) >= len(self._stable_prefix):
                self._stable_prefix = common

        changed = normalized != self._previous
        if changed:
            self._revision += 1
            self._previous = normalized

        return StableTranscriptUpdate(
            text=normalized,
            stable_prefix=self._stable_prefix,
            changed=changed,
            revision=self._revision,
        )

    def clear(self) -> None:
        self._previous = ""
        self._stable_prefix = ""
        self._revision = 0


def _longest_common_prefix(left: str, right: str) -> str:
    maximum = min(len(left), len(right))
    index = 0
    while index < maximum and left[index] == right[index]:
        index += 1
    prefix = left[:index]

    # Space-delimited languages avoid promoting a half-decoded word. CJK text
    # normally has no spaces, so its character prefix remains useful.
    if " " in left[: index + 1] or " " in right[: index + 1]:
        boundary = prefix.rfind(" ")
        if boundary >= 0 and index < len(left) and index < len(right):
            return prefix[: boundary + 1]
    return prefix
