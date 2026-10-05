"""Responses text deltas에서 완전히 닫힌 keyed JSON 문자열만 추출한다."""

from __future__ import annotations

import json


class ProgressiveFieldError(ValueError):
    pass


class ProgressiveFields:
    def __init__(
        self, expected_keys: list[str], max_chars: int = 1_500_000, *, recover_holes: bool = False
    ):
        if not expected_keys or len(set(expected_keys)) != len(expected_keys):
            raise ProgressiveFieldError("PROGRESSIVE_KEYS_INVALID")
        self.expected_keys = expected_keys
        self.max_chars = max_chars
        self.buffer = ""
        self.position = 0
        self.index = 0
        self.state = "open"
        self.pending: tuple[str, str] | None = None
        self.values: dict[str, str] = {}
        self.recover_holes = recover_holes
        self.failures: list[tuple[str, str]] = []
        self.salvage: list[tuple[str, str]] = []
        self.decoder = json.JSONDecoder()

    def feed(self, delta: str) -> list[tuple[str, str]]:
        self.salvage = []
        self.buffer += delta
        if len(self.buffer) > self.max_chars:
            raise ProgressiveFieldError("PROGRESSIVE_BODY_TOO_LARGE")
        emitted: list[tuple[str, str]] = []
        while True:
            self._space()
            if self.state == "open":
                if not self._take("{"):
                    break
                self.state = "key"
            elif self.state == "key":
                if (
                    self.recover_holes
                    and self.position < len(self.buffer)
                    and self.buffer[self.position] == "}"
                ):
                    self.position += 1
                    self._missing_until(len(self.expected_keys))
                    self.state = "done"
                    continue
                if self.index == len(self.expected_keys):
                    if not self._take("}"):
                        break
                    self.state = "done"
                    continue
                parsed = self._string()
                if parsed is None:
                    break
                key, self.position = parsed
                if key != self.expected_keys[self.index]:
                    # 알려진 단조 증가 ID의 누락만 격리한다.
                    # 중복·역순·임의 ID는 경계를 신뢰하지 않는다.
                    if not self.recover_holes or key not in self.expected_keys[self.index + 1 :]:
                        raise ProgressiveFieldError("PROGRESSIVE_KEY_ORDER_INVALID")
                    self._missing_until(self.expected_keys.index(key))
                self.state = "colon"
            elif self.state == "colon":
                if not self._take(":"):
                    break
                self.state = "value"
            elif self.state == "value":
                parsed = self._string()
                if parsed is None:
                    break
                value, self.position = parsed
                if not value.strip() and not self.recover_holes:
                    raise ProgressiveFieldError("PROGRESSIVE_VALUE_BLANK")
                self.pending = (self.expected_keys[self.index], value)
                self.state = "delimiter"
            elif self.state == "delimiter":
                self._space()
                if self.position >= len(self.buffer):
                    break
                expected = "," if self.index + 1 < len(self.expected_keys) else "}"
                if self.recover_holes and self.buffer[self.position] == "}":
                    expected = "}"
                if self.buffer[self.position] != expected:
                    raise ProgressiveFieldError("PROGRESSIVE_DELIMITER_INVALID")
                self.position += 1
                key, value = self.pending  # type: ignore[misc]
                self.values[key] = value
                if value.strip():
                    emitted.append((key, value))
                    self.salvage.append((key, value))
                else:
                    self.failures.append((key, "PROGRESSIVE_VALUE_BLANK"))
                self.pending = None
                self.index += 1
                self.state = "key" if expected == "," else "done"
                if expected == "}" and self.recover_holes:
                    self._missing_until(len(self.expected_keys))
            else:
                if self.position < len(self.buffer):
                    raise ProgressiveFieldError("PROGRESSIVE_TRAILING_DATA")
                break
        self.salvage = []
        return emitted

    def drain_salvage(self) -> list[tuple[str, str]]:
        # 같은 SDK delta 안에서 오류 경계보다 앞서 완전히 닫힌 필드만 보존한다.
        values, self.salvage = self.salvage, []
        return values

    def _missing_until(self, index: int) -> None:
        while self.index < index:
            self.failures.append((self.expected_keys[self.index], "PROGRESSIVE_KEY_MISSING"))
            self.index += 1

    def drain_failures(self) -> list[tuple[str, str]]:
        failures, self.failures = self.failures, []
        return failures

    def complete(self, output: object) -> None:
        self._space()
        if self.state != "done" or self.position != len(self.buffer) or output != self.values:
            raise ProgressiveFieldError("PROGRESSIVE_FINAL_MISMATCH")

    def _space(self) -> None:
        while self.position < len(self.buffer) and self.buffer[self.position] in " \t\r\n":
            self.position += 1

    def _take(self, char: str) -> bool:
        if self.position >= len(self.buffer):
            return False
        if self.buffer[self.position] != char:
            raise ProgressiveFieldError("PROGRESSIVE_JSON_INVALID")
        self.position += 1
        return True

    def _string(self) -> tuple[str, int] | None:
        if self.position >= len(self.buffer):
            return None
        if self.buffer[self.position] != '"':
            raise ProgressiveFieldError("PROGRESSIVE_STRING_REQUIRED")
        try:
            value, end = self.decoder.raw_decode(self.buffer, self.position)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, str):
            raise ProgressiveFieldError("PROGRESSIVE_STRING_REQUIRED")
        return value, end
