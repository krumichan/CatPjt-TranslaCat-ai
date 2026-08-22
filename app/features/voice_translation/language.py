from __future__ import annotations

import math
from dataclasses import dataclass

from app.features.voice_translation.errors import VoicePipelineException
from app.schemas.voice_translation import (
    SUPPORTED_VOICE_LANGUAGES,
    VoiceErrorCode,
    VoiceStage,
)


@dataclass(frozen=True)
class VoiceLanguageObservation:
    detected_language: str | None
    confidence: float | None
    locked_language: str | None


class LanguageStateManager:
    """Maintains one representative input language per channel connection."""

    def __init__(
        self,
        *,
        lock_confidence: float,
        switch_confidence: float,
        switch_consecutive_count: int,
        manual_language: str | None = None,
        initial_locked_language: str | None = None,
        minimum_detection_confidence: float = 0.50,
    ) -> None:
        self.lock_confidence = lock_confidence
        self.switch_confidence = switch_confidence
        self.switch_consecutive_count = switch_consecutive_count
        self.manual_language = normalize_voice_language(manual_language)
        self.minimum_detection_confidence = minimum_detection_confidence
        self._locked_language: str | None = (
            self.manual_language or normalize_voice_language(initial_locked_language)
        )
        self._switch_candidate: str | None = None
        self._switch_count = 0

    @property
    def locked_language(self) -> str | None:
        return self.manual_language or self._locked_language

    def observe(
        self,
        language: str | None,
        confidence: float | None,
    ) -> VoiceLanguageObservation:
        detected = normalize_voice_language(language)
        normalized_confidence = normalize_confidence(confidence)
        if detected is None:
            normalized_confidence = None

        if self.manual_language is not None:
            return VoiceLanguageObservation(
                detected_language=detected,
                confidence=normalized_confidence,
                locked_language=self.manual_language,
            )

        # Unknown/no-evidence observations do not participate in hysteresis.
        if detected is None or normalized_confidence is None:
            return VoiceLanguageObservation(
                detected_language=detected,
                confidence=normalized_confidence,
                locked_language=self._locked_language,
            )

        if self._locked_language is None:
            if normalized_confidence >= self.lock_confidence:
                self._locked_language = detected
            return VoiceLanguageObservation(
                detected_language=detected,
                confidence=normalized_confidence,
                locked_language=self._locked_language,
            )

        if detected == self._locked_language:
            self._reset_switch_candidate()
        elif normalized_confidence >= self.switch_confidence:
            if detected == self._switch_candidate:
                self._switch_count += 1
            else:
                self._switch_candidate = detected
                self._switch_count = 1

            if self._switch_count >= self.switch_consecutive_count:
                self._locked_language = detected
                self._reset_switch_candidate()
        else:
            self._reset_switch_candidate()

        return VoiceLanguageObservation(
            detected_language=detected,
            confidence=normalized_confidence,
            locked_language=self._locked_language,
        )

    def resolve_translation_source(
        self,
        *,
        detected_language: str | None,
        confidence: float | None,
    ) -> str:
        if self.manual_language is not None:
            return self.manual_language

        detected = normalize_voice_language(detected_language)
        normalized_confidence = normalize_confidence(confidence)
        if (
            detected is not None
            and normalized_confidence is not None
            and normalized_confidence >= self.minimum_detection_confidence
        ):
            return detected

        if self._locked_language is not None:
            return self._locked_language

        raise VoicePipelineException(
            code=VoiceErrorCode.LANGUAGE_UNDETERMINED,
            stage=VoiceStage.LANGUAGE,
            message="입력 언어를 신뢰할 수 있는 수준으로 확인하지 못했습니다.",
            retryable=False,
        )

    def _reset_switch_candidate(self) -> None:
        self._switch_candidate = None
        self._switch_count = 0


def normalize_voice_language(language: str | None) -> str | None:
    if not language:
        return None
    normalized = language.strip().lower().split("-")[0]
    if normalized == "und" or normalized not in SUPPORTED_VOICE_LANGUAGES:
        return None
    return normalized


def normalize_confidence(confidence: float | None) -> float | None:
    if confidence is None:
        return None
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return max(0.0, min(1.0, value))
