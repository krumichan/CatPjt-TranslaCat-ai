from __future__ import annotations

import math
import sys
from array import array
from collections import deque
from dataclasses import dataclass

from app.features.voice_translation.errors import VoicePipelineException
from app.schemas.voice_translation import VoiceAudioFormat, VoiceErrorCode, VoiceStage


@dataclass(frozen=True)
class PcmFrame:
    data: bytes
    duration_ms: int
    rms: float
    peak: float
    clipping_ratio: float
    is_speech: bool


@dataclass(frozen=True)
class TimedPcmFrame:
    frame: PcmFrame
    started_at_offset_ms: int
    ended_at_offset_ms: int


@dataclass(frozen=True)
class VadPartial:
    pcm_bytes: bytes
    started_at_offset_ms: int
    ended_at_offset_ms: int


@dataclass(frozen=True)
class VadUtterance:
    pcm_bytes: bytes
    started_at_offset_ms: int
    ended_at_offset_ms: int
    speech_duration_ms: int
    endpointing_ms: int
    forced_split: bool
    rms: float
    peak: float
    clipping_ratio: float


@dataclass(frozen=True)
class VadNoSpeech:
    started_at_offset_ms: int
    ended_at_offset_ms: int


@dataclass(frozen=True)
class VadProcessResult:
    speech_started_at_offset_ms: int | None = None
    partial: VadPartial | None = None
    utterance: VadUtterance | None = None
    no_speech: VadNoSpeech | None = None


class AudioFrameValidator:
    def __init__(
        self,
        audio_format: VoiceAudioFormat,
        *,
        minimum_frame_duration_ms: int,
        maximum_frame_duration_ms: int,
        speech_rms_threshold: float,
    ) -> None:
        self.audio_format = audio_format
        self.minimum_frame_duration_ms = minimum_frame_duration_ms
        self.maximum_frame_duration_ms = maximum_frame_duration_ms
        self.speech_rms_threshold = speech_rms_threshold

    def validate(self, data: bytes) -> PcmFrame:
        if not isinstance(data, bytes) or not data:
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_AUDIO_FRAME,
                stage=VoiceStage.AUDIO,
                message="비어 있지 않은 PCM Binary Frame이 필요합니다.",
                retryable=False,
            )
        if len(data) % 2 != 0:
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_AUDIO_FRAME,
                stage=VoiceStage.AUDIO,
                message="PCM_S16LE Frame의 Byte 길이는 짝수여야 합니다.",
                retryable=False,
            )

        bytes_per_ms = (
            self.audio_format.sample_rate * self.audio_format.channels * 2 // 1000
        )
        if len(data) % bytes_per_ms != 0:
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_AUDIO_FRAME,
                stage=VoiceStage.AUDIO,
                message="PCM Frame 길이가 Millisecond 경계와 일치하지 않습니다.",
                retryable=False,
            )
        duration_ms = len(data) // bytes_per_ms
        if not (
            self.minimum_frame_duration_ms
            <= duration_ms
            <= self.maximum_frame_duration_ms
        ):
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_AUDIO_FRAME,
                stage=VoiceStage.AUDIO,
                message=(
                    "PCM Frame 길이는 "
                    f"{self.minimum_frame_duration_ms}~"
                    f"{self.maximum_frame_duration_ms}ms 범위여야 합니다."
                ),
                retryable=False,
            )
        if duration_ms != self.audio_format.frame_duration_ms:
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_AUDIO_FRAME,
                stage=VoiceStage.AUDIO,
                message="PCM Frame 길이가 STREAM_OPEN의 frameDurationMs와 다릅니다.",
                retryable=False,
            )

        samples = array("h")
        samples.frombytes(data)
        if sys.byteorder != "little":
            samples.byteswap()
        sample_count = len(samples)
        square_sum = sum(float(sample) * float(sample) for sample in samples)
        rms = math.sqrt(square_sum / sample_count) / 32768.0
        peak = max(abs(sample) for sample in samples) / 32768.0
        clipping_count = sum(1 for sample in samples if abs(sample) >= 32760)
        clipping_ratio = clipping_count / sample_count

        return PcmFrame(
            data=data,
            duration_ms=duration_ms,
            rms=rms,
            peak=peak,
            clipping_ratio=clipping_ratio,
            is_speech=rms >= self.speech_rms_threshold,
        )


class VoiceActivityDetector:
    """Low-cost authoritative endpointing for normalized PCM input.

    The detector intentionally owns only speech boundaries. Whisper performs the
    final transcription and language detection after a boundary is confirmed.
    """

    def __init__(
        self,
        *,
        endpointing_silence_ms: int,
        minimum_utterance_ms: int,
        maximum_utterance_ms: int,
        start_evidence_ms: int,
        partial_interval_ms: int,
        pre_roll_ms: int,
        post_roll_ms: int,
        force_split_overlap_ms: int,
    ) -> None:
        self.endpointing_silence_ms = endpointing_silence_ms
        self.minimum_utterance_ms = minimum_utterance_ms
        self.maximum_utterance_ms = maximum_utterance_ms
        self.start_evidence_ms = start_evidence_ms
        self.partial_interval_ms = partial_interval_ms
        self.pre_roll_ms = pre_roll_ms
        self.post_roll_ms = post_roll_ms
        self.force_split_overlap_ms = force_split_overlap_ms

        self._offset_ms = 0
        self._pre_roll: deque[TimedPcmFrame] = deque()
        self._active_frames: list[TimedPcmFrame] = []
        self._active = False
        self._candidate_speech_ms = 0
        self._first_speech_offset_ms: int | None = None
        self._last_speech_offset_ms: int | None = None
        self._trailing_silence_ms = 0
        self._next_partial_at_ms = partial_interval_ms
        self._idle_started_at_ms: int | None = None

    @property
    def current_offset_ms(self) -> int:
        return self._offset_ms

    @property
    def active(self) -> bool:
        return self._active

    def process(self, frame: PcmFrame) -> VadProcessResult:
        timed = TimedPcmFrame(
            frame=frame,
            started_at_offset_ms=self._offset_ms,
            ended_at_offset_ms=self._offset_ms + frame.duration_ms,
        )
        self._offset_ms = timed.ended_at_offset_ms

        if not self._active:
            return self._process_idle(timed)
        return self._process_active(timed)

    def flush(self) -> VadProcessResult:
        if self._active:
            return self._finalize_active(forced_split=False)

        if (
            self._idle_started_at_ms is not None
            and self._offset_ms > self._idle_started_at_ms
        ):
            no_speech = VadNoSpeech(
                started_at_offset_ms=self._idle_started_at_ms,
                ended_at_offset_ms=self._offset_ms,
            )
            self._clear_idle_buffer()
            return VadProcessResult(no_speech=no_speech)
        return VadProcessResult()

    def clear(self) -> None:
        self._pre_roll.clear()
        self._active_frames.clear()
        self._active = False
        self._candidate_speech_ms = 0
        self._first_speech_offset_ms = None
        self._last_speech_offset_ms = None
        self._trailing_silence_ms = 0
        self._idle_started_at_ms = None

    def _process_idle(self, timed: TimedPcmFrame) -> VadProcessResult:
        if self._idle_started_at_ms is None:
            self._idle_started_at_ms = timed.started_at_offset_ms

        self._pre_roll.append(timed)
        self._trim_pre_roll(self.pre_roll_ms + self.start_evidence_ms + 200)

        if timed.frame.is_speech:
            self._candidate_speech_ms += timed.frame.duration_ms
        else:
            self._candidate_speech_ms = 0

        if self._candidate_speech_ms < self.start_evidence_ms:
            return VadProcessResult()

        selected = self._select_start_frames()
        self._active = True
        self._active_frames = selected
        self._pre_roll.clear()
        speech_frames = [item for item in selected if item.frame.is_speech]
        self._first_speech_offset_ms = speech_frames[0].started_at_offset_ms
        self._last_speech_offset_ms = speech_frames[-1].ended_at_offset_ms
        self._trailing_silence_ms = 0
        self._next_partial_at_ms = self.partial_interval_ms
        self._idle_started_at_ms = None

        return VadProcessResult(
            speech_started_at_offset_ms=selected[0].started_at_offset_ms,
        )

    def _process_active(self, timed: TimedPcmFrame) -> VadProcessResult:
        self._active_frames.append(timed)
        if timed.frame.is_speech:
            self._last_speech_offset_ms = timed.ended_at_offset_ms
            self._trailing_silence_ms = 0
        else:
            self._trailing_silence_ms += timed.frame.duration_ms

        assert self._first_speech_offset_ms is not None
        active_elapsed_ms = timed.ended_at_offset_ms - self._first_speech_offset_ms

        if active_elapsed_ms >= self.maximum_utterance_ms:
            return self._finalize_active(forced_split=True)

        partial: VadPartial | None = None
        if active_elapsed_ms >= self._next_partial_at_ms:
            partial = VadPartial(
                pcm_bytes=b"".join(item.frame.data for item in self._active_frames),
                started_at_offset_ms=self._active_frames[0].started_at_offset_ms,
                ended_at_offset_ms=timed.ended_at_offset_ms,
            )
            while active_elapsed_ms >= self._next_partial_at_ms:
                self._next_partial_at_ms += self.partial_interval_ms

        if self._trailing_silence_ms >= self.endpointing_silence_ms:
            finalized = self._finalize_active(forced_split=False)
            return VadProcessResult(
                partial=partial,
                utterance=finalized.utterance,
                no_speech=finalized.no_speech,
            )

        return VadProcessResult(partial=partial)

    def _finalize_active(self, *, forced_split: bool) -> VadProcessResult:
        assert self._active_frames
        assert self._first_speech_offset_ms is not None
        assert self._last_speech_offset_ms is not None

        speech_duration_ms = max(
            0,
            self._last_speech_offset_ms - self._first_speech_offset_ms,
        )
        if forced_split:
            ended_at_ms = self._active_frames[-1].ended_at_offset_ms
            endpointing_ms = 0
        else:
            ended_at_ms = self._last_speech_offset_ms + min(
                self._trailing_silence_ms,
                self.post_roll_ms,
            )
            endpointing_ms = self._trailing_silence_ms

        frames = self._frames_until(self._active_frames, ended_at_ms)
        started_at_ms = frames[0].started_at_offset_ms
        preserved = (
            self._tail_frames(frames, self.force_split_overlap_ms)
            if forced_split
            else []
        )

        if speech_duration_ms < self.minimum_utterance_ms:
            no_speech = VadNoSpeech(
                started_at_offset_ms=started_at_ms,
                ended_at_offset_ms=ended_at_ms,
            )
            self._reset_after_utterance(preserved)
            return VadProcessResult(no_speech=no_speech)

        sample_count = sum(len(item.frame.data) // 2 for item in frames)
        if sample_count:
            weighted_square = sum(
                (item.frame.rms**2) * (len(item.frame.data) // 2) for item in frames
            )
            rms = math.sqrt(weighted_square / sample_count)
            clipping_ratio = (
                sum(
                    item.frame.clipping_ratio * (len(item.frame.data) // 2)
                    for item in frames
                )
                / sample_count
            )
        else:
            rms = 0.0
            clipping_ratio = 0.0

        utterance = VadUtterance(
            pcm_bytes=b"".join(item.frame.data for item in frames),
            started_at_offset_ms=started_at_ms,
            ended_at_offset_ms=ended_at_ms,
            speech_duration_ms=speech_duration_ms,
            endpointing_ms=endpointing_ms,
            forced_split=forced_split,
            rms=rms,
            peak=max((item.frame.peak for item in frames), default=0.0),
            clipping_ratio=clipping_ratio,
        )
        self._reset_after_utterance(preserved)
        return VadProcessResult(utterance=utterance)

    def _reset_after_utterance(self, preserved: list[TimedPcmFrame]) -> None:
        self._active_frames.clear()
        self._active = False
        self._pre_roll = deque(preserved)
        self._candidate_speech_ms = sum(
            item.frame.duration_ms for item in preserved if item.frame.is_speech
        )
        self._first_speech_offset_ms = None
        self._last_speech_offset_ms = None
        self._trailing_silence_ms = 0
        self._next_partial_at_ms = self.partial_interval_ms
        self._idle_started_at_ms = self._offset_ms

    def _clear_idle_buffer(self) -> None:
        self._pre_roll.clear()
        self._candidate_speech_ms = 0
        self._idle_started_at_ms = None

    def _select_start_frames(self) -> list[TimedPcmFrame]:
        items = list(self._pre_roll)
        first_speech_index = next(
            index for index, item in enumerate(items) if item.frame.is_speech
        )
        prefix: list[TimedPcmFrame] = []
        kept_ms = 0
        for index in range(first_speech_index - 1, -1, -1):
            item = items[index]
            remaining_ms = self.pre_roll_ms - kept_ms
            if remaining_ms <= 0:
                break
            if item.frame.duration_ms <= remaining_ms:
                prefix.append(item)
                kept_ms += item.frame.duration_ms
                continue

            bytes_per_ms = len(item.frame.data) // item.frame.duration_ms
            prefix.append(
                TimedPcmFrame(
                    frame=PcmFrame(
                        data=item.frame.data[-remaining_ms * bytes_per_ms :],
                        duration_ms=remaining_ms,
                        rms=item.frame.rms,
                        peak=item.frame.peak,
                        clipping_ratio=item.frame.clipping_ratio,
                        is_speech=item.frame.is_speech,
                    ),
                    started_at_offset_ms=item.ended_at_offset_ms - remaining_ms,
                    ended_at_offset_ms=item.ended_at_offset_ms,
                )
            )
            break
        prefix.reverse()
        return prefix + items[first_speech_index:]

    def _trim_pre_roll(self, maximum_ms: int) -> None:
        duration_ms = sum(item.frame.duration_ms for item in self._pre_roll)
        while self._pre_roll and duration_ms > maximum_ms:
            duration_ms -= self._pre_roll.popleft().frame.duration_ms

    @staticmethod
    def _frames_until(
        frames: list[TimedPcmFrame],
        ended_at_ms: int,
    ) -> list[TimedPcmFrame]:
        selected: list[TimedPcmFrame] = []
        for item in frames:
            if item.started_at_offset_ms >= ended_at_ms:
                break
            if item.ended_at_offset_ms <= ended_at_ms:
                selected.append(item)
                continue

            kept_duration_ms = ended_at_ms - item.started_at_offset_ms
            bytes_per_ms = len(item.frame.data) // item.frame.duration_ms
            kept_data = item.frame.data[: kept_duration_ms * bytes_per_ms]
            selected.append(
                TimedPcmFrame(
                    frame=PcmFrame(
                        data=kept_data,
                        duration_ms=kept_duration_ms,
                        rms=item.frame.rms,
                        peak=item.frame.peak,
                        clipping_ratio=item.frame.clipping_ratio,
                        is_speech=item.frame.is_speech,
                    ),
                    started_at_offset_ms=item.started_at_offset_ms,
                    ended_at_offset_ms=ended_at_ms,
                )
            )
            break
        return selected

    @staticmethod
    def _tail_frames(
        frames: list[TimedPcmFrame],
        maximum_ms: int,
    ) -> list[TimedPcmFrame]:
        selected: list[TimedPcmFrame] = []
        duration_ms = 0
        for item in reversed(frames):
            remaining_ms = maximum_ms - duration_ms
            if remaining_ms <= 0:
                break
            if item.frame.duration_ms > remaining_ms:
                bytes_per_ms = len(item.frame.data) // item.frame.duration_ms
                kept_data = item.frame.data[-remaining_ms * bytes_per_ms :]
                selected.append(
                    TimedPcmFrame(
                        frame=PcmFrame(
                            data=kept_data,
                            duration_ms=remaining_ms,
                            rms=item.frame.rms,
                            peak=item.frame.peak,
                            clipping_ratio=item.frame.clipping_ratio,
                            is_speech=item.frame.is_speech,
                        ),
                        started_at_offset_ms=item.ended_at_offset_ms - remaining_ms,
                        ended_at_offset_ms=item.ended_at_offset_ms,
                    )
                )
                duration_ms += remaining_ms
                break
            selected.append(item)
            duration_ms += item.frame.duration_ms
        selected.reverse()
        return selected
