from app.core.config import Settings
from app.features.speech_to_text import FasterWhisperRuntime
from app.features.speech_to_text.runtime_policy import speaking_runtime


def test_speaking_selected_runtime_is_scoped_and_lazy():
    # 준비
    config = Settings.model_construct(
        AI_VOICE_STT_MODEL_NAME="base", AI_SPEAKING_STT_MODEL_NAME="small"
    )
    shared = FasterWhisperRuntime(
        model_name="base", model_revision="", device="cpu", compute_type="int8", cpu_threads=2
    )

    # 실행
    selected = speaking_runtime(shared, config)

    # 검증
    assert selected is not shared
    assert (shared.model_name, shared.cpu_threads) == ("base", 2)
    assert (selected.model_name, selected.cpu_threads) == ("small", 4)
    assert not shared.ready and not selected.ready
    assert shared._model is None and selected._model is None


def test_equal_runtime_policy_reuses_existing_model_without_duplicate_load():
    # 준비
    config = Settings.model_construct(
        AI_SPEAKING_STT_MODEL_NAME="base", AI_SPEAKING_STT_CPU_THREADS=2
    )
    shared = FasterWhisperRuntime(
        model_name="base", model_revision="", device="cpu", compute_type="int8", cpu_threads=2
    )

    # 실행
    selected = speaking_runtime(shared, config)

    # 검증
    assert selected is shared


def test_effective_di_preserves_other_features_runtime():
    # 준비: 업무 STT adapter가 제거돼도 기술 런타임 선택은 그대로 유지한다.
    from app.api import dependencies as d
    from app.api.internal.speech_transcription import get_speech_evidence_runtime

    # 실행 및 검증
    assert d._voice_stt_provider.runtime is d._speech_runtime
    assert d.get_speech_runtime() is d._speech_runtime
    assert d.get_speaking_speech_runtime() is d._speaking_speech_runtime
    assert get_speech_evidence_runtime().enabled
