from app.core.config import Settings
from app.features.speech_to_text import FasterWhisperRuntime
from app.features.speech_to_text.runtime_policy import speaking_runtime
from app.features.language_learning.speaking.stt_service import FasterWhisperSpeakingSttProvider


def test_speaking_selected_runtime_is_scoped_and_lazy():
    config = Settings.model_construct(AI_VOICE_STT_MODEL_NAME="base", AI_SPEAKING_STT_MODEL_NAME="small")
    shared = FasterWhisperRuntime(model_name="base", model_revision="", device="cpu", compute_type="int8", cpu_threads=2)
    selected = speaking_runtime(shared, config)
    assert selected is not shared
    assert (shared.model_name, shared.cpu_threads) == ("base", 2)
    assert (selected.model_name, selected.cpu_threads) == ("small", 4)
    assert not shared.ready and not selected.ready
    assert shared._model is None and selected._model is None


def test_equal_runtime_policy_reuses_existing_model_without_duplicate_load():
    config = Settings.model_construct(AI_SPEAKING_STT_MODEL_NAME="base", AI_SPEAKING_STT_CPU_THREADS=2)
    shared = FasterWhisperRuntime(model_name="base", model_revision="", device="cpu", compute_type="int8", cpu_threads=2)
    assert speaking_runtime(shared, config) is shared


def test_effective_di_preserves_other_features_runtime():
    from app.api import dependencies as d
    assert d._listening_stt_provider.runtime is d._speech_runtime
    assert d._voice_stt_provider.runtime is d._speech_runtime
    assert isinstance(d._level_test_stt_service.provider, FasterWhisperSpeakingSttProvider)
    assert d._level_test_stt_service.provider.runtime is d._speech_runtime
    assert d._speaking_stt_provider.runtime is d._speaking_speech_runtime
    assert d._speaking_stt_provider.speech_guard is d._speaking_speech_evidence_guard
    assert d._speaking_speech_evidence_guard.enabled
