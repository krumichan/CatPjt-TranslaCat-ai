"""Explicit feature-scoped runtime selection without duplicate equal models."""
from app.core.config import Settings
from app.features.speech_to_text.runtime import FasterWhisperRuntime


def speaking_runtime(shared: FasterWhisperRuntime, configuration: Settings) -> FasterWhisperRuntime:
    requested = (
        configuration.AI_SPEAKING_STT_MODEL_NAME,
        configuration.AI_SPEAKING_STT_MODEL_REVISION or None,
        configuration.AI_SPEAKING_STT_DEVICE,
        configuration.AI_SPEAKING_STT_COMPUTE_TYPE,
        configuration.AI_SPEAKING_STT_CPU_THREADS,
    )
    if requested == (shared.model_name, shared.model_revision, shared.device,
                     shared.compute_type, shared.cpu_threads):
        return shared
    return FasterWhisperRuntime(
        model_name=requested[0], model_revision=requested[1] or "",
        device=requested[2], compute_type=requested[3], cpu_threads=requested[4],
        num_workers=shared.num_workers, max_concurrency=shared.max_concurrency,
        queue_capacity=shared.queue_capacity, run_warm_up_inference=shared.run_warm_up_inference,
    )
