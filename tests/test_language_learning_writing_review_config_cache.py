"""Actual config manager control flow with SDK config *builders* doubled.

The google-genai SDK is not required; these tests do not assert SDK serialization.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

TASKS = (
    "LANGUAGE_LEARNING_WRITING_DIFFICULTY_PRESCREEN",
    "LANGUAGE_LEARNING_WRITING_TASK_VERIFICATION",
    "LANGUAGE_LEARNING_WRITING_DIFFICULTY_VERIFICATION",
    "LANGUAGE_LEARNING_WRITING_NOTE_VERIFICATION",
    "LANGUAGE_LEARNING_WRITING_NOTE_LOCALIZATION",
    "LANGUAGE_LEARNING_WRITING_SOURCE_LOCALIZATION",
)


def load_manager(monkeypatch):
    configs = ModuleType("app.ai.providers.gemini.configs")
    names = (
        "chat_ai_reply", "default_gemini", "fast_translation",
        "language_learning_choice_verification", "language_learning_evaluation",
        "language_learning_generation", "language_learning_task_verification",
        "language_learning_writing_verification", "language_learning_vocab_design",
        "language_learning_vocab_repair", "voice_translation",
    )
    for name in names:
        setattr(configs, f"build_{name}_config", lambda **kwargs: copy.deepcopy(kwargs))
    monkeypatch.setitem(sys.modules, configs.__name__, configs)
    path = Path(__file__).parents[1] / "app/ai/providers/gemini/config_manager.py"
    spec = importlib.util.spec_from_file_location("_writing_config_manager_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GeminiConfigManager()


@pytest.mark.parametrize("task", TASKS)
def test_candidate_bound_schemas_are_not_retained_in_shared_cache(task, monkeypatch):
    manager = load_manager(monkeypatch)
    returned = []
    for i in range(100):
        schema = {"type": "object", "properties": {"candidateId": {"type": "string", "enum": [str(i)]}}}
        returned.append(manager.get_cached_config(task, schema))
    assert manager._config_cache == {}
    assert len({id(item) for item in returned}) == 100
    assert [item["schema"]["properties"]["candidateId"]["enum"] for item in returned] == [[str(i)] for i in range(100)]


def test_non_writing_configs_keep_existing_cache_semantics(monkeypatch):
    manager = load_manager(monkeypatch)
    first = manager.get_cached_config("LANGUAGE_LEARNING_WRITING_EVALUATION", {"type": "object"})
    second = manager.get_cached_config("LANGUAGE_LEARNING_WRITING_EVALUATION", {"type": "object"})
    assert first is second
    assert len(manager._config_cache) == 1
