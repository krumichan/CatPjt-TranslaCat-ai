import hashlib

from app.features.language_learning.listening.audio_store import TemporaryListeningAudioStore


def test_duplicate_cache_key_preserves_matching_audio_and_metadata(tmp_path):
    store = TemporaryListeningAudioStore()
    store.base_dir = tmp_path
    first = store.put(
        b"first-audio", cache_key="same-reference", content_type="audio/wav",
        duration_seconds=1.0,
    )
    duplicate = store.put(
        b"different-audio", cache_key="same-reference", content_type="audio/mp3",
        duration_seconds=2.0,
    )

    assert duplicate == first
    assert duplicate.path.read_bytes() == b"first-audio"
    assert duplicate.checksum == hashlib.sha256(duplicate.path.read_bytes()).hexdigest()


def test_unindexed_audio_file_is_not_adopted_with_new_metadata(tmp_path):
    store = TemporaryListeningAudioStore()
    store.base_dir = tmp_path
    reference = store.reference_for("restart-reference")
    (tmp_path / f"{reference}.audio").write_bytes(b"stale-before-restart")

    stored = store.put(
        b"new-audio", cache_key="restart-reference", content_type="audio/wav",
        duration_seconds=3.0,
    )

    assert stored.path.read_bytes() == b"new-audio"
    assert stored.checksum == hashlib.sha256(stored.path.read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_audio_is_regenerated_with_consistent_metadata(tmp_path):
    store = TemporaryListeningAudioStore()
    store.base_dir = tmp_path
    first = store.put(
        b"missing-audio", cache_key="deleted-reference", content_type="audio/wav",
        duration_seconds=1.0,
    )
    first.path.unlink()
    replacement = store.put(
        b"replacement-audio", cache_key="deleted-reference", content_type="audio/wav",
        duration_seconds=2.0,
    )

    assert replacement.reference == first.reference
    assert replacement.path.read_bytes() == b"replacement-audio"
    assert replacement.checksum == hashlib.sha256(replacement.path.read_bytes()).hexdigest()
    assert replacement.duration_seconds == 2.0
