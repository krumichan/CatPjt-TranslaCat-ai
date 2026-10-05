import pytest

from app.ai.providers.openai.progressive_fields import ProgressiveFieldError, ProgressiveFields


def test_identifiable_blank_is_a_hole_and_later_valid_fields_survive():
    parser = ProgressiveFields(["s0", "s1", "s2"], recover_holes=True)
    assert parser.feed('{"s0":"앞","s1":" ","s2":"뒤"}') == [("s0", "앞"), ("s2", "뒤")]
    assert parser.drain_failures() == [("s1", "PROGRESSIVE_VALUE_BLANK")]
    parser.complete({"s0": "앞", "s1": " ", "s2": "뒤"})


def test_known_monotonic_missing_key_is_distinct_from_wrong_or_duplicate_key():
    parser = ProgressiveFields(["s0", "s1", "s2"], recover_holes=True)
    assert parser.feed('{"s0":"앞","s2":"뒤"}') == [("s0", "앞"), ("s2", "뒤")]
    assert parser.drain_failures() == [("s1", "PROGRESSIVE_KEY_MISSING")]
    parser.complete({"s0": "앞", "s2": "뒤"})


@pytest.mark.parametrize("tail", ['"s0":"중복"}', '"s9":"임의"}', '"s1":42}', '"s1":"뒤"x'])
def test_unsafe_boundary_stops_but_closed_prefix_in_same_delta_is_retained(tail):
    parser = ProgressiveFields(["s0", "s1"], recover_holes=True)
    with pytest.raises(ProgressiveFieldError):
        parser.feed('{"s0":"앞",' + tail)
    assert parser.drain_salvage() == [("s0", "앞")]


def test_final_missing_tail_is_identified_but_partial_string_is_never_guessed():
    parser = ProgressiveFields(["s0", "s1", "s2"], recover_holes=True)
    assert parser.feed('{"s0":"앞"}') == [("s0", "앞")]
    assert parser.drain_failures() == [
        ("s1", "PROGRESSIVE_KEY_MISSING"),
        ("s2", "PROGRESSIVE_KEY_MISSING"),
    ]
    parser.complete({"s0": "앞"})
    interrupted = ProgressiveFields(["s0", "s1"], recover_holes=True)
    assert interrupted.feed('{"s0":"앞","s1":"미완') == [("s0", "앞")]
    with pytest.raises(ProgressiveFieldError):
        interrupted.complete({"s0": "앞", "s1": "미완"})
