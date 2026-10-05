import pytest

from app.ai.providers.openai.progressive_fields import ProgressiveFieldError, ProgressiveFields


def test_split_escapes_and_closed_delimiter_only():
    parser = ProgressiveFields(["s0", "s1"])
    assert parser.feed('{"s0":"가\\') == []
    assert parser.feed('"나"') == []
    assert parser.feed(",") == [("s0", '가"나')]
    assert parser.feed('"s1":"\ud55c') == []
    assert parser.feed('글"}') == [("s1", "한글")]
    parser.complete({"s0": '가"나', "s1": "한글"})


@pytest.mark.parametrize(
    "body",
    [
        '{"s1":"뒤","s0":"앞"}',
        '{"s0":"앞","s0":"다시"}',
        '{"s0":"앞","extra":"값"}',
        '{"s0":" "}',
        '{"s0":"앞"}',
    ],
)
def test_invalid_or_missing_fields(body):
    parser = ProgressiveFields(["s0", "s1"])
    with pytest.raises(ProgressiveFieldError):
        parser.feed(body)
        parser.complete({"s0": "앞", "s1": "뒤"})


def test_late_final_mismatch_is_not_success():
    parser = ProgressiveFields(["s0"])
    assert parser.feed('{"s0":"보임"}') == [("s0", "보임")]
    with pytest.raises(ProgressiveFieldError, match="FINAL_MISMATCH"):
        parser.complete({"s0": "바뀜"})
