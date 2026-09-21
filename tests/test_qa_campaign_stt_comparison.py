from scripts.qa_campaign_stt_comparison import edit_distance, normalized_text


def test_comparison_normalization_does_not_hide_negation_or_script_differences():
    assert normalized_text("Ａ、 B。") == "AB"
    assert edit_distance("停止する", "停止しない") > 0
    assert edit_distance("追加", "ついか") > 0
    assert edit_distance("", "音声") == 2
    assert edit_distance("同じ", "同じ") == 0
