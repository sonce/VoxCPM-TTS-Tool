import numpy as np

from voxcpm_tts_tool.long_text import concat_waveforms, split_for_generation

CHAR_BUDGET_CJK = 80


def test_short_input_returns_one_chunk():
    out = split_for_generation("你好", char_budget=CHAR_BUDGET_CJK)
    assert out == ["你好"]


def test_splits_on_cjk_period():
    out = split_for_generation("第一句。第二句。第三句。",
                               char_budget=CHAR_BUDGET_CJK)
    assert out == ["第一句。", "第二句。", "第三句。"]


def test_splits_on_ascii_period_and_question_mark():
    out = split_for_generation("First. Second? Third!",
                               char_budget=200)
    assert out == ["First.", " Second?", " Third!"]


def test_falls_back_to_comma_when_sentence_too_long():
    long_sentence = "一段没有句号但很长的话，再来一截，又来一段，最后一截"
    out = split_for_generation(long_sentence, char_budget=10)
    # Each chunk must be <= budget OR a single comma-bounded segment.
    assert all("。" not in c for c in out)
    assert len(out) > 1


def test_square_bracket_tag_indivisible():
    # Budget is small but [laughing] cannot be split.
    out = split_for_generation("ha [laughing] ha", char_budget=4)
    joined = "".join(out)
    assert "[laughing]" in joined
    assert all(c == "" or "[laughing]" in c or "[" not in c for c in out)


def test_empty_input_yields_no_chunks():
    assert split_for_generation("", char_budget=80) == []


def test_concat_preserves_order():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0], dtype=np.float32)
    c = np.array([4.0, 5.0], dtype=np.float32)
    out = concat_waveforms([a, b, c])
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_concat_empty_list_returns_empty_array():
    out = concat_waveforms([])
    assert out.shape == (0,)
    assert out.dtype == np.float32
