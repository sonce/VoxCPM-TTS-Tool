from voxcpm_tts_tool.script_parser import (
    NON_LANG_TAG_MAP_ZH,
    ParsedSegment,
    localize_non_lang_tags,
    parse_script,
)


def _names(segments: list[ParsedSegment]) -> list[str | None]:
    return [s.voice_name for s in segments]


def test_no_switches_uses_default():
    segs, warns = parse_script("hello world", default_voice="alpha",
                               known_names={"alpha"})
    assert _names(segs) == ["alpha"]
    assert segs[0].text == "hello world"
    assert warns == []


def test_inline_switch_changes_voice():
    segs, _ = parse_script("hi <bob> there", default_voice="alpha",
                           known_names={"alpha", "bob"})
    assert _names(segs) == ["alpha", "bob"]
    assert segs[0].text == "hi "
    assert segs[1].text == " there"


def test_newline_resets_to_default():
    segs, _ = parse_script("<bob>line1\nline2", default_voice="alpha",
                           known_names={"alpha", "bob"})
    assert _names(segs) == ["bob", "alpha"]


def test_crlf_split():
    segs, _ = parse_script("a\r\nb\rc", default_voice="d",
                           known_names={"d"})
    assert [s.text for s in segs] == ["a", "b", "c"]


def test_empty_lines_skipped():
    segs, _ = parse_script("a\n\n   \nb", default_voice="d",
                           known_names={"d"})
    assert [s.text for s in segs] == ["a", "b"]


def test_case_insensitive_match():
    segs, _ = parse_script("<BOB>x", default_voice="a",
                           known_names={"a", "bob"})
    assert segs[0].voice_name == "bob"


def test_trim_whitespace_inside_brackets():
    segs, _ = parse_script("<  bob  >x", default_voice="a",
                           known_names={"a", "bob"})
    assert segs[0].voice_name == "bob"


def test_unknown_tag_preserved_with_warning():
    segs, warns = parse_script("hi <ghost> there", default_voice="a",
                               known_names={"a"})
    assert "<ghost>" in segs[0].text
    assert any("ghost" in w and "line 1" in w.lower() for w in warns)


def test_square_bracket_tags_preserved():
    segs, _ = parse_script("hello [laughing] world", default_voice="a",
                           known_names={"a"})
    assert "[laughing]" in segs[0].text


def test_cjk_voice_name():
    segs, _ = parse_script("<女声>大家好", default_voice="a",
                           known_names={"a", "女声"})
    assert segs[0].voice_name == "女声"
    assert segs[0].text == "大家好"


# ---- `<voice>(control)` script-level control instruction ----

def test_no_control_yields_none():
    segs, _ = parse_script("<bob>hello", default_voice="a",
                           known_names={"a", "bob"})
    assert segs[0].voice_name == "bob"
    assert segs[0].text == "hello"
    assert segs[0].control is None


def test_with_control_extracts_parens_content():
    segs, _ = parse_script("<bob>(温柔，低声)你好", default_voice="a",
                           known_names={"a", "bob"})
    assert segs[0].voice_name == "bob"
    assert segs[0].control == "温柔，低声"
    assert segs[0].text == "你好"


def test_default_voice_starts_with_no_control():
    segs, _ = parse_script("hello", default_voice="a", known_names={"a"})
    assert segs[0].voice_name == "a"
    assert segs[0].control is None


def test_each_switch_overrides_control_independently():
    segs, _ = parse_script(
        "<bob>(angry)hi <bob>(calm)bye",
        default_voice="a", known_names={"a", "bob"},
    )
    assert len(segs) == 2
    assert segs[0].control == "angry"
    assert segs[0].text == "hi "
    assert segs[1].control == "calm"
    assert segs[1].text == "bye"


def test_unknown_voice_tag_with_control_preserved_verbatim():
    segs, warns = parse_script("<ghost>(eerie)boo", default_voice="a",
                               known_names={"a"})
    assert "<ghost>(eerie)" in segs[0].text
    assert any("ghost" in w for w in warns)


def test_empty_control_is_empty_string_not_none():
    """`<bob>()hi` means user explicitly wrote empty control — distinct from no parens."""
    segs, _ = parse_script("<bob>()hi", default_voice="a",
                           known_names={"a", "bob"})
    assert segs[0].control == ""
    assert segs[0].text == "hi"


def test_localize_non_lang_tags_converts_zh_to_en():
    out = localize_non_lang_tags("hi[笑声]there[叹息]")
    assert out == "hi[laughing]there[sigh]"


def test_localize_non_lang_tags_passes_through_english():
    """Idempotent: English tokens already substituted stay as-is."""
    out = localize_non_lang_tags("hi[laughing]there")
    assert out == "hi[laughing]there"


def test_localize_non_lang_tags_leaves_unknown_alone():
    out = localize_non_lang_tags("hi[未知标签]bye")
    assert out == "hi[未知标签]bye"


def test_localize_non_lang_tags_handles_overlapping_keys():
    """`[嗯]` and `[疑问-嗯]` must not collide; longer key wins."""
    out = localize_non_lang_tags("a[嗯]b[疑问-嗯]c")
    assert out == "a[Uhm]b[Question-en]c"


def test_non_lang_tag_map_zh_covers_expected_set():
    assert "笑声" in NON_LANG_TAG_MAP_ZH
    assert NON_LANG_TAG_MAP_ZH["笑声"] == "laughing"
    assert NON_LANG_TAG_MAP_ZH["不满-哼"] == "Dissatisfaction-hnn"
