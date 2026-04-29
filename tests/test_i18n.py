from voxcpm_tts_tool import i18n


def test_returns_zh_when_lang_is_zh():
    assert i18n.t("tab.generation", "zh") == "语音生成"


def test_returns_en_when_lang_is_en():
    assert i18n.t("tab.generation", "en") == "Speech Generation"


def test_falls_back_to_zh_when_en_missing(monkeypatch):
    monkeypatch.setitem(i18n.STRINGS, "only.zh", {"zh": "中"})
    assert i18n.t("only.zh", "en") == "中"


def test_falls_back_to_key_when_both_missing():
    assert i18n.t("definitely.missing.key", "en") == "definitely.missing.key"


def test_chinese_dictionary_covers_every_key():
    missing = [k for k, v in i18n.STRINGS.items() if "zh" not in v]
    assert missing == [], f"keys missing zh translation: {missing}"
