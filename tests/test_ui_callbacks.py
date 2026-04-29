from voxcpm_tts_tool.ui_callbacks import (
    effective_mode,
    field_visibility,
    insert_tag,
    insert_voice_tag,
    voice_dropdown_choices,
)
from voxcpm_tts_tool.app_state import EPHEMERAL_DEFAULT_VOICE_ID, ephemeral_default_voice
from voxcpm_tts_tool.voice_library import Voice


def _design(name: str = "x") -> Voice:
    return Voice(id="i" + name, name=name, mode="design")


def test_design_visibility():
    vis = field_visibility("design")
    assert vis == {
        "voice_name": True,
        "control": True,
        "reference_audio": False,
        "prompt_text": False,
        "denoise": False,
        "normalize": True,
        "seed_text": True,
    }


def test_clone_shows_control_prompt_seed_text():
    """Controllable Cloning: shows control + transcript + seed_text + denoise."""
    vis = field_visibility("clone")
    assert vis["reference_audio"] is True
    assert vis["denoise"] is True
    assert vis["prompt_text"] is True
    assert vis["control"] is True
    assert vis["seed_text"] is True   # all modes now use seed_text
    assert vis["normalize"] is True


def test_hifi_hides_control_shows_prompt_seed():
    """Ultimate Cloning: hides control (SDK ignores it), shows transcript + seed_text."""
    vis = field_visibility("hifi")
    assert vis["control"] is False
    assert vis["prompt_text"] is True
    assert vis["seed_text"] is True
    assert vis["denoise"] is True
    assert vis["normalize"] is True


def test_seed_text_visible_in_all_modes():
    for mode in ("design", "clone", "hifi"):
        assert field_visibility(mode)["seed_text"] is True


def test_effective_mode_design():
    assert effective_mode("design", "controllable") == "design"
    assert effective_mode("design", "ultimate") == "design"  # sub ignored


def test_effective_mode_cloning_to_clone_or_hifi():
    assert effective_mode("cloning", "controllable") == "clone"
    assert effective_mode("cloning", "ultimate") == "hifi"


def test_effective_mode_unknown_top_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown top mode"):
        effective_mode("bogus", "controllable")


def test_normalize_visible_in_all_modes():
    for mode in ("design", "clone", "hifi"):
        assert field_visibility(mode)["normalize"] is True


def test_transcribe_button_no_longer_in_visibility_map():
    """The explicit Transcribe button was replaced by auto-fire on upload."""
    for mode in ("design", "clone", "hifi"):
        assert "transcribe_button" not in field_visibility(mode)


def test_voice_dropdown_includes_real_voices():
    voices = [_design("alpha"), _design("beta")]
    choices = voice_dropdown_choices(voices, lang="zh", ephemeral=ephemeral_default_voice())
    # Returns list of (label, id) tuples; ephemeral first only when no real voices.
    assert ("alpha", "ialpha") in choices
    assert ("beta", "ibeta") in choices
    assert all(label != "__default__" for label, _ in choices)


def test_voice_dropdown_shows_ephemeral_when_no_voices():
    choices = voice_dropdown_choices([], lang="en", ephemeral=ephemeral_default_voice())
    assert choices == [("Default", EPHEMERAL_DEFAULT_VOICE_ID)]


def test_insert_voice_tag_appends_brackets_to_script():
    out = insert_voice_tag("hello ", voice_name="bob")
    assert out == "hello <bob>"


def test_insert_tag_appends_bracket_tag():
    out = insert_tag("hi ", tag="laughing")
    assert out == "hi [laughing]"
