from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxcpm_tts_tool.voice_library import (
    Voice,
    VoiceLibrary,
    VoiceLibraryError,
)


@pytest.fixture
def lib(project_root) -> VoiceLibrary:
    return VoiceLibrary(project_root / "voices")


def _wav(path: Path, name: str) -> Path:
    p = path / name
    p.write_bytes(b"RIFF...")
    return p


@pytest.fixture
def fake_audio(project_root) -> str:
    """A staged wav file path usable as audio_upload= in lib.create().

    All voices now require `audio_upload` (the generated preview) — this
    fixture saves boilerplate of writing a wav per test.
    """
    return str(_wav(project_root, "_preview.wav"))


def test_create_design_voice_persists(lib, project_root, fake_audio):
    v = lib.create(name="旁白女声", mode="design", control="温柔清晰",
                   audio_upload=fake_audio)
    assert v.id and v.id != ""
    on_disk = json.loads((project_root / "voices/voices.json").read_text("utf-8"))
    assert on_disk[0]["name"] == "旁白女声"
    assert on_disk[0]["mode"] == "design"
    assert "control" in on_disk[0]


def test_design_audio_field_set_reference_empty(lib, project_root, fake_audio):
    """Design voices have audio (generated) but no reference_audio (no upload)."""
    v = lib.create(name="x", mode="design",
                   audio_upload=fake_audio,
                   prompt_text="seed text")
    assert v.audio.endswith(".wav") and ".original" not in v.audio
    assert v.reference_audio == ""  # design has no original upload


def test_clone_stores_both_audio_and_reference(lib, project_root, fake_audio):
    """Clone keeps original upload as reference_audio AND generated as audio."""
    upload = _wav(project_root, "user_upload.wav")
    v = lib.create(name="c", mode="clone",
                   reference_audio_upload=str(upload),
                   audio_upload=fake_audio,
                   prompt_text="asr result")
    assert v.audio.endswith(".wav") and ".original" not in v.audio
    assert v.reference_audio.endswith(".original.wav")


def test_clone_requires_reference_audio(lib, fake_audio):
    with pytest.raises(VoiceLibraryError, match="clone.*reference"):
        lib.create(name="c", mode="clone", audio_upload=fake_audio)


def test_clone_keeps_prompt_text_as_metadata(lib, project_root, fake_audio):
    """All modes may now store prompt_text (post-refactor)."""
    upload = _wav(project_root, "user_upload.wav")
    v = lib.create(name="c", mode="clone",
                   reference_audio_upload=str(upload), audio_upload=fake_audio,
                   prompt_text="kept as metadata")
    assert v.prompt_text == "kept as metadata"


def test_hifi_requires_prompt_text(lib, project_root, fake_audio):
    upload = _wav(project_root, "user_upload.wav")
    with pytest.raises(VoiceLibraryError, match="hifi.*prompt"):
        lib.create(name="h", mode="hifi",
                   reference_audio_upload=str(upload), audio_upload=fake_audio)


def test_hifi_clears_control(lib, project_root, fake_audio):
    upload = _wav(project_root, "user_upload.wav")
    v = lib.create(name="h", mode="hifi",
                   reference_audio_upload=str(upload), audio_upload=fake_audio,
                   prompt_text="hi", control="ignored")
    assert v.control == ""


def test_voice_without_audio_rejected(lib):
    """All modes now require audio (the generated preview)."""
    with pytest.raises(VoiceLibraryError, match="generated audio"):
        lib.create(name="x", mode="design")  # no audio_upload


def test_duplicate_name_case_insensitive_after_trim(lib, fake_audio):
    lib.create(name="alpha", mode="design", audio_upload=fake_audio)
    with pytest.raises(VoiceLibraryError, match="duplicate"):
        lib.create(name=" Alpha ", mode="design", audio_upload=fake_audio)


def test_voice_name_with_angle_bracket_rejected(lib, fake_audio):
    with pytest.raises(VoiceLibraryError, match="< or >"):
        lib.create(name="bad<name", mode="design", audio_upload=fake_audio)


def test_only_wav_accepted(lib, project_root, fake_audio):
    mp3 = project_root / "x.mp3"
    mp3.write_bytes(b"id3")
    with pytest.raises(VoiceLibraryError, match=".wav"):
        lib.create(name="m", mode="clone",
                   reference_audio_upload=str(mp3), audio_upload=fake_audio)


def test_audio_files_saved_with_correct_suffixes(lib, project_root, fake_audio):
    upload = _wav(project_root, "user_upload.wav")
    v = lib.create(name="a", mode="clone",
                   reference_audio_upload=str(upload), audio_upload=fake_audio)
    # Generated preview = <id>.wav; original upload = <id>.original.wav
    assert (project_root / "voices/audio" / f"{v.id}.wav").exists()
    assert (project_root / "voices/audio" / f"{v.id}.original.wav").exists()
    assert v.audio == f"voices/audio/{v.id}.wav"
    assert v.reference_audio == f"voices/audio/{v.id}.original.wav"


def test_atomic_write_uses_replace(lib, project_root, fake_audio, monkeypatch):
    """If process crashes between tmp write and replace, original survives."""
    lib.create(name="a", mode="design", audio_upload=fake_audio)
    original = (project_root / "voices/voices.json").read_text("utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(RuntimeError):
        lib.create(name="b", mode="design", audio_upload=fake_audio)
    # Original file unchanged
    assert (project_root / "voices/voices.json").read_text("utf-8") == original


def test_malformed_json_recovery(project_root):
    bad = project_root / "voices/voices.json"
    bad.write_text("not json", encoding="utf-8")
    lib = VoiceLibrary(project_root / "voices")
    assert lib.list_voices() == []
    backups = list((project_root / "voices").glob("voices.broken-*.json"))
    assert len(backups) == 1


def test_delete_removes_both_audio_files(lib, project_root, fake_audio):
    upload = _wav(project_root, "user_upload.wav")
    v = lib.create(name="a", mode="clone",
                   reference_audio_upload=str(upload), audio_upload=fake_audio)
    preview = project_root / "voices/audio" / f"{v.id}.wav"
    original = project_root / "voices/audio" / f"{v.id}.original.wav"
    assert preview.exists() and original.exists()
    lib.delete(v.id)
    assert not preview.exists() and not original.exists()


def test_rename_returns_warning(lib, fake_audio):
    v = lib.create(name="old", mode="design", audio_upload=fake_audio)
    msg = lib.update(v.id, name="new")
    assert "rename" in msg.lower() or "<old>" in msg


def test_id_is_uuid4_hex(lib, fake_audio):
    v = lib.create(name="a", mode="design", audio_upload=fake_audio)
    assert len(v.id) == 32
    assert all(c in "0123456789abcdef" for c in v.id)


def test_lookup_by_name_case_insensitive(lib, fake_audio):
    v = lib.create(name="MixedCase", mode="design", audio_upload=fake_audio)
    assert lib.find_by_name("  mixedcase  ").id == v.id


def test_design_denoise_forced_false(lib, fake_audio):
    v = lib.create(name="a", mode="design", denoise=True, audio_upload=fake_audio)
    assert v.denoise is False


def test_normalize_persisted_per_voice(lib, project_root, fake_audio):
    v = lib.create(name="a", mode="design", normalize=True, audio_upload=fake_audio)
    assert v.normalize is True
    on_disk = json.loads((project_root / "voices/voices.json").read_text("utf-8"))
    assert on_disk[0]["normalize"] is True


def test_normalize_defaults_false(lib, fake_audio):
    v = lib.create(name="a", mode="design", audio_upload=fake_audio)
    assert v.normalize is False


def test_normalize_allowed_in_all_modes(lib, project_root, fake_audio):
    upload = _wav(project_root, "user_upload.wav")
    a = lib.create(name="a", mode="design", normalize=True, audio_upload=fake_audio)
    b = lib.create(name="b", mode="clone", reference_audio_upload=str(upload),
                   audio_upload=fake_audio, normalize=True)
    c = lib.create(name="c", mode="hifi", reference_audio_upload=str(upload),
                   audio_upload=fake_audio, prompt_text="hi", normalize=True)
    assert all(v.normalize is True for v in (a, b, c))


def test_update_can_change_normalize(lib, fake_audio):
    v = lib.create(name="a", mode="design", normalize=False, audio_upload=fake_audio)
    lib.update(v.id, normalize=True)
    assert lib.find_by_id(v.id).normalize is True


def test_update_audio_upload_overwrites_preview(lib, project_root, fake_audio):
    """update(audio_upload=...) must overwrite v.audio on disk in place."""
    v = lib.create(name="a", mode="design", audio_upload=fake_audio)
    audio_path = project_root / v.audio
    assert audio_path.read_bytes() == b"RIFF..."

    new_preview = _wav(project_root, "new_preview.wav")
    new_preview.write_bytes(b"RIFF-NEW")
    lib.update(v.id, audio_upload=str(new_preview))

    # Same id-keyed path, different content.
    assert lib.find_by_id(v.id).audio == v.audio
    assert audio_path.read_bytes() == b"RIFF-NEW"


def test_update_reference_audio_upload_overwrites_original(lib, project_root, fake_audio):
    """update(reference_audio_upload=...) overwrites v.reference_audio on disk."""
    upload = _wav(project_root, "orig.wav")
    v = lib.create(name="a", mode="clone", reference_audio_upload=str(upload),
                   audio_upload=fake_audio)
    ref_path = project_root / v.reference_audio
    assert ref_path.exists()

    new_upload = _wav(project_root, "new_orig.wav")
    new_upload.write_bytes(b"RIFF-ORIG2")
    lib.update(v.id, reference_audio_upload=str(new_upload))

    assert lib.find_by_id(v.id).reference_audio == v.reference_audio
    assert ref_path.read_bytes() == b"RIFF-ORIG2"


def test_seed_text_persists_through_create(lib, project_root, fake_audio):
    """seed_text (the "样本朗读" phrase the user typed) must round-trip through
    JSON so we can reuse it as prompt_text at hifi generation time."""
    upload = _wav(project_root, "user_upload.wav")
    v = lib.create(name="h", mode="hifi",
                   reference_audio_upload=str(upload), audio_upload=fake_audio,
                   prompt_text="ASR of upload",
                   seed_text="样本朗读文本")
    assert v.seed_text == "样本朗读文本"
    on_disk = json.loads((project_root / "voices/voices.json").read_text("utf-8"))
    assert on_disk[0]["seed_text"] == "样本朗读文本"


def test_update_can_change_seed_text(lib, fake_audio):
    v = lib.create(name="a", mode="design", audio_upload=fake_audio,
                   seed_text="first")
    lib.update(v.id, seed_text="second")
    assert lib.find_by_id(v.id).seed_text == "second"


def test_seed_text_defaults_empty(lib, fake_audio):
    """Voices created without seed_text get an empty string (back-compat path)."""
    v = lib.create(name="a", mode="design", audio_upload=fake_audio)
    assert v.seed_text == ""


def test_load_old_voices_json_without_new_fields(project_root):
    """Older voices.json without `normalize` or `audio` must still load.

    Migration semantics: old `reference_audio` (which used to hold the only
    audio path) becomes a fallback for `audio` at script-generation time,
    but the loaded Voice itself just gets the dataclass defaults.
    """
    legacy = [{
        "id": "abc", "name": "x", "mode": "design",
        "control": "", "reference_audio": "voices/audio/abc.wav", "prompt_text": "",
        "denoise": False,
        "created_at": "2026-04-26T00:00:00Z", "updated_at": "2026-04-26T00:00:00Z",
    }]
    (project_root / "voices/voices.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )
    lib2 = VoiceLibrary(project_root / "voices")
    voices = lib2.list_voices()
    assert len(voices) == 1
    assert voices[0].normalize is False  # default for legacy records
    assert voices[0].audio == ""  # default; generation falls back to reference_audio
