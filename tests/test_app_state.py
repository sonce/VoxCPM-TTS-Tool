from pathlib import Path

from voxcpm_tts_tool.app_state import (
    AppPaths,
    AppState,
    EPHEMERAL_DEFAULT_VOICE_ID,
    ephemeral_default_voice,
    paths_for,
)
from voxcpm_tts_tool.transcription import SenseVoiceTranscriber
from voxcpm_tts_tool.voice_library import VoiceLibrary


def test_paths_for_creates_runtime_dirs(tmp_path):
    p = paths_for(tmp_path)
    for sub in (p.voices, p.voices_audio, p.outputs,
                p.voxcpm_dir, p.sensevoice_dir, p.zipenhancer_dir):
        assert sub.exists() and sub.is_dir()


def test_ephemeral_default_voice_unaddressable():
    v = ephemeral_default_voice()
    assert v.id == EPHEMERAL_DEFAULT_VOICE_ID
    assert v.id.startswith("__")
    assert v.mode == "design"
    assert v.reference_audio == ""


# ---------------------------------------------------------------------------
# gen_stop_flag — the only generation-related state retained after the
# 2026-04-27 flatten (chunks-table queue fields were removed).
# ---------------------------------------------------------------------------

def _state(tmp_path: Path) -> AppState:
    paths = AppPaths(
        root=tmp_path, voices=tmp_path / "voices",
        voices_audio=tmp_path / "voices/audio",
        outputs=tmp_path / "outputs",
        voxcpm_dir=tmp_path / "vox",
        sensevoice_dir=tmp_path / "sv",
        zipenhancer_dir=tmp_path / "ze",
        cache_dir=tmp_path / "cache",
    )
    for d in (paths.voices, paths.voices_audio, paths.outputs):
        d.mkdir(parents=True, exist_ok=True)
    return AppState(
        paths=paths,
        library=VoiceLibrary(paths.voices),
        model=object(),
        transcriber=SenseVoiceTranscriber.unavailable("test"),
        zipenhancer_loaded=False,
    )


def test_appstate_gen_stop_flag_defaults_false(tmp_path):
    s = _state(tmp_path)
    assert s.gen_stop_flag is False


def test_appstate_gen_stop_flag_per_instance(tmp_path):
    a = _state(tmp_path)
    b = _state(tmp_path)
    a.gen_stop_flag = True
    assert b.gen_stop_flag is False
