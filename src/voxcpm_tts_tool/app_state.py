"""Holds resolved paths, the model singleton, the library, and the recognizer.

Created once in app.py at startup. Tests can construct partial AppState by
passing fakes directly into the dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .transcription import SenseVoiceTranscriber
from .voice_library import Voice, VoiceLibrary

EPHEMERAL_DEFAULT_VOICE_ID = "__default__"


@dataclass
class AppPaths:
    root: Path
    voices: Path
    voices_audio: Path
    outputs: Path
    voxcpm_dir: Path
    sensevoice_dir: Path
    zipenhancer_dir: Path
    cache_dir: Path


def paths_for(project_root: Path) -> AppPaths:
    root = Path(project_root)
    p = AppPaths(
        root=root,
        voices=root / "voices",
        voices_audio=root / "voices" / "audio",
        outputs=root / "outputs",
        voxcpm_dir=root / "pretrained_models" / "VoxCPM2",
        sensevoice_dir=root / "pretrained_models" / "SenseVoiceSmall",
        zipenhancer_dir=root / "pretrained_models" / "ZipEnhancer",
        cache_dir=root / ".cache",
    )
    for d in (p.voices, p.voices_audio, p.outputs,
              p.voxcpm_dir, p.sensevoice_dir, p.zipenhancer_dir, p.cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    return p


def ephemeral_default_voice() -> Voice:
    """Per spec §Error Handling: an in-memory voice used only when the library is empty."""
    return Voice(
        id=EPHEMERAL_DEFAULT_VOICE_ID,
        name=EPHEMERAL_DEFAULT_VOICE_ID,
        mode="design",
        control="",
    )


@dataclass
class AppState:
    paths: AppPaths
    library: VoiceLibrary
    model: object  # voxcpm.VoxCPM at runtime; FakeVoxCPM in tests
    transcriber: SenseVoiceTranscriber
    zipenhancer_loaded: bool

    # Stop flag for the streaming run_generation loop. Flipped by the 停止
    # button handler; checked at the top of each chunk by the generator.
    gen_stop_flag: bool = False

    def default_voice(self, requested_id: str | None) -> Voice:
        """Resolve the user's selected default; fall back to first voice or ephemeral."""
        if requested_id:
            v = self.library.find_by_id(requested_id)
            if v is not None:
                return v
        voices = self.library.list_voices()
        if voices:
            return voices[0]
        return ephemeral_default_voice()
