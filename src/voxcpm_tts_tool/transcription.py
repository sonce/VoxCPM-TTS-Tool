"""Thin wrapper around SenseVoiceSmall via FunASR.

Importing FunASR/torch is expensive and may fail; this module isolates the
import and exposes a typed wrapper that the rest of the app uses.
See spec §Reference-audio transcription behavior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AsrUnavailable(RuntimeError):
    """Raised when transcription is requested but no recognizer was loaded."""


class _Recognizer(Protocol):
    def recognize(self, wav_path: str) -> str: ...


class SenseVoiceTranscriber:
    """Wrapper exposing .is_available + .transcribe(wav_path) -> str."""

    def __init__(self, recognizer: _Recognizer | None, error: str = ""):
        self._rec = recognizer
        self._err = error

    @classmethod
    def from_recognizer(cls, recognizer: _Recognizer) -> "SenseVoiceTranscriber":
        return cls(recognizer)

    @classmethod
    def unavailable(cls, error: str) -> "SenseVoiceTranscriber":
        return cls(None, error=error)

    @property
    def is_available(self) -> bool:
        return self._rec is not None

    def transcribe(self, wav_path: str) -> str:
        if self._rec is None:
            raise AsrUnavailable(f"ASR is not available: {self._err}")
        return self._rec.recognize(wav_path).strip()


def load_real_transcriber(model_dir: Path) -> SenseVoiceTranscriber:
    """Construct a real SenseVoice recognizer; on import failure return unavailable."""
    try:
        from funasr import AutoModel  # type: ignore
    except Exception as exc:
        return SenseVoiceTranscriber.unavailable(f"funasr import failed: {exc}")

    try:
        model = AutoModel(model=str(model_dir), disable_update=True)
    except Exception as exc:
        return SenseVoiceTranscriber.unavailable(f"SenseVoice load failed: {exc}")

    class _Adapter:
        def recognize(self, wav_path: str) -> str:
            # Decode the audio ourselves with soundfile (libsndfile, no
            # ffmpeg / torchcodec needed for wav/flac/ogg). Resample to 16k
            # mono via torchaudio.functional. Pass the resulting numpy array
            # to funasr.generate so its internal load path (which on newer
            # torchaudio requires torchcodec, then falls back to spawning
            # ffmpeg) is bypassed entirely. See upstream issue: torchaudio
            # >=2.5 delegates `load()` to torchcodec.
            audio, sr = _decode_to_16k_mono_float32(wav_path)
            res = model.generate(
                input=audio, fs=sr, language="auto", use_itn=True,
            )
            if not res:
                return ""
            text = res[0].get("text", "")
            return _strip_funasr_tags(text)

    return SenseVoiceTranscriber.from_recognizer(_Adapter())


def _strip_funasr_tags(text: str) -> str:
    """Remove FunASR special tokens like <|zh|>, <|woitn|>, etc."""
    import re

    return re.sub(r"<\|[^|]+\|>", "", text).strip()


def _decode_to_16k_mono_float32(src_path: str):
    """Decode `src_path` to a 16 kHz mono float32 numpy array.

    Uses soundfile (libsndfile) for I/O — no ffmpeg or torchcodec needed for
    wav / flac / ogg. Resampling is done via `torchaudio.functional.resample`
    which is pure tensor math (no subprocess).

    Returns ``(numpy_array, 16000)``. Raises on unreadable input — the caller
    should let the exception bubble up so the user sees the real error.
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(src_path, dtype="float32", always_2d=True)
    # Mix to mono.
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    target_sr = 16000
    if sr != target_sr:
        import torch
        import torchaudio.functional as F
        tensor = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
        tensor = F.resample(tensor, sr, target_sr)
        data = tensor.squeeze(0).cpu().numpy()
        sr = target_sr

    return data, int(sr)
