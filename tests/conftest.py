"""Shared test fixtures: fakes for VoxCPM SDK and SenseVoice ASR."""
from __future__ import annotations

import numpy as np
import pytest


class FakeVoxCPM:
    """Records calls to .generate(); returns a tiny zero waveform.

    Mirrors the surface used by orchestration: .generate(...) and .sample_rate.
    """

    def __init__(self):
        self.sample_rate = 16000
        self.calls: list[dict] = []

    def generate(self, **kwargs) -> np.ndarray:
        self.calls.append(kwargs)
        return np.zeros(8, dtype=np.float32)


class FakeRecognizer:
    """Returns the canned text for any wav path."""

    def __init__(self, text: str = "fake transcript"):
        self.text = text
        self.calls: list[str] = []

    def recognize(self, wav_path: str) -> str:
        self.calls.append(wav_path)
        return self.text


@pytest.fixture
def fake_model() -> FakeVoxCPM:
    return FakeVoxCPM()


@pytest.fixture
def fake_recognizer() -> FakeRecognizer:
    return FakeRecognizer()


@pytest.fixture
def project_root(tmp_path):
    """A tmp directory standing in for the project root with all runtime dirs."""
    for sub in ("voices", "voices/audio", "outputs",
                "pretrained_models/VoxCPM2",
                "pretrained_models/SenseVoiceSmall",
                "pretrained_models/ZipEnhancer"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path
