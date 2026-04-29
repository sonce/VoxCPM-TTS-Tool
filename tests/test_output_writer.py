import re
from pathlib import Path

import numpy as np

from voxcpm_tts_tool.output_writer import write_output_wav


def test_writes_wav_with_correct_sample_rate(tmp_path):
    wav = np.zeros(16, dtype=np.float32)
    out = write_output_wav(wav, sample_rate=22050, outputs_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".wav"
    import soundfile as sf
    info = sf.info(str(out))
    assert info.samplerate == 22050


def test_filename_uses_yyyymmdd_hhmmss_mmm(tmp_path):
    wav = np.zeros(8, dtype=np.float32)
    out = write_output_wav(wav, sample_rate=16000, outputs_dir=tmp_path)
    assert re.match(r"\d{8}-\d{6}-\d{3}\.wav", out.name)


def test_collision_appends_suffix(tmp_path, monkeypatch):
    """Force two writes to yield the same base filename."""
    monkeypatch.setattr("voxcpm_tts_tool.output_writer._timestamp",
                        lambda: "20260426-120000-000")
    wav = np.zeros(8, dtype=np.float32)
    a = write_output_wav(wav, sample_rate=16000, outputs_dir=tmp_path)
    b = write_output_wav(wav, sample_rate=16000, outputs_dir=tmp_path)
    assert a.name == "20260426-120000-000.wav"
    assert b.name == "20260426-120000-000-1.wav"
