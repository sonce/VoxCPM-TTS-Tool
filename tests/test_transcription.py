import pytest

from voxcpm_tts_tool.transcription import (
    AsrUnavailable,
    SenseVoiceTranscriber,
)


def test_calls_underlying_recognizer(fake_recognizer, tmp_path):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF...")
    t = SenseVoiceTranscriber.from_recognizer(fake_recognizer)
    assert t.transcribe(str(wav)) == "fake transcript"
    assert fake_recognizer.calls == [str(wav)]


def test_unavailable_raises(tmp_path):
    t = SenseVoiceTranscriber.unavailable("import failed")
    with pytest.raises(AsrUnavailable, match="import failed"):
        t.transcribe(str(tmp_path / "x.wav"))


def test_is_available_flag():
    t1 = SenseVoiceTranscriber.unavailable("nope")
    t2 = SenseVoiceTranscriber.from_recognizer(object())
    assert t1.is_available is False
    assert t2.is_available is True
