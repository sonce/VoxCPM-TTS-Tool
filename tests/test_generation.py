from voxcpm_tts_tool.generation import build_generate_kwargs
from voxcpm_tts_tool.voice_library import Voice


def _design(control: str = "", denoise: bool = False, normalize: bool = False) -> Voice:
    # Post-refactor: design voices have audio (the generated preview).
    return Voice(id="x", name="x", mode="design", control=control,
                 audio="voices/audio/x.wav",
                 denoise=denoise, normalize=normalize)


def _clone(control: str = "", denoise: bool = False, normalize: bool = False) -> Voice:
    return Voice(id="x", name="x", mode="clone", control=control,
                 reference_audio="voices/audio/x.original.wav",
                 audio="voices/audio/x.wav",
                 denoise=denoise, normalize=normalize)


def _hifi(denoise: bool = False, normalize: bool = False) -> Voice:
    return Voice(id="x", name="x", mode="hifi",
                 reference_audio="voices/audio/x.original.wav",
                 audio="voices/audio/x.wav",
                 prompt_text="hello", denoise=denoise, normalize=normalize)


def test_design_with_audio_and_control_routes_through_clone_branch():
    """Post-refactor: design voices have audio (the saved preview), so they
    behave like clone at script-generation time — text gets the (control)
    prefix and reference_wav_path is the saved audio."""
    kw = build_generate_kwargs(_design(control="温柔"), "你好",
                               zipenhancer_loaded=True, audio_root=".")
    assert kw["text"] == "(温柔)你好"
    assert kw["reference_wav_path"].endswith("x.wav")
    assert kw["denoise"] is False  # design voices always have denoise=False


def test_design_with_audio_no_control_no_prefix():
    kw = build_generate_kwargs(_design(control=""), "你好",
                               zipenhancer_loaded=True, audio_root=".")
    assert kw["text"] == "你好"
    assert kw["reference_wav_path"].endswith("x.wav")


def test_legacy_design_without_audio_text_only():
    """Legacy fallback: voices with no audio field at all (old voices.json)
    use text-only generation with control as prefix."""
    legacy = Voice(id="x", name="x", mode="design", control="温柔")
    kw = build_generate_kwargs(legacy, "你好", zipenhancer_loaded=True, audio_root=".")
    assert kw == {"text": "(温柔)你好"}


def test_clone_with_control_and_reference_path():
    kw = build_generate_kwargs(_clone(control="温柔"), "你好",
                               zipenhancer_loaded=True, audio_root="/root")
    assert kw["text"] == "(温柔)你好"
    assert kw["reference_wav_path"] == "/root/voices/audio/x.wav".replace("/", __import__("os").sep)
    assert "prompt_wav_path" not in kw
    assert "prompt_text" not in kw


def test_clone_without_control():
    kw = build_generate_kwargs(_clone(), "你好",
                               zipenhancer_loaded=True, audio_root="/r")
    assert kw["text"] == "你好"


def test_hifi_passes_prompt_pair_and_reference():
    kw = build_generate_kwargs(_hifi(), "target",
                               zipenhancer_loaded=True, audio_root="/r")
    import os as _os
    expected_path = _os.path.normpath("/r/voices/audio/x.wav")
    assert kw == {
        "text": "target",
        "prompt_wav_path": expected_path,
        "prompt_text": "hello",
        "reference_wav_path": expected_path,
        "denoise": False,
    }


def test_hifi_prefers_seed_text_over_prompt_text():
    """`prompt_wav_path` is `voice.audio` (the generated preview wav, which
    actually says `voice.seed_text`). The SDK's prompt_text must therefore
    match seed_text, not the original upload's ASR transcript stored in
    `voice.prompt_text`.
    """
    voice = Voice(
        id="x", name="x", mode="hifi",
        reference_audio="voices/audio/x.original.wav",
        audio="voices/audio/x.wav",
        prompt_text="upload ASR text",
        seed_text="样本朗读文本",
    )
    kw = build_generate_kwargs(voice, "target",
                               zipenhancer_loaded=False, audio_root="/r")
    assert kw["prompt_text"] == "样本朗读文本"


def test_hifi_falls_back_to_prompt_text_when_no_seed_text():
    """Legacy voices saved before the seed_text field still work — the hifi
    branch falls back to voice.prompt_text."""
    voice = Voice(
        id="x", name="x", mode="hifi",
        reference_audio="voices/audio/x.original.wav",
        audio="voices/audio/x.wav",
        prompt_text="legacy transcript",
        seed_text="",
    )
    kw = build_generate_kwargs(voice, "target",
                               zipenhancer_loaded=False, audio_root="/r")
    assert kw["prompt_text"] == "legacy transcript"


def test_denoise_true_only_when_voice_on_AND_zipenhancer_loaded():
    on = _hifi(denoise=True)
    kw_yes = build_generate_kwargs(on, "t", zipenhancer_loaded=True, audio_root="/r")
    kw_no_zh = build_generate_kwargs(on, "t", zipenhancer_loaded=False, audio_root="/r")
    off = _hifi(denoise=False)
    kw_off = build_generate_kwargs(off, "t", zipenhancer_loaded=True, audio_root="/r")
    assert kw_yes["denoise"] is True
    assert kw_no_zh["denoise"] is False
    assert kw_off["denoise"] is False


def test_legacy_design_no_audio_omits_denoise_kwarg():
    """A legacy voice with no audio field gets bare {text:...} (no denoise key)."""
    legacy = Voice(id="x", name="x", mode="design")  # no audio field
    kw = build_generate_kwargs(legacy, "x", zipenhancer_loaded=True, audio_root=".")
    assert "denoise" not in kw
    assert "reference_wav_path" not in kw


# ---- Per-call CFG / inference_timesteps + per-voice normalize / denoise ----

def test_default_cfg_value_omitted_from_kwargs():
    """When cfg_value matches SDK default (2.0), it must not be in kwargs."""
    kw = build_generate_kwargs(_design(), "x", zipenhancer_loaded=True, audio_root=".")
    assert "cfg_value" not in kw


def test_non_default_cfg_value_included():
    kw = build_generate_kwargs(_design(), "x", zipenhancer_loaded=True, audio_root=".",
                               cfg_value=3.5)
    assert kw["cfg_value"] == 3.5


def test_voice_normalize_false_omits_kwarg():
    kw = build_generate_kwargs(_design(normalize=False), "x",
                               zipenhancer_loaded=True, audio_root=".")
    assert "normalize" not in kw


def test_voice_normalize_true_includes_kwarg():
    kw = build_generate_kwargs(_design(normalize=True), "x",
                               zipenhancer_loaded=True, audio_root=".")
    assert kw["normalize"] is True


def test_inference_timesteps_omitted_when_default():
    kw = build_generate_kwargs(_design(), "x", zipenhancer_loaded=True, audio_root=".",
                               inference_timesteps=10)
    assert "inference_timesteps" not in kw


def test_inference_timesteps_included_when_overridden():
    kw = build_generate_kwargs(_design(), "x", zipenhancer_loaded=True, audio_root=".",
                               inference_timesteps=20)
    assert kw["inference_timesteps"] == 20


def test_voice_denoise_true_with_zipenhancer_loaded():
    voice = _clone(denoise=True)
    kw = build_generate_kwargs(voice, "x", zipenhancer_loaded=True, audio_root="/r")
    assert kw["denoise"] is True


def test_voice_denoise_false_yields_false_kwarg():
    voice = _clone(denoise=False)
    kw = build_generate_kwargs(voice, "x", zipenhancer_loaded=True, audio_root="/r")
    assert kw["denoise"] is False


def test_voice_denoise_true_but_zipenhancer_not_loaded():
    voice = _clone(denoise=True)
    kw = build_generate_kwargs(voice, "x", zipenhancer_loaded=False, audio_root="/r")
    assert kw["denoise"] is False  # gated by zipenhancer_loaded


def test_all_voice_props_and_per_call_knobs_combine_for_hifi():
    voice = _hifi(denoise=True, normalize=True)
    kw = build_generate_kwargs(voice, "target", zipenhancer_loaded=True, audio_root="/r",
                               cfg_value=4.0, inference_timesteps=15)
    assert kw["cfg_value"] == 4.0
    assert kw["normalize"] is True  # from voice
    assert kw["denoise"] is True    # from voice
    assert kw["inference_timesteps"] == 15


# ---- Script-level `(control)` routes through clone-style ----

def test_script_control_forces_clone_style_for_voice_with_audio_and_prompt():
    """Voice has audio+prompt (would default to hifi); script `(control)` overrides to clone."""
    voice = _hifi()  # has reference_audio + prompt_text
    kw = build_generate_kwargs(voice, "你好", zipenhancer_loaded=True, audio_root="/r",
                               script_control="温柔，低声")
    assert kw["text"] == "(温柔，低声)你好"
    assert kw["reference_wav_path"].endswith("voices/audio/x.wav") or \
           kw["reference_wav_path"].endswith("voices\\audio\\x.wav")
    assert "prompt_wav_path" not in kw
    assert "prompt_text" not in kw


def test_script_control_overrides_voice_stored_control():
    voice = _clone(control="voice_default_control")
    kw = build_generate_kwargs(voice, "你好", zipenhancer_loaded=True, audio_root="/r",
                               script_control="script_control")
    assert "(script_control)你好" == kw["text"]


def test_no_script_control_picks_hifi_when_voice_has_prompt():
    voice = _hifi()  # has audio + prompt_text
    kw = build_generate_kwargs(voice, "target", zipenhancer_loaded=True, audio_root="/r")
    assert kw["text"] == "target"
    assert "prompt_wav_path" in kw
    assert kw["prompt_text"] == "hello"


def test_no_script_control_falls_back_to_clone_when_voice_lacks_prompt():
    voice = _clone()  # has audio, no prompt_text
    kw = build_generate_kwargs(voice, "target", zipenhancer_loaded=True, audio_root="/r")
    assert "prompt_wav_path" not in kw
    assert "reference_wav_path" in kw
    assert kw["text"] == "target"  # no control prefix since voice.control is empty


def test_empty_string_script_control_is_clone_with_no_prefix():
    """`<voice>()text` means user explicitly chose clone with no style description."""
    voice = _hifi()
    kw = build_generate_kwargs(voice, "你好", zipenhancer_loaded=True, audio_root="/r",
                               script_control="")
    assert kw["text"] == "你好"  # no parenthetical prefix (empty control)
    assert "reference_wav_path" in kw
    assert "prompt_wav_path" not in kw  # forced clone style by script_control != None


def test_synthesize_voice_preview_design_with_control(fake_model):
    from pathlib import Path
    from voxcpm_tts_tool.generation import synthesize_voice_preview

    tmp_path, transcript = synthesize_voice_preview(
        fake_model, mode="design", control="温柔的女声", seed_text="你好",
    )
    try:
        assert fake_model.calls[0]["text"] == "(温柔的女声)你好"
        # No audio paths for design.
        assert "reference_wav_path" not in fake_model.calls[0]
        assert "prompt_wav_path" not in fake_model.calls[0]
        # Transcript echoes the seed text so the saved voice can be re-used hifi-style.
        assert transcript == "你好"
        import soundfile as sf
        _, sr = sf.read(tmp_path)
        assert sr == 16000
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_synthesize_voice_preview_design_no_control(fake_model):
    from pathlib import Path
    from voxcpm_tts_tool.generation import synthesize_voice_preview

    tmp_path, _ = synthesize_voice_preview(
        fake_model, mode="design", control="", seed_text="just seed",
    )
    try:
        assert fake_model.calls[0]["text"] == "just seed"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_synthesize_voice_preview_clone_speaks_seed_text_with_control(fake_model, tmp_path):
    """Clone now speaks seed_text (not the transcript) prefixed with (control)."""
    from voxcpm_tts_tool.generation import synthesize_voice_preview

    upload = tmp_path / "ref.wav"
    upload.write_bytes(b"RIFF...")

    out, transcript = synthesize_voice_preview(
        fake_model,
        mode="clone",
        control="温柔，低声",
        seed_text="样本朗读文本",
        upload_path=str(upload),
        transcript="原始参考音频转写",
    )
    try:
        call = fake_model.calls[0]
        assert call["text"] == "(温柔，低声)样本朗读文本"
        assert call["reference_wav_path"] == str(upload)
        # Clone does NOT use prompt_wav_path / prompt_text per VoxCPM2 guide.
        assert "prompt_wav_path" not in call
        assert "prompt_text" not in call
        # Stored transcript is the upload's ASR result so the saved voice can
        # later be addressed via <voice>text (hifi-style).
        assert transcript == "原始参考音频转写"
    finally:
        from pathlib import Path
        Path(out).unlink(missing_ok=True)


def test_synthesize_voice_preview_clone_seed_text_required(fake_model, tmp_path):
    import pytest
    from voxcpm_tts_tool.generation import synthesize_voice_preview

    upload = tmp_path / "ref.wav"
    upload.write_bytes(b"RIFF...")
    with pytest.raises(ValueError, match="seed_text"):
        synthesize_voice_preview(
            fake_model, mode="clone", control="x", seed_text="",
            upload_path=str(upload), transcript="t",
        )


def test_synthesize_voice_preview_clone_falls_back_transcript_when_no_seed_arg(fake_model, tmp_path):
    """If transcript is missing, the saved voice's prompt_text falls back to seed_text."""
    from voxcpm_tts_tool.generation import synthesize_voice_preview

    upload = tmp_path / "ref.wav"
    upload.write_bytes(b"RIFF...")
    out, transcript = synthesize_voice_preview(
        fake_model, mode="clone", control="", seed_text="seed only",
        upload_path=str(upload), transcript="",
    )
    try:
        assert transcript == "seed only"  # storage fallback
    finally:
        from pathlib import Path
        Path(out).unlink(missing_ok=True)


def test_synthesize_voice_preview_hifi_uses_prompt_pair_and_seed(fake_model, tmp_path):
    from voxcpm_tts_tool.generation import synthesize_voice_preview

    upload = tmp_path / "ref.wav"
    upload.write_bytes(b"RIFF...")

    out, transcript = synthesize_voice_preview(
        fake_model,
        mode="hifi",
        seed_text="测试音色",
        upload_path=str(upload),
        transcript="原始转写",
    )
    try:
        call = fake_model.calls[0]
        # Per VoxCPM2 guide: hifi text is plain (no control prefix).
        assert call["text"] == "测试音色"
        assert call["prompt_wav_path"] == str(upload)
        assert call["prompt_text"] == "原始转写"
        assert call["reference_wav_path"] == str(upload)
        assert transcript == "原始转写"
    finally:
        from pathlib import Path
        Path(out).unlink(missing_ok=True)


def test_synthesize_voice_preview_hifi_seed_text_required(fake_model, tmp_path):
    import pytest
    from voxcpm_tts_tool.generation import synthesize_voice_preview

    upload = tmp_path / "ref.wav"
    upload.write_bytes(b"RIFF...")
    with pytest.raises(ValueError, match="seed_text"):
        synthesize_voice_preview(
            fake_model, mode="hifi", upload_path=str(upload),
            transcript="t", seed_text="",
        )


def test_synthesize_voice_preview_unknown_mode_raises(fake_model):
    import pytest
    from voxcpm_tts_tool.generation import synthesize_voice_preview
    with pytest.raises(ValueError, match="unknown mode"):
        synthesize_voice_preview(fake_model, mode="bogus", seed_text="x")


# ---- Event types for run_generation ----------------------------------------

def test_progress_dataclass_has_done_and_total():
    from voxcpm_tts_tool.generation import Progress
    p = Progress(done=3, total=10)
    assert p.done == 3
    assert p.total == 10


def test_result_dataclass_defaults_was_stopped_false():
    import numpy as np
    from voxcpm_tts_tool.generation import Result
    r = Result(wav=np.zeros(4, dtype=np.float32), sample_rate=16000)
    assert r.was_stopped is False
    assert r.wav.shape == (4,)
    assert r.sample_rate == 16000


def test_result_dataclass_was_stopped_set_true():
    import numpy as np
    from voxcpm_tts_tool.generation import Result
    r = Result(wav=np.zeros(0, dtype=np.float32), sample_rate=16000, was_stopped=True)
    assert r.was_stopped is True


# ---- _voice_for_segment ----------------------------------------------------

def test_voice_for_segment_returns_default_when_name_matches():
    from voxcpm_tts_tool.generation import _voice_for_segment
    from voxcpm_tts_tool.voice_library import Voice

    default = Voice(id="d", name="default", mode="design")
    # library has nothing matching; helper must short-circuit on name == default.
    class _Lib:
        def find_by_name(self, name): return None

    got = _voice_for_segment("default", default_voice=default, library=_Lib())
    assert got is default


def test_voice_for_segment_returns_library_voice_by_name():
    from voxcpm_tts_tool.generation import _voice_for_segment
    from voxcpm_tts_tool.voice_library import Voice

    default = Voice(id="d", name="default", mode="design")
    bob = Voice(id="b", name="bob", mode="design")
    class _Lib:
        def find_by_name(self, name): return bob if name.lower() == "bob" else None

    got = _voice_for_segment("bob", default_voice=default, library=_Lib())
    assert got is bob


def test_voice_for_segment_ephemeral_default_short_circuits():
    """Default voice with id starting with __ is ephemeral; library lookup
    would return None for it, so the name short-circuit MUST fire first."""
    from voxcpm_tts_tool.generation import _voice_for_segment
    from voxcpm_tts_tool.voice_library import Voice

    ephemeral = Voice(id="__default__", name="__default__", mode="design")
    class _Lib:
        def find_by_name(self, name): return None

    got = _voice_for_segment("__default__", default_voice=ephemeral, library=_Lib())
    assert got is ephemeral


def test_voice_for_segment_unknown_name_raises():
    """Parser invariant says this can't happen, but defensive raise."""
    import pytest
    from voxcpm_tts_tool.generation import _voice_for_segment
    from voxcpm_tts_tool.voice_library import Voice

    default = Voice(id="d", name="default", mode="design")
    class _Lib:
        def find_by_name(self, name): return None

    with pytest.raises(ValueError, match="voice not found"):
        _voice_for_segment("nobody", default_voice=default, library=_Lib())


# ---- run_generation: happy path --------------------------------------------

class _RecordingModel:
    """Like FakeVoxCPM but each generate() returns a distinguishable wav so
    we can assert concatenation order."""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.calls: list[dict] = []
        self._counter = 0

    def generate(self, **kwargs):
        import numpy as np
        self._counter += 1
        self.calls.append(kwargs)
        # Each chunk emits a 4-sample wav whose value == the call ordinal.
        return np.full(4, float(self._counter), dtype=np.float32)


def test_run_generation_single_voice_two_terminator_chunks():
    """`split_for_generation` splits on '。' → 2 chunks; assert yield order:
    Progress(1,2), Progress(2,2), Result(concat, was_stopped=False)."""
    import numpy as np
    from voxcpm_tts_tool.generation import Progress, Result, run_generation
    from voxcpm_tts_tool.voice_library import Voice

    default = Voice(id="d", name="default", mode="design", audio="voices/audio/d.wav")

    class _Lib:
        def list_voices(self): return []
        def find_by_name(self, name): return None

    model = _RecordingModel()
    events = list(run_generation(
        "你好。再见。",
        library=_Lib(), default_voice=default, model=model,
        audio_root=".", zipenhancer_loaded=False,
        char_budget=80, cfg_value=2.0, inference_timesteps=10,
        stop_flag=lambda: False,
    ))

    assert len(events) == 3
    assert events[0] == Progress(done=1, total=2)
    assert events[1] == Progress(done=2, total=2)
    assert isinstance(events[2], Result)
    assert events[2].was_stopped is False
    assert events[2].sample_rate == 16000
    # Wavs concatenated in order: [1,1,1,1, 2,2,2,2]
    assert np.array_equal(events[2].wav, np.array(
        [1, 1, 1, 1, 2, 2, 2, 2], dtype=np.float32))
    # Each chunk became one SDK call.
    assert [c["text"] for c in model.calls] == ["你好。", "再见。"]


def test_run_generation_multi_voice_with_script_control(project_root):
    """`<bob>你好<alice>(温柔)再见` → 2 segments → 2 SDK calls.
    Bob uses hifi (no script_control), Alice uses clone (script_control='温柔')."""
    from voxcpm_tts_tool.generation import Result, run_generation
    from voxcpm_tts_tool.voice_library import Voice

    bob = Voice(id="b", name="bob", mode="hifi",
                reference_audio="voices/audio/b.original.wav",
                audio="voices/audio/b.wav",
                prompt_text="bob seed")
    alice = Voice(id="a", name="alice", mode="clone",
                  audio="voices/audio/a.wav")
    class _Lib:
        def list_voices(self): return [bob, alice]
        def find_by_name(self, name):
            return {"bob": bob, "alice": alice}.get(name.lower())

    model = _RecordingModel()
    events = list(run_generation(
        "<bob>你好<alice>(温柔)再见",
        library=_Lib(), default_voice=alice, model=model,
        audio_root=str(project_root), zipenhancer_loaded=False,
        char_budget=80, cfg_value=2.0, inference_timesteps=10,
        stop_flag=lambda: False,
    ))

    assert len([e for e in events if isinstance(e, Result)]) == 1
    assert len(model.calls) == 2

    bob_call = model.calls[0]
    assert bob_call["text"] == "你好"
    assert "prompt_wav_path" in bob_call
    assert bob_call["prompt_text"] == "bob seed"

    alice_call = model.calls[1]
    assert alice_call["text"] == "(温柔)再见"
    assert "prompt_wav_path" not in alice_call
    assert "reference_wav_path" in alice_call


def test_run_generation_splits_long_line_into_chunks_by_budget():
    """A single line with multiple terminators exceeds char_budget; each
    sentence becomes its own SDK call."""
    from voxcpm_tts_tool.generation import Progress, Result, run_generation
    from voxcpm_tts_tool.long_text import split_for_generation
    from voxcpm_tts_tool.voice_library import Voice

    default = Voice(id="d", name="default", mode="design", audio="voices/audio/d.wav")

    class _Lib:
        def list_voices(self): return []
        def find_by_name(self, name): return None

    text = "句子一。句子二。句子三。句子四。"
    expected_chunks = split_for_generation(text, char_budget=4)
    assert len(expected_chunks) == 4   # sanity: budget forces per-sentence split

    model = _RecordingModel()
    events = list(run_generation(
        text, library=_Lib(), default_voice=default, model=model,
        audio_root=".", zipenhancer_loaded=False,
        char_budget=4, cfg_value=2.0, inference_timesteps=10,
        stop_flag=lambda: False,
    ))

    progresses = [e for e in events if isinstance(e, Progress)]
    results = [e for e in events if isinstance(e, Result)]
    assert len(progresses) == 4
    assert progresses[-1] == Progress(done=4, total=4)
    assert len(results) == 1
    assert len(model.calls) == 4
    assert [c["text"] for c in model.calls] == expected_chunks


def test_run_generation_stop_yields_result_with_was_stopped_true():
    """stop_flag returns True after 2 chunks completed; assert generator
    yields Result(was_stopped=True) with the 2 wavs concatenated, no third
    SDK call, no third Progress."""
    import numpy as np
    from voxcpm_tts_tool.generation import Progress, Result, run_generation
    from voxcpm_tts_tool.voice_library import Voice

    default = Voice(id="d", name="default", mode="design", audio="voices/audio/d.wav")

    class _Lib:
        def list_voices(self): return []
        def find_by_name(self, name): return None

    model = _RecordingModel()

    def _stop_flag():
        # Top-of-iteration check: returns True once 2 SDK calls have completed.
        return len(model.calls) >= 2

    events = list(run_generation(
        "句一。句二。句三。句四。",
        library=_Lib(), default_voice=default, model=model,
        audio_root=".", zipenhancer_loaded=False,
        char_budget=4, cfg_value=2.0, inference_timesteps=10,
        stop_flag=_stop_flag,
    ))

    progresses = [e for e in events if isinstance(e, Progress)]
    results = [e for e in events if isinstance(e, Result)]
    assert len(progresses) == 2                    # only 2 chunks finished
    assert len(results) == 1
    assert results[0].was_stopped is True
    # Result wav is the 2 partial wavs concatenated.
    assert np.array_equal(results[0].wav, np.array(
        [1, 1, 1, 1, 2, 2, 2, 2], dtype=np.float32))
    assert len(model.calls) == 2                   # no 3rd SDK call


def test_run_generation_failure_propagates_no_result():
    """fake model raises on the third call. Generator yields 2 Progresses,
    then RuntimeError propagates; NO Result is yielded."""
    import pytest
    from voxcpm_tts_tool.generation import Progress, Result, run_generation
    from voxcpm_tts_tool.voice_library import Voice

    default = Voice(id="d", name="default", mode="design", audio="voices/audio/d.wav")

    class _Lib:
        def list_voices(self): return []
        def find_by_name(self, name): return None

    class _BoomingModel:
        sample_rate = 16000

        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            import numpy as np
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("boom")
            return np.zeros(4, dtype=np.float32)

    gen = run_generation(
        "句一。句二。句三。句四。",
        library=_Lib(), default_voice=default, model=_BoomingModel(),
        audio_root=".", zipenhancer_loaded=False,
        char_budget=4, cfg_value=2.0, inference_timesteps=10,
        stop_flag=lambda: False,
    )

    collected = []
    with pytest.raises(RuntimeError, match="boom"):
        for ev in gen:
            collected.append(ev)

    assert all(isinstance(e, Progress) for e in collected)
    assert len(collected) == 2
    assert all(not isinstance(e, Result) for e in collected)


def test_run_generation_empty_input_yields_only_empty_result():
    from voxcpm_tts_tool.generation import Result, run_generation
    from voxcpm_tts_tool.voice_library import Voice

    default = Voice(id="d", name="default", mode="design")
    class _Lib:
        def list_voices(self): return []
        def find_by_name(self, name): return None

    model = _RecordingModel()
    events = list(run_generation(
        "", library=_Lib(), default_voice=default, model=model,
        audio_root=".", zipenhancer_loaded=False,
        char_budget=80, cfg_value=2.0, inference_timesteps=10,
        stop_flag=lambda: False,
    ))
    assert len(events) == 1
    assert isinstance(events[0], Result)
    assert events[0].wav.size == 0
    assert events[0].was_stopped is False
    assert len(model.calls) == 0


def test_run_generation_whitespace_only_input_yields_only_empty_result():
    from voxcpm_tts_tool.generation import Result, run_generation
    from voxcpm_tts_tool.voice_library import Voice

    default = Voice(id="d", name="default", mode="design")
    class _Lib:
        def list_voices(self): return []
        def find_by_name(self, name): return None

    model = _RecordingModel()
    events = list(run_generation(
        "   \n  \n",
        library=_Lib(), default_voice=default, model=model,
        audio_root=".", zipenhancer_loaded=False,
        char_budget=80, cfg_value=2.0, inference_timesteps=10,
        stop_flag=lambda: False,
    ))
    assert len(events) == 1
    assert isinstance(events[0], Result)
    assert events[0].wav.size == 0
    assert len(model.calls) == 0


def test_run_generation_ephemeral_default_voice_with_empty_library():
    """Empty library + ephemeral default → single Result with one wav.
    _voice_for_segment must short-circuit on default name match because
    library.find_by_name would return None for `__default__`."""
    from voxcpm_tts_tool.generation import Result, run_generation
    from voxcpm_tts_tool.voice_library import Voice

    ephemeral = Voice(id="__default__", name="__default__", mode="design")
    class _Lib:
        def list_voices(self): return []
        def find_by_name(self, name): return None

    model = _RecordingModel()
    events = list(run_generation(
        "你好",
        library=_Lib(), default_voice=ephemeral, model=model,
        audio_root=".", zipenhancer_loaded=False,
        char_budget=80, cfg_value=2.0, inference_timesteps=10,
        stop_flag=lambda: False,
    ))

    results = [e for e in events if isinstance(e, Result)]
    assert len(results) == 1
    assert len(model.calls) == 1
    # Ephemeral default has no audio → text-only call.
    assert model.calls[0] == {"text": "你好"}


