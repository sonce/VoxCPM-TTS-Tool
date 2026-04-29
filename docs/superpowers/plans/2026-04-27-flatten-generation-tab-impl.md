# Flatten Generation Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Roll back the parse-then-table Generation flow to a single streaming generator: one button → progress text → one wav. Keeps `<voice>` script switching, `(control)` parens, and char-budget chunk splitting.

**Architecture:** Restore `run_generation` in `generation.py` as an `Iterator[Progress | Result]` event-stream generator (two new dataclasses). The Gradio handler in `app.py` iterates events, yielding progress text + locked UI on `Progress` and the final wav on `Result`. Stop and failure are intentionally asymmetric: stop yields a `Result(was_stopped=True)` with the partial wav, failure raises and discards everything.

**Tech Stack:** Python 3.12, Gradio 6.x, voxcpm 2.0.2, numpy, soundfile, pytest.

**Spec:** `docs/superpowers/specs/2026-04-27-flatten-generation-tab-design.md`

---

## File structure

**Modified:**
- `src/voxcpm_tts_tool/generation.py` — add `Progress`, `Result`, `_voice_for_segment`, `run_generation`. Keep existing `build_generate_kwargs` and `synthesize_voice_preview`.
- `src/voxcpm_tts_tool/app_state.py` — delete fields `gen_chunks`, `gen_audio`, `gen_status`, `gen_errors`, `gen_queue`, `gen_running_idx`, method `reset_generation`, and the `if TYPE_CHECKING: from .chunking import ChunkRow` import. Keep `gen_stop_flag`.
- `app.py` — rewrite Generation tab block to a flat layout; remove queue / table imports and handlers.
- `tests/test_generation.py` — add tests for `Progress`, `Result`, `_voice_for_segment`, `run_generation`.
- `tests/test_app_state.py` — delete tests asserting deleted fields.

**Deleted:**
- `src/voxcpm_tts_tool/chunking.py`
- `src/voxcpm_tts_tool/generation_queue.py`
- `tests/test_chunking.py`
- `tests/test_generation_queue.py`

**Untouched:**
- `src/voxcpm_tts_tool/script_parser.py` (`localize_non_lang_tags`, `parse_script` — used as-is)
- `src/voxcpm_tts_tool/long_text.py` (`split_for_generation`, `concat_waveforms` — both reused)
- `src/voxcpm_tts_tool/output_writer.py` (`write_output_wav` — reused for final wav write)
- `src/voxcpm_tts_tool/voice_library.py` (uncommitted seed_text / audio_upload changes left as-is — they belong to a different work item)

---

## Task 1: Add `Progress` and `Result` dataclasses

**Files:**
- Modify: `src/voxcpm_tts_tool/generation.py`
- Test: `tests/test_generation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:/Users/vibecoding/workspace/wys/VoxCPM-TTS-Tool
.venv/Scripts/python.exe -m pytest tests/test_generation.py::test_progress_dataclass_has_done_and_total -v
```

Expected: `ImportError: cannot import name 'Progress' from 'voxcpm_tts_tool.generation'`

If `pytest` is missing in `.venv`, install: `.venv/Scripts/python.exe -m pip install pytest pytest-mock`. Pytest is listed in `requirements.txt` as a dev dep — installing it in this venv is intentional, not a side quest.

- [ ] **Step 3: Add the dataclasses**

Edit `src/voxcpm_tts_tool/generation.py`:

After the existing imports block (around line 14, after `from .voice_library import Voice`), add:

```python
from dataclasses import dataclass


@dataclass
class Progress:
    """Emitted by run_generation after each successful SDK call."""
    done: int        # SDK calls completed so far
    total: int       # SDK calls planned for this run


@dataclass
class Result:
    """Final event from run_generation — partial on stop, complete on success.

    Failure does NOT yield a Result; the underlying exception propagates instead.
    """
    wav: np.ndarray
    sample_rate: int
    was_stopped: bool = False    # True iff stop_flag triggered the early return
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py -v -k "progress_dataclass or result_dataclass"
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/generation.py tests/test_generation.py
git commit -m "feat(generation): add Progress/Result dataclasses for run_generation event stream"
```

---

## Task 2: Add `_voice_for_segment` helper

**Files:**
- Modify: `src/voxcpm_tts_tool/generation.py`
- Test: `tests/test_generation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generation.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py -v -k "voice_for_segment"
```

Expected: 4 FAIL with `ImportError: cannot import name '_voice_for_segment'`

- [ ] **Step 3: Implement `_voice_for_segment`**

Add to `src/voxcpm_tts_tool/generation.py`, after `Result` dataclass:

```python
def _voice_for_segment(
    name: str,
    *,
    default_voice: Voice,
    library,
) -> Voice:
    """Resolve a parser-emitted voice name to a Voice.

    Parser invariant: ``name`` is either ``default_voice.name`` or the
    canonical name of a library voice (case-canonicalized). Default-voice
    name match short-circuits library lookup so that an ephemeral default
    (id starting with ``__``, never in the library) still resolves.

    Raises ValueError if the name resolves to nothing — this is unreachable
    given the parser invariant, but defensive so a typo upstream doesn't
    silently fall through to a no-voice generation.
    """
    if name == default_voice.name:
        return default_voice
    v = library.find_by_name(name)
    if v is None:
        raise ValueError(f"voice not found: {name!r}")
    return v
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py -v -k "voice_for_segment"
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/generation.py tests/test_generation.py
git commit -m "feat(generation): add _voice_for_segment name-based resolver"
```

---

## Task 3: `run_generation` — single-voice happy path

**Files:**
- Modify: `src/voxcpm_tts_tool/generation.py`
- Test: `tests/test_generation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py::test_run_generation_single_voice_two_terminator_chunks -v
```

Expected: FAIL with `ImportError: cannot import name 'run_generation'`

- [ ] **Step 3: Implement `run_generation`**

Add to `src/voxcpm_tts_tool/generation.py`:

```python
from typing import Callable, Iterator

from .long_text import concat_waveforms, split_for_generation
from .script_parser import localize_non_lang_tags, parse_script


def run_generation(
    script: str,
    *,
    library,
    default_voice: Voice,
    model,
    audio_root: str,
    zipenhancer_loaded: bool,
    char_budget: int,
    cfg_value: float,
    inference_timesteps: int,
    stop_flag: Callable[[], bool],
) -> Iterator:
    """Stream Progress events per chunk + a final Result.

    Stop semantics: when ``stop_flag()`` returns True, yield
    ``Result(concat(done_so_far), sample_rate, was_stopped=True)`` and return.
    Failure semantics: any exception from ``model.generate`` propagates;
    no Result is yielded and partial wavs are discarded.

    See spec §"Stop vs failure" for why these are intentionally asymmetric.
    """
    sample_rate = int(getattr(model, "sample_rate", 16000))

    script = localize_non_lang_tags(script)

    by_name = {v.name: v for v in library.list_voices()}
    is_ephemeral_default = default_voice.id.startswith("__")
    if not is_ephemeral_default:
        by_name[default_voice.name] = default_voice

    segments, _warnings = parse_script(
        script,
        default_voice=default_voice.name,
        known_names=list(by_name.keys()),
    )

    # Pre-compute total chunk count for the Progress denominator.
    plan: list[tuple[Voice, str, str | None]] = []
    for seg in segments:
        voice = _voice_for_segment(seg.voice_name, default_voice=default_voice, library=library)
        for chunk in split_for_generation(seg.text, char_budget=char_budget):
            plan.append((voice, chunk, seg.control))
    total = len(plan)

    if total == 0:
        yield Result(wav=concat_waveforms([]), sample_rate=sample_rate)
        return

    all_wavs: list[np.ndarray] = []
    for i, (voice, chunk, ctrl) in enumerate(plan, start=1):
        if stop_flag():
            yield Result(wav=concat_waveforms(all_wavs),
                         sample_rate=sample_rate, was_stopped=True)
            return
        kwargs = build_generate_kwargs(
            voice, chunk,
            zipenhancer_loaded=zipenhancer_loaded,
            audio_root=audio_root,
            script_control=ctrl,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        wav = model.generate(**kwargs)
        all_wavs.append(wav)
        yield Progress(done=i, total=total)

    yield Result(wav=concat_waveforms(all_wavs), sample_rate=sample_rate)
```

Note: `parse_script` and `localize_non_lang_tags` are imported at the top of the function block (after the existing `from .voice_library import Voice`); `concat_waveforms` is in `long_text.py`. The `_warnings` are intentionally swallowed — the handler invokes `parse_script` separately to surface warnings before calling this generator (see Task 11).

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py::test_run_generation_single_voice_two_terminator_chunks -v
```

Expected: PASS

- [ ] **Step 5: Run all generation tests to confirm no regressions**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py -v
```

Expected: All previous build_generate_kwargs / synthesize_voice_preview tests still PASS, plus the new ones.

- [ ] **Step 6: Commit**

```bash
git add src/voxcpm_tts_tool/generation.py tests/test_generation.py
git commit -m "feat(generation): restore run_generation as Progress|Result event stream"
```

---

## Task 4: `run_generation` — multi-voice + script `(control)` round-trip

**Files:**
- Test: `tests/test_generation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generation.py`:

```python
def test_run_generation_multi_voice_with_script_control(tmp_path, project_root):
    """`<bob>你好<alice>(温柔)再见` → 2 segments → 2 SDK calls.
    Bob uses hifi (no script_control), Alice uses clone (script_control='温柔')."""
    import numpy as np
    from voxcpm_tts_tool.generation import Result, run_generation
    from voxcpm_tts_tool.voice_library import Voice

    # Library has bob (hifi) and alice (clone). Default voice is alice.
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

    # Two SDK calls; final event is Result.
    assert len([e for e in events if isinstance(e, Result)]) == 1
    assert len(model.calls) == 2

    # Bob call: hifi-style (prompt_wav_path + prompt_text), text == "你好"
    bob_call = model.calls[0]
    assert bob_call["text"] == "你好"
    assert "prompt_wav_path" in bob_call
    assert bob_call["prompt_text"] == "bob seed"

    # Alice call: script_control wins → clone-style with (温柔) prefix
    alice_call = model.calls[1]
    assert alice_call["text"] == "(温柔)再见"
    assert "prompt_wav_path" not in alice_call
    assert "reference_wav_path" in alice_call
```

- [ ] **Step 2: Run test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py::test_run_generation_multi_voice_with_script_control -v
```

Expected: PASS (the implementation from Task 3 already handles this).

- [ ] **Step 3: Commit**

```bash
git add tests/test_generation.py
git commit -m "test(generation): cover multi-voice + script-control through run_generation"
```

---

## Task 5: `run_generation` — long-text chunk splitting via `split_for_generation`

**Files:**
- Test: `tests/test_generation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generation.py`:

```python
def test_run_generation_splits_long_line_into_chunks_by_budget():
    """A single line with multiple terminators exceeds char_budget; each
    sentence becomes its own SDK call."""
    import numpy as np
    from voxcpm_tts_tool.generation import Progress, Result, run_generation
    from voxcpm_tts_tool.voice_library import Voice
    from voxcpm_tts_tool.long_text import split_for_generation

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
```

- [ ] **Step 2: Run test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py::test_run_generation_splits_long_line_into_chunks_by_budget -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_generation.py
git commit -m "test(generation): cover long-text chunk splitting through run_generation"
```

---

## Task 6: `run_generation` — stop preserves partial output

**Files:**
- Test: `tests/test_generation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generation.py`:

```python
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

    # Flag flips to True after 2 SDK calls have completed (i.e. checked at
    # the top of iteration 3 → True → early Result).
    stop_state = {"flips_after": 2, "calls_seen": [0]}
    model = _RecordingModel()

    def _stop_flag():
        # Read how many SDK calls have happened so far.
        return len(model.calls) >= stop_state["flips_after"]

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
```

- [ ] **Step 2: Run test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py::test_run_generation_stop_yields_result_with_was_stopped_true -v
```

Expected: PASS (Task 3 implementation already covers this; this test pins the contract).

- [ ] **Step 3: Commit**

```bash
git add tests/test_generation.py
git commit -m "test(generation): pin stop-yields-Result(was_stopped=True) contract"
```

---

## Task 7: `run_generation` — failure propagates and discards everything

**Files:**
- Test: `tests/test_generation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generation.py`:

```python
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
        calls = 0
        def generate(self, **kwargs):
            import numpy as np
            type(self).calls += 1
            if type(self).calls == 3:
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
```

- [ ] **Step 2: Run test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py::test_run_generation_failure_propagates_no_result -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_generation.py
git commit -m "test(generation): pin failure-propagates-no-Result contract"
```

---

## Task 8: `run_generation` — empty / whitespace input

**Files:**
- Test: `tests/test_generation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generation.py`:

```python
def test_run_generation_empty_input_yields_only_empty_result():
    import numpy as np
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
    import numpy as np
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
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py -v -k "empty_input or whitespace_only"
```

Expected: 2 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_generation.py
git commit -m "test(generation): empty/whitespace input yields a single empty Result"
```

---

## Task 9: `run_generation` — ephemeral default voice (empty library)

**Files:**
- Test: `tests/test_generation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generation.py`:

```python
def test_run_generation_ephemeral_default_voice_with_empty_library():
    """Empty library + ephemeral default → single Result with one wav.
    _voice_for_segment must short-circuit on default name match because
    library.find_by_name would return None for `__default__`."""
    import numpy as np
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
```

- [ ] **Step 2: Run test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py::test_run_generation_ephemeral_default_voice_with_empty_library -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_generation.py
git commit -m "test(generation): cover ephemeral default voice + empty library"
```

---

## Task 10: Strip table fields from `AppState`

**Files:**
- Modify: `src/voxcpm_tts_tool/app_state.py`
- Modify: `tests/test_app_state.py`

- [ ] **Step 1: Read the current state**

```bash
.venv/Scripts/python.exe -m pytest tests/test_app_state.py -v
```

Note any tests that exercise `gen_chunks`, `gen_audio`, `gen_status`, `gen_errors`, `gen_queue`, `gen_running_idx`, or `reset_generation`. Those are the ones to delete in Step 3.

- [ ] **Step 2: Edit `src/voxcpm_tts_tool/app_state.py`**

Replace the file's `AppState` class body so it looks like:

```python
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
```

Also delete these from the file (they are now dead):
- The `if TYPE_CHECKING: from .chunking import ChunkRow` import block at the top of the file.
- The `from typing import TYPE_CHECKING` line if it has no other consumers.
- The `reset_generation` method.
- The `from dataclasses import dataclass, field` line should change back to `from dataclasses import dataclass` if `field` is no longer used (it isn't, after removing the `default_factory=...` fields).

- [ ] **Step 3: Delete dead tests in `tests/test_app_state.py`**

For each test enumerated in Step 1 that asserts deleted fields or `reset_generation`, delete it. Tests that exercise `default_voice()` or fields we're keeping (`gen_stop_flag`, paths, library, model, etc.) stay.

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_app_state.py -v
```

Expected: All remaining tests PASS, no errors about missing fields.

- [ ] **Step 5: Run full test suite to catch indirect breakage**

```bash
.venv/Scripts/python.exe -m pytest -x -q 2>&1 | tail -30
```

Expected: Only failures should be in `tests/test_chunking.py` and `tests/test_generation_queue.py` (still importing the to-be-deleted modules) — those get cleaned up in Task 12.

- [ ] **Step 6: Commit**

```bash
git add src/voxcpm_tts_tool/app_state.py tests/test_app_state.py
git commit -m "refactor(app_state): drop chunks/queue fields; keep gen_stop_flag for streaming"
```

---

## Task 11: Rewrite the Generation tab in `app.py`

**Files:**
- Modify: `app.py` (Generation tab block, ~lines 233–730)

This is the largest task. Do it in three sub-steps so the diff stays auditable.

- [ ] **Step 1: Replace imports**

In `app.py`, lines ~47–75, change:

```python
from voxcpm_tts_tool.generation import synthesize_voice_preview
from voxcpm_tts_tool.output_writer import write_output_wav
...
from voxcpm_tts_tool.chunking import split_for_table
from voxcpm_tts_tool.long_text import concat_waveforms
from voxcpm_tts_tool.generation_queue import compute_fresh_queue, enqueue_regen, resolve_voice, run_queue
from voxcpm_tts_tool.script_parser import NON_LANG_TAG_MAP_ZH
```

to:

```python
from voxcpm_tts_tool.generation import (
    Progress,
    Result,
    run_generation,
    synthesize_voice_preview,
)
from voxcpm_tts_tool.output_writer import write_output_wav
...
# (remove chunking import)
# (remove concat_waveforms import — run_generation handles concat internally)
# (remove generation_queue imports)
from voxcpm_tts_tool.script_parser import NON_LANG_TAG_MAP_ZH, localize_non_lang_tags, parse_script
```

Also delete any module-level `COL_*` index constants that were used by the deleted row-select handler (search `COL_ACTION`, `COL_VOICE`, `COL_CONTROL`, etc., and remove their definitions).

- [ ] **Step 2: Replace the Generation tab block (UI declaration + handlers)**

Locate the block beginning at:

```python
        # ---- Generation tab ----
        with gr.Tab(i18n.t("tab.generation", "zh")) as tab_gen:
            with gr.Tabs() as gen_subtabs:
                with gr.TabItem("输入", id="gen-input") as gen_input_tab:
                    ...
```

and ending at the `chunks_df.input(...)` wiring (around line 696–700, just before the `# ---- Voice library tab ----` block).

Replace the **entire block** (UI declaration + every Generation-tab handler / wiring statement up to but not including `# ---- Voice library tab ----`) with:

```python
        # ---- Generation tab (flat: one button → progress → one wav) ----
        with gr.Tab(i18n.t("tab.generation", "zh")) as tab_gen:
            default_voice_dd = gr.Dropdown(
                choices=voice_dropdown_choices(state.library.list_voices(), lang="zh", ephemeral=ephemeral),
                label=i18n.t("field.default_voice", "zh"),
            )
            text_box = gr.Textbox(label="文本", lines=10)
            with gr.Row():
                voice_picker = gr.Dropdown(
                    choices=[v.name for v in state.library.list_voices()],
                    label=i18n.t("btn.insert_voice", "zh"),
                )
                insert_voice_btn = gr.Button(i18n.t("btn.insert_voice", "zh"))
                tag_picker = gr.Dropdown(choices=NON_LANG_TAGS, label=i18n.t("btn.insert_tag", "zh"))
                insert_tag_btn = gr.Button(i18n.t("btn.insert_tag", "zh"))
            max_duration_slider = gr.Slider(
                minimum=10, maximum=30, step=1, value=20,
                label="单段最长时长（秒，× 4 ≈ 字符预算）",
            )
            with gr.Accordion(i18n.t("field.advanced", "zh"), open=False):
                cfg_value_slider = gr.Slider(
                    minimum=1.0, maximum=3.0, step=0.1, value=2.0,
                    label=i18n.t("field.cfg_value", "zh"),
                )
                inference_timesteps_slider = gr.Slider(
                    minimum=4, maximum=30, step=1, value=10,
                    label=i18n.t("field.inference_timesteps", "zh"),
                )
            generate_btn = gr.Button("开始生成", variant="primary")
            stop_btn = gr.Button("停止", variant="secondary")
            status_md = gr.Markdown(value="")
            audio_out = gr.Audio(
                label="结果", type="filepath", interactive=False,
            )
```

Then, **after** the existing `insert_tag_btn.click(...)` wiring (which stays in the cross-tab callback section further down — look for it around the end of the file's UI build), add the new handlers:

```python
        # ---- Generation handlers (insert _on_generate near the existing
        # insert_voice_btn.click / insert_tag_btn.click wiring) ----

        def _on_generate(default_id, text, max_duration, cfg_value, inference_timesteps):
            default = state.default_voice(default_id)

            # Pre-flight parse to surface unknown-tag warnings before locking UI.
            prepped = localize_non_lang_tags(text)
            by_name = {v.name: v for v in state.library.list_voices()}
            if not default.id.startswith("__"):
                by_name[default.name] = default
            _segs, warnings = parse_script(
                prepped, default_voice=default.name, known_names=list(by_name),
            )
            warn_md = "" if not warnings else "⚠ " + "; ".join(warnings) + "\n\n"

            state.gen_stop_flag = False  # arm
            try:
                yield (
                    gr.update(value=None),                      # audio_out: clear
                    warn_md + "▶ 准备生成…",
                    gr.update(interactive=False),               # generate_btn
                    gr.update(interactive=False),               # text_box
                    gr.update(interactive=False),               # default_voice_dd
                    gr.update(value="停止"),                     # stop_btn (label restore)
                )
                for ev in run_generation(
                        prepped,
                        library=state.library,
                        default_voice=default,
                        model=state.model,
                        audio_root=str(state.paths.root),
                        zipenhancer_loaded=state.zipenhancer_loaded,
                        char_budget=int(max_duration) * 4,
                        cfg_value=float(cfg_value),
                        inference_timesteps=int(inference_timesteps),
                        stop_flag=lambda: state.gen_stop_flag,
                ):
                    if isinstance(ev, Progress):
                        yield (
                            gr.update(),
                            warn_md + f"▶ 正在生成第 {ev.done}/{ev.total} 段",
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(),
                        )
                    else:  # Result
                        if ev.wav.size == 0:
                            yield (
                                gr.update(value=None),
                                warn_md + "ℹ 没有可生成的内容",
                                gr.update(interactive=True),
                                gr.update(interactive=True),
                                gr.update(interactive=True),
                                gr.update(value="停止"),
                            )
                        else:
                            out_path = write_output_wav(
                                ev.wav, sample_rate=ev.sample_rate,
                                outputs_dir=state.paths.outputs,
                            )
                            msg = ("⏸ 已停止，已导出前面段" if ev.was_stopped
                                   else "✅ 已完成")
                            yield (
                                gr.update(value=str(out_path)),
                                warn_md + msg,
                                gr.update(interactive=True),
                                gr.update(interactive=True),
                                gr.update(interactive=True),
                                gr.update(value="停止"),
                            )
            except Exception as e:
                traceback.print_exc()
                yield (
                    gr.update(value=None),
                    warn_md + f"❌ 生成失败：{e}",
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(value="停止"),
                )
            finally:
                state.gen_stop_flag = False

        generate_btn.click(
            _on_generate,
            inputs=[default_voice_dd, text_box, max_duration_slider,
                    cfg_value_slider, inference_timesteps_slider],
            outputs=[audio_out, status_md, generate_btn, text_box,
                     default_voice_dd, stop_btn],
            concurrency_id="gen",
            concurrency_limit=1,
        )

        def _on_stop():
            state.gen_stop_flag = True
            return gr.update(value="⏳ 停止中…")

        stop_btn.click(
            _on_stop,
            outputs=stop_btn,
            concurrency_id="control",
            concurrency_limit=None,
        )
```

The existing `insert_voice_btn.click(...)` and `insert_tag_btn.click(...)` wiring stays — they still target `text_box`, which still exists.

- [ ] **Step 3: Verify imports + syntax**

```bash
.venv/Scripts/python.exe -c "import ast; ast.parse(open('app.py', encoding='utf-8').read())"
```

Expected: No output (clean parse). Then check the imports block actually compiles:

```bash
.venv/Scripts/python.exe -c "from voxcpm_tts_tool.generation import Progress, Result, run_generation; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Run the test suite**

```bash
.venv/Scripts/python.exe -m pytest -x -q 2>&1 | tail -40
```

Expected: All `test_generation.py` and `test_app_state.py` PASS. The only remaining failures should be in `test_chunking.py` and `test_generation_queue.py` (those modules still exist; we delete in Task 12).

- [ ] **Step 5: Manual smoke (cannot be automated)**

Launch the app via `./run.ps1`. In the browser:
1. Open the page; confirm the Generation tab shows a flat layout (no inner sub-tabs).
2. Type `你好世界。` and click 开始生成 → audio player populates with a wav, status shows `✅ 已完成`.
3. Long input (e.g. 5 sentences) + click 开始生成 → status cycles through `▶ 正在生成第 X/N 段`.
4. Click 停止 mid-run → label flips to `⏳ 停止中…`, current chunk finishes, audio player gets a partial wav, status shows `⏸ 已停止，已导出前面段`.
5. Multi-voice script (`<bob>你好<alice>(温柔)再见` with bob and alice in library) → both chunks generate, audio is concatenation.
6. Empty input + click → status shows `ℹ 没有可生成的内容`, audio empty.
7. Confirm the unrelated tabs (Voice Library, Usage) still work unchanged.

If any step fails, fix and re-test.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat(ui): flatten Generation tab to streaming run_generation handler"
```

---

## Task 12: Delete obsolete modules and tests

**Files:**
- Delete: `src/voxcpm_tts_tool/chunking.py`
- Delete: `src/voxcpm_tts_tool/generation_queue.py`
- Delete: `tests/test_chunking.py`
- Delete: `tests/test_generation_queue.py`

- [ ] **Step 1: Delete the files**

```bash
git rm src/voxcpm_tts_tool/chunking.py src/voxcpm_tts_tool/generation_queue.py tests/test_chunking.py tests/test_generation_queue.py
```

- [ ] **Step 2: Run the full test suite**

```bash
.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -20
```

Expected: All tests PASS. No "module not found" errors anywhere.

- [ ] **Step 3: Search for any leftover references to the deleted symbols**

```bash
.venv/Scripts/python.exe -c "
import subprocess, sys
for sym in ['split_for_table', 'ChunkRow', 'compute_fresh_queue', 'enqueue_regen',
            'resolve_voice', 'run_queue', 'reset_generation', 'gen_chunks',
            'gen_audio', 'gen_status', 'gen_errors', 'gen_queue', 'gen_running_idx']:
    r = subprocess.run(['git', 'grep', '-n', sym], capture_output=True, text=True)
    if r.stdout.strip():
        print(f'--- {sym} ---'); print(r.stdout)
" 2>&1 | head -60
```

Expected: Empty output (or only references inside `docs/superpowers/specs/2026-04-26-...md` and the old `2026-04-26-...impl.md` plan, which are historical and stay).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: drop chunks-table scaffolding (chunking + generation_queue + their tests)"
```

---

## Task 13: End-to-end verification

**Files:** N/A (verification only)

- [ ] **Step 1: Full test run**

```bash
.venv/Scripts/python.exe -m pytest -v 2>&1 | tail -40
```

Expected: All tests PASS. Snapshot the final summary line (e.g. `==== 78 passed in 4.21s ====`).

- [ ] **Step 2: Manual smoke — end-to-end happy path**

Launch via `./run.ps1`. In the browser:
1. Verify Generation tab is flat (no sub-tabs).
2. Empty library + ephemeral default → text-only generate works (case 6 from spec).
3. Library has voices → multi-voice script generates correctly.
4. Long text → progress text cycles, final wav plays in browser, download works.
5. Click 停止 mid-run → partial wav saved, label restores after the run ends.
6. Voice Library tab unchanged (still creates, lists, plays, deletes).
7. Switch tabs back and forth — should be instant (the original "tab stuck" symptom was env-level, but verify the new UI doesn't reintroduce it).

- [ ] **Step 3: Final cleanup — only if needed**

If Step 1 or 2 surfaced anything, fix and re-test. No commit unless changes were made.

- [ ] **Step 4: Final commit (only if Step 3 needed changes)**

```bash
git add -A
git commit -m "fix: address smoke-test findings post-flatten"
```

---

## Self-review notes

- **Spec coverage:** UI shape (Tasks 11), `Progress`/`Result` (Task 1), `_voice_for_segment` (Task 2), `run_generation` algorithm (Task 3), tests cases 1–8 (Tasks 3–9), AppState cleanup (Task 10), code organization (Tasks 10–12), tests deletion (Tasks 10, 12). All spec sections traced.
- **Type consistency:** `Progress(done, total)`, `Result(wav, sample_rate, was_stopped=False)`, `_voice_for_segment(name, *, default_voice, library)`, `run_generation(script, *, library, default_voice, model, audio_root, zipenhancer_loaded, char_budget, cfg_value, inference_timesteps, stop_flag)` — used identically across tasks.
- **Stop vs failure asymmetry** is pinned by Tasks 6 + 7; the test names spell out the contract.
- **Pytest install** is treated as a one-line install in Task 1 step 2, not a side-quest task. Pytest is in `requirements.txt` as a dev dep.
- **Empty/whitespace + ephemeral default** are explicitly covered (Tasks 8–9) because they were the regression risks identified in the prior table-design spec.
