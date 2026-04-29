# Generation Table-View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-shot Generation tab (textarea → one big wav) with a parse-then-batch flow whose unit of work is a row in a chunks table; users review and regenerate per-row, then merge to one final wav.

**Architecture:** New `chunking` module emits one `ChunkRow` per (voice, line, char-budgeted chunk). New `generation_queue` module owns pure-Python state mutators and the `run_queue` generator (so the loop is unit-testable without Gradio). Existing `build_generate_kwargs` / `concat_waveforms` / `write_output_wav` / `split_for_generation` / `parse_script` / `localize_non_lang_tags` / `VoiceLibrary` are reused unchanged. Old `run_generation` + `GenerationResult` + `GenerationError` are deleted; `synthesize_voice_preview` stays.

**Tech Stack:** Python 3.11, Gradio 6.x (sub-tabs, dataframe, per-row select / .input events, concurrency groups), pytest, voxcpm 2.0.2 SDK, soundfile, numpy.

**Spec:** `docs/superpowers/specs/2026-04-26-generation-table-view-design.md`

---

## File structure

| Path | Purpose | Status |
|---|---|---|
| `src/voxcpm_tts_tool/chunking.py` | `ChunkRow` dataclass + `split_for_table()` (parse → table rows). | NEW (~90 lines) |
| `src/voxcpm_tts_tool/generation_queue.py` | `resolve_voice()`, `compute_fresh_queue()`, `enqueue_regen()`, `run_queue()` generator. Pure-Python; no Gradio imports. | NEW (~140 lines) |
| `src/voxcpm_tts_tool/app_state.py` | Add `gen_*` fields (queue / status / audio / errors / running_idx / stop_flag) and `reset_generation()` helper. | MODIFY |
| `src/voxcpm_tts_tool/generation.py` | Delete `run_generation`, `GenerationResult`, `GenerationError`. Keep `build_generate_kwargs`, `synthesize_voice_preview`. | MODIFY |
| `app.py` | Generation tab rewritten as two sub-tabs (`输入` / `分段`) with parse / generate-all / stop / per-row regen / per-row play / inline-edit / merge handlers. | MODIFY |
| `tests/test_chunking.py` | Cover all `split_for_table` behaviors per spec §Tests. | NEW |
| `tests/test_generation_queue.py` | `resolve_voice`, `compute_fresh_queue`, `enqueue_regen`, `run_queue` cleanup invariants. | NEW |
| `tests/test_generation.py` | Drop `run_generation` / `GenerationError` / `synthesize_voice_preview`-via-`run_generation` tests. Keep `build_generate_kwargs` and standalone `synthesize_voice_preview` tests. | MODIFY |

The split between `chunking.py` (parse-time) and `generation_queue.py` (runtime / loop / state mutation) keeps each file under ~150 lines and lets tests target the loop without standing up Gradio.

---

### Task 1: Extend `AppState` with generation-queue fields

**Files:**
- Modify: `src/voxcpm_tts_tool/app_state.py`
- Test: `tests/test_app_state.py` (create if absent)

- [ ] **Step 1: Inspect current AppState**

Read `src/voxcpm_tts_tool/app_state.py`. Note that `AppState` is a `@dataclass`; `field(default_factory=...)` is required for mutable defaults. The new fields must appear AFTER all existing required fields (no defaults before required fields rule).

- [ ] **Step 2: Write the failing test**

Create `tests/test_app_state.py`:

```python
from pathlib import Path

from voxcpm_tts_tool.app_state import AppState, AppPaths
from voxcpm_tts_tool.transcription import SenseVoiceTranscriber
from voxcpm_tts_tool.voice_library import VoiceLibrary


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


def test_appstate_gen_fields_default_to_empty(tmp_path):
    s = _state(tmp_path)
    assert s.gen_chunks == []
    assert s.gen_audio == {}
    assert s.gen_status == {}
    assert s.gen_errors == {}
    assert s.gen_queue == []
    assert s.gen_running_idx is None
    assert s.gen_stop_flag is False


def test_appstate_gen_fields_are_distinct_per_instance(tmp_path):
    """Mutable defaults must use field(default_factory=...) — otherwise two
    AppState instances would share the same list/dict."""
    a = _state(tmp_path)
    b = _state(tmp_path)
    a.gen_queue.append(0)
    a.gen_status[0] = "running"
    assert b.gen_queue == []
    assert b.gen_status == {}


def test_reset_generation_clears_all_gen_fields(tmp_path):
    s = _state(tmp_path)
    s.gen_chunks.append(object())
    s.gen_audio[0] = "/tmp/a.wav"
    s.gen_status[0] = "done"
    s.gen_errors[0] = "boom"
    s.gen_queue.append(0)
    s.gen_running_idx = 0
    s.gen_stop_flag = True

    s.reset_generation()

    assert s.gen_chunks == []
    assert s.gen_audio == {}
    assert s.gen_status == {}
    assert s.gen_errors == {}
    assert s.gen_queue == []
    assert s.gen_running_idx is None
    assert s.gen_stop_flag is False
```

- [ ] **Step 3: Run the tests, verify they fail**

Run: `pytest tests/test_app_state.py -v`
Expected: FAIL with `AttributeError: 'AppState' object has no attribute 'gen_chunks'`.

- [ ] **Step 4: Implement the fields and helper**

Edit `src/voxcpm_tts_tool/app_state.py`. Add `field` import, then append fields + helper to the existing `@dataclass class AppState`:

```python
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chunking import ChunkRow

# ... existing code ...

@dataclass
class AppState:
    paths: AppPaths
    library: VoiceLibrary
    model: object
    transcriber: SenseVoiceTranscriber
    zipenhancer_loaded: bool

    # ---- Generation tab queue state (added 2026-04-26) ----
    # See spec §Data model. ChunkRow is a forward reference to avoid
    # importing chunking at module load (chunking imports voice_library
    # which is fine, but keeps the dep direction one-way).
    gen_chunks: list["ChunkRow"] = field(default_factory=list)
    gen_audio: dict[int, str] = field(default_factory=dict)        # idx → wav path
    gen_status: dict[int, str] = field(default_factory=dict)       # idx → "pending"|"running"|"done"|"failed"
    gen_errors: dict[int, str] = field(default_factory=dict)       # idx → user-readable message
    gen_queue: list[int] = field(default_factory=list)             # PENDING idx only; never includes running
    gen_running_idx: int | None = None
    gen_stop_flag: bool = False

    def default_voice(self, requested_id: str | None) -> Voice:
        # ... existing body, unchanged ...

    def reset_generation(self) -> None:
        """Clear all gen_* fields. Called when user clicks 解析 to repopulate."""
        self.gen_chunks.clear()
        self.gen_audio.clear()
        self.gen_status.clear()
        self.gen_errors.clear()
        self.gen_queue.clear()
        self.gen_running_idx = None
        self.gen_stop_flag = False
```

Keep the existing `default_voice` method body verbatim — only the field block + new method are additions.

- [ ] **Step 5: Run all tests, verify they pass**

Run: `pytest tests/test_app_state.py -v`
Expected: PASS — three tests.

Run: `pytest -q`
Expected: every existing test still passes (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/voxcpm_tts_tool/app_state.py tests/test_app_state.py
git commit -m "feat(app_state): add gen_* queue fields + reset_generation() helper"
```

---

### Task 2: Create `chunking.py` with `ChunkRow` dataclass

**Files:**
- Create: `src/voxcpm_tts_tool/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunking.py`:

```python
from voxcpm_tts_tool.chunking import ChunkRow


def test_chunkrow_construction_full():
    row = ChunkRow(
        line_no=1, voice_id="v1", voice_name="alpha",
        text="hello", control="温柔",
    )
    assert row.line_no == 1
    assert row.voice_id == "v1"
    assert row.voice_name == "alpha"
    assert row.text == "hello"
    assert row.control == "温柔"


def test_chunkrow_control_defaults_to_none():
    row = ChunkRow(line_no=1, voice_id="v1", voice_name="a", text="x")
    assert row.control is None


def test_chunkrow_control_three_state():
    """control=None / "" / "text" must round-trip — these are the three
    distinguishable script syntaxes the parser produces."""
    a = ChunkRow(line_no=1, voice_id="v", voice_name="n", text="t", control=None)
    b = ChunkRow(line_no=1, voice_id="v", voice_name="n", text="t", control="")
    c = ChunkRow(line_no=1, voice_id="v", voice_name="n", text="t", control="温柔")
    assert a.control is None
    assert b.control == ""
    assert c.control == "温柔"
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: voxcpm_tts_tool.chunking`.

- [ ] **Step 3: Implement minimal `ChunkRow`**

Create `src/voxcpm_tts_tool/chunking.py`:

```python
"""Parse-time chunking: maps script text → list of ChunkRow per spec §Splitting.

`split_for_table` is added in a later task; this file currently exposes only
the dataclass that the queue / app modules import.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkRow:
    line_no: int
    voice_id: str         # canonical reference; see spec §Voice identity
    voice_name: str       # display + inline-edit cache
    text: str
    control: str | None = None
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS — three tests.

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/chunking.py tests/test_chunking.py
git commit -m "feat(chunking): add ChunkRow dataclass"
```

---

### Task 3: Implement `resolve_voice` helper

**Files:**
- Create: `src/voxcpm_tts_tool/generation_queue.py`
- Test: `tests/test_generation_queue.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generation_queue.py`:

```python
from pathlib import Path

import pytest

from voxcpm_tts_tool.app_state import ephemeral_default_voice
from voxcpm_tts_tool.generation_queue import resolve_voice
from voxcpm_tts_tool.voice_library import Voice, VoiceLibrary


@pytest.fixture
def lib(tmp_path):
    return VoiceLibrary(tmp_path / "voices")


def _make_voice(lib: VoiceLibrary, name: str, tmp_path: Path) -> Voice:
    upload = tmp_path / f"{name}.wav"
    upload.write_bytes(b"RIFF...")
    return lib.create(name=name, mode="design", audio_upload=str(upload))


def test_resolve_voice_ephemeral_default_short_circuits(lib):
    eph = ephemeral_default_voice()
    # Library is empty — the helper must not call find_by_id, since the
    # ephemeral id (`__default__`) was never persisted.
    assert resolve_voice(eph.id, default_voice=eph, library=lib) is eph


def test_resolve_voice_library_id_resolves(lib, tmp_path):
    bob = _make_voice(lib, "bob", tmp_path)
    eph = ephemeral_default_voice()
    got = resolve_voice(bob.id, default_voice=eph, library=lib)
    assert got is not None
    assert got.id == bob.id
    assert got.name == "bob"


def test_resolve_voice_returns_none_when_library_voice_deleted(lib, tmp_path):
    bob = _make_voice(lib, "bob", tmp_path)
    bob_id = bob.id
    eph = ephemeral_default_voice()
    lib.delete(bob_id)
    assert resolve_voice(bob_id, default_voice=eph, library=lib) is None


def test_resolve_voice_non_ephemeral_default_goes_through_library(lib, tmp_path):
    """A non-ephemeral default that gets deleted must NOT be returned from
    the cached default_voice — we want deletion to surface as None so the
    loop can mark the row failed with 'voice not found'."""
    bob = _make_voice(lib, "bob", tmp_path)
    cached_default = bob   # caller passes this as default_voice
    lib.delete(bob.id)
    # Library no longer has bob; the helper must NOT short-circuit just
    # because voice_id == default_voice.id, because default isn't ephemeral.
    assert resolve_voice(bob.id, default_voice=cached_default, library=lib) is None
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_generation_queue.py -v`
Expected: FAIL — `ModuleNotFoundError: voxcpm_tts_tool.generation_queue`.

- [ ] **Step 3: Implement `resolve_voice`**

Create `src/voxcpm_tts_tool/generation_queue.py`:

```python
"""Pure-Python state mutators and run-loop generator for the Generation tab.

Kept free of Gradio imports so the loop's cleanup invariants are unit-testable
with a fake model. See spec §Generation loop.
"""
from __future__ import annotations

from .voice_library import Voice, VoiceLibrary


def resolve_voice(
    voice_id: str,
    *,
    default_voice: Voice,
    library: VoiceLibrary,
) -> Voice | None:
    """Resolve a stored voice_id to a live Voice object, or None if missing.

    Short-circuits ONLY for an ephemeral default (id starts with `__`), whose
    id is never in the library. A non-ephemeral default still goes through
    library.find_by_id, so deletion is surfaced as None — the loop relies on
    this to mark rows `failed` with 'voice not found'.
    """
    if default_voice.id.startswith("__") and voice_id == default_voice.id:
        return default_voice
    return library.find_by_id(voice_id)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_generation_queue.py -v`
Expected: PASS — four tests.

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/generation_queue.py tests/test_generation_queue.py
git commit -m "feat(generation_queue): add resolve_voice helper"
```

---

### Task 4: Implement `compute_fresh_queue`

**Files:**
- Modify: `src/voxcpm_tts_tool/generation_queue.py`
- Modify: `tests/test_generation_queue.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generation_queue.py`:

```python
from voxcpm_tts_tool.chunking import ChunkRow
from voxcpm_tts_tool.generation_queue import compute_fresh_queue


def _row(idx_marker: int) -> ChunkRow:
    return ChunkRow(line_no=1, voice_id="v", voice_name="n",
                    text=f"chunk-{idx_marker}")


def test_compute_fresh_queue_all_pending_returns_all_in_order():
    rows = [_row(0), _row(1), _row(2)]
    q = compute_fresh_queue(rows, statuses={})
    assert q == [0, 1, 2]


def test_compute_fresh_queue_skips_failed_and_done():
    rows = [_row(0), _row(1), _row(2), _row(3)]
    statuses = {0: "pending", 1: "failed", 2: "done", 3: "pending"}
    q = compute_fresh_queue(rows, statuses=statuses)
    assert q == [0, 3]   # failed rows are NOT auto-requeued; user must regen


def test_compute_fresh_queue_treats_missing_status_as_pending():
    """A row that was never touched has no status entry; default = pending."""
    rows = [_row(0), _row(1)]
    q = compute_fresh_queue(rows, statuses={1: "done"})
    assert q == [0]
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_generation_queue.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_fresh_queue'`.

- [ ] **Step 3: Implement**

Append to `src/voxcpm_tts_tool/generation_queue.py`:

```python
from .chunking import ChunkRow


def compute_fresh_queue(
    chunks: list[ChunkRow],
    *,
    statuses: dict[int, str],
) -> list[int]:
    """Return the indices to enqueue on a fresh start, in row order.

    Per spec §Start / resume semantics: only `pending` rows. `failed` rows are
    NOT auto-requeued — the user must fix the row (inline-edit, which resets
    failed → pending) or click 重生 (which resets to pending and inserts at
    queue front). `done` rows likewise require explicit 重生.
    """
    return [
        i for i in range(len(chunks))
        if statuses.get(i, "pending") == "pending"
    ]
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_generation_queue.py -v`
Expected: PASS — three new tests + the four resolve_voice tests.

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/generation_queue.py tests/test_generation_queue.py
git commit -m "feat(generation_queue): add compute_fresh_queue (pending-only on fresh start)"
```

---

### Task 5: Implement `enqueue_regen`

**Files:**
- Modify: `src/voxcpm_tts_tool/generation_queue.py`
- Modify: `tests/test_generation_queue.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generation_queue.py`:

```python
from voxcpm_tts_tool.generation_queue import enqueue_regen


class _MiniState:
    """Minimal AppState surface used by enqueue_regen — avoids spinning up a
    full AppState fixture for queue-mutation tests."""
    def __init__(self):
        self.gen_chunks: list[ChunkRow] = []
        self.gen_audio: dict[int, str] = {}
        self.gen_status: dict[int, str] = {}
        self.gen_errors: dict[int, str] = {}
        self.gen_queue: list[int] = []
        self.gen_running_idx: int | None = None


def _state_with_n_rows(n: int) -> _MiniState:
    s = _MiniState()
    s.gen_chunks = [_row(i) for i in range(n)]
    return s


def test_enqueue_regen_idle_inserts_at_front_and_resets_status():
    s = _state_with_n_rows(3)
    s.gen_status = {0: "done", 1: "done", 2: "done"}
    s.gen_audio = {0: "/tmp/a.wav", 1: "/tmp/b.wav", 2: "/tmp/c.wav"}
    s.gen_errors = {}

    enqueue_regen(s, 1)

    assert s.gen_queue == [1]
    assert s.gen_status[1] == "pending"
    assert 1 not in s.gen_audio       # wav reference dropped
    assert 1 not in s.gen_errors


def test_enqueue_regen_running_row_inserts_at_front_anyway():
    """voxcpm has no cancel — the in-flight row will overwrite once, and the
    re-queue causes one MORE generate after that. User-visible: row flips
    back to pending, finishes in-flight run, then runs once more."""
    s = _state_with_n_rows(3)
    s.gen_running_idx = 1
    s.gen_status = {0: "done", 1: "running", 2: "pending"}
    s.gen_queue = [2]

    enqueue_regen(s, 1)

    assert s.gen_queue == [1, 2]
    # status stays running (in-flight call still going); the loop's per-row
    # finally will reconcile it. enqueue_regen does NOT yank running.


def test_enqueue_regen_already_in_queue_moves_to_front():
    s = _state_with_n_rows(4)
    s.gen_queue = [1, 2, 3]
    s.gen_status = {1: "pending", 2: "pending", 3: "pending"}

    enqueue_regen(s, 3)

    assert s.gen_queue == [3, 1, 2]   # 3 moved to front, no duplicate


def test_enqueue_regen_failed_row_resets_to_pending():
    s = _state_with_n_rows(2)
    s.gen_status = {0: "failed", 1: "done"}
    s.gen_errors = {0: "voice not found"}

    enqueue_regen(s, 0)

    assert s.gen_queue == [0]
    assert s.gen_status[0] == "pending"
    assert 0 not in s.gen_errors
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_generation_queue.py -v`
Expected: FAIL — `ImportError: cannot import name 'enqueue_regen'`.

- [ ] **Step 3: Implement**

Append to `src/voxcpm_tts_tool/generation_queue.py`:

```python
import os
from typing import Protocol


class _QueueState(Protocol):
    """Structural type covering the AppState surface enqueue_regen touches."""
    gen_audio: dict[int, str]
    gen_status: dict[int, str]
    gen_errors: dict[int, str]
    gen_queue: list[int]
    gen_running_idx: int | None


def enqueue_regen(state: _QueueState, idx: int) -> None:
    """Reset row `idx` and insert it at the front of the pending queue.

    Per spec §enqueue_regen:
      - If idx is currently running: the in-flight call has no cancel, so we
        leave its status alone (the per-row finally in run_queue will settle
        it) and just insert idx at queue index 0 so the loop picks it up next.
      - Else: drop wav + error, mark pending, remove from queue if already
        present, then insert at index 0.
    """
    # Drop any prior wav reference. We unlink the file too so the disk doesn't
    # leak a stale wav for a row whose "current" output is about to be replaced.
    prior = state.gen_audio.pop(idx, "")
    if prior:
        try:
            os.unlink(prior)
        except OSError:
            pass
    state.gen_errors.pop(idx, None)

    if state.gen_running_idx == idx:
        # Don't touch status — the running row's per-row finally owns it.
        # Insert at front so the loop picks it up after the in-flight call.
        if idx in state.gen_queue:
            state.gen_queue.remove(idx)
        state.gen_queue.insert(0, idx)
        return

    state.gen_status[idx] = "pending"
    if idx in state.gen_queue:
        state.gen_queue.remove(idx)
    state.gen_queue.insert(0, idx)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_generation_queue.py -v`
Expected: PASS — four enqueue_regen tests + previous tests.

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/generation_queue.py tests/test_generation_queue.py
git commit -m "feat(generation_queue): add enqueue_regen with running-row safety"
```

---

### Task 6: Implement `split_for_table` — single-line single-voice happy path

**Files:**
- Modify: `src/voxcpm_tts_tool/chunking.py`
- Modify: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chunking.py`:

```python
import pytest

from voxcpm_tts_tool.app_state import ephemeral_default_voice
from voxcpm_tts_tool.chunking import split_for_table
from voxcpm_tts_tool.voice_library import VoiceLibrary


@pytest.fixture
def lib(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    upload = tmp_path / "alpha.wav"
    upload.write_bytes(b"RIFF...")
    lib.create(name="alpha", mode="design", audio_upload=str(upload))
    return lib


def test_single_line_single_voice_one_row(lib):
    alpha = lib.find_by_name("alpha")
    rows, warnings = split_for_table(
        "你好世界",
        library=lib, default_voice=alpha, char_budget=80,
    )
    assert warnings == []
    assert len(rows) == 1
    assert rows[0].line_no == 1
    assert rows[0].voice_id == alpha.id
    assert rows[0].voice_name == "alpha"
    assert rows[0].text == "你好世界"
    assert rows[0].control is None
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_chunking.py::test_single_line_single_voice_one_row -v`
Expected: FAIL — `ImportError: cannot import name 'split_for_table'`.

- [ ] **Step 3: Implement minimal `split_for_table`**

Append to `src/voxcpm_tts_tool/chunking.py`:

```python
from .long_text import split_for_generation
from .script_parser import localize_non_lang_tags, parse_script
from .voice_library import Voice, VoiceLibrary


def split_for_table(
    script: str,
    *,
    library: VoiceLibrary,
    default_voice: Voice,
    char_budget: int,
) -> tuple[list[ChunkRow], list[str]]:
    """Parse `script` and emit one ChunkRow per (voice, line, char-budgeted chunk).

    See spec §Splitting for the full algorithm. This module reuses
    `localize_non_lang_tags`, `parse_script`, and `split_for_generation` —
    no duplicate logic.
    """
    # 1. Localize zh non-language tag labels to the SDK's English tokens.
    script = localize_non_lang_tags(script)

    # 2. Build name → Voice map for resolution. Ephemeral default is added
    #    AFTER parse so <__default__> stays unknown text (matches generation.py:243).
    is_ephemeral = default_voice.id.startswith("__")
    by_name: dict[str, Voice] = {v.name: v for v in library.list_voices()}
    if not is_ephemeral:
        by_name[default_voice.name] = default_voice

    # 3. Parse.
    segments, warnings = parse_script(
        script,
        default_voice=default_voice.name,
        known_names=list(by_name.keys()),
    )

    # 4 + 5. Add ephemeral default for post-parse resolution; the parser used
    #        it as the line-start name even though it wasn't a known_name.
    by_name.setdefault(default_voice.name, default_voice)

    # 6. Split each segment by char_budget; map each chunk → ChunkRow.
    rows: list[ChunkRow] = []
    for seg in segments:
        voice = by_name[seg.voice_name]   # always resolves; see spec §Voice identity
        chunks = split_for_generation(seg.text, char_budget=char_budget)
        for chunk in chunks:
            rows.append(ChunkRow(
                line_no=seg.line_no,
                voice_id=voice.id,
                voice_name=voice.name,
                text=chunk,
                control=seg.control,
            ))
    return rows, warnings
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_chunking.py::test_single_line_single_voice_one_row -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/chunking.py tests/test_chunking.py
git commit -m "feat(chunking): split_for_table happy-path single-line single-voice"
```

---

### Task 7: `split_for_table` — voice switch within a line

**Files:**
- Modify: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chunking.py`:

```python
@pytest.fixture
def lib_two_voices(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    for name in ("alpha", "bob"):
        upload = tmp_path / f"{name}.wav"
        upload.write_bytes(b"RIFF...")
        lib.create(name=name, mode="design", audio_upload=str(upload))
    return lib


def test_voice_switch_within_line_produces_multiple_rows(lib_two_voices):
    alpha = lib_two_voices.find_by_name("alpha")
    bob = lib_two_voices.find_by_name("bob")
    rows, warnings = split_for_table(
        "你好<bob>大家好",
        library=lib_two_voices, default_voice=alpha, char_budget=80,
    )
    assert warnings == []
    assert len(rows) == 2
    assert rows[0].voice_id == alpha.id and rows[0].text == "你好"
    assert rows[1].voice_id == bob.id and rows[1].text == "大家好"


def test_each_line_resets_to_default_voice(lib_two_voices):
    """Per script_parser.py:100 — newline resets to default voice. A <bob>
    on line 1 does NOT carry to line 2."""
    alpha = lib_two_voices.find_by_name("alpha")
    bob = lib_two_voices.find_by_name("bob")
    rows, _ = split_for_table(
        "<bob>line one\nline two",
        library=lib_two_voices, default_voice=alpha, char_budget=80,
    )
    assert len(rows) == 2
    assert rows[0].voice_id == bob.id and rows[0].line_no == 1
    assert rows[1].voice_id == alpha.id and rows[1].line_no == 2
```

- [ ] **Step 2: Run — should pass already (parser already supports this)**

Run: `pytest tests/test_chunking.py::test_voice_switch_within_line_produces_multiple_rows tests/test_chunking.py::test_each_line_resets_to_default_voice -v`
Expected: PASS — `split_for_table` already delegates to `parse_script`, which handles both cases.

- [ ] **Step 3: Commit**

```bash
git add tests/test_chunking.py
git commit -m "test(chunking): voice-switch-within-line + each-line-resets behavior"
```

---

### Task 8: `split_for_table` — long-text splitting + control three-state preservation

**Files:**
- Modify: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunking.py`:

```python
def test_long_line_split_by_char_budget(lib):
    alpha = lib.find_by_name("alpha")
    # Three sentences; budget 6 → each becomes its own chunk.
    rows, _ = split_for_table(
        "一二三。四五六。七八九。",
        library=lib, default_voice=alpha, char_budget=6,
    )
    assert len(rows) == 3
    assert all(r.voice_id == alpha.id for r in rows)
    assert [r.text for r in rows] == ["一二三。", "四五六。", "七八九。"]


def test_control_three_state_round_trips(lib):
    alpha = lib.find_by_name("alpha")
    rows, _ = split_for_table(
        "<alpha>none\n<alpha>()empty\n<alpha>(温柔)styled",
        library=lib, default_voice=alpha, char_budget=80,
    )
    assert len(rows) == 3
    assert rows[0].control is None     # script wrote `<alpha>` only
    assert rows[1].control == ""       # script wrote `()`
    assert rows[2].control == "温柔"   # script wrote `(温柔)`


def test_localize_non_lang_tags_runs_before_parse(lib):
    alpha = lib.find_by_name("alpha")
    rows, _ = split_for_table(
        "<alpha>[笑声]你好",
        library=lib, default_voice=alpha, char_budget=80,
    )
    assert len(rows) == 1
    assert "[laughing]" in rows[0].text
    assert "[笑声]" not in rows[0].text


def test_empty_or_whitespace_input(lib):
    alpha = lib.find_by_name("alpha")
    assert split_for_table("", library=lib, default_voice=alpha, char_budget=80) == ([], [])
    assert split_for_table("   \n\n", library=lib, default_voice=alpha, char_budget=80) == ([], [])
```

- [ ] **Step 2: Run — should pass already**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS — all built on the same delegation.

- [ ] **Step 3: Commit**

```bash
git add tests/test_chunking.py
git commit -m "test(chunking): long-text split, control three-state, tag localize, empty input"
```

---

### Task 9: `split_for_table` — empty library + ephemeral default + voice-id stability

**Files:**
- Modify: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunking.py`:

```python
def test_empty_library_uses_ephemeral_default(tmp_path):
    """Spec §Tests: 'Plain text under an empty library still produces a row
    with voice_id == default_voice.id'. This is the regression target most
    likely to break the empty-library happy path."""
    lib = VoiceLibrary(tmp_path / "voices")
    eph = ephemeral_default_voice()   # id == name == "__default__"
    rows, warnings = split_for_table(
        "你好",
        library=lib, default_voice=eph, char_budget=80,
    )
    assert warnings == []
    assert len(rows) == 1
    assert rows[0].voice_id == eph.id
    assert rows[0].voice_name == eph.name


def test_ephemeral_default_tag_is_unknown_text(tmp_path):
    """`<__default__>` must be preserved verbatim with a parser warning,
    NOT resolved as a voice switch."""
    lib = VoiceLibrary(tmp_path / "voices")
    eph = ephemeral_default_voice()
    rows, warnings = split_for_table(
        "<__default__>hello",
        library=lib, default_voice=eph, char_budget=80,
    )
    assert any("__default__" in w for w in warnings)
    assert len(rows) == 1
    assert "__default__" in rows[0].text     # tag preserved as text


def test_voice_id_stable_across_rename(lib):
    """Spec §Tests: 'parse against a library where one voice is named bob →
    row stores voice_id == bob.id. After library.update(name="robert"), the
    row's voice_id is still resolvable.'"""
    bob = lib.find_by_name("alpha")   # the lib fixture creates "alpha"
    rows, _ = split_for_table(
        "<alpha>hi",
        library=lib, default_voice=bob, char_budget=80,
    )
    bob_id = bob.id
    assert rows[0].voice_id == bob_id

    # Rename — the row's voice_id must still resolve.
    lib.update(bob_id, name="robert")
    from voxcpm_tts_tool.generation_queue import resolve_voice
    resolved = resolve_voice(rows[0].voice_id, default_voice=bob, library=lib)
    assert resolved is not None
    assert resolved.name == "robert"
```

- [ ] **Step 2: Run — should pass with current implementation**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS — `split_for_table` already adds the ephemeral default via `setdefault` after parse.

- [ ] **Step 3: Commit**

```bash
git add tests/test_chunking.py
git commit -m "test(chunking): empty-library ephemeral default + voice-id rename stability"
```

---

### Task 10: `run_queue` happy path

**Files:**
- Modify: `src/voxcpm_tts_tool/generation_queue.py`
- Modify: `tests/test_generation_queue.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generation_queue.py`:

```python
import numpy as np

from voxcpm_tts_tool.app_state import AppState, AppPaths, ephemeral_default_voice
from voxcpm_tts_tool.chunking import ChunkRow
from voxcpm_tts_tool.generation_queue import run_queue
from voxcpm_tts_tool.transcription import SenseVoiceTranscriber


class _Recorder:
    """Capture model.generate kwargs and return a tiny waveform."""
    def __init__(self):
        self.sample_rate = 16000
        self.calls: list[dict] = []
    def generate(self, **kw):
        self.calls.append(kw)
        return np.zeros(8, dtype=np.float32)


def _appstate(tmp_path):
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
    lib = VoiceLibrary(paths.voices)
    upload = tmp_path / "alpha.wav"
    upload.write_bytes(b"RIFF...")
    lib.create(name="alpha", mode="design", audio_upload=str(upload))
    return AppState(
        paths=paths, library=lib, model=_Recorder(),
        transcriber=SenseVoiceTranscriber.unavailable("test"),
        zipenhancer_loaded=False,
    )


def test_run_queue_happy_path_processes_all_rows(tmp_path):
    s = _appstate(tmp_path)
    alpha = s.library.find_by_name("alpha")
    s.gen_chunks = [
        ChunkRow(line_no=1, voice_id=alpha.id, voice_name=alpha.name, text="一"),
        ChunkRow(line_no=1, voice_id=alpha.id, voice_name=alpha.name, text="二"),
    ]
    s.gen_status = {0: "pending", 1: "pending"}
    s.gen_queue = [0, 1]

    # Drain the generator — it yields after each state transition.
    list(run_queue(
        s, library=s.library, default_voice=alpha, model=s.model,
        audio_root=str(s.paths.root), zipenhancer_loaded=False,
        cfg_value=2.0, inference_timesteps=10,
        outputs_dir=s.paths.outputs,
    ))

    assert s.gen_status == {0: "done", 1: "done"}
    assert s.gen_audio[0] and s.gen_audio[1]
    assert s.gen_running_idx is None
    assert s.gen_stop_flag is False
    assert s.gen_queue == []
    assert len(s.model.calls) == 2
    assert s.model.calls[0]["text"] == "一"
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_generation_queue.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_queue'`.

- [ ] **Step 3: Implement `run_queue`**

Append to `src/voxcpm_tts_tool/generation_queue.py`:

```python
import os
from pathlib import Path
from typing import Iterator

import soundfile as sf

from .generation import build_generate_kwargs


def _write_row_wav(waveform, sample_rate: int, outputs_dir: Path, idx: int) -> Path:
    """Write the row's wav to outputs/row-<idx>.wav (stable name, overwrite-safe)."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    path = outputs_dir / f"row-{idx}.wav"
    sf.write(str(path), waveform, samplerate=int(sample_rate), subtype="PCM_16")
    return path


def run_queue(
    state,
    *,
    library: VoiceLibrary,
    default_voice: Voice,
    model,
    audio_root: str,
    zipenhancer_loaded: bool,
    cfg_value: float,
    inference_timesteps: int,
    outputs_dir: Path,
) -> Iterator[int]:
    """Drain `state.gen_queue` one row at a time. Yields `idx` after each
    state transition so a Gradio handler can re-render the table.

    See spec §Generation loop for invariants. The outer try/finally guarantees
    `gen_stop_flag` reset even if an unexpected exception leaks past the
    per-row handler. The per-row try/except/finally guarantees `gen_running_idx`
    cleanup AND a fail-safe status assignment.
    """
    try:
        while state.gen_queue and not state.gen_stop_flag:
            idx = state.gen_queue.pop(0)
            state.gen_running_idx = idx
            state.gen_status[idx] = "running"
            yield idx

            try:
                row = state.gen_chunks[idx]
                voice = resolve_voice(row.voice_id,
                                      default_voice=default_voice, library=library)
                if voice is None:
                    state.gen_errors[idx] = f"voice not found: {row.voice_name}"
                    state.gen_status[idx] = "failed"
                    continue

                active_audio = getattr(voice, "audio", "") or voice.reference_audio
                if active_audio:
                    abs_audio = os.path.normpath(os.path.join(audio_root, active_audio))
                    if not os.path.exists(abs_audio):
                        state.gen_errors[idx] = f"voice audio missing at {abs_audio}"
                        state.gen_status[idx] = "failed"
                        continue

                kwargs = build_generate_kwargs(
                    voice, row.text,
                    zipenhancer_loaded=zipenhancer_loaded,
                    audio_root=audio_root,
                    script_control=row.control,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                )
                wav = model.generate(**kwargs)
                out_path = _write_row_wav(wav, model.sample_rate, outputs_dir, idx)
                state.gen_audio[idx] = str(out_path)
                state.gen_status[idx] = "done"
            except Exception as exc:
                preview = state.gen_chunks[idx].text[:40].replace("\n", " ")
                state.gen_errors[idx] = f"{exc} [text: {preview!r}]"
                state.gen_status[idx] = "failed"
            finally:
                # State cleanup first — must NOT be guarded by yield.
                state.gen_running_idx = None
                # Fail-safe: if no branch set a terminal status, mark failed.
                if state.gen_status.get(idx) == "running":
                    state.gen_status[idx] = "failed"
                    state.gen_errors.setdefault(idx, "row aborted before completion")
                # Best-effort UI yield. If the caller's handler raises during
                # re-render, the outer finally still runs.
                try:
                    yield idx
                except GeneratorExit:
                    raise
    finally:
        state.gen_stop_flag = False
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_generation_queue.py::test_run_queue_happy_path_processes_all_rows -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/generation_queue.py tests/test_generation_queue.py
git commit -m "feat(generation_queue): add run_queue happy-path generator"
```

---

### Task 11: `run_queue` — pre-flight: voice not found + audio missing

**Files:**
- Modify: `tests/test_generation_queue.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generation_queue.py`:

```python
def test_run_queue_voice_deleted_marks_failed_with_message(tmp_path):
    s = _appstate(tmp_path)
    alpha = s.library.find_by_name("alpha")
    s.gen_chunks = [
        ChunkRow(line_no=1, voice_id=alpha.id, voice_name="alpha", text="hi"),
    ]
    s.gen_status = {0: "pending"}
    s.gen_queue = [0]
    s.library.delete(alpha.id)   # voice gone before loop runs

    list(run_queue(
        s, library=s.library, default_voice=alpha, model=s.model,
        audio_root=str(s.paths.root), zipenhancer_loaded=False,
        cfg_value=2.0, inference_timesteps=10,
        outputs_dir=s.paths.outputs,
    ))

    assert s.gen_status[0] == "failed"
    assert "voice not found" in s.gen_errors[0]
    assert "alpha" in s.gen_errors[0]
    assert s.gen_running_idx is None
    assert s.model.calls == []   # SDK never called


def test_run_queue_audio_file_missing_marks_failed_with_message(tmp_path):
    s = _appstate(tmp_path)
    alpha = s.library.find_by_name("alpha")
    # Delete the audio file behind the library's back.
    audio_path = s.paths.root / alpha.audio
    os.remove(audio_path)

    s.gen_chunks = [ChunkRow(line_no=1, voice_id=alpha.id, voice_name="alpha", text="hi")]
    s.gen_status = {0: "pending"}
    s.gen_queue = [0]

    list(run_queue(
        s, library=s.library, default_voice=alpha, model=s.model,
        audio_root=str(s.paths.root), zipenhancer_loaded=False,
        cfg_value=2.0, inference_timesteps=10,
        outputs_dir=s.paths.outputs,
    ))

    assert s.gen_status[0] == "failed"
    assert "voice audio missing" in s.gen_errors[0]
    assert s.model.calls == []   # SDK never called


def test_run_queue_ephemeral_default_text_only_succeeds(tmp_path):
    """Empty library + ephemeral default → row resolves via short-circuit
    in resolve_voice → build_generate_kwargs takes the text-only branch."""
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
    s = AppState(
        paths=paths, library=VoiceLibrary(paths.voices), model=_Recorder(),
        transcriber=SenseVoiceTranscriber.unavailable("test"),
        zipenhancer_loaded=False,
    )
    eph = ephemeral_default_voice()
    s.gen_chunks = [ChunkRow(line_no=1, voice_id=eph.id, voice_name=eph.name, text="hi")]
    s.gen_status = {0: "pending"}
    s.gen_queue = [0]

    list(run_queue(
        s, library=s.library, default_voice=eph, model=s.model,
        audio_root=str(paths.root), zipenhancer_loaded=False,
        cfg_value=2.0, inference_timesteps=10,
        outputs_dir=paths.outputs,
    ))

    assert s.gen_status[0] == "done"
    assert s.gen_audio[0]
    assert s.model.calls == [{"text": "hi"}]
```

- [ ] **Step 2: Run, verify pass**

Run: `pytest tests/test_generation_queue.py -v`
Expected: PASS — pre-flight branches already implemented in Task 10.

- [ ] **Step 3: Commit**

```bash
git add tests/test_generation_queue.py
git commit -m "test(generation_queue): pre-flight voice-not-found + audio-missing + ephemeral default"
```

---

### Task 12: `run_queue` — SDK error and cleanup invariants

**Files:**
- Modify: `tests/test_generation_queue.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generation_queue.py`:

```python
class _BoomModel:
    sample_rate = 16000
    def generate(self, **kw):
        raise RuntimeError("model exploded")


def test_run_queue_sdk_error_records_message_and_cleans_up(tmp_path):
    s = _appstate(tmp_path)
    alpha = s.library.find_by_name("alpha")
    s.model = _BoomModel()
    s.gen_chunks = [ChunkRow(line_no=1, voice_id=alpha.id, voice_name="alpha", text="hello world")]
    s.gen_status = {0: "pending"}
    s.gen_queue = [0]

    list(run_queue(
        s, library=s.library, default_voice=alpha, model=s.model,
        audio_root=str(s.paths.root), zipenhancer_loaded=False,
        cfg_value=2.0, inference_timesteps=10,
        outputs_dir=s.paths.outputs,
    ))

    assert s.gen_status[0] == "failed"
    assert "model exploded" in s.gen_errors[0]
    assert "hello world" in s.gen_errors[0]   # chunk preview embedded
    # Cleanup invariants — see spec §Generation loop outer/per-row finally.
    assert s.gen_running_idx is None
    assert s.gen_stop_flag is False


def test_run_queue_outer_finally_resets_stop_flag_when_set_mid_loop(tmp_path):
    s = _appstate(tmp_path)
    alpha = s.library.find_by_name("alpha")
    s.gen_chunks = [
        ChunkRow(line_no=1, voice_id=alpha.id, voice_name="alpha", text=f"chunk-{i}")
        for i in range(3)
    ]
    s.gen_status = {0: "pending", 1: "pending", 2: "pending"}
    s.gen_queue = [0, 1, 2]

    # Drain manually so we can flip stop_flag between iterations.
    gen = run_queue(
        s, library=s.library, default_voice=alpha, model=s.model,
        audio_root=str(s.paths.root), zipenhancer_loaded=False,
        cfg_value=2.0, inference_timesteps=10,
        outputs_dir=s.paths.outputs,
    )
    next(gen)   # row 0 starts running
    next(gen)   # row 0 settles
    s.gen_stop_flag = True
    list(gen)   # exhaust — outer finally must reset the flag

    assert s.gen_status[0] == "done"
    assert 1 not in s.gen_audio   # never started
    assert s.gen_running_idx is None
    assert s.gen_stop_flag is False   # OUTER finally guarantees this
    assert s.gen_queue == [1, 2]      # remaining rows preserved for resume
```

- [ ] **Step 2: Run, verify pass**

Run: `pytest tests/test_generation_queue.py -v`
Expected: PASS — both behaviors implemented in Task 10.

- [ ] **Step 3: Commit**

```bash
git add tests/test_generation_queue.py
git commit -m "test(generation_queue): SDK error capture + outer finally stop-flag reset"
```

---

### Task 13: Delete `run_generation` / `GenerationResult` / `GenerationError`

**Files:**
- Modify: `src/voxcpm_tts_tool/generation.py`
- Modify: `tests/test_generation.py`
- Modify: `app.py`

- [ ] **Step 1: Drop `run_generation`-based tests from test_generation.py**

Delete from `tests/test_generation.py`:
- The whole import block `from voxcpm_tts_tool.generation import GenerationError, run_generation` (line ~251)
- `_make_lib_with_design` helper (only used by deleted tests)
- `test_run_generation_calls_model_per_chunk`
- `test_empty_script_raises`
- `test_missing_voice_audio_aborts`
- `test_one_segment_failure_carries_index`
- `test_ephemeral_default_voice_not_addressable_via_script_switch`

Keep all `build_generate_kwargs` tests, all standalone `synthesize_voice_preview` tests, and the file's first import block (`from voxcpm_tts_tool.generation import build_generate_kwargs`).

- [ ] **Step 2: Run remaining test_generation.py tests**

Run: `pytest tests/test_generation.py -v`
Expected: PASS — all surviving tests still pass.

- [ ] **Step 3: Delete `run_generation`, `GenerationResult`, `GenerationError`**

In `src/voxcpm_tts_tool/generation.py` delete:
- `class GenerationResult` (~lines 121-125)
- `class GenerationError` (~lines 128-129)
- `def run_generation` and its body (~lines 218-298)

Also remove now-unused imports at top of file: `from .long_text import concat_waveforms, split_for_generation` → keep only what's still used. Verify `from .script_parser import localize_non_lang_tags, parse_script` is removable (no other refs in the file). Verify `from .voice_library import VoiceLibrary` is removable (only `Voice` is still used). Run `grep` mentally / via tooling to confirm.

- [ ] **Step 4: Update `app.py` import + remove old `_on_generate` handler**

The app.py imports list (line ~51):
```python
from voxcpm_tts_tool.generation import GenerationError, run_generation, synthesize_voice_preview
```
becomes
```python
from voxcpm_tts_tool.generation import synthesize_voice_preview
```

Remove `_on_generate`, the `generate_btn.click(...)` block, and the now-unused `audio_out` / `log_out` / `generate_btn` widgets. (Their replacements are added in Task 14.)

This will leave `app.py` temporarily with a Generation tab that has the script box but no generate button — that's fine; Task 14 immediately rebuilds the tab.

- [ ] **Step 5: Run all tests**

Run: `pytest -q`
Expected: PASS — chunking, generation_queue, app_state, surviving generation tests all green.

- [ ] **Step 6: Commit**

```bash
git add src/voxcpm_tts_tool/generation.py tests/test_generation.py app.py
git commit -m "refactor: delete run_generation / GenerationResult / GenerationError"
```

---

### Task 14: Build the new Generation tab UI shell

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace the Generation tab block with sub-tabs**

Inside `build_ui`, replace the existing `with gr.Tab(i18n.t("tab.generation", "zh")) as tab_gen:` block (the one that ends right before `# ---- Voice library tab ----`) with:

```python
with gr.Tab(i18n.t("tab.generation", "zh")) as tab_gen:
    with gr.Tabs() as gen_subtabs:
        with gr.TabItem("输入", id="gen-input") as gen_input_tab:
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
            parse_btn = gr.Button("解析为分段表格", variant="primary")
        with gr.TabItem("分段", id="gen-table") as gen_table_tab:
            chunks_df = gr.Dataframe(
                headers=["#", "音色", "风格", "文本", "状态", "排队", "操作"],
                value=[],
                interactive=True,
                wrap=True,
            )
            current_status = gr.Markdown(value="")
            chunk_audio = gr.Audio(
                type="filepath", autoplay=True, interactive=False,
                show_label=False, container=False,
                elem_id="voxcpm-chunk-audio",
            )
            with gr.Row():
                generate_all_btn = gr.Button("开始生成", variant="primary")
                stop_btn = gr.Button("停止", variant="secondary")
                merge_dl_btn = gr.Button("合并下载", variant="secondary")
            merged_audio_out = gr.Audio(
                label="合并结果", type="filepath",
                interactive=False, show_download_button=True,
            )
```

Add CSS to hide the per-row chunk_audio (matching the voice-library pattern):

```python
_VOICE_TAB_CSS = """
#voxcpm-listen-audio { position: absolute; left: -9999px; width: 1px; height: 1px;
                       overflow: hidden; opacity: 0; pointer-events: none; }
#voxcpm-chunk-audio  { position: absolute; left: -9999px; width: 1px; height: 1px;
                       overflow: hidden; opacity: 0; pointer-events: none; }
"""
```

- [ ] **Step 2: Re-wire the existing `insert_voice_btn` / `insert_tag_btn` callbacks**

The existing `_on_insert_voice` / `_on_insert_tag` closures already operate on `script_box`. Rename references: `script_box` → `text_box`. The two `.click(...)` registrations follow.

- [ ] **Step 3: Launch the app and verify layout**

Run: `python app.py --port 8808` and open the Generation tab. Manually verify:
- The 输入 sub-tab is visible with text box, voice picker, tag picker, max duration slider, advanced accordion, and "解析为分段表格" button.
- The 分段 sub-tab opens to an empty dataframe with the seven columns, the status markdown, the three buttons, and the (visually hidden) per-row audio.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(ui): scaffold new Generation tab with sub-tabs + chunks dataframe"
```

---

### Task 15: Wire `parse_btn` → `split_for_table` → populate table + switch to 分段 sub-tab

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the parse handler**

In `build_ui`, before the wiring section, add a helper that renders rows for the dataframe:

```python
from voxcpm_tts_tool.chunking import split_for_table
from voxcpm_tts_tool.generation_queue import resolve_voice

def _render_chunks_df(state, default_voice):
    """Build the dataframe rows from state.gen_chunks + state.gen_status.
    Display name comes from resolve_voice(voice_id) so a rename done after
    parse is reflected without re-parse (spec §Voice identity)."""
    rows = []
    queue_pos = {idx: pos for pos, idx in enumerate(state.gen_queue)}
    for i, ch in enumerate(state.gen_chunks):
        v = resolve_voice(ch.voice_id, default_voice=default_voice, library=state.library)
        display_name = v.name if v is not None else f"{ch.voice_name} (已删除)"
        status = state.gen_status.get(i, "pending")
        status_glyph = {
            "pending": "⏳", "running": "⚙", "done": "✓", "failed": "✗",
        }.get(status, status)
        if i == state.gen_running_idx:
            queue_cell = "▶"
        elif i in queue_pos:
            queue_cell = str(queue_pos[i] + 1)
        else:
            queue_cell = ""
        action = "▶ 播放 / 重生" if status == "done" else "重生"
        rows.append([
            str(i + 1), display_name,
            ch.control if ch.control is not None else "",
            ch.text, status_glyph, queue_cell, action,
        ])
    return rows
```

Then wire `parse_btn`:

```python
def _on_parse(default_id, text, max_duration):
    default = state.default_voice(default_id)
    char_budget = int(max_duration) * 4
    # If non-pending rows exist, the user is overwriting prior work — for
    # v1 we just clear silently; the spec mentions a confirm-toast, deferred.
    state.reset_generation()
    rows, warnings = split_for_table(
        text, library=state.library, default_voice=default, char_budget=char_budget,
    )
    state.gen_chunks = rows
    state.gen_status = {i: "pending" for i in range(len(rows))}
    df_rows = _render_chunks_df(state, default)
    status_md = ("✅ 已解析 %d 行" % len(rows)) if not warnings else (
        "⚠ 解析警告：\n- " + "\n- ".join(warnings)
    )
    return df_rows, status_md, gr.Tabs(selected="gen-table")

parse_btn.click(
    _on_parse,
    inputs=[default_voice_dd, text_box, max_duration_slider],
    outputs=[chunks_df, current_status, gen_subtabs],
)
```

- [ ] **Step 2: Manual smoke test**

Run: `python app.py`
Type `<alpha>第一句。<alpha>(温柔)第二句` (substituting an existing voice name) into 文本 → click 解析为分段表格. Expect the table to populate with two rows and the UI to switch to the 分段 sub-tab.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(ui): wire parse_btn to split_for_table + switch to table sub-tab"
```

---

### Task 16: Wire `generate_all_btn` + `stop_btn` (queue loop with concurrency groups)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the loop wiring**

```python
from voxcpm_tts_tool.generation_queue import run_queue, compute_fresh_queue, enqueue_regen

def _run_loop(default_id, cfg_value, inference_timesteps):
    """The Gradio handler that wraps run_queue. Yields table updates between
    rows. The outer generator's finally guarantees stop_flag reset; Gradio
    handles GeneratorExit on disconnect."""
    default = state.default_voice(default_id)

    # Resume vs fresh start.
    if not state.gen_queue:
        state.gen_queue = compute_fresh_queue(state.gen_chunks, statuses=state.gen_status)

    if not state.gen_queue:
        yield (
            _render_chunks_df(state, default),
            "ℹ 没有待生成的行（请先解析或点击 重生）",
            gr.update(interactive=True),   # generate_all_btn
        )
        return

    yield (
        _render_chunks_df(state, default),
        "▶ 开始生成",
        gr.update(interactive=False),
    )
    for _idx in run_queue(
        state, library=state.library, default_voice=default, model=state.model,
        audio_root=str(state.paths.root),
        zipenhancer_loaded=state.zipenhancer_loaded,
        cfg_value=float(cfg_value),
        inference_timesteps=int(inference_timesteps),
        outputs_dir=state.paths.outputs,
    ):
        running = state.gen_running_idx
        status_md = (
            f"⚙ 正在生成第 {running + 1} 行" if running is not None
            else "▶ 生成中…"
        )
        yield (
            _render_chunks_df(state, default),
            status_md,
            gr.update(interactive=False),
        )
    yield (
        _render_chunks_df(state, default),
        "✅ 全部完成" if all(state.gen_status.get(i) == "done"
                            for i in range(len(state.gen_chunks)))
        else "⚠ 已结束（有失败行，请检查）",
        gr.update(interactive=True),
    )

generate_all_btn.click(
    _run_loop,
    inputs=[default_voice_dd, cfg_value_slider, inference_timesteps_slider],
    outputs=[chunks_df, current_status, generate_all_btn],
    concurrency_id="gen",
    concurrency_limit=1,
)

def _on_stop():
    """Set the stop flag. Lives in concurrency_id="control" so it runs even
    while `gen` is busy. Per spec §Stop semantics: just flips the flag."""
    state.gen_stop_flag = True
    return gr.update(value="⏳ 停止中…")

stop_btn.click(
    _on_stop,
    outputs=stop_btn,
    concurrency_id="control",
    concurrency_limit=None,
)
```

- [ ] **Step 2: Manual smoke test — happy path**

Run app, parse a 2-3 row script, click 开始生成. Expect each row's status to flip to ⚙ → ✓ in turn, the running button to disable then re-enable, and `current_status` to update.

- [ ] **Step 3: Manual smoke test — stop**

Parse a 5-row script, click 开始生成, then click 停止 mid-batch. Expect the in-flight row to complete, the loop to exit, the input tab to remain accessible (lock comes in Task 21), and `current_status` to read "已结束".

- [ ] **Step 4: Manual smoke test — resume**

After the previous stop test, click 开始生成 again. Expect the loop to pick up the un-processed rows from the existing `gen_queue` (no rebuild).

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(ui): wire generate_all_btn + stop_btn with concurrency groups"
```

---

### Task 17: Wire per-row click dispatcher (regenerate + play)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the row-select handler**

Add at top of build_ui, before any handlers:

```python
COL_INDEX, COL_VOICE, COL_CONTROL, COL_TEXT, COL_STATUS, COL_QUEUE, COL_ACTION = range(7)
```

Define the action dispatcher. The 操作 column does double duty (▶ 播放 / 重生); we treat any click in that column as: if status is `done` AND the action label was "▶ 播放 / 重生", play; otherwise regen. We also support clicks on the `状态` column for play (idempotent UX).

```python
def _on_row_select(default_id, evt: gr.SelectData):
    """Per-row click dispatcher. Column 6 (操作) is the primary action;
    state mutations enqueue regen or stage the per-row audio for autoplay.
    """
    default = state.default_voice(default_id)
    if isinstance(evt.index, (list, tuple)):
        row_idx, col_idx = evt.index[0], evt.index[1]
    else:
        row_idx, col_idx = evt.index, 0
    if not (0 <= row_idx < len(state.gen_chunks)):
        return _render_chunks_df(state, default), gr.update(), ""
    status = state.gen_status.get(row_idx, "pending")

    if col_idx == COL_ACTION:
        # If the row has audio and status is done, treat as "play"; otherwise regen.
        if status == "done" and row_idx in state.gen_audio:
            state._pending_chunk_path = state.gen_audio[row_idx]
            return (
                _render_chunks_df(state, default),
                gr.update(value=None),    # clear first; .then sets it (autoplay re-fire)
                f"🔊 试听第 {row_idx + 1} 行",
            )
        # Regen.
        enqueue_regen(state, row_idx)
        return (
            _render_chunks_df(state, default),
            gr.update(),
            f"♻ 已将第 {row_idx + 1} 行加入队列首位（点 开始生成 处理）",
        )
    # Other columns: no-op (inline edit handled by .input).
    return _render_chunks_df(state, default), gr.update(), ""

def _apply_pending_chunk():
    """Phase 2: push the staged path into chunk_audio so the browser sees a
    fresh value and autoplay re-fires."""
    pending = getattr(state, "_pending_chunk_path", "")
    state._pending_chunk_path = ""
    if pending:
        return gr.update(value=pending)
    return gr.update()

if not hasattr(state, "_pending_chunk_path"):
    state._pending_chunk_path = ""

chunks_df.select(
    _on_row_select,
    inputs=[default_voice_dd],
    outputs=[chunks_df, chunk_audio, current_status],
    concurrency_id="enqueue",
    concurrency_limit=None,
).then(
    _apply_pending_chunk,
    outputs=chunk_audio,
)
```

- [ ] **Step 2: Manual smoke test — play**

Generate a row to `done`, click its 操作 cell. Expect the (visually hidden) chunk_audio to play that row's wav.

- [ ] **Step 3: Manual smoke test — regen mid-batch**

Start a 5-row batch, while it's running click 操作 on a yet-to-be-processed row. Expect that row's queue position to flip to "1" and the loop (after the in-flight row completes) to pick it up next.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(ui): per-row click dispatch — play (done rows) and regenerate"
```

---

### Task 18: Wire inline edit (.input) — voice + control validation

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the inline-edit handler**

```python
def _on_chunks_input(default_id, df_value):
    """.input fires for every cell edit; payload is the whole dataframe.
    Diff against state.gen_chunks to find changed cell, validate, persist.
    Reverts (re-renders from state) on invalid voice or non-editable column.
    """
    default = state.default_voice(default_id)
    # Normalize df_value (Gradio gives pandas DataFrame or list-of-lists).
    try:
        new_rows = df_value.values.tolist() if hasattr(df_value, "values") else list(df_value)
    except Exception:
        return _render_chunks_df(state, default), ""

    # Find the changed (row, col).
    changed = None
    for i, new_row in enumerate(new_rows):
        if i >= len(state.gen_chunks):
            break
        old_voice = state.gen_chunks[i].voice_name
        v = resolve_voice(state.gen_chunks[i].voice_id,
                          default_voice=default, library=state.library)
        old_voice_display = v.name if v else f"{old_voice} (已删除)"
        old_control = state.gen_chunks[i].control if state.gen_chunks[i].control is not None else ""
        if str(new_row[COL_VOICE]).strip() != old_voice_display:
            changed = (i, COL_VOICE, str(new_row[COL_VOICE]).strip())
            break
        if str(new_row[COL_CONTROL]) != old_control:
            changed = (i, COL_CONTROL, str(new_row[COL_CONTROL]))
            break

    if changed is None:
        # Edit was on a non-editable column (text/status/queue/action) — revert.
        return _render_chunks_df(state, default), ""

    idx, col, value = changed
    # Edits to the running row are rejected.
    if idx == state.gen_running_idx:
        return _render_chunks_df(state, default), "❌ 不能编辑正在生成的行"

    if col == COL_VOICE:
        v = state.library.find_by_name(value)
        if v is None and value.strip().lower() == default.name.strip().lower():
            v = default   # ephemeral default also editable
        if v is None:
            return _render_chunks_df(state, default), f"❌ 未知音色：`{value}`（已撤销）"
        state.gen_chunks[idx].voice_id = v.id
        state.gen_chunks[idx].voice_name = v.name
        if state.gen_status.get(idx) == "failed":
            state.gen_status[idx] = "pending"
            state.gen_errors.pop(idx, None)
        return _render_chunks_df(state, default), f"✏ 第 {idx + 1} 行音色更新为 `{v.name}`"

    if col == COL_CONTROL:
        # Empty cell → None; non-empty → that string. Three-state preserved.
        new_control = value if value else None
        state.gen_chunks[idx].control = new_control
        if state.gen_status.get(idx) == "failed":
            state.gen_status[idx] = "pending"
            state.gen_errors.pop(idx, None)
        return _render_chunks_df(state, default), f"✏ 第 {idx + 1} 行风格更新"

    return _render_chunks_df(state, default), ""

chunks_df.input(
    _on_chunks_input,
    inputs=[default_voice_dd, chunks_df],
    outputs=[chunks_df, current_status],
    concurrency_id="enqueue",
    concurrency_limit=None,
)
```

- [ ] **Step 2: Manual smoke test — voice rename**

Parse a row, double-click its 音色 cell, type a different known voice name → confirm. Expect the cell to update and `current_status` to show "音色更新".

- [ ] **Step 3: Manual smoke test — invalid voice reverts**

Type a non-existent voice name in 音色 → confirm. Expect the cell to revert and `current_status` to show "未知音色".

- [ ] **Step 4: Manual smoke test — control three-state**

Edit the 风格 cell from blank to `温柔` → confirm. Then clear it → confirm. Verify `state.gen_chunks[idx].control` flips between `None` and `"温柔"` (check via a print or by re-running generate and observing the SDK call shape).

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(ui): inline edit for 音色 + 风格 columns with validation"
```

---

### Task 19: Wire `merge_dl_btn` — concatenate done rows + waveform output

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the merge handler**

```python
import soundfile as sf
import numpy as np
from voxcpm_tts_tool.long_text import concat_waveforms
from voxcpm_tts_tool.output_writer import write_output_wav

def _on_merge():
    sample_rate = int(getattr(state.model, "sample_rate", 16000))
    waveforms = []
    for i in range(len(state.gen_chunks)):
        if state.gen_status.get(i) != "done":
            continue
        path = state.gen_audio.get(i)
        if not path:
            continue
        wav, sr = sf.read(path, dtype="float32")
        if sr != sample_rate:
            sample_rate = sr   # take whatever the file says
        waveforms.append(wav)
    if not waveforms:
        return gr.update(), "❌ 没有已完成的行可合并"
    merged = concat_waveforms(waveforms)
    out_path = write_output_wav(
        merged, sample_rate=sample_rate, outputs_dir=state.paths.outputs,
    )
    return gr.update(value=str(out_path)), f"✅ 已合并 {len(waveforms)} 行 → {out_path.name}"

merge_dl_btn.click(
    _on_merge,
    outputs=[merged_audio_out, current_status],
    concurrency_id="enqueue",
    concurrency_limit=None,
)
```

- [ ] **Step 2: Manual smoke test**

Generate a 3-row batch to completion, click 合并下载. Expect `merged_audio_out` to populate with a waveform, the play button to work, and the download icon to save the wav.

- [ ] **Step 3: Manual smoke test — empty merge**

With a fresh table (no done rows), click 合并下载. Expect `current_status` to show "没有已完成的行可合并" and the audio component to remain empty.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(ui): merge_dl_btn → concat done rows → write_output_wav → audio + download"
```

---

### Task 20: Lock 输入 sub-tab during generation

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Make `text_box` + `parse_btn` reflect lock state**

Extend the `_run_loop` yields to also disable `text_box` and `parse_btn` (set `interactive=False`) at the start of the loop and restore (`interactive=True`) at the end. Add the two components to the `outputs` list:

```python
generate_all_btn.click(
    _run_loop,
    inputs=[default_voice_dd, cfg_value_slider, inference_timesteps_slider],
    outputs=[chunks_df, current_status, generate_all_btn,
             text_box, parse_btn, default_voice_dd],
    concurrency_id="gen",
    concurrency_limit=1,
)
```

In `_run_loop` change each `yield (...)` to a 6-tuple:
- During the loop: `gr.update(interactive=False)` for text_box / parse_btn / default_voice_dd.
- At entry-no-rows / completion: `gr.update(interactive=True)`.

- [ ] **Step 2: Manual smoke test**

Start a long batch, observe that text_box, parse_btn, default_voice_dd grey out. Click 停止, observe that they re-enable after the in-flight row completes.

- [ ] **Step 3: Restore stop_btn label**

After the loop completes, also reset the stop button label (which `_on_stop` flipped to "⏳ 停止中…"):

```python
# At the END of _run_loop, in the final yield, also include:
yield (
    _render_chunks_df(state, default), final_status,
    gr.update(interactive=True),
    gr.update(interactive=True),
    gr.update(interactive=True),
    gr.update(interactive=True),
    gr.update(value="停止"),   # restore stop button label
)
```

Add `stop_btn` to the outputs list. Update intermediate yields to pass `gr.update()` for `stop_btn` (no change mid-loop).

- [ ] **Step 4: Manual smoke test**

Repeat the stop test from Task 16. Verify the stop button label resets to "停止" after the loop ends.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(ui): lock 输入 sub-tab during generation + restore stop button label"
```

---

### Task 21: End-to-end smoke + final cleanup

**Files:**
- Various — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all green. No regressions in unrelated tests.

- [ ] **Step 2: Manual end-to-end smoke**

Run the app from a fresh launch. Verify the full user flow per spec §User flow:
1. Type a 4-row script (mix of voices, with and without `(control)`). Click 解析.
2. Sub-tab switches to 分段; table shows 4 rows with ⏳.
3. Click 开始生成. Watch rows go ⚙ → ✓ in order.
4. Click 操作 on row 2 to play. Hear the audio.
5. Click 操作 on row 1 to regen — its queue position becomes "1". Click 开始生成 to resume (loop is idle since batch finished). Row 1 goes ⚙ → ✓.
6. Inline-edit row 3's 音色 to a different voice. Status flips back to ⏳ if it was failed; otherwise stays ✓ (edit doesn't re-run; user must click 重生 too if they want to regenerate). Click 重生 then 开始生成.
7. Click 合并下载. Expect `merged_audio_out` to populate with the concatenated waveform; play it; click download.
8. Try an empty-library case: rename the only voice to nothing the script references, parse plain text → row gets the ephemeral default → 开始生成 still works (text-only branch).

- [ ] **Step 3: Verify no orphaned imports / dead code in app.py**

Grep for `audio_out` (the deleted one-shot output) and `log_out` and `generate_btn` (deleted one-shot button). They should not appear except as the new `merged_audio_out`. Also verify `from voxcpm_tts_tool.generation import ...` does not still import `GenerationError` or `run_generation`.

- [ ] **Step 4: Commit final cleanup if needed**

```bash
git add app.py
git commit -m "chore: final cleanup of dead refs after Generation tab rewrite"
```

---

## Self-review notes

**Spec coverage check (each section → task):**
- §UI shape (sub-tabs, components) → Tasks 14, 19
- §Data model (ChunkRow, voice identity, AppState fields) → Tasks 1, 2
- §Splitting (split_for_table) → Tasks 6–9
- §Generation loop (run_queue, pre-flight, finally) → Tasks 10–12
- §Stop semantics → Task 16
- §Start / resume semantics (compute_fresh_queue) → Task 4 + 16
- §Inline edit → Task 18
- §Re-parse semantics (loose: silent reset) → Task 15 (note: confirm-toast deferred — spec §Re-parse is a "should"; flagged)
- §Per-row playback → Task 17
- §Final concatenation → Task 19
- §Components removed → Task 13
- §Tests → Tasks 1, 2, 6–12
- §Voice identity (resolve_voice, rename stability, ephemeral default) → Tasks 3, 9, 11
- §enqueue_regen running-row safety → Task 5
- §Triggering note (only generate_all_btn enters loop) → Task 16
- §run_queue cleanup invariants → Task 12
- §User flow → Task 21 smoke

**Re-parse confirmation toast (spec §Re-parse semantics line ~190):** Task 15 currently does silent `reset_generation()` instead of the documented two-click confirm. This is a UX nicety; deferring keeps Task 15 small and doesn't block the workflow. Add a follow-up task here if the user wants it before merge:

- *Optional Task 22*: Wire a one-time toast on parse when `any(state.gen_status[i] != "pending" for i in ...)`; second click within ~3s proceeds.

**Type / signature consistency check:**
- `ChunkRow` field order matches across all uses (line_no, voice_id, voice_name, text, control).
- `resolve_voice(voice_id, *, default_voice, library)` — same kwargs everywhere.
- `run_queue(state, *, library, default_voice, model, audio_root, zipenhancer_loaded, cfg_value, inference_timesteps, outputs_dir)` — signature consistent in test (Task 10) and handler (Task 16).
- `compute_fresh_queue(chunks, *, statuses)` — kwarg-only `statuses` matches usage.
- `enqueue_regen(state, idx)` — positional args; no audio_dir kwarg (we removed it; the function does best-effort os.unlink internally).

---
