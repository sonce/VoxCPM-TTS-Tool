# Generation tab: table-driven per-row workflow

Date: 2026-04-26
Status: Pending review

## Goal

Replace the single-shot Generation flow (textarea → one big wav) with a parse-then-batch flow whose unit of work is a row in a chunks table. Each row is one SDK `generate()` call. Users can listen to, regenerate, and edit individual rows; mid-batch regenerate inserts as the next row to process.

### User flow

1. **Parse**: user writes a script in `输入`, clicks 解析 → table populates in `分段`.
2. **Generate**: user clicks 开始生成. The loop processes rows one at a time. Each completed row becomes its own wav, playable inline.
3. **Review per row**: user plays each row's audio in the table; if a row is wrong, click 重生 to regenerate just that row (queued as next-up; resume the loop if idle). Inline-edit voice / control to adjust before regenerating. Repeat until every row sounds right.
4. **Merge & download**: user clicks 合并下载. All `done` rows are concatenated in row order and exposed as a single `gr.Audio` for in-browser playback and download.

The merge step is **explicitly user-triggered**, not automatic — generating all rows does NOT produce a merged file, because step 3 is the whole point of the new design.

## Current state vs new state

Today's `run_generation()` in `generation.py` is **not queue-driven**: it runs a `for` loop over segments × chunks, concatenates the wavs, and returns one combined result. There's no per-row state, no way to retry one chunk, and no way to interleave a regenerate during the run.

This spec **introduces the queue** as a first-class concept. The new loop reads `state.gen_queue` between rows; other handlers (regenerate) mutate the queue while the loop is running. The old `run_generation()` function is deleted; its tests get rewritten against the new chunking + per-row generate path.

## Scope (this spec)

`app.py` Generation tab restructure, plus:

1. New `chunking` module that wraps the existing parser + splitter to emit one `ChunkRow` per (voice, line, char-budgeted chunk) — see "Splitting" for why "paragraph" is just "line" here.
2. New `AppState` fields for the live queue/results.
3. New batch generator handler with mid-batch regenerate via separate `concurrency_id`.
4. Per-row click dispatch (play / regenerate) on the chunks table.
5. Inline edit of voice + control cells, validated on `.input`.
6. Auto-tab-switch on Parse and on Generate. Lock the input tab during generation.
7. Final concatenation: a "合并下载" button that joins all completed rows in row order.

## Out of scope

- Persisted generation history. This is a process-local single-user tool: `AppState` is constructed once at startup and shared by every browser tab pointed at the same process (see `app.py` `build_ui(state, ...)`). "Starts empty" means the process starts empty, not the browser session — opening a second tab will see the in-progress queue. Multi-session isolation would require moving `gen_chunks` / `gen_queue` / `gen_audio` / `gen_status` / `gen_errors` / `gen_running_idx` / `gen_stop_flag` into a `gr.State` per session; deferred.
- Streaming partial audio (each row only becomes playable after its `generate()` call returns).
- True mid-row cancellation (voxcpm SDK has no cancel primitive — see Stop semantics).

## UI shape

Generation outer Tab now contains two **sub-tabs** in a single `gr.Tabs`:

- **`输入`** sub-tab (shown by default):
  - `default_voice_dd` — default voice dropdown (unchanged).
  - `text_box` — `gr.Textbox(label="文本", lines=10)` (renamed from "脚本文本"; visible label change only).
  - Insert-voice / insert-tag pickers (unchanged).
  - `max_duration_slider` — Slider [10, 30] seconds, default 20s. Maps to `char_budget = seconds * 4`.
  - `parse_btn` — "解析为分段表格". Builds chunks, populates the table, switches to `分段` tab.

- **`分段`** sub-tab:
  - `chunks_df` — `gr.Dataframe`, columns `[# / 音色 / 风格 / 文本 / 状态 / 排队 / 操作]`. The 音色 and 风格 columns are interactive; the rest are read-only display.
  - `current_status` — Markdown showing the row currently being generated.
  - `chunk_audio` — hidden `gr.Audio(autoplay=True)` for per-row playback (same trick as voice-library tab).
  - `generate_all_btn` — "开始生成". Enters `run_queue` with the existing queue (resume) or rebuilds it from `pending` rows only when the queue is empty (fresh start). `failed` rows are not auto-requeued — see "Start / resume semantics".
  - `stop_btn` — "停止". Sets a stop flag; the loop exits after the current row finishes.
  - `merge_dl_btn` — "合并下载". Concatenates the wavs of all rows whose status is ✓, in row order, and writes the result to `merged_audio_out`.
  - `merged_audio_out` — `gr.Audio(label="合并结果", type="filepath", interactive=False, show_download_button=True, waveform_options=gr.WaveformOptions(...))`. Renders waveform + transport controls (play / pause / scrub) AND a built-in download button. Hidden / empty until the user clicks 合并下载. This is the same component family already in use elsewhere (`app.py:223`, `app.py:321`).

Sub-tab transitions:
- Click 解析 → switch to `分段`.
- Click 开始生成 → already in `分段`; `输入` sub-tab is **disabled** (interactive=False on its components, plus a lock indicator) until the loop exits.
- Click 停止 → unlocks `输入` once the in-flight row completes.

## Data model

```python
@dataclass
class ChunkRow:
    line_no: int                    # source line
    voice_id: str                   # canonical reference; always set at parse time (see "Voice identity" below)
    voice_name: str                 # display + inline-edit cache; refreshed from voice_id when the row is rendered
    text: str                       # immutable after parse (re-parse to change)
    control: str | None = None      # None == "no (control) in script"; "" == script wrote (); a string == that text
```

`control` is `str | None`, not `str`, because `build_generate_kwargs` already distinguishes the three cases (see `generation.py:37+` — `script_control is None` skips the script-control branch entirely; `""` enters clone mode with empty prefix). Flattening to `""` would silently force every row into clone-style and bypass the hifi branch for voices that have a `prompt_text`. The 风格 column renders `None` as empty; an inline edit that **clears** the cell sets the field back to `None` (not `""`), and an edit that types text sets it to that text.

**Voice identity — id vs name.** `voice_id` is the canonical reference (`Voice.id` is a stable UUID for library voices; ephemeral default voices have a synthetic id starting with `__`). `Voice.name` can be renamed via `VoiceLibrary.update`, which emits a "scripts containing <old> will no longer match" warning at `voice_library.py:166`, so storing the id keeps the row stable across rename. `voice_id` is **always** set at parse time — the parser preserves unknown `<voice>` tags as text under the active voice (`script_parser.py:118-124`), so segments only ever carry the default voice's name or a known library voice's name; both resolve. To resolve `voice_id → Voice` at render / generate time, callers must check the in-memory default voice first (its id won't be in the library) before falling back to `library.find_by_id`:

```python
def resolve(voice_id: str, *, default_voice: Voice, library: VoiceLibrary) -> Voice | None:
    # Short-circuit ONLY for ephemeral default — its id (`__...`) is never
    # in the library, so library.find_by_id would always return None.
    # A non-ephemeral default goes through library.find_by_id like any other
    # voice; if the user deleted it from the library, we MUST report it as
    # deleted rather than silently using the parse-time cached object.
    if default_voice.id.startswith("__") and voice_id == default_voice.id:
        return default_voice
    return library.find_by_id(voice_id)
```

The 音色 column displays `resolve(...).name` (so a rename done after parse is reflected without re-parse) and falls back to `row.voice_name` annotated `(已删除)` when `resolve` returns `None` (the library voice was deleted after parse). A deleted voice causes the loop to mark the row `failed` with "voice not found" — see "Generation loop" pre-flight.

Added to `AppState` (using `field(default_factory=...)` for the mutable defaults — Python dataclasses reject bare `[]` / `{}`):

```python
from dataclasses import field

gen_chunks: list[ChunkRow] = field(default_factory=list)
gen_audio: dict[int, str] = field(default_factory=dict)        # chunk_idx → wav path on disk
gen_status: dict[int, str] = field(default_factory=dict)       # chunk_idx → "pending"|"running"|"done"|"failed"
gen_errors: dict[int, str] = field(default_factory=dict)       # chunk_idx → message (only when failed)
gen_queue: list[int] = field(default_factory=list)             # PENDING rows only — does NOT include the running row
gen_running_idx: int | None = None                             # the chunk currently inside model.generate(); None when idle
gen_stop_flag: bool = False
```

Key invariant: `gen_running_idx` and `gen_queue` are **disjoint**. A row is either pending (in queue), running (in `gen_running_idx`), or settled (status `done` / `failed` / `pending` for never-started). This avoids the "regenerate the currently running row pops the wrong index" bug.

## Splitting

New module `src/voxcpm_tts_tool/chunking.py`:

```python
def split_for_table(
    script: str,
    *,
    library: VoiceLibrary,
    default_voice: Voice,
    char_budget: int,               # = seconds * 4
) -> tuple[list[ChunkRow], list[str]]:
    ...
```

The signature takes the full `Voice` for the default and the `VoiceLibrary`, not name strings, so step 5 can resolve names → ids without an extra lookup map plumbed in from the caller. "Ephemeral" is detected internally via `default_voice.id.startswith("__")`, mirroring `generation.py:243`.

Algorithm:
1. **Localize non-language tags first** — run `localize_non_lang_tags(script)` so `[笑声]` → `[laughing]` etc. before parsing. Same as today's `run_generation()`.
2. **Build name → Voice map.** Start with `{v.name: v for v in library.list_voices()}`. If `default_voice` is **not** ephemeral, add it under its name. The map is used both for parser `known_names` AND for post-parse name → id resolution. (Ephemeral default is added back only after parsing — see step 5 — so the parser still treats `<__default__>` as unknown text, identical to `generation.py:243-248`.)
3. Run existing `parse_script(script, default_voice=default_voice.name, known_names=<map keys>)` → `(ParsedSegment[], warnings)`. Each `ParsedSegment` already corresponds to one (voice, optional script-control) run within a single source line — `parse_script` resets to the default voice on every newline (`script_parser.py:100`). Segment text never contains `\n`.
4. For each segment, run existing `split_for_generation(seg.text, char_budget=...)` → list of chunk strings.
5. **Resolve `seg.voice_name` → `Voice`.** Look up in the map from step 2; if not found, fall back to `default_voice` (this is the ephemeral default case — the parser starts each line with `default_voice.name`, so an ephemeral default produces segments named after it that the map deliberately doesn't contain). The resolution is exhaustive given the parser invariant in §"Voice identity" — there is no "typo" branch because the parser preserves unknown tags as text under the active voice (`script_parser.py:118-124`).
6. Each chunk string → one `ChunkRow(line_no=seg.line_no, voice_id=resolved.id, voice_name=resolved.name, text=chunk, control=seg.control)`. `seg.control` is `None | str` per parser (None when script didn't write `(...)`, `""` when it wrote `()`, string when it wrote `(text)`) — pass through unchanged.
7. Return `(rows, warnings)`. The caller surfaces warnings on the table page.

Reuses `localize_non_lang_tags`, `parse_script`, and `split_for_generation` — no duplicate logic. The "paragraph" wording was a misread of the existing parser; current behavior already gives one segment per line, which IS the paragraph boundary the user asked for. Tests must not assert any "paragraph inside a segment" splitting; that case can't arise from the current parser.

## Generation loop

Three handlers in three concurrency groups. They never share a queue slot.

**`run_queue` (the loop)** — `concurrency_id="gen"`, `concurrency_limit=1`. Generator wrapped in an outer `try / finally`: the `finally` always runs (queue drained, stop flag set, OR an unexpected exception leaks past the per-row handler) and is responsible for `state.gen_stop_flag = False`, clearing `current_status`, and unlocking the `输入` sub-tab. Without this, an uncaught exception during table re-render or wav write would leave the UI permanently locked.

Inside that outer try, while `state.gen_queue` is non-empty AND `not state.gen_stop_flag`:
- `idx = state.gen_queue.pop(0)` — pop **first**, then assign `state.gen_running_idx = idx`. The running row is never in the queue, so concurrent mutations to the queue can't shift the wrong index out from under us.
- Mark `running`, yield table update + `current_status` markdown.
- Wrap the row body in `try / except / finally`. The `finally` does **state cleanup first** — `state.gen_running_idx = None`, plus updating `gen_status[idx]` to `failed` if the row never reached its success/failure assignment — and only then attempts a final table-update yield. State cleanup must NOT be guarded by the yield: if the table re-render itself raises (Gradio component error, df serialization, etc.), state must already be consistent so the outer `finally` can still unlock the input tab. Tests assert against the state cleanup; the final UI yield is best-effort.
- Inside the `try`:
  - **Pre-flight validation (mirrors the deleted `run_generation`'s check at `generation.py:267`).**
    - Resolve voice via the `resolve(row.voice_id, default_voice=..., library=...)` helper from §"Voice identity". If `None` (the row's library voice was deleted after parse), set `gen_errors[idx] = f"voice not found: {row.voice_name}"`, mark `failed`, `continue` (the per-row `finally` clears `gen_running_idx`). An ephemeral default voice short-circuits to `default_voice` itself and proceeds; this preserves the empty-library text-only generation path that `run_generation` had at `generation.py:256-257`.
    - For voices with audio (clone/hifi/legacy with a `reference_audio` fallback), compute the absolute path and `os.path.exists` it. If missing, set `gen_errors[idx] = f"voice audio missing at {path}"`, mark `failed`, `continue`. Voices with no audio (legacy design / ephemeral default) skip this check and fall through to `build_generate_kwargs`'s text-only branch (`generation.py:75-79`).
    - Both messages must be user-readable; do NOT let raw SDK file errors surface from inside `model.generate`.
  - Build kwargs from `state.gen_chunks[idx]` + voice + global cfg/timesteps.
  - Call `state.model.generate(**kwargs)`.
  - On success: write wav to a per-row tmp file, store path in `state.gen_audio[idx]`, mark `done`.
- The `except Exception as e` branch (catches SDK errors AND wav-write errors): store `gen_errors[idx] = f"{e} [text: {chunk_preview!r}]"` (preview pattern from `generation.py:286-290`), mark `failed`. Errors are recorded as user-readable strings in `gen_errors[idx]`; there is **no custom exception class** (the deleted `GenerationError` is not replaced — the loop is the only consumer, and it just needs message text). Tests assert against the string contents in `gen_errors`.

**`enqueue_regen` (per-row regenerate)** — `concurrency_id="enqueue"`, `concurrency_limit=None`. Handler that:
- Reset row state: `gen_status[idx] = "pending"`, drop `gen_audio[idx]` (unlink the wav file if present), drop `gen_errors[idx]`.
- Compute target queue position:
  - If `idx == state.gen_running_idx`: it's already running. voxcpm has no cancel, so the in-flight call will complete and overwrite the wav once. To **also** regenerate after that, insert at queue index 0 — the loop will pick it up as the next row. (User-visible effect: the row flips back to ⏳, finishes the in-flight run, finishes once more, ends as the latest result.) An alternative — silently no-op when `idx == running` — is rejected because the user explicitly clicked 重生 expecting a fresh result.
  - Else: remove `idx` from `state.gen_queue` if present, then insert at index 0 (the first pending row that the loop will pick up).
- Re-render the table with updated 排队 numbers (running row shows `▶`; queue slots 1..N show their position; settled rows show empty).
- If the loop is **not** running (`gen_running_idx is None`), the user must click 开始生成 to start it — see "Triggering" below. The new entry is now sitting at the front of the queue, ready.

The three concurrency groups (`gen` / `enqueue` / `control`) ensure regenerate runs immediately even while the loop is busy. Because the loop pops *before* yielding, the queue index that `enqueue_regen` mutates can never collide with the running row.

### Triggering note

Gradio can't programmatically click a button from inside a Python handler. Two options handled:
- **Option 1 (chosen):** `generate_all_btn` is the only loop entry point (`concurrency_id="gen"`). `stop_btn` is a `control` event that just sets `gen_stop_flag` (see Stop semantics) and never enters `run_queue`. `enqueue_regen` only mutates state; if the loop isn't running, the user has to click 开始生成 to start it. This is acceptable because regenerate-when-idle is rare (user typically regenerates after batch completes; clicking 重生 in that moment then 开始生成 is one extra click).
- **Option 2 (deferred):** wire a hidden `gr.State` counter that increments on regenerate; its `.change` triggers the loop. Adds complexity for marginal UX benefit.

If you want option 2 instead, say so and I'll switch.

## Stop semantics

voxcpm `generate()` is a blocking torch op with no cancel. "停止" means:
- `stop_btn.click(set_stop_flag, ...)` runs in `concurrency_id="control"` with `concurrency_limit=None`. It must NOT share a concurrency group with `run_queue`; if it did, the click would queue behind the running loop and never take effect. The handler does exactly two things: `state.gen_stop_flag = True` and flip the button label to "⏳ 停止中…". No queue mutation, no SDK call.
- The loop checks `gen_stop_flag` at the top of each iteration.
- The currently-running row **completes** (no mid-row interrupt). Its result is kept.
- After it completes, the loop exits, the `输入` sub-tab unlocks, queue is preserved.
- User can resume via 开始生成 (the queue picks up where it left off — see "Resume" below).

## Start / resume semantics

`generate_all_btn` click handler (the same one that's wired to `run_queue` as the entry point):

1. **Block during a running loop.** While `state.gen_running_idx is not None`, the button is `interactive=False`. There's no way to enter the handler twice; the concurrency-1 limit on `gen` would also serialize duplicate clicks.
2. **Resume case** — loop is idle AND `state.gen_queue` is non-empty: just enter `run_queue` with the existing queue. Don't rebuild. This preserves any 重生 insertions made while paused, plus any unfinished rows from before 停止.
3. **Fresh start case** — loop is idle AND queue is empty: rebuild the queue from `state.gen_chunks`, including **only rows whose status is `pending`**. `failed` rows are NOT auto-requeued — re-running them requires either fixing the cause (e.g. inline-edit the voice cell to a valid voice, which resets the row's status to `pending`) or an explicit 重生 click (which also resets to `pending` and inserts at queue front). Otherwise the same broken row would re-fail on every "开始生成" click, which is annoying and gives no signal that the user must take action. `done` rows likewise require explicit 重生. Then enter `run_queue`.

The 排队 column reflects the queue indices, not chunk indices. The running row shows `▶`. Queue position 0 is "next-up", 1 is "after that", and so on. Done / failed / never-queued rows show empty.

## Inline edit (voice + control)

`chunks_df` is constructed with `interactive=True`. The 文本, 状态, 排队, 操作 columns ignore edits (revert on `.input`).

On `.input` event:
- Diff the new dataframe value against `state.gen_chunks` to find the changed row + column.
- 音色 column: resolve the typed value via `library.find_by_name` (case-insensitive trim — `voice_library.find_by_name` already does that at `voice_library.py:80`); also accept `default_voice.name` so an ephemeral default is editable too. If found, persist BOTH `voice_id = voice.id` and `voice_name = voice.name` (canonicalized) to `state.gen_chunks[idx]`. If the row's status was `failed`, reset it to `pending` (so the next 开始生成 picks it up — see "Start / resume semantics"). If not found, revert and show a warning toast.
- 风格 column: free-text. **Empty cell → `None`**, non-empty → that string. This preserves the three-state distinction documented in Data model. Persist to `state.gen_chunks[idx].control`.
- All other columns: revert by re-rendering from state.

Edits made on the row that is currently `running` are **rejected** (revert + toast). Edits to other rows during a batch are fine.

## Re-parse semantics (Q4=A)

- The 文本 textbox is independent from `state.gen_chunks` until the user clicks 解析.
- 解析 click: if any row has status ≠ `pending`, show a confirm-style toast "重新解析将丢弃已生成的音频，再点确认"; second click within a short window proceeds.
- During generation: the `输入` sub-tab is locked (the textbox + parse button are non-interactive), so the user cannot reach 解析 until 停止 finishes.

## Per-row playback

Same pattern as voice library: shared hidden `gr.Audio(autoplay=True)`. Clicking the "▶ 播放" cell sets the audio to that row's wav and autoplays. Two-phase clear→set chain so replays of the same row re-fire autoplay (carries over the fix already in voice library).

## Final concatenation (Q3=A)

`merge_dl_btn` click:
- Read each row in order. For rows whose status is `done`, load their wav.
- Concatenate via existing `concat_waveforms`.
- Write the result to `outputs/` via the existing `write_output_wav(waveform, sample_rate=..., outputs_dir=...)` (`output_writer.py:19`). It produces a timestamped filename like `outputs/20260426-143022-481.wav` (with `-1`, `-2`, ... suffixes on collision). This is a persisted file under the project's outputs dir, **not** a tempfile — its lifecycle is the same as today's one-shot output, so the user keeps every merge they trigger. The returned `Path` is what we hand to `merged_audio_out`.
- Update `merged_audio_out` (the `gr.Audio` declared in UI shape: `type="filepath"`, `show_download_button=True`, with `waveform_options` so the user gets a visible waveform). The user can play, scrub, and download via the component's built-in transport + download UI — no separate "save as" step.
- If no rows are done, show a toast and do nothing (leave `merged_audio_out` unchanged).

## Components removed

- The original "Generate" button + audio output (one-shot whole-script generation) is **removed**. The new "开始生成" + "合并下载" replace it. Per Q5=B, the user wanted to keep the one-shot UX, which is achieved by `开始生成 + 合并下载` (still a one-click batch then one-click merge).

## Tests

New module `tests/test_chunking.py`:
- One row per `<voice>` switch (parser already does this; `split_for_table` preserves it).
- Each source line is parsed independently — `parse_script` resets to the default voice on every newline (`script_parser.py:100`). The realistic "multiple lines under same voice → multiple rows" case is "two lines, both prefixed `<bob>`, both produce bob-voiced rows"; a `<bob>` written only on line 1 does NOT carry over to line 2 (line 2 reverts to default voice). Tests must reflect this; do not assert that voice state persists across `\n`. There is no "split inside a segment by `\n`" case either — segment text never contains `\n`.
- Long line honors `char_budget` and falls back through `split_for_generation`'s sentence/comma rules.
- `seg.control` round-trips: `None` stays `None`, `""` stays `""`, `"温柔"` stays `"温柔"`. Lock the three-state distinction.
- `localize_non_lang_tags` runs before parse: `<voice>[笑声]你好` → row with text containing `[laughing]`.
- Ephemeral default (id starts with `__`): `<__default__>` is an unknown tag (preserved verbatim, parser warning). Plain text under an empty library still produces a row with `voice_id == default_voice.id`, and the loop's `resolve(...)` short-circuits to `default_voice` so generation proceeds via `build_generate_kwargs`'s text-only branch (`generation.py:75-79`). This is the regression target most likely to break the empty-library happy path.
- Voice id stability across rename: parse against a library where one voice is named `bob` → row stores `voice_id == bob.id`. After `library.update(bob.id, name="robert")`, calling `resolve(row.voice_id, ...)` still returns the renamed voice, and re-rendering the table shows `robert` in the 音色 column. The textbox is unchanged (still says `<bob>`); re-parse would now warn-and-preserve, but the parsed rows remain valid.
- Empty / whitespace-only input → empty list, no crash.

`tests/test_generation.py` — drop `run_generation`-based tests; keep `build_generate_kwargs` tests as-is (they exercise the per-row path that the new loop uses).

UI / generation-loop behavior is exercised manually (Gradio handlers are hard to unit-test without a real model). Three pure-Python paths *are* unit-testable and should be covered:
- `enqueue_regen` queue mutation: given various `(gen_running_idx, gen_queue)` shapes, assert the resulting queue.
- The "fresh start vs resume" branch in the start handler. Specifically: fresh start with a mix of `pending` / `failed` / `done` rows queues only `pending`.
- `run_queue` cleanup invariants — drive the loop with a fake model that raises on `generate()`, and a fake "wav writer" that raises on the success path. After the loop returns, assert: `gen_running_idx is None`, the offending row's status is `failed` with a non-empty `gen_errors[idx]`, `gen_stop_flag is False`, and `current_status` is cleared. Both inner per-row failure and an unexpected leak past the inner handler must hit the outer `finally`. This guards the "UI permanently locked" regression that motivates the try/finally structure.

## File-by-file change list

- `src/voxcpm_tts_tool/chunking.py` — new module, ~80 lines.
- `src/voxcpm_tts_tool/app_state.py` — add the `gen_*` fields + a `reset_generation()` helper.
- `src/voxcpm_tts_tool/generation.py` — keep `build_generate_kwargs` (still used per-row); **delete** `run_generation`, `GenerationResult`, `GenerationError`. The new loop does not need a custom exception class — failures are written to `gen_errors[idx]` as user-readable strings (see "Generation loop"). `synthesize_voice_preview` already raises `ValueError` for its own input checks (`generation.py:172`, `188`, `197`) and is unaffected. Their callers in `app.py` are removed in this same change.
- `app.py` — Generation tab rewritten (sub-tabs, table, queue handlers, stop, merge); ~280 lines net new.
- `tests/test_chunking.py` — new, ~120 lines.
- `tests/test_generation.py` — drop the `run_generation`-based tests (they no longer have a target); keep `build_generate_kwargs` tests as-is.

## Risks & open questions

- **Concurrency-group support in Gradio v6**: I'll verify `concurrency_id` per-event behaves as documented before wiring the parallel-regenerate path. Fallback (option 1 above) doesn't depend on it.
- **Dataframe `.input` event granularity**: Gradio fires `.input` for every cell edit but the payload is the whole dataframe. Diff cost is O(rows); fine for typical scripts.
- **Per-row tmp wav lifecycle**: each generation overwrites the row's previous wav (`os.unlink` first). On session exit, all per-row wavs are orphaned in tmp; OS cleans on reboot. Acceptable for a local dev tool.
