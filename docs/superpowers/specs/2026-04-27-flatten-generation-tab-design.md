# Flatten Generation tab: roll back chunks-table to streaming one-shot

Date: 2026-04-27
Status: Pending review

## Goal

Replace the parse-then-table workflow (输入/分段 sub-tabs, `chunks_df`, per-row regen, merge) with a single streaming generator: one button → progress text → one wav.

The script-parsing layer (`<voice>` switching + `(control)` parens + non-lang tag localization) is **kept**. The "long-text splitting" (`split_for_generation` by char budget) is **kept**. Only the table view, queue, per-row regen, and merge step are removed.

This is a deliberate revert of the table-driven design at `2026-04-26-generation-table-view-design.md`, prompted by environmental issues (remote-port-forwarded WS instability surfaced as "tabs stuck") and a re-evaluation of UX value: per-row regen turned out to be more complexity than the user wanted to maintain for now.

## Out of scope

- Per-row regenerate, inline edit, individual row playback.
- Mid-row cancellation (voxcpm has no cancel primitive — same as before).
- Persisted queue across page reloads.

## UI (Generation tab)

Single flat layout under the existing `gr.Tab("生成")`. **No inner `gr.Tabs`**.

Top-to-bottom:

1. `default_voice_dd` — default voice dropdown (unchanged).
2. `text_box` — `gr.Textbox(label="文本", lines=10)` (unchanged).
3. `voice_picker` + `insert_voice_btn` / `tag_picker` + `insert_tag_btn` — script-authoring helpers (unchanged; multi-voice retained).
4. `max_duration_slider` — `gr.Slider(10, 30, step=1, value=20, label="单段最长时长（秒，× 4 ≈ 字符预算）")`. Maps to `char_budget = sec * 4`, fed into `split_for_generation`.
5. Advanced accordion: `cfg_value_slider`, `inference_timesteps_slider` (unchanged).
6. `generate_btn` — `gr.Button("开始生成", variant="primary")`. **The only entry point.**
7. `stop_btn` — `gr.Button("停止", variant="secondary")`. Visible always; only effective during a run.
8. `status_md` — `gr.Markdown(value="")`. Progress text, written by the generator on each yield.
9. `audio_out` — `gr.Audio(label="结果", type="filepath", interactive=False)`. Empty until a Result is yielded; then shows the generated wav with built-in transport + download.

**Components removed from current code:** `gen_subtabs`, `gen_input_tab`, `gen_table_tab`, `chunks_df`, `current_status` (replaced by `status_md`), `chunk_audio`, `generate_all_btn` (renamed to `generate_btn`), `merge_dl_btn`, `merged_audio_out` (replaced by `audio_out`).

## Generation flow

### Module surface — `generation.py`

Restored `run_generation` is an event-stream generator. Two small dataclasses define the events:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Progress:
    done: int        # how many SDK calls have completed
    total: int       # total SDK calls planned (computed up front)

@dataclass
class Result:
    wav: np.ndarray
    sample_rate: int
    was_stopped: bool = False    # True iff stop_flag triggered the early return

def run_generation(
    script: str,
    *,
    library: VoiceLibrary,
    default_voice: Voice,
    model,
    audio_root: str,
    zipenhancer_loaded: bool,
    char_budget: int,
    cfg_value: float,
    inference_timesteps: int,
    stop_flag: Callable[[], bool],
) -> Iterator[Progress | Result]:
    ...
```

**Algorithm:**

1. `script = localize_non_lang_tags(script)` — same prep as the deleted `split_for_table` step 1.
2. Build name→Voice map from `library.list_voices()`; add `default_voice` under its name unless ephemeral (`default_voice.id.startswith("__")`).
3. `segments, warnings = parse_script(script, default_voice=default_voice.name, known_names=...)`. Warnings are surfaced via the generator's first yield (see "Warnings" below).
4. Pre-compute total chunk count: `total = sum(len(split_for_generation(seg.text, char_budget=char_budget)) for seg in segments)`. This is the denominator in `Progress.done/total`. If `total == 0` (empty input), yield `Result(np.zeros(0, dtype=np.float32), sample_rate=model.sample_rate)` and return.
5. Iterate:
   ```python
   all_wavs: list[np.ndarray] = []
   done = 0
   for seg in segments:
       voice = _voice_for_segment(seg.voice_name, default_voice=default_voice, library=library)
       chunks = split_for_generation(seg.text, char_budget=char_budget)
       for chunk in chunks:
           if stop_flag():
               yield Result(_concat(all_wavs), sample_rate, was_stopped=True)
               return
           kwargs = build_generate_kwargs(
               text=chunk, voice=voice, script_control=seg.control,
               audio_root=audio_root, zipenhancer_loaded=zipenhancer_loaded,
               cfg_value=cfg_value, inference_timesteps=inference_timesteps,
           )
           wav = model.generate(**kwargs)
           all_wavs.append(wav)
           done += 1
           yield Progress(done=done, total=total)
   yield Result(_concat(all_wavs), sample_rate)
   ```
6. `sample_rate` comes from `model.sample_rate` (per the existing `1d4b066 fix: ... trust model sample_rate in merge` decision). Same for the empty-input path.
7. `_concat` is `np.concatenate(all_wavs).astype(np.float32, copy=False)`, plus the empty fallback `np.zeros(0, dtype=np.float32)` for stop-on-first-chunk.

**Warnings.** `parse_script` returns warnings for unknown `<voice>` tags (preserved as text). The generator yields them via a one-time prelude `Progress(done=0, total=total)` — the handler shows the warnings in `status_md` before the first chunk lands. Implementation detail: pass them out via a separate field, or just emit them through the handler's own `parse_script` call before invoking `run_generation`. **Decision: caller invokes `parse_script` itself for the warnings**, then calls `run_generation` for execution. This keeps the generator monomorphic (only `Progress | Result`). See "Code organization" for where the parse-and-warn happens.

### Handler — `app.py`

```python
def _on_generate(default_id, text, max_duration, cfg_value, inference_timesteps):
    default = state.default_voice(default_id)

    # Pre-flight: localize + parse to surface warnings before locking the UI.
    prepped = localize_non_lang_tags(text)
    by_name = {v.name: v for v in state.library.list_voices()}
    if not default.id.startswith("__"):
        by_name[default.name] = default
    _, warnings = parse_script(prepped, default_voice=default.name, known_names=list(by_name))
    warn_md = "" if not warnings else "⚠ " + "; ".join(warnings) + "\n\n"

    state.gen_stop_flag = False  # arm
    try:
        # Initial lock + status.
        yield (gr.update(value=None),                          # audio_out: clear
               warn_md + "▶ 准备生成…",
               gr.update(interactive=False),                   # generate_btn
               gr.update(interactive=False),                   # text_box
               gr.update(interactive=False),                   # default_voice_dd
               gr.update(value="停止"))                         # stop_btn (label restore)
        for ev in run_generation(
                prepped, library=state.library, default_voice=default, model=state.model,
                audio_root=str(state.paths.root),
                zipenhancer_loaded=state.zipenhancer_loaded,
                char_budget=int(max_duration) * 4,
                cfg_value=float(cfg_value),
                inference_timesteps=int(inference_timesteps),
                stop_flag=lambda: state.gen_stop_flag):
            if isinstance(ev, Progress):
                yield (gr.update(), warn_md + f"▶ 正在生成第 {ev.done}/{ev.total} 段",
                       gr.update(interactive=False), gr.update(interactive=False),
                       gr.update(interactive=False), gr.update())
            else:  # Result
                if ev.wav.size == 0:
                    yield (gr.update(value=None), warn_md + "ℹ 没有可生成的内容",
                           gr.update(interactive=True), gr.update(interactive=True),
                           gr.update(interactive=True), gr.update(value="停止"))
                else:
                    out_path = write_output_wav(
                        ev.wav, sample_rate=ev.sample_rate, outputs_dir=state.paths.outputs)
                    msg = ("⏸ 已停止，已导出前面段" if ev.was_stopped
                           else "✅ 已完成")
                    yield (gr.update(value=str(out_path)), warn_md + msg,
                           gr.update(interactive=True), gr.update(interactive=True),
                           gr.update(interactive=True), gr.update(value="停止"))
    except Exception as e:
        yield (gr.update(value=None),
               warn_md + f"❌ 生成失败：{e}",
               gr.update(interactive=True), gr.update(interactive=True),
               gr.update(interactive=True), gr.update(value="停止"))
    finally:
        state.gen_stop_flag = False
```

`generate_btn.click(_on_generate, ..., concurrency_id="gen", concurrency_limit=1)`.
`stop_btn.click(lambda: setattr(state, "gen_stop_flag", True) or gr.update(value="⏳ 停止中…"), outputs=stop_btn, concurrency_id="control", concurrency_limit=None)`. Two concurrency lanes: stop can flip the flag while gen is running.

## Stop vs failure — the asymmetry

Encoded as separate code paths because the user wants different outputs:

| Trigger | What `run_generation` does | What handler emits |
|---|---|---|
| User clicks 停止 (mid-loop) | `stop_flag()` returns True → yield `Result(concat(all_wavs), sr)` and `return` | Write the partial wav to `outputs/`; status `⏸ 已停止，已导出前面段` |
| `model.generate(...)` raises | Generator propagates the exception | Handler `except` clears `audio_out`, status `❌ 生成失败：<msg>`; **all wavs discarded** |
| Successful completion | Loop runs to end, yield final `Result(concat(all_wavs), sr)` | Write wav, status `✅ 已完成` |

**Why stop ≠ failure:** stop is user-initiated and intentional — the user wants a result on disk to listen to what they got. Failure is a runtime fault — partial output would mislead about what they asked for. Tests assert this asymmetry (see Tests).

## Code organization

**New / restored:**

- `src/voxcpm_tts_tool/generation.py` — restore `run_generation` as the event-stream generator above. Keep existing `build_generate_kwargs` and `synthesize_voice_preview` unchanged. Define `Progress` and `Result` here (one file is plenty; they're the public output shape of `run_generation`).
- Inside `generation.py`: a **module-private** helper `_voice_for_segment(name, *, default_voice, library) -> Voice`. Resolution is by name (the parser canonicalizes case and only emits names from `known_names ∪ {default_voice.name}`). No id-based logic, no rename-stability concern (the whole script runs in one handler invocation; rename can't happen mid-run). This replaces the old `resolve_voice` from `generation_queue.py`. **Decision rationale:** the old module-level `resolve_voice` existed because the table view stored `voice_id` on each row and needed to re-resolve at render time across renames. The flat flow has neither requirement, so an inline helper is correct — splitting it into a `voice_resolve.py` module would be premature abstraction (one caller, ~8 lines).

**Deleted:**

- `src/voxcpm_tts_tool/chunking.py` — only consumer was `_on_parse`; gone.
- `src/voxcpm_tts_tool/generation_queue.py` — only consumer was `_run_loop` / `enqueue_regen`; gone.
- `tests/test_chunking.py`
- `tests/test_generation_queue.py`

**Modified:**

- `src/voxcpm_tts_tool/app_state.py`:
  - **Delete** fields: `gen_chunks`, `gen_audio`, `gen_status`, `gen_errors`, `gen_queue`, `gen_running_idx`.
  - **Delete** method: `reset_generation()`.
  - **Keep** field: `gen_stop_flag: bool` (the new handler still uses it).
  - **Delete** `if TYPE_CHECKING: from .chunking import ChunkRow` import.
- `src/voxcpm_tts_tool/script_parser.py`: no change. `localize_non_lang_tags` and `parse_script` keep their current uncommitted form.
- `app.py` (Generation tab block, ~app.py:233 to app.py:730 area):
  - Replace the `with gr.Tabs() as gen_subtabs:` ... `with gr.TabItem("分段") ...` structure with a flat `with gr.Tab("生成"):` body containing the components in "UI" above.
  - Delete handlers: `_on_parse`, `_run_loop`, `_on_stop` (replaced by simpler one), `_on_row_select`, `_apply_inline_edit`, `_apply_pending_chunk` (or whichever names the chunk row dispatcher uses), `_on_merge_download`, `_render_chunks_df`.
  - Delete the module-level `_render_chunks_df` helper (only used by the deleted handlers).
  - Add `_on_generate` (the handler above) + the new `_on_stop` (one-liner that flips the flag).
  - Remove the `from voxcpm_tts_tool.chunking import split_for_table` import and the `from voxcpm_tts_tool.generation_queue import ...` imports.
  - Delete the COL_ACTION constant and any other table-column index constants.
- `tests/test_app_state.py`: delete the `reset_generation` test if present; delete any test that asserts the deleted fields.
- `tests/test_generation.py`: see "Tests" below.

**Untouched but worth noting:**

- `voice_library.py` — the uncommitted seed_text / audio_upload changes are unrelated to this work; left as-is. They belong to `2026-04-26-voice-library-edit-and-controls-design.md`.
- `output_writer.py` — `write_output_wav` is reused without change.
- `long_text.py` — `split_for_generation` is reused without change.

## Tests

`tests/test_generation.py` gets the new spec coverage. Test against a fake `model` whose `generate(**kwargs)` returns a deterministic small ndarray (e.g., `np.full(N, idx, dtype=np.float32)` where `idx` is a counter), and a fake `model.sample_rate = 16000`.

Required cases:

1. **Single-voice happy path** — script `"你好世界。再见。"` with one library voice → 2 segments × 1 chunk each (terminator splits) → assert yield sequence is `[Progress(1,2), Progress(2,2), Result(...)]`, and `Result.wav` is the concatenation of the two fake wavs in order.

2. **Multi-voice + control round-trip** — script `"<bob>你好<alice>(温柔)再见"` → 2 segments (one bob no-control, one alice with control="温柔"). Assert each segment was called with the correct `reference_wav_path` and `script_control` via `build_generate_kwargs` (mock or capture kwargs).

3. **Long-text chunk splitting** — script with a single long line containing `。` terminators that exceed `char_budget` → assert `total` matches `split_for_generation`'s chunk count and yield count == chunks + 1 (`Progress` × N + final `Result`).

4. **Stop preserves partial output** — `stop_flag` returns True after the second `Progress`. Assert yield sequence is `[Progress(1,N), Progress(2,N), Result(wav, was_stopped=True)]` where `wav.size == 2 * fake_chunk_size` and the generator returned cleanly. Critically: assert `Result.was_stopped is True` (this is what the handler dispatches on, not the global flag).

5. **Failure discards everything** — fake model raises `RuntimeError("boom")` on the third call. Assert the generator yields `Progress(1,N), Progress(2,N)` then `RuntimeError("boom")` propagates out. Assert no `Result` was yielded.

6. **Empty input** — `script=""` → yield is exactly `[Result(empty_wav, sample_rate)]`, no `Progress`. Empty wav is `np.zeros(0, dtype=np.float32)`.

7. **Whitespace-only input** — `script="   \n  \n"` → same as case 6 (parser returns no segments).

8. **Ephemeral default voice (empty library)** — script `"你好"` with default_voice = ephemeral → `Result` is the single fake wav, `_voice_for_segment` returned default_voice without library lookup.

`build_generate_kwargs` tests stay as-is (already cover the per-segment call construction).

UI / Gradio handler is exercised manually (consistent with the rest of the project — `app.py`'s handler tests would require booting Gradio).

## Risks / notes

- **The "tab clicks stuck" symptom that prompted this redesign was diagnosed as remote-port-forward WS instability**, not a bug in the parse-table code. Flattening the UI removes a lot of complexity but doesn't itself fix that environment issue. If the new flat UI also "sticks", the root cause is the forward / remote process — not this design.
- **Concurrency lanes (`gen` / `control`) are intentionally separate** so 停止 can flip the flag while a `generate(...)` call is mid-flight. Without this split, the click would queue behind the running generator and the flag-flip would only fire after `generate` completes — which already completes the segment naturally and renders 停止 useless.
- **Stop semantics intentionally diverge from failure semantics.** Encoded explicitly because the user asked for it. Tests pin this in (cases 4 and 5).
- **Warnings prelude.** Caller pre-parses to extract warnings before invoking `run_generation`. This means `parse_script` runs twice (once in handler for warnings, once inside `run_generation` for execution). This is fine: parsing is microseconds. The alternative — making `run_generation` yield a `Warnings` event before `Progress` — bloats the union type for a one-time prelude. Not worth the abstraction tax.
