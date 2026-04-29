# Voice Library: edit, listen, delete-confirm, refresh placement

Date: 2026-04-26
Status: Approved

## Goal

Make the voice library tab usable for managing existing voices, not just creating new ones. Today the only row-level action is "click a row to delete it"; users have no way to listen to a saved voice or change anything about it short of deleting and recreating.

## Scope

Four UI changes, all confined to the voice-library tab in `app.py` and a small extension to `VoiceLibrary.update()`:

1. **Row-level listen button** — play a saved voice's audio without re-rendering the whole script.
2. **Row-level edit** — click a row to load that voice into the form and edit *all* its fields (full overwrite, including audio files).
3. **Delete confirmation** — browser-native `confirm()` before destruction.
4. **Refresh button placement** — moved to the top-left, immediately above the table.

## Table layout

Columns become: `名称 | 模式 | 🔊 | 🗑`. The combined "操作" column from before is split.

Click semantics by column index (handled in `voice_list.select` via `evt.index[1]`):

| col | label | action |
|----|-------|--------|
| 0  | 名称  | enter edit mode for this voice |
| 1  | 模式  | enter edit mode for this voice |
| 2  | 🔊    | load + autoplay this voice's `voice.audio` in the listen widget |
| 3  | 🗑    | confirm-then-delete (see §Delete confirmation) |

## Listen widget

A `gr.Audio(autoplay=True, visible=False)` placed below the dataframe. Clicking the 🔊 column resolves the row to a voice, computes `state.paths.root / voice.audio`, and updates the widget with that path + `visible=True`. Clicking 🔊 on another row just swaps the source.

`gr.Dataframe` has no in-cell media support in Gradio 6, so a single shared widget is the only realistic approach.

## Refresh button placement

```
gr.Row():
    refresh_btn  # compact, left-aligned, scale=0
voice_list = gr.Dataframe(...)
listen_audio = gr.Audio(visible=False, autoplay=True)
```

The existing `refresh_btn = gr.Button(i18n.t("btn.refresh", "zh"))` moves up into a `gr.Row()` immediately above the dataframe. Use `scale=0` (or `size="sm"`) so it doesn't stretch full-width.

## Delete confirmation

When the user clicks the 🗑 column, the browser must show a native `confirm("确定删除该音色？此操作不可撤销。")` and the Python deletion path runs only when the user confirms.

Two implementation paths are acceptable; the implementation plan picks based on what wires cleanly into Gradio 6's `select` event:

- **Path A — JS guard on `voice_list.select`:** pass `js=` that inspects the SelectData and returns `null` to abort when clicked column is 3 and `confirm()` returns false.
- **Path B — pending-delete state machine:** the select handler sets a `pending_delete_id` state and returns a follow-up Markdown ("确认删除 <name>?") with [确认]/[取消] buttons; only the [确认] button calls `library.delete`. The native `confirm()` dialog is then triggered on the [确认] button via `js=` (which works on Buttons reliably).

Path A is preferred for minimal UI churn. If Gradio's `select`-event `js=` does not reliably intercept, fall back to Path B.

## Edit mode

### State

Add `edit_voice_id = gr.State("")`. Empty string = "creating a new voice"; non-empty = "editing voice <id>".

### Entering edit mode

Clicking col 0 or col 1 of a voice row calls a handler that:

1. Sets `edit_voice_id` to that voice's id.
2. Populates form fields from the voice:
   - `name_box` ← `voice.name`
   - `top_mode_radio` / `sub_mode_radio` ← derived from `voice.mode` (design → "design"; clone → "cloning"+"controllable"; hifi → "cloning"+"ultimate")
   - `control_box` ← `voice.control`
   - `ref_audio` ← absolute path to `voice.reference_audio` if non-empty
   - `prompt_box` ← `voice.prompt_text`
   - `denoise_box` ← `voice.denoise`
   - `normalize_box` ← `voice.normalize`
   - `seed_text_box` ← `voice.prompt_text` if present, else current default seed text
3. Runs the existing `_apply_visibility(top, sub)` so the right fields show for the voice's mode.
4. Resets the preview state (any pending preview from a previous edit/create attempt is invalidated).
5. Shows an edit banner above the form: a `gr.Markdown(value="✏ 正在编辑：**<name>**", visible=True)` next to a `cancel_edit_btn = gr.Button("取消编辑", visible=True)` (both hidden in create mode).

### Cancel edit

`cancel_edit_btn.click` clears `edit_voice_id`, hides the banner, resets all form fields to their create-mode defaults, hides the rename button (see below).

### Save flow

`_on_save` checks `edit_voice_id`:

- **Empty (create)** — current behavior: `library.create(...)`.
- **Non-empty (edit)** — `library.update(voice_id, ..., audio_upload=preview_path, reference_audio_upload=ref_path)`. Both audio files are *overwritten in place* on disk at their existing paths (`voices/audio/<id>.wav` and `voices/audio/<id>.original.wav`).

Same as create, save requires that the user has clicked Preview first (preview generates the new `voice.audio`). After successful save, `edit_voice_id` clears and the form resets to create mode.

### Rename-only shortcut

Next to `name_box`, add a `rename_btn = gr.Button("保存名字", visible=False)`. The button is shown only while in edit mode (`edit_voice_id != ""`).

Click handler: if `edit_voice_id` is set and `name_box` differs from the voice's current name, call `library.update(voice_id, name=...)` (no audio touched). On success, refresh the table and dropdowns; status reads `✅ 已重命名为 <new_name>`. On `VoiceLibraryError` (e.g. duplicate name), show error inline; the voice's name in `_voices` is unchanged.

This lets users rename without going through preview→save.

### `VoiceLibrary.update()` extension

Current signature already supports `name`, `mode`, `control`, `reference_audio_upload`, `prompt_text`, `denoise`, `normalize`. It does **not** support overwriting `audio` (the generated preview). Add an `audio_upload` parameter mirroring `reference_audio_upload`:

```python
def update(
    self,
    voice_id: str,
    *,
    name: str | None = None,
    mode: Mode | None = None,
    control: str | None = None,
    reference_audio_upload: str | None = None,
    audio_upload: str | None = None,        # NEW
    prompt_text: str | None = None,
    denoise: bool | None = None,
    normalize: bool | None = None,
) -> str:
```

When `audio_upload` is given, stage it via `_stage_audio(audio_upload, v.id, suffix=".wav")` (same path as create, which means it overwrites the existing file by design — `shutil.copyfile` truncates). Set `v.audio` to the returned relative path.

`_validate_mode_invariants` is already called at end of update, so mode-switch invariants (design ⇒ no denoise, hifi ⇒ requires prompt_text, clone/hifi ⇒ requires reference_audio) are enforced post-edit.

### Mode switch during edit

Allowed. If user picks a new mode whose invariants aren't satisfied by the current form (e.g. switching to hifi without an upload), `_validate_mode_invariants` raises `VoiceLibraryError` and we surface that in `lib_status` without persisting.

## Out of scope

- Bulk operations (delete-many, export).
- Undo of delete or rename.
- Editing voices via the API (this is a UI-only change; the underlying `library.update` extension is reusable but no new API endpoints are added).
- Showing waveform thumbnails or duration in the listen column.

## Files affected

- `app.py` — voice-library tab UI, `_on_save`, `_on_voice_row_select`, new `_on_enter_edit`, `_on_cancel_edit`, `_on_rename_only`.
- `src/voxcpm_tts_tool/voice_library.py` — `VoiceLibrary.update()` gains `audio_upload` parameter.
- Tests in `tests/` — extend voice_library tests to cover `update(audio_upload=...)` overwriting both files; UI flow tests are out of scope (no existing UI tests).
