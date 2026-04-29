# VoxCPM TTS Tool Design

## Summary

Create a standalone project under `VoxCPM-TTS-Tool/` that uses the installed `voxcpm` Python package as its TTS SDK. The tool provides a Gradio web UI for voice-library management, script-based speech generation, non-language tag helpers, long-text generation, downloadable audio output, reference-audio transcription, optional reference denoising, and automatic model resolution.

The project is independent from local checkout folders. It must not import code from external repositories, depend on local source installs, use `PYTHONPATH` to reach external source trees, or read model files from another project. The behavior needed from prior exploration is written directly in this design; the implementation must live in this project and use `pip install voxcpm` for the SDK.

## Goals

- Build a standalone TTS web tool in a new directory.
- Manage reusable voices: create, update, delete, refresh.
- Support voice design, controllable cloning, and high-fidelity cloning records.
- Support high-fidelity cloning prompt text entry and optional reference-audio transcription via SenseVoiceSmall, triggered by an explicit Transcribe button (never auto-fired on upload).
- Support optional reference-audio denoising with ZipEnhancer for cloning workflows.
- Make the three generation modes visually distinct and mutually exclusive in the UI so users do not mix Voice Design inputs with reference-audio cloning inputs.
- Support UI language switching between Chinese and English.
- Generate speech from script text with default voice selection and inline `<voice name>` switches.
- Reset voice scope to the selected default voice at each new line.
- Support VoxCPM non-language tags such as `[laughing]`, `[sigh]`, `[Uhm]`, `[Shh]`, `[Question-ah]`, `[Question-ei]`, `[Question-en]`, `[Question-oh]`, `[Surprise-wa]`, `[Surprise-yo]`, and `[Dissatisfaction-hnn]`.
- Support long text by splitting into smaller generation segments and concatenating generated audio.
- Save generated audio to a downloadable file.
- Resolve required models automatically at startup: local project models first, then ModelScope, then Hugging Face where supported.

## Non-Goals

- Do not fork or rewrite `voxcpm` package internals.
- Do not run real VoxCPM model inference in automated tests.
- Do not implement a custom JavaScript code editor in the first version.
- Do not require users to manually edit JSON files.
- Do not import from external repositories or reuse model files stored outside `VoxCPM-TTS-Tool`, except when the user explicitly sets a supported model-path environment variable.

## Project Layout

```text
VoxCPM-TTS-Tool/
  app.py
  pyproject.toml
  README.md
  .gitignore
  src/
    voxcpm_tts_tool/
      __init__.py
      app_state.py
      generation.py
      long_text.py
      model_resolver.py
      script_parser.py
      voice_library.py
  tests/
    test_generation.py
    test_long_text.py
    test_script_parser.py
    test_voice_library.py
  voices/
    voices.json
    audio/
  outputs/
  pretrained_models/
    VoxCPM2/
    SenseVoiceSmall/
    ZipEnhancer/
```

Runtime data directories:

- `voices/voices.json`: voice metadata.
- `voices/audio/`: copied reference audio files.
- `outputs/`: generated `.wav` files.
- `pretrained_models/VoxCPM2/`: project-local VoxCPM2 model directory.
- `pretrained_models/SenseVoiceSmall/`: project-local ASR model directory used to transcribe reference audio into `prompt_text`.
- `pretrained_models/ZipEnhancer/`: project-local denoising model directory used by VoxCPM when reference denoising is enabled.

These runtime data directories should be ignored by Git except for optional `.gitkeep` placeholders.

## SDK Dependency

The standalone project should use the published package only:

- Install with `pip install voxcpm`. Always use the latest published version; do not pin minor/patch in `pyproject.toml`.
- The app imports the SDK as `import voxcpm`.
- The app must avoid importing private model internals from the package.
- The app must not use local source installs, external source trees, or local repository paths as SDK dependencies.
- `VoxCPM2` in this design refers to the model identifier (the folder under `pretrained_models/` and the ModelScope/Hugging Face repo IDs), not the Python package name. The Python package is always lowercase `voxcpm`.

Other runtime dependencies (declared in `pyproject.toml`):

- `gradio>=6,<7` — required transitively by `voxcpm>=2.0.2`. 3.x/4.x/5.x are not supported by this design (the v6 APIs `gr.Blocks`, queueing, file outputs are largely backward-compatible with v5 for our usage).
- `numpy`, `soundfile` — for waveform concatenation and `.wav` writing.
- `modelscope` — for ModelScope downloads in the resolver.
- `huggingface_hub` — already a transitive of `voxcpm`; declared explicitly because the resolver always calls `snapshot_download` itself for HF fallback (both VoxCPM2 and SenseVoiceSmall) so model files land in the project's `pretrained_models/` tree rather than HF's user cache.
- `funasr` — required for SenseVoiceSmall ASR inference. The app must start successfully without it; if the import fails, the Transcribe button is disabled and a status message explains why.

Versions other than `gradio` follow the same "use latest, don't pin minor/patch" preference unless a future incident proves otherwise.

## SDK Surface (verified against voxcpm 2.0.2)

Verified by inspecting the installed `voxcpm` package source. Implementation must conform to these signatures.

Construction (one instance per process, reused across requests):

```python
voxcpm.VoxCPM.from_pretrained(
    hf_model_id: str = "openbmb/VoxCPM2",   # ALSO accepts a local directory path; uses snapshot_download otherwise
    load_denoiser: bool = True,             # set False if ZipEnhancer is unavailable
    zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",  # accepts local path too
    cache_dir: str | None = None,
    local_files_only: bool = False,
    optimize: bool = True,                  # torch.compile + warmup generation; set False in tests
)
```

Generation:

```python
model.generate(
    text: str,                              # required; control instructions are encoded as a "(control)" prefix inside text
    prompt_wav_path: str | None = None,     # MUST be paired with prompt_text (SDK raises if exactly one is set)
    prompt_text: str | None = None,
    reference_wav_path: str | None = None,  # ONLY supported when the loaded model is VoxCPM2; required for clone/hifi
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    min_len: int = 2,
    max_len: int = 4096,
    normalize: bool = False,
    denoise: bool = False,                  # per-call toggle; only effective if denoiser was loaded at init
    retry_badcase: bool = True,
    retry_badcase_max_times: int = 3,
    retry_badcase_ratio_threshold: float = 6.0,
)
```

Behavior facts derived from the SDK:

- `from_pretrained` resolves `hf_model_id` as a local directory if `os.path.isdir(...)` is true, else calls `huggingface_hub.snapshot_download`. This design always passes a local directory path (see §Model Resolution); we never delegate the HF download to the SDK, so model files always land under `pretrained_models/`.
- ModelScope is not handled by the SDK. Project resolver must run the ModelScope download itself and then pass the resulting local path to `from_pretrained`.
- The SDK reads `config.json` `architecture` (`voxcpm` or `voxcpm2`) to choose the model class. `reference_wav_path` raises if the loaded model is not VoxCPM2; this project uses VoxCPM2 and is unaffected.
- ZipEnhancer is loaded once at init via `load_denoiser=True` + `zipenhancer_model_id=<path or id>`. Per-request enabling uses the `denoise=` flag on `generate`. Denoising the prompt/reference audio happens inside `generate` to a temporary wav.
- There is no separate `control` kwarg. The `(control)` text prefix is a text-level convention that the model interprets; this design treats control instructions purely as a prefix to `text`.

## Incorporated Behavior

These behaviors are part of this design and will be implemented inside this project:

- Reference-only cloning passes uploaded audio as `reference_wav_path` only.
- Ultimate cloning passes uploaded audio as both `reference_wav_path` and `prompt_wav_path`, plus editable `prompt_text`. The pairing constraint between `prompt_wav_path` and `prompt_text` is enforced by the SDK.
- On-demand transcription uses SenseVoiceSmall to fill `prompt_text`, triggered by an explicit Transcribe button (never on upload). Denoising is opt-in per voice/request via the `denoise=` flag, and is only effective if ZipEnhancer was successfully loaded at startup.

## Voice Library

Voice record format:

```json
{
  "id": "a1b2c3d4e5f6...",
  "name": "旁白女声",
  "control": "年轻女性，温柔清晰，适合旁白",
  "reference_audio": "voices/audio/<id>.wav",
  "prompt_text": "",
  "mode": "design",
  "denoise": false,
  "created_at": "2026-04-26T00:00:00Z",
  "updated_at": "2026-04-26T00:00:00Z"
}
```

Field rules:

- `id`: opaque string generated by the app as `uuid.uuid4().hex` at create time. Never derived from `name`, never user-editable, never reused after deletion.
- `name`: unique (case-insensitive after trimming whitespace) and non-empty. Used as the lookup key for inline `<voice name>` script switches.
- `mode`: one of `design`, `clone`, `hifi`.
- `denoise`: persisted per voice. Only meaningful for `clone` and `hifi`; ignored (and stored as `false`) for `design`. Drives the per-call `denoise=` flag at generation time.
- `reference_audio`: relative path under `voices/audio/`. v1 accepts only `.wav` uploads; other formats (`.mp3`, `.flac`, `.m4a`, etc.) are rejected at upload time with a UI message. This restriction avoids unverified assumptions about which audio loader the SDK's `build_prompt_cache` uses internally. The on-disk filename is `<id>.wav`. Multi-format support can be added in a future version after verifying the SDK loader's accepted extensions.
- `prompt_text`: free text (used only in `hifi` mode).

Mode-specific validation:

- `design` must not store `reference_audio`, `prompt_text`, or `denoise=true`; switching an existing voice to `design` clears those fields.
- `clone` requires `reference_audio`. May store `control` and `denoise`. `prompt_text` is cleared.
- `hifi` requires `reference_audio` and non-empty `prompt_text`. May store `denoise`. `control` is cleared.

Persistence rules:

- All writes to `voices.json` are atomic: write to `voices.json.tmp` in the same directory, `fsync`, then `os.replace` onto `voices.json`.
- If `voices.json` is malformed at load time, move it to `voices.broken-<UTC-timestamp>.json` (timestamped to avoid clobbering a previous broken file) and start with an empty library.
- If a saved reference audio file is missing at generation time, keep the record but block only that generation request with a clear error.

Rename and deletion:

- Renaming a voice updates `name` and `updated_at` but does not migrate references in user scripts. Existing scripts containing `<old_name>` will surface as unknown-tag warnings via the script-parser path. The UI shows a one-line warning at rename time so users are aware their scripts may need updating.
- Deleting a voice removes the record and its `reference_audio` file. If the deleted voice is the current default, the UI falls back to the first available voice; if no voices remain, it falls back to the ephemeral `__default__` voice per §Error Handling.

## Model Resolution

Startup should automatically resolve all required model paths before the first generation request. Each resolver should treat a directory as available only when it exists and contains at least one expected model/config file, so empty folders do not block download attempts. The UI should show the source selected for each model and any failed download messages.

The resolver should set runtime cache environment variables before importing heavy model libraries:

- `TOKENIZERS_PARALLELISM=false`
- Optional defaults for `HF_HOME`, `MODELSCOPE_CACHE`, `TRANSFORMERS_CACHE`, and `HF_DATASETS_CACHE` under a local or user cache directory when the variables are unset.

VoxCPM2 TTS model resolution:

1. Use `VOXCPM_MODEL_DIR` if it points to a valid local directory.
2. Use local `pretrained_models/VoxCPM2` if it is valid.
3. Try ModelScope download `OpenBMB/VoxCPM2` to `pretrained_models/VoxCPM2`.
4. Fall back to Hugging Face download `openbmb/VoxCPM2` to `pretrained_models/VoxCPM2`.

SenseVoiceSmall ASR model resolution:

1. Use `VOXCPM_ASR_MODEL_DIR` if it points to a valid local directory.
2. Use local `pretrained_models/SenseVoiceSmall` if it is valid.
3. Try ModelScope download `iic/SenseVoiceSmall` to `pretrained_models/SenseVoiceSmall`.
4. Fall back to Hugging Face download `FunAudioLLM/SenseVoiceSmall` to `pretrained_models/SenseVoiceSmall`.

ZipEnhancer denoising model resolution:

1. Use `ZIPENHANCER_MODEL_PATH` if it points to a valid local directory.
2. Use local `pretrained_models/ZipEnhancer` if it is valid.
3. Try ModelScope download `iic/speech_zipenhancer_ans_multiloss_16k_base` to `pretrained_models/ZipEnhancer`.

A "valid" local directory in all three model resolvers above means: directory exists AND contains at least one of the model's expected files (e.g. `config.json` or `model.safetensors` for VoxCPM2, `model.pt` or equivalent for SenseVoiceSmall, `pytorch_model.bin` or equivalent for ZipEnhancer). Empty directories or directories holding only `.gitkeep` are treated as missing so the next resolver step still runs. The same validity check applies to env-var-pointed paths (`VOXCPM_MODEL_DIR`, `VOXCPM_ASR_MODEL_DIR`, `ZIPENHANCER_MODEL_PATH`) — pointing the env var at an empty directory does not skip downloading.

For the Hugging Face fallback steps, the resolver downloads via `huggingface_hub.snapshot_download(repo_id=..., local_dir="pretrained_models/<name>")` and then passes the resulting local path to `from_pretrained`. The resolver never passes a remote HF id directly to the SDK; this keeps all model files under the project's `pretrained_models/` tree rather than scattered through HF's user-cache directory.

ASR and denoising are feature dependencies, not substitutes for the VoxCPM2 TTS model. VoxCPM2 must resolve successfully before generation can run. ASR failure disables the Transcribe button but still allows manual `prompt_text`. ZipEnhancer failure disables reference denoising but still allows generation without denoising.

`pretrained_models/` is ignored by Git and Mutagen. Remote machines must perform their own startup detection and downloads instead of receiving model files through file sync.

## UI Design

Use Gradio tabs:

- `语音生成` / `Speech Generation`: Default voice selection, script text input, insertion helpers, advanced generation settings, generate button, audio preview, download file, and segment log.
- `音色管理` / `Voice Library`: Voice list, mode selector, mode-specific fields, save new, update, delete, and refresh.
- `使用说明` / `Usage`: Syntax examples, newline reset rule, non-language tag list, long-text behavior, and quality tips.

The generation tab should not directly edit voices. Voice changes happen in the management tab and refresh the dropdown choices used by generation.

UI language switching:

- A top-level language control switches visible UI labels, helper text, tab names, status messages, and usage docs between `中文` and `English`.
- The selected UI language does not change generation language. Users still control speech language through target text and control instructions.
- The first implementation uses a small in-app translation dictionary keyed by stable string IDs. It does not need browser locale detection.
- Missing-key fallback order: requested language → Chinese (`zh`) → the literal string ID. Tests assert that the Chinese dictionary contains every ID used in the app.

Mode-specific voice editing UI:

- The voice management tab must show a mode selector first, with exactly three choices: `声音设计 / Voice Design`, `可控克隆 / Controllable Cloning`, and `极致克隆 / Ultimate Cloning`.
- Only fields relevant to the selected mode are visible. Hidden fields are not submitted for that mode.
- `Voice Design`: show `Voice Name`, `Control Instruction`, and optional mode-specific tips. Do not show `Reference Audio`, `Prompt Text`, `Transcribe button`, or denoise controls.
- `Controllable Cloning`: show `Voice Name`, `Reference Audio`, optional `Control Instruction`, and optional denoise control. Do not show `Prompt Text` or `Transcribe button`.
- `Ultimate Cloning`: show `Voice Name`, `Reference Audio`, `Prompt Text / Transcript`, `Transcribe button`, and optional denoise control. Do not show `Control Instruction`.
- Loading an existing voice switches the mode selector and re-renders the correct field group for that record.

Because Gradio Textbox does not provide robust editor-level autocomplete, the first version uses stable insertion helpers:

- A searchable voice dropdown plus an insert button that appends `<voice name>` to the script box.
- A non-language tag dropdown plus an insert button that appends tags such as `[laughing]`.

Reference-audio transcription behavior applies only inside Ultimate Cloning:

- In `hifi` mode, uploading a reference audio file does NOT auto-fire transcription. Instead, an explicit `Transcribe / 识别转写` button next to the `prompt_text` field runs SenseVoiceSmall with automatic language detection and inverse text normalization, then writes the result into `prompt_text`. The button label deliberately avoids the word "自动" so users do not expect upload-time auto-fire. This avoids silently overwriting any text the user already typed.
- `prompt_text` remains editable both before and after transcription.
- If ASR is unavailable (model failed to resolve), the Transcribe button is disabled and the UI shows a model-status message; manual `prompt_text` entry still works.

Reference denoising behavior:

- The denoise toggle is available for clone and hifi records only and is persisted on the voice record.
- ZipEnhancer is loaded once at app startup via `from_pretrained(load_denoiser=True, zipenhancer_model_id=<resolved path>)`. The path is NOT passed per-call.
- For each generation request, the SDK call uses `generate(..., denoise=<voice.denoise AND zipenhancer_loaded>)`. See §Generation Flow for the full conjunction rule.
- If ZipEnhancer was not loaded at startup, the toggle has no effect; generation proceeds without denoising and a one-time UI warning explains why.

## Script Semantics

Line splitting:

- The script is split on any of `\r\n`, `\r`, or `\n` (CRLF, CR, LF). Each resulting line is parsed independently.
- Empty lines and whitespace-only lines produce no segments and yield no audio chunks.

Default voice:

- The selected default voice is used at the start of every line.
- If a line contains no valid `<voice name>` switch, the whole line uses the default voice.

Voice switch (`<voice name>`):

- Pattern: a literal `<`, the voice name, and a literal `>`. The voice name is the raw substring between the brackets; leading/trailing whitespace inside the brackets is trimmed before matching.
- Matching against the voice library is case-insensitive after trimming whitespace (consistent with the uniqueness rule in §Voice Library).
- The voice name itself may contain spaces, CJK characters, digits, and punctuation, but MUST NOT contain `<` or `>`. There is no escape syntax for `<`/`>` inside a voice name; voices created with such characters in the name are rejected at save time.
- A switch remains active until another valid `<voice name>` appears in the same line.
- A new line resets the active voice back to the selected default voice.

Unknown angle-bracket text:

- If `<...>` does not match any voice (after the trim+case-insensitive comparison), the literal `<...>` substring is preserved verbatim in the output text and a warning is appended to the segment log identifying the line number and the unmatched substring.

Non-language tags:

- Square-bracket tags such as `[laughing]` are preserved verbatim in text and passed through to VoxCPM as part of the `text` argument.
- The helper UI lists only the tags enumerated in §Goals (the official recommended set).
- Square-bracket tags never affect voice switching and are not validated against any closed list at parse time — unknown bracket content is passed through silently.

Example:

```text
<女声>大家好。[laughing] 今天我们介绍 VoxCPM。<男声>下面换一个声音。
这一行没有指定音色，所以回到默认音色。
```

## Generation Flow

1. Configure model cache environment variables.
2. Resolve VoxCPM2, SenseVoiceSmall, and ZipEnhancer project-local model paths.
3. Load voice library and resolve the default voice.
4. Parse script text line by line into logical voice segments.
5. Split each logical segment into smaller chunks for long-text generation.
6. For each chunk, build VoxCPM generation arguments from the active voice.
7. Generate waveforms sequentially. Concurrency is enforced at two layers in Gradio 5: (a) `Blocks.queue(default_concurrency_limit=1)` at app construction, AND (b) `concurrency_limit=1` on the generate button's `.click(...)` handler. Both are required because per-event limits override the default.
8. Concatenate waveforms with `numpy.concatenate`.
9. Save the final waveform to `outputs/YYYYMMDD-HHMMSS-mmm.wav` (millisecond suffix to avoid collisions on rapid retries; if a collision still occurs, append `-1`, `-2`, ... until a free name is found). The sample rate written to the `.wav` header is read from the loaded VoxCPM model instance at write time and never hard-coded in the orchestration code.
10. Return audio preview, downloadable file path, and a segment log.

Generation arguments by mode (see §SDK Surface for full signature):

- `design`:
  - `text = f"({control}){chunk}"` when control is non-empty, else `text = chunk`.
  - No `prompt_wav_path`, no `prompt_text`, no `reference_wav_path`.
- `clone`:
  - `text = f"({control}){chunk}"` when control is non-empty, else `text = chunk`.
  - `reference_wav_path = <voice.reference_audio>`.
  - No `prompt_wav_path`, no `prompt_text`.
- `hifi`:
  - `text = chunk` (control is ignored for hifi).
  - `prompt_wav_path = <voice.reference_audio>`, `prompt_text = <voice.prompt_text>` (must be non-empty; SDK enforces pairing).
  - `reference_wav_path = <voice.reference_audio>` (same path as `prompt_wav_path`).

Per-call `denoise` flag:

- For each `generate(...)` call, set `denoise=True` only if (a) the active voice's denoise toggle is on, AND (b) ZipEnhancer was loaded successfully at startup. Otherwise `denoise=False`.

Model construction (one instance, reused across requests):

- Call `voxcpm.VoxCPM.from_pretrained(hf_model_id=<resolved VoxCPM2 local path>, load_denoiser=<ZipEnhancer available>, zipenhancer_model_id=<resolved ZipEnhancer local path or default id>, optimize=True)`. Tests pass `optimize=False` (see §SDK Surface) to skip `torch.compile` and the warmup generation.
- `from_pretrained` always receives a local directory path. HF/ModelScope downloads happen inside §Model Resolution before this call; the SDK is never asked to download.
- If ZipEnhancer is unavailable, pass `load_denoiser=False`. The conjunction rule above forces the per-call `denoise=` to `False`, and the UI shows a one-time warning at startup explaining that the denoise toggle has no effect.

## Long Text

Long text splitting should happen after script parsing so voice boundaries are preserved. The first implementation is a self-contained punctuation-aware splitter that:

- Splits on the union of CJK and ASCII sentence terminators: `。！？；…!?;` plus newline-equivalents already removed by the script parser.
- Falls back to comma-class boundaries `，、,` only when a single sentence exceeds the per-chunk character budget.
- Treats any `[xxx]` square-bracket tag as an indivisible token: never splits inside the brackets and never inserts a chunk boundary between an immediately-preceding word and the bracket.
- Treats any `<voice name>` switch as already-resolved before this stage, so the splitter never sees angle brackets.

Per-chunk character budget is a single tunable constant in the splitter module (suggested initial value: 80 CJK characters or 200 ASCII characters per chunk; tune in plan stage).

Concatenation should preserve chunk order. Optional silence insertion can be added later, but the first version should avoid artificial pauses unless a generated chunk already contains them.

## Error Handling

- Empty script text: block generation with a clear message.
- No voices defined: synthesize an in-memory ephemeral voice `{id: "__default__", name: "__default__", mode: "design", control: ""}` for use as the default. This voice is NOT persisted to `voices.json`, NOT listed in the Voice Library tab, and NOT addressable via `<voice name>` script switches (the script parser excludes ids beginning with `__`). The Speech Generation tab's voice dropdown shows it under an i18n-translated label (`默认` / `Default`). It disappears as soon as the user creates any real voice.
- Deleted selected voice: refresh choices and fall back to first available voice.
- Missing reference audio: block only the affected generation request.
- VoxCPM2 model download failure: block generation and show local path expectation plus upstream errors.
- SenseVoiceSmall download failure: disable the Transcribe button, keep manual `prompt_text`, and show the ASR error.
- ZipEnhancer download failure: leave the per-voice denoise toggle visible but inert, keep generation available, and show a one-time startup warning explaining the toggle has no effect (matches §Reference denoising behavior).
- `funasr` import failure at startup: same effect as SenseVoiceSmall download failure — disable the Transcribe button and show a startup status message; manual `prompt_text` entry still works.
- ASR transcription failure for one file: keep the uploaded reference audio, preserve any existing `prompt_text`, and show the audio path plus the recognizer error.
- Generation failure in one segment: stop the job and show the segment index, voice name, and text preview.

## Testing Strategy

Automated tests use fake model objects instead of real VoxCPM inference. A fake model implements the same small interface used by orchestration, records calls, and returns a tiny zero-filled NumPy waveform. This keeps tests fast, deterministic, offline, and GPU-free.

Test coverage:

- Voice library create, update, delete, duplicate-name validation (case-insensitive after trim), and malformed JSON recovery (timestamped backup).
- Voice library atomic write: simulate a crash between `tmp` write and `os.replace` and verify the original `voices.json` is intact.
- Reference-audio upload accepts only `.wav` (case-insensitive); `.mp3`/`.flac`/`.m4a` etc. are rejected with a UI message. Saved filename is `<id>.wav`.
- Script parser: line splitting on `\r\n`/`\r`/`\n`; empty/whitespace-only lines produce no segments; voice switch trimming and case-insensitive matching; CJK voice names; unknown `<...>` preserved verbatim with line-numbered warning; rejection of voice names containing `<` or `>` at save time.
- Long-text splitting without crossing voice boundaries and without breaking square-bracket tags.
- Generation orchestration maps `design`, `clone`, and `hifi` voices to the expected `voxcpm.VoxCPM.generate(...)` kwargs as specified in §Generation Flow, including the empty-control branch (`text=chunk`) and the populated-control branch (`text="(control)chunk"`).
- Per-call `denoise=` flag is True only when both the voice's denoise toggle is on AND ZipEnhancer was loaded at startup.
- Voice validation clears or rejects mode-incompatible fields: no reference audio in design, no prompt text in clone, no control instruction in hifi, no `denoise=true` in design.
- UI visibility tests or lightweight callback tests verify mode changes expose only the relevant field group.
- I18N tests verify the Chinese dictionary contains every string ID; missing-key fallback returns Chinese, then the literal ID.
- Model resolver tests cover VoxCPM2, SenseVoiceSmall, and ZipEnhancer local hits, environment-variable overrides, ModelScope downloads, Hugging Face fallback for VoxCPM2 and SenseVoiceSmall, partial auxiliary-model failures, and rejection of empty/`.gitkeep`-only directories as "valid".
- ASR orchestration tests use a fake recognizer and verify the explicit Transcribe button writes into `prompt_text` and never auto-fires on upload.
- Output filename collision handling: simulate two writes within the same millisecond and verify the second gets a `-1` suffix.
- Final waveform concatenation preserves segment order.

Manual verification:

- Install dependencies.
- Launch `python app.py --port 8808`.
- Confirm automatic model resolution for VoxCPM2, SenseVoiceSmall, and ZipEnhancer.
- Confirm UI language can switch between Chinese and English without changing saved voice data.
- Confirm Voice Design does not show reference-audio upload controls.
- Confirm Controllable Cloning shows reference audio and optional Control Instruction, but no transcript field.
- Confirm Ultimate Cloning shows reference audio and transcript controls, but no Control Instruction field.
- Confirm hifi reference audio fills `prompt_text` only after clicking the Transcribe button (never on upload), and that the field remains manually editable.
- Confirm clone/hifi generation works with denoising enabled and disabled.
- Create design, clone, and hifi voices.
- Generate a multi-line script with inline voice switches and non-language tags.
- Download and play the generated audio.
