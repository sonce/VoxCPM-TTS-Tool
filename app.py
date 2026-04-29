"""Gradio entry point. Thin wiring layer over voxcpm_tts_tool modules."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

def _silence_proactor_conn_reset() -> None:
    """Suppress Windows-specific asyncio noise on browser disconnect.

    On Windows, Gradio runs over the asyncio Proactor event loop. When a
    browser tab closes/refreshes, the underlying socket is half-closed and
    ProactorBasePipeTransport tries `socket.shutdown(SHUT_RDWR)`, which
    raises `ConnectionResetError [WinError 10054]`. asyncio's default
    exception handler logs this at ERROR level. The condition is benign
    (the peer just went away) but it floods the console. We attach a
    narrow filter to the asyncio logger that drops only this specific
    pattern; all other asyncio errors are still logged normally.
    """
    if sys.platform != "win32":
        return

    class _ConnResetFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.exc_info:
                exc = record.exc_info[1]
                if isinstance(exc, ConnectionResetError):
                    return False
            msg = record.getMessage()
            return not ("ConnectionResetError" in msg and "10054" in msg)

    logging.getLogger("asyncio").addFilter(_ConnResetFilter())


_silence_proactor_conn_reset()


import gradio as gr

from voxcpm_tts_tool import i18n, model_resolver
from voxcpm_tts_tool.app_state import (
    AppState,
    EPHEMERAL_DEFAULT_VOICE_ID,
    ephemeral_default_voice,
    paths_for,
)
from voxcpm_tts_tool.generation import (
    Progress,
    Result,
    run_generation,
    synthesize_voice_preview,
)
from voxcpm_tts_tool.output_writer import write_output_wav
from voxcpm_tts_tool.transcription import (
    AsrUnavailable,
    SenseVoiceTranscriber,
    load_real_transcriber,
)
from voxcpm_tts_tool.ui_callbacks import (
    effective_mode,
    field_visibility,
    insert_tag,
    insert_voice_tag,
    voice_dropdown_choices,
)
from voxcpm_tts_tool.script_parser import (
    NON_LANG_TAG_MAP_ZH,
    localize_non_lang_tags,
    parse_script,
)
from voxcpm_tts_tool.voice_library import (
    VoiceLibrary,
    VoiceLibraryError,
)

# Dropdown shows the Chinese labels (keys); the inserted text gets `[zh_label]`.
# Generation-time substitution (see script_parser.localize_non_lang_tags) maps
# these to the English tokens the SDK expects.
NON_LANG_TAGS = list(NON_LANG_TAG_MAP_ZH.keys())


def initialize(project_root: Path) -> tuple[AppState, list[str]]:
    """Resolve all models and build the AppState. Returns (state, startup_messages)."""
    paths = paths_for(project_root)
    model_resolver.configure_runtime_caches(paths.cache_dir)
    messages: list[str] = []

    # VoxCPM2 — required.
    voxcpm_path = model_resolver.resolve_voxcpm(
        paths.voxcpm_dir,
        modelscope_download=model_resolver.real_modelscope_download,
        hf_download=model_resolver.real_hf_download,
    )
    messages.append(f"VoxCPM2 ready at {voxcpm_path}")

    # ZipEnhancer — optional.
    zipenhancer_loaded = False
    zipenhancer_path = None
    try:
        zipenhancer_path = model_resolver.resolve_zipenhancer(
            paths.zipenhancer_dir,
            modelscope_download=model_resolver.real_modelscope_download,
        )
        zipenhancer_loaded = True
        messages.append(f"ZipEnhancer ready at {zipenhancer_path}")
    except model_resolver.ModelResolutionError as exc:
        messages.append(f"ZipEnhancer unavailable: {exc} ({i18n.t('status.denoise_unavailable', 'zh')})")

    # SenseVoiceSmall — optional.
    transcriber: SenseVoiceTranscriber
    try:
        sv_path = model_resolver.resolve_sensevoice(
            paths.sensevoice_dir,
            modelscope_download=model_resolver.real_modelscope_download,
            hf_download=model_resolver.real_hf_download,
        )
        transcriber = load_real_transcriber(sv_path)
        if transcriber.is_available:
            messages.append(f"SenseVoice ready at {sv_path}")
        else:
            messages.append(f"SenseVoice loaded path but funasr import failed: {i18n.t('status.asr_unavailable', 'zh')}")
    except model_resolver.ModelResolutionError as exc:
        transcriber = SenseVoiceTranscriber.unavailable(str(exc))
        messages.append(f"SenseVoice unavailable: {exc}")

    # Patch voxcpm CPU attention-mask bug (issue #71) before importing the SDK.
    # See src/voxcpm_tts_tool/voxcpm_patch.py for details.
    from voxcpm_tts_tool import voxcpm_patch
    if voxcpm_patch.apply():
        messages.append("Applied voxcpm CPU attention-mask patch (upstream issue #71)")

    # Build VoxCPM model singleton.
    import voxcpm
    import inspect
    # Diagnostic: print SDK version + _generate signature so we can spot any
    # API drift (e.g. reference_wav_path missing in older voxcpm).
    try:
        ver = getattr(voxcpm, "__version__", "unknown")
        sig = inspect.signature(voxcpm.VoxCPM._generate)
        print(f"[voxcpm] version={ver}", file=sys.stderr, flush=True)
        print(f"[voxcpm] VoxCPM._generate{sig}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[voxcpm] could not introspect: {exc}", file=sys.stderr, flush=True)

    model = voxcpm.VoxCPM.from_pretrained(
        hf_model_id=str(voxcpm_path),
        load_denoiser=zipenhancer_loaded,
        zipenhancer_model_id=str(zipenhancer_path) if zipenhancer_path is not None else None,
        optimize=True,
    )
    # voxcpm doesn't expose sample_rate uniformly; sniff once.
    if not hasattr(model, "sample_rate"):
        model.sample_rate = getattr(getattr(model, "tts_model", None), "sample_rate", 16000)

    library = VoiceLibrary(paths.voices)
    return (
        AppState(
            paths=paths,
            library=library,
            model=model,
            transcriber=transcriber,
            zipenhancer_loaded=zipenhancer_loaded,
        ),
        messages,
    )


def build_ui(state: AppState, startup_messages: list[str]) -> gr.Blocks:
    """Build the Gradio UI. `startup_messages` is accepted but not rendered;
    the same content is printed to the console by `main()` for ops visibility.
    """
    ephemeral = ephemeral_default_voice()
    _ = startup_messages  # kept in signature for callers; intentionally not shown in UI

    # Per-row ▶/⏹ toggle reads this. Single-user local app, runtime attribute
    # is sufficient (no per-session isolation needed).
    if not hasattr(state, "_playing_voice_id"):
        state._playing_voice_id = ""
    # Two-click confirmation pending state for the 🗑 column. First click on a
    # row's 🗑 sets this; second click on the SAME row's 🗑 within the timeout
    # actually deletes. Any other interaction clears it.
    if not hasattr(state, "_delete_pending"):
        state._delete_pending = {"id": "", "t": 0.0}
    # Path to push into the listen widget in the .then() phase after a row
    # click. Phase 1 always clears the audio (value=None); phase 2 sets it to
    # this path. Two-phase is required because Gradio collapses "set to same
    # value" updates, which breaks autoplay on the second click of the same row.
    if not hasattr(state, "_pending_listen_path"):
        state._pending_listen_path = ""
    if not hasattr(state, "gen_audio_result"):
        state.gen_audio_result = None

    # CSS-hide the listen audio: keep it visible to Gradio (autoplay needs the
    # element actually mounted in the DOM, not display:none), but remove the
    # waveform/transport UI so the per-row ▶/⏹ toggle is the only control.
    _VOICE_TAB_CSS = """
    #voxcpm-listen-audio { position: absolute; left: -9999px; width: 1px; height: 1px;
                           overflow: hidden; opacity: 0; pointer-events: none; }
    #voxcpm-chunk-audio  { position: absolute; left: -9999px; width: 1px; height: 1px;
                           overflow: hidden; opacity: 0; pointer-events: none; }
    """

    with gr.Blocks(title="VoxCPM TTS Tool", css=_VOICE_TAB_CSS) as demo:

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
            stop_btn = gr.Button("停止", variant="secondary", visible=False)
            status_md = gr.Markdown(value="")
            audio_out = gr.Audio(
                label="结果", type="filepath", interactive=False,
            )
        # ---- Voice library tab ----
        with gr.Tab(i18n.t("tab.voice_library", "zh")) as tab_lib:
            # Refresh button sits in its own narrow row above the table.
            with gr.Row():
                refresh_btn = gr.Button(i18n.t("btn.refresh", "zh"), scale=0, size="sm")
            # Per-row click semantics by column index (see _on_voice_row_select):
            #   0 名称 / 1 模式 → enter edit mode
            #   2 🔊            → load this voice into listen_audio
            #   3 🗑            → confirm-then-delete
            voice_list = gr.Dataframe(
                headers=["名称", "模式", "🔊", "🗑"],
                value=_voice_list_rows(state.library.list_voices()),
                interactive=False,
                wrap=True,
            )
            # Shared playback widget for the 🔊 column. Visible to Gradio so
            # autoplay actually fires (browsers block autoplay on display:none
            # elements); CSS-hidden via elem_id so the user only sees the
            # per-row ▶/⏹ toggle in the table.
            listen_audio = gr.Audio(
                label="试听",
                type="filepath",
                autoplay=True,
                interactive=False,
                show_label=False,
                container=False,
                elem_id="voxcpm-listen-audio",
            )
            # Edit-mode banner: visible only while edit_voice_id is non-empty.
            with gr.Row(visible=False) as edit_banner_row:
                edit_banner = gr.Markdown(value="", elem_id="voice-edit-banner")
                cancel_edit_btn = gr.Button("取消编辑", variant="secondary", scale=0, size="sm")
            top_mode_radio = gr.Radio(
                choices=[("声音设计 / Voice Design", "design"),
                         ("克隆 / Cloning", "cloning")],
                value="design",
                label="模式",
            )
            name_box = gr.Textbox(label=i18n.t("field.voice_name", "zh"))
            # Rename-only shortcut — sits directly under the name textbox.
            # Visible only while editing an existing voice.
            rename_btn = gr.Button(
                "保存名字",
                variant="secondary",
                size="sm",
                visible=False,
            )
            # Reference-audio block: upload + auto-ASR transcript + denoise toggle
            # all stay together so user sees the full "voice source" group.
            with gr.Group():
                ref_audio = gr.Audio(label=i18n.t("field.reference_audio", "zh"),
                                     type="filepath", visible=False)
                prompt_box = gr.Textbox(label=i18n.t("field.prompt_text", "zh"), visible=False)
                denoise_box = gr.Checkbox(label=i18n.t("field.denoise", "zh"), visible=False)
            # Sub-radio sits right above control_box: switching to "Ultimate"
            # hides the control textbox per user spec.
            sub_mode_radio = gr.Radio(
                choices=[("可控克隆 / Controllable", "controllable"),
                         ("极致克隆 / Ultimate", "ultimate")],
                value="controllable",
                label="克隆方式",
                visible=False,  # shown only when top_mode_radio == "cloning"
            )
            control_box = gr.Textbox(
                label=i18n.t("field.control", "zh"),
                placeholder="例：年轻女性，温柔甜美 / 中年男性，低沉浑厚 / 语速稍快，情绪激动",
                info="生成时会作为 (控制指令) 前缀拼到文本前；留空则不加前缀。",
                visible=True,
            )
            seed_text_box = gr.Textbox(
                label=i18n.t("field.seed_text", "zh"),
                value="你好，这是一个用于固化音色的样本朗读。",
                lines=2,
                visible=True,
            )
            # Normalize is a text-handling property — sits right under the text
            # it normalizes (seed_text).
            normalize_box = gr.Checkbox(label=i18n.t("field.normalize", "zh"), visible=True)
            # Advanced tuning, mirrored from the Generation tab. Defaults match
            # the VoxCPM SDK; only forwarded when user changes them. Used for
            # the preview generation in all modes (design / clone / hifi).
            with gr.Accordion(i18n.t("field.advanced", "zh"), open=False):
                lib_cfg_value_slider = gr.Slider(
                    minimum=1.0, maximum=3.0, step=0.1, value=2.0,
                    label=i18n.t("field.cfg_value", "zh"),
                )
                lib_inference_timesteps_slider = gr.Slider(
                    minimum=4, maximum=30, step=1, value=10,
                    label=i18n.t("field.inference_timesteps", "zh"),
                )
            # Two-step save flow: preview → save (save shown only after a successful preview).
            preview_btn = gr.Button(i18n.t("btn.preview", "zh"), variant="primary")
            # Always visible so Gradio's per-component loading skin can render
            # in place during the SDK call. Empty (value=None) until preview
            # is generated; cleared back to None on cancel/save/error.
            preview_audio_out = gr.Audio(
                label=i18n.t("field.preview_audio", "zh"),
                type="filepath",
                visible=True,
                interactive=False,
            )
            with gr.Row():
                save_btn = gr.Button(i18n.t("btn.save", "zh"), variant="primary", visible=False)
                # "Save as" — create a NEW voice from the current form + preview
                # without touching the voice being edited. Visible only in edit
                # mode after a successful preview.
                save_as_btn = gr.Button("另存为", variant="secondary", visible=False)
            lib_status = gr.Markdown()
            # Holds (preview_wav_path, transcript) between Preview and Save clicks.
            preview_state = gr.State(("", ""))
            # Empty string = creating a new voice; non-empty = editing this voice id.
            edit_voice_id = gr.State("")

        # ---- Usage tab ----
        with gr.Tab(i18n.t("tab.usage", "zh")):
            gr.Markdown(_usage_doc("zh"))

        # ---- Wire callbacks ----
        def _hide_preview_and_save():
            """Reset preview state + hide save / save-as buttons + preview audio.
            Called whenever the user changes mode or any input that invalidates the preview."""
            return (
                ("", ""),                              # preview_state
                gr.update(value=None),                 # preview_audio_out
                gr.update(visible=False),              # save_btn
                gr.update(visible=False),              # save_as_btn
            )

        def _apply_visibility(top: str, sub: str):
            """Update all mode-controlled widgets based on top + sub radios."""
            mode = effective_mode(top, sub)
            vis = field_visibility(mode)
            sub_visible = (top == "cloning")
            preview_state_v, preview_audio_v, save_btn_v, save_as_btn_v = _hide_preview_and_save()
            return (
                gr.update(visible=sub_visible),         # sub_mode_radio
                gr.update(visible=vis["reference_audio"]),  # ref_audio
                gr.update(visible=vis["prompt_text"]),  # prompt_box
                gr.update(visible=vis["control"]),      # control_box
                gr.update(visible=vis["denoise"]),      # denoise_box
                gr.update(visible=vis["seed_text"]),    # seed_text_box
                preview_state_v,
                preview_audio_v,
                save_btn_v,
                save_as_btn_v,
            )

        _MODE_OUTPUTS = [
            sub_mode_radio, ref_audio, prompt_box, control_box,
            denoise_box, seed_text_box,
            preview_state, preview_audio_out, save_btn, save_as_btn,
        ]
        top_mode_radio.change(_apply_visibility,
                              inputs=[top_mode_radio, sub_mode_radio],
                              outputs=_MODE_OUTPUTS)
        sub_mode_radio.change(_apply_visibility,
                              inputs=[top_mode_radio, sub_mode_radio],
                              outputs=_MODE_OUTPUTS)

        def _on_insert_voice(script: str, name: str | None) -> str:
            if not name:
                return script
            return insert_voice_tag(script, voice_name=name)

        insert_voice_btn.click(_on_insert_voice, inputs=[text_box, voice_picker], outputs=text_box)

        def _on_insert_tag(script: str, tag: str | None) -> str:
            if not tag:
                return script
            return insert_tag(script, tag=tag)

        insert_tag_btn.click(_on_insert_tag, inputs=[text_box, tag_picker], outputs=text_box)

        # ---- Generation: three-phase chain to match the preview's loading UX.
        # Phase 1 (click, instant): clear audio_out + switch buttons.
        # Phase 2 (then, streaming): progress in status_md, store wav path in state.
        # Phase 3 (then, show_progress="full"): deliver audio with loading overlay.

        def _gen_phase1():
            return (
                gr.update(value=None, label="⏳ 结果（生成中…）"),   # audio_out
                "▶ 准备生成…",                                        # status_md
                gr.update(interactive=False, visible=False),           # generate_btn
                gr.update(interactive=False),                          # text_box
                gr.update(interactive=False),                          # default_voice_dd
                gr.update(visible=True, value="停止"),                 # stop_btn
            )

        def _gen_phase2(default_id, text, max_duration, cfg_value, inference_timesteps):
            default = state.default_voice(default_id)
            prepped = localize_non_lang_tags(text)
            by_name = {v.name: v for v in state.library.list_voices()}
            if not default.id.startswith("__"):
                by_name[default.name] = default
            _segs, warnings = parse_script(
                prepped, default_voice=default.name, known_names=list(by_name),
            )
            warn_md = "" if not warnings else "⚠ " + "; ".join(warnings) + "\n\n"

            state.gen_stop_flag = False
            state.gen_audio_result = None
            try:
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
                        yield warn_md + f"▶ 正在生成第 {ev.done}/{ev.total} 段"
                    else:  # Result
                        if ev.wav.size == 0:
                            yield warn_md + "ℹ 没有可生成的内容"
                        else:
                            out_path = write_output_wav(
                                ev.wav,
                                sample_rate=ev.sample_rate,
                                outputs_dir=state.paths.outputs,
                            )
                            state.gen_audio_result = str(out_path)
                            msg = ("⏸ 已停止，已导出前面段" if ev.was_stopped
                                   else "✅ 已完成")
                            yield warn_md + msg
            except Exception as e:
                traceback.print_exc()
                yield warn_md + f"❌ 生成失败：{e}"
            finally:
                state.gen_stop_flag = False

        def _gen_phase3():
            path = state.gen_audio_result
            state.gen_audio_result = None
            return (
                gr.update(value=path, label="结果") if path else gr.update(value=None, label="结果"),
                gr.update(interactive=True, visible=True),   # generate_btn
                gr.update(visible=False),                    # stop_btn
                gr.update(interactive=True),                 # text_box
                gr.update(interactive=True),                 # default_voice_dd
            )

        generate_btn.click(
            _gen_phase1,
            outputs=[audio_out, status_md, generate_btn, text_box, default_voice_dd, stop_btn],
        ).then(
            _gen_phase2,
            inputs=[default_voice_dd, text_box, max_duration_slider,
                    cfg_value_slider, inference_timesteps_slider],
            outputs=[status_md],
            concurrency_id="gen",
            concurrency_limit=1,
        ).then(
            _gen_phase3,
            outputs=[audio_out, generate_btn, stop_btn, text_box, default_voice_dd],
            show_progress="full",
        )

        def _on_stop():
            """Flip the stop flag. Lives in concurrency_id="control" so it runs
            even while `gen` is busy."""
            state.gen_stop_flag = True
            return gr.update(value="⏳ 停止中…")

        stop_btn.click(
            _on_stop,
            outputs=stop_btn,
            concurrency_id="control",
            concurrency_limit=None,
        )

        # Auto-transcribe whenever the reference audio is uploaded or changed
        # (Cloning modes only). Replaces the old explicit Transcribe button.
        # In edit mode, entering a voice programmatically sets ref_audio; that
        # also fires .change. We detect "this is the edit-mode programmatic
        # load" by comparing the audio path against the voice's stored
        # reference_audio and skipping ASR (which would overwrite the prompt
        # we just populated). Genuine new uploads while editing still re-ASR.
        def _on_ref_audio_change(audio_path, top, sub, edit_id):
            preview_state_v, preview_audio_v, save_btn_v, save_as_btn_v = _hide_preview_and_save()
            mode = effective_mode(top, sub)
            if not audio_path or mode not in ("clone", "hifi"):
                return gr.update(), preview_state_v, preview_audio_v, save_btn_v, save_as_btn_v
            if edit_id:
                v = state.library.find_by_id(edit_id)
                if v and v.reference_audio:
                    voice_audio_abs = str(state.paths.root / v.reference_audio)
                    if os.path.normpath(audio_path) == os.path.normpath(voice_audio_abs):
                        return gr.update(), preview_state_v, preview_audio_v, save_btn_v, save_as_btn_v
            print(f"[ASR] transcribe begin: path={audio_path}", file=sys.stderr, flush=True)
            try:
                text = state.transcriber.transcribe(audio_path)
                print(f"[ASR] transcribe ok ({len(text)} chars)", file=sys.stderr, flush=True)
            except AsrUnavailable:
                text = i18n.t("status.asr_unavailable", "zh")
            except Exception as exc:
                # Surface the FULL traceback to the console so the underlying
                # call (which line/which library) is visible. The in-UI
                # textbox only shows a one-line summary.
                tb = traceback.format_exc()
                print(f"[ASR] FAILED:\n{tb}", file=sys.stderr, flush=True)
                text = (
                    f"[识别失败] {type(exc).__name__}: {exc}\n"
                    "完整 traceback 已打印到控制台 — 把那段输出贴回来定位。"
                )
            return text, preview_state_v, preview_audio_v, save_btn_v, save_as_btn_v

        ref_audio.change(
            _on_ref_audio_change,
            inputs=[ref_audio, top_mode_radio, sub_mode_radio, edit_voice_id],
            outputs=[prompt_box, preview_state, preview_audio_out, save_btn, save_as_btn],
        )

        # Generate-preview: runs the SDK with the appropriate kwargs per
        # effective_mode(top, sub), stores the resulting wav path in
        # `preview_state`, and reveals the Save button. Save uses the stored
        # preview path as the voice's audio.
        def _on_preview(top, sub, control, ref, prompt, seed_text,
                        cfg_value, inference_timesteps, edit_id):
            # Diagnostic: surface the edit_id we read so a future "save_as_btn
            # didn't appear" report can be triaged from the console alone.
            print(f"[Preview] edit_id={edit_id!r}", file=sys.stderr, flush=True)
            mode = effective_mode(top, sub)
            effective_control = control if mode in ("design", "clone") else ""
            transcript = (prompt or "").strip()
            # User rule: if seed_text is empty, fall back to transcript; if
            # transcript also empty, use a Chinese default.
            effective_seed = (
                (seed_text or "").strip()
                or transcript
                or "你好，这是一个用于固化音色的样本朗读。"
            )
            cfg_v = float(cfg_value)
            timesteps = int(inference_timesteps)
            err = lambda msg: (
                ("", ""),
                gr.update(value=None),     # preview_audio_out — always visible, just clear
                gr.update(visible=False),  # save_btn
                gr.update(visible=False),  # save_as_btn
                msg,
            )
            try:
                if mode == "design":
                    tmp_wav, voice_transcript = synthesize_voice_preview(
                        state.model, mode="design",
                        control=effective_control, seed_text=effective_seed,
                        cfg_value=cfg_v, inference_timesteps=timesteps,
                    )
                elif mode == "clone":
                    if not ref:
                        return err("❌ 请先上传参考音频")
                    tmp_wav, voice_transcript = synthesize_voice_preview(
                        state.model, mode="clone",
                        control=effective_control, seed_text=effective_seed,
                        upload_path=ref, transcript=transcript,
                        cfg_value=cfg_v, inference_timesteps=timesteps,
                    )
                elif mode == "hifi":
                    if not ref:
                        return err("❌ 请先上传参考音频")
                    if not transcript:
                        return err("❌ 参考音频转写为空（识别失败时请手动填写或重新上传）")
                    tmp_wav, voice_transcript = synthesize_voice_preview(
                        state.model, mode="hifi",
                        seed_text=effective_seed,
                        upload_path=ref, transcript=transcript,
                        cfg_value=cfg_v, inference_timesteps=timesteps,
                    )
                else:
                    return err(f"❌ 未知模式 {mode!r}")
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[Preview] FAILED:\n{tb}", file=sys.stderr, flush=True)
                return err(
                    f"❌ 预览生成失败 {type(exc).__name__}: {exc}\n"
                    "完整 traceback 已打印到控制台。"
                )

            return (
                (tmp_wav, voice_transcript),
                gr.update(value=tmp_wav),                      # preview_audio_out
                gr.update(visible=True),                       # save_btn
                gr.update(visible=bool(edit_id)),              # save_as_btn (only in edit mode)
                "✅ 预览已生成，点击保存以入库",
            )

        # voxcpm.generate is a blocking torch op with no cancel primitive,
        # so the button is disabled for the duration; users wait it out.
        # Three-stage chain:
        #   1. Disable button + flip label to a busy hint, set status message
        #   2. Run the actual preview generation
        #   3. Restore button regardless of success/failure (Gradio runs .then
        #      after the previous handler returns, error or not)
        def _enter_busy():
            # Clear any prior preview wav. The audio widget is always rendered
            # (visible=True at definition) so Gradio's per-component loading
            # overlay can attach to it during the SDK call.
            return (
                gr.update(interactive=False, value="⏳ 生成中...请稍候"),
                "⏳ 正在调用 SDK 生成预览，根据文本长度约需数秒到数十秒…",
                gr.update(value=None),
            )

        def _exit_busy():
            return gr.update(interactive=True, value=i18n.t("btn.preview", "zh"))

        preview_btn.click(
            _enter_busy,
            outputs=[preview_btn, lib_status, preview_audio_out],
        ).then(
            _on_preview,
            inputs=[top_mode_radio, sub_mode_radio, control_box,
                    ref_audio, prompt_box, seed_text_box,
                    lib_cfg_value_slider, lib_inference_timesteps_slider,
                    edit_voice_id],
            outputs=[preview_state, preview_audio_out, save_btn, save_as_btn, lib_status],
            show_progress="full",
        ).then(
            _exit_busy,
            outputs=preview_btn,
        )

        # ---- Save flow (create or update depending on edit_voice_id) ----
        def _save_impl(
            top, sub, edit_id, name, control, ref, denoise, normalize, seed_text, preview_pair,
            *, force_create: bool,
        ):
            """Shared body for save & save-as.

            force_create=True (save-as in edit mode) → always library.create,
            leaving the edited voice untouched.
            force_create=False (regular save) → library.update if editing, else
            library.create.
            """
            mode = effective_mode(top, sub)
            preview_path, transcript = preview_pair if preview_pair else ("", "")
            empty_form = _form_reset_outputs()
            if not preview_path or not os.path.exists(preview_path):
                table = _refresh_outputs(
                    state, ephemeral, status="❌ 请先点击 “生成预览” 再保存",
                )
                # Don't exit edit mode on this validation error — user may still
                # want to fix and re-preview.
                return (*table, gr.update(), gr.update(), gr.update(), *empty_form)
            effective_control = control if mode in ("design", "clone") else ""
            effective_denoise = bool(denoise) if mode in ("clone", "hifi") else False
            # Original upload only exists for clone/hifi; design has no upload.
            effective_ref = ref if mode in ("clone", "hifi") else None
            try:
                # seed_text is what the *generated* preview wav (saved as v.audio)
                # actually says, so it's also what the SDK needs as `prompt_text`
                # at script-generation time when it uses v.audio as prompt_wav_path.
                # See generation.build_generate_kwargs hifi-style branch.
                effective_seed = (seed_text or "").strip()
                if edit_id and not force_create:
                    state.library.update(
                        edit_id,
                        name=name,
                        mode=mode,
                        control=effective_control,
                        reference_audio_upload=effective_ref,
                        audio_upload=preview_path,
                        prompt_text=transcript,
                        seed_text=effective_seed,
                        denoise=effective_denoise,
                        normalize=bool(normalize),
                    )
                    status = f"✅ 已更新音色 `{name}`"
                else:
                    state.library.create(
                        name=name,
                        mode=mode,
                        control=effective_control,
                        reference_audio_upload=effective_ref,
                        audio_upload=preview_path,
                        prompt_text=transcript,
                        seed_text=effective_seed,
                        denoise=effective_denoise,
                        normalize=bool(normalize),
                    )
                    status = (
                        f"✅ 已另存为新音色 `{name}`" if (force_create and edit_id)
                        else f"✅ 已保存音色 `{name}`"
                    )
                table = _refresh_outputs(state, ephemeral, status=status)
                # Successful save → exit edit mode + clear form back to defaults.
                return (
                    *table,
                    "",                          # edit_voice_id
                    gr.update(visible=False),    # edit_banner_row
                    gr.update(visible=False),    # rename_btn
                    *empty_form,
                )
            except VoiceLibraryError as exc:
                table = _refresh_outputs(state, ephemeral, status=f"❌ {exc}")
                return (*table, gr.update(), gr.update(), gr.update(), *empty_form)
            finally:
                try:
                    os.unlink(preview_path)
                except OSError:
                    pass

        def _on_save(top, sub, edit_id, name, control, ref, denoise, normalize, seed_text, preview_pair):
            return _save_impl(top, sub, edit_id, name, control, ref, denoise, normalize,
                              seed_text, preview_pair, force_create=False)

        def _on_save_as(top, sub, edit_id, name, control, ref, denoise, normalize, seed_text, preview_pair):
            return _save_impl(top, sub, edit_id, name, control, ref, denoise, normalize,
                              seed_text, preview_pair, force_create=True)

        def _form_reset_outputs():
            """Return updates that reset the inline preview state without touching
            the form fields. Used when save fails — we want the user's inputs to
            stay so they can fix and retry."""
            return (
                ("", ""),                              # preview_state
                gr.update(value=None),                 # preview_audio_out
                gr.update(visible=False),              # save_btn
                gr.update(visible=False),              # save_as_btn
            )

        _SAVE_OUTPUTS = [
            voice_list, voice_picker, default_voice_dd, lib_status,
            edit_voice_id, edit_banner_row, rename_btn,
            preview_state, preview_audio_out, save_btn, save_as_btn,
        ]

        save_btn.click(
            _on_save,
            inputs=[top_mode_radio, sub_mode_radio, edit_voice_id, name_box, control_box,
                    ref_audio, denoise_box, normalize_box, seed_text_box, preview_state],
            outputs=_SAVE_OUTPUTS,
        )

        save_as_btn.click(
            _on_save_as,
            inputs=[top_mode_radio, sub_mode_radio, edit_voice_id, name_box, control_box,
                    ref_audio, denoise_box, normalize_box, seed_text_box, preview_state],
            outputs=_SAVE_OUTPUTS,
        )

        # ---- Row click dispatcher ----
        # Column 0/1 (name/mode) → enter edit mode; col 2 → play; col 3 → delete.
        # Delete relies on a JS guard (see voice_list.select.js below) that
        # rewrites the cell index to (-1, -1) when the user cancels confirm(),
        # so the Python handler treats it as a no-op.
        EDIT_COLS = (0, 1)
        LISTEN_COL = 2
        DELETE_COL = 3

        def _row_select_payload(
            *,
            table_voices=None,
            status=None,
            listen_path=None,
            listen_clear=False,
            edit_voice=None,
            current_edit_id="",
        ):
            """Build the full output tuple for voice_list.select.

            Each kwarg controls one slice; unspecified slices default to
            gr.update() (no change). Order matches `_ROW_SELECT_OUTPUTS`.

            ``listen_path`` (truthy str) → set audio source + autoplay.
            ``listen_clear`` (True)      → set audio to None (stops playback).
            Both default false → don't touch audio widget.

            ``current_edit_id``: pass-through value for the edit_voice_id State
            when ``edit_voice is None``. ``gr.update()`` is unreliable for
            ``gr.State`` (some Gradio versions reset it to the default), which
            would silently exit edit mode and hide save_as_btn the next time
            the user clicked Preview. Always returning the actual id keeps
            edit mode sticky across listen/delete/no-op row clicks.
            """
            playing_id = getattr(state, "_playing_voice_id", "")
            delete_pending_id = state._delete_pending.get("id", "")

            # Table + dropdowns
            if table_voices is None:
                v_list = gr.update()
                v_picker = gr.update()
                v_dd = gr.update()
            else:
                v_list = _voice_list_rows(
                    table_voices,
                    playing_id=playing_id,
                    delete_pending_id=delete_pending_id,
                )
                v_picker = gr.update(choices=[v.name for v in table_voices])
                v_dd = gr.update(
                    choices=voice_dropdown_choices(
                        table_voices, lang="zh", ephemeral=ephemeral,
                    )
                )

            status_v = "" if status is None else status

            # Listen widget — always rendered hidden; we only swap value.
            if listen_clear:
                listen_v = gr.update(value=None)
            elif listen_path is not None:
                listen_v = gr.update(value=listen_path)
            else:
                listen_v = gr.update()

            # Edit-mode bundle
            if edit_voice is None:
                edit_id_v = current_edit_id
                banner_row_v = gr.update()
                banner_md_v = gr.update()
                rename_v = gr.update()
                # form fields untouched
                name_v = gr.update()
                top_v = gr.update()
                sub_v = gr.update()
                ctrl_v = gr.update()
                ref_v = gr.update()
                prompt_v = gr.update()
                denoise_v = gr.update()
                norm_v = gr.update()
                seed_v = gr.update()
                preview_state_v = gr.update()
                preview_audio_v = gr.update()
                save_btn_v = gr.update()
                save_as_btn_v = gr.update()
            else:
                v = edit_voice
                edit_id_v = v.id
                banner_row_v = gr.update(visible=True)
                banner_md_v = gr.update(value=f"✏ 正在编辑：**{v.name}**")
                rename_v = gr.update(visible=True)
                # mode → top/sub radio values
                if v.mode == "design":
                    top_val, sub_val = "design", "controllable"
                elif v.mode == "clone":
                    top_val, sub_val = "cloning", "controllable"
                else:  # hifi
                    top_val, sub_val = "cloning", "ultimate"
                vis = field_visibility(v.mode)
                sub_visible = (top_val == "cloning")
                ref_path = (
                    str(state.paths.root / v.reference_audio)
                    if v.reference_audio
                    else None
                )
                name_v = gr.update(value=v.name)
                top_v = gr.update(value=top_val)
                sub_v = gr.update(value=sub_val, visible=sub_visible)
                ctrl_v = gr.update(value=v.control, visible=vis["control"])
                ref_v = gr.update(value=ref_path, visible=vis["reference_audio"])
                prompt_v = gr.update(value=v.prompt_text, visible=vis["prompt_text"])
                denoise_v = gr.update(value=v.denoise, visible=vis["denoise"])
                norm_v = gr.update(value=v.normalize)
                # Prefer v.seed_text (the original "样本朗读文本" the user typed
                # when creating this voice). Legacy voices saved before the
                # seed_text field was added fall back to prompt_text.
                seed_v = gr.update(
                    value=v.seed_text or v.prompt_text or "你好，这是一个用于固化音色的样本朗读。",
                    visible=vis["seed_text"],
                )
                # Entering edit mode invalidates any pending preview.
                preview_state_v = ("", "")
                preview_audio_v = gr.update(value=None)
                save_btn_v = gr.update(visible=False)
                save_as_btn_v = gr.update(visible=False)

            return (
                v_list, v_picker, v_dd, status_v, listen_v,
                edit_id_v, banner_row_v, banner_md_v, rename_v,
                name_v, top_v, sub_v, ctrl_v,
                ref_v, prompt_v, denoise_v, norm_v, seed_v,
                preview_state_v, preview_audio_v, save_btn_v, save_as_btn_v,
            )

        _ROW_SELECT_OUTPUTS = [
            voice_list, voice_picker, default_voice_dd, lib_status, listen_audio,
            edit_voice_id, edit_banner_row, edit_banner, rename_btn,
            name_box, top_mode_radio, sub_mode_radio, control_box,
            ref_audio, prompt_box, denoise_box, normalize_box, seed_text_box,
            preview_state, preview_audio_out, save_btn, save_as_btn,
        ]

        DELETE_CONFIRM_TIMEOUT = 8.0  # seconds the warning stays valid

        def _on_voice_row_select(current_edit_id: str, evt: gr.SelectData):
            # `current_edit_id`: pass-through value of the edit_voice_id State,
            # used so listen/delete/no-op branches don't accidentally clear
            # edit mode (which would hide save_as_btn on the next preview).
            import time
            if isinstance(evt.index, (list, tuple)):
                row_idx, col_idx = evt.index[0], evt.index[1]
            else:
                row_idx, col_idx = evt.index, 0
            voices = state.library.list_voices()
            if not (0 <= row_idx < len(voices)):
                return _row_select_payload(current_edit_id=current_edit_id)

            v = voices[row_idx]

            # Any non-delete interaction clears any pending delete confirmation.
            if col_idx != DELETE_COL:
                state._delete_pending = {"id": "", "t": 0.0}

            if col_idx == DELETE_COL:
                pending = state._delete_pending
                now = time.time()
                same_row = (pending["id"] == v.id)
                fresh = (now - pending["t"] < DELETE_CONFIRM_TIMEOUT)
                if same_row and fresh:
                    # Confirmed.
                    state.library.delete(v.id)
                    state._delete_pending = {"id": "", "t": 0.0}
                    listen_clear = (state._playing_voice_id == v.id)
                    if listen_clear:
                        state._playing_voice_id = ""
                    # If we just deleted the voice the user was editing, exit
                    # edit mode; otherwise preserve the current edit.
                    surviving_edit_id = (
                        "" if current_edit_id == v.id else current_edit_id
                    )
                    voices = state.library.list_voices()
                    return _row_select_payload(
                        table_voices=voices,
                        listen_clear=listen_clear,
                        status=f"✅ 已删除音色 `{v.name}`",
                        current_edit_id=surviving_edit_id,
                    )
                # First click (or stale / different row) — record + warn.
                # Three signals: (1) toast (gr.Warning) so the user sees a
                # popup even if their eyes are on the table; (2) table cell
                # flips to "⚠ 再点确认" via delete_pending_id readback in
                # _row_select_payload; (3) status line below the table.
                state._delete_pending = {"id": v.id, "t": now}
                gr.Warning(
                    f"再次点击 `{v.name}` 这一行的 🗑 即可删除"
                    f"（{int(DELETE_CONFIRM_TIMEOUT)} 秒内有效）"
                )
                return _row_select_payload(
                    table_voices=voices,
                    status=(
                        f"⚠ 再次点击 `{v.name}` 这一行的 🗑 即可删除"
                        f"（{int(DELETE_CONFIRM_TIMEOUT)} 秒内有效）"
                    ),
                    current_edit_id=current_edit_id,
                )
            if col_idx == LISTEN_COL:
                # Toggle: if this row is the currently-playing one, stop;
                # otherwise switch to playing this row.
                if state._playing_voice_id == v.id:
                    state._playing_voice_id = ""
                    return _row_select_payload(
                        table_voices=voices,
                        listen_clear=True,
                        status=f"⏹ 已停止：**{v.name}**",
                        current_edit_id=current_edit_id,
                    )
                if not v.audio:
                    return _row_select_payload(
                        status=f"❌ 该音色没有可播放的音频：`{v.name}`",
                        current_edit_id=current_edit_id,
                    )
                state._playing_voice_id = v.id
                state._pending_listen_path = str(state.paths.root / v.audio)
                # Phase 1 clears the audio widget; the chained .then() handler
                # (_apply_pending_listen) sets it to the path in phase 2 so the
                # browser sees a real value change and autoplay re-fires even
                # when the user replays the same row.
                return _row_select_payload(
                    table_voices=voices,
                    listen_clear=True,
                    status=f"🔊 试听：**{v.name}**",
                    current_edit_id=current_edit_id,
                )
            if col_idx in EDIT_COLS:
                return _row_select_payload(
                    edit_voice=v,
                    status=f"✏ 正在编辑：**{v.name}**",
                )
            return _row_select_payload(current_edit_id=current_edit_id)

        # Two-click confirm pattern (see _on_voice_row_select above): a JS
        # `confirm()` guard would have been simpler, but Gradio's `js=` for
        # select events does not receive the SelectData reliably, so we
        # implement confirmation in Python via state._delete_pending.
        def _apply_pending_listen():
            """Phase 2 of the row-click chain — push the queued listen path into
            the audio widget so it's a fresh value change (forcing autoplay)."""
            pending = getattr(state, "_pending_listen_path", "")
            state._pending_listen_path = ""
            if pending:
                return gr.update(value=pending)
            return gr.update()

        voice_list.select(
            _on_voice_row_select,
            inputs=[edit_voice_id],
            outputs=_ROW_SELECT_OUTPUTS,
        ).then(
            _apply_pending_listen,
            outputs=listen_audio,
        )

        # ---- Cancel-edit ----
        def _on_cancel_edit_explicit():
            # When cancelling, we DO want to reset the form to defaults — so
            # rather than leaving fields as-is (the default for edit_voice=None),
            # we send explicit resets.
            return (
                gr.update(),                            # voice_list
                gr.update(),                            # voice_picker
                gr.update(),                            # default_voice_dd
                "",                                     # lib_status
                gr.update(),                            # listen_audio
                "",                                     # edit_voice_id
                gr.update(visible=False),               # edit_banner_row
                gr.update(value=""),                    # edit_banner (md)
                gr.update(visible=False),               # rename_btn
                gr.update(value=""),                    # name_box
                gr.update(value="design"),              # top_mode_radio
                gr.update(value="controllable",
                          visible=False),               # sub_mode_radio
                gr.update(value="", visible=True),      # control_box
                gr.update(value=None, visible=False),   # ref_audio
                gr.update(value="", visible=False),     # prompt_box
                gr.update(value=False, visible=False),  # denoise_box
                gr.update(value=False),                 # normalize_box
                gr.update(
                    value="你好，这是一个用于固化音色的样本朗读。",
                    visible=True,
                ),                                       # seed_text_box
                ("", ""),                                # preview_state
                gr.update(value=None),                   # preview_audio_out
                gr.update(visible=False),                # save_btn
                gr.update(visible=False),                # save_as_btn
            )

        cancel_edit_btn.click(
            _on_cancel_edit_explicit,
            outputs=_ROW_SELECT_OUTPUTS,
        )

        # ---- Listen auto-revert ----
        # When the underlying <audio> element fires its `ended` event (natural
        # end of playback), Gradio surfaces it as the .stop event. We reset
        # the playback state and rebuild the table so the cell reverts from
        # ⏹ 停止 back to ▶ 播放. If the SDK version doesn't emit .stop, the
        # cell stays as ⏹ until the user clicks it (acceptable degradation).
        def _on_listen_stopped():
            # Reset the audio source on natural end too — otherwise the next
            # ▶ click on the same row pushes the same filepath, which Gradio
            # treats as "no change" and the browser never re-fires autoplay.
            # Preserve any in-flight delete-confirmation indicator so a row
            # finishing playback doesn't erase the "⚠ 再点确认" cell.
            state._playing_voice_id = ""
            voices = state.library.list_voices()
            delete_pending_id = state._delete_pending.get("id", "")
            return (
                _voice_list_rows(
                    voices, playing_id="", delete_pending_id=delete_pending_id,
                ),
                gr.update(value=None),
            )

        try:
            listen_audio.stop(_on_listen_stopped, outputs=[voice_list, listen_audio])
        except AttributeError:
            # Older Gradio versions don't expose `.stop` on Audio.
            pass

        # ---- Rename-only shortcut (edit mode only) ----
        def _on_rename_only(edit_id, new_name):
            if not edit_id:
                # Defensive: button is hidden when not editing, but guard anyway.
                table = _refresh_outputs(state, ephemeral, status="❌ 不在编辑模式")
                return (*table, gr.update())
            try:
                state.library.update(edit_id, name=new_name)
                v = state.library.find_by_id(edit_id)
                status = f"✅ 已重命名为 `{v.name}`" if v else "✅ 已重命名"
                table = _refresh_outputs(state, ephemeral, status=status)
                # Refresh the banner so it shows the new name too.
                banner = (
                    gr.update(value=f"✏ 正在编辑：**{v.name}**")
                    if v
                    else gr.update()
                )
                return (*table, banner)
            except VoiceLibraryError as exc:
                table = _refresh_outputs(state, ephemeral, status=f"❌ {exc}")
                return (*table, gr.update())

        rename_btn.click(
            _on_rename_only,
            inputs=[edit_voice_id, name_box],
            outputs=[voice_list, voice_picker, default_voice_dd, lib_status, edit_banner],
        )

        # ---- Refresh button ----
        refresh_btn.click(
            lambda: _refresh_outputs(state, ephemeral),
            outputs=[voice_list, voice_picker, default_voice_dd, lib_status],
        )

        # Initial values for the voice list + dropdowns are baked into the UI
        # config at build_ui time. After a save, the table is updated via
        # callback, but a browser refresh re-downloads the config snapshot and
        # would show the startup state. Re-populate from state.library on every
        # page load so refresh stays in sync with disk. Also reset the
        # ▶/⏹ playback state — a fresh tab has no audio playing.
        def _on_page_load():
            state._playing_voice_id = ""
            return _refresh_outputs(state, ephemeral, status="")

        demo.load(
            _on_page_load,
            outputs=[voice_list, voice_picker, default_voice_dd, lib_status],
        )

    return demo


def _voice_list_rows(
    voices, playing_id: str = "", delete_pending_id: str = ""
) -> list[list[str]]:
    """Build the 4-column dataframe rows: name / mode / play / delete.

    ``playing_id`` controls per-row toggle: the row whose voice id matches
    is rendered with ⏹ 停止; all others with ▶ 播放.

    ``delete_pending_id`` controls the 🗑 column: a row whose voice is
    awaiting delete-confirmation shows ⚠ 再点确认, so the user gets immediate
    in-table feedback instead of having to read the status line below.
    """
    return [
        [
            v.name,
            v.mode,
            "⏹ 停止" if v.id == playing_id else "▶ 播放",
            "⚠ 再点确认" if v.id == delete_pending_id else "🗑 删除",
        ]
        for v in voices
    ]


def _refresh_outputs(state: AppState, ephemeral, status: str = "✅ updated") -> tuple:
    voices = state.library.list_voices()
    playing_id = getattr(state, "_playing_voice_id", "")
    delete_pending_id = state._delete_pending.get("id", "")
    return (
        _voice_list_rows(
            voices, playing_id=playing_id, delete_pending_id=delete_pending_id,
        ),
        gr.update(choices=[v.name for v in voices]),
        gr.update(choices=voice_dropdown_choices(voices, lang="zh", ephemeral=ephemeral)),
        status,
    )


def _usage_doc(lang: str) -> str:
    if lang == "en":
        return (
            "## Usage\n\n"
            "### Script syntax → SDK call mapping\n\n"
            "Per the official [VoxCPM 2 usage guide]"
            "(https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html), "
            "each segment is routed to one of two SDK call shapes based on script syntax:\n\n"
            "| Script | Mode | SDK call |\n"
            "|---|---|---|\n"
            "| `<voice>(control)text` | Controllable Cloning | `generate(text=\"(control)text\", reference_wav_path=voice.audio)` |\n"
            "| `<voice>text` *(voice has audio + transcript)* | High-Fidelity Cloning | `generate(text=\"text\", prompt_wav_path=..., prompt_text=..., reference_wav_path=...)` |\n"
            "| (no `<voice>` switch and no real voices) | Voice Design | `generate(text=\"(control)text\")` (no audio) |\n\n"
            "### Other rules\n\n"
            "- `<voice name>` switches voice for the **rest of the current line**; a newline resets to the default voice.\n"
            "- Square-bracket tags like `[laughing]` are passed through to the model verbatim and are never split across chunks.\n"
            "- Long text is split on sentence terminators (`。！？.!?;…`), with comma boundaries (`，、,`) as fallback when a sentence exceeds the per-chunk budget.\n"
            "- Voice names are matched case-insensitively after trimming whitespace.\n"
            "- A Voice Design save synthesises one audio sample on the fly to fix the voice character; that sample is reused for all subsequent generations.\n"
        )
    return (
        "## 使用说明\n\n"
        "### 脚本语法 → SDK 调用映射\n\n"
        "依据 [VoxCPM 2 官方使用指南]"
        "(https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html)，"
        "每一段会按脚本语法路由到下列两种 SDK 调用之一：\n\n"
        "| 脚本写法 | 模式 | SDK 调用 |\n"
        "|---|---|---|\n"
        "| `<音色>(控制指令)文本` | 可控声音克隆 | `generate(text=\"(控制指令)文本\", reference_wav_path=音色音频)` |\n"
        "| `<音色>文本`（该音色有音频 + 转写） | 高保真克隆 | `generate(text=\"文本\", prompt_wav_path=..., prompt_text=..., reference_wav_path=...)` |\n"
        "| 无 `<音色>` 切换且无任何音色 | 声音设定 | `generate(text=\"(控制)文本\")`（无音频） |\n\n"
        "### 其他规则\n\n"
        "- `<音色名>` 切换该**行后续文本**的音色，新行自动复位到默认音色。\n"
        "- 方括号标签如 `[laughing]` 原样传给模型，不会被切分到不同 chunk。\n"
        "- 长文本按句末标点切分（`。！？.!?;…`），单句超长时退而按逗号 `，、,` 切。\n"
        "- 音色名匹配是 trim + 大小写不敏感。\n"
        "- 声音设定保存时会先调一次 SDK 生成样本固化音色，之后所有生成都复用该样本。\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VoxCPM TTS Tool")
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--root", default=".", help="Project root (where pretrained_models/ etc. live)")
    args = parser.parse_args(argv)

    state, messages = initialize(Path(args.root).resolve())
    for m in messages:
        print(m)

    demo = build_ui(state, messages)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
