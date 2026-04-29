"""Generation kwargs builder + end-to-end orchestration.

`build_generate_kwargs` maps a (Voice, chunk, runtime flags) to the dict
passed to `voxcpm.VoxCPM.generate`. `run_generation` streams Progress events
per chunk and a final Result. See spec §Generation Flow for the rules.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

import numpy as np

from .long_text import concat_waveforms, split_for_generation
from .script_parser import localize_non_lang_tags, parse_script
from .voice_library import Voice


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


def _voice_for_segment(
    name: str,
    *,
    default_voice: Voice,
    library,
) -> Voice:
    """Resolve a parser-emitted voice name to a Voice.

    Parser invariant: ``name`` is either ``default_voice.name`` or the
    canonical name of a library voice (case-canonicalized). Default-voice
    name match short-circuits library lookup so an ephemeral default
    (id starting with ``__``, never in the library) still resolves.

    Raises ValueError if the name resolves to nothing — unreachable given
    the parser invariant, but defensive so a typo upstream doesn't silently
    fall through to a no-voice generation.
    """
    if name == default_voice.name:
        return default_voice
    v = library.find_by_name(name)
    if v is None:
        raise ValueError(f"voice not found: {name!r}")
    return v


class _GeneratesAudio(Protocol):
    sample_rate: int

    def generate(self, **kwargs) -> np.ndarray: ...


def _absolute_audio_path(audio_root: str, rel: str) -> str:
    return os.path.normpath(os.path.join(audio_root, rel))


def build_generate_kwargs(
    voice: Voice,
    chunk: str,
    *,
    zipenhancer_loaded: bool,
    audio_root: str,
    script_control: str | None = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
) -> dict:
    """Return the kwargs dict passed to voxcpm.VoxCPM.generate(...).

    Routing rules (per user request — script syntax dictates clone vs hifi):

      1. `script_control is not None` (script wrote `<voice>(control)text`):
         → CLONE-style: text = "(control)chunk", reference_wav_path. The script
           control wins over voice.control. Empty string control means "use the
           voice as-is in clone mode" (text = chunk, no prefix).
      2. `script_control is None` AND voice has audio AND voice has prompt_text:
         → HIFI-style: prompt_wav_path + prompt_text + reference_wav_path.
      3. `script_control is None` AND voice has audio only:
         → CLONE-style with voice.control as the prefix (legacy).
      4. Voice has no audio (legacy design):
         → text-only, with voice.control as prefix if set (legacy).

    Per-call tuning knobs (cfg_value, inference_timesteps) are included only
    when they differ from SDK defaults. `normalize` and `denoise` are
    properties of the voice (see spec §Voice Library); `denoise` is still
    gated by `zipenhancer_loaded`.
    """
    extras: dict = {}
    if cfg_value != 2.0:
        extras["cfg_value"] = cfg_value
    if voice.normalize:
        extras["normalize"] = True
    if inference_timesteps != 10:
        extras["inference_timesteps"] = inference_timesteps

    # Prefer the generated preview (`voice.audio`); fall back to legacy
    # `voice.reference_audio` field for voices saved before the schema split.
    active_audio = getattr(voice, "audio", "") or voice.reference_audio
    has_audio = bool(active_audio)

    # Branch 4: voice has no audio (legacy design path).
    if not has_audio:
        # Script control still takes precedence if set.
        effective = script_control if script_control is not None else voice.control
        text = f"({effective}){chunk}" if effective else chunk
        return {"text": text, **extras}

    abs_audio = _absolute_audio_path(audio_root, active_audio)
    denoise = bool(voice.denoise) and zipenhancer_loaded

    # Branch 1: script provided `(control)` — clone mode, script wins.
    if script_control is not None:
        text = f"({script_control}){chunk}" if script_control else chunk
        return {
            "text": text,
            "reference_wav_path": abs_audio,
            "denoise": denoise,
            **extras,
        }

    # Branch 2: hifi (voice has audio + transcript, no script control).
    # `prompt_wav_path` here is `voice.audio` (the GENERATED preview wav, which
    # actually says `voice.seed_text`), so the SDK's prompt_text must match
    # seed_text — not the original upload's ASR transcript stored in
    # `voice.prompt_text`. Fall back to prompt_text for legacy voices saved
    # before the seed_text field existed.
    hifi_prompt_text = getattr(voice, "seed_text", "") or voice.prompt_text
    if hifi_prompt_text:
        return {
            "text": chunk,
            "prompt_wav_path": abs_audio,
            "prompt_text": hifi_prompt_text,
            "reference_wav_path": abs_audio,
            "denoise": denoise,
            **extras,
        }

    # Branch 3: clone with voice's stored control (legacy clone mode).
    text = f"({voice.control}){chunk}" if voice.control else chunk
    return {
        "text": text,
        "reference_wav_path": abs_audio,
        "denoise": denoise,
        **extras,
    }


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
    Warnings from ``parse_script`` (unknown ``<voice>`` tags) are intentionally
    swallowed here; the handler invokes ``parse_script`` separately to surface
    them in UI before locking the inputs.
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
        voice = _voice_for_segment(
            seg.voice_name, default_voice=default_voice, library=library,
        )
        for chunk in split_for_generation(seg.text, char_budget=char_budget):
            plan.append((voice, chunk, seg.control))
    total = len(plan)

    if total == 0:
        yield Result(wav=concat_waveforms([]), sample_rate=sample_rate)
        return

    all_wavs: list[np.ndarray] = []
    for i, (voice, chunk, ctrl) in enumerate(plan, start=1):
        if stop_flag():
            yield Result(
                wav=concat_waveforms(all_wavs),
                sample_rate=sample_rate,
                was_stopped=True,
            )
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


def synthesize_voice_preview(
    model: _GeneratesAudio,
    *,
    mode: str,
    control: str = "",
    seed_text: str = "",
    upload_path: str | None = None,
    transcript: str = "",
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
) -> tuple[str, str]:
    """Generate a preview wav for a voice in the given creation mode.

    Returns ``(tmp_wav_path, transcript_for_voice)`` where ``transcript_for_voice``
    is the text the caller should persist as the voice's ``prompt_text``.

    Per `VoxCPM 2 usage guide`_, the SDK call shape per mode is:

      - ``design``: ``generate(text="(control)seed_text")``. No upload, no
        transcript. The preview "says" the seed text in the described style.
        ``transcript_for_voice`` is set to the seed text so the saved voice can
        be addressed via ``<voice>text`` (hifi-style) afterwards.
      - ``clone``: ``generate(text="(control)transcript", reference_wav_path=upload)``.
        The transcript is provided by the caller (typically pre-filled by ASR
        on upload). The preview "says" that transcript in the cloned voice +
        control style. ``transcript_for_voice`` echoes the input transcript.
      - ``hifi``: ``generate(text=seed_text, prompt_wav_path=upload,
        prompt_text=transcript, reference_wav_path=upload)``. ``seed_text``
        is the verification phrase the user wants to hear; ``transcript`` is
        the recorded reference's actual content. Default seed if blank.

    The waveform is written to a tempfile; the caller must unlink it after
    staging into the library or rendering to the UI.

    .. _VoxCPM 2 usage guide:
       https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html
    """
    import soundfile as sf

    if not seed_text:
        raise ValueError("seed_text is required (caller may auto-fill from transcript or default)")

    # Only forward cfg_value / inference_timesteps when they differ from SDK
    # defaults so older SDK builds without these kwargs aren't broken.
    extras: dict = {}
    if cfg_value != 2.0:
        extras["cfg_value"] = cfg_value
    if inference_timesteps != 10:
        extras["inference_timesteps"] = inference_timesteps

    if mode == "design":
        text = f"({control}){seed_text}" if control else seed_text
        wav = model.generate(text=text, **extras)
        transcript_for_voice = seed_text
    elif mode == "clone":
        if not upload_path:
            raise ValueError("clone preview requires upload_path")
        # NOTE: per the user's UX rules, clone now speaks `seed_text` (not the
        # ASR transcript). The transcript is still kept for voice storage so
        # that `<voice>text` (hifi-style script switch) works on the saved voice.
        text = f"({control}){seed_text}" if control else seed_text
        wav = model.generate(text=text, reference_wav_path=upload_path, **extras)
        transcript_for_voice = transcript or seed_text
    elif mode == "hifi":
        if not upload_path:
            raise ValueError("hifi preview requires upload_path")
        if not transcript:
            raise ValueError("hifi preview requires transcript (caller should ASR the upload first)")
        wav = model.generate(
            text=seed_text,
            prompt_wav_path=upload_path,
            prompt_text=transcript,
            reference_wav_path=upload_path,
            **extras,
        )
        transcript_for_voice = transcript
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    sr = int(getattr(model, "sample_rate", 16000))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, wav, samplerate=sr, subtype="PCM_16")
    return tmp.name, transcript_for_voice
