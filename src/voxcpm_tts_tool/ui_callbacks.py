"""UI-pure callback helpers extracted so they can be unit-tested without Gradio.

Spec refs: §Mode-specific voice editing UI (visibility), §UI Design insertion
helpers, §Error Handling no-voices ephemeral default.
"""
from __future__ import annotations

from typing import Iterable

from . import i18n
from .app_state import EPHEMERAL_DEFAULT_VOICE_ID
from .voice_library import Voice


_VISIBILITY_BY_MODE: dict[str, dict[str, bool]] = {
    "design": {
        "voice_name": True, "control": True,
        "reference_audio": False, "prompt_text": False,
        "denoise": False, "normalize": True,
        "seed_text": True,
    },
    "clone": {
        # Controllable Cloning: shows control box. seed_text is the text
        # that will be spoken during preview (auto-fills from transcript if blank).
        "voice_name": True, "control": True,
        "reference_audio": True, "prompt_text": True,
        "denoise": True, "normalize": True,
        "seed_text": True,
    },
    "hifi": {
        # Ultimate Cloning: hides control box (control instructions are ignored
        # by the SDK in hifi mode per VoxCPM2 guide).
        "voice_name": True, "control": False,
        "reference_audio": True, "prompt_text": True,
        "denoise": True, "normalize": True,
        "seed_text": True,
    },
}


def effective_mode(top: str, sub: str) -> str:
    """Map two-level UI radio (top, sub) to the internal storage mode.

    top: "design" | "cloning"
    sub: "controllable" | "ultimate"  (only used when top == "cloning")
    Returns one of "design" | "clone" | "hifi" — the mode value stored on
    the Voice and consumed by the rest of the system.
    """
    if top == "design":
        return "design"
    if top == "cloning":
        return "hifi" if sub == "ultimate" else "clone"
    raise ValueError(f"unknown top mode: {top!r}")


def field_visibility(mode: str) -> dict[str, bool]:
    return dict(_VISIBILITY_BY_MODE[mode])


def voice_dropdown_choices(
    voices: Iterable[Voice], *, lang: str, ephemeral: Voice
) -> list[tuple[str, str]]:
    voices = list(voices)
    if not voices:
        return [(i18n.t("voice.default", lang), ephemeral.id)]
    return [(v.name, v.id) for v in voices if v.id != EPHEMERAL_DEFAULT_VOICE_ID]


def insert_voice_tag(current_script: str, *, voice_name: str) -> str:
    return f"{current_script}<{voice_name}>"


def insert_tag(current_script: str, *, tag: str) -> str:
    return f"{current_script}[{tag}]"
