"""Voice metadata persistence with atomic JSON writes and audio file lifecycle.

See spec §Voice Library for field rules, mode validation, and persistence rules.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Mode = Literal["design", "clone", "hifi"]


class VoiceLibraryError(ValueError):
    """Validation or persistence error in the voice library."""


@dataclass
class Voice:
    id: str
    name: str
    mode: Mode
    control: str = ""
    # reference_audio: ORIGINAL user upload kept for record (clone/hifi only).
    # audio: the GENERATED preview wav — this is what every SDK call at
    # script-generation time actually uses (see generation.build_generate_kwargs).
    reference_audio: str = ""  # rel path "voices/audio/<id>.original.wav" or ""
    audio: str = ""            # rel path "voices/audio/<id>.wav" or ""
    # prompt_text: transcript of the ORIGINAL upload (reference_audio). For
    #   hifi this is the ASR result captured at preview time; needed when
    #   re-running the hifi preview in edit mode (SDK requires prompt_text
    #   matching prompt_wav_path = the original upload).
    # seed_text: the phrase the user typed for "样本朗读"; this is what the
    #   GENERATED preview wav (`audio`) actually says. At script-generation
    #   time the hifi-style branch uses this as `prompt_text` because
    #   `prompt_wav_path` there is `audio`, not the original upload.
    prompt_text: str = ""
    seed_text: str = ""
    denoise: bool = False
    normalize: bool = False  # per-voice text normalization preference (all modes)
    created_at: str = ""
    updated_at: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_name(name: str) -> str:
    return name.strip().lower()


class VoiceLibrary:
    """File-backed voice store rooted at <voices_dir>/voices.json.

    Audio files live under <voices_dir>/audio/<id>.wav.
    """

    def __init__(self, voices_dir: Path):
        self.dir = Path(voices_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.dir / "voices.json"
        self._voices: list[Voice] = self._load()

    # ---- public API ----

    def list_voices(self) -> list[Voice]:
        return list(self._voices)

    def find_by_id(self, voice_id: str) -> Voice | None:
        return next((v for v in self._voices if v.id == voice_id), None)

    def find_by_name(self, name: str) -> Voice | None:
        norm = _normalize_name(name)
        return next((v for v in self._voices if _normalize_name(v.name) == norm), None)

    def create(
        self,
        *,
        name: str,
        mode: Mode,
        control: str = "",
        reference_audio_upload: str | None = None,
        audio_upload: str | None = None,
        prompt_text: str = "",
        seed_text: str = "",
        denoise: bool = False,
        normalize: bool = False,
    ) -> Voice:
        """Create a voice.

        `reference_audio_upload`: original user upload (clone/hifi only).
            Stored at ``voices/audio/<id>.original.wav`` if present. Optional;
            design voices have no upload.
        `audio_upload`: the generated preview wav from synthesize_voice_preview.
            Stored at ``voices/audio/<id>.wav``. This is the file the rest of
            the system uses at script-generation time.
        """
        self._validate_name(name)
        if self.find_by_name(name) is not None:
            raise VoiceLibraryError(f"duplicate name: {name!r}")
        voice_id = uuid.uuid4().hex
        rel_audio = self._stage_audio(audio_upload, voice_id, suffix=".wav")
        rel_ref = self._stage_audio(reference_audio_upload, voice_id, suffix=".original.wav")
        v = Voice(
            id=voice_id,
            name=name.strip(),
            mode=mode,
            control=control if mode != "hifi" else "",
            reference_audio=rel_ref,
            audio=rel_audio,
            # All modes may store prompt_text now: hifi uses it as the recorded
            # transcript; design uses the seed text from auto-generation; clone
            # stores the upload's ASR result (or seed_text fallback).
            prompt_text=prompt_text or "",
            seed_text=seed_text or "",
            denoise=bool(denoise) if mode != "design" else False,
            normalize=bool(normalize),
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self._validate_mode_invariants(v)
        self._voices.append(v)
        self._save()
        return v

    def update(
        self,
        voice_id: str,
        *,
        name: str | None = None,
        mode: Mode | None = None,
        control: str | None = None,
        reference_audio_upload: str | None = None,
        audio_upload: str | None = None,
        prompt_text: str | None = None,
        seed_text: str | None = None,
        denoise: bool | None = None,
        normalize: bool | None = None,
    ) -> str:
        """Update voice in place. Returns a human-readable warning string (may be empty).

        ``audio_upload`` overwrites the generated-preview wav (`v.audio`) at the
        same id-keyed path used by ``create``; ``reference_audio_upload`` does
        the same for the original-upload wav. Both are optional; pass only what
        the caller wants to replace.
        """
        v = self.find_by_id(voice_id)
        if v is None:
            raise VoiceLibraryError(f"unknown voice id: {voice_id}")
        warnings: list[str] = []

        if name is not None and name.strip() != v.name:
            self._validate_name(name)
            other = self.find_by_name(name)
            if other is not None and other.id != v.id:
                raise VoiceLibraryError(f"duplicate name: {name!r}")
            old = v.name
            v.name = name.strip()
            warnings.append(
                f"rename: scripts containing <{old}> will no longer match this voice"
            )

        if mode is not None:
            v.mode = mode
        if control is not None:
            v.control = control
        if prompt_text is not None:
            v.prompt_text = prompt_text
        if seed_text is not None:
            v.seed_text = seed_text
        if denoise is not None:
            v.denoise = bool(denoise)
        if normalize is not None:
            v.normalize = bool(normalize)

        if reference_audio_upload is not None:
            v.reference_audio = self._stage_audio(
                reference_audio_upload, v.id, suffix=".original.wav"
            )
        if audio_upload is not None:
            v.audio = self._stage_audio(audio_upload, v.id, suffix=".wav")

        # Apply mode-specific clearing rules.
        # Design no longer wipes audio/prompt_text — those are populated by the
        # auto-generation flow at save time (see app.py _on_save).
        if v.mode == "design":
            v.denoise = False
        elif v.mode == "hifi":
            v.control = ""

        self._validate_mode_invariants(v)
        v.updated_at = _now_iso()
        self._save()
        return " | ".join(warnings)

    def delete(self, voice_id: str) -> None:
        v = self.find_by_id(voice_id)
        if v is None:
            return
        self._remove_audio(v.id)
        self._voices = [x for x in self._voices if x.id != voice_id]
        self._save()

    # ---- internals ----

    def _validate_name(self, name: str) -> None:
        stripped = name.strip()
        if not stripped:
            raise VoiceLibraryError("name must be non-empty")
        if "<" in stripped or ">" in stripped:
            raise VoiceLibraryError("name must not contain < or >")

    def _validate_mode_invariants(self, v: Voice) -> None:
        # The "active audio" (v.audio = generated preview) is what the rest of
        # the system uses, so all modes after a successful preview-then-save
        # must have it populated. The original-upload `reference_audio` is
        # only required for clone/hifi (design has no upload).
        if not v.audio:
            raise VoiceLibraryError(f"{v.mode} voice requires generated audio (preview)")
        if v.mode == "design":
            if v.denoise:
                raise VoiceLibraryError("design voice must not set denoise=true")
        elif v.mode == "clone":
            if not v.reference_audio:
                raise VoiceLibraryError("clone voice requires reference_audio (original upload)")
        elif v.mode == "hifi":
            if not v.reference_audio:
                raise VoiceLibraryError("hifi voice requires reference_audio (original upload)")
            if not v.prompt_text:
                raise VoiceLibraryError("hifi voice requires non-empty prompt_text")

    def _stage_audio(
        self, upload_path: str | None, voice_id: str, *, suffix: str = ".wav"
    ) -> str:
        """Copy the source wav into the voices audio dir under <id><suffix>.

        Returns the relative path stored in the Voice (or "" if no upload)."""
        if upload_path is None:
            return ""
        if not upload_path.lower().endswith(".wav"):
            raise VoiceLibraryError("only .wav uploads are supported in v1")
        dst = self.audio_dir / f"{voice_id}{suffix}"
        shutil.copyfile(upload_path, dst)
        return f"voices/audio/{voice_id}{suffix}"

    def _remove_audio(self, voice_id: str) -> None:
        # Remove both the active audio and the original upload kept on record.
        for suffix in (".wav", ".original.wav"):
            path = self.audio_dir / f"{voice_id}{suffix}"
            if path.exists():
                path.unlink()

    def _load(self) -> list[Voice]:
        if not self.json_path.exists():
            return []
        try:
            raw = json.loads(self.json_path.read_text("utf-8"))
        except json.JSONDecodeError:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = self.dir / f"voices.broken-{stamp}.json"
            self.json_path.rename(backup)
            return []
        # Filter unknown keys for forward-compat; older records without newer fields
        # (e.g. `normalize`) get dataclass defaults.
        allowed = set(Voice.__dataclass_fields__)
        return [Voice(**{k: v for k, v in rec.items() if k in allowed}) for rec in raw]

    def _save(self) -> None:
        tmp = self.json_path.with_suffix(".json.tmp")
        data = [asdict(v) for v in self._voices]
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.json_path)
