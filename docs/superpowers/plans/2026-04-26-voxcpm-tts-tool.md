# VoxCPM TTS Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Gradio web tool that wraps the published `voxcpm` SDK, manages reusable voices (design / clone / hifi), parses script-style `<voice>` switches, splits long text, optionally transcribes/denoises reference audio, and saves downloadable WAV output.

**Architecture:** Pure-Python `src/` layout. Domain logic (voice library, parser, splitter, resolver, generation kwargs) lives in small focused modules with no Gradio/torch coupling and is fully unit-tested with fakes. `app.py` is a thin Gradio 5 wiring layer over those modules. Models are downloaded at startup into `pretrained_models/` and a single `voxcpm.VoxCPM` instance is reused for the process lifetime.

**Tech Stack:** Python 3.10–3.12 (3.13 currently has wheel issues with `editdistance`, a transitive dep of `voxcpm`), `voxcpm`, `gradio>=5,<6`, `numpy`, `soundfile`, `huggingface_hub`, `modelscope`, `funasr`, `pytest`.

**Source spec:** `docs/superpowers/specs/2026-04-26-voxcpm-tts-tool-design.md` — every behavior in this plan traces back to a section there. Read it before starting.

---

## File Structure

Production code (`src/voxcpm_tts_tool/`):

- `__init__.py` — package marker + `__version__`.
- `i18n.py` — translation dictionary, `t(key, lang)` lookup with fallback.
- `voice_library.py` — voice CRUD, atomic JSON write, audio file lifecycle, validation.
- `script_parser.py` — line-based `<voice>` tokenization, returns `[(line_no, voice_name_or_None, text), ...]` plus warnings.
- `long_text.py` — punctuation-aware splitter for one parsed segment, plus `concat_waveforms`.
- `model_resolver.py` — three resolvers (VoxCPM2 / SenseVoiceSmall / ZipEnhancer), each returns local path or raises.
- `transcription.py` — thin SenseVoice wrapper (lazy load, takes wav path → returns string).
- `generation.py` — `build_kwargs(voice, chunk, zipenhancer_loaded)` + `run_generation(...)` orchestration.
- `output_writer.py` — `outputs/YYYYMMDD-HHMMSS-mmm.wav` filename + soundfile write.
- `app_state.py` — dataclass holding resolved paths, model singleton, library, recognizer, ZipEnhancer flag.

Entry point:

- `app.py` — argparse CLI + Gradio `Blocks` UI with three tabs.

Tests (`tests/`):

- `conftest.py` — `FakeVoxCPM`, `FakeRecognizer`, tmp-path fixtures.
- `test_i18n.py`, `test_voice_library.py`, `test_script_parser.py`, `test_long_text.py`, `test_model_resolver.py`, `test_transcription.py`, `test_generation.py`, `test_output_writer.py`, `test_app_state.py`, `test_ui_callbacks.py`.

Project meta:

- `pyproject.toml`, `.gitignore`, `README.md`.

Runtime data dirs (created at startup, gitignored): `voices/`, `voices/audio/`, `outputs/`, `pretrained_models/{VoxCPM2,SenseVoiceSmall,ZipEnhancer}/`.

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md` (skeleton only; populated in Task 16)
- Create: `src/voxcpm_tts_tool/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Initialize git**

```bash
cd /c/Users/vibecoding/workspace/wys/VoxCPM-TTS-Tool && git init && git branch -M main
```
Expected: `Initialized empty Git repository in .../VoxCPM-TTS-Tool/.git/`

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "voxcpm-tts-tool"
version = "0.1.0"
description = "Standalone Gradio TTS tool wrapping the voxcpm SDK"
requires-python = ">=3.10,<3.13"
dependencies = [
    "voxcpm",
    "gradio>=5,<6",
    "numpy",
    "soundfile",
    "huggingface_hub",
    "modelscope",
    "funasr",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-mock"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

Note: Python 3.13 fails to build `editdistance` (transitive of voxcpm) without a C toolchain. Use 3.10–3.12 for development.

- [ ] **Step 3: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.venv*/
*.egg-info/
build/
dist/

voices/
outputs/
pretrained_models/
*.wav.tmp
voices.json.tmp
voices.broken-*.json

.env
mutagen.yml
mutagen.yml.lock
```

- [ ] **Step 4: Write `README.md` skeleton**

```markdown
# VoxCPM TTS Tool

Standalone Gradio web tool that wraps the [voxcpm](https://pypi.org/project/voxcpm/) TTS SDK.

See [`docs/superpowers/specs/2026-04-26-voxcpm-tts-tool-design.md`](docs/superpowers/specs/2026-04-26-voxcpm-tts-tool-design.md) for the complete design.

Detailed install/run/usage notes will be added in Task 16.
```

- [ ] **Step 5: Create package marker**

`src/voxcpm_tts_tool/__init__.py`:
```python
__version__ = "0.1.0"
```

- [ ] **Step 6: Create empty test package**

`tests/__init__.py`:
```python
```

- [ ] **Step 7: Write `tests/conftest.py` with shared fixtures**

```python
"""Shared test fixtures: fakes for VoxCPM SDK and SenseVoice ASR."""
from __future__ import annotations

import numpy as np
import pytest


class FakeVoxCPM:
    """Records calls to .generate(); returns a tiny zero waveform.

    Mirrors the surface used by orchestration: .generate(...) and .sample_rate.
    """

    sample_rate = 16000

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kwargs) -> np.ndarray:
        self.calls.append(kwargs)
        return np.zeros(8, dtype=np.float32)


class FakeRecognizer:
    """Returns the canned text for any wav path."""

    def __init__(self, text: str = "fake transcript"):
        self.text = text
        self.calls: list[str] = []

    def recognize(self, wav_path: str) -> str:
        self.calls.append(wav_path)
        return self.text


@pytest.fixture
def fake_model() -> FakeVoxCPM:
    return FakeVoxCPM()


@pytest.fixture
def fake_recognizer() -> FakeRecognizer:
    return FakeRecognizer()


@pytest.fixture
def project_root(tmp_path):
    """A tmp directory standing in for the project root with all runtime dirs."""
    for sub in ("voices", "voices/audio", "outputs",
                "pretrained_models/VoxCPM2",
                "pretrained_models/SenseVoiceSmall",
                "pretrained_models/ZipEnhancer"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path
```

- [ ] **Step 8: Verify package installs in editable mode**

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -e ".[dev]"
```
Expected (last line): `Successfully installed ... voxcpm-tts-tool-0.1.0`

If this fails with `editdistance` build errors, install Python 3.11 or 3.12 and retry.

- [ ] **Step 9: Verify pytest collects zero tests cleanly**

```bash
.venv/Scripts/python.exe -m pytest
```
Expected: `no tests ran in ...s` (exit 5 is fine).

- [ ] **Step 10: First commit**

```bash
git add pyproject.toml .gitignore README.md src tests
git commit -m "chore: project scaffold with pyproject, conftest fakes"
```

---

## Task 2: i18n Module

**Files:**
- Create: `src/voxcpm_tts_tool/i18n.py`
- Create: `tests/test_i18n.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_i18n.py`:
```python
from voxcpm_tts_tool import i18n


def test_returns_zh_when_lang_is_zh():
    assert i18n.t("tab.generation", "zh") == "语音生成"


def test_returns_en_when_lang_is_en():
    assert i18n.t("tab.generation", "en") == "Speech Generation"


def test_falls_back_to_zh_when_en_missing(monkeypatch):
    monkeypatch.setitem(i18n.STRINGS, "only.zh", {"zh": "中"})
    assert i18n.t("only.zh", "en") == "中"


def test_falls_back_to_key_when_both_missing():
    assert i18n.t("definitely.missing.key", "en") == "definitely.missing.key"


def test_chinese_dictionary_covers_every_key():
    missing = [k for k, v in i18n.STRINGS.items() if "zh" not in v]
    assert missing == [], f"keys missing zh translation: {missing}"
```

- [ ] **Step 2: Run and confirm failure**

```bash
.venv/Scripts/python.exe -m pytest tests/test_i18n.py -v
```
Expected: ImportError / 5 failures.

- [ ] **Step 3: Implement `i18n.py`**

```python
"""Minimal in-app translation dictionary with zh→key fallback.

Keys are stable string IDs (e.g. "tab.generation"); per spec §UI Design,
Chinese must cover every key; English may have gaps.
"""
from __future__ import annotations

Lang = str  # "zh" or "en"

STRINGS: dict[str, dict[Lang, str]] = {
    # tabs
    "tab.generation": {"zh": "语音生成", "en": "Speech Generation"},
    "tab.voice_library": {"zh": "音色管理", "en": "Voice Library"},
    "tab.usage": {"zh": "使用说明", "en": "Usage"},
    # mode labels
    "mode.design": {"zh": "声音设计", "en": "Voice Design"},
    "mode.clone": {"zh": "可控克隆", "en": "Controllable Cloning"},
    "mode.hifi": {"zh": "极致克隆", "en": "Ultimate Cloning"},
    # field labels
    "field.voice_name": {"zh": "音色名称", "en": "Voice Name"},
    "field.control": {"zh": "风格描述", "en": "Control Instruction"},
    "field.reference_audio": {"zh": "参考音频", "en": "Reference Audio"},
    "field.prompt_text": {"zh": "对应文本", "en": "Prompt Text / Transcript"},
    "field.denoise": {"zh": "启用降噪", "en": "Enable Denoise"},
    "field.default_voice": {"zh": "默认音色", "en": "Default Voice"},
    "field.script": {"zh": "脚本文本", "en": "Script Text"},
    # buttons
    "btn.transcribe": {"zh": "识别转写", "en": "Transcribe"},
    "btn.generate": {"zh": "开始生成", "en": "Generate"},
    "btn.save": {"zh": "保存", "en": "Save"},
    "btn.delete": {"zh": "删除", "en": "Delete"},
    "btn.refresh": {"zh": "刷新", "en": "Refresh"},
    "btn.insert_voice": {"zh": "插入音色标签", "en": "Insert Voice Tag"},
    "btn.insert_tag": {"zh": "插入非语言标签", "en": "Insert Tag"},
    # status / messages
    "status.asr_unavailable": {
        "zh": "ASR 模型不可用，识别按钮已禁用",
        "en": "ASR model unavailable; Transcribe button disabled",
    },
    "status.denoise_unavailable": {
        "zh": "ZipEnhancer 不可用，降噪开关无效",
        "en": "ZipEnhancer unavailable; denoise toggle has no effect",
    },
    "err.empty_script": {"zh": "脚本为空", "en": "Script is empty"},
    "err.missing_reference": {
        "zh": "找不到参考音频文件",
        "en": "Reference audio file is missing",
    },
    "err.unsupported_audio": {
        "zh": "仅支持 .wav 文件",
        "en": "Only .wav files are supported",
    },
    "err.duplicate_name": {"zh": "音色名重复", "en": "Voice name already exists"},
    "err.invalid_voice_name": {
        "zh": "音色名不能包含 < 或 >",
        "en": "Voice name must not contain < or >",
    },
    # default voice display name
    "voice.default": {"zh": "默认", "en": "Default"},
}


def t(key: str, lang: Lang) -> str:
    """Return localized string. Fallback order: lang → zh → key."""
    entry = STRINGS.get(key)
    if entry is None:
        return key
    if lang in entry:
        return entry[lang]
    if "zh" in entry:
        return entry["zh"]
    return key
```

- [ ] **Step 4: Run tests and confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_i18n.py -v
```
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/i18n.py tests/test_i18n.py
git commit -m "feat(i18n): translation dict with zh→key fallback"
```

---

## Task 3: Voice Library

**Files:**
- Create: `src/voxcpm_tts_tool/voice_library.py`
- Create: `tests/test_voice_library.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_voice_library.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxcpm_tts_tool.voice_library import (
    Voice,
    VoiceLibrary,
    VoiceLibraryError,
)


@pytest.fixture
def lib(project_root) -> VoiceLibrary:
    return VoiceLibrary(project_root / "voices")


def _wav(path: Path, name: str) -> Path:
    p = path / name
    p.write_bytes(b"RIFF...")
    return p


def test_create_design_voice_persists(lib, project_root):
    v = lib.create(name="旁白女声", mode="design", control="温柔清晰")
    assert v.id and v.id != ""
    on_disk = json.loads((project_root / "voices/voices.json").read_text("utf-8"))
    assert on_disk[0]["name"] == "旁白女声"
    assert on_disk[0]["mode"] == "design"
    assert "control" in on_disk[0]


def test_design_rejects_reference_audio(lib, project_root):
    wav = _wav(project_root, "x.wav")
    with pytest.raises(VoiceLibraryError, match="design.*reference"):
        lib.create(name="x", mode="design", reference_audio_upload=str(wav))


def test_clone_requires_reference_audio(lib):
    with pytest.raises(VoiceLibraryError, match="clone.*reference"):
        lib.create(name="c", mode="clone")


def test_clone_clears_prompt_text(lib, project_root):
    wav = _wav(project_root, "x.wav")
    v = lib.create(name="c", mode="clone", reference_audio_upload=str(wav),
                   prompt_text="should be dropped")
    assert v.prompt_text == ""


def test_hifi_requires_prompt_text(lib, project_root):
    wav = _wav(project_root, "x.wav")
    with pytest.raises(VoiceLibraryError, match="hifi.*prompt"):
        lib.create(name="h", mode="hifi", reference_audio_upload=str(wav))


def test_hifi_clears_control(lib, project_root):
    wav = _wav(project_root, "x.wav")
    v = lib.create(name="h", mode="hifi", reference_audio_upload=str(wav),
                   prompt_text="hi", control="ignored")
    assert v.control == ""


def test_duplicate_name_case_insensitive_after_trim(lib):
    lib.create(name="alpha", mode="design")
    with pytest.raises(VoiceLibraryError, match="duplicate"):
        lib.create(name=" Alpha ", mode="design")


def test_voice_name_with_angle_bracket_rejected(lib):
    with pytest.raises(VoiceLibraryError, match="< or >"):
        lib.create(name="bad<name", mode="design")


def test_only_wav_accepted(lib, project_root):
    mp3 = project_root / "x.mp3"
    mp3.write_bytes(b"id3")
    with pytest.raises(VoiceLibraryError, match=".wav"):
        lib.create(name="m", mode="clone", reference_audio_upload=str(mp3))


def test_audio_saved_as_id_dot_wav(lib, project_root):
    wav = _wav(project_root, "src.wav")
    v = lib.create(name="a", mode="clone", reference_audio_upload=str(wav))
    saved = project_root / "voices/audio" / f"{v.id}.wav"
    assert saved.exists()
    assert v.reference_audio == f"voices/audio/{v.id}.wav"


def test_atomic_write_uses_replace(lib, project_root, monkeypatch):
    """If process crashes between tmp write and replace, original survives."""
    lib.create(name="a", mode="design")
    original = (project_root / "voices/voices.json").read_text("utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(RuntimeError):
        lib.create(name="b", mode="design")
    # Original file unchanged
    assert (project_root / "voices/voices.json").read_text("utf-8") == original


def test_malformed_json_recovery(project_root):
    bad = project_root / "voices/voices.json"
    bad.write_text("not json", encoding="utf-8")
    lib = VoiceLibrary(project_root / "voices")
    assert lib.list_voices() == []
    backups = list((project_root / "voices").glob("voices.broken-*.json"))
    assert len(backups) == 1


def test_delete_removes_audio_file(lib, project_root):
    wav = _wav(project_root, "x.wav")
    v = lib.create(name="a", mode="clone", reference_audio_upload=str(wav))
    audio = project_root / "voices/audio" / f"{v.id}.wav"
    assert audio.exists()
    lib.delete(v.id)
    assert not audio.exists()


def test_rename_returns_warning(lib):
    v = lib.create(name="old", mode="design")
    msg = lib.update(v.id, name="new")
    assert "rename" in msg.lower() or "<old>" in msg


def test_id_is_uuid4_hex(lib):
    v = lib.create(name="a", mode="design")
    assert len(v.id) == 32
    assert all(c in "0123456789abcdef" for c in v.id)


def test_lookup_by_name_case_insensitive(lib):
    v = lib.create(name="MixedCase", mode="design")
    assert lib.find_by_name("  mixedcase  ").id == v.id


def test_design_denoise_forced_false(lib):
    v = lib.create(name="a", mode="design", denoise=True)
    assert v.denoise is False
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_voice_library.py -v
```
Expected: ImportError or many failures.

- [ ] **Step 3: Implement `voice_library.py`**

```python
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
    reference_audio: str = ""  # relative path "voices/audio/<id>.wav" or ""
    prompt_text: str = ""
    denoise: bool = False
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
        prompt_text: str = "",
        denoise: bool = False,
    ) -> Voice:
        self._validate_name(name)
        if self.find_by_name(name) is not None:
            raise VoiceLibraryError(f"duplicate name: {name!r}")
        voice_id = uuid.uuid4().hex
        rel_audio = self._stage_audio(reference_audio_upload, voice_id, mode)
        v = Voice(
            id=voice_id,
            name=name.strip(),
            mode=mode,
            control=control if mode != "hifi" else "",
            reference_audio=rel_audio,
            prompt_text=prompt_text if mode == "hifi" else "",
            denoise=bool(denoise) if mode != "design" else False,
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
        prompt_text: str | None = None,
        denoise: bool | None = None,
    ) -> str:
        """Update voice in place. Returns a human-readable warning string (may be empty)."""
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
        if denoise is not None:
            v.denoise = bool(denoise)

        if reference_audio_upload is not None:
            v.reference_audio = self._stage_audio(reference_audio_upload, v.id, v.mode)

        # Apply mode-specific clearing rules.
        if v.mode == "design":
            v.reference_audio = ""
            v.prompt_text = ""
            v.denoise = False
            self._remove_audio(v.id)
        elif v.mode == "clone":
            v.prompt_text = ""
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
        if v.mode == "design":
            if v.reference_audio or v.prompt_text or v.denoise:
                raise VoiceLibraryError(
                    "design voice must not store reference_audio, prompt_text, or denoise=true"
                )
        elif v.mode == "clone":
            if not v.reference_audio:
                raise VoiceLibraryError("clone voice requires reference_audio")
        elif v.mode == "hifi":
            if not v.reference_audio:
                raise VoiceLibraryError("hifi voice requires reference_audio")
            if not v.prompt_text:
                raise VoiceLibraryError("hifi voice requires non-empty prompt_text")

    def _stage_audio(
        self, upload_path: str | None, voice_id: str, mode: Mode
    ) -> str:
        if upload_path is None:
            return ""
        if mode == "design":
            raise VoiceLibraryError("design mode must not upload reference_audio")
        if not upload_path.lower().endswith(".wav"):
            raise VoiceLibraryError("only .wav uploads are supported in v1")
        dst = self.audio_dir / f"{voice_id}.wav"
        shutil.copyfile(upload_path, dst)
        return f"voices/audio/{voice_id}.wav"

    def _remove_audio(self, voice_id: str) -> None:
        path = self.audio_dir / f"{voice_id}.wav"
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
        return [Voice(**rec) for rec in raw]

    def _save(self) -> None:
        tmp = self.json_path.with_suffix(".json.tmp")
        data = [asdict(v) for v in self._voices]
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.json_path)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_voice_library.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/voice_library.py tests/test_voice_library.py
git commit -m "feat(library): voice CRUD with atomic JSON, mode validation, .wav-only"
```

---

## Task 4: Script Parser

**Files:**
- Create: `src/voxcpm_tts_tool/script_parser.py`
- Create: `tests/test_script_parser.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_script_parser.py`:
```python
from voxcpm_tts_tool.script_parser import ParsedSegment, parse_script


def _names(segments: list[ParsedSegment]) -> list[str | None]:
    return [s.voice_name for s in segments]


def test_no_switches_uses_default():
    segs, warns = parse_script("hello world", default_voice="alpha",
                               known_names={"alpha"})
    assert _names(segs) == ["alpha"]
    assert segs[0].text == "hello world"
    assert warns == []


def test_inline_switch_changes_voice():
    segs, _ = parse_script("hi <bob> there", default_voice="alpha",
                           known_names={"alpha", "bob"})
    assert _names(segs) == ["alpha", "bob"]
    assert segs[0].text == "hi "
    assert segs[1].text == " there"


def test_newline_resets_to_default():
    segs, _ = parse_script("<bob>line1\nline2", default_voice="alpha",
                           known_names={"alpha", "bob"})
    assert _names(segs) == ["bob", "alpha"]


def test_crlf_split():
    segs, _ = parse_script("a\r\nb\rc", default_voice="d",
                           known_names={"d"})
    assert [s.text for s in segs] == ["a", "b", "c"]


def test_empty_lines_skipped():
    segs, _ = parse_script("a\n\n   \nb", default_voice="d",
                           known_names={"d"})
    assert [s.text for s in segs] == ["a", "b"]


def test_case_insensitive_match():
    segs, _ = parse_script("<BOB>x", default_voice="a",
                           known_names={"a", "bob"})
    assert segs[0].voice_name == "bob"


def test_trim_whitespace_inside_brackets():
    segs, _ = parse_script("<  bob  >x", default_voice="a",
                           known_names={"a", "bob"})
    assert segs[0].voice_name == "bob"


def test_unknown_tag_preserved_with_warning():
    segs, warns = parse_script("hi <ghost> there", default_voice="a",
                               known_names={"a"})
    assert "<ghost>" in segs[0].text
    assert any("ghost" in w and "line 1" in w.lower() for w in warns)


def test_square_bracket_tags_preserved():
    segs, _ = parse_script("hello [laughing] world", default_voice="a",
                           known_names={"a"})
    assert "[laughing]" in segs[0].text


def test_cjk_voice_name():
    segs, _ = parse_script("<女声>大家好", default_voice="a",
                           known_names={"a", "女声"})
    assert segs[0].voice_name == "女声"
    assert segs[0].text == "大家好"
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_script_parser.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `script_parser.py`**

```python
"""Line-scoped script parser for `<voice name>` switches.

See spec §Script Semantics for line-splitting rules, voice-name matching,
and unknown-tag handling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_SWITCH_RE = re.compile(r"<([^<>\n\r]+)>")


@dataclass
class ParsedSegment:
    line_no: int          # 1-based
    voice_name: str | None  # None only inside warnings; segments always carry a resolved name
    text: str


def parse_script(
    script: str,
    *,
    default_voice: str,
    known_names: Iterable[str],
) -> tuple[list[ParsedSegment], list[str]]:
    """Tokenize script into [(line_no, voice_name, text), ...] segments + warnings.

    `default_voice` and `known_names` are matched case-insensitively after trim.
    """
    name_lookup = {n.strip().lower(): n for n in known_names}
    default_norm = default_voice.strip().lower()
    if default_norm not in name_lookup:
        # Allow the default to be used even if not in known_names (ephemeral default voice).
        name_lookup[default_norm] = default_voice

    segments: list[ParsedSegment] = []
    warnings: list[str] = []

    # Split on \r\n, \r, or \n in any order.
    lines = re.split(r"\r\n|\r|\n", script)

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        active_name = name_lookup[default_norm]
        cursor = 0
        accumulator = ""

        for match in _SWITCH_RE.finditer(line):
            # text before the tag goes to the active voice
            accumulator += line[cursor:match.start()]
            raw_name = match.group(1).strip()
            norm = raw_name.lower()
            if norm in name_lookup:
                # Flush accumulator and switch voice.
                if accumulator:
                    segments.append(ParsedSegment(idx, active_name, accumulator))
                    accumulator = ""
                active_name = name_lookup[norm]
            else:
                # Unknown tag: preserve verbatim and warn.
                accumulator += match.group(0)
                warnings.append(
                    f"line {idx}: unknown voice tag {match.group(0)!r} preserved as text"
                )
            cursor = match.end()

        accumulator += line[cursor:]
        if accumulator:
            segments.append(ParsedSegment(idx, active_name, accumulator))

    return segments, warnings
```

- [ ] **Step 4: Run, confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_script_parser.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/script_parser.py tests/test_script_parser.py
git commit -m "feat(parser): line-scoped <voice> parsing with CJK + unknown-tag warnings"
```

---

## Task 5: Long-Text Splitter & Concatenation

**Files:**
- Create: `src/voxcpm_tts_tool/long_text.py`
- Create: `tests/test_long_text.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_long_text.py`:
```python
import numpy as np

from voxcpm_tts_tool.long_text import concat_waveforms, split_for_generation

CHAR_BUDGET_CJK = 80


def test_short_input_returns_one_chunk():
    out = split_for_generation("你好", char_budget=CHAR_BUDGET_CJK)
    assert out == ["你好"]


def test_splits_on_cjk_period():
    out = split_for_generation("第一句。第二句。第三句。",
                               char_budget=CHAR_BUDGET_CJK)
    assert out == ["第一句。", "第二句。", "第三句。"]


def test_splits_on_ascii_period_and_question_mark():
    out = split_for_generation("First. Second? Third!",
                               char_budget=200)
    assert out == ["First.", " Second?", " Third!"]


def test_falls_back_to_comma_when_sentence_too_long():
    long_sentence = "一段没有句号但很长的话，再来一截，又来一段，最后一截"
    out = split_for_generation(long_sentence, char_budget=10)
    # Each chunk must be <= budget OR a single comma-bounded segment.
    assert all("。" not in c for c in out)
    assert len(out) > 1


def test_square_bracket_tag_indivisible():
    # Budget is small but [laughing] cannot be split.
    out = split_for_generation("ha [laughing] ha", char_budget=4)
    joined = "".join(out)
    assert "[laughing]" in joined
    assert all(c == "" or "[laughing]" in c or "[" not in c for c in out)


def test_empty_input_yields_no_chunks():
    assert split_for_generation("", char_budget=80) == []


def test_concat_preserves_order():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0], dtype=np.float32)
    c = np.array([4.0, 5.0], dtype=np.float32)
    out = concat_waveforms([a, b, c])
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_concat_empty_list_returns_empty_array():
    out = concat_waveforms([])
    assert out.shape == (0,)
    assert out.dtype == np.float32
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_long_text.py -v
```

- [ ] **Step 3: Implement `long_text.py`**

```python
"""Punctuation-aware splitter and waveform concatenation.

See spec §Long Text for splitting rules. Square-bracket tags are indivisible;
sentence terminators are preferred boundaries; commas are a fallback when a
single sentence exceeds the per-chunk character budget.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

_SENTENCE_TERMINATORS = "。！？；…!?;"
_COMMA_BOUNDARIES = "，、,"
_TAG_RE = re.compile(r"\[[^\[\]\n\r]*\]")


def _tokenize(text: str) -> list[str]:
    """Split text into atomic tokens: bracket-tag groups stay whole."""
    tokens: list[str] = []
    cursor = 0
    for m in _TAG_RE.finditer(text):
        if m.start() > cursor:
            tokens.extend(text[cursor:m.start()])  # one-char tokens for non-tag text
        tokens.append(m.group(0))
        cursor = m.end()
    if cursor < len(text):
        tokens.extend(text[cursor:])
    return tokens


def split_for_generation(text: str, *, char_budget: int) -> list[str]:
    """Split `text` into chunks suitable for one generate() call.

    Order of preference: sentence terminators, then comma-class boundaries
    when a sentence alone exceeds `char_budget`. Bracket tags are atomic.
    """
    if not text:
        return []
    tokens = _tokenize(text)
    chunks: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            chunks.append("".join(buf))
            buf.clear()

    for tok in tokens:
        buf.append(tok)
        # Prefer sentence terminator boundaries.
        if len(tok) == 1 and tok in _SENTENCE_TERMINATORS:
            flush()
            continue
        # Comma fallback only if buffer has overflowed budget.
        if len(tok) == 1 and tok in _COMMA_BOUNDARIES and sum(len(t) for t in buf) >= char_budget:
            flush()
            continue
    flush()
    return chunks


def concat_waveforms(waveforms: Iterable[np.ndarray]) -> np.ndarray:
    """Concatenate 1-D float32 waveforms in order. Empty input → empty array."""
    arrays = list(waveforms)
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(arrays).astype(np.float32, copy=False)
```

- [ ] **Step 4: Run, confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_long_text.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/long_text.py tests/test_long_text.py
git commit -m "feat(long-text): punctuation splitter with bracket-tag protection"
```

---

## Task 6: Model Resolver

**Files:**
- Create: `src/voxcpm_tts_tool/model_resolver.py`
- Create: `tests/test_model_resolver.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_model_resolver.py`:
```python
from pathlib import Path

import pytest

from voxcpm_tts_tool import model_resolver as mr


def _seed(dir_: Path, files: list[str]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for f in files:
        (dir_ / f).write_text("dummy", encoding="utf-8")


# ---- valid-directory predicate ----

def test_dir_with_expected_file_is_valid(tmp_path):
    _seed(tmp_path / "m", ["config.json"])
    assert mr._has_any(tmp_path / "m", ["config.json", "model.safetensors"])


def test_empty_dir_is_invalid(tmp_path):
    (tmp_path / "m").mkdir()
    assert not mr._has_any(tmp_path / "m", ["config.json"])


def test_gitkeep_only_dir_is_invalid(tmp_path):
    _seed(tmp_path / "m", [".gitkeep"])
    assert not mr._has_any(tmp_path / "m", ["config.json"])


# ---- env-var override ----

def test_env_var_path_used_when_valid(tmp_path, monkeypatch):
    target = tmp_path / "external/voxcpm"
    _seed(target, ["config.json"])
    monkeypatch.setenv("VOXCPM_MODEL_DIR", str(target))
    path = mr.resolve_voxcpm(tmp_path / "pretrained_models/VoxCPM2",
                              modelscope_download=lambda *a, **k: pytest.fail("called"),
                              hf_download=lambda *a, **k: pytest.fail("called"))
    assert path == target


def test_env_var_pointing_to_empty_dir_does_not_skip_download(tmp_path, monkeypatch):
    empty = tmp_path / "external/voxcpm"
    empty.mkdir(parents=True)
    monkeypatch.setenv("VOXCPM_MODEL_DIR", str(empty))
    local_dir = tmp_path / "pretrained_models/VoxCPM2"
    calls = []

    def fake_ms(repo_id, local_dir):
        calls.append(("ms", repo_id, local_dir))
        _seed(local_dir, ["config.json"])
        return Path(local_dir)

    path = mr.resolve_voxcpm(local_dir, modelscope_download=fake_ms,
                              hf_download=lambda *a, **k: pytest.fail("called"))
    assert path == local_dir
    assert calls and calls[0][0] == "ms"


# ---- modelscope hit ----

def test_modelscope_download_then_hit(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXCPM_MODEL_DIR", raising=False)
    local_dir = tmp_path / "pretrained_models/VoxCPM2"
    calls = []

    def fake_ms(repo_id, local_dir):
        calls.append(repo_id)
        _seed(local_dir, ["config.json"])
        return Path(local_dir)

    path = mr.resolve_voxcpm(local_dir, modelscope_download=fake_ms,
                              hf_download=lambda *a, **k: pytest.fail("called"))
    assert path == local_dir
    assert calls == ["OpenBMB/VoxCPM2"]


# ---- HF fallback ----

def test_hf_fallback_when_modelscope_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXCPM_MODEL_DIR", raising=False)
    local_dir = tmp_path / "pretrained_models/VoxCPM2"

    def failing_ms(*a, **k):
        raise RuntimeError("modelscope down")

    def fake_hf(repo_id, local_dir):
        _seed(local_dir, ["config.json"])
        return Path(local_dir)

    path = mr.resolve_voxcpm(local_dir, modelscope_download=failing_ms, hf_download=fake_hf)
    assert path == local_dir


# ---- complete failure ----

def test_voxcpm_failure_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXCPM_MODEL_DIR", raising=False)
    local_dir = tmp_path / "pretrained_models/VoxCPM2"

    with pytest.raises(mr.ModelResolutionError):
        mr.resolve_voxcpm(
            local_dir,
            modelscope_download=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ms fail")),
            hf_download=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("hf fail")),
        )


# ---- SenseVoice has HF fallback too ----

def test_sensevoice_falls_back_to_hf(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXCPM_ASR_MODEL_DIR", raising=False)
    local_dir = tmp_path / "pretrained_models/SenseVoiceSmall"

    def failing_ms(*a, **k):
        raise RuntimeError("ms fail")

    def fake_hf(repo_id, local_dir):
        assert repo_id == "FunAudioLLM/SenseVoiceSmall"
        _seed(local_dir, ["model.pt"])
        return Path(local_dir)

    path = mr.resolve_sensevoice(local_dir, modelscope_download=failing_ms, hf_download=fake_hf)
    assert path == local_dir


# ---- ZipEnhancer has no HF fallback ----

def test_zipenhancer_no_hf_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("ZIPENHANCER_MODEL_PATH", raising=False)
    local_dir = tmp_path / "pretrained_models/ZipEnhancer"

    def failing_ms(*a, **k):
        raise RuntimeError("ms fail")

    with pytest.raises(mr.ModelResolutionError):
        mr.resolve_zipenhancer(local_dir, modelscope_download=failing_ms)
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_model_resolver.py -v
```

- [ ] **Step 3: Implement `model_resolver.py`**

```python
"""Resolves VoxCPM2, SenseVoiceSmall, and ZipEnhancer model paths.

Strategy per spec §Model Resolution:
  1. env-var path (if valid)
  2. project-local pretrained_models/<name> (if valid)
  3. ModelScope download to project-local path
  4. HF download to project-local path (only VoxCPM2 + SenseVoiceSmall)

Downloaders are passed in (dependency injection) so tests don't hit the network.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

# Files we expect to find inside a "valid" model directory.
VOXCPM_EXPECTED = ("config.json", "model.safetensors", "pytorch_model.bin")
SENSEVOICE_EXPECTED = ("model.pt", "config.yaml", "tokens.json")
ZIPENHANCER_EXPECTED = ("pytorch_model.bin", "configuration.json")

Downloader = Callable[[str, Path], Path]


class ModelResolutionError(RuntimeError):
    """Raised when no resolver step yields a valid local model directory."""


def _has_any(directory: Path, expected: Iterable[str]) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    names = {p.name for p in directory.iterdir()}
    return any(name in names for name in expected)


def _try_path(env: str | None, expected: Iterable[str]) -> Path | None:
    if not env:
        return None
    p = Path(env)
    return p if _has_any(p, expected) else None


def _resolve(
    *,
    env_var: str,
    local_dir: Path,
    expected: Iterable[str],
    modelscope_repo: str,
    modelscope_download: Downloader,
    hf_repo: str | None,
    hf_download: Downloader | None,
) -> Path:
    expected = tuple(expected)
    # 1. env var
    env_hit = _try_path(os.environ.get(env_var), expected)
    if env_hit is not None:
        return env_hit
    # 2. local
    if _has_any(local_dir, expected):
        return local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    # 3. ModelScope
    try:
        result = modelscope_download(modelscope_repo, local_dir)
        if _has_any(Path(result), expected):
            return Path(result)
    except Exception:
        pass
    # 4. HF (optional)
    if hf_repo is not None and hf_download is not None:
        try:
            result = hf_download(hf_repo, local_dir)
            if _has_any(Path(result), expected):
                return Path(result)
        except Exception:
            pass
    raise ModelResolutionError(
        f"failed to resolve model {modelscope_repo} into {local_dir}"
    )


def resolve_voxcpm(
    local_dir: Path,
    *,
    modelscope_download: Downloader,
    hf_download: Downloader,
) -> Path:
    return _resolve(
        env_var="VOXCPM_MODEL_DIR",
        local_dir=Path(local_dir),
        expected=VOXCPM_EXPECTED,
        modelscope_repo="OpenBMB/VoxCPM2",
        modelscope_download=modelscope_download,
        hf_repo="openbmb/VoxCPM2",
        hf_download=hf_download,
    )


def resolve_sensevoice(
    local_dir: Path,
    *,
    modelscope_download: Downloader,
    hf_download: Downloader,
) -> Path:
    return _resolve(
        env_var="VOXCPM_ASR_MODEL_DIR",
        local_dir=Path(local_dir),
        expected=SENSEVOICE_EXPECTED,
        modelscope_repo="iic/SenseVoiceSmall",
        modelscope_download=modelscope_download,
        hf_repo="FunAudioLLM/SenseVoiceSmall",
        hf_download=hf_download,
    )


def resolve_zipenhancer(
    local_dir: Path,
    *,
    modelscope_download: Downloader,
) -> Path:
    return _resolve(
        env_var="ZIPENHANCER_MODEL_PATH",
        local_dir=Path(local_dir),
        expected=ZIPENHANCER_EXPECTED,
        modelscope_repo="iic/speech_zipenhancer_ans_multiloss_16k_base",
        modelscope_download=modelscope_download,
        hf_repo=None,
        hf_download=None,
    )


# ---- Default downloader factories (used by app, not by tests) ----

def real_modelscope_download(repo_id: str, local_dir: Path) -> Path:
    """Download a repo via modelscope into local_dir. Imported lazily."""
    from modelscope import snapshot_download as ms_download

    return Path(ms_download(repo_id, cache_dir=str(local_dir.parent), local_dir=str(local_dir)))


def real_hf_download(repo_id: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download as hf_download

    return Path(hf_download(repo_id=repo_id, local_dir=str(local_dir)))


def configure_runtime_caches(cache_root: Path) -> None:
    """Set TOKENIZERS_PARALLELISM and default HF/MODELSCOPE caches if unset."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache_root.mkdir(parents=True, exist_ok=True)
    for var in ("HF_HOME", "MODELSCOPE_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
        os.environ.setdefault(var, str(cache_root))
```

- [ ] **Step 4: Run, confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_model_resolver.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/model_resolver.py tests/test_model_resolver.py
git commit -m "feat(resolver): three-tier model resolution with HF fallback for VoxCPM2/SenseVoice"
```

---

## Task 7: Transcription Wrapper

**Files:**
- Create: `src/voxcpm_tts_tool/transcription.py`
- Create: `tests/test_transcription.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_transcription.py`:
```python
import pytest

from voxcpm_tts_tool.transcription import (
    AsrUnavailable,
    SenseVoiceTranscriber,
)


def test_calls_underlying_recognizer(fake_recognizer, tmp_path):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF...")
    t = SenseVoiceTranscriber.from_recognizer(fake_recognizer)
    assert t.transcribe(str(wav)) == "fake transcript"
    assert fake_recognizer.calls == [str(wav)]


def test_unavailable_raises(tmp_path):
    t = SenseVoiceTranscriber.unavailable("import failed")
    with pytest.raises(AsrUnavailable, match="import failed"):
        t.transcribe(str(tmp_path / "x.wav"))


def test_is_available_flag():
    t1 = SenseVoiceTranscriber.unavailable("nope")
    t2 = SenseVoiceTranscriber.from_recognizer(object())
    assert t1.is_available is False
    assert t2.is_available is True
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_transcription.py -v
```

- [ ] **Step 3: Implement `transcription.py`**

```python
"""Thin wrapper around SenseVoiceSmall via FunASR.

Importing FunASR/torch is expensive and may fail; this module isolates the
import and exposes a typed wrapper that the rest of the app uses.
See spec §Reference-audio transcription behavior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AsrUnavailable(RuntimeError):
    """Raised when transcription is requested but no recognizer was loaded."""


class _Recognizer(Protocol):
    def recognize(self, wav_path: str) -> str: ...


class SenseVoiceTranscriber:
    """Wrapper exposing .is_available + .transcribe(wav_path) -> str."""

    def __init__(self, recognizer: _Recognizer | None, error: str = ""):
        self._rec = recognizer
        self._err = error

    @classmethod
    def from_recognizer(cls, recognizer: _Recognizer) -> "SenseVoiceTranscriber":
        return cls(recognizer)

    @classmethod
    def unavailable(cls, error: str) -> "SenseVoiceTranscriber":
        return cls(None, error=error)

    @property
    def is_available(self) -> bool:
        return self._rec is not None

    def transcribe(self, wav_path: str) -> str:
        if self._rec is None:
            raise AsrUnavailable(f"ASR is not available: {self._err}")
        text = self._rec.recognize(wav_path)
        return text.strip()


def load_real_transcriber(model_dir: Path) -> SenseVoiceTranscriber:
    """Construct a real SenseVoice recognizer; on import failure return unavailable."""
    try:
        from funasr import AutoModel  # type: ignore
    except Exception as exc:
        return SenseVoiceTranscriber.unavailable(f"funasr import failed: {exc}")

    try:
        model = AutoModel(model=str(model_dir), disable_update=True)
    except Exception as exc:
        return SenseVoiceTranscriber.unavailable(f"SenseVoice load failed: {exc}")

    class _Adapter:
        def recognize(self, wav_path: str) -> str:
            res = model.generate(input=wav_path, language="auto", use_itn=True)
            if not res:
                return ""
            text = res[0].get("text", "")
            return _strip_funasr_tags(text)

    return SenseVoiceTranscriber.from_recognizer(_Adapter())


def _strip_funasr_tags(text: str) -> str:
    """Remove FunASR special tokens like <|zh|>, <|woitn|>, etc."""
    import re

    return re.sub(r"<\|[^|]+\|>", "", text).strip()
```

- [ ] **Step 4: Run, confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_transcription.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/transcription.py tests/test_transcription.py
git commit -m "feat(asr): SenseVoice wrapper with availability flag and tag stripping"
```

---

## Task 8: Generation Kwargs Builder

**Files:**
- Create: `src/voxcpm_tts_tool/generation.py`
- Create: `tests/test_generation.py`

- [ ] **Step 1: Write the failing tests for `build_generate_kwargs`**

`tests/test_generation.py`:
```python
from voxcpm_tts_tool.generation import build_generate_kwargs
from voxcpm_tts_tool.voice_library import Voice


def _design(control: str = "", denoise: bool = False) -> Voice:
    return Voice(id="x", name="x", mode="design", control=control, denoise=denoise)


def _clone(control: str = "", denoise: bool = False) -> Voice:
    return Voice(id="x", name="x", mode="clone", control=control,
                 reference_audio="voices/audio/x.wav", denoise=denoise)


def _hifi(denoise: bool = False) -> Voice:
    return Voice(id="x", name="x", mode="hifi",
                 reference_audio="voices/audio/x.wav",
                 prompt_text="hello", denoise=denoise)


def test_design_with_control_prefixes_text():
    kw = build_generate_kwargs(_design(control="温柔"), "你好",
                               zipenhancer_loaded=True, audio_root=".")
    assert kw == {"text": "(温柔)你好"}


def test_design_without_control_passes_text_only():
    kw = build_generate_kwargs(_design(control=""), "你好",
                               zipenhancer_loaded=True, audio_root=".")
    assert kw == {"text": "你好"}


def test_clone_with_control_and_reference_path():
    kw = build_generate_kwargs(_clone(control="温柔"), "你好",
                               zipenhancer_loaded=True, audio_root="/root")
    assert kw["text"] == "(温柔)你好"
    assert kw["reference_wav_path"] == "/root/voices/audio/x.wav"
    assert "prompt_wav_path" not in kw
    assert "prompt_text" not in kw


def test_clone_without_control():
    kw = build_generate_kwargs(_clone(), "你好",
                               zipenhancer_loaded=True, audio_root="/r")
    assert kw["text"] == "你好"


def test_hifi_passes_prompt_pair_and_reference():
    kw = build_generate_kwargs(_hifi(), "target",
                               zipenhancer_loaded=True, audio_root="/r")
    expected_path = "/r/voices/audio/x.wav"
    assert kw == {
        "text": "target",
        "prompt_wav_path": expected_path,
        "prompt_text": "hello",
        "reference_wav_path": expected_path,
        "denoise": False,
    }


def test_denoise_true_only_when_voice_on_AND_zipenhancer_loaded():
    on = _hifi(denoise=True)
    kw_yes = build_generate_kwargs(on, "t", zipenhancer_loaded=True, audio_root="/r")
    kw_no_zh = build_generate_kwargs(on, "t", zipenhancer_loaded=False, audio_root="/r")
    off = _hifi(denoise=False)
    kw_off = build_generate_kwargs(off, "t", zipenhancer_loaded=True, audio_root="/r")
    assert kw_yes["denoise"] is True
    assert kw_no_zh["denoise"] is False
    assert kw_off["denoise"] is False


def test_design_never_includes_denoise_kwarg():
    kw = build_generate_kwargs(_design(), "x", zipenhancer_loaded=True, audio_root=".")
    assert "denoise" not in kw  # design has no audio path; denoise is irrelevant
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py -v
```

- [ ] **Step 3: Implement `build_generate_kwargs` in `generation.py`**

```python
"""Generation kwargs builder + end-to-end orchestration.

`build_generate_kwargs` maps a (Voice, chunk, runtime flags) to the dict
passed to `voxcpm.VoxCPM.generate`. See spec §Generation Flow for the rules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .long_text import concat_waveforms, split_for_generation
from .script_parser import parse_script
from .voice_library import Voice, VoiceLibrary


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
) -> dict:
    """Return the kwargs dict passed to voxcpm.VoxCPM.generate(...)."""
    if voice.mode == "design":
        text = f"({voice.control}){chunk}" if voice.control else chunk
        return {"text": text}

    abs_audio = _absolute_audio_path(audio_root, voice.reference_audio)
    denoise = bool(voice.denoise) and zipenhancer_loaded

    if voice.mode == "clone":
        text = f"({voice.control}){chunk}" if voice.control else chunk
        return {
            "text": text,
            "reference_wav_path": abs_audio,
            "denoise": denoise,
        }

    if voice.mode == "hifi":
        return {
            "text": chunk,
            "prompt_wav_path": abs_audio,
            "prompt_text": voice.prompt_text,
            "reference_wav_path": abs_audio,
            "denoise": denoise,
        }

    raise ValueError(f"unknown mode: {voice.mode}")


@dataclass
class GenerationResult:
    waveform: np.ndarray
    sample_rate: int
    log: list[str]


class GenerationError(RuntimeError):
    """Raised when generation aborts mid-script. Carries segment context."""


def run_generation(
    *,
    script: str,
    default_voice: Voice,
    library: VoiceLibrary,
    model: _GeneratesAudio,
    audio_root: str,
    zipenhancer_loaded: bool,
    char_budget: int = 80,
) -> GenerationResult:
    """End-to-end: parse → split → generate per chunk → concat. Pure of I/O."""
    if not script.strip():
        raise GenerationError("script is empty")

    by_name: dict[str, Voice] = {v.name: v for v in library.list_voices()}
    by_name[default_voice.name] = default_voice

    segments, warnings = parse_script(
        script,
        default_voice=default_voice.name,
        known_names=by_name.keys(),
    )

    log: list[str] = list(warnings)
    waveforms: list[np.ndarray] = []

    for seg_idx, seg in enumerate(segments):
        voice = by_name[seg.voice_name]
        if voice.mode in ("clone", "hifi"):
            audio_path = _absolute_audio_path(audio_root, voice.reference_audio)
            if not os.path.exists(audio_path):
                raise GenerationError(
                    f"segment {seg_idx} ({voice.name!r}): reference audio missing at {audio_path}"
                )
        chunks = split_for_generation(seg.text, char_budget=char_budget)
        for chunk_idx, chunk in enumerate(chunks):
            kwargs = build_generate_kwargs(
                voice, chunk,
                zipenhancer_loaded=zipenhancer_loaded,
                audio_root=audio_root,
            )
            try:
                wav = model.generate(**kwargs)
            except Exception as exc:
                preview = chunk[:40].replace("\n", " ")
                raise GenerationError(
                    f"segment {seg_idx} chunk {chunk_idx} ({voice.name!r}): {exc} "
                    f"[text: {preview!r}]"
                ) from exc
            waveforms.append(wav)
            log.append(f"seg {seg_idx} chunk {chunk_idx} voice={voice.name} chars={len(chunk)}")

    return GenerationResult(
        waveform=concat_waveforms(waveforms),
        sample_rate=int(model.sample_rate),
        log=log,
    )
```

- [ ] **Step 4: Run kwargs tests, confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py -v
```

- [ ] **Step 5: Add end-to-end orchestration tests**

Append to `tests/test_generation.py`:
```python
import os

import numpy as np
import pytest

from voxcpm_tts_tool.generation import GenerationError, run_generation
from voxcpm_tts_tool.voice_library import Voice, VoiceLibrary


def _make_lib_with_design(project_root):
    lib = VoiceLibrary(project_root / "voices")
    lib.create(name="alpha", mode="design")
    return lib


def test_run_generation_calls_model_per_chunk(project_root, fake_model):
    lib = _make_lib_with_design(project_root)
    default = lib.find_by_name("alpha")
    result = run_generation(
        script="第一句。第二句。",
        default_voice=default,
        library=lib,
        model=fake_model,
        audio_root=str(project_root),
        zipenhancer_loaded=True,
        char_budget=80,
    )
    assert len(fake_model.calls) == 2
    assert all(c["text"] in ("第一句。", "第二句。") for c in fake_model.calls)
    assert result.waveform.dtype == np.float32
    assert result.sample_rate == 16000


def test_empty_script_raises(project_root, fake_model):
    lib = _make_lib_with_design(project_root)
    default = lib.find_by_name("alpha")
    with pytest.raises(GenerationError, match="empty"):
        run_generation(script="   ", default_voice=default, library=lib,
                       model=fake_model, audio_root=str(project_root),
                       zipenhancer_loaded=True)


def test_missing_reference_audio_aborts(project_root, fake_model):
    lib = VoiceLibrary(project_root / "voices")
    wav = project_root / "ref.wav"
    wav.write_bytes(b"RIFF...")
    v = lib.create(name="bob", mode="clone", reference_audio_upload=str(wav))
    # Now delete the audio file behind the library's back.
    os.remove(project_root / "voices/audio" / f"{v.id}.wav")
    with pytest.raises(GenerationError, match="reference audio missing"):
        run_generation(script="hi", default_voice=v, library=lib,
                       model=fake_model, audio_root=str(project_root),
                       zipenhancer_loaded=True)


def test_one_segment_failure_carries_index(project_root):
    lib = _make_lib_with_design(project_root)
    default = lib.find_by_name("alpha")

    class BoomModel:
        sample_rate = 16000
        def generate(self, **kwargs):
            raise RuntimeError("model exploded")

    with pytest.raises(GenerationError, match="segment 0 chunk 0.*model exploded"):
        run_generation(script="hi", default_voice=default, library=lib,
                       model=BoomModel(), audio_root=str(project_root),
                       zipenhancer_loaded=True)
```

- [ ] **Step 6: Run all generation tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_generation.py -v
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/voxcpm_tts_tool/generation.py tests/test_generation.py
git commit -m "feat(generation): kwargs builder + end-to-end orchestration with fake model"
```

---

## Task 9: Output WAV Writer

**Files:**
- Create: `src/voxcpm_tts_tool/output_writer.py`
- Create: `tests/test_output_writer.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_output_writer.py`:
```python
import re
from pathlib import Path

import numpy as np

from voxcpm_tts_tool.output_writer import write_output_wav


def test_writes_wav_with_correct_sample_rate(tmp_path):
    wav = np.zeros(16, dtype=np.float32)
    out = write_output_wav(wav, sample_rate=22050, outputs_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".wav"
    import soundfile as sf
    info = sf.info(str(out))
    assert info.samplerate == 22050


def test_filename_uses_yyyymmdd_hhmmss_mmm(tmp_path):
    wav = np.zeros(8, dtype=np.float32)
    out = write_output_wav(wav, sample_rate=16000, outputs_dir=tmp_path)
    assert re.match(r"\d{8}-\d{6}-\d{3}\.wav", out.name)


def test_collision_appends_suffix(tmp_path, monkeypatch):
    """Force two writes to yield the same base filename."""
    monkeypatch.setattr("voxcpm_tts_tool.output_writer._timestamp",
                        lambda: "20260426-120000-000")
    wav = np.zeros(8, dtype=np.float32)
    a = write_output_wav(wav, sample_rate=16000, outputs_dir=tmp_path)
    b = write_output_wav(wav, sample_rate=16000, outputs_dir=tmp_path)
    assert a.name == "20260426-120000-000.wav"
    assert b.name == "20260426-120000-000-1.wav"
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_output_writer.py -v
```

- [ ] **Step 3: Implement `output_writer.py`**

```python
"""Write generated waveforms to outputs/<timestamp>.wav with collision suffix.

See spec §Generation Flow step 9 for filename convention.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


def _timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"


def write_output_wav(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    outputs_dir: Path,
) -> Path:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    base = _timestamp()
    candidate = outputs_dir / f"{base}.wav"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = outputs_dir / f"{base}-{suffix}.wav"
    sf.write(str(candidate), waveform, samplerate=int(sample_rate), subtype="PCM_16")
    return candidate
```

- [ ] **Step 4: Run, confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_output_writer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/output_writer.py tests/test_output_writer.py
git commit -m "feat(output): timestamped wav writer with collision suffix"
```

---

## Task 10: App State Container

**Files:**
- Create: `src/voxcpm_tts_tool/app_state.py`
- Create: `tests/test_app_state.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_app_state.py`:
```python
from pathlib import Path

from voxcpm_tts_tool.app_state import (
    AppPaths,
    EPHEMERAL_DEFAULT_VOICE_ID,
    ephemeral_default_voice,
    paths_for,
)


def test_paths_for_creates_runtime_dirs(tmp_path):
    p = paths_for(tmp_path)
    for sub in (p.voices, p.voices_audio, p.outputs,
                p.voxcpm_dir, p.sensevoice_dir, p.zipenhancer_dir):
        assert sub.exists() and sub.is_dir()


def test_ephemeral_default_voice_unaddressable():
    v = ephemeral_default_voice()
    assert v.id == EPHEMERAL_DEFAULT_VOICE_ID
    assert v.id.startswith("__")
    assert v.mode == "design"
    assert v.reference_audio == ""
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_app_state.py -v
```

- [ ] **Step 3: Implement `app_state.py`**

```python
"""Holds resolved paths, the model singleton, the library, and the recognizer.

Created once in app.py at startup. Tests can construct partial AppState by
passing fakes directly into the dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .transcription import SenseVoiceTranscriber
from .voice_library import Voice, VoiceLibrary

EPHEMERAL_DEFAULT_VOICE_ID = "__default__"


@dataclass
class AppPaths:
    root: Path
    voices: Path
    voices_audio: Path
    outputs: Path
    voxcpm_dir: Path
    sensevoice_dir: Path
    zipenhancer_dir: Path
    cache_dir: Path


def paths_for(project_root: Path) -> AppPaths:
    root = Path(project_root)
    p = AppPaths(
        root=root,
        voices=root / "voices",
        voices_audio=root / "voices" / "audio",
        outputs=root / "outputs",
        voxcpm_dir=root / "pretrained_models" / "VoxCPM2",
        sensevoice_dir=root / "pretrained_models" / "SenseVoiceSmall",
        zipenhancer_dir=root / "pretrained_models" / "ZipEnhancer",
        cache_dir=root / ".cache",
    )
    for d in (p.voices, p.voices_audio, p.outputs,
              p.voxcpm_dir, p.sensevoice_dir, p.zipenhancer_dir, p.cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    return p


def ephemeral_default_voice() -> Voice:
    """Per spec §Error Handling: an in-memory voice used only when the library is empty."""
    return Voice(
        id=EPHEMERAL_DEFAULT_VOICE_ID,
        name=EPHEMERAL_DEFAULT_VOICE_ID,
        mode="design",
        control="",
    )


@dataclass
class AppState:
    paths: AppPaths
    library: VoiceLibrary
    model: object  # voxcpm.VoxCPM at runtime; FakeVoxCPM in tests
    transcriber: SenseVoiceTranscriber
    zipenhancer_loaded: bool

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

- [ ] **Step 4: Run, confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_app_state.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/app_state.py tests/test_app_state.py
git commit -m "feat(state): AppPaths, AppState, ephemeral default voice"
```

---

## Task 11: UI Callback Functions (UI-pure logic)

Pull the field-visibility, dropdown-refresh, and label-relabel logic into a
testable module so the Gradio assembly itself can stay free of branching.

**Files:**
- Create: `src/voxcpm_tts_tool/ui_callbacks.py`
- Create: `tests/test_ui_callbacks.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_ui_callbacks.py`:
```python
from voxcpm_tts_tool.ui_callbacks import (
    field_visibility,
    insert_tag,
    insert_voice_tag,
    voice_dropdown_choices,
)
from voxcpm_tts_tool.app_state import EPHEMERAL_DEFAULT_VOICE_ID, ephemeral_default_voice
from voxcpm_tts_tool.voice_library import Voice


def _design(name: str = "x") -> Voice:
    return Voice(id="i" + name, name=name, mode="design")


def test_design_hides_audio_and_prompt_and_transcribe_and_denoise():
    vis = field_visibility("design")
    assert vis == {
        "voice_name": True,
        "control": True,
        "reference_audio": False,
        "prompt_text": False,
        "transcribe_button": False,
        "denoise": False,
    }


def test_clone_shows_audio_and_denoise_hides_prompt_transcribe():
    vis = field_visibility("clone")
    assert vis["reference_audio"] is True
    assert vis["denoise"] is True
    assert vis["prompt_text"] is False
    assert vis["transcribe_button"] is False
    assert vis["control"] is True


def test_hifi_shows_prompt_and_transcribe_hides_control():
    vis = field_visibility("hifi")
    assert vis["prompt_text"] is True
    assert vis["transcribe_button"] is True
    assert vis["denoise"] is True
    assert vis["control"] is False


def test_voice_dropdown_includes_real_voices():
    voices = [_design("alpha"), _design("beta")]
    choices = voice_dropdown_choices(voices, lang="zh", ephemeral=ephemeral_default_voice())
    # Returns list of (label, id) tuples; ephemeral first only when no real voices.
    assert ("alpha", "ialpha") in choices
    assert ("beta", "ibeta") in choices
    assert all(label != "__default__" for label, _ in choices)


def test_voice_dropdown_shows_ephemeral_when_no_voices():
    choices = voice_dropdown_choices([], lang="en", ephemeral=ephemeral_default_voice())
    assert choices == [("Default", EPHEMERAL_DEFAULT_VOICE_ID)]


def test_insert_voice_tag_appends_brackets_to_script():
    out = insert_voice_tag("hello ", voice_name="bob")
    assert out == "hello <bob>"


def test_insert_tag_appends_bracket_tag():
    out = insert_tag("hi ", tag="laughing")
    assert out == "hi [laughing]"
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ui_callbacks.py -v
```

- [ ] **Step 3: Implement `ui_callbacks.py`**

```python
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
        "transcribe_button": False, "denoise": False,
    },
    "clone": {
        "voice_name": True, "control": True,
        "reference_audio": True, "prompt_text": False,
        "transcribe_button": False, "denoise": True,
    },
    "hifi": {
        "voice_name": True, "control": False,
        "reference_audio": True, "prompt_text": True,
        "transcribe_button": True, "denoise": True,
    },
}


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
```

- [ ] **Step 4: Run, confirm pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ui_callbacks.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm_tts_tool/ui_callbacks.py tests/test_ui_callbacks.py
git commit -m "feat(ui): pure callbacks for visibility, voice dropdown, insert helpers"
```

---

## Task 12: Gradio App Wiring

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write `app.py` with CLI + Gradio assembly**

```python
"""Gradio entry point. Thin wiring layer over voxcpm_tts_tool modules."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import gradio as gr

from voxcpm_tts_tool import i18n, model_resolver
from voxcpm_tts_tool.app_state import (
    AppState,
    EPHEMERAL_DEFAULT_VOICE_ID,
    ephemeral_default_voice,
    paths_for,
)
from voxcpm_tts_tool.generation import GenerationError, run_generation
from voxcpm_tts_tool.output_writer import write_output_wav
from voxcpm_tts_tool.transcription import (
    AsrUnavailable,
    SenseVoiceTranscriber,
    load_real_transcriber,
)
from voxcpm_tts_tool.ui_callbacks import (
    field_visibility,
    insert_tag,
    insert_voice_tag,
    voice_dropdown_choices,
)
from voxcpm_tts_tool.voice_library import (
    VoiceLibrary,
    VoiceLibraryError,
)

NON_LANG_TAGS = [
    "laughing", "sigh", "Uhm", "Shh",
    "Question-ah", "Question-ei", "Question-en", "Question-oh",
    "Surprise-wa", "Surprise-yo", "Dissatisfaction-hnn",
]


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

    # Build VoxCPM model singleton.
    import voxcpm
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
    lang_state = gr.State("zh")
    ephemeral = ephemeral_default_voice()

    with gr.Blocks(title="VoxCPM TTS Tool") as demo:
        lang_radio = gr.Radio(["zh", "en"], value="zh", label="Language / 语言")
        startup_md = gr.Markdown("\n".join(f"- {m}" for m in startup_messages))

        # ---- Generation tab ----
        with gr.Tab(i18n.t("tab.generation", "zh")) as tab_gen:
            default_voice_dd = gr.Dropdown(
                choices=voice_dropdown_choices(state.library.list_voices(), lang="zh", ephemeral=ephemeral),
                label=i18n.t("field.default_voice", "zh"),
            )
            script_box = gr.Textbox(label=i18n.t("field.script", "zh"), lines=10)
            with gr.Row():
                voice_picker = gr.Dropdown(
                    choices=[v.name for v in state.library.list_voices()],
                    label=i18n.t("btn.insert_voice", "zh"),
                )
                insert_voice_btn = gr.Button(i18n.t("btn.insert_voice", "zh"))
                tag_picker = gr.Dropdown(choices=NON_LANG_TAGS, label=i18n.t("btn.insert_tag", "zh"))
                insert_tag_btn = gr.Button(i18n.t("btn.insert_tag", "zh"))
            generate_btn = gr.Button(i18n.t("btn.generate", "zh"), variant="primary")
            audio_out = gr.Audio(label="Output", type="filepath")
            log_out = gr.Textbox(label="Log", lines=8)

        # ---- Voice library tab ----
        with gr.Tab(i18n.t("tab.voice_library", "zh")) as tab_lib:
            voice_list = gr.Dataframe(
                headers=["id", "name", "mode"],
                value=[[v.id, v.name, v.mode] for v in state.library.list_voices()],
                interactive=False,
            )
            mode_radio = gr.Radio(
                choices=[("声音设计 / Voice Design", "design"),
                         ("可控克隆 / Controllable Cloning", "clone"),
                         ("极致克隆 / Ultimate Cloning", "hifi")],
                value="design",
                label="Mode",
            )
            name_box = gr.Textbox(label=i18n.t("field.voice_name", "zh"))
            control_box = gr.Textbox(label=i18n.t("field.control", "zh"), visible=True)
            ref_audio = gr.Audio(label=i18n.t("field.reference_audio", "zh"),
                                 type="filepath", visible=False)
            prompt_box = gr.Textbox(label=i18n.t("field.prompt_text", "zh"), visible=False)
            transcribe_btn = gr.Button(i18n.t("btn.transcribe", "zh"), visible=False)
            denoise_box = gr.Checkbox(label=i18n.t("field.denoise", "zh"), visible=False)
            save_btn = gr.Button(i18n.t("btn.save", "zh"), variant="primary")
            delete_btn = gr.Button(i18n.t("btn.delete", "zh"))
            refresh_btn = gr.Button(i18n.t("btn.refresh", "zh"))
            lib_status = gr.Markdown()

        # ---- Usage tab ----
        with gr.Tab(i18n.t("tab.usage", "zh")):
            gr.Markdown(_usage_doc("zh"))

        # ---- Wire callbacks ----
        def _on_mode_change(mode: str):
            vis = field_visibility(mode)
            return (
                gr.update(visible=vis["control"]),
                gr.update(visible=vis["reference_audio"]),
                gr.update(visible=vis["prompt_text"]),
                gr.update(visible=vis["transcribe_button"], interactive=state.transcriber.is_available),
                gr.update(visible=vis["denoise"]),
            )

        mode_radio.change(
            _on_mode_change,
            inputs=mode_radio,
            outputs=[control_box, ref_audio, prompt_box, transcribe_btn, denoise_box],
        )

        def _on_insert_voice(script: str, name: str | None) -> str:
            if not name:
                return script
            return insert_voice_tag(script, voice_name=name)

        insert_voice_btn.click(_on_insert_voice, inputs=[script_box, voice_picker], outputs=script_box)

        def _on_insert_tag(script: str, tag: str | None) -> str:
            if not tag:
                return script
            return insert_tag(script, tag=tag)

        insert_tag_btn.click(_on_insert_tag, inputs=[script_box, tag_picker], outputs=script_box)

        def _on_transcribe(audio_path: str | None) -> str:
            if not audio_path:
                return ""
            try:
                return state.transcriber.transcribe(audio_path)
            except AsrUnavailable as exc:
                return f"[ASR unavailable: {exc}]"

        transcribe_btn.click(_on_transcribe, inputs=ref_audio, outputs=prompt_box)

        def _on_save(mode, name, control, ref, prompt, denoise):
            try:
                state.library.create(
                    name=name,
                    mode=mode,
                    control=control or "",
                    reference_audio_upload=ref,
                    prompt_text=prompt or "",
                    denoise=bool(denoise),
                )
                return _refresh_outputs(state, ephemeral)
            except VoiceLibraryError as exc:
                return _refresh_outputs(state, ephemeral, status=f"❌ {exc}")

        save_btn.click(
            _on_save,
            inputs=[mode_radio, name_box, control_box, ref_audio, prompt_box, denoise_box],
            outputs=[voice_list, voice_picker, default_voice_dd, lib_status],
        )

        def _on_delete(selected_id: str):
            if selected_id and selected_id != EPHEMERAL_DEFAULT_VOICE_ID:
                state.library.delete(selected_id)
            return _refresh_outputs(state, ephemeral)

        delete_btn.click(_on_delete, inputs=default_voice_dd,
                         outputs=[voice_list, voice_picker, default_voice_dd, lib_status])
        refresh_btn.click(lambda: _refresh_outputs(state, ephemeral),
                          outputs=[voice_list, voice_picker, default_voice_dd, lib_status])

        def _on_generate(default_id: str, script: str):
            try:
                default = state.default_voice(default_id)
                result = run_generation(
                    script=script,
                    default_voice=default,
                    library=state.library,
                    model=state.model,
                    audio_root=str(state.paths.root),
                    zipenhancer_loaded=state.zipenhancer_loaded,
                )
                out_path = write_output_wav(
                    result.waveform,
                    sample_rate=result.sample_rate,
                    outputs_dir=state.paths.outputs,
                )
                return str(out_path), "\n".join(result.log)
            except GenerationError as exc:
                return None, f"❌ {exc}"
            except Exception:
                return None, traceback.format_exc()

        # Per spec §Generation Flow: BOTH the queue default AND per-event limit must be 1.
        generate_btn.click(_on_generate, inputs=[default_voice_dd, script_box],
                           outputs=[audio_out, log_out], concurrency_limit=1)

        # Language switch (relabels the most-visible widgets only).
        def _on_lang_change(new_lang: str):
            return (
                new_lang,
                gr.update(label=i18n.t("field.default_voice", new_lang)),
                gr.update(label=i18n.t("field.script", new_lang)),
                gr.update(value=i18n.t("btn.generate", new_lang)),
                gr.update(value=i18n.t("btn.transcribe", new_lang)),
                gr.update(value=i18n.t("btn.save", new_lang)),
                gr.update(value=i18n.t("btn.delete", new_lang)),
                gr.update(value=i18n.t("btn.refresh", new_lang)),
            )

        lang_radio.change(
            _on_lang_change,
            inputs=lang_radio,
            outputs=[lang_state, default_voice_dd, script_box,
                     generate_btn, transcribe_btn, save_btn, delete_btn, refresh_btn],
        )

    return demo


def _refresh_outputs(state: AppState, ephemeral) -> tuple:
    voices = state.library.list_voices()
    return (
        [[v.id, v.name, v.mode] for v in voices],
        gr.update(choices=[v.name for v in voices]),
        gr.update(choices=voice_dropdown_choices(voices, lang="zh", ephemeral=ephemeral)),
        "✅ updated",
    )


def _usage_doc(lang: str) -> str:
    if lang == "en":
        return (
            "## Usage\n\n"
            "- `<voice name>` switches voice for the rest of the line.\n"
            "- A new line resets to the default voice.\n"
            "- Square-bracket tags like `[laughing]` are passed through to the model.\n"
            "- Long text is split on punctuation; bracket tags are atomic.\n"
        )
    return (
        "## 使用说明\n\n"
        "- `<音色名>` 切换当前行后续文本的音色。\n"
        "- 新行自动恢复到默认音色。\n"
        "- 方括号标签如 `[laughing]` 会原样传给模型。\n"
        "- 长文本按标点切分，方括号标签不会被切断。\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VoxCPM TTS Tool")
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--host", default="127.0.0.1")
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
```

- [ ] **Step 2: Smoke-import to catch syntax errors**

```bash
.venv/Scripts/python.exe -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Run the full test suite to confirm nothing regressed**

```bash
.venv/Scripts/python.exe -m pytest -v
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): Gradio UI with three tabs, generation, voice CRUD, language switch"
```

---

## Task 13: README & Manual Smoke

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` with full documentation**

```markdown
# VoxCPM TTS Tool

Standalone Gradio web tool that wraps the [voxcpm](https://pypi.org/project/voxcpm/)
TTS SDK. Manages reusable voices, parses script-style `<voice>` switches,
splits long text, transcribes/denoises reference audio, and saves downloadable
`.wav` output.

See [`docs/superpowers/specs/2026-04-26-voxcpm-tts-tool-design.md`](docs/superpowers/specs/2026-04-26-voxcpm-tts-tool-design.md)
for the full design.

## Requirements

- Python 3.10–3.12 (3.13 currently fails to build the `editdistance` transitive dependency).
- ~6 GB free disk for model downloads (VoxCPM2 + SenseVoiceSmall + ZipEnhancer).
- Network access for first-time model downloads (ModelScope, with HF fallback for VoxCPM2 and SenseVoiceSmall).

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

## Run

```bash
.venv/Scripts/python.exe app.py --port 8808
```

The first launch downloads ~6 GB of model files into `./pretrained_models/`.

CLI flags:
- `--port` (default `8808`)
- `--host` (default `127.0.0.1`)
- `--share` enables Gradio's public tunnel
- `--root` overrides the project root (where `voices/`, `outputs/`, `pretrained_models/` live)

## Test

```bash
.venv/Scripts/python.exe -m pytest -v
```

All tests use fakes; no real model inference, no network.

## Script syntax

```text
<女声>大家好。[laughing] 今天我们介绍 VoxCPM。<男声>下面换一个声音。
这一行没有指定音色，所以回到默认音色。
```

- `<voice name>` switches the active voice for the rest of the **line**.
- A new line resets to the selected default voice.
- `[tag]` (e.g. `[laughing]`) is passed through to VoxCPM as-is.
- Voice names are matched case-insensitively after trimming whitespace.

## Voice modes

- **Voice Design (`design`)**: pure prompt-driven. Provide name + control instruction.
- **Controllable Cloning (`clone`)**: upload reference audio, optional control instruction.
- **Ultimate Cloning (`hifi`)**: upload reference audio + transcript. The Transcribe button
  uses SenseVoiceSmall to fill the transcript automatically (button-triggered, never on upload).

## Data layout

- `voices/voices.json` — voice metadata (atomic write).
- `voices/audio/<id>.wav` — reference audio files.
- `outputs/YYYYMMDD-HHMMSS-mmm.wav` — generated audio.
- `pretrained_models/<name>/` — local model cache. Gitignored.

## Environment variables (optional overrides)

- `VOXCPM_MODEL_DIR` — pre-existing VoxCPM2 model directory.
- `VOXCPM_ASR_MODEL_DIR` — pre-existing SenseVoiceSmall directory.
- `ZIPENHANCER_MODEL_PATH` — pre-existing ZipEnhancer directory.

A pointed-at directory must contain the model's expected files (e.g. `config.json`).
Empty directories or those with only `.gitkeep` are treated as missing.
```

- [ ] **Step 2: Manual smoke checklist (perform once, do not commit results)**

Verify each item:
- [ ] `pip install -e .[dev]` completes.
- [ ] `pytest -v` shows all green.
- [ ] `python app.py --port 8808` starts and prints model resolution status for VoxCPM2, SenseVoiceSmall, ZipEnhancer.
- [ ] UI loads at `http://127.0.0.1:8808`.
- [ ] Language radio switches button labels between zh/en.
- [ ] Voice Design tab does NOT show audio upload, prompt text, transcribe, or denoise.
- [ ] Controllable Cloning shows reference audio + denoise; no prompt/transcribe.
- [ ] Ultimate Cloning shows reference audio + prompt + transcribe + denoise; no control.
- [ ] Upload `.mp3` to clone: rejected with `.wav`-only message.
- [ ] Hifi: upload wav, click Transcribe, prompt_text fills.
- [ ] Generate a 2-line script with `<voice>` switches; download wav and play.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with install, run, syntax, modes, and data layout"
```

---

## Self-Review Notes

- **Spec coverage**: every spec section has at least one task — i18n (Task 2), Voice Library
  (Task 3), Script Semantics (Task 4), Long Text (Task 5), Model Resolution (Task 6),
  Reference-audio transcription (Task 7), Generation Flow + denoise conjunction (Task 8),
  Output filename (Task 9), App state + ephemeral default (Task 10), UI visibility +
  insertion helpers (Task 11), Gradio assembly + language switching + concurrency limits
  (Task 12), README + manual verification (Task 13).
- **Type consistency**: `Voice` field names (`id`, `name`, `mode`, `control`,
  `reference_audio`, `prompt_text`, `denoise`) are stable across all tasks. `VoiceLibrary`
  method names (`create`, `update`, `delete`, `find_by_id`, `find_by_name`, `list_voices`)
  match across Tasks 3, 8, 10, 11, 12.
- **No placeholders**: all code blocks contain runnable code; all commands include expected
  output; no "implement later" or "similar to" references.
- **Gradio 5 concurrency**: spec requires both `Blocks.queue(default_concurrency_limit=1)`
  AND `.click(concurrency_limit=1)` on the generate handler. Task 12 sets both.
