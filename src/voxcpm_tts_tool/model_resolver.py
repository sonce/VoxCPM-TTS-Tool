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
