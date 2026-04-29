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


def test_dir_with_git_lfs_pointer_model_file_is_invalid():
    model_dir = Path(__file__).parent / "fixtures" / "lfs_pointer_model"
    assert not mr._has_any(model_dir, ["pytorch_model.bin", "configuration.json"])


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


# ---- preferred shared paths ----

def test_preferred_path_used_before_local_dir_and_download(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXCPM_MODEL_DIR", raising=False)
    shared = tmp_path / "models/openbmb__VoxCPM2"
    local_dir = tmp_path / "VoxCPM-TTS-Tool/pretrained_models/VoxCPM2"
    _seed(shared, ["config.json"])
    _seed(local_dir, ["config.json"])

    path = mr.resolve_voxcpm(
        local_dir,
        preferred_dirs=[shared],
        modelscope_download=lambda *a, **k: pytest.fail("called"),
        hf_download=lambda *a, **k: pytest.fail("called"),
    )

    assert path == shared


def test_invalid_preferred_path_falls_back_to_local_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXCPM_MODEL_DIR", raising=False)
    shared = tmp_path / "models/openbmb__VoxCPM2"
    shared.mkdir(parents=True)
    local_dir = tmp_path / "VoxCPM-TTS-Tool/pretrained_models/VoxCPM2"
    _seed(local_dir, ["config.json"])

    path = mr.resolve_voxcpm(
        local_dir,
        preferred_dirs=[shared],
        modelscope_download=lambda *a, **k: pytest.fail("called"),
        hf_download=lambda *a, **k: pytest.fail("called"),
    )

    assert path == local_dir


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
