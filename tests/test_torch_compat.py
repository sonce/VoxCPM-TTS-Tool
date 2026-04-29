from pathlib import Path


def test_zipenhancer_torch_load_forces_weights_only_false(monkeypatch):
    from voxcpm_tts_tool.torch_compat import zipenhancer_torch_load_compat

    calls = []

    def fake_load(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        return {"generator": {}}

    import torch

    monkeypatch.setattr(torch, "load", fake_load)
    model_dir = Path("models") / "ZipEnhancer"
    checkpoint = model_dir / "pytorch_model.bin"

    with zipenhancer_torch_load_compat(model_dir):
        torch.load(str(checkpoint), map_location="cpu")

    assert calls == [(str(checkpoint), (), {"map_location": "cpu", "weights_only": False})]


def test_zipenhancer_torch_load_leaves_explicit_weights_only(monkeypatch):
    from voxcpm_tts_tool.torch_compat import zipenhancer_torch_load_compat

    calls = []

    def fake_load(path, *args, **kwargs):
        calls.append(kwargs)
        return {}

    import torch

    monkeypatch.setattr(torch, "load", fake_load)

    model_dir = Path("models") / "ZipEnhancer"
    with zipenhancer_torch_load_compat(model_dir):
        torch.load(model_dir / "pytorch_model.bin", weights_only=True)

    assert calls == [{"weights_only": True}]


def test_zipenhancer_torch_load_leaves_other_paths_unchanged(monkeypatch):
    from voxcpm_tts_tool.torch_compat import zipenhancer_torch_load_compat

    calls = []

    def fake_load(path, *args, **kwargs):
        calls.append(kwargs)
        return {}

    import torch

    monkeypatch.setattr(torch, "load", fake_load)

    with zipenhancer_torch_load_compat(Path("models") / "ZipEnhancer"):
        torch.load(Path("other") / "pytorch_model.bin", map_location="cpu")

    assert calls == [{"map_location": "cpu"}]
