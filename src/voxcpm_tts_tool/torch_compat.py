"""Compatibility shims for third-party torch integrations."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _is_zipenhancer_checkpoint(path: object, model_dir: Path) -> bool:
    if not isinstance(path, (str, Path)):
        return False

    candidate = Path(path).resolve(strict=False)
    root = model_dir.resolve(strict=False)
    return candidate.name == "pytorch_model.bin" and candidate.is_relative_to(root)


@contextmanager
def zipenhancer_torch_load_compat(model_dir: Path) -> Iterator[None]:
    """Load the trusted ZipEnhancer checkpoint on PyTorch 2.6+.

    ModelScope's ZipEnhancer loader calls ``torch.load`` without
    ``weights_only``. PyTorch 2.6 changed that default to True, but the
    released ZipEnhancer checkpoint needs the legacy unpickler. Keep the
    override scoped to the resolved ZipEnhancer checkpoint path.
    """
    import torch

    original_load = torch.load

    def load_with_zipenhancer_compat(f: Any, *args: Any, **kwargs: Any) -> Any:
        if "weights_only" not in kwargs and _is_zipenhancer_checkpoint(f, model_dir):
            kwargs["weights_only"] = False
        return original_load(f, *args, **kwargs)

    torch.load = load_with_zipenhancer_compat
    try:
        yield
    finally:
        torch.load = original_load
