"""Runtime patch for voxcpm CPU attention-mask shape bug.

Upstream issue: https://github.com/OpenBMB/VoxCPM/issues/71

In `voxcpm/modules/minicpm4/model.py:MiniCPMAttention.forward_step`, the code
builds a 1-D attention mask (`torch.arange(...) <= position_id`) and passes it
to `torch.nn.functional.scaled_dot_product_attention` along with 4-D Q/K/V.
PyTorch's SDPA on CPU rejects the 1-D mask (it can't broadcast onto
`[batch, heads, q_seq, k_seq]`), raising:

    IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)

Fix strategy: wrap `torch.nn.functional.scaled_dot_product_attention` with a
shim that, whenever it sees a 1-D attn_mask, reshapes it to `(1, 1, 1, -1)` so
broadcasting works. The wrapper is a no-op for already-correct mask shapes, so
it's safe even if the user has other torch code in the process.

This is more reliable than rewriting the installed `model.py` source: file
edits get defeated by stale `.pyc` cache (e.g. when Mutagen sync doesn't
preserve mtime), whereas a function replacement takes effect in the running
process immediately.

`apply()` returns True on first call, False on subsequent calls (idempotent).
"""
from __future__ import annotations

_PATCH_FLAG = "_voxcpm_tts_tool_sdpa_patched"


def apply() -> bool:
    """Wrap torch SDPA to expand 1-D attn_mask to 4-D. Idempotent."""
    try:
        import torch.nn.functional as F
    except ImportError:
        return False

    sdpa = getattr(F, "scaled_dot_product_attention", None)
    if sdpa is None:
        return False
    if getattr(sdpa, _PATCH_FLAG, False):
        return False  # already patched

    original = sdpa

    def patched_sdpa(query, key, value, attn_mask=None, *args, **kwargs):
        if attn_mask is not None and hasattr(attn_mask, "ndim") and attn_mask.ndim == 1:
            attn_mask = attn_mask.view(1, 1, 1, -1)
        return original(query, key, value, attn_mask=attn_mask, *args, **kwargs)

    setattr(patched_sdpa, _PATCH_FLAG, True)
    F.scaled_dot_product_attention = patched_sdpa
    return True
