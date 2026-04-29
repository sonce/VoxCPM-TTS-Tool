"""Tests for the voxcpm CPU attention-mask SDPA shim.

Requires torch. Skipped on environments without torch (e.g. system Python where
the project venv hasn't been installed). The patch and these tests exercise
behavior that only matters at runtime with a real model loaded.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from voxcpm_tts_tool import voxcpm_patch  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_sdpa():
    """Snapshot/restore F.scaled_dot_product_attention so tests don't leak state."""
    original = F.scaled_dot_product_attention
    yield
    F.scaled_dot_product_attention = original


def test_apply_first_time_returns_true():
    assert voxcpm_patch.apply() is True


def test_apply_is_idempotent():
    voxcpm_patch.apply()
    assert voxcpm_patch.apply() is False
    assert voxcpm_patch.apply() is False


def test_patched_sdpa_passes_through_4d_mask_unchanged():
    """A correctly-shaped mask must be forwarded without reshape."""
    voxcpm_patch.apply()

    # 4D q/k/v that SDPA can accept directly.
    q = torch.zeros(1, 2, 1, 4, dtype=torch.float32)
    k = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
    v = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
    mask4d = torch.ones(1, 1, 1, 3, dtype=torch.bool)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask4d)
    assert out.shape == (1, 2, 1, 4)


def test_patched_sdpa_expands_1d_mask_for_voxcpm_case():
    """The exact failure mode from voxcpm: 1D mask + 4D Q/K/V on CPU."""
    voxcpm_patch.apply()

    # Mirror voxcpm's MiniCPMAttention.forward_step shapes.
    bsz, num_heads, head_dim = 1, 4, 8
    q = torch.zeros(bsz, num_heads, 1, head_dim, dtype=torch.float32)
    k_cache = torch.zeros(bsz, num_heads, 5, head_dim, dtype=torch.float32)
    v_cache = torch.zeros(bsz, num_heads, 5, head_dim, dtype=torch.float32)

    # The buggy 1D mask: torch.arange(5) <= 2  -> [True, True, True, False, False]
    position_id = 2
    mask_1d = torch.arange(k_cache.size(2)) <= position_id

    # Without the patch this would raise IndexError. With the patch it works.
    out = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=mask_1d)
    assert out.shape == (bsz, num_heads, 1, head_dim)


def test_patched_sdpa_handles_no_mask():
    """attn_mask=None must still work (most common case)."""
    voxcpm_patch.apply()

    q = torch.zeros(1, 2, 1, 4, dtype=torch.float32)
    k = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
    v = torch.zeros(1, 2, 3, 4, dtype=torch.float32)

    out = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == (1, 2, 1, 4)
