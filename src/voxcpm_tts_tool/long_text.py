"""Punctuation-aware splitter and waveform concatenation.

See spec §Long Text for splitting rules. Square-bracket tags are indivisible;
sentence terminators are preferred boundaries; commas are a fallback when a
single sentence exceeds the per-chunk character budget.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

_SENTENCE_TERMINATORS = "。！？；…!?.;"
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
    buf_len = 0  # running length so the comma-budget check stays O(1) per token

    def flush() -> None:
        nonlocal buf_len
        if buf:
            chunks.append("".join(buf))
            buf.clear()
            buf_len = 0

    for i, tok in enumerate(tokens):
        buf.append(tok)
        buf_len += len(tok)
        # Prefer sentence terminator boundaries.
        if len(tok) == 1 and tok in _SENTENCE_TERMINATORS:
            prev_tok = tokens[i - 1] if i > 0 else ""
            next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
            if tok == "." and prev_tok.isdigit() and next_tok.isdigit():
                continue
            flush()
            continue
        # Comma fallback only if buffer has overflowed budget.
        if len(tok) == 1 and tok in _COMMA_BOUNDARIES and buf_len >= char_budget:
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
