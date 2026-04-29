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
