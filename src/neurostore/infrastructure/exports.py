"""Writing an export folder to disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_export(dest: Path, signal: np.ndarray, metadata: dict[str, object]) -> Path:
    """Create `dest` with `signal.npy` and `metadata.json`. Fails if `dest` already exists."""
    dest.mkdir(parents=True)
    np.save(dest / "signal.npy", signal, allow_pickle=False)
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return dest
