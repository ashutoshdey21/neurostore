"""Small synthetic recording shared by all tests."""

from datetime import UTC, datetime

import numpy as np

FS = 100.0
CHANNELS = ("Fz", "Cz", "Pz")
N_SAMPLES = 1000  # 10 s at 100 Hz
DURATION = N_SAMPLES / FS
T0 = datetime(2026, 1, 15, 9, 30, tzinfo=UTC)


def make_signal() -> np.ndarray:
    """Each value encodes its position: channel * 10_000 + sample index."""
    rows = np.arange(len(CHANNELS))[:, None] * 10_000
    return (rows + np.arange(N_SAMPLES)[None, :]).astype(np.float32)
