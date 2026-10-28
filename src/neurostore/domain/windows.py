"""Converting times to sample ranges, and cutting windows and epochs out of a signal.

All functions are pure: they take arrays and numbers and touch no storage.
A time t maps to sample round(t * fs); ranges are half-open [start, end).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neurostore.domain.errors import ValidationError
from neurostore.domain.models import Epochs


def to_sample(t: float, fs: float) -> int:
    """Nearest sample index for a time in seconds.

    Rounding (not truncating) matters: 0.29 * 100 is 28.999999999999996 in floating point,
    which truncates to 28 but rounds to the intended 29.
    """
    return round(t * fs)


def window_range(t_start: float, t_end: float, fs: float, n_samples: int) -> tuple[int, int]:
    """Sample range [start, end) covering t_start <= t < t_end."""
    duration = n_samples / fs
    if not 0 <= t_start < t_end <= duration:
        raise ValidationError(
            f"window [{t_start}, {t_end}) must satisfy 0 <= t_start < t_end <= {duration}"
        )
    return to_sample(t_start, fs), to_sample(t_end, fs)


def channel_rows(available: Sequence[str], requested: Sequence[str] | None) -> list[int]:
    """Row indices for the requested channel names, in the requested order."""
    if requested is None:
        return list(range(len(available)))
    unknown = [c for c in requested if c not in available]
    if unknown:
        raise ValidationError(f"unknown channels {unknown}; available: {list(available)}")
    return [list(available).index(c) for c in requested]


def extract_epochs(
    signal: np.ndarray, fs: float, event_times: Sequence[float], pre_s: float, post_s: float
) -> Epochs:
    """Cut [t - pre_s, t + post_s) around each event time.

    Events whose epoch would run past either end of the signal are skipped and counted.
    """
    if pre_s < 0 or post_s < 0 or pre_s + post_s <= 0:
        raise ValidationError("pre_s and post_s must be >= 0 and not both zero")
    n_pre, n_post = to_sample(pre_s, fs), to_sample(post_s, fs)
    n_channels, n_samples = signal.shape
    segments = []
    for t in event_times:
        centre = to_sample(t, fs)
        start, end = centre - n_pre, centre + n_post
        if start >= 0 and end <= n_samples:
            segments.append(signal[:, start:end])
    if segments:
        data = np.stack(segments)
    else:
        data = np.empty((0, n_channels, n_pre + n_post), dtype=signal.dtype)
    return Epochs(data=data, n_skipped=len(event_times) - len(segments))
