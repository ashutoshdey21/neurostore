"""Input rules for the data model. Each function raises ValidationError with a clear message."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from neurostore.domain.errors import ValidationError
from neurostore.domain.models import Event

# Short machine-readable labels only, so these fields cannot quietly hold free text
# such as names or clinical notes.
_LABEL_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def validate_label(value: str, what: str) -> str:
    if not isinstance(value, str) or not _LABEL_RE.fullmatch(value):
        raise ValidationError(
            f"{what} must be 1-64 characters of letters, digits, '_', '.', or '-'; got {value!r}"
        )
    return value


def to_utc(value: datetime, what: str) -> datetime:
    """Convert a timezone-aware datetime to UTC. Naive datetimes are rejected."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{what} must be timezone-aware (e.g. datetime.now(UTC))")
    return value.astimezone(UTC)


def validate_recording(signal: np.ndarray, fs: float, channels: Sequence[str]) -> None:
    if not _is_finite_number(fs) or fs <= 0:
        raise ValidationError(f"sampling rate must be a finite number > 0 Hz; got {fs!r}")
    if not isinstance(signal, np.ndarray) or signal.ndim != 2:
        raise ValidationError("signal must be a 2-D numpy array shaped (channels, samples)")
    if signal.dtype != np.float32:
        raise ValidationError(
            f"signal must be float32; got {signal.dtype} (convert with signal.astype(np.float32))"
        )
    if signal.shape[1] == 0:
        raise ValidationError("signal must contain at least one sample")
    if len(channels) != signal.shape[0]:
        raise ValidationError(
            f"channel count mismatch: {len(channels)} names for {signal.shape[0]} signal rows"
        )
    if any(not isinstance(c, str) or not c for c in channels):
        raise ValidationError("channel names must be non-empty strings")
    duplicates = sorted(name for name, n in Counter(channels).items() if n > 1)
    if duplicates:
        raise ValidationError(f"duplicate channel names: {duplicates}")


def validate_events(events: Sequence[Event], duration: float) -> None:
    for ev in events:
        validate_label(ev.type, "event type")
        if not _is_finite_number(ev.t) or not 0 <= ev.t < duration:
            raise ValidationError(
                f"event {ev.type!r} at t={ev.t}s is outside the recording [0, {duration}s)"
            )
        for key, val in ev.params.items():
            validate_label(key, "event parameter name")
            if not _is_finite_number(val):
                raise ValidationError(f"event parameter {key!r} must be a finite number")
