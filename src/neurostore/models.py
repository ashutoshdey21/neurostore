"""Plain data types, errors, and input validation for neurostore."""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import NamedTuple

import numpy as np

# Short machine-readable labels only, so these fields cannot quietly hold free text
# such as names or clinical notes.
_LABEL_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class NeurostoreError(Exception):
    """Base class for all neurostore errors."""


class ValidationError(NeurostoreError, ValueError):
    """Input data does not satisfy the data model."""


class AccessDeniedError(NeurostoreError, PermissionError):
    """The store's role does not permit the requested action."""


class ChecksumError(NeurostoreError):
    """A signal file on disk does not match the checksum recorded at write time."""


class NotFoundError(NeurostoreError, KeyError):
    """A referenced subject, session, or recording does not exist."""


class Role(enum.Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class Event:
    """A marker within a recording, e.g. a TMS pulse.

    `t` is in seconds from the start of the recording.
    """

    type: str
    t: float
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordingInfo:
    """Metadata for a stored recording (the signal itself is loaded separately)."""

    id: str
    session_id: str
    subject_id: str
    protocol: str
    channels: tuple[str, ...]
    fs: float
    start_time: datetime
    n_samples: int

    @property
    def duration(self) -> float:
        """Recording length in seconds."""
        return self.n_samples / self.fs


class Epochs(NamedTuple):
    """Event-locked segments: `data` is (n_events, channels, samples)."""

    data: np.ndarray
    n_skipped: int


def validate_label(value: str, what: str) -> str:
    if not isinstance(value, str) or not _LABEL_RE.fullmatch(value):
        raise ValidationError(
            f"{what} must be 1-64 characters of letters, digits, '_', '.', or '-'; got {value!r}"
        )
    return value


def to_utc(value: datetime, what: str) -> datetime:
    """Normalise a timezone-aware datetime to UTC. Naive datetimes are rejected."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{what} must be timezone-aware (e.g. datetime.now(UTC))")
    return value.astimezone(UTC)


def validate_recording(signal: np.ndarray, fs: float, channels: Sequence[str]) -> None:
    if not (isinstance(fs, int | float) and math.isfinite(fs) and fs > 0):
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
    duplicates = sorted({c for c in channels if list(channels).count(c) > 1})
    if duplicates:
        raise ValidationError(f"duplicate channel names: {duplicates}")


def validate_events(events: Sequence[Event], duration: float) -> None:
    for ev in events:
        validate_label(ev.type, "event type")
        if not math.isfinite(ev.t) or not 0 <= ev.t < duration:
            raise ValidationError(
                f"event {ev.type!r} at t={ev.t}s is outside the recording [0, {duration}s)"
            )
        for key, val in ev.params.items():
            validate_label(key, "event parameter name")
            if not isinstance(val, int | float) or not math.isfinite(val):
                raise ValidationError(f"event parameter {key!r} must be a finite number")
