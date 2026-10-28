"""Plain data types shared by every layer."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy as np


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
