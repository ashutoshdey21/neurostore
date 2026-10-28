from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from neurostore import Event, ValidationError
from neurostore.domain.validation import (
    to_utc,
    validate_events,
    validate_label,
    validate_recording,
)
from tests.sample_data import CHANNELS, DURATION, FS, make_signal


def test_valid_recording_passes() -> None:
    validate_recording(make_signal(), FS, CHANNELS)


@pytest.mark.parametrize("fs", [0, -250.0, float("nan"), float("inf"), True])
def test_sampling_rate_must_be_a_positive_number(fs: float) -> None:
    with pytest.raises(ValidationError, match="sampling rate must be a finite number > 0"):
        validate_recording(make_signal(), fs, CHANNELS)


@pytest.mark.parametrize(
    ("signal", "channels", "message"),
    [
        (make_signal(), ["Fz", "Cz"], "2 names for 3 signal rows"),
        (make_signal(), ["Cz", "Fz", "Cz"], r"duplicate channel names: \['Cz'\]"),
        (make_signal(), ["Fz", "", "Pz"], "non-empty strings"),
        (make_signal().astype(np.float64), CHANNELS, "must be float32; got float64"),
        (make_signal()[0], CHANNELS, "2-D"),
        (np.empty((3, 0), np.float32), CHANNELS, "at least one sample"),
    ],
)
def test_invalid_recordings_are_rejected(
    signal: np.ndarray, channels: list[str], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        validate_recording(signal, FS, channels)


def test_events_inside_recording_pass() -> None:
    validate_events([Event("tms_pulse", 0.0), Event("stim_off", DURATION - 0.01)], DURATION)


@pytest.mark.parametrize("t", [-0.001, DURATION, DURATION + 2.5, float("nan")])
def test_events_must_fall_inside_recording(t: float) -> None:
    with pytest.raises(ValidationError, match=r"outside the recording \[0, 10.0s\)"):
        validate_events([Event("tms_pulse", t)], DURATION)


def test_event_parameters_must_be_finite_numbers() -> None:
    with pytest.raises(ValidationError, match="'intensity_pct' must be a finite number"):
        validate_events([Event("tms_pulse", 1.0, {"intensity_pct": float("inf")})], DURATION)


@pytest.mark.parametrize("label", ["Jane Doe", "", "a" * 65, "visit/2"])
def test_labels_reject_free_text(label: str) -> None:
    with pytest.raises(ValidationError, match="protocol must be 1-64 characters"):
        validate_label(label, "protocol")


def test_to_utc_converts_other_timezones() -> None:
    local = datetime(2026, 1, 15, 1, 30, tzinfo=timezone(timedelta(hours=-8)))
    assert to_utc(local, "start").isoformat() == "2026-01-15T09:30:00+00:00"


def test_to_utc_rejects_naive_datetimes() -> None:
    with pytest.raises(ValidationError, match="start must be timezone-aware"):
        to_utc(datetime(2026, 1, 1), "start")
