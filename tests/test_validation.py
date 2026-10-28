from datetime import datetime

import numpy as np
import pytest

from neurostore import Event, NotFoundError, Store, ValidationError
from tests.conftest import CHANNELS, FS, T0, make_signal


@pytest.mark.parametrize("fs", [0, -250.0, float("nan"), float("inf")])
def test_sampling_rate_must_be_positive(store: Store, session_id: str, fs: float) -> None:
    with pytest.raises(ValidationError, match="sampling rate must be a finite number > 0"):
        store.add_recording(session_id, make_signal(), fs, CHANNELS, T0)


def test_channel_count_must_match_signal(store: Store, session_id: str) -> None:
    with pytest.raises(ValidationError, match="2 names for 3 signal rows"):
        store.add_recording(session_id, make_signal(), FS, ["Fz", "Cz"], T0)


def test_duplicate_channel_names_rejected(store: Store, session_id: str) -> None:
    with pytest.raises(ValidationError, match=r"duplicate channel names: \['Cz'\]"):
        store.add_recording(session_id, make_signal(), FS, ["Cz", "Fz", "Cz"], T0)


def test_signal_must_be_2d_float32(store: Store, session_id: str) -> None:
    with pytest.raises(ValidationError, match="must be float32; got float64"):
        store.add_recording(session_id, make_signal().astype(np.float64), FS, CHANNELS, T0)
    with pytest.raises(ValidationError, match="2-D"):
        store.add_recording(session_id, make_signal()[0], FS, CHANNELS, T0)


def test_naive_datetimes_rejected(store: Store, session_id: str) -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        store.add_recording(session_id, make_signal(), FS, CHANNELS, datetime(2026, 1, 1))


@pytest.mark.parametrize("t", [-0.001, 10.0, 12.5])
def test_events_must_fall_inside_recording(store: Store, recording_id: str, t: float) -> None:
    with pytest.raises(ValidationError, match=r"outside the recording \[0, 10.0s\)"):
        store.add_events(recording_id, [Event("tms_pulse", 1.0), Event("tms_pulse", t)])
    assert store.get_events(recording_id) == []  # nothing partially written


def test_free_text_labels_rejected(store: Store) -> None:
    subject = store.add_subject()
    with pytest.raises(ValidationError, match="protocol must be 1-64 characters"):
        store.add_session(subject, T0, protocol="Jane Doe, follow-up visit")


def test_unknown_parent_ids_rejected(store: Store) -> None:
    with pytest.raises(NotFoundError, match="no session"):
        store.add_recording("missing", make_signal(), FS, CHANNELS, T0)
    with pytest.raises(NotFoundError, match="no recording"):
        store.get_window("missing", 0.0, 1.0)
