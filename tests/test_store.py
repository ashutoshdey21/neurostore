"""Tests for the public Store API.

Rules and arithmetic are tested in tests/domain and storage in tests/infrastructure.
These tests check what Store adds on top: wiring the layers together, access control,
the audit trail, and cleanup when a write fails.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from neurostore import (
    AccessDeniedError,
    ChecksumError,
    Event,
    NotFoundError,
    Store,
    ValidationError,
)
from tests.sample_data import CHANNELS, DURATION, FS, T0, make_signal


def test_write_then_read_returns_identical_data(store: Store, recording_id: str) -> None:
    events = [Event("stim_on", 1.0), Event("tms_pulse", 2.5, {"intensity_pct": 110.0})]
    store.add_events(recording_id, events)

    assert [r.id for r in store.list_recordings()] == [recording_id]
    np.testing.assert_array_equal(store.get_window(recording_id, 0.0, DURATION), make_signal())
    assert store.get_events(recording_id) == events


def test_get_window_applies_time_range_and_channels(store: Store, recording_id: str) -> None:
    out = store.get_window(recording_id, 0.1, 0.3, channels=["Pz", "Fz"])
    np.testing.assert_array_equal(out, make_signal()[[2, 0], 10:30])


def test_get_epochs_uses_only_the_requested_event_type(store: Store, recording_id: str) -> None:
    store.add_events(recording_id, [Event("stim_on", 5.0), Event("tms_pulse", 2.0)])
    data, _ = store.get_epochs(recording_id, "tms_pulse", pre_s=0.1, post_s=0.2)
    assert data.shape == (1, len(CHANNELS), 30)
    np.testing.assert_array_equal(data[0], make_signal()[:, 190:220])


def test_invalid_events_are_rejected_and_nothing_is_written(
    store: Store, recording_id: str
) -> None:
    with pytest.raises(ValidationError):
        store.add_events(recording_id, [Event("tms_pulse", 1.0), Event("tms_pulse", DURATION)])
    assert store.get_events(recording_id) == []


def test_invalid_recording_leaves_no_file_or_row(store: Store, session_id: str) -> None:
    with pytest.raises(ValidationError, match="channel count mismatch"):
        store.add_recording(session_id, make_signal(), FS, ["Fz"], T0)
    assert store.list_recordings() == []
    assert list((store.root / "signals").glob("*.npy")) == []


def test_signal_file_is_removed_if_database_write_fails(store: Store, session_id: str) -> None:
    with (
        patch.object(store._db, "insert_recording", side_effect=RuntimeError("disk full")),
        pytest.raises(RuntimeError),
    ):
        store.add_recording(session_id, make_signal(), FS, CHANNELS, T0)
    assert list((store.root / "signals").glob("*.npy")) == []


def test_unknown_ids_raise_not_found(store: Store) -> None:
    with pytest.raises(NotFoundError, match="no subject"):
        store.add_session("missing", T0, "sp_tms")
    with pytest.raises(NotFoundError, match="no session"):
        store.add_recording("missing", make_signal(), FS, CHANNELS, T0)
    with pytest.raises(NotFoundError, match="no recording"):
        store.get_window("missing", 0.0, 1.0)


def test_tampered_signal_is_refused_by_every_read(
    tmp_path: Path, store: Store, recording_id: str
) -> None:
    [path] = (store.root / "signals").glob("*.npy")
    data = np.load(path)
    data[0, 0] += 1.0
    np.save(path, data)

    with pytest.raises(ChecksumError):
        store.get_window(recording_id, 0.0, 1.0)
    with pytest.raises(ChecksumError):
        store.get_epochs(recording_id, "tms_pulse", 0.1, 0.1)
    with pytest.raises(ChecksumError):
        store.export_deidentified(recording_id, tmp_path / "export")


def test_export_writes_signal_and_events_under_a_new_id(
    tmp_path: Path, store: Store, recording_id: str
) -> None:
    store.add_events(recording_id, [Event("tms_pulse", 2.0)])

    dest = store.export_deidentified(recording_id, tmp_path / "export")

    np.testing.assert_array_equal(np.load(dest / "signal.npy"), make_signal())
    meta = json.loads((dest / "metadata.json").read_text())
    assert meta["export_id"] == dest.name != recording_id
    assert [e["t"] for e in meta["events"]] == [2.0]


# -- access control and audit ----------------------------------------------------------


def test_read_only_store_can_read_but_not_write(store: Store, recording_id: str) -> None:
    with Store(store.root, actor="analyst") as read_only:
        assert len(read_only.list_recordings()) == 1
        with pytest.raises(AccessDeniedError, match="add_subject requires a read-write store"):
            read_only.add_subject()
        with pytest.raises(AccessDeniedError):
            read_only.add_events(recording_id, [Event("tms_pulse", 1.0)])
    assert store.get_events(recording_id) == []


def test_read_only_cannot_create_a_store(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError, match="no store"):
        Store(tmp_path / "nothing", actor="analyst")
    assert not (tmp_path / "nothing").exists()


def test_actor_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="actor must be a non-empty string"):
        Store(tmp_path, actor="  ")


def test_audit_log_records_every_read_write_and_denial(
    tmp_path: Path, store: Store, recording_id: str
) -> None:
    store.add_events(recording_id, [Event("tms_pulse", 2.0)])
    store.get_window(recording_id, 0.0, 1.0)
    store.get_epochs(recording_id, "tms_pulse", 0.1, 0.1)
    store.export_deidentified(recording_id, tmp_path / "export")
    with Store(store.root, actor="analyst") as read_only:
        read_only.get_events(recording_id)
        read_only.list_recordings()
        with pytest.raises(AccessDeniedError):
            read_only.add_subject()

    log = store.audit_log()
    [info] = store.list_recordings()
    assert [(actor, action.split(":")[0], obj) for actor, action, obj, _ in log] == [
        ("tester", "add_subject", info.subject_id),
        ("tester", "add_session", info.session_id),
        ("tester", "add_recording", recording_id),
        ("tester", "add_events", recording_id),
        ("tester", "get_window", recording_id),
        ("tester", "get_epochs", recording_id),
        ("tester", "export_deidentified", recording_id),
        ("analyst", "get_events", recording_id),
        ("analyst", "list_recordings", None),
        ("analyst", "add_subject", None),
    ]
    assert log[-1][1] == "add_subject:denied"
    assert all(at.endswith("+00:00") for *_, at in log)
