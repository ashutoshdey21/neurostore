import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from neurostore import Event
from neurostore.infrastructure.database import Database
from tests.sample_data import CHANNELS, FS, N_SAMPLES, T0


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


def add_recording(db: Database, recording_id: str, subject_id: str, protocol: str) -> None:
    with db.transaction():
        if not db.exists("subjects", subject_id):
            db.insert_subject(subject_id)
        db.insert_session(f"ses-{recording_id}", subject_id, T0, protocol)
        db.insert_recording(
            recording_id, f"ses-{recording_id}", CHANNELS, FS, T0, N_SAMPLES, "p.npy", "abc"
        )


def test_recording_round_trip(db: Database) -> None:
    add_recording(db, "rec1", "sub1", "sp_tms")
    [info] = db.find_recordings(recording_id="rec1")
    assert info.subject_id == "sub1"
    assert info.protocol == "sp_tms"
    assert info.channels == CHANNELS
    assert (info.fs, info.n_samples, info.start_time) == (FS, N_SAMPLES, T0)
    assert db.signal_location("rec1") == ("p.npy", "abc")


def test_find_recordings_filters_combine(db: Database) -> None:
    add_recording(db, "rec1", "sub1", "sp_tms")
    add_recording(db, "rec2", "sub2", "sp_tms")
    add_recording(db, "rec3", "sub2", "rtms_10hz")

    def ids(**filters: str) -> list[str]:
        return [r.id for r in db.find_recordings(**filters)]

    assert ids() == ["rec1", "rec2", "rec3"]
    assert ids(subject_id="sub2") == ["rec2", "rec3"]
    assert ids(protocol="sp_tms") == ["rec1", "rec2"]
    assert ids(subject_id="sub2", protocol="sp_tms") == ["rec2"]
    assert ids(recording_id="missing") == []


def test_events_come_back_in_time_order_and_filter_by_type(db: Database) -> None:
    add_recording(db, "rec1", "sub1", "sp_tms")
    later, earlier = Event("tms_pulse", 5.0, {"intensity_pct": 110.0}), Event("stim_on", 1.0)
    with db.transaction():
        db.insert_events("rec1", [later, earlier])
    assert db.events("rec1") == [earlier, later]
    assert db.events("rec1", "tms_pulse") == [later]


def test_transaction_rolls_back_everything_on_error(db: Database) -> None:
    with pytest.raises(RuntimeError), db.transaction():
        db.insert_subject("sub1")
        db.insert_audit("tester", "add_subject", "sub1", datetime.now(UTC))
        raise RuntimeError("fail mid-write")
    assert not db.exists("subjects", "sub1")
    assert db.audit_rows() == []


def test_foreign_keys_are_enforced(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError), db.transaction():
        db.insert_session("ses1", "no-such-subject", T0, "sp_tms")


def test_audit_rows_are_kept_in_insertion_order(db: Database) -> None:
    at = datetime(2026, 1, 1, tzinfo=UTC)
    with db.transaction():
        db.insert_audit("alice", "add_subject", "s1", at)
        db.insert_audit("bob", "list_recordings", None, at)
    assert db.audit_rows() == [
        ("alice", "add_subject", "s1", at.isoformat()),
        ("bob", "list_recordings", None, at.isoformat()),
    ]


def test_data_survives_reopening(tmp_path: Path, db: Database) -> None:
    add_recording(db, "rec1", "sub1", "sp_tms")
    reopened = Database(tmp_path / "test.db")
    assert [r.id for r in reopened.find_recordings()] == ["rec1"]
    reopened.close()


def test_exists_only_accepts_known_tables(db: Database) -> None:
    with pytest.raises(ValueError, match="unknown table"):
        db.exists("audit_log", "1")
