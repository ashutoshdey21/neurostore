"""SQLite storage for metadata, events, and the audit log.

Write methods do not commit on their own. Callers group them with `transaction()`, so a
change and its audit entry are saved together or not at all.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from neurostore.domain.models import Event, RecordingInfo

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    id TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES subjects(id),
    start_time TEXT NOT NULL,
    protocol   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recordings (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    channels   TEXT NOT NULL,
    fs         REAL NOT NULL CHECK (fs > 0),
    start_time TEXT NOT NULL,
    n_samples  INTEGER NOT NULL CHECK (n_samples > 0),
    path       TEXT NOT NULL,
    sha256     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES recordings(id),
    type         TEXT NOT NULL,
    t            REAL NOT NULL,
    params       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_recording ON events(recording_id, type);
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,
    object_id TEXT,
    at        TEXT NOT NULL
);
"""

_ID_TABLES = frozenset({"subjects", "sessions", "recordings"})

AuditRow = tuple[str, str, str | None, str]


class Database:
    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit everything written inside the block, or roll it all back on error."""
        with self._conn:
            yield

    # -- writes (call inside transaction()) -----------------------------------------

    def insert_subject(self, subject_id: str) -> None:
        self._conn.execute("INSERT INTO subjects (id) VALUES (?)", (subject_id,))

    def insert_session(
        self, session_id: str, subject_id: str, start_time: datetime, protocol: str
    ) -> None:
        self._conn.execute(
            "INSERT INTO sessions (id, subject_id, start_time, protocol) VALUES (?, ?, ?, ?)",
            (session_id, subject_id, start_time.isoformat(), protocol),
        )

    def insert_recording(
        self,
        recording_id: str,
        session_id: str,
        channels: Sequence[str],
        fs: float,
        start_time: datetime,
        n_samples: int,
        path: str,
        sha256: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO recordings (id, session_id, channels, fs, start_time, n_samples,"
            " path, sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                recording_id,
                session_id,
                json.dumps(list(channels)),
                float(fs),
                start_time.isoformat(),
                n_samples,
                path,
                sha256,
            ),
        )

    def insert_events(self, recording_id: str, events: Sequence[Event]) -> None:
        self._conn.executemany(
            "INSERT INTO events (recording_id, type, t, params) VALUES (?, ?, ?, ?)",
            [(recording_id, e.type, float(e.t), json.dumps(e.params)) for e in events],
        )

    def insert_audit(self, actor: str, action: str, object_id: str | None, at: datetime) -> None:
        self._conn.execute(
            "INSERT INTO audit_log (actor, action, object_id, at) VALUES (?, ?, ?, ?)",
            (actor, action, object_id, at.isoformat()),
        )

    # -- reads -----------------------------------------------------------------------

    def exists(self, table: str, object_id: str) -> bool:
        if table not in _ID_TABLES:
            raise ValueError(f"unknown table {table!r}")
        query = f"SELECT 1 FROM {table} WHERE id = ?"
        return self._conn.execute(query, (object_id,)).fetchone() is not None

    def find_recordings(
        self,
        recording_id: str | None = None,
        subject_id: str | None = None,
        protocol: str | None = None,
    ) -> list[RecordingInfo]:
        """Recordings matching every filter that is not None, oldest first."""
        rows = self._conn.execute(
            "SELECT r.id, r.session_id, s.subject_id, s.protocol, r.channels, r.fs,"
            " r.start_time, r.n_samples"
            " FROM recordings r JOIN sessions s ON r.session_id = s.id"
            " WHERE (:rid IS NULL OR r.id = :rid)"
            " AND (:subject IS NULL OR s.subject_id = :subject)"
            " AND (:protocol IS NULL OR s.protocol = :protocol)"
            " ORDER BY r.start_time, r.id",
            {"rid": recording_id, "subject": subject_id, "protocol": protocol},
        ).fetchall()
        return [
            RecordingInfo(
                id=rid,
                session_id=sid,
                subject_id=subject,
                protocol=proto,
                channels=tuple(json.loads(channels)),
                fs=fs,
                start_time=datetime.fromisoformat(start),
                n_samples=n,
            )
            for rid, sid, subject, proto, channels, fs, start, n in rows
        ]

    def signal_location(self, recording_id: str) -> tuple[str, str]:
        """(relative path, sha256) of a recording's signal file."""
        row = self._conn.execute(
            "SELECT path, sha256 FROM recordings WHERE id = ?", (recording_id,)
        ).fetchone()
        if row is None:
            raise KeyError(recording_id)
        path, checksum = row
        return str(path), str(checksum)

    def events(self, recording_id: str, event_type: str | None = None) -> list[Event]:
        """A recording's events in time order, optionally of one type."""
        rows = self._conn.execute(
            "SELECT type, t, params FROM events WHERE recording_id = :rid"
            " AND (:type IS NULL OR type = :type) ORDER BY t, id",
            {"rid": recording_id, "type": event_type},
        ).fetchall()
        return [Event(type=kind, t=t, params=json.loads(params)) for kind, t, params in rows]

    def audit_rows(self) -> list[AuditRow]:
        rows = self._conn.execute(
            "SELECT actor, action, object_id, at FROM audit_log ORDER BY id"
        ).fetchall()
        return [tuple(r) for r in rows]
