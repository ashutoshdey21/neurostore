"""The Store: SQLite for metadata, events, and audit log; checksummed .npy files for signals."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neurostore.models import (
    AccessDeniedError,
    ChecksumError,
    Epochs,
    Event,
    NotFoundError,
    RecordingInfo,
    Role,
    ValidationError,
    to_utc,
    validate_events,
    validate_label,
    validate_recording,
)

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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Store:
    """A directory holding `neurostore.db` plus a `signals/` folder of .npy files.

    `actor` is recorded in the audit log for every read and write. `role` is
    self-declared by the caller; it guards against accidental writes, not against
    a malicious user (see README, "What I would change for real clinical data").
    """

    def __init__(self, root: Path, actor: str, role: Role = Role.READ_ONLY) -> None:
        if not isinstance(actor, str) or not actor.strip():
            raise ValidationError("actor must be a non-empty string")
        self.root = Path(root)
        self.actor = actor
        self.role = role
        if role is Role.READ_WRITE:
            (self.root / "signals").mkdir(parents=True, exist_ok=True)
        elif not (self.root / "neurostore.db").exists():
            raise NotFoundError(f"no store at {self.root} (open read-write once to create it)")
        self._db = sqlite3.connect(self.root / "neurostore.db")
        self._db.execute("PRAGMA foreign_keys = ON")
        with self._db:
            self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- access control and audit -------------------------------------------------

    def _audit(self, action: str, object_id: str | None) -> None:
        self._db.execute(
            "INSERT INTO audit_log (actor, action, object_id, at) VALUES (?, ?, ?, ?)",
            (self.actor, action, object_id, datetime.now(UTC).isoformat()),
        )

    def _require_write(self, action: str, object_id: str | None) -> None:
        if self.role is not Role.READ_WRITE:
            with self._db:
                self._audit(f"{action}:denied", object_id)
            raise AccessDeniedError(f"{action} requires a read-write store; role is {self.role}")

    def audit_log(self) -> list[tuple[str, str, str | None, str]]:
        """Return (actor, action, object_id, utc_time) rows, oldest first."""
        rows = self._db.execute(
            "SELECT actor, action, object_id, at FROM audit_log ORDER BY id"
        ).fetchall()
        return [tuple(r) for r in rows]

    # -- writes ----------------------------------------------------------------------

    def add_subject(self) -> str:
        """Create a subject with a random pseudonymous ID. No other attributes are stored."""
        self._require_write("add_subject", None)
        subject_id = str(uuid.uuid4())
        with self._db:
            self._db.execute("INSERT INTO subjects (id) VALUES (?)", (subject_id,))
            self._audit("add_subject", subject_id)
        return subject_id

    def add_session(self, subject_id: str, start_time: datetime, protocol: str) -> str:
        self._require_write("add_session", subject_id)
        start = to_utc(start_time, "session start_time")
        validate_label(protocol, "protocol")
        self._ensure_exists("subjects", subject_id)
        session_id = str(uuid.uuid4())
        with self._db:
            self._db.execute(
                "INSERT INTO sessions (id, subject_id, start_time, protocol) VALUES (?, ?, ?, ?)",
                (session_id, subject_id, start.isoformat(), protocol),
            )
            self._audit("add_session", session_id)
        return session_id

    def add_recording(
        self,
        session_id: str,
        signal: np.ndarray,
        fs: float,
        channels: Sequence[str],
        start_time: datetime,
    ) -> str:
        """Store a (channels x samples) float32 signal and return the new recording ID."""
        self._require_write("add_recording", session_id)
        validate_recording(signal, fs, channels)
        start = to_utc(start_time, "recording start_time")
        self._ensure_exists("sessions", session_id)

        recording_id = str(uuid.uuid4())
        rel_path = Path("signals") / f"{recording_id}.npy"
        buf = io.BytesIO()
        np.save(buf, signal, allow_pickle=False)
        data = buf.getvalue()
        (self.root / rel_path).write_bytes(data)
        try:
            with self._db:
                self._db.execute(
                    "INSERT INTO recordings (id, session_id, channels, fs, start_time,"
                    " n_samples, path, sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        recording_id,
                        session_id,
                        json.dumps(list(channels)),
                        float(fs),
                        start.isoformat(),
                        signal.shape[1],
                        rel_path.as_posix(),
                        _sha256(data),
                    ),
                )
                self._audit("add_recording", recording_id)
        except BaseException:
            (self.root / rel_path).unlink(missing_ok=True)
            raise
        return recording_id

    def add_events(self, recording_id: str, events: Iterable[Event]) -> int:
        """Attach events to a recording. All are validated before any are written."""
        self._require_write("add_events", recording_id)
        events = list(events)
        validate_events(events, self._info(recording_id).duration)
        with self._db:
            self._db.executemany(
                "INSERT INTO events (recording_id, type, t, params) VALUES (?, ?, ?, ?)",
                [(recording_id, e.type, float(e.t), json.dumps(e.params)) for e in events],
            )
            self._audit("add_events", recording_id)
        return len(events)

    # -- reads -----------------------------------------------------------------------

    def list_recordings(
        self, subject_id: str | None = None, protocol: str | None = None
    ) -> list[RecordingInfo]:
        query = (
            "SELECT r.id, r.session_id, s.subject_id, s.protocol, r.channels, r.fs,"
            " r.start_time, r.n_samples FROM recordings r JOIN sessions s ON r.session_id = s.id"
            " WHERE (? IS NULL OR s.subject_id = ?) AND (? IS NULL OR s.protocol = ?)"
            " ORDER BY r.start_time, r.id"
        )
        rows = self._db.execute(query, (subject_id, subject_id, protocol, protocol)).fetchall()
        with self._db:
            self._audit("list_recordings", subject_id)
        return [self._row_to_info(r) for r in rows]

    def get_events(self, recording_id: str, event_type: str | None = None) -> list[Event]:
        """Return a recording's events in time order, optionally filtered by type."""
        self._info(recording_id)
        with self._db:
            self._audit("get_events", recording_id)
        return self._events(recording_id, event_type)

    def get_window(
        self,
        recording_id: str,
        t_start: float,
        t_end: float,
        channels: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return samples with t_start <= t < t_end as (channels, samples).

        Times are seconds from recording start and are rounded to the nearest
        sample, so the window covers sample indices [round(t_start*fs), round(t_end*fs)).
        """
        info = self._info(recording_id)
        if not 0 <= t_start < t_end <= info.duration:
            raise ValidationError(
                f"window [{t_start}, {t_end}) must satisfy 0 <= t_start < t_end <= {info.duration}"
            )
        rows = self._channel_indices(info, channels)
        signal = self._load_signal(recording_id)
        i0, i1 = round(t_start * info.fs), round(t_end * info.fs)
        with self._db:
            self._audit("get_window", recording_id)
        return signal[rows, i0:i1]

    def get_epochs(self, recording_id: str, event_type: str, pre_s: float, post_s: float) -> Epochs:
        """Cut a fixed-length window around each event of `event_type`.

        Each epoch spans [event - pre_s, event + post_s). Events whose window would
        extend past either end of the recording are skipped and counted in `n_skipped`.
        """
        if pre_s < 0 or post_s < 0 or pre_s + post_s <= 0:
            raise ValidationError("pre_s and post_s must be >= 0 and not both zero")
        info = self._info(recording_id)
        signal = self._load_signal(recording_id)
        n_pre, n_post = round(pre_s * info.fs), round(post_s * info.fs)
        segments, skipped = [], 0
        for ev in self._events(recording_id, event_type):
            centre = round(ev.t * info.fs)
            lo, hi = centre - n_pre, centre + n_post
            if lo < 0 or hi > info.n_samples:
                skipped += 1
                continue
            segments.append(signal[:, lo:hi])
        with self._db:
            self._audit("get_epochs", recording_id)
        shape = (0, len(info.channels), n_pre + n_post)
        data = np.stack(segments) if segments else np.empty(shape, dtype=np.float32)
        return Epochs(data=data, n_skipped=skipped)

    def export_deidentified(self, recording_id: str, out_dir: Path) -> Path:
        """Write signal + events under a fresh export ID, with no internal IDs or paths.

        Absolute start times are also left out; event times stay relative to
        recording start. Returns the directory written.
        """
        info = self._info(recording_id)
        signal = self._load_signal(recording_id)
        export_id = str(uuid.uuid4())
        dest = Path(out_dir) / export_id
        dest.mkdir(parents=True)
        np.save(dest / "signal.npy", signal, allow_pickle=False)
        meta = {
            "export_id": export_id,
            "channels": list(info.channels),
            "fs": info.fs,
            "n_samples": info.n_samples,
            "events": [
                {"type": e.type, "t": e.t, "params": e.params}
                for e in self._events(recording_id, None)
            ],
        }
        (dest / "metadata.json").write_text(json.dumps(meta, indent=2))
        with self._db:
            self._audit(f"export_deidentified:{export_id}", recording_id)
        return dest

    # -- internals -------------------------------------------------------------------

    def _ensure_exists(self, table: str, object_id: str) -> None:
        row = self._db.execute(f"SELECT 1 FROM {table} WHERE id = ?", (object_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"no {table[:-1]} with id {object_id!r}")

    def _info(self, recording_id: str) -> RecordingInfo:
        row = self._db.execute(
            "SELECT r.id, r.session_id, s.subject_id, s.protocol, r.channels, r.fs,"
            " r.start_time, r.n_samples FROM recordings r JOIN sessions s ON r.session_id = s.id"
            " WHERE r.id = ?",
            (recording_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"no recording with id {recording_id!r}")
        return self._row_to_info(row)

    @staticmethod
    def _row_to_info(row: tuple) -> RecordingInfo:
        rid, sid, subj, protocol, channels, fs, start, n = row
        return RecordingInfo(
            id=rid,
            session_id=sid,
            subject_id=subj,
            protocol=protocol,
            channels=tuple(json.loads(channels)),
            fs=fs,
            start_time=datetime.fromisoformat(start),
            n_samples=n,
        )

    def _events(self, recording_id: str, event_type: str | None) -> list[Event]:
        rows = self._db.execute(
            "SELECT type, t, params FROM events WHERE recording_id = ?"
            " AND (? IS NULL OR type = ?) ORDER BY t, id",
            (recording_id, event_type, event_type),
        ).fetchall()
        return [Event(type=t, t=time, params=json.loads(p)) for t, time, p in rows]

    @staticmethod
    def _channel_indices(info: RecordingInfo, channels: Sequence[str] | None) -> list[int]:
        if channels is None:
            return list(range(len(info.channels)))
        unknown = [c for c in channels if c not in info.channels]
        if unknown:
            raise ValidationError(f"unknown channels {unknown}; available: {list(info.channels)}")
        return [info.channels.index(c) for c in channels]

    def _load_signal(self, recording_id: str) -> np.ndarray:
        rel_path, expected = self._db.execute(
            "SELECT path, sha256 FROM recordings WHERE id = ?", (recording_id,)
        ).fetchone()
        data = (self.root / rel_path).read_bytes()
        if _sha256(data) != expected:
            raise ChecksumError(
                f"signal file for recording {recording_id} failed SHA-256 verification;"
                " it was modified or corrupted after it was written"
            )
        return np.load(io.BytesIO(data), allow_pickle=False)
