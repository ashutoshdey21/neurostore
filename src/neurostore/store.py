"""The public entry point. Store checks access, validates input, calls the storage
classes, and records every action in the audit log. It holds no SQL and no file I/O."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neurostore.domain import windows
from neurostore.domain.errors import AccessDeniedError, NotFoundError, ValidationError
from neurostore.domain.export import deidentified_metadata
from neurostore.domain.models import Epochs, Event, RecordingInfo, Role
from neurostore.domain.validation import (
    to_utc,
    validate_events,
    validate_label,
    validate_recording,
)
from neurostore.infrastructure.database import AuditRow, Database
from neurostore.infrastructure.exports import write_export
from neurostore.infrastructure.signal_files import SignalFiles

DB_FILE = "neurostore.db"


def _new_id() -> str:
    return str(uuid.uuid4())


class Store:
    """A directory holding `neurostore.db` plus a `signals/` folder of .npy files.

    `actor` is recorded in the audit log for every read and write. `role` is
    declared by the caller; it guards against accidental writes, not against
    a malicious user (see README, "What I would change for real clinical data").
    """

    def __init__(self, root: Path, actor: str, role: Role = Role.READ_ONLY) -> None:
        if not isinstance(actor, str) or not actor.strip():
            raise ValidationError("actor must be a non-empty string")
        self.root = Path(root)
        self.actor = actor
        self.role = role
        if role is Role.READ_WRITE:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not (self.root / DB_FILE).exists():
            raise NotFoundError(f"no store at {self.root} (open read-write once to create it)")
        self._db = Database(self.root / DB_FILE)
        self._files = SignalFiles(self.root)

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes ----------------------------------------------------------------------

    def add_subject(self) -> str:
        """Create a subject with a random pseudonymous ID. No other attributes are stored."""
        self._require_write("add_subject")
        subject_id = _new_id()
        with self._db.transaction():
            self._db.insert_subject(subject_id)
            self._log("add_subject", subject_id)
        return subject_id

    def add_session(self, subject_id: str, start_time: datetime, protocol: str) -> str:
        """Create a session for an existing subject and return the new session ID."""
        self._require_write("add_session", subject_id)
        start = to_utc(start_time, "session start_time")
        validate_label(protocol, "protocol")
        self._require_exists("subjects", subject_id)
        session_id = _new_id()
        with self._db.transaction():
            self._db.insert_session(session_id, subject_id, start, protocol)
            self._log("add_session", session_id)
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
        self._require_exists("sessions", session_id)
        recording_id = _new_id()
        path, checksum = self._files.save(recording_id, signal)
        try:
            with self._db.transaction():
                self._db.insert_recording(
                    recording_id, session_id, channels, fs, start, signal.shape[1], path, checksum
                )
                self._log("add_recording", recording_id)
        except BaseException:
            self._files.delete(path)
            raise
        return recording_id

    def add_events(self, recording_id: str, events: Iterable[Event]) -> int:
        """Attach events to a recording. All are validated before any are written."""
        self._require_write("add_events", recording_id)
        events = list(events)
        validate_events(events, self._recording(recording_id).duration)
        with self._db.transaction():
            self._db.insert_events(recording_id, events)
            self._log("add_events", recording_id)
        return len(events)

    # -- reads -----------------------------------------------------------------------

    def list_recordings(
        self, subject_id: str | None = None, protocol: str | None = None
    ) -> list[RecordingInfo]:
        """Return recordings, optionally filtered by subject and protocol, oldest first."""
        recordings = self._db.find_recordings(subject_id=subject_id, protocol=protocol)
        self._log_read("list_recordings", subject_id)
        return recordings

    def get_events(self, recording_id: str, event_type: str | None = None) -> list[Event]:
        """Return a recording's events in time order, optionally filtered by type."""
        self._recording(recording_id)
        events = self._db.events(recording_id, event_type)
        self._log_read("get_events", recording_id)
        return events

    def get_window(
        self,
        recording_id: str,
        t_start: float,
        t_end: float,
        channels: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return samples with t_start <= t < t_end as (channels, samples).

        Times are seconds from recording start, rounded to the nearest sample.
        """
        info = self._recording(recording_id)
        start, end = windows.window_range(t_start, t_end, info.fs, info.n_samples)
        rows = windows.channel_rows(info.channels, channels)
        window = self._load_signal(recording_id)[rows, start:end]
        self._log_read("get_window", recording_id)
        return window

    def get_epochs(self, recording_id: str, event_type: str, pre_s: float, post_s: float) -> Epochs:
        """Cut [event - pre_s, event + post_s) around each event of `event_type`.

        Events whose window would run past either end of the recording are skipped
        and counted in `n_skipped`.
        """
        info = self._recording(recording_id)
        times = [e.t for e in self._db.events(recording_id, event_type)]
        epochs = windows.extract_epochs(
            self._load_signal(recording_id), info.fs, times, pre_s, post_s
        )
        self._log_read("get_epochs", recording_id)
        return epochs

    def export_deidentified(self, recording_id: str, out_dir: Path) -> Path:
        """Write signal + events under a fresh export ID, with no internal IDs or paths.

        Returns the directory written.
        """
        info = self._recording(recording_id)
        export_id = _new_id()
        metadata = deidentified_metadata(export_id, info, self._db.events(recording_id))
        dest = write_export(Path(out_dir) / export_id, self._load_signal(recording_id), metadata)
        self._log_read(f"export_deidentified:{export_id}", recording_id)
        return dest

    def audit_log(self) -> list[AuditRow]:
        """Return (actor, action, object_id, utc_time) rows, oldest first."""
        return self._db.audit_rows()

    # -- helpers ---------------------------------------------------------------------

    def _log(self, action: str, object_id: str | None) -> None:
        """Add an audit entry to the current transaction."""
        self._db.insert_audit(self.actor, action, object_id, datetime.now(UTC))

    def _log_read(self, action: str, object_id: str | None) -> None:
        with self._db.transaction():
            self._log(action, object_id)

    def _require_write(self, action: str, object_id: str | None = None) -> None:
        if self.role is not Role.READ_WRITE:
            self._log_read(f"{action}:denied", object_id)
            raise AccessDeniedError(f"{action} requires a read-write store; role is {self.role}")

    def _require_exists(self, table: str, object_id: str) -> None:
        if not self._db.exists(table, object_id):
            raise NotFoundError(f"no {table.removesuffix('s')} with id {object_id!r}")

    def _recording(self, recording_id: str) -> RecordingInfo:
        found = self._db.find_recordings(recording_id=recording_id)
        if not found:
            raise NotFoundError(f"no recording with id {recording_id!r}")
        return found[0]

    def _load_signal(self, recording_id: str) -> np.ndarray:
        return self._files.load(*self._db.signal_location(recording_id))
