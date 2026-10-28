# neurostore

A small Python package for storing and querying multi-channel time-series recordings
(EEG-like) together with brain-stimulation event markers such as TMS pulses. It is a code
sample: a deliberately minimal take on the collect → store → retrieve loop behind
neurostimulation experiments. All data used here is synthetic.

The **[illustrated walkthrough](https://ashutoshdey21.github.io/neurostore/)** has diagrams of
epoching and sample boundaries, the data model, the full API table, and what each test file
covers (source: [docs/index.html](docs/index.html)).
For setup, inspecting a store, handling errors, and backup and restore, see
[docs/RUNBOOK.md](docs/RUNBOOK.md).

## Quickstart

```python
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
from neurostore import Event, Role, Store

store = Store(Path("my_store"), actor="alice", role=Role.READ_WRITE)
session = store.add_session(store.add_subject(), datetime.now(UTC), protocol="sp_tms")
signal = np.random.default_rng(0).normal(size=(8, 250 * 60)).astype(np.float32)
rec = store.add_recording(session, signal, 250.0, [f"ch{i}" for i in range(8)], datetime.now(UTC))
store.add_events(rec, [Event("tms_pulse", t, {"intensity_pct": 110.0}) for t in (5, 20, 40)])
window = store.get_window(rec, 10.0, 12.0, channels=["ch0", "ch3"])  # (2, 500)
epochs, skipped = store.get_epochs(rec, "tms_pulse", pre_s=0.1, post_s=0.3)  # (3, 8, 100)
```

A fuller example: `python scripts/demo.py`. Development setup:

```bash
pip install -e ".[dev]" && ruff check . && pytest
```

## Architecture

```
            Store(root, actor, role)
   ┌───────────────┴────────────────┐
   │ check role → validate input    │
   │ → do the work → append audit   │
   └───────┬─────────────────┬──────┘
           │                 │
 root/neurostore.db     root/signals/<recording_id>.npy
 ─────────────────      ───────────────────────────────
 subjects  (uuid)       float32, channels x samples
 sessions  → subject    written once, never modified
 recordings → session,
   path + sha256  ──────▶ verified on every read
 events    → recording
 audit_log (actor, action, object_id, utc)
```

The code is in three layers. Each depends only on the layers below it, and
`tests/test_architecture.py` fails if that rule is broken.

**Where to start reading:** `store.py`. Each public method there is a short list of steps
(check role, validate, load, compute, log), so it shows the whole flow. From there, follow
whichever step you are curious about into `domain/` (the rules and the slicing math) or
`infrastructure/` (SQL and files).

```
src/neurostore/
├── store.py              Store: the public entry point. Checks the role, validates input,
│                         calls storage, writes the audit log. No SQL or file formats.
├── infrastructure/       How things are saved
│   ├── database.py       Database: every SQL statement (schema, inserts, queries)
│   ├── signal_files.py   SignalFiles: .npy save/load with SHA-256 verification
│   └── exports.py        write_export: writes an export folder
└── domain/               Rules and arithmetic, no file or database access
    ├── models.py         Event, RecordingInfo, Epochs, Role
    ├── errors.py         ValidationError, AccessDeniedError, ChecksumError, NotFoundError
    ├── validation.py     data-model rules (sampling rate, channels, events, labels, UTC)
    ├── windows.py        time → sample ranges, window and epoch cutting on plain arrays
    └── export.py         what a de-identified export may contain
```

Tests mirror the layers. `tests/domain` tests rules and arithmetic on plain arrays,
`tests/infrastructure` tests each storage class against a temporary directory, and
`tests/test_store.py` covers only what `Store` adds: wiring, access control, the audit trail,
and cleanup when a write fails. Each rule is tested once, in the layer that owns it.

Times are seconds from recording start, windows are half-open, and datetimes must be
timezone-aware (stored in UTC). The walkthrough explains the
[time and sample rules](https://ashutoshdey21.github.io/neurostore/#time) with diagrams.

## Design decisions

**Why SQLite plus .npy.** Metadata and events are small, relational, and queried by
subject, protocol, and event type, which suits SQL. Signals are large, dense arrays that are
read in bulk, which suits a flat binary file. Keeping the two apart keeps the database small
and the signal format trivial to read from any numpy-capable tool. SQLite and `.npy` are both
single-file, need no server, and ship with Python and numpy.

**Why the checksum.** The database and the signal files can drift apart: a file gets
overwritten, partially copied, or edited by another tool. A SHA-256 recorded at write time and
checked on every read turns that silent corruption into a loud `ChecksumError` before any
analysis runs on bad data. The cost is reading the whole file on each access, which is fine
at this scale (see "Left out on purpose").

**Why the audit log.** For human-subject data, who looked at or changed what, and when, is
a question you need to be able to answer after the fact. Every read and write appends a row
with the actor, action, object ID, and UTC time, in the same transaction as the write it
describes. Denied write attempts are logged too.

**Privacy-related choices.** Subjects are random UUIDs with no other attributes. Protocol
labels, event types, and parameter names are restricted to short `[A-Za-z0-9_.-]` labels, so
there is no free-text field that could quietly hold a name or a clinical note.
`export_deidentified` writes the signal and events under a fresh export ID and leaves out
internal IDs, file paths, and absolute start times. These are design choices, not a
guarantee of de-identification.

**Roles.** A store is opened read-only by default; writes need `Role.READ_WRITE`. The role
is declared by the caller, so this prevents accidents, not misuse.

## Left out on purpose

Each of these would be needed in a real system. They were left out to keep the sample small
and the core paths easy to read.

- **Update and delete.** Recordings are append-only. Changing a recording in place would
  silently change the data earlier analyses used, and a delete has to remove the file, the
  rows, and the events consistently while keeping an audit entry. A real system needs deletion
  at least for consent withdrawal: an audited, admin-only delete, with corrections stored as
  new versions instead of edits.
- **Memory-mapped or partial reads.** `get_window` loads the whole file, verifies the checksum
  over all of it, then slices. That is fine for minutes of 8-channel data. A one-hour,
  256-channel, 2 kHz float32 recording is about 3.7 GB, too much to read for a two-second
  window. The fix is chunked storage (HDF5 or Zarr) with a checksum per chunk.
- **Concurrent writers.** SQLite allows one writer at a time; a second writer waits, then fails
  with `database is locked`. No WAL mode, retry, or busy timeout is configured. UUID file names
  do stop two writers from overwriting each other's signal files. Multi-user use would move
  metadata to Postgres.
- **A command-line interface.** The package is Python-only. Commands such as `ingest`, `list`,
  and `export` would be a thin wrapper over the same methods.

Two small additions to the requested API: `get_epochs` returns an `Epochs(data, n_skipped)`
named tuple so the skip count travels with the data, and `get_events` and `audit_log` are
read helpers.

## What I would change for real clinical data

This package is not suitable for real patient data as it stands. The gaps:

- **Encryption at rest.** Signals and the database are plain files. Real data would need
  encrypted storage and managed keys.
- **Real authentication and authorization.** `actor` and `role` are whatever the caller
  passes in. They would need to come from an identity provider, with per-study access
  control enforced outside the client process.
- **Tamper-evident audit.** The audit log lives in the same SQLite file a user with write
  access can edit. It would need to go to an append-only store the user cannot modify.
- **Formal de-identification review.** Dropping IDs and start times is not de-identification
  on its own. Signals, event timing, and metadata can still be identifying, and the process
  would need review against the applicable standard by qualified people.
- **Institutional compliance review.** IRB, data-use agreements, and the institution's
  security and privacy offices would need to sign off on storage, access, and retention.
- **Scalable storage.** Whole-file reads and a single SQLite file do not scale to long,
  high-density recordings or many concurrent users. HDF5 or Zarr for chunked signal access,
  Parquet for events and tabular exports, and Postgres for multi-user metadata would be the
  next steps.
- **Standard formats.** Interoperability would need import and export in BIDS (EEG/iEEG
  extensions) and EDF/EDF+, rather than a custom `.npy` + JSON layout.

## License

MIT. See [LICENSE](LICENSE).
