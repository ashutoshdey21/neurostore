# neurostore

A small Python package for storing and querying multi-channel time-series recordings
(EEG-like) together with brain-stimulation event markers such as TMS pulses. It is a code
sample: a deliberately minimal take on the collect → store → retrieve loop behind
neurostimulation experiments. All data used here is synthetic.

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
   │ validate inputs → check role   │
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

Source layout: `models.py` holds the data types, errors, and validation; `store.py` holds the
SQLite schema and the `Store` class.

Conventions:

- Times are seconds from recording start. `get_window(t_start, t_end)` returns the half-open
  sample range `[round(t_start*fs), round(t_end*fs))`, so adjacent windows tile exactly.
- An epoch spans `[event - pre_s, event + post_s)`. Events whose epoch would run past either
  end of the recording are skipped and counted in `Epochs.n_skipped`.
- Datetimes must be timezone-aware and are stored in UTC.

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
at this scale (see below).

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

## Scope notes

Kept out on purpose, to stay small: no update or delete operations (recordings are
append-only), no partial/memory-mapped signal reads, no concurrency handling beyond what
SQLite provides, and no CLI. `get_epochs` returns an `Epochs(data, n_skipped)` named tuple
rather than a bare array so the skip count travels with the data. `get_events` and
`audit_log` exist as small read helpers alongside the specified API.

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
