# neurostore runbook

How to set up, check, inspect, and recover a neurostore store. For how the system works, see
[index.html](index.html).

A store is one directory:

```
my_store/
├── neurostore.db     metadata, events, audit log (SQLite)
└── signals/*.npy     one signal file per recording
```

The two halves belong together. Never copy, move, or restore one without the other.

## Setup and checks

```bash
python3.11 -m venv .venv          # any Python 3.11+
.venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest -q               # expect: 41 passed
.venv/bin/python scripts/demo.py  # expect: epochs (7, 8, 100), skipped 2
```

CI runs the same ruff and pytest commands on Python 3.11 and 3.12.

## Inspecting a store

Use the Python API when you can, because it records each read in the audit log:

```python
from pathlib import Path
from neurostore import Store

with Store(Path("my_store"), actor="your-name") as store:  # read-only by default
    for rec in store.list_recordings():
        print(rec.id, rec.protocol, rec.fs, rec.n_samples, rec.channels)
    for row in store.audit_log()[-20:]:
        print(row)
```

For a quick look without Python, open the database read-only with the `sqlite3` CLI. Reads made
this way **are not audited**, so keep them to troubleshooting.

```bash
sqlite3 -readonly -header -column my_store/neurostore.db
```

```sql
-- Recordings with their subject and protocol
SELECT r.id, s.subject_id, s.protocol, r.fs, r.n_samples, r.start_time
FROM recordings r JOIN sessions s ON r.session_id = s.id
ORDER BY r.start_time;

-- Event counts per recording and type
SELECT recording_id, type, COUNT(*) AS n FROM events GROUP BY recording_id, type;

-- Latest audit entries, and any denied writes
SELECT actor, action, object_id, at FROM audit_log ORDER BY id DESC LIMIT 20;
SELECT actor, action, object_id, at FROM audit_log WHERE action LIKE '%:denied';
```

## Errors and first steps

| Error | What it means | What to do |
|---|---|---|
| `ValidationError` | Input breaks a data-model rule. The message names the rule and the value. | Fix the input. Nothing was written: validation runs before any write. |
| `NotFoundError` | An ID doesn't exist, or a read-only `Store` was pointed at a directory with no store. | Check the ID with `list_recordings()` and the path. Open read-write once to create a new store. |
| `AccessDeniedError` | A write was attempted on a read-only `Store`. The attempt is logged as `<action>:denied`. | Reopen with `role=Role.READ_WRITE` if the write is intended. |
| `ChecksumError` | A signal file no longer matches the SHA-256 recorded when it was written. | See below. |
| `sqlite3.OperationalError: database is locked` | Another process is writing. SQLite allows one writer at a time. | Wait and retry, or find the other writer. Don't delete the database or `-journal` files to clear it. |

### When a checksum fails

1. **Stop using that recording.** Don't analyse it or export it.
2. **Leave the file as it is.** Don't re-save, "repair", or overwrite it. You may need it to work out what happened.
3. **Find out what changed it.** Check the file's modification time
   (`ls -l my_store/signals/<recording_id>.npy`) and the audit log for that recording.
4. **Check the other files** with the script in the next section.
5. **Restore that one file from a backup** whose checksum matches the database, then check it again.
   neurostore has no command to rewrite a recording, so a clean backup is the only fix.

## Checking every signal file

This reads each file and compares it with the checksum in the database. It changes nothing and
prints one line per recording.

```bash
.venv/bin/python - my_store <<'EOF'
import hashlib, sqlite3, sys
from pathlib import Path

root = Path(sys.argv[1])
db = sqlite3.connect(f"file:{root / 'neurostore.db'}?mode=ro", uri=True)
bad = 0
for rid, rel, expected in db.execute("SELECT id, path, sha256 FROM recordings"):
    path = root / rel
    status = "MISSING" if not path.exists() else (
        "ok" if hashlib.sha256(path.read_bytes()).hexdigest() == expected else "MISMATCH")
    bad += status != "ok"
    print(f"{status:8} {rid}")
print(f"{bad} problem(s)")
sys.exit(1 if bad else 0)
EOF
```

## Backup and restore

**Back up**, while nothing is writing to the store:

```bash
tar -czf my_store-$(date -u +%Y%m%dT%H%M%SZ).tar.gz my_store
```

Copying the database while a write is running can catch it half-finished. If writers can't be
paused, copy the database with `sqlite3 my_store/neurostore.db ".backup backup.db"` and then copy
`signals/`. Signal files are written before their database row, so every recording in a
database copy taken this way will have its file.

**Restore**: unpack into a new directory, run the checksum script above against it, and switch
to it only if it reports `0 problem(s)`.

```bash
mkdir restored && tar -xzf my_store-<timestamp>.tar.gz -C restored
```

Back-ups contain the same data as the store. Keep them under the same access rules.

## Not covered

neurostore has no update or delete commands, no migrations (the schema is created on first open
and has one version), and no encryption. See "What I would change for real clinical data" in
the [README](../README.md).
