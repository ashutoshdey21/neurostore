from pathlib import Path

import pytest

from neurostore import AccessDeniedError, Event, Store, ValidationError
from tests.conftest import CHANNELS, FS, T0, make_signal


def test_read_only_store_can_read_but_not_write(store: Store, recording_id: str) -> None:
    with Store(store.root, actor="analyst") as ro:
        assert len(ro.list_recordings()) == 1
        with pytest.raises(AccessDeniedError, match="requires a read-write store"):
            ro.add_subject()
        with pytest.raises(AccessDeniedError):
            ro.add_events(recording_id, [Event("tms_pulse", 1.0)])
        with pytest.raises(AccessDeniedError):
            ro.add_recording("any", make_signal(), FS, CHANNELS, T0)
    assert store.get_events(recording_id) == []


def test_read_only_cannot_create_a_new_store(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no store"):
        Store(tmp_path / "nothing", actor="analyst")
    assert not (tmp_path / "nothing").exists()


def test_actor_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="actor"):
        Store(tmp_path, actor="  ")


def test_audit_log_records_reads_writes_and_denials(
    tmp_path: Path, store: Store, recording_id: str
) -> None:
    store.add_events(recording_id, [Event("tms_pulse", 2.0)])
    store.get_window(recording_id, 0.0, 1.0)
    store.get_epochs(recording_id, "tms_pulse", 0.1, 0.1)
    store.export_deidentified(recording_id, tmp_path / "export")
    with Store(store.root, actor="analyst") as ro:
        ro.get_events(recording_id)
        with pytest.raises(AccessDeniedError):
            ro.add_subject()

    log = store.audit_log()
    actions = [(actor, action.split(":")[0], obj) for actor, action, obj, _ in log]
    [info] = store.list_recordings()
    assert actions == [
        ("tester", "add_subject", info.subject_id),
        ("tester", "add_session", info.session_id),
        ("tester", "add_recording", recording_id),
        ("tester", "add_events", recording_id),
        ("tester", "get_window", recording_id),
        ("tester", "get_epochs", recording_id),
        ("tester", "export_deidentified", recording_id),
        ("analyst", "get_events", recording_id),
        ("analyst", "add_subject", None),
    ]
    assert log[-1][1] == "add_subject:denied"
    assert all(at.endswith("+00:00") for *_, at in log)  # UTC timestamps
