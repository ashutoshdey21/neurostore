import json
from pathlib import Path

import numpy as np

from neurostore import Event, Store
from tests.conftest import CHANNELS, FS, N_SAMPLES, T0, make_signal


def test_signal_and_metadata_round_trip(store: Store, recording_id: str) -> None:
    out = store.get_window(recording_id, 0.0, N_SAMPLES / FS)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, make_signal())

    [info] = store.list_recordings()
    assert info.id == recording_id
    assert info.channels == tuple(CHANNELS)
    assert info.fs == FS
    assert info.start_time == T0
    assert info.n_samples == N_SAMPLES


def test_events_round_trip_in_time_order(store: Store, recording_id: str) -> None:
    events = [
        Event("tms_pulse", 2.5, {"intensity_pct": 110.0}),
        Event("stim_on", 1.0),
        Event("stim_off", 9.0),
    ]
    assert store.add_events(recording_id, events) == 3
    assert store.get_events(recording_id) == sorted(events, key=lambda e: e.t)
    assert store.get_events(recording_id, "tms_pulse") == [events[0]]


def test_store_reopens_from_disk(tmp_path: Path, store: Store, recording_id: str) -> None:
    with Store(store.root, actor="reader") as reopened:
        np.testing.assert_array_equal(
            reopened.get_window(recording_id, 0.0, 1.0), make_signal()[:, :100]
        )


def test_list_recordings_filters(store: Store, recording_id: str) -> None:
    other_subject = store.add_subject()
    other_session = store.add_session(other_subject, T0, protocol="rtms_10hz")
    other_rec = store.add_recording(other_session, make_signal(), FS, CHANNELS, T0)

    assert {r.id for r in store.list_recordings()} == {recording_id, other_rec}
    assert [r.id for r in store.list_recordings(subject_id=other_subject)] == [other_rec]
    assert [r.id for r in store.list_recordings(protocol="sp_tms")] == [recording_id]
    assert store.list_recordings(subject_id=other_subject, protocol="sp_tms") == []


def test_export_deidentified_has_no_internal_ids(
    tmp_path: Path, store: Store, recording_id: str
) -> None:
    store.add_events(recording_id, [Event("tms_pulse", 2.0, {"intensity_pct": 100.0})])
    [info] = store.list_recordings()

    dest = store.export_deidentified(recording_id, tmp_path / "export")

    np.testing.assert_array_equal(np.load(dest / "signal.npy"), make_signal())
    meta_text = (dest / "metadata.json").read_text()
    meta = json.loads(meta_text)
    assert meta["export_id"] == dest.name
    assert meta["events"] == [{"type": "tms_pulse", "t": 2.0, "params": {"intensity_pct": 100.0}}]
    for internal in (info.id, info.session_id, info.subject_id, "signals/", str(store.root)):
        assert internal not in meta_text
    assert "start_time" not in meta
