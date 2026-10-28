import json

from neurostore import Event, RecordingInfo
from neurostore.domain.export import deidentified_metadata
from tests.sample_data import CHANNELS, FS, N_SAMPLES, T0


def test_metadata_keeps_signal_facts_and_drops_identifiers() -> None:
    info = RecordingInfo(
        "rec-id", "session-id", "subject-id", "sp_tms", CHANNELS, FS, T0, N_SAMPLES
    )
    events = [Event("tms_pulse", 2.0, {"intensity_pct": 110.0})]

    meta = deidentified_metadata("export-id", info, events)

    assert meta == {
        "export_id": "export-id",
        "channels": list(CHANNELS),
        "fs": FS,
        "n_samples": N_SAMPLES,
        "events": [{"type": "tms_pulse", "t": 2.0, "params": {"intensity_pct": 110.0}}],
    }
    text = json.dumps(meta)
    for identifier in ("rec-id", "session-id", "subject-id", "sp_tms", T0.isoformat()):
        assert identifier not in text
