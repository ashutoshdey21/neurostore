import numpy as np
import pytest

from neurostore import ChecksumError, Store


def test_tampered_signal_file_is_detected(store: Store, recording_id: str) -> None:
    [path] = (store.root / "signals").glob("*.npy")
    data = np.load(path)
    data[0, 0] += 1.0  # a single changed sample, file size unchanged
    np.save(path, data)

    with pytest.raises(ChecksumError, match="SHA-256"):
        store.get_window(recording_id, 0.0, 1.0)
    with pytest.raises(ChecksumError):
        store.get_epochs(recording_id, "tms_pulse", 0.1, 0.1)
