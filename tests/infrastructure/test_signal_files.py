from pathlib import Path

import numpy as np
import pytest

from neurostore import ChecksumError
from neurostore.infrastructure.signal_files import SignalFiles, sha256
from tests.sample_data import make_signal


@pytest.fixture
def files(tmp_path: Path) -> SignalFiles:
    return SignalFiles(tmp_path)


def test_save_and_load_round_trip(tmp_path: Path, files: SignalFiles) -> None:
    path, checksum = files.save("rec1", make_signal())
    assert path == "signals/rec1.npy"
    assert checksum == sha256((tmp_path / path).read_bytes())
    loaded = files.load(path, checksum)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, make_signal())


def test_one_changed_sample_is_detected(tmp_path: Path, files: SignalFiles) -> None:
    path, checksum = files.save("rec1", make_signal())
    data = np.load(tmp_path / path)
    data[0, 0] += 1.0  # same file size, one value different
    np.save(tmp_path / path, data)
    with pytest.raises(ChecksumError, match=r"signals/rec1\.npy failed SHA-256"):
        files.load(path, checksum)


def test_delete_removes_the_file_and_ignores_missing(tmp_path: Path, files: SignalFiles) -> None:
    path, _ = files.save("rec1", make_signal())
    files.delete(path)
    assert not (tmp_path / path).exists()
    files.delete(path)  # second delete is a no-op
