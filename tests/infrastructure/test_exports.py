import json
from pathlib import Path

import numpy as np
import pytest

from neurostore.infrastructure.exports import write_export
from tests.sample_data import make_signal


def test_writes_signal_and_metadata(tmp_path: Path) -> None:
    dest = write_export(tmp_path / "exp1", make_signal(), {"export_id": "exp1"})
    np.testing.assert_array_equal(np.load(dest / "signal.npy"), make_signal())
    assert json.loads((dest / "metadata.json").read_text()) == {"export_id": "exp1"}


def test_refuses_to_overwrite_an_existing_export(tmp_path: Path) -> None:
    (tmp_path / "exp1").mkdir()
    with pytest.raises(FileExistsError):
        write_export(tmp_path / "exp1", make_signal(), {})
