from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from neurostore import Role, Store

FS = 100.0
CHANNELS = ["Fz", "Cz", "Pz"]
N_SAMPLES = 1000  # 10 s at 100 Hz
T0 = datetime(2026, 1, 15, 9, 30, tzinfo=UTC)


def make_signal() -> np.ndarray:
    """Each value encodes its position: channel * 10_000 + sample index."""
    rows = np.arange(len(CHANNELS))[:, None] * 10_000
    return (rows + np.arange(N_SAMPLES)[None, :]).astype(np.float32)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "store", actor="tester", role=Role.READ_WRITE) as s:
        yield s


@pytest.fixture
def session_id(store: Store) -> str:
    return store.add_session(store.add_subject(), T0, protocol="sp_tms")


@pytest.fixture
def recording_id(store: Store, session_id: str) -> str:
    return store.add_recording(session_id, make_signal(), FS, CHANNELS, T0)
