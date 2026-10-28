from collections.abc import Iterator
from pathlib import Path

import pytest

from neurostore import Role, Store
from tests.sample_data import CHANNELS, FS, T0, make_signal


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
