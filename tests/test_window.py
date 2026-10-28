import numpy as np
import pytest

from neurostore import Store, ValidationError
from tests.conftest import FS, N_SAMPLES, make_signal

DURATION = N_SAMPLES / FS


@pytest.mark.parametrize(
    ("t_start", "t_end", "i0", "i1"),
    [
        (0.0, 0.01, 0, 1),  # first sample only
        (0.0, 1.0, 0, 100),  # end is exclusive
        (0.5, 0.51, 50, 51),
        (0.1, 0.3, 10, 30),  # 0.1 * 100 is 10.000000000000002 in floating point
        (DURATION - 0.01, DURATION, N_SAMPLES - 1, N_SAMPLES),  # last sample only
        (0.0, DURATION, 0, N_SAMPLES),  # whole recording
    ],
)
def test_window_boundaries(
    store: Store, recording_id: str, t_start: float, t_end: float, i0: int, i1: int
) -> None:
    out = store.get_window(recording_id, t_start, t_end)
    np.testing.assert_array_equal(out, make_signal()[:, i0:i1])


def test_adjacent_windows_tile_without_gap_or_overlap(store: Store, recording_id: str) -> None:
    parts = [store.get_window(recording_id, t, t + 0.25) for t in np.arange(0, DURATION, 0.25)]
    np.testing.assert_array_equal(np.concatenate(parts, axis=1), make_signal())


def test_channel_selection_respects_requested_order(store: Store, recording_id: str) -> None:
    out = store.get_window(recording_id, 0.0, 0.05, channels=["Pz", "Fz"])
    np.testing.assert_array_equal(out, make_signal()[[2, 0], :5])


@pytest.mark.parametrize(
    ("t_start", "t_end"), [(-0.01, 1.0), (0.0, DURATION + 0.01), (2.0, 2.0), (3.0, 1.0)]
)
def test_out_of_range_windows_are_rejected(
    store: Store, recording_id: str, t_start: float, t_end: float
) -> None:
    with pytest.raises(ValidationError, match="window"):
        store.get_window(recording_id, t_start, t_end)


def test_unknown_channel_is_rejected(store: Store, recording_id: str) -> None:
    with pytest.raises(ValidationError, match=r"unknown channels \['O1'\]"):
        store.get_window(recording_id, 0.0, 1.0, channels=["O1"])
