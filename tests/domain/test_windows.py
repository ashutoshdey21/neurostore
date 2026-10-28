from itertools import pairwise

import numpy as np
import pytest

from neurostore import ValidationError
from neurostore.domain.windows import channel_rows, extract_epochs, window_range
from tests.sample_data import CHANNELS, DURATION, FS, N_SAMPLES, make_signal


@pytest.mark.parametrize(
    ("t_start", "t_end", "expected"),
    [
        (0.0, 0.01, (0, 1)),  # first sample only
        (0.0, 1.0, (0, 100)),  # end is exclusive
        (0.29, 0.5, (29, 50)),  # 0.29 * 100 is 28.999999999999996; truncating would give 28
        (DURATION - 0.01, DURATION, (N_SAMPLES - 1, N_SAMPLES)),  # last sample only
        (0.0, DURATION, (0, N_SAMPLES)),  # whole recording
    ],
)
def test_window_range_boundaries(t_start: float, t_end: float, expected: tuple[int, int]) -> None:
    assert window_range(t_start, t_end, FS, N_SAMPLES) == expected


def test_adjacent_windows_tile_without_gap_or_overlap() -> None:
    ranges = [window_range(t, t + 0.25, FS, N_SAMPLES) for t in np.arange(0, DURATION, 0.25)]
    assert ranges[0][0] == 0 and ranges[-1][1] == N_SAMPLES
    assert all(prev[1] == nxt[0] for prev, nxt in pairwise(ranges))


@pytest.mark.parametrize(
    ("t_start", "t_end"), [(-0.01, 1.0), (0.0, DURATION + 0.01), (2.0, 2.0), (3.0, 1.0)]
)
def test_out_of_range_windows_are_rejected(t_start: float, t_end: float) -> None:
    with pytest.raises(ValidationError, match="window"):
        window_range(t_start, t_end, FS, N_SAMPLES)


def test_channel_rows_follow_requested_order() -> None:
    assert channel_rows(CHANNELS, None) == [0, 1, 2]
    assert channel_rows(CHANNELS, ["Pz", "Fz"]) == [2, 0]


def test_unknown_channel_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"unknown channels \['O1'\]"):
        channel_rows(CHANNELS, ["O1"])


def test_epoch_shape_and_content() -> None:
    epochs = extract_epochs(make_signal(), FS, [2.0, 5.0, 8.0], pre_s=0.1, post_s=0.2)
    assert epochs.data.shape == (3, len(CHANNELS), 30)
    assert epochs.n_skipped == 0
    np.testing.assert_array_equal(epochs.data[1], make_signal()[:, 490:520])


def test_epochs_too_close_to_edges_are_skipped_and_counted() -> None:
    times = [0.05, 0.1, 5.0, 9.8, 9.81]  # first and last windows fall outside the signal
    epochs = extract_epochs(make_signal(), FS, times, pre_s=0.1, post_s=0.2)
    assert epochs.n_skipped == 2
    np.testing.assert_array_equal(epochs.data[0], make_signal()[:, 0:30])  # flush with start
    np.testing.assert_array_equal(epochs.data[-1], make_signal()[:, 970:1000])  # flush with end


def test_no_events_gives_empty_array_of_the_right_shape() -> None:
    epochs = extract_epochs(make_signal(), FS, [], pre_s=0.1, post_s=0.1)
    assert epochs.data.shape == (0, len(CHANNELS), 20)
    assert epochs.data.dtype == np.float32
    assert epochs.n_skipped == 0


@pytest.mark.parametrize(("pre_s", "post_s"), [(-0.1, 0.2), (0.1, -0.2), (0.0, 0.0)])
def test_invalid_epoch_lengths_are_rejected(pre_s: float, post_s: float) -> None:
    with pytest.raises(ValidationError, match="pre_s and post_s"):
        extract_epochs(make_signal(), FS, [5.0], pre_s, post_s)
