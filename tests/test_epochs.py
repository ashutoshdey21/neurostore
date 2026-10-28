import numpy as np
import pytest

from neurostore import Event, Store, ValidationError
from tests.conftest import CHANNELS, make_signal


def test_epoch_shape_and_content(store: Store, recording_id: str) -> None:
    store.add_events(recording_id, [Event("tms_pulse", t) for t in (2.0, 5.0, 8.0)])

    epochs = store.get_epochs(recording_id, "tms_pulse", pre_s=0.1, post_s=0.2)

    assert epochs.data.shape == (3, len(CHANNELS), 30)
    assert epochs.n_skipped == 0
    np.testing.assert_array_equal(epochs.data[1], make_signal()[:, 490:520])


def test_events_too_close_to_edges_are_skipped_and_counted(store: Store, recording_id: str) -> None:
    times = [0.05, 0.1, 5.0, 9.8, 9.81]  # first and last windows fall outside the recording
    store.add_events(recording_id, [Event("tms_pulse", t) for t in times])

    epochs = store.get_epochs(recording_id, "tms_pulse", pre_s=0.1, post_s=0.2)

    assert epochs.n_skipped == 2
    assert epochs.data.shape == (3, len(CHANNELS), 30)
    np.testing.assert_array_equal(epochs.data[0], make_signal()[:, 0:30])  # starts at sample 0
    np.testing.assert_array_equal(epochs.data[-1], make_signal()[:, 970:1000])  # ends at last


def test_only_requested_event_type_is_used(store: Store, recording_id: str) -> None:
    store.add_events(recording_id, [Event("stim_on", 1.0), Event("tms_pulse", 2.0)])
    data, skipped = store.get_epochs(recording_id, "tms_pulse", pre_s=0.0, post_s=0.5)
    assert data.shape == (1, len(CHANNELS), 50)
    assert skipped == 0


def test_no_matching_events_gives_empty_array(store: Store, recording_id: str) -> None:
    epochs = store.get_epochs(recording_id, "tms_pulse", pre_s=0.1, post_s=0.1)
    assert epochs.data.shape == (0, len(CHANNELS), 20)
    assert epochs.n_skipped == 0


def test_invalid_epoch_lengths_are_rejected(store: Store, recording_id: str) -> None:
    with pytest.raises(ValidationError, match="pre_s and post_s"):
        store.get_epochs(recording_id, "tms_pulse", pre_s=-0.1, post_s=0.2)
