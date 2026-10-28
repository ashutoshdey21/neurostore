"""Generate a synthetic TMS-EEG-like recording, store it, and print the evoked average.

All data here is random noise plus a made-up evoked response. Run with:
    python scripts/demo.py
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neurostore import Event, Role, Store

FS = 250.0
DURATION_S = 60.0
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
PULSE_TIMES = [0.05, 5.0, 12.0, 20.0, 27.5, 35.0, 42.0, 50.0, 59.9]  # first, last too near edges


def synthesize(rng: np.random.Generator) -> np.ndarray:
    n = int(DURATION_S * FS)
    signal = rng.normal(0.0, 5.0, size=(len(CHANNELS), n))  # background "EEG", in microvolts
    t = np.arange(int(0.3 * FS)) / FS
    evoked = 20.0 * np.exp(-t / 0.05) * np.sin(2 * np.pi * 10 * t)  # damped 10 Hz response
    gain = np.linspace(1.0, 0.3, len(CHANNELS))[:, None]  # stronger near the "coil"
    for pulse in PULSE_TIMES:
        i = round(pulse * FS)
        seg = signal[:, i : i + len(t)]
        seg += gain * evoked[: seg.shape[1]]
    return signal.astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(seed=0)
    with (
        tempfile.TemporaryDirectory() as tmp,
        Store(Path(tmp), actor="demo", role=Role.READ_WRITE) as store,
    ):
        subject = store.add_subject()
        session = store.add_session(subject, datetime.now(UTC), protocol="sp_tms_demo")
        recording = store.add_recording(
            session, synthesize(rng), FS, CHANNELS, start_time=datetime.now(UTC)
        )
        store.add_events(
            recording, [Event("tms_pulse", t, {"intensity_pct": 110.0}) for t in PULSE_TIMES]
        )

        epochs = store.get_epochs(recording, "tms_pulse", pre_s=0.1, post_s=0.3)
        average = epochs.data.mean(axis=0)
        print(f"recording {recording}: {len(CHANNELS)} ch @ {FS:g} Hz, {DURATION_S:g} s")
        print(f"epochs: {epochs.data.shape}  (skipped {epochs.n_skipped} near the edges)")
        print("\nevoked average, peak |amplitude| 0-300 ms after pulse (uV):")
        post = average[:, round(0.1 * FS) :]
        for name, peak in zip(CHANNELS, np.abs(post).max(axis=1), strict=True):
            print(f"  {name:>4} {peak:6.2f} {'#' * round(peak)}")
        print(f"\naudit log entries: {len(store.audit_log())}")


if __name__ == "__main__":
    main()
