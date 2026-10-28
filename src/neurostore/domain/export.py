"""What a de-identified export contains. Deciding the content is a domain rule;
writing it to disk is done by the infrastructure layer."""

from __future__ import annotations

from collections.abc import Sequence

from neurostore.domain.models import Event, RecordingInfo


def deidentified_metadata(
    export_id: str, info: RecordingInfo, events: Sequence[Event]
) -> dict[str, object]:
    """Metadata for an export: no internal IDs, file paths, or absolute start times.

    Event times stay relative to recording start.
    """
    return {
        "export_id": export_id,
        "channels": list(info.channels),
        "fs": info.fs,
        "n_samples": info.n_samples,
        "events": [{"type": e.type, "t": e.t, "params": e.params} for e in events],
    }
