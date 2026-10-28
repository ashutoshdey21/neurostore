"""Signal arrays as .npy files, each verified against a SHA-256 checksum when read."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

from neurostore.domain.errors import ChecksumError


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SignalFiles:
    """Files live under `root/<dir_name>/`; callers store the returned relative path."""

    def __init__(self, root: Path, dir_name: str = "signals") -> None:
        self._root = root
        self._dir_name = dir_name

    def save(self, name: str, signal: np.ndarray) -> tuple[str, str]:
        """Write `signal` and return (path relative to root, sha256 of the file)."""
        buf = io.BytesIO()
        np.save(buf, signal, allow_pickle=False)
        data = buf.getvalue()
        rel_path = f"{self._dir_name}/{name}.npy"
        (self._root / self._dir_name).mkdir(parents=True, exist_ok=True)
        (self._root / rel_path).write_bytes(data)
        return rel_path, sha256(data)

    def load(self, rel_path: str, expected_sha256: str) -> np.ndarray:
        """Read a signal, raising ChecksumError if the file changed since it was saved."""
        data = (self._root / rel_path).read_bytes()
        if sha256(data) != expected_sha256:
            raise ChecksumError(
                f"{rel_path} failed SHA-256 verification;"
                " it was modified or corrupted after it was written"
            )
        return np.load(io.BytesIO(data), allow_pickle=False)

    def delete(self, rel_path: str) -> None:
        (self._root / rel_path).unlink(missing_ok=True)
