"""Errors raised by neurostore. All share the `NeurostoreError` base class."""


class NeurostoreError(Exception):
    """Base class for all neurostore errors."""


class ValidationError(NeurostoreError, ValueError):
    """Input data does not satisfy the data model."""


class AccessDeniedError(NeurostoreError, PermissionError):
    """The store's role does not permit the requested action."""


class ChecksumError(NeurostoreError):
    """A signal file on disk does not match the checksum recorded at write time."""


class NotFoundError(NeurostoreError, KeyError):
    """A referenced subject, session, recording, or store does not exist."""
