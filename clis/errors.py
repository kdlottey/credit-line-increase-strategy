"""Typed errors raised at module boundaries so callers can react precisely."""


class CLIError(Exception):
    """Base class for every error this package raises on purpose."""


class DataLoadError(CLIError):
    """Source data could not be read from disk or downloaded."""


class SchemaError(CLIError):
    """Data does not have the columns or types the pipeline expects."""


class ArtifactError(CLIError):
    """A saved model or result file is missing, unreadable or inconsistent."""


class StrategyError(CLIError):
    """A strategy could not be evaluated (e.g. no cut-off meets the risk appetite)."""
