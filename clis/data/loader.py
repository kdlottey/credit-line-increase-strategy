"""Load the UCI credit card file and map it onto the project's column names."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clis.config import DATA_FILE, RAW_COLUMNS
from clis.errors import DataLoadError, SchemaError

_EXPECTED_SOURCE_COLS = len(RAW_COLUMNS)


def _read_local(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataLoadError(f"Could not read {path}: {exc}") from exc


def _download() -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=350)
    except Exception as exc:  # network / package errors are all "could not load"
        raise DataLoadError(
            f"{DATA_FILE.name} is missing and the UCI download failed ({exc}). "
            "Place the file in data/ (see data/README.md)."
        ) from exc
    return pd.concat([ds.data.features, ds.data.targets], axis=1)


def load_raw(path: Path = DATA_FILE) -> pd.DataFrame:
    """Return the raw dataset with standard column names and an account_id."""
    raw = _read_local(path) if path.exists() else _download()
    raw = raw.drop(columns=[c for c in raw.columns if str(c).upper() == "ID"])

    if raw.shape[1] != _EXPECTED_SOURCE_COLS:
        raise SchemaError(f"Expected {_EXPECTED_SOURCE_COLS} columns, found {raw.shape[1]}.")
    raw.columns = RAW_COLUMNS
    non_numeric = [c for c in RAW_COLUMNS if not pd.api.types.is_numeric_dtype(raw[c])]
    if non_numeric:
        raise SchemaError(f"Non-numeric columns in source data: {non_numeric}")

    raw.insert(0, "account_id", np.arange(1, len(raw) + 1))
    return raw
