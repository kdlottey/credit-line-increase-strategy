"""Read/write the files the dashboard depends on. The app only ever reads these;
it never retrains, which keeps it fast on free hosting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from clis.config import ARTIFACT_DIR
from clis.errors import ArtifactError

BUILD_HINT = "Run `python scripts/build_artifacts.py` to regenerate artifacts."


def save_table(df: pd.DataFrame, name: str, directory: Path = ARTIFACT_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    df.to_parquet(directory / f"{name}.parquet", index=False)


def load_table(name: str, directory: Path = ARTIFACT_DIR) -> pd.DataFrame:
    path = directory / f"{name}.parquet"
    if not path.exists():
        raise ArtifactError(f"Missing artifact {path.name}. {BUILD_HINT}")
    try:
        return pd.read_parquet(path)
    except Exception as exc:   # pyarrow raises several unrelated types for bad files
        raise ArtifactError(f"Could not read {path.name}: {exc}. {BUILD_HINT}") from exc


def save_json(obj: dict[str, Any], name: str, directory: Path = ARTIFACT_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))


def load_json(name: str, directory: Path = ARTIFACT_DIR) -> dict[str, Any]:
    path = directory / f"{name}.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ArtifactError(f"Missing artifact {path.name}. {BUILD_HINT}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path.name} is corrupt: {exc}. {BUILD_HINT}") from exc
