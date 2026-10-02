"""Build features with DuckDB SQL for a snapshot at month offset 0 (now) or 1 (one month earlier)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from clis.config import LOOKBACK, N_MONTHS_RAW
from clis.errors import SchemaError

SQL_DIR = Path(__file__).parent / "sql"
SEGMENT_COLUMNS = {"util_band", "customer_type"}
_PASSTHROUGH = ["account_id", "limit_amt", "sex", "age", "education", "marriage", "default_next_month"]


@lru_cache(maxsize=None)
def read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def align_months(raw: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    """Rename month columns so s1/b1/p1 is the snapshot's latest month.

    offset=1 rebuilds the account as it looked one month earlier (months 2..6), which gives a
    genuine out-of-time population to compare score distributions against.
    """
    if not 0 <= offset <= N_MONTHS_RAW - LOOKBACK:
        raise ValueError(f"offset must be between 0 and {N_MONTHS_RAW - LOOKBACK}")
    missing = [c for c in _PASSTHROUGH if c not in raw.columns]
    if missing:
        raise SchemaError(f"Missing columns for feature build: {missing}")

    out = raw[_PASSTHROUGH].copy()
    for k in range(1, LOOKBACK + 1):
        src = k + offset
        out[f"s{k}"] = raw[f"pay_status_{src}"]
        out[f"b{k}"] = raw[f"bill_{src}"].astype(float)
        out[f"p{k}"] = raw[f"paid_{src}"].astype(float)
    return out


def build_features(raw: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    snap = align_months(raw, offset)
    with duckdb.connect() as con:
        con.register("snapshot", snap)
        return con.sql(read_sql("features.sql")).df()


def segment_table(acct: pd.DataFrame, by: str) -> pd.DataFrame:
    if by not in SEGMENT_COLUMNS:   # guards the f-string below against SQL injection
        raise ValueError(f"Unknown segment column {by!r}; choose from {sorted(SEGMENT_COLUMNS)}")
    with duckdb.connect() as con:
        con.register("accounts", acct)
        return con.sql(read_sql("segments.sql").format(by=by)).df()
