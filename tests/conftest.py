"""Shared fixtures: a small synthetic book so tests run in seconds without the real file."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clis.config import RAW_COLUMNS  # noqa: E402


@pytest.fixture
def raw_accounts() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "limit_amt": rng.choice([50_000, 100_000, 200_000], n).astype(float),
        "sex": rng.choice([1, 2], n), "education": rng.choice([1, 2, 3, 5], n),
        "marriage": rng.choice([1, 2, 0], n), "age": rng.integers(21, 70, n),
    })
    for i in range(1, 7):
        df[f"pay_status_{i}"] = rng.choice([-2, -1, 0, 0, 0, 1, 2], n)
    for i in range(1, 7):
        df[f"bill_{i}"] = (df.limit_amt * rng.uniform(0, 1.1, n)).round()
    for i in range(1, 7):
        df[f"paid_{i}"] = (df[f"bill_{i}"] * rng.uniform(0, 0.3, n)).round()
    risk = (df.pay_status_1.clip(0) + df[[f"pay_status_{i}" for i in range(1, 7)]].max(axis=1).clip(0)) / 4
    df["default_next_month"] = (rng.uniform(0, 1, n) < 0.1 + 0.6 * risk).astype(int)
    df = df[RAW_COLUMNS]
    df.insert(0, "account_id", np.arange(1, n + 1))
    return df
