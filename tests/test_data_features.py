import pandas as pd
import pytest

from clis.data.controls import run_controls
from clis.data.loader import load_raw
from clis.errors import SchemaError
from clis.features.builder import align_months, build_features, segment_table


def test_controls_regroup_codes(raw_accounts):
    clean, log = run_controls(raw_accounts)
    assert clean.education.isin([1, 2, 3, 4]).all()
    assert clean.marriage.isin([1, 2, 3]).all()
    assert {"PASS", "FIXED", "INFO"} <= set(log.status)


def test_offset_shifts_months(raw_accounts):
    snap1 = align_months(raw_accounts, offset=1)
    assert (snap1.s1 == raw_accounts.pay_status_2).all()
    assert (snap1.b5 == raw_accounts.bill_6).all()
    with pytest.raises(ValueError):
        align_months(raw_accounts, offset=2)


def test_features_values(raw_accounts):
    f = build_features(raw_accounts)
    assert len(f) == len(raw_accounts)
    assert (f.util_now >= 0).all()
    assert f.pay_ratio.between(0, 1).all()
    row = raw_accounts.iloc[0]
    expected_worst = max(row[f"pay_status_{i}"] for i in range(1, 6))
    assert f.iloc[0].worst_delay == expected_worst


def test_dormant_flag(raw_accounts):
    df = raw_accounts.copy()
    for i in range(1, 6):
        df.loc[0, f"bill_{i}"] = 0
    f = build_features(df)
    assert bool(f.iloc[0].is_dormant)
    assert f.iloc[0].customer_type.startswith("0. Dormant")


def test_segment_rejects_unknown_column(raw_accounts):
    f = build_features(raw_accounts)
    with pytest.raises(ValueError):
        segment_table(f, "1; DROP TABLE accounts")


def test_loader_schema_error(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad, index=False)
    with pytest.raises(SchemaError):
        load_raw(bad)
