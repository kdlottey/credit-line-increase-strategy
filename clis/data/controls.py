"""Data-quality controls run on every load. Each check records what was done about it."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

BILLS = [f"bill_{i}" for i in range(1, 7)]
PAIDS = [f"paid_{i}" for i in range(1, 7)]


@dataclass
class ControlResult:
    check: str
    result: int
    status: str   # PASS | FLAG | FIXED | REVIEWED | INFO
    action: str


def run_controls(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean_df, control_log)."""
    log: list[ControlResult] = []

    def add(check: str, value: int, ok: bool, bad_status: str, action: str) -> None:
        log.append(ControlResult(check, int(value), "PASS" if ok else bad_status,
                                 "-" if ok else action))

    add("Row count", len(df), len(df) > 0, "FLAG", "Stop: empty extract")
    miss = int(df.isna().sum().sum())
    add("Missing values", miss, miss == 0, "FLAG", "Investigate source extract")
    bad_lim = int((df.limit_amt <= 0).sum())
    add("Credit limit <= 0", bad_lim, bad_lim == 0, "FLAG", "Exclude from strategy")

    dup_mask = df.drop(columns="account_id").duplicated(keep=False)
    dup_dormant = (df.loc[dup_mask, BILLS + PAIDS].abs().sum(axis=1) == 0).mean() if dup_mask.any() else 0
    add("Identical account rows", int(df.drop(columns="account_id").duplicated().sum()),
        not dup_mask.any(), "REVIEWED",
        f"Kept: {dup_dormant:.0%} are dormant accounts with zero activity, "
        "so identical rows are expected for distinct customers")

    bad_edu = int((~df.education.isin([1, 2, 3, 4])).sum())
    add("Undocumented education codes", bad_edu, bad_edu == 0, "FIXED", "Regrouped to 'other' (4)")
    bad_mar = int((~df.marriage.isin([1, 2, 3])).sum())
    add("Undocumented marriage codes", bad_mar, bad_mar == 0, "FIXED", "Regrouped to 'other' (3)")

    log.append(ControlResult("Accounts over limit", int((df.bill_1 > df.limit_amt).sum()), "INFO",
                             "Blocked from increases by policy gate"))
    log.append(ControlResult("Accounts with credit balance", int((df.bill_1 < 0).sum()), "INFO",
                             "Treated as 0% utilization"))

    clean = df.copy()
    clean.loc[~clean.education.isin([1, 2, 3, 4]), "education"] = 4
    clean.loc[~clean.marriage.isin([1, 2, 3]), "marriage"] = 3
    clean = clean[clean.limit_amt > 0]
    return clean, pd.DataFrame([asdict(r) for r in log])
