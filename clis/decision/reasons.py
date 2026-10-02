"""Reason codes. Customer-facing reasons are specific: a decline caused by the risk score
is explained by the factors that pushed the score up, never by "score too high" alone."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Policy-rule reasons
RULE_REASONS: dict[str, str] = {
    "R01": "Account currently past due",
    "R02": "Serious delinquency (2+ months) in recent months",
    "R03": "Balance over credit limit",
    "R05": "Low repayment relative to balance",
}

# Score-driver reasons: each groups the model features that express the same behaviour
DRIVER_REASONS: dict[str, tuple[str, list[str]]] = {
    "R10": ("Recent repayment status", ["pay_status_1", "pay_status_2"]),
    "R11": ("History of late payments", ["worst_delay", "months_delayed"]),
    "R12": ("High balance relative to credit limit", ["util_now", "util_avg", "bill_1"]),
    "R13": ("Payments small relative to balances", ["pay_ratio", "paid_1"]),
    "R14": ("Size of current credit line", ["limit_amt"]),
}

ALL_REASONS: dict[str, str] = {**RULE_REASONS, **{k: v[0] for k, v in DRIVER_REASONS.items()}}


def driver_codes(contrib: pd.DataFrame, top_n: int = 2) -> list[list[str]]:
    """Top reason codes per account, from features that increased its risk score.

    Every score-based decline must carry at least one reason, so if no factor pushed the
    score up, the single strongest factor is returned.
    """
    grouped = pd.DataFrame({code: contrib[feats].sum(axis=1)
                            for code, (_, feats) in DRIVER_REASONS.items()})
    vals, codes = grouped.to_numpy(), np.array(grouped.columns)
    order = np.argsort(-vals, axis=1)[:, :top_n]
    out = []
    for row_vals, row_order in zip(vals, order):
        picked = [codes[i] for i in row_order if row_vals[i] > 0]
        out.append(picked or [codes[row_order[0]]])
    return out
