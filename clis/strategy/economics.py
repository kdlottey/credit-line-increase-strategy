"""Increase sizing and the per-account profit model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from clis.config import Economics, LineSizing


def flat_increase(limit: pd.Series | np.ndarray, s: LineSizing) -> np.ndarray:
    return np.minimum(np.round(np.asarray(limit, float) * s.flat_pct, -3), s.cap)


def tiered_increase(limit: pd.Series | np.ndarray, pd_: pd.Series | np.ndarray, s: LineSizing) -> np.ndarray:
    """Lower predicted risk earns a larger increase."""
    limit, pd_ = np.asarray(limit, float), np.asarray(pd_, float)
    uppers = np.array([u for u, _ in s.tiers])
    pcts = np.array([pct for _, pct in s.tiers])
    pct = pcts[np.clip(np.searchsorted(uppers, pd_, side="right"), 0, len(pcts) - 1)]
    return np.minimum(np.round(limit * pct, -3), s.cap)


def account_economics(t: pd.DataFrame, increase: np.ndarray, e: Economics,
                      default_col: str = "pd") -> pd.DataFrame:
    """12-month value of each increase.

    new balance = increase x current utilization (customers use new line like the old)
    revenue     = new balance x (APR - cost of funds) x (1 - P(default))
    loss        = P(default) x LGD x (increase x CCF)  (defaulters draw down more of the line)

    default_col="pd" gives expected values; default_col=TARGET gives realized values.
    """
    p = t[default_col].to_numpy(float)
    new_bal = increase * t.util_now.clip(0, 1).to_numpy()
    revenue = new_bal * (e.apr - e.cost_of_funds) * e.horizon_years * (1 - p)
    loss = p * e.lgd * increase * e.ccf
    return pd.DataFrame({"increase": increase, "new_balance": new_bal, "revenue": revenue,
                         "loss": loss, "profit": revenue - loss}, index=t.index)
