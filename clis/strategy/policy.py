"""Eligibility rules shared by the proactive (PLI) and reactive (RLI) strategies."""
from __future__ import annotations

import pandas as pd

from clis.config import Policy


def policy_gates(t: pd.DataFrame, p: Policy) -> pd.Series:
    """Hard rules every line increase must pass, whatever the model says."""
    return ((t.pay_status_1 <= 0)
            & (t.worst_delay <= p.max_worst_delay)
            & (t.util_now <= p.max_util)
            & (t.pay_ratio >= p.min_pay_ratio))   # no-op while min_pay_ratio is 0


def champion_mask(t: pd.DataFrame, p: Policy, min_util: float | None = None) -> pd.Series:
    """Current rule-based strategy: never late in the lookback and heavy line usage."""
    u = p.champion_min_util if min_util is None else min_util
    return policy_gates(t, p) & (t.worst_delay <= 0) & (t.util_now >= u)


def challenger_mask(t: pd.DataFrame, p: Policy, pd_cutoff: float, min_util: float | None = None) -> pd.Series:
    """Proposed strategy: the same gates, with the PD model replacing the payment rule."""
    u = p.challenger_min_util if min_util is None else min_util
    return policy_gates(t, p) & (t.util_now >= u) & (t["pd"] <= pd_cutoff)
