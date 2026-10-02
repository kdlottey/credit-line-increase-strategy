"""Champion vs challenger PLI: tune the PD cut-off on validation, report on holdout."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from clis.config import TARGET, Settings
from clis.errors import StrategyError
from clis.strategy.economics import account_economics, flat_increase, tiered_increase
from clis.strategy.policy import champion_mask, challenger_mask


@dataclass
class CutoffChoice:
    cutoff: float
    sweep: pd.DataFrame
    champion_default_rate: float
    reason: str


def _summary(t: pd.DataFrame, mask: pd.Series, increase: np.ndarray, cfg: Settings, name: str) -> dict:
    s = t[mask]
    inc = increase[mask.to_numpy()]
    exp = account_economics(s, inc, cfg.economics, "pd")
    real = account_economics(s, inc, cfg.economics, TARGET)
    return {"Strategy": name, "Eligible accounts": int(mask.sum()),
            "% of book": float(mask.mean() * 100),
            "Default rate %": float(s[TARGET].mean() * 100) if len(s) else np.nan,
            "Line granted": float(inc.sum()), "New balance": float(exp.new_balance.sum()),
            "Expected profit": float(exp.profit.sum()), "Realized profit": float(real.profit.sum())}


def champion_result(t: pd.DataFrame, cfg: Settings, min_util: float | None = None) -> tuple[pd.Series, dict]:
    m = champion_mask(t, cfg.policy, min_util)
    return m, _summary(t, m, flat_increase(t.limit_amt, cfg.sizing), cfg, "Champion (rules)")


def challenger_result(t: pd.DataFrame, cfg: Settings, cutoff: float,
                      min_util: float | None = None) -> tuple[pd.Series, dict]:
    m = challenger_mask(t, cfg.policy, cutoff, min_util)
    inc = tiered_increase(t.limit_amt, t["pd"], cfg.sizing)
    return m, _summary(t, m, inc, cfg, f"Challenger (PD ≤ {cutoff:.2f})")


def sweep_cutoffs(t: pd.DataFrame, cfg: Settings, min_util: float | None = None) -> pd.DataFrame:
    rows = []
    for c in cfg.policy.cutoff_grid:
        m, r = challenger_result(t, cfg, c, min_util)
        if m.sum() >= cfg.policy.min_segment_size:
            rows.append({"pd_cutoff": c, "eligible": r["Eligible accounts"],
                         "default_rate_pct": r["Default rate %"],
                         "expected_profit": r["Expected profit"]})
    return pd.DataFrame(rows)


def choose_cutoff(valid: pd.DataFrame, cfg: Settings, min_util: float | None = None,
                  champ_min_util: float | None = None) -> CutoffChoice:
    """Highest expected profit among cut-offs whose validation default rate is no worse
    than the champion's (the risk appetite). Uses validation data only."""
    _, champ = champion_result(valid, cfg, champ_min_util)
    sweep = sweep_cutoffs(valid, cfg, min_util)
    if sweep.empty:
        raise StrategyError("No PD cut-off approves enough accounts to evaluate.")
    ok = sweep[sweep.default_rate_pct <= champ["Default rate %"]]
    if ok.empty:
        raise StrategyError("No cut-off keeps the default rate at or below the champion's.")
    best = ok.loc[ok.expected_profit.idxmax()]
    reason = ("profit peak" if best.pd_cutoff < ok.pd_cutoff.max()
              else "risk-appetite limit (profit still rising)")
    return CutoffChoice(float(best.pd_cutoff), sweep, champ["Default rate %"], reason)


def swap_sets(t: pd.DataFrame, champ: pd.Series, chall: pd.Series) -> pd.DataFrame:
    groups = {"Both approve": champ & chall, "Swap-in (challenger only)": ~champ & chall,
              "Swap-out (champion only)": champ & ~chall}
    return pd.DataFrame([{"Group": g, "Accounts": int(m.sum()),
                          "Default rate %": t.loc[m, TARGET].mean() * 100 if m.any() else np.nan}
                         for g, m in groups.items()])
