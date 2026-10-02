"""Reactive line increase (RLI): a customer asks for more credit and gets a decision with reasons.

Fully vectorized, so scoring the whole portfolio is a few array operations rather than a
row-by-row loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from clis.config import TARGET, Settings
from clis.decision.reasons import ALL_REASONS, driver_codes
from clis.model.scoring import ScoringModel
from clis.strategy.economics import tiered_increase


@dataclass
class Decision:
    decision: str           # APPROVE | DECLINE
    increase: int
    pd: float
    codes: list[str]

    @property
    def reasons(self) -> list[str]:
        return [f"{c} {ALL_REASONS[c]}" for c in self.codes]


def rule_flags(t: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    p = cfg.policy
    flags = {"R01": t.pay_status_1 > 0,
             "R02": t.worst_delay > p.max_worst_delay,
             "R03": t.util_now > p.max_util}
    if p.min_pay_ratio > 0:
        flags["R05"] = t.pay_ratio < p.min_pay_ratio
    return pd.DataFrame(flags, index=t.index)


def rule_impact(t: pd.DataFrame, decisions: pd.DataFrame, threshold: float) -> dict[str, float]:
    """What a candidate repayment-ratio rule would add on top of the current decisions:
    the accounts it would newly decline, and whether they are actually riskier."""
    approved = decisions.rli_decision == "APPROVE"
    newly = approved & (t.pay_ratio < threshold)
    keep = approved & ~newly
    return {"threshold": threshold, "would_decline": int(newly.sum()),
            "default_rate_newly_declined": float(t.loc[newly, TARGET].mean() * 100),
            "default_rate_remaining_approved": float(t.loc[keep, TARGET].mean() * 100)}


def decide_portfolio(t: pd.DataFrame, model: ScoringModel, cfg: Settings, cutoff: float) -> pd.DataFrame:
    """Decision, increase and reason codes for every account in t (which must have 'pd')."""
    flags = rule_flags(t, cfg)
    score_decline = (t["pd"] > cutoff).to_numpy()

    codes = _vector_codes(flags)
    if score_decline.any():
        drivers = driver_codes(model.contributions(t.loc[score_decline]))
        for pos, d in zip(np.flatnonzero(score_decline), drivers):
            codes[pos] = codes[pos] + [c for c in d if c not in codes[pos]]

    approve = ~flags.any(axis=1).to_numpy() & ~score_decline
    inc = np.where(approve, tiered_increase(t.limit_amt, t["pd"], cfg.sizing), 0).astype(int)
    return pd.DataFrame({"rli_decision": np.where(approve, "APPROVE", "DECLINE"),
                         "rli_increase": inc, "rli_score_decline": score_decline,
                         "rli_codes": ["; ".join(c) if c else "-" for c in codes]}, index=t.index)


def _vector_codes(flags: pd.DataFrame) -> list[list[str]]:
    arr, names = flags.to_numpy(), np.array(flags.columns)
    return [list(names[row]) for row in arr]


def decide_one(account: pd.DataFrame, model: ScoringModel, cfg: Settings, cutoff: float) -> Decision:
    if len(account) != 1:
        raise ValueError("decide_one expects exactly one account row")
    scored = account.assign(pd=model.predict_pd(account))
    r = decide_portfolio(scored, model, cfg, cutoff).iloc[0]
    codes = [] if r.rli_codes == "-" else r.rli_codes.split("; ")
    return Decision(r.rli_decision, int(r.rli_increase), float(scored["pd"].iloc[0]), codes)


def reason_validation(t: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Does each reason actually point at riskier accounts? Compares default rates."""
    exploded = decisions.rli_codes.str.split("; ").explode()
    exploded = exploded[exploded != "-"]
    only_one = decisions.rli_codes.str.count(";") == 0
    approved_rate = t.loc[decisions.rli_decision == "APPROVE", TARGET].mean() * 100
    rows = []
    for code in sorted(exploded.unique()):
        cited = exploded[exploded == code].index.unique()
        sole = decisions.index[only_one & (decisions.rli_codes == code)]
        rows.append({"Code": code, "Reason": ALL_REASONS[code], "Declines citing": len(cited),
                     "Default rate % (cited)": t.loc[cited, TARGET].mean() * 100,
                     "Sole-reason declines": len(sole),
                     "Default rate % (sole reason)": t.loc[sole, TARGET].mean() * 100 if len(sole) else np.nan,
                     "Approved default rate %": approved_rate})
    return pd.DataFrame(rows)
