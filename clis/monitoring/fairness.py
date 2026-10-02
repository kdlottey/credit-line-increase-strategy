"""Disparate-impact testing. Protected attributes are used here only, never in the model."""
from __future__ import annotations

import numpy as np
import pandas as pd

FOUR_FIFTHS = 0.80

AGE_BANDS = [0, 25, 35, 45, 60, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45-59", "60+"]


def protected_groups(t: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Sex": t.sex.map({1: "Male", 2: "Female"}).fillna("Unknown"),
        "Age": pd.cut(t.age, AGE_BANDS, labels=AGE_LABELS, right=False).astype(str),
    }, index=t.index)


def adverse_impact(t: pd.DataFrame, approvals: dict[str, pd.Series]) -> pd.DataFrame:
    """Approval rate per group and its ratio to the best-treated group (AIR).

    AIR below 0.80 (the four-fifths rule) is a common screen for possible disparate impact.
    """
    groups = protected_groups(t)
    rows = []
    for strategy, approved in approvals.items():
        for attr in groups.columns:
            rates = approved.groupby(groups[attr]).mean()
            sizes = groups[attr].value_counts()
            for g, r in rates.items():
                rows.append({"Strategy": strategy, "Attribute": attr, "Group": g,
                             "Accounts": int(sizes[g]), "Approval rate %": r * 100,
                             "AIR": r / rates.max() if rates.max() > 0 else np.nan})
    out = pd.DataFrame(rows)
    out["Flag"] = np.where(out.AIR < FOUR_FIFTHS, "Below 0.80", "OK")
    return out
