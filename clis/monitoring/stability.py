"""Population stability: score PSI and per-feature CSI between two snapshots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from clis.config import FEATURES

THRESHOLDS = (0.10, 0.25)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("PSI needs non-empty distributions")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    lo, hi = THRESHOLDS
    return "Stable" if value < lo else "Monitor" if value < hi else "Significant shift"


def feature_stability(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """CSI for each model feature: which inputs moved, not just whether the score did."""
    rows = [{"Feature": f, "CSI": psi(reference[f], current[f])} for f in FEATURES]
    out = pd.DataFrame(rows).sort_values("CSI", ascending=False)
    out["Status"] = out.CSI.map(status)
    return out
