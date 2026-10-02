"""Model performance metrics: discrimination (AUC/KS/Gini) and calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from clis.config import TARGET
from clis.model.scoring import ScoringModel


def expected_calibration_error(score: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(score, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, bins - 1)
    gaps = [abs(score[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
            for b in range(bins) if (idx == b).any()]
    return float(np.sum(gaps))


def ks_stat(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float((tpr - fpr).max())


def metrics_table(candidates: list[ScoringModel], split) -> pd.DataFrame:
    rows = []
    for m in candidates:
        r = {"Model": m.name}
        for label, df in [("Train", split.train), ("Validation", split.valid), ("Holdout", split.holdout)]:
            r[f"{label} AUC"] = roc_auc_score(df[TARGET], m.raw_score(df))
        s = m.raw_score(split.holdout)
        r["Holdout KS"] = ks_stat(split.holdout[TARGET].to_numpy(), s)
        r["Holdout Gini"] = 2 * r["Holdout AUC"] - 1
        rows.append(r)
    return pd.DataFrame(rows)


def roc_points(candidates: list[ScoringModel], holdout: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """ROC curves thinned to n points per model so the artifact stays small."""
    out = []
    for m in candidates:
        fpr, tpr, _ = roc_curve(holdout[TARGET], m.raw_score(holdout))
        idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
        out.append(pd.DataFrame({"model": m.name, "fpr": fpr[idx], "tpr": tpr[idx]}))
    return pd.concat(out, ignore_index=True)


def add_risk_band(scored: pd.DataFrame, edges: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Decile bands on PD. Edges come from the reference population and are reused elsewhere."""
    if edges is None:
        edges = np.quantile(scored["pd"], np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
    out = scored.copy()
    out["risk_band"] = np.clip(np.searchsorted(edges, out["pd"], side="right"), 1, 10)
    return out, edges


def band_table(scored: pd.DataFrame) -> pd.DataFrame:
    t = (scored.groupby("risk_band")
               .agg(accounts=("pd", "size"), avg_pd=("pd", "mean"),
                    actual_default_rate=(TARGET, "mean"), defaults=(TARGET, "sum"))
               .reset_index())
    t["cum_pct_defaults_captured"] = t["defaults"][::-1].cumsum()[::-1] / t["defaults"].sum() * 100
    return t


def calibration_summary(model: ScoringModel, holdout: pd.DataFrame) -> dict[str, float]:
    y = holdout[TARGET].to_numpy()
    pd_ = model.predict_pd(holdout)
    return {"brier": float(brier_score_loss(y, pd_)),
            "holdout_ece": expected_calibration_error(pd_, y),
            "validation_ece": float(model.params.get("validation_ece", np.nan)),
            "recalibrated": bool(model.params.get("recalibrated", False)),
            "mean_pd": float(pd_.mean()), "actual_rate": float(y.mean())}
