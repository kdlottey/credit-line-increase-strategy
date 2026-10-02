"""Train candidate PD models. The model is chosen and calibrated on validation data only;
the holdout set is never touched until final reporting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from clis.config import FEATURES, RANDOM_STATE, TARGET
from clis.model.evaluate import expected_calibration_error
from clis.model.scoring import ScoringModel


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame      # early stopping, model choice, calibration, PD cut-off tuning
    holdout: pd.DataFrame    # reported results only


def split_accounts(acct: pd.DataFrame, valid_size: float = 0.2, holdout_size: float = 0.2) -> Split:
    rest, holdout = train_test_split(acct, test_size=holdout_size, random_state=RANDOM_STATE,
                                     stratify=acct[TARGET])
    train, valid = train_test_split(rest, test_size=valid_size / (1 - holdout_size),
                                    random_state=RANDOM_STATE, stratify=rest[TARGET])
    return Split(train.copy(), valid.copy(), holdout.copy())


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df[FEATURES].astype(float).fillna(0.0), df[TARGET].to_numpy()


def fit_logistic(split: Split) -> ScoringModel:
    X, y = _xy(split.train)
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000).fit(scaler.transform(X), y)
    params = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
              "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    return ScoringModel(kind="logistic", name="Logistic Regression", params=params)


def fit_xgboost(split: Split) -> ScoringModel:
    X, y = _xy(split.train)
    Xv, yv = _xy(split.valid)
    clf = xgb.XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                            eval_metric="auc", early_stopping_rounds=50, random_state=RANDOM_STATE)
    clf.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    booster = clf.get_booster()
    best = clf.best_iteration + 1
    booster = booster[:best]   # keep only the trees early stopping chose
    return ScoringModel(kind="xgboost", name="XGBoost", booster=booster,
                        params={"n_trees": best})


ECE_TOLERANCE = 0.02   # recalibrate only if average |predicted - actual| across deciles exceeds 2 pts


def calibrate_if_needed(model: ScoringModel, valid: pd.DataFrame) -> float:
    """Check calibration on validation; fit isotonic knots only when scores are off.

    Returns the validation ECE of the raw scores. Isotonic regression on an already
    calibrated model just adds ties and noise, so it is skipped in that case.
    """
    raw, y = model.raw_score(valid), valid[TARGET].to_numpy()
    ece = expected_calibration_error(raw, y)
    if ece > ECE_TOLERANCE:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(raw, y)
        model.calib_x = iso.X_thresholds_.tolist()
        model.calib_y = iso.y_thresholds_.tolist()
    model.params["validation_ece"] = ece
    model.params["recalibrated"] = ece > ECE_TOLERANCE
    return ece


def train_and_select(split: Split) -> tuple[ScoringModel, list[ScoringModel]]:
    """Fit both models, pick the higher validation AUC, check its calibration."""
    candidates = [fit_logistic(split), fit_xgboost(split)]
    yv = split.valid[TARGET]
    best = max(candidates, key=lambda m: roc_auc_score(yv, m.raw_score(split.valid)))
    calibrate_if_needed(best, split.valid)
    return best, candidates
