"""A portable scoring model: no pickles, so artifacts load on any library version.

XGBoost is stored as its native JSON; logistic regression as plain coefficients;
calibration as isotonic knots applied with np.interp.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xgboost as xgb

from clis.config import FEATURES
from clis.errors import ArtifactError

ModelKind = Literal["xgboost", "logistic"]


@dataclass
class ScoringModel:
    kind: ModelKind
    name: str
    params: dict[str, Any] = field(default_factory=dict)   # logistic: mean, scale, coef, intercept
    calib_x: list[float] = field(default_factory=list)     # isotonic knots (raw score -> PD)
    calib_y: list[float] = field(default_factory=list)
    booster: xgb.Booster | None = None

    # ------------------------------------------------------------ scoring
    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        missing = [f for f in FEATURES if f not in X.columns]
        if missing:
            raise ArtifactError(f"Scoring input is missing features: {missing}")
        return X[FEATURES].astype(float).fillna(0.0).to_numpy()

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        m = self._matrix(X)
        if self.kind == "xgboost":
            if self.booster is None:
                raise ArtifactError("XGBoost model has no booster loaded.")
            return self.booster.predict(xgb.DMatrix(m, feature_names=FEATURES))
        p = self.params
        z = ((m - np.array(p["mean"])) / np.array(p["scale"])) @ np.array(p["coef"]) + p["intercept"]
        return 1.0 / (1.0 + np.exp(-z))

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.raw_score(X)
        if not self.calib_x:
            return raw
        return np.interp(raw, self.calib_x, self.calib_y)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to the risk score (log-odds scale), for reason codes."""
        m = self._matrix(X)
        if self.kind == "xgboost":
            c = self.booster.predict(xgb.DMatrix(m, feature_names=FEATURES), pred_contribs=True)[:, :-1]
        else:
            p = self.params
            c = ((m - np.array(p["mean"])) / np.array(p["scale"])) * np.array(p["coef"])
        return pd.DataFrame(c, columns=FEATURES, index=X.index)

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"kind": self.kind, "name": self.name, "params": self.params,
                "calib_x": self.calib_x, "calib_y": self.calib_y, "features": FEATURES}
        (directory / "model_meta.json").write_text(json.dumps(meta, indent=1))
        if self.kind == "xgboost":
            self.booster.save_model(str(directory / "model_xgb.json"))

    @classmethod
    def load(cls, directory: Path) -> "ScoringModel":
        meta_path = directory / "model_meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except FileNotFoundError as exc:
            raise ArtifactError(f"{meta_path} not found. Run `python scripts/build_artifacts.py`.") from exc
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"{meta_path} is corrupt: {exc}") from exc
        if meta.get("features") != FEATURES:
            raise ArtifactError("Saved model was trained on different features. Rebuild artifacts.")

        model = cls(kind=meta["kind"], name=meta["name"], params=meta["params"],
                    calib_x=meta["calib_x"], calib_y=meta["calib_y"])
        if model.kind == "xgboost":
            booster = xgb.Booster()
            try:
                booster.load_model(str(directory / "model_xgb.json"))
            except xgb.core.XGBoostError as exc:
                raise ArtifactError(f"Could not load XGBoost model: {exc}") from exc
            model.booster = booster
        return model
