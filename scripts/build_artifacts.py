"""Offline pipeline: load -> controls -> features -> train -> tune -> score -> save artifacts.

Run once (and after any change to the logic):  python scripts/build_artifacts.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from clis import artifacts as art
from clis.config import ARTIFACT_DIR, FEATURES, SETTINGS, TARGET
from clis.data.controls import run_controls
from clis.data.loader import load_raw
from clis.decision.rli import decide_portfolio, reason_validation, rule_impact
from clis.errors import CLIError
from clis.features.builder import build_features, segment_table
from clis.model.evaluate import (add_risk_band, band_table, calibration_summary,
                                 metrics_table, roc_points)
from clis.model.train import split_accounts, train_and_select
from clis.monitoring.fairness import adverse_impact
from clis.monitoring.stability import feature_stability, psi, status
from clis.strategy.pli import challenger_result, champion_result, choose_cutoff, swap_sets


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    cfg = SETTINGS
    log("Loading data and running controls")
    raw = load_raw()
    clean, control_log = run_controls(raw)

    log("Building SQL features (current snapshot and one month earlier)")
    acct = build_features(clean, offset=0)
    prior = build_features(clean, offset=1)

    log("Training models (train), choosing and calibrating on validation")
    split = split_accounts(acct)
    model, candidates = train_and_select(split)
    model.save(ARTIFACT_DIR)

    for part in ("train", "valid", "holdout"):
        df = getattr(split, part)
        df["pd"] = model.predict_pd(df)
        df["split"] = part
    _, band_edges = add_risk_band(split.valid)
    valid, holdout = (add_risk_band(df, band_edges)[0] for df in (split.valid, split.holdout))

    log("Choosing the PLI cut-off on validation only")
    choice = choose_cutoff(valid, cfg)
    champ_m, champ = champion_result(holdout, cfg)
    chall_m, chall = challenger_result(holdout, cfg, choice.cutoff)

    log("Scoring RLI decisions on holdout")
    rli = decide_portfolio(holdout, model, cfg, choice.cutoff)
    holdout = holdout.join(rli)

    log("Monitoring: month-over-month stability and fairness")
    prior["pd"] = model.predict_pd(prior)
    acct_scored = pd.concat([split.train, valid, holdout]).sort_index()
    score_psi = psi(prior["pd"], acct_scored["pd"])
    build_vs_holdout = psi(split.train["pd"], holdout["pd"])
    fairness = adverse_impact(holdout, {"Champion": champ_m, "Challenger": chall_m})

    importance = (model.contributions(holdout).abs().mean()
                  .rename("mean_abs_contribution").rename_axis("feature").reset_index()
                  .sort_values("mean_abs_contribution"))

    log("Saving artifacts")
    keep = ["account_id", "split", "pd", "risk_band", *FEATURES, TARGET, "sex", "age",
            "customer_type", "util_band", "is_dormant", "rli_decision", "rli_increase",
            "rli_score_decline", "rli_codes"]
    art.save_table(acct_scored.reindex(columns=keep).sort_values("account_id"), "accounts")
    art.save_table(control_log, "control_log")
    art.save_table(metrics_table(candidates, split), "model_metrics")
    art.save_table(roc_points(candidates, split.holdout), "roc")
    art.save_table(band_table(holdout), "risk_bands")
    art.save_table(importance, "importance")
    art.save_table(segment_table(acct, "util_band"), "seg_util_band")
    art.save_table(segment_table(acct, "customer_type"), "seg_customer_type")
    art.save_table(choice.sweep, "cutoff_sweep_valid")
    art.save_table(swap_sets(holdout, champ_m, chall_m), "swap_sets")
    art.save_table(reason_validation(holdout, rli), "reason_validation")
    art.save_table(feature_stability(prior, acct), "feature_stability")
    art.save_table(fairness, "fairness")
    art.save_table(pd.DataFrame({"pd_prior": prior["pd"].to_numpy(),
                                 "pd_current": acct_scored["pd"].to_numpy()}),
                   "score_snapshots")

    summary = {
        "model": model.name, "n_trees": model.params.get("n_trees"),
        "calibration": calibration_summary(model, holdout),
        "cutoff": choice.cutoff, "cutoff_reason": choice.reason,
        "champion_default_rate_valid": choice.champion_default_rate,
        "champion": champ, "challenger": chall,
        "r05_rule_test": rule_impact(holdout, rli, cfg.policy.tested_pay_ratio_rule),
        "rli_approval_rate": float((rli.rli_decision == "APPROVE").mean() * 100),
        "psi_month_over_month": score_psi, "psi_month_over_month_status": status(score_psi),
        "psi_build_vs_holdout": build_vs_holdout,
        "fairness_min_air": float(fairness.AIR.min()),
        "split_sizes": {p: len(getattr(split, p)) for p in ("train", "valid", "holdout")},
    }
    art.save_json(summary, "summary")
    log(f"Done. Cut-off {choice.cutoff:.2f} ({choice.reason}); "
        f"challenger {chall['Eligible accounts']} vs champion {champ['Eligible accounts']} eligible")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CLIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
