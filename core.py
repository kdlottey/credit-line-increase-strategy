"""
Core logic for the Credit Line Increase Strategy project.
Shared by the Streamlit app (app.py). Every step is a small, readable function.
"""
import os
import numpy as np
import pandas as pd
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "UCI_Credit_Card.csv")

COLUMNS = (["limit_amt", "sex", "education", "marriage", "age"]
           + [f"pay_status_{i}" for i in range(1, 7)]   # 1 = most recent month
           + [f"bill_{i}" for i in range(1, 7)]         # statement balance
           + [f"paid_{i}" for i in range(1, 7)]         # amount paid
           + ["default_next_month"])

# Behavioral features only: no sex, age, marriage or education (fair lending)
FEATURES = ["limit_amt", "util_now", "util_avg6", "worst_delay_6m", "months_delayed_6m",
            "pay_ratio_6m", "pay_status_1", "pay_status_2", "bill_1", "paid_1"]

REASONS = {
    "R01": "Account currently past due",
    "R02": "Serious delinquency (2+ months) in last 6 months",
    "R03": "Account over credit limit",
    "R04": "Risk score above approval threshold",
    "R05": "Low repayment relative to balance",
}


# ---------------------------------------------------------------- 1. Load
def load_raw() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        raw = pd.read_csv(DATA_FILE)
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=350)
        raw = pd.concat([ds.data.features, ds.data.targets], axis=1)
    raw = raw.drop(columns=[c for c in raw.columns if c.upper() == "ID"])
    raw.columns = COLUMNS
    raw.insert(0, "account_id", np.arange(1, len(raw) + 1))
    return raw


# ---------------------------------------------------------------- 2. Controls
def run_controls(df: pd.DataFrame):
    """Data-quality checks. Returns (clean_df, control_log)."""
    log = []

    def add(check, value, status):
        log.append({"Check": check, "Result": int(value), "Status": status})

    add("Row count", len(df), "PASS" if len(df) > 0 else "FLAG")
    miss = df.isna().sum().sum()
    add("Missing values", miss, "PASS" if miss == 0 else "FLAG")
    dup = df.drop(columns="account_id").duplicated().sum()
    add("Duplicate account rows", dup, "PASS" if dup == 0 else "FLAG")
    bad_lim = (df.limit_amt <= 0).sum()
    add("Credit limit <= 0", bad_lim, "PASS" if bad_lim == 0 else "FLAG")
    bad_edu = (~df.education.isin([1, 2, 3, 4])).sum()
    add("Undocumented education codes (regrouped)", bad_edu, "PASS" if bad_edu == 0 else "FIXED")
    bad_mar = (~df.marriage.isin([1, 2, 3])).sum()
    add("Undocumented marriage codes (regrouped)", bad_mar, "PASS" if bad_mar == 0 else "FIXED")
    add("Accounts over limit", (df.bill_1 > df.limit_amt).sum(), "INFO")
    add("Accounts with credit balance", (df.bill_1 < 0).sum(), "INFO")

    clean = df.copy()
    clean.loc[~clean.education.isin([1, 2, 3, 4]), "education"] = 4
    clean.loc[~clean.marriage.isin([1, 2, 3]), "marriage"] = 3
    return clean, pd.DataFrame(log)


# ---------------------------------------------------------------- 3. SQL features
FEATURE_SQL = """
SELECT
    *,
    bill_1 / NULLIF(limit_amt, 0)                                         AS util_now,
    (bill_1 + bill_2 + bill_3 + bill_4 + bill_5 + bill_6) / 6.0
        / NULLIF(limit_amt, 0)                                            AS util_avg6,
    GREATEST(pay_status_1, pay_status_2, pay_status_3,
             pay_status_4, pay_status_5, pay_status_6)                    AS worst_delay_6m,
    (CAST(pay_status_1 > 0 AS INT) + CAST(pay_status_2 > 0 AS INT) +
     CAST(pay_status_3 > 0 AS INT) + CAST(pay_status_4 > 0 AS INT) +
     CAST(pay_status_5 > 0 AS INT) + CAST(pay_status_6 > 0 AS INT))       AS months_delayed_6m,
    LEAST(COALESCE((paid_1 + paid_2 + paid_3 + paid_4 + paid_5 + paid_6)
        / NULLIF(GREATEST(bill_1 + bill_2 + bill_3 + bill_4 + bill_5 + bill_6, 0), 0), 1), 1)
                                                                          AS pay_ratio_6m,
    CASE WHEN pay_status_1 IN (-2, -1) THEN 'Transactor'
         WHEN pay_status_1 = 0         THEN 'Revolver'
         WHEN pay_status_1 = 1         THEN '1 month late'
         ELSE '2+ months late' END                                        AS customer_type,
    CASE WHEN bill_1 / NULLIF(limit_amt, 0) < 0.10 THEN '1. <10%'
         WHEN bill_1 / NULLIF(limit_amt, 0) < 0.30 THEN '2. 10-30%'
         WHEN bill_1 / NULLIF(limit_amt, 0) < 0.50 THEN '3. 30-50%'
         WHEN bill_1 / NULLIF(limit_amt, 0) < 0.80 THEN '4. 50-80%'
         WHEN bill_1 / NULLIF(limit_amt, 0) <= 1.0 THEN '5. 80-100%'
         ELSE '6. Over limit' END                                         AS util_band
FROM raw_accounts
"""


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("raw_accounts", df)
    out = con.sql(FEATURE_SQL).df()
    con.close()
    return out


def segment_table(acct: pd.DataFrame, by: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("accounts", acct)
    out = con.sql(f"""
        SELECT {by},
               COUNT(*)                                            AS accounts,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)  AS pct_of_book,
               ROUND(100.0 * AVG(default_next_month), 2)           AS default_rate_pct,
               ROUND(AVG(util_now) * 100, 1)                       AS avg_util_pct
        FROM accounts GROUP BY {by} ORDER BY {by}
    """).df()
    con.close()
    return out


# ---------------------------------------------------------------- 4. Model
def ks_stat(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return float((tpr - fpr).max())


def train_models(acct: pd.DataFrame):
    X = acct[FEATURES].fillna(0)
    y = acct["default_next_month"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="auc", random_state=42).fit(X_tr, y_tr)

    rows, fpr_tpr = [], {}
    for name, m in [("Logistic Regression", logit), ("XGBoost", xgb)]:
        p_tr, p_te = m.predict_proba(X_tr)[:, 1], m.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p_te)
        rows.append({"Model": name, "Train AUC": roc_auc_score(y_tr, p_tr), "Test AUC": auc,
                     "Test KS": ks_stat(y_te, p_te), "Test Gini": 2 * auc - 1})
        fpr, tpr, _ = roc_curve(y_te, p_te)
        fpr_tpr[name] = (fpr, tpr)
    metrics = pd.DataFrame(rows)

    best_name = metrics.sort_values("Test AUC", ascending=False).iloc[0]["Model"]
    best = xgb if best_name == "XGBoost" else logit

    test = acct.loc[X_te.index].copy()
    test["pd"] = best.predict_proba(X_te)[:, 1]
    test["risk_band"] = pd.qcut(test["pd"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    test["increase_amt"] = np.minimum((test.limit_amt * 0.20).round(-3), 50_000)
    train_scores = best.predict_proba(X_tr)[:, 1]

    importance = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values()
    return dict(metrics=metrics, best_name=best_name, best=best, test=test,
                train_scores=train_scores, roc=fpr_tpr, importance=importance)


def band_table(test: pd.DataFrame) -> pd.DataFrame:
    t = (test.groupby("risk_band")
             .agg(accounts=("pd", "size"), avg_pd=("pd", "mean"),
                  actual_default_rate=("default_next_month", "mean"),
                  defaults=("default_next_month", "sum"))
             .reset_index())
    t["cum_pct_defaults_captured"] = t["defaults"][::-1].cumsum()[::-1] / t["defaults"].sum() * 100
    return t


# ---------------------------------------------------------------- 5. PLI strategy
def policy_gates(t: pd.DataFrame) -> pd.Series:
    """Hard rules every line increase must pass."""
    return (t.pay_status_1 <= 0) & (t.worst_delay_6m < 2) & (t.util_now <= 1.0)


def champion_mask(t, min_util=0.5):
    return policy_gates(t) & (t.worst_delay_6m <= 0) & (t.util_now >= min_util)


def challenger_mask(t, pd_cutoff, min_util=0.3):
    return policy_gates(t) & (t.util_now >= min_util) & (t.pd <= pd_cutoff)


def evaluate(t, mask, name):
    s = t[mask]
    used = s.util_now.clip(0, 1)
    new_bal = float((s.increase_amt * used).sum())
    exp_loss = float((s.pd * s.increase_amt * used).sum())
    return {"Strategy": name, "Eligible accounts": int(mask.sum()),
            "% of book": float(mask.mean() * 100),
            "Actual default rate %": float(s.default_next_month.mean() * 100) if len(s) else np.nan,
            "Expected new balance": new_bal, "Expected loss": exp_loss}


def best_cutoff(t, min_util=0.3, champ_min_util=0.5):
    """Largest approval volume with default rate <= champion's."""
    champ_dr = t.loc[champion_mask(t, champ_min_util), "default_next_month"].mean()
    rows = []
    for c in np.round(np.arange(0.05, 0.41, 0.01), 2):
        m = challenger_mask(t, c, min_util)
        if m.sum() >= 50:  # ignore tiny groups: their default rate is too noisy
            rows.append({"pd_cutoff": c, "eligible": int(m.sum()),
                         "default_rate_pct": t.loc[m, "default_next_month"].mean() * 100})
    sweep = pd.DataFrame(rows)
    ok = sweep[sweep.default_rate_pct <= champ_dr * 100]
    cut = float(ok.sort_values("eligible").iloc[-1]["pd_cutoff"]) if len(ok) else float(sweep.iloc[0]["pd_cutoff"])
    return cut, sweep


# ---------------------------------------------------------------- 6. RLI decision
def rli_decision(pay_status_1, worst_delay_6m, util_now, pd_score, pay_ratio_6m,
                 limit_amt, pd_cutoff):
    codes = []
    if pay_status_1 > 0:
        codes.append("R01")
    if worst_delay_6m >= 2:
        codes.append("R02")
    if util_now > 1.0:
        codes.append("R03")
    if pd_score > pd_cutoff:
        codes.append("R04")
    if pay_ratio_6m < 0.05:
        codes.append("R05")
    if codes:
        return "DECLINE", 0, codes
    return "APPROVE", int(min(round(limit_amt * 0.20, -3), 50_000)), []


def apply_rli(t, pd_cutoff):
    out = t.apply(lambda a: rli_decision(a.pay_status_1, a.worst_delay_6m, a.util_now, a.pd,
                                         a.pay_ratio_6m, a.limit_amt, pd_cutoff), axis=1)
    t = t.copy()
    t["rli_decision"] = [o[0] for o in out]
    t["rli_increase"] = [o[1] for o in out]
    t["rli_reasons"] = ["; ".join(o[2]) if o[2] else "-" for o in out]
    return t


# ---------------------------------------------------------------- 7. Monitoring
def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
