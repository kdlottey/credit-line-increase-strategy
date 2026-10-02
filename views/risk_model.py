"""Risk model: discrimination, calibration and what drives the score."""
import streamlit as st

from clis.config import FEATURE_LABELS
from ui import charts, loaders as L

s = L.guard(L.summary)
metrics = L.guard(L.table, "model_metrics")
cal = s["calibration"]

st.subheader("Probability-of-default (PD) model")
st.write(f"Predicts which accounts will default next month from **behavioral features only** over a "
         f"5-month lookback. Sex, age, marital status and education are never model inputs. Data is split "
         f"**{s['split_sizes']['train']:,} train / {s['split_sizes']['valid']:,} validation / "
         f"{s['split_sizes']['holdout']:,} holdout**: the model is chosen, early-stopped and checked for calibration "
         "on validation, and every number reported below comes from the untouched holdout.")

num = [c for c in metrics.columns if c != "Model"]
st.dataframe(metrics.style.format({c: "{:.3f}" for c in num}), hide_index=True, width="stretch")
st.caption(f"Selected: **{s['model']}** ({s['n_trees']} trees after early stopping), chosen on validation AUC. "
           "AUC = chance a random defaulter is scored riskier than a random non-defaulter; "
           "KS = maximum separation between the two groups.")

c = st.columns(3)
c[0].metric("Brier score (holdout)", f"{cal['brier']:.3f}")
c[1].metric("Calibration error (holdout ECE)", f"{cal['holdout_ece'] * 100:.1f} pts")
c[2].metric("Average PD vs actual default rate", f"{cal['mean_pd'] * 100:.1f}% vs {cal['actual_rate'] * 100:.1f}%")
st.caption(("Raw scores were already well calibrated on validation (ECE "
            f"{cal['validation_ece'] * 100:.1f} pts ≤ 2 pts), so no recalibration was applied."
            if not cal["recalibrated"] else
            "Scores were recalibrated with isotonic regression fitted on validation.")
           + " This matters because the strategy uses PD as a probability, not just a ranking.")

l, r = st.columns(2)
l.plotly_chart(charts.roc(L.guard(L.table, "roc")), width="stretch")
r.plotly_chart(charts.predicted_vs_actual(L.guard(L.table, "risk_bands")), width="stretch")

imp = L.guard(L.table, "importance")
imp = imp.assign(feature=imp.feature.map(FEATURE_LABELS)).set_index("feature")["mean_abs_contribution"]
st.plotly_chart(charts.hbar(imp, "What drives risk? (mean |SHAP contribution|, holdout)",
                            "Average impact on log-odds of default"), width="stretch")
st.caption("SHAP contributions show how much each feature moves an individual account's score. "
           "The same contributions generate the customer-facing decline reasons in the RLI engine.")
