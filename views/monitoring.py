"""Monitoring and controls: data quality, stability over time, fairness, limitations."""
import streamlit as st

from clis.config import FEATURE_LABELS
from clis.monitoring.fairness import FOUR_FIFTHS
from clis.monitoring.stability import THRESHOLDS
from ui import charts, loaders as L

s = L.guard(L.summary)

st.subheader("Data-quality controls")
st.write("Run on every data refresh before anyone uses the numbers. Each check records the action taken.")
controls = L.guard(L.table, "control_log").rename(columns=str.capitalize)
st.dataframe(controls, hide_index=True, width="stretch",
             column_config={"Action": st.column_config.TextColumn(width="large")})

st.subheader("Score stability over time")
st.write("The data has 6 months of history and the model uses 5, so every account can be rebuilt **as it "
         "looked one month earlier**. Comparing last month's score distribution with this month's is a real "
         "out-of-time stability check.")
snap = L.guard(L.table, "score_snapshots")
c = st.columns(2)
c[0].metric("Score PSI: last month vs this month", f"{s['psi_month_over_month']:.4f}",
            s["psi_month_over_month_status"], delta_color="off")
c[1].metric("Score PSI: build vs holdout (sanity check)", f"{s['psi_build_vs_holdout']:.4f}",
            help="Random split from the same month, so this is near zero by construction. "
                 "It only confirms the split is unbiased; it is not monitoring.")
l, r = st.columns([3, 2])
l.plotly_chart(charts.overlaid_hist(snap.pd_prior, snap.pd_current, "Last month", "This month",
                                    "Score distribution, month over month"), width="stretch")
csi = L.guard(L.table, "feature_stability")
csi = csi.assign(Feature=csi.Feature.map(FEATURE_LABELS))
r.markdown("**Which inputs moved? (CSI per feature)**")
r.dataframe(csi.style.format({"CSI": "{:.4f}"}), hide_index=True, width="stretch")
st.caption(f"< {THRESHOLDS[0]} stable · {THRESHOLDS[0]}–{THRESHOLDS[1]} monitor · > {THRESHOLDS[1]} review the strategy.")

st.subheader("Fair-lending check")
st.write("Protected attributes are kept out of the model, but that alone doesn't guarantee fair outcomes. "
         "Here they are used only to test approval rates. The **adverse impact ratio (AIR)** divides each "
         f"group's approval rate by the best-treated group's; below {FOUR_FIFTHS:.2f} (the four-fifths rule) "
         "warrants review.")
fair = L.guard(L.table, "fairness")
pivot = fair.pivot_table(index=["Attribute", "Group"], columns="Strategy", values="AIR").reset_index()[["Attribute", "Group", "Champion", "Challenger"]]
st.dataframe(pivot.style.format({"Champion": "{:.2f}", "Challenger": "{:.2f}"})
             .highlight_between(subset=["Champion", "Challenger"], left=0, right=FOUR_FIFTHS - 1e-9,
                                color="#F6D6D9"), hide_index=True, width="stretch")
flags = fair[fair.AIR < FOUR_FIFTHS].groupby("Strategy").size()
st.caption(f"Groups below {FOUR_FIFTHS:.2f}: champion {int(flags.get('Champion', 0))}, "
           f"challenger {int(flags.get('Challenger', 0))}. The model-based challenger treats groups more evenly "
           "than the rule-based champion. Small groups (e.g. 60+) have noisy rates, so a production review would "
           "also test statistical significance and look for proxy variables.")

st.subheader("Limitations")
st.markdown(
    "- Target is default **next month**; production line-increase models predict over 12+ months. The profit "
    "model treats this PD as the default probability for the 12-month horizon.\n"
    "- No income, months-on-book or bureau data, so increase amounts are simple risk tiers.\n"
    "- Profit assumes customers use new line like the old, with fixed APR, LGD and drawdown at default "
    "(adjustable on the PLI tab).\n"
    "- One snapshot of 2005 Taiwan data; true out-of-time validation would need later cohorts.\n"
    "- A real rollout would A/B test the challenger on a small random group and track 30/60/90-day "
    "delinquency by vintage.")
