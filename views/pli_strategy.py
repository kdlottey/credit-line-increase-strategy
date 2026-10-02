"""Proactive line increase: champion vs challenger with a profit objective."""
import numpy as np
import pandas as pd
import streamlit as st

from clis.config import SETTINGS
from ui import charts, loaders as L

s = L.guard(L.summary)
E, P, Z = SETTINGS.economics, SETTINGS.policy, SETTINGS.sizing

st.subheader("Proactive Line Increase: champion vs challenger")
st.write(
    f"**Champion** (current rules): never late in 5 months, utilization ≥ {P.champion_min_util:.0%}, "
    f"flat increase of {Z.flat_pct:.0%} of limit.  \n"
    f"**Challenger** (proposed): same safety gates, but the **PD model** replaces the payment rule, it targets "
    f"utilization ≥ {P.challenger_min_util:.0%}, and the increase is **tiered by risk** (30% / 20% / 10% of limit). "
    f"Both capped at NT${Z.cap:,}.  \n"
    "The PD cut-off is chosen on **validation** data to maximize expected 12-month profit, without letting the "
    "default rate exceed the champion's. All results below are on the separate **holdout** set.")

with st.expander("Business assumptions and overrides"):
    a1, a2, a3, a4 = st.columns(4)
    apr = a1.slider("APR", 0.10, 0.30, E.apr, 0.01, format="%.2f")
    lgd = a2.slider("Loss given default", 0.50, 1.00, E.lgd, 0.05)
    ch_util = a3.slider("Challenger min utilization", 0.0, 0.8, P.challenger_min_util, 0.05)
    cp_util = a4.slider("Champion min utilization", 0.0, 0.8, P.champion_min_util, 0.05)
    override = st.toggle("Override the PD cut-off manually")
    manual = st.slider("PD cut-off", 0.02, 0.80, float(s["cutoff"]), 0.01, disabled=not override)
    st.caption(f"Profit model: new balance × (APR − {E.cost_of_funds:.0%} funding cost) × (1 − PD) "
               f"− PD × LGD × increase × {E.ccf:.0%} drawdown at default.")

sc = L.guard(L.pli_scenario, apr, lgd, ch_util, cp_util, manual if override else None)
ce, cl = sc["champion"], sc["challenger"]
if sc["best"] is not None and not override:
    st.success(f"Chosen PD cut-off **{sc['used']:.2f}** ({sc['reason']}).")
elif sc["best"] is None:
    st.warning(sc["reason"])


def pct(a: float, b: float) -> str:
    return f"{(a / b - 1) * 100:+.0f}% vs champion" if b else "n/a"


c = st.columns(4)
c[0].metric("Eligible accounts", f"{cl['Eligible accounts']:,}", pct(cl["Eligible accounts"], ce["Eligible accounts"]))
c[1].metric("Default rate", f"{cl['Default rate %']:.2f}%",
            f"{cl['Default rate %'] - ce['Default rate %']:+.2f} pts vs champion", delta_color="inverse")
c[2].metric("Expected 12-mo profit", f"NT${cl['Expected profit'] / 1e6:.2f}M",
            pct(cl["Expected profit"], ce["Expected profit"]))
c[3].metric("Credit line granted", f"NT${cl['Line granted'] / 1e6:.1f}M", pct(cl["Line granted"], ce["Line granted"]))

res = pd.DataFrame([ce, cl])
money = ["Line granted", "New balance", "Expected profit", "Realized profit"]
st.dataframe(res.style.format({"% of book": "{:.1f}", "Default rate %": "{:.2f}",
                               **{m: "{:,.0f}" for m in money}}), hide_index=True, width="stretch")
st.caption("Expected profit uses predicted PD; realized profit replaces it with what actually happened on holdout.")

l, r = st.columns(2)
l.plotly_chart(charts.profit_curve(sc["sweep"], sc["used"]), width="stretch")
r.plotly_chart(charts.tradeoff(sc["holdout_sweep"], ce, cl), width="stretch")

st.markdown("**Swap-set analysis (holdout)**")
st.dataframe(sc["swap"].style.format({"Default rate %": "{:.2f}"}, na_rep="-"), hide_index=True, width="stretch")
sw = sc["swap"].set_index("Group")["Default rate %"]
if not np.isnan(sw.iloc[1]) and not np.isnan(sw.iloc[2]):
    st.caption(f"The challenger adds customers who default at {sw.iloc[1]:.1f}% and drops customers the old "
               f"rules approved who default at {sw.iloc[2]:.1f}%: it swaps worse accounts for better ones.")
