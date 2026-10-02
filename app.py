"""
Credit Line Increase Strategy: interactive dashboard
Run locally:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="Credit Line Increase Strategy", page_icon="📈", layout="wide")

BLUE, RED, GREY, GREEN = "#1F4E79", "#B23A48", "#9AA5B1", "#2E7D5B"
CHART = dict(template="simple_white", margin=dict(l=10, r=10, t=40, b=10), height=340)


# ------------------------------------------------------------------ pipeline (runs once)
@st.cache_resource(show_spinner="Loading data and training models (first load only)...")
def pipeline():
    raw = core.load_raw()
    clean, control_log = core.run_controls(raw)
    acct = core.build_features(clean)
    model = core.train_models(acct)
    cut, sweep = core.best_cutoff(model["test"])
    return dict(acct=acct, control_log=control_log, model=model, best_cut=cut, sweep=sweep)


P = pipeline()
acct, M = P["acct"], P["model"]
test = M["test"]

# ------------------------------------------------------------------ header
st.title("Credit Line Increase Strategy & Risk Analysis")
st.caption("Who should get a higher credit limit? Proactive (PLI) and reactive (RLI) line-increase "
           "decisions on 30,000 real credit card accounts, using SQL, a probability-of-default model and "
           "champion vs challenger testing. Data: UCI Default of Credit Card Clients (Taiwan, amounts in NT$).")

tabs = st.tabs(["📊 Portfolio", "🎯 Risk model", "⚖️ PLI strategy", "🧾 RLI decision engine", "🛡️ Monitoring & controls"])

# ================================================================== 1. PORTFOLIO
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Accounts", f"{len(acct):,}")
    c[1].metric("Default rate (next month)", f"{acct.default_next_month.mean()*100:.1f}%")
    c[2].metric("Avg utilization", f"{acct.util_now.clip(-1, 2).mean()*100:.1f}%")
    c[3].metric("Revolvers (carry a balance)", f"{(acct.customer_type=='Revolver').mean()*100:.1f}%")

    util = core.segment_table(acct, "util_band")
    ctype = core.segment_table(acct, "customer_type").sort_values("default_rate_pct")

    l, r = st.columns(2)
    fig = px.bar(util, x="util_band", y="default_rate_pct", text="default_rate_pct",
                 hover_data=["accounts", "pct_of_book"], color_discrete_sequence=[BLUE])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(title="Default rate by utilization band", xaxis_title="Utilization (balance ÷ limit)",
                      yaxis_title="Default rate (%)", **CHART)
    l.plotly_chart(fig, width="stretch")

    fig = px.bar(ctype, x="customer_type", y="default_rate_pct", text="default_rate_pct",
                 hover_data=["accounts", "pct_of_book"], color_discrete_sequence=[BLUE])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(title="Default rate by customer type (latest month)", xaxis_title="",
                      yaxis_title="Default rate (%)", **CHART)
    r.plotly_chart(fig, width="stretch")

    lo = util.iloc[0]["default_rate_pct"]
    hi = util[util.util_band == "5. 80-100%"]["default_rate_pct"].iloc[0]
    rev = ctype.set_index("customer_type").loc["Revolver", "default_rate_pct"]
    late = ctype.set_index("customer_type").loc["2+ months late", "default_rate_pct"]
    st.info(f"**Key insight:** default risk climbs from **{lo:.1f}%** for accounts using under 10% of their "
            f"line to **{hi:.1f}%** for accounts at 80–100%. Customers 2+ months late default at "
            f"**{late/rev:.1f}×** the rate of revolvers. High utilization signals both *need* and *stress*, "
            f"so it can't drive line increases on its own: it has to be combined with repayment behavior.")

    with st.expander("See the SQL behind these features"):
        st.code(core.FEATURE_SQL.strip(), language="sql")

# ================================================================== 2. RISK MODEL
with tabs[1]:
    st.subheader("Probability-of-default (PD) model")
    st.write("Predicts which accounts will default next month using **behavioral features only**. "
             "Sex, age, marital status and education are excluded on fair-lending grounds.")
    st.dataframe(M["metrics"].style.format({c: "{:.3f}" for c in ["Train AUC", "Test AUC", "Test KS", "Test Gini"]}),
                 hide_index=True, width="stretch")
    st.caption(f"Selected model: **{M['best_name']}** (highest test AUC). AUC = chance a random defaulter is "
               "scored riskier than a random non-defaulter. KS = maximum separation between the two groups.")

    l, r = st.columns(2)
    fig = go.Figure()
    for (name, (fpr, tpr)), col in zip(M["roc"].items(), [BLUE, RED]):
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, line=dict(color=col)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=GREY, dash="dash")))
    fig.update_layout(title="ROC curve (test set)", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", **CHART)
    l.plotly_chart(fig, width="stretch")

    bands = core.band_table(test)
    fig = px.bar(bands, x="risk_band", y=bands.actual_default_rate * 100, color_discrete_sequence=[BLUE],
                 hover_data={"accounts": True, "avg_pd": ":.3f"})
    fig.update_layout(title="Actual default rate by risk band (1 = safest)", xaxis_title="Risk band",
                      yaxis_title="Actual default rate (%)", xaxis=dict(dtick=1), **CHART)
    r.plotly_chart(fig, width="stretch")

    imp = M["importance"]
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[BLUE])
    fig.update_layout(title="What drives risk? (XGBoost feature importance)", xaxis_title="Importance",
                      yaxis_title="", **CHART)
    st.plotly_chart(fig, width="stretch")

# ================================================================== 3. PLI STRATEGY
with tabs[2]:
    st.subheader("Proactive Line Increase: champion vs challenger")
    st.write("**Champion** (current rules): no late payment in 6 months and utilization ≥ 50%.  \n"
             "**Challenger** (proposed): the same safety gates, but uses the **PD model** instead of a hard "
             "payment rule, and targets customers using ≥ 30% of their line.  \n"
             "Both are tested on accounts the model never saw. Increase offered = 20% of limit, capped at NT$50,000.")

    s1, s2, s3 = st.columns(3)
    cut = s1.slider("Challenger PD cut-off", 0.05, 0.40, float(P["best_cut"]), 0.01,
                    help="Approve only accounts whose predicted default probability is at or below this.")
    ch_util = s2.slider("Challenger min utilization", 0.0, 0.8, 0.3, 0.05)
    cp_util = s3.slider("Champion min utilization", 0.0, 0.8, 0.5, 0.05)

    champ, chall = core.champion_mask(test, cp_util), core.challenger_mask(test, cut, ch_util)
    res = pd.DataFrame([core.evaluate(test, champ, "Champion (rules)"),
                        core.evaluate(test, chall, f"Challenger (PD ≤ {cut:.2f})")])

    c = st.columns(4)
    ce, cl = res.iloc[0], res.iloc[1]
    lift = (cl["Eligible accounts"] / ce["Eligible accounts"] - 1) * 100 if ce["Eligible accounts"] else np.nan
    c[0].metric("Champion eligible", f"{int(ce['Eligible accounts']):,}")
    c[1].metric("Challenger eligible", f"{int(cl['Eligible accounts']):,}", f"{lift:+.0f}% vs champion")
    c[2].metric("Champion default rate", f"{ce['Actual default rate %']:.2f}%")
    c[3].metric("Challenger default rate", f"{cl['Actual default rate %']:.2f}%",
                f"{cl['Actual default rate %'] - ce['Actual default rate %']:+.2f} pts", delta_color="inverse")

    st.dataframe(res.style.format({"% of book": "{:.1f}", "Actual default rate %": "{:.2f}",
                                   "Expected new balance": "{:,.0f}", "Expected loss": "{:,.0f}"}),
                 hide_index=True, width="stretch")

    l, r = st.columns([3, 2])
    _, sweep = core.best_cutoff(test, ch_util, cp_util)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep.eligible, y=sweep.default_rate_pct, mode="lines+markers",
                             name="Challenger at each cut-off", line=dict(color=BLUE),
                             customdata=sweep.pd_cutoff, hovertemplate="cut-off %{customdata}<br>%{x:,} accounts<br>%{y:.2f}%"))
    fig.add_trace(go.Scatter(x=[ce["Eligible accounts"]], y=[ce["Actual default rate %"]], mode="markers",
                             name="Champion", marker=dict(color=RED, size=14, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[cl["Eligible accounts"]], y=[cl["Actual default rate %"]], mode="markers",
                             name="Selected challenger", marker=dict(color=GREEN, size=12)))
    fig.update_layout(title="Volume vs risk trade-off", xaxis_title="Accounts eligible",
                      yaxis_title="Actual default rate (%)", **CHART)
    l.plotly_chart(fig, width="stretch")

    swap = pd.DataFrame({
        "Group": ["Both approve", "Swap-in (challenger only)", "Swap-out (champion only)"],
        "Accounts": [int((champ & chall).sum()), int((~champ & chall).sum()), int((champ & ~chall).sum())],
        "Default rate %": [test.loc[m, "default_next_month"].mean() * 100 if m.sum() else np.nan
                           for m in [champ & chall, ~champ & chall, champ & ~chall]]})
    r.markdown("**Swap-set analysis**")
    r.dataframe(swap.style.format({"Default rate %": "{:.2f}"}), hide_index=True, width="stretch")
    r.caption("Swap-ins are customers the new strategy adds. Risk teams check that their default rate is acceptable.")

# ================================================================== 4. RLI ENGINE
with tabs[3]:
    st.subheader("Reactive Line Increase: decision engine with reason codes")
    st.write("A customer **asks** for a higher limit. Every decision returns reason codes, so declines can be "
             "explained to the customer (adverse-action rules) and audited.")

    PRESETS = {
        "Good revolver": dict(limit=200000, bal=90000, avg_bal=85000, pay=9000, avg_pay=8000, status=[0] * 6),
        "Recently late": dict(limit=150000, bal=120000, avg_bal=110000, pay=2000, avg_pay=3000, status=[2, 1, 0, 0, 0, 0]),
        "Maxed out": dict(limit=50000, bal=56000, avg_bal=49000, pay=1500, avg_pay=2000, status=[0, 0, 1, 0, 0, 0]),
    }

    def load_preset(name):
        p = PRESETS[name]
        for k in ["limit", "bal", "avg_bal", "pay", "avg_pay"]:
            st.session_state[f"rli_{k}"] = p[k]
        for i, v in enumerate(p["status"]):
            st.session_state[f"rli_s{i}"] = v

    if "rli_limit" not in st.session_state:
        load_preset("Good revolver")

    b = st.columns(len(PRESETS) + 2)
    b[0].markdown("**Try an example:**")
    for i, name in enumerate(PRESETS):
        b[i + 1].button(name, on_click=load_preset, args=(name,), width="stretch")

    STATUS = {-2: "No balance", -1: "Paid in full", 0: "Paid min / revolved", 1: "1 month late",
              2: "2 months late", 3: "3 months late", 4: "4+ months late"}
    with st.form("rli"):
        a1, a2, a3 = st.columns(3)
        limit = a1.number_input("Current credit limit (NT$)", 10000, 1000000, step=10000, key="rli_limit")
        bal = a2.number_input("Current balance (NT$)", -10000, 1200000, step=1000, key="rli_bal")
        pay = a3.number_input("Last payment (NT$)", 0, 1000000, step=500, key="rli_pay")
        a4, a5 = st.columns(2)
        avg_bal = a4.number_input("Typical monthly balance, previous 5 months (NT$)", -10000, 1200000, step=1000, key="rli_avg_bal")
        avg_pay = a5.number_input("Typical monthly payment, previous 5 months (NT$)", 0, 1000000, step=500, key="rli_avg_pay")
        st.markdown("Repayment status, most recent month first")
        sc = st.columns(6)
        status = [sc[i].selectbox(f"Month -{i+1}", list(STATUS), format_func=STATUS.get, key=f"rli_s{i}")
                  for i in range(6)]
        submitted = st.form_submit_button("Decide", type="primary")

    row = {"account_id": 0, "limit_amt": limit, "sex": 1, "education": 1, "marriage": 1, "age": 30,
           **{f"pay_status_{i+1}": status[i] for i in range(6)},
           "bill_1": bal, **{f"bill_{i}": avg_bal for i in range(2, 7)},
           "paid_1": pay, **{f"paid_{i}": avg_pay for i in range(2, 7)}, "default_next_month": 0}
    f = core.build_features(pd.DataFrame([row])).iloc[0]
    pd_score = float(M["best"].predict_proba(pd.DataFrame([f[core.FEATURES]]).astype(float))[:, 1][0])
    decision, inc, codes = core.rli_decision(f.pay_status_1, f.worst_delay_6m, f.util_now, pd_score,
                                             f.pay_ratio_6m, limit, P["best_cut"])

    d1, d2, d3 = st.columns(3)
    d1.metric("Decision", decision)
    d2.metric("Increase", f"NT${inc:,}")
    d3.metric("Predicted default probability", f"{pd_score*100:.1f}%",
              help=f"Approval threshold: {P['best_cut']*100:.0f}% (the PLI challenger cut-off)")
    if decision == "APPROVE":
        st.success(f"Approved: new limit NT${limit + inc:,}. Utilization {f.util_now*100:.0f}%, "
                   f"no serious delinquency, risk below threshold.")
    else:
        st.error("Declined. Reasons:\n" + "\n".join(f"- **{c}** {core.REASONS[c]}" for c in codes))

    st.divider()
    st.markdown("**Across the test portfolio**")
    rli = core.apply_rli(test, P["best_cut"])
    l, r = st.columns([1, 2])
    l.metric("RLI approval rate", f"{(rli.rli_decision=='APPROVE').mean()*100:.1f}%")
    rc = (rli.loc[rli.rli_decision == "DECLINE", "rli_reasons"].str.split("; ").explode()
              .map(lambda c: f"{c} {core.REASONS[c]}").value_counts().sort_values())
    fig = px.bar(x=rc.values, y=rc.index, orientation="h", color_discrete_sequence=[RED])
    fig.update_layout(title="How often each decline reason is cited", xaxis_title="Declines citing reason",
                      yaxis_title="", **CHART)
    r.plotly_chart(fig, width="stretch")

# ================================================================== 5. MONITORING
with tabs[4]:
    st.subheader("Data-quality controls")
    st.write("Run on every data refresh before anyone uses the numbers. FIXED = issue found and corrected; "
             "INFO = tracked, no action needed.")
    st.dataframe(P["control_log"], hide_index=True, width="stretch")

    st.subheader("Score stability (PSI)")
    v = core.psi(M["train_scores"], test["pd"])
    status = "Stable" if v < 0.10 else "Monitor" if v < 0.25 else "Significant shift"
    st.metric("Population Stability Index: build vs test", f"{v:.4f}", status, delta_color="off")
    st.caption("PSI compares today's score distribution with the one the model was built on. "
               "< 0.10 stable · 0.10–0.25 monitor · > 0.25 review the strategy.")

    st.subheader("Limitations")
    st.markdown("- Target is default **next month**; production line-increase models predict 12+ months.\n"
                "- No income, months-on-book or bureau data, so the increase amount is a simple rule.\n"
                "- Expected loss assumes 100% loss given default and that customers use new line like the old.\n"
                "- A real rollout would A/B test the challenger on a small random group first.")

st.caption("Built by Keshav Dhiman · Python, DuckDB SQL, scikit-learn, XGBoost, Streamlit")
