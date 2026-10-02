"""Reactive line increase: a customer asks for more credit; decide with specific reasons."""
import pandas as pd
import streamlit as st

from clis.config import LOOKBACK, SETTINGS
from clis.decision.reasons import ALL_REASONS
from clis.decision.rli import decide_one
from clis.features.builder import build_features
from ui import charts, loaders as L

s = L.guard(L.summary)
CUTOFF = float(s["cutoff"])

st.subheader("Reactive Line Increase: decision engine with reason codes")
st.write("A customer **asks** for a higher limit. Policy rules run first, then the PD model. Every decline "
         "lists specific reasons: when the score causes a decline, the reasons are the factors that pushed "
         "that customer's score up (from SHAP), never just \"score too high\". That is what adverse-action "
         "notices require.")

PRESETS = {
    "Good revolver": dict(limit=200000, bal=90000, avg_bal=85000, pay=9000, avg_pay=8000, status=[0] * LOOKBACK),
    "Recently late": dict(limit=150000, bal=120000, avg_bal=110000, pay=2000, avg_pay=3000, status=[2, 1, 0, 0, 0]),
    "Maxed out": dict(limit=50000, bal=56000, avg_bal=49000, pay=1500, avg_pay=2000, status=[0, 0, 1, 0, 0]),
    "Low payer": dict(limit=80000, bal=70000, avg_bal=68000, pay=1000, avg_pay=1200, status=[0, 0, 0, 0, 1]),
}
STATUS = {-2: "No balance", -1: "Paid in full", 0: "Paid min / revolved", 1: "1 month late",
          2: "2 months late", 3: "3 months late", 4: "4+ months late"}


def load_preset(name: str) -> None:
    p = PRESETS[name]
    for k in ["limit", "bal", "avg_bal", "pay", "avg_pay"]:
        st.session_state[f"rli_{k}"] = p[k]
    for i, v in enumerate(p["status"]):
        st.session_state[f"rli_s{i}"] = v


if "rli_limit" not in st.session_state:
    load_preset("Good revolver")

b = st.columns(len(PRESETS) + 1)
b[0].markdown("**Try an example:**")
for i, name in enumerate(PRESETS):
    b[i + 1].button(name, on_click=load_preset, args=(name,), width="stretch")

with st.form("rli"):
    a1, a2, a3 = st.columns(3)
    limit = a1.number_input("Current credit limit (NT$)", 10000, 1000000, step=10000, key="rli_limit")
    bal = a2.number_input("Current balance (NT$)", -10000, 1200000, step=1000, key="rli_bal")
    pay = a3.number_input("Last payment (NT$)", 0, 1000000, step=500, key="rli_pay")
    a4, a5 = st.columns(2)
    avg_bal = a4.number_input("Typical monthly balance, previous 4 months (NT$)", -10000, 1200000,
                              step=1000, key="rli_avg_bal")
    avg_pay = a5.number_input("Typical monthly payment, previous 4 months (NT$)", 0, 1000000,
                              step=500, key="rli_avg_pay")
    st.markdown("Repayment status, most recent month first")
    sc = st.columns(LOOKBACK)
    status = [sc[i].selectbox(f"Month -{i + 1}", list(STATUS), format_func=STATUS.get, key=f"rli_s{i}")
              for i in range(LOOKBACK)]
    st.form_submit_button("Decide", type="primary")
st.caption("Months 2–5 use the typical balance and payment you enter; a real system would read each month "
           "from the account history.")


@st.cache_data(show_spinner=False)
def decide(limit: int, bal: int, pay: int, avg_bal: int, avg_pay: int, status: tuple[int, ...]):
    row = {"account_id": 0, "limit_amt": limit, "sex": 0, "education": 0, "marriage": 0, "age": 0,
           "default_next_month": 0, "bill_1": bal, "paid_1": pay,
           **{f"pay_status_{i + 1}": status[min(i, LOOKBACK - 1)] for i in range(6)},
           **{f"bill_{i}": avg_bal for i in range(2, 7)}, **{f"paid_{i}": avg_pay for i in range(2, 7)}}
    feats = build_features(pd.DataFrame([row]))
    return decide_one(feats, L.model(), SETTINGS, CUTOFF)


d = L.guard(decide, int(limit), int(bal), int(pay), int(avg_bal), int(avg_pay), tuple(status))
c = st.columns(3)
c[0].metric("Decision", d.decision)
c[1].metric("Increase", f"NT${d.increase:,}")
c[2].metric("Predicted default probability", f"{d.pd * 100:.1f}%",
            help=f"Approval threshold: {CUTOFF * 100:.0f}% (the PLI cut-off chosen on validation)")
if d.decision == "APPROVE":
    st.success(f"Approved: new limit NT${limit + d.increase:,}. No policy rule failed and predicted risk "
               f"is below the {CUTOFF:.0%} threshold. Lower-risk customers receive a larger share of their limit.")
else:
    st.error("Declined. Reasons:\n" + "\n".join(f"- **{r}**" for r in d.reasons))

st.divider()
st.markdown("**Across the holdout portfolio**")
hold = L.guard(L.split, "holdout")
val = L.guard(L.table, "reason_validation")
pct_cols = [c for c in val.columns if "%" in c]
val[pct_cols] = val[pct_cols].astype(float)
l, r = st.columns([1, 2])
l.metric("RLI approval rate", f"{s['rli_approval_rate']:.1f}%")
approved_rate = hold.loc[hold.rli_decision == "APPROVE", "default_next_month"].mean() * 100
l.metric("Default rate of approved", f"{approved_rate:.1f}%")
counts = val.set_index(val.Code + " " + val.Reason)["Declines citing"].sort_values()
r.plotly_chart(charts.hbar(counts, "How often each decline reason is cited", "Declines citing reason",
                           charts.RED), width="stretch")

st.markdown("**Do the reasons point at real risk?**")
st.dataframe(val.style.format({c: "{:.1f}" for c in pct_cols}, na_rep="-"),
             hide_index=True, width="stretch")
t = s["r05_rule_test"]
st.caption(f"Rules are kept only if they earn their place. A candidate rule, *decline if repayment ratio < "
           f"{t['threshold']:.0%}* (old R05), would decline {t['would_decline']:,} more holdout accounts whose "
           f"default rate is {t['default_rate_newly_declined']:.1f}%, barely above the "
           f"{t['default_rate_remaining_approved']:.1f}% of those still approved. It was retired; the model already "
           f"captures repayment behaviour (R13).")
