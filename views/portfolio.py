"""Portfolio: who is in the book and how risk varies by usage and payment behaviour."""
import streamlit as st

from clis.features.builder import read_sql
from ui import charts, loaders as L

acct = L.guard(L.table, "accounts")
util = L.guard(L.table, "seg_util_band")
ctype = L.guard(L.table, "seg_customer_type")

c = st.columns(4)
c[0].metric("Accounts", f"{len(acct):,}")
c[1].metric("Default rate (next month)", f"{acct.default_next_month.mean() * 100:.1f}%")
c[2].metric("Avg utilization", f"{acct.util_now.clip(0, 2).mean() * 100:.1f}%")
c[3].metric("Revolvers (carry a balance)", f"{(acct.customer_type == '3. Revolver').mean() * 100:.1f}%")

l, r = st.columns(2)
l.plotly_chart(charts.bar(util, "segment", "default_rate_pct", "Default rate by utilization band",
                          "Utilization (balance ÷ limit)", "Default rate (%)",
                          hover=["accounts", "pct_of_book"]), width="stretch")
r.plotly_chart(charts.bar(ctype, "segment", "default_rate_pct", "Default rate by customer type (latest month)",
                          "", "Default rate (%)", hover=["accounts", "pct_of_book"]), width="stretch")

# Insight text is computed from the data so it always matches the charts above.
u = util.set_index("segment")["default_rate_pct"]
t = ctype.set_index("segment")["default_rate_pct"]
active = u.drop(index=["0. No balance", "6. Over limit"], errors="ignore")
low_band, high_band = active.idxmin(), active.idxmax()
tr = acct[acct.customer_type == "2. Transactor (paid in full)"]
tr_late = tr.loc[tr.worst_delay > 0, "default_next_month"].mean() * 100
tr_clean = tr.loc[tr.worst_delay <= 0, "default_next_month"].mean() * 100
st.info(
    f"**Key insights.** (1) Among active accounts, risk rises from **{active[low_band]:.1f}%** "
    f"({low_band[3:]} utilization) to **{active[high_band]:.1f}%** ({high_band[3:]}), and over-limit "
    f"accounts reach **{u.get('6. Over limit', float('nan')):.1f}%**. The rise is not smooth, so utilization "
    "alone can't drive line increases.  \n"
    f"(2) **Dormant accounts** (no balance for 5 months) default at **{t['0. Dormant (no balance 5 mo)']:.1f}%**, "
    f"far above revolvers (**{t['3. Revolver']:.1f}%**). Low usage is not the same as low risk.  \n"
    f"(3) Customers who paid in full last month default *more* than revolvers "
    f"(**{t['2. Transactor (paid in full)']:.1f}%** vs **{t['3. Revolver']:.1f}%**). Splitting them shows why: "
    f"those with any late payment in the last 5 months default at **{tr_late:.1f}%**, clean ones at "
    f"**{tr_clean:.1f}%**. Payment *history* matters more than the latest payment type.  \n"
    f"(4) Customers 2+ months late default at **{t['5. 2+ months late'] / t['3. Revolver']:.1f}×** the revolver rate."
)

with st.expander("See the SQL behind these features"):
    st.code(read_sql("features.sql").strip(), language="sql")
    st.code(read_sql("segments.sql").strip(), language="sql")
