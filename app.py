"""
Credit Line Increase Strategy: interactive dashboard (entry point).

Run locally:
    python scripts/build_artifacts.py   # once, or after changing logic
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Credit Line Increase Strategy", page_icon="📈", layout="wide")

st.title("Credit Line Increase Strategy & Risk Analysis")
st.caption("Who should get a higher credit limit? Proactive (PLI) and reactive (RLI) line-increase decisions on "
           "30,000 real credit card accounts, using SQL, a probability-of-default model, profit-based champion vs "
           "challenger testing and fair-lending checks. Data: UCI Default of Credit Card Clients (Taiwan, NT$).")

pages = [
    st.Page("views/portfolio.py", title="Portfolio", icon="📊", default=True),
    st.Page("views/risk_model.py", title="Risk model", icon="🎯"),
    st.Page("views/pli_strategy.py", title="PLI strategy", icon="⚖️"),
    st.Page("views/rli_engine.py", title="RLI decision engine", icon="🧾"),
    st.Page("views/monitoring.py", title="Monitoring & controls", icon="🛡️"),
]
st.navigation(pages, position="top").run()

st.divider()
st.caption("Built by Keshav Dhiman · Python, DuckDB SQL, XGBoost, SHAP, Streamlit · "
           "[GitHub](https://github.com/kdlottey/credit-line-increase-strategy)")
