# Credit Line Increase Strategy & Risk Analysis

**Who should get a higher credit limit?** An end-to-end analysis of proactive (PLI) and reactive (RLI) credit line increase decisions on 30,000 real credit card accounts, from SQL analysis to a risk model, a strategy test and a live decision engine.

### 🔗 [Live app](https://YOUR-APP-NAME.streamlit.app) · [Notebook](notebooks/credit_line_increase_strategy.ipynb)

![App screenshot](images/portfolio.png)

## Business problem
Card issuers grow balances by raising credit limits, but giving more credit to the wrong customer increases losses. A good strategy gives more line to customers who will **use** it and **repay** it.

| Question | Where it's answered |
|---|---|
| How do utilization and repayment behavior relate to default risk? | Portfolio tab · SQL analysis |
| Can a model rank customers by default risk? | Risk model tab |
| Does a model-based strategy beat today's rule-based one? | PLI strategy tab |
| How should a customer's request for more credit be decided, with reasons? | RLI decision engine tab |
| How do we keep the data and the strategy under control after launch? | Monitoring & controls tab |

## Key results
<!-- Replace with the numbers from your app / notebook section 9 -->
- **Risk model:** [Logistic Regression / XGBoost] reached test **AUC [0.XX]**, **KS [XX]**, using behavioral features only.
- **PLI strategy:** the challenger approves **[X]% more accounts** than the rule-based champion at a default rate of **[X]%** vs **[X]%**.
- **Insight:** default rate rises from **[X]%** at under 10% utilization to **[X]%** at 80–100%, so utilization alone can't drive line increases.
- **RLI engine:** [X]% approval rate; every decline carries auditable reason codes.
- **Monitoring:** score PSI of **[0.XXX]** (stable).

## Approach
| Step | What | Tools |
|---|---|---|
| 1 | Data-quality and control checks on every load | Pandas |
| 2 | Feature building and segment analysis: utilization bands, customer types, deciles with window functions | SQL (DuckDB) |
| 3 | Probability-of-default model: Logistic Regression vs XGBoost, evaluated on AUC, KS and Gini | scikit-learn, XGBoost |
| 4 | Champion vs challenger PLI strategy, PD cut-off sweep, swap-set analysis | Python |
| 5 | RLI decision engine with reason codes | Python |
| 6 | Score stability monitoring (PSI) | Python |
| 7 | Interactive dashboard | Streamlit, Plotly |

**Fair lending:** sex, age, marital status and education are deliberately excluded from the model.

## Screenshots
| PLI strategy | RLI decision engine |
|---|---|
| ![](images/pli_strategy.png) | ![](images/rli_engine.png) |

## Project structure
```
├── app.py                  # Streamlit dashboard (5 tabs)
├── core.py                 # All logic: controls, SQL features, model, strategies, RLI engine, PSI
├── notebooks/
│   └── credit_line_increase_strategy.ipynb   # Step-by-step analysis with commentary
├── data/                   # UCI_Credit_Card.csv (see data/README.md)
├── images/                 # Screenshots for this README
└── requirements.txt
```

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations and next steps
- Target is default **next month**; production line-increase models predict over 12+ months.
- No income, months-on-book or bureau data, so the increase amount is a simple rule (20% of limit, capped).
- Expected loss assumes 100% loss given default.
- A real rollout would **A/B test** the challenger on a small random group and track 30/60/90-day delinquency by vintage.

## Data
Yeh, I. C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0). Taiwan, 2005; amounts in NT$.

---
Built by **Keshav Dhiman** · [LinkedIn](https://www.linkedin.com/in/keshav-dhiman-85164724a/)
