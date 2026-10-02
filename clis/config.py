"""Central configuration: paths, schema, features and business assumptions.

Every threshold the strategy depends on lives here so it can be reviewed in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "UCI_Credit_Card.csv"
ARTIFACT_DIR = ROOT / "artifacts"

N_MONTHS_RAW = 6          # months of history in the source data
LOOKBACK = 5              # months used per snapshot; leaves 1 month to rebuild a T-1 snapshot
RANDOM_STATE = 42

RAW_COLUMNS: list[str] = (
    ["limit_amt", "sex", "education", "marriage", "age"]
    + [f"pay_status_{i}" for i in range(1, 7)]   # 1 = most recent month
    + [f"bill_{i}" for i in range(1, 7)]         # statement balance
    + [f"paid_{i}" for i in range(1, 7)]         # amount paid
    + ["default_next_month"]
)
TARGET = "default_next_month"

# Behavioral features only. Protected attributes are never model inputs;
# they are kept aside solely to test the strategy for disparate impact.
FEATURES: list[str] = [
    "limit_amt", "util_now", "util_avg", "worst_delay", "months_delayed",
    "pay_ratio", "pay_status_1", "pay_status_2", "bill_1", "paid_1",
]
PROTECTED: list[str] = ["sex", "age"]

FEATURE_LABELS: dict[str, str] = {
    "limit_amt": "Credit limit", "util_now": "Current utilization",
    "util_avg": "Average utilization (5 mo)", "worst_delay": "Worst delinquency (5 mo)",
    "months_delayed": "Months delinquent (5 mo)", "pay_ratio": "Repayment ratio (5 mo)",
    "pay_status_1": "Latest repayment status", "pay_status_2": "Prior-month repayment status",
    "bill_1": "Current balance", "paid_1": "Latest payment",
}


@dataclass(frozen=True)
class Policy:
    """Hard gates and rule thresholds shared by PLI and RLI."""
    max_util: float = 1.0               # no increases for over-limit accounts
    max_worst_delay: int = 1            # 2+ months late in lookback -> ineligible
    # Repayment-ratio rule (R05). Disabled (0.0): tested at 0.05 it declined ~1,000 holdout accounts
    # whose default rate was barely above approved accounts', so the model handles repayment instead.
    min_pay_ratio: float = 0.0
    tested_pay_ratio_rule: float = 0.05
    champion_min_util: float = 0.50
    challenger_min_util: float = 0.30
    min_segment_size: int = 50          # ignore cut-offs that approve fewer accounts
    cutoff_grid: tuple[float, ...] = tuple(round(0.02 + 0.01 * i, 2) for i in range(79))  # 0.02..0.80


@dataclass(frozen=True)
class LineSizing:
    """Increase amount. Champion: flat %. Challenger: tiered by predicted risk."""
    flat_pct: float = 0.20
    cap: int = 50_000
    tiers: tuple[tuple[float, float], ...] = ((0.08, 0.30), (0.15, 0.20), (1.01, 0.10))  # (pd upper, pct)


@dataclass(frozen=True)
class Economics:
    """Simple 12-month profit model for one line increase. All values are assumptions."""
    apr: float = 0.18             # interest on revolving balance
    lgd: float = 0.85             # loss given default
    ccf: float = 0.75             # share of the new line drawn by an account that defaults
    cost_of_funds: float = 0.03
    horizon_years: float = 1.0


@dataclass(frozen=True)
class Settings:
    policy: Policy = field(default_factory=Policy)
    sizing: LineSizing = field(default_factory=LineSizing)
    economics: Economics = field(default_factory=Economics)


SETTINGS = Settings()
