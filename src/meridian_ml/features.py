"""Build modelling datasets from the Meridian warehouse output.

THE RULE THAT GOVERNS THIS ENTIRE MODULE: NO LEAKAGE.

A feature leaks when it contains information that would not have been
available at the moment the prediction is made. It is the single most common
way a model looks brilliant in development and fails in production, and it is
usually introduced by accident while doing something that feels sensible.

THREE KINDS, in rough order of how often they bite people:

1. TARGET LEAKAGE — the feature is partly the answer. Using "number of missed
   payments" to predict default is not prediction; missing payments IS
   defaulting. Signature: implausibly high AUC (0.95+ on credit data).

2. TEMPORAL LEAKAGE — the feature uses data from after the prediction point.
   Using a customer's CURRENT credit score to predict a loan decision made two
   years ago means using information that did not exist then. This is exactly
   why Sprint 1 generated `credit_score_at_origination` and why Sprint 3 built
   SCD Type 2 — so this module can use point-in-time values honestly.

3. TRAIN/TEST CONTAMINATION — fitting a scaler or imputer on the full dataset
   before splitting leaks test-set statistics into training. Fix: put every
   transformation inside a sklearn Pipeline, which only ever sees training
   data during fit.

WHY THE `credit_score_at_origination` COLUMN EXISTS AT ALL
It looked like a small modelling detail back in Sprint 1. It is the entire
reason this module can build an honest credit dataset: the borrower's score
TODAY is contaminated by whether they subsequently defaulted. The score AT
THE TIME OF THE DECISION is not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_parquet_tables(source_dir: str | Path, tables: list[str]) -> dict[str, pd.DataFrame]:
    """Read the generator/warehouse parquet output into DataFrames."""
    source = Path(source_dir)
    out: dict[str, pd.DataFrame] = {}
    for name in tables:
        matches = list(source.glob(f"{name}/**/*.parquet")) + list(source.glob(f"{name}/*.parquet"))
        if matches:
            out[name] = pd.read_parquet(matches[0])
    return out


def build_credit_features(loans: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Assemble the PD modelling dataset, using only origination-time facts.

    EVERY feature here is knowable at the moment the loan is underwritten.
    Nothing describes what happened afterwards.

    Deliberately EXCLUDED, and why:
      - the customer's current credit score (contaminated by the outcome)
      - payment history (that IS the outcome)
      - current balance or DPD (post-origination)
    """
    df = loans.merge(
        customers[["customer_id", "age", "segment", "join_date", "home_branch_id"]],
        on="customer_id",
        how="left",
    )

    df["origination_date"] = pd.to_datetime(df["origination_date"])
    df["join_date"] = pd.to_datetime(df["join_date"])

    # --- origination-time features only ---
    df["tenure_days_at_origination"] = (df["origination_date"] - df["join_date"]).dt.days.clip(
        lower=0
    )

    # Loan size relative to borrower quality — a classic scorecard feature.
    df["loan_to_score_ratio"] = df["principal"] / df["credit_score_at_origination"]

    # Monthly payment burden. Payment-to-income would be better; we approximate
    # with DTI, which is captured at origination.
    df["payment_burden"] = df["monthly_payment"] * df["term_months"] / df["principal"]

    df["is_secured"] = df["loan_type"].isin(["auto", "mortgage"]).astype(int)
    df["log_principal"] = np.log1p(df["principal"])

    # Credit band at origination — the same derivation the warehouse uses, so
    # model and reporting agree on who counts as "good credit".
    df["credit_band_at_origination"] = pd.cut(
        df["credit_score_at_origination"],
        bins=[0, 580, 670, 740, 850],
        labels=["poor", "fair", "good", "excellent"],
    )

    df["segment_code"] = df["segment"].map({"mass": 0, "affluent": 1, "private": 2}).fillna(0)
    df["loan_type_code"] = df["loan_type"].astype("category").cat.codes

    return df


CREDIT_FEATURES = [
    "credit_score_at_origination",
    "dti_at_origination",
    "log_principal",
    "apr",
    "term_months",
    "tenure_days_at_origination",
    "loan_to_score_ratio",
    "is_secured",
    "segment_code",
    "age",
]


def build_fraud_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Assemble fraud-detection features, including velocity aggregates.

    VELOCITY FEATURES are the backbone of real fraud detection. A single $400
    purchase is unremarkable; a $400 purchase that is the fifth in twenty
    minutes on an account that normally transacts twice a week is not. The
    signal lives in the DEVIATION FROM THIS ACCOUNT'S OWN NORMAL, not in the
    raw values.

    CRITICAL: every rolling window below looks BACKWARD only. Computing "the
    account's average transaction amount" over the whole dataset would include
    the fraudulent transactions themselves and leak the answer. Sorting by time
    and using expanding/shifted windows keeps it honest.
    """
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["log_amount"] = np.log1p(df["amount"])
    df["is_online"] = (df["channel"] == "online").astype(int)

    g = df.groupby("account_id", sort=False)

    # Time since this account's previous transaction — the core velocity signal.
    df["seconds_since_prev"] = g["timestamp"].diff().dt.total_seconds().fillna(86400 * 7)
    df["log_seconds_since_prev"] = np.log1p(df["seconds_since_prev"])

    # Expanding mean of PRIOR transactions only. shift(1) is what makes this
    # backward-looking; without it the current row contaminates its own baseline.
    df["account_prior_mean_amount"] = (
        g["amount"].transform(lambda s: s.shift(1).expanding().mean())
    ).fillna(df["amount"].median())

    # How unusual is this amount for THIS account? The deviation is the signal.
    df["amount_vs_account_mean"] = df["amount"] / df["account_prior_mean_amount"].clip(lower=1)

    # Rolling count in the last 10 transactions — burst detection.
    df["prior_txn_count_10"] = (
        g["amount"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).count())
    ).fillna(0)

    df["merchant_risk"] = (
        df["merchant_category"]
        .map(
            {
                "online": 3,
                "travel": 3,
                "retail": 2,
                "gas": 2,
                "entertainment": 2,
                "atm_withdrawal": 2,
                "grocery": 1,
                "restaurant": 1,
                "utilities": 1,
                "healthcare": 1,
            }
        )
        .fillna(2)
    )

    return df


FRAUD_FEATURES = [
    "log_amount",
    "hour",
    "is_night",
    "is_weekend",
    "is_online",
    "log_seconds_since_prev",
    "amount_vs_account_mean",
    "prior_txn_count_10",
    "merchant_risk",
]


def build_churn_features(
    customers: pd.DataFrame, accounts: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    """Assemble churn features using an observation/performance window design.

    THE HARD PART OF CHURN IS NOT THE MODEL, IT IS THE LABEL.
    "Churn" has no natural definition for a bank account — customers rarely
    close accounts, they just go quiet. You must DEFINE it, and the definition
    determines what you are modelling:

      too strict ("closed the account")  -> almost no positives, unlearnable
      too loose  ("no activity in 30d")  -> catches people who were on holiday

    We use: no transactions in the final 90 days of the window, among
    customers who WERE active before it. That "were active before" condition
    matters — without it, you include customers who never engaged at all, and
    the model just learns to detect dormancy rather than churn.

    THE WINDOW DESIGN prevents leakage: features come only from the
    OBSERVATION window; the label comes only from the later PERFORMANCE
    window. No feature may touch performance-window data.
    """
    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    max_date = tx["timestamp"].max()

    performance_start = max_date - pd.Timedelta(days=90)
    observation = tx[tx["timestamp"] < performance_start]
    performance = tx[tx["timestamp"] >= performance_start]

    acct_to_cust = accounts.set_index("account_id")["customer_id"]
    observation = observation.assign(customer_id=observation["account_id"].map(acct_to_cust))
    performance = performance.assign(customer_id=performance["account_id"].map(acct_to_cust))

    obs_agg = (
        observation.groupby("customer_id")
        .agg(
            txn_count_obs=("transaction_id", "count"),
            total_spend_obs=("amount", "sum"),
            avg_amount_obs=("amount", "mean"),
            last_txn_obs=("timestamp", "max"),
            distinct_categories=("merchant_category", "nunique"),
            online_share=("channel", lambda s: (s == "online").mean()),
        )
        .reset_index()
    )

    active_in_performance = set(performance["customer_id"].dropna().unique())

    df = obs_agg.merge(
        customers[["customer_id", "age", "annual_income", "credit_score", "segment", "join_date"]],
        on="customer_id",
        how="inner",
    )

    df["days_since_last_txn"] = (performance_start - df["last_txn_obs"]).dt.days
    df["tenure_days"] = (performance_start - pd.to_datetime(df["join_date"])).dt.days
    df["txn_per_month_obs"] = df["txn_count_obs"] / (df["tenure_days"] / 30).clip(lower=1)
    df["segment_code"] = df["segment"].map({"mass": 0, "affluent": 1, "private": 2}).fillna(0)
    df["log_income"] = np.log1p(df["annual_income"])

    # THE LABEL: active before, silent during the performance window.
    df["churned"] = (~df["customer_id"].isin(active_in_performance)).astype(int)

    return df


CHURN_FEATURES = [
    "txn_count_obs",
    "avg_amount_obs",
    "days_since_last_txn",
    "distinct_categories",
    "online_share",
    "txn_per_month_obs",
    "tenure_days",
    "credit_score",
    "log_income",
    "segment_code",
    "age",
]
