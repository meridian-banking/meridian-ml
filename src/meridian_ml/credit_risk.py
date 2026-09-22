"""Credit risk: probability of default (PD) modelling.

THE BUSINESS CONTEXT THAT SHAPES EVERY DECISION HERE

Expected Loss = PD x LGD x EAD
  PD  probability of default   <- this module
  LGD loss given default       (what fraction you fail to recover)
  EAD exposure at default      (how much is outstanding when it happens)

That equation drives loan pricing, capital held under Basel, and provisioning.
It needs an actual PROBABILITY, not a ranking — which is why calibration
matters here in a way it does not for, say, a recommendation engine.

WHY THIS MODULE DOES THINGS THAT LOOK OLD-FASHIONED

1. LOGISTIC REGRESSION IS A FIRST-CLASS CITIZEN, not a baseline to beat.
   Banks still deploy logistic scorecards in production in 2026. Not from
   inertia — because a regulator can read the coefficients, because adverse
   action reasons fall out directly, and because a model you can explain to a
   compliance officer is deployable while a better one you cannot explain is
   not. We fit XGBoost too, and compare honestly.

2. WEIGHT OF EVIDENCE (WoE) AND INFORMATION VALUE (IV).
   The traditional credit scorecard toolkit. WoE transforms a predictor into
   log-odds units, which linearises its relationship with the target and
   handles outliers and missing values gracefully. IV summarises how
   predictive a variable is before you fit anything.

3. OUT-OF-TIME VALIDATION IS MANDATORY.
   Sprint 7 showed random splits leak on time-ordered data. Credit has that
   problem plus a worse one: POPULATION DRIFT. The people who borrowed in 2022
   are not the people who borrow in 2024 — the economy moved, your marketing
   moved, your own underwriting moved. A model validated on a random shuffle
   of its own era flatters itself twice: once from leakage, once from being
   tested on a population it will never see again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import ClassificationMetrics, evaluate_classifier

# --- Weight of Evidence / Information Value --------------------------------


@dataclass
class WoEBin:
    """One bin of a WoE transformation."""

    bin_label: str
    n: int
    n_good: int
    n_bad: int
    bad_rate: float
    woe: float
    iv_contribution: float


@dataclass
class WoEResult:
    feature: str
    bins: list[WoEBin]
    information_value: float

    @property
    def predictive_strength(self) -> str:
        """The conventional IV interpretation bands used in credit scoring.

        These are rules of thumb from the scorecard tradition, not laws:
          < 0.02  useless
          0.02-0.1 weak
          0.1-0.3  medium
          0.3-0.5  strong
          > 0.5    suspiciously strong — check for leakage before celebrating
        """
        iv = self.information_value
        if iv < 0.02:
            return "useless"
        if iv < 0.1:
            return "weak"
        if iv < 0.3:
            return "medium"
        if iv < 0.5:
            return "strong"
        return "suspicious (check for leakage)"

    def summary(self) -> str:
        lines = [
            f"{self.feature}: IV={self.information_value:.4f} ({self.predictive_strength})",
            f"  {'bin':<22}{'n':>7}{'bad rate':>11}{'WoE':>9}",
        ]
        for b in self.bins:
            lines.append(f"  {b.bin_label:<22}{b.n:>7,}{b.bad_rate:>11.2%}{b.woe:>9.3f}")
        return "\n".join(lines)


def calculate_woe_iv(
    feature: pd.Series, target: pd.Series, n_bins: int = 5, epsilon: float = 0.5
) -> WoEResult:
    """Weight of Evidence and Information Value for one predictor.

    WoE_bin = ln( (% of goods in bin) / (% of bads in bin) )
    IV      = sum over bins of (%goods - %bads) x WoE

    WHY WoE IS USEFUL BEYOND BEING TRADITIONAL
      - It is in log-odds units, the same scale logistic regression works in,
        so the relationship becomes linear by construction.
      - Outliers land in an edge bin and stop mattering.
      - Missing values get their own bin rather than being imputed away — and
        "missing" is often genuinely predictive in credit data.
      - The bins are inspectable: a risk manager can read the table and say
        "yes, worse credit scores should have higher bad rates" or "no, that
        pattern is backwards and something is wrong."

    epsilon guards against a bin with zero bads or zero goods, where the log
    would be infinite. Adding 0.5 is the conventional correction.
    """
    df = pd.DataFrame({"x": feature, "y": target.astype(int)}).dropna()

    if pd.api.types.is_numeric_dtype(df["x"]) and df["x"].nunique() > n_bins:
        df["bin"] = pd.qcut(df["x"], q=n_bins, duplicates="drop")
    else:
        df["bin"] = df["x"].astype(str)

    total_good = int((df["y"] == 0).sum())
    total_bad = int((df["y"] == 1).sum())

    bins: list[WoEBin] = []
    iv_total = 0.0

    for label, grp in df.groupby("bin", observed=True):
        n_good = int((grp["y"] == 0).sum())
        n_bad = int((grp["y"] == 1).sum())

        pct_good = (n_good + epsilon) / (total_good + epsilon)
        pct_bad = (n_bad + epsilon) / (total_bad + epsilon)

        woe = float(np.log(pct_good / pct_bad))
        iv_contrib = float((pct_good - pct_bad) * woe)
        iv_total += iv_contrib

        bins.append(
            WoEBin(
                bin_label=str(label),
                n=len(grp),
                n_good=n_good,
                n_bad=n_bad,
                bad_rate=n_bad / len(grp) if len(grp) else 0.0,
                woe=woe,
                iv_contribution=iv_contrib,
            )
        )

    return WoEResult(feature=str(feature.name), bins=bins, information_value=float(iv_total))


def rank_features_by_iv(
    df: pd.DataFrame, target_col: str, features: list[str], n_bins: int = 5
) -> pd.DataFrame:
    """IV for every candidate feature, strongest first.

    Standard first step in building a scorecard: it tells you what is worth
    modelling before you fit anything, and it flags suspiciously strong
    predictors that usually turn out to be leakage.
    """
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        res = calculate_woe_iv(df[f], df[target_col], n_bins=n_bins)
        rows.append(
            {
                "feature": f,
                "information_value": res.information_value,
                "strength": res.predictive_strength,
            }
        )
    return pd.DataFrame(rows).sort_values("information_value", ascending=False)


# --- out-of-time split ------------------------------------------------------


@dataclass
class TemporalSplit:
    """A train/test split made by TIME, never at random."""

    train: pd.DataFrame
    test: pd.DataFrame
    split_date: pd.Timestamp
    date_column: str

    def summary(self) -> str:
        return (
            f"Out-of-time split at {self.split_date.date()}\n"
            f"  train  {len(self.train):>7,} rows  "
            f"{self.train[self.date_column].min().date()} to "
            f"{self.train[self.date_column].max().date()}\n"
            f"  test   {len(self.test):>7,} rows  "
            f"{self.test[self.date_column].min().date()} to "
            f"{self.test[self.date_column].max().date()}"
        )


def out_of_time_split(
    df: pd.DataFrame, date_column: str, split_date: str | pd.Timestamp
) -> TemporalSplit:
    """Split so that ALL training data precedes ALL test data.

    This is the non-negotiable validation design for credit risk. The test set
    is a genuinely later period, which is the only honest simulation of how the
    model will be used: trained on history, applied to applicants who have not
    happened yet.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    split = pd.Timestamp(split_date)
    return TemporalSplit(
        train=df[df[date_column] < split].copy(),
        test=df[df[date_column] >= split].copy(),
        split_date=split,
        date_column=date_column,
    )


def random_split_for_comparison(
    df: pd.DataFrame, test_size: float = 0.3, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A random split, provided ONLY to demonstrate why it is wrong.

    Used by the comparison below. Never use this to report performance.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_size
    return df[~mask].copy(), df[mask].copy()


# --- the PD model -----------------------------------------------------------


@dataclass
class PDModel:
    """A fitted probability-of-default model."""

    name: str
    pipeline: Pipeline
    features: list[str]
    train_metrics: ClassificationMetrics | None = None
    test_metrics: ClassificationMetrics | None = None
    coefficients: pd.DataFrame | None = field(default=None)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.features])[:, 1]


def fit_logistic_pd(
    train: pd.DataFrame,
    features: list[str],
    target: str = "defaulted",
    class_weight: str | None = "balanced",
) -> PDModel:
    """Fit a logistic regression PD model — the scorecard workhorse.

    class_weight='balanced' up-weights the minority class so the model does not
    simply learn "almost nobody defaults". NOTE THE TRADE-OFF, because it is a
    real one and interviewers probe it: balancing improves RANKING but
    DESTROYS CALIBRATION — the predicted probabilities come out systematically
    too high. For a pure ranking use (who to decline) that is fine. For pricing
    or expected-loss you must recalibrate afterwards, or fit unbalanced.
    """
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=42),
            ),
        ]
    )
    X, y = train[features], train[target].astype(int)
    pipeline.fit(X, y)

    clf = pipeline.named_steps["clf"]
    coefs = pd.DataFrame(
        {
            "feature": features,
            "coefficient": clf.coef_[0],
            # Odds ratio: how the odds of default multiply per 1 SD increase.
            # This is the number you explain to a risk committee.
            "odds_ratio": np.exp(clf.coef_[0]),
        }
    ).sort_values("coefficient", key=np.abs, ascending=False)

    return PDModel(
        name="logistic_regression", pipeline=pipeline, features=features, coefficients=coefs
    )


def fit_gradient_boosted_pd(
    train: pd.DataFrame,
    features: list[str],
    target: str = "defaulted",
    scale_pos_weight: float | None = None,
) -> PDModel:
    """Fit an XGBoost PD model, for honest comparison against the scorecard.

    Usually wins on AUC. Usually loses on deployability, because "the model
    said no" is not an acceptable adverse action reason under ECOA. The point
    of fitting both is to quantify what the interpretability is costing you —
    if gradient boosting buys 0.01 AUC, the scorecard is obviously right; if it
    buys 0.08, that is a real conversation with compliance.
    """
    from xgboost import XGBClassifier

    y = train[target].astype(int)
    if scale_pos_weight is None:
        pos = int(y.sum())
        scale_pos_weight = (len(y) - pos) / pos if pos else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,  # shallow: credit signal is mostly additive, deep trees overfit
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    pipeline = Pipeline([("clf", model)])
    pipeline.fit(train[features], y)

    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return PDModel(name="xgboost", pipeline=pipeline, features=features, coefficients=importance)


def evaluate_pd_model(
    model: PDModel, train: pd.DataFrame, test: pd.DataFrame, target: str = "defaulted"
) -> PDModel:
    """Score the model on both periods and attach the metrics.

    Reporting BOTH matters: a large train-test gap is the signature of
    overfitting, and on an out-of-time split some degradation is expected and
    normal — that gap IS the population drift, made visible.
    """
    model.train_metrics = evaluate_classifier(
        train[target].astype(int).values,
        model.predict_proba(train),
        f"{model.name} (train)",
    )
    model.test_metrics = evaluate_classifier(
        test[target].astype(int).values,
        model.predict_proba(test),
        f"{model.name} (out-of-time test)",
    )
    return model
