"""Churn prediction and customer segmentation.

CHURN: THE MODEL IS EASY, THE LABEL IS HARD

There is no natural definition of "churn" for a bank account. Customers rarely
close accounts; they go quiet, then quieter. So you must DEFINE churn, and
that definition IS the modelling decision:

  "closed the account"        -> almost no positives, nothing to learn
  "no activity in 30 days"    -> catches everyone who went on holiday
  "no activity in 90 days,
   among previously active"   -> what we use

A FINDING FROM OUR OWN DATA, recorded because it is instructive:
the Sprint 1 generator gives every account steady activity for the whole
window — median gap between transactions is 3 days, and the LONGEST silence
across 10,700 accounts is 42 days. So the 90-day definition yields exactly
zero positives, and `fit_churn_model` raises rather than training.

The tempting fix is to drop the window to 30 days until positives appear.
That is backwards: it defines churn to fit the data rather than to fit the
business question, and produces a model that predicts "was briefly quiet"
rather than "left us". The honest conclusion is that this synthetic source
does not model attrition, and the churn model needs either a generator that
does or real data.

The "among previously active" clause matters more than it looks. Without it
you include customers who never engaged at all, and the model stops predicting
churn and starts detecting dormancy — a different and much easier problem that
nobody asked you to solve.

THE OBSERVATION/PERFORMANCE WINDOW DESIGN is what keeps it honest:

    |------- observation window -------|--- performance window ---|
     features come only from here        label comes only from here

Any feature that touches the performance window leaks the answer. This is the
same discipline as out-of-time validation in credit risk, applied to label
construction rather than to the train/test split.

SEGMENTATION: UNSUPERVISED, SO "CORRECT" IS NOT DEFINED

There is no ground truth for a cluster. k=4 is not more right than k=5. The
honest way to choose is a combination of:
  - silhouette score (statistical cohesion)
  - the elbow in inertia (diminishing returns)
  - WHETHER THE SEGMENTS ARE ACTIONABLE, which is a business judgement

The third criterion usually dominates in practice. A statistically superb
7-cluster solution that marketing cannot build campaigns for is worse than a
mediocre 4-cluster one that they can.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import ClassificationMetrics, evaluate_classifier

# --- churn ------------------------------------------------------------------


@dataclass
class ChurnModel:
    pipeline: Pipeline
    features: list[str]
    metrics: ClassificationMetrics | None = None
    importance: pd.DataFrame | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.features])[:, 1]


class InsufficientPositivesError(ValueError):
    """Raised when the target has too few positives to learn from.

    This deserves its own exception rather than a cryptic sklearn error,
    because it is a DATA finding, not a bug, and the correct response is to
    investigate the label definition or the data — never to loosen the
    definition until positives appear. Defining churn to fit the data you
    happen to have is how you end up with a model that predicts something
    nobody asked about.
    """


def fit_churn_model(
    train: pd.DataFrame, features: list[str], target: str = "churned", min_positives: int = 30
) -> ChurnModel:
    """Gradient boosting for churn.

    Tree models suit churn well: the relationships are genuinely non-linear
    and interaction-heavy (a low balance matters much more for a short-tenure
    customer than a long-tenure one), and unlike credit there is no regulatory
    requirement to explain an individual prediction — nobody has a legal right
    to know why they were targeted by a retention campaign.
    """
    n_positives = int(train[target].astype(int).sum())
    if n_positives < min_positives:
        raise InsufficientPositivesError(
            f"only {n_positives} positive labels (need >= {min_positives}). "
            "Check the label definition and whether the source data actually "
            "contains the behaviour you are trying to model."
        )

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
    pipeline = Pipeline([("clf", model)])
    pipeline.fit(train[features], train[target].astype(int))

    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return ChurnModel(pipeline=pipeline, features=features, importance=importance)


def check_churn_leakage(
    df: pd.DataFrame, features: list[str], target: str = "churned", threshold: float = 0.95
) -> dict:
    """Flag features suspiciously correlated with the churn label.

    A correlation above ~0.95 with the target almost always means the feature
    encodes the answer rather than predicting it. The canonical churn leak is
    "days since last transaction" computed across the WHOLE period including
    the performance window — for a churner that is by definition large, so the
    feature is just the label wearing a hat.

    Our `days_since_last_txn` is computed only up to the performance window
    start, which is what makes it a legitimate predictor rather than a leak.
    """
    y = df[target].astype(int)
    suspicious = {}
    for f in features:
        if f not in df.columns or not pd.api.types.is_numeric_dtype(df[f]):
            continue
        corr = float(np.corrcoef(df[f].fillna(df[f].median()), y)[0, 1])
        if abs(corr) > threshold:
            suspicious[f] = corr
    return {
        "suspicious_features": suspicious,
        "passed": len(suspicious) == 0,
        "threshold": threshold,
    }


def evaluate_churn_model(
    model: ChurnModel, test: pd.DataFrame, target: str = "churned"
) -> ChurnModel:
    model.metrics = evaluate_classifier(
        test[target].astype(int).values, model.predict_proba(test), "churn model"
    )
    return model


# --- segmentation -----------------------------------------------------------


@dataclass
class SegmentationResult:
    k: int
    labels: np.ndarray
    silhouette: float
    inertia: float
    profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    names: dict[int, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"k={self.k}  silhouette={self.silhouette:.4f}  inertia={self.inertia:,.0f}"]
        if not self.profiles.empty:
            lines.append(self.profiles.to_string())
        return "\n".join(lines)


def choose_k(X: pd.DataFrame, k_range: range = range(2, 9), seed: int = 42) -> pd.DataFrame:
    """Evaluate several k values on silhouette and inertia.

    NEITHER METRIC PICKS k FOR YOU.
      silhouette — how cohesive and separated clusters are, in [-1, 1].
        Often monotonically favours small k, so it alone tends to say "2".
      inertia — within-cluster sum of squares. ALWAYS falls as k rises (with
        k = n it reaches zero), so you look for the ELBOW, the point where
        extra clusters stop buying much.

    In practice the deciding criterion is usually the third one: can marketing
    actually do something different for each segment? A statistically optimal
    segmentation nobody can act on has no value.
    """
    scaled = StandardScaler().fit_transform(X)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(scaled)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(scaled, labels)),
                "inertia": float(km.inertia_),
            }
        )
    out = pd.DataFrame(rows)
    out["inertia_drop_pct"] = -out["inertia"].pct_change() * 100
    return out


def segment_customers(
    df: pd.DataFrame, features: list[str], k: int = 4, seed: int = 42
) -> SegmentationResult:
    """Cluster customers and profile the resulting segments.

    The PROFILE is the deliverable, not the labels. A cluster id means nothing
    to a marketer; "high-balance, low-activity, long-tenure" is something they
    can write a campaign against.
    """
    X = df[features].fillna(df[features].median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(scaled)

    profiled = df.copy()
    profiled["segment"] = labels
    profiles = profiled.groupby("segment")[features].mean().round(2)
    profiles["n_customers"] = profiled.groupby("segment").size()
    profiles["pct_of_base"] = (profiles["n_customers"] / len(profiled) * 100).round(1)

    return SegmentationResult(
        k=k,
        labels=labels,
        silhouette=float(silhouette_score(scaled, labels)),
        inertia=float(km.inertia_),
        profiles=profiles,
    )


def name_segments(result: SegmentationResult, value_col: str, activity_col: str) -> dict[int, str]:
    """Attach human-readable names based on value and activity.

    Naming is not cosmetic. An unnamed segment does not get used; "Segment 3"
    never appears in a campaign brief, while "High Value, At Risk" does.
    """
    profiles = result.profiles
    value_median = profiles[value_col].median()
    activity_median = profiles[activity_col].median()

    names = {}
    for seg in profiles.index:
        high_value = profiles.loc[seg, value_col] >= value_median
        high_activity = profiles.loc[seg, activity_col] >= activity_median
        if high_value and high_activity:
            names[seg] = "Champions (high value, engaged)"
        elif high_value and not high_activity:
            names[seg] = "At Risk (high value, disengaging)"
        elif not high_value and high_activity:
            names[seg] = "Growth Potential (engaged, low value)"
        else:
            names[seg] = "Low Engagement"
    result.names = names
    return names
