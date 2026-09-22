"""Model evaluation metrics, including the ones banks actually use.

WHY THIS MODULE EXISTS SEPARATELY
Generic ML tutorials evaluate with accuracy and maybe AUC. Credit risk and
fraud teams use a different vocabulary, and using the wrong metric is not a
stylistic difference — it produces models that look excellent and are useless.

THE THREE METRICS A CREDIT RISK TEAM WILL ASK YOU ABOUT

  AUC (area under the ROC curve) — the probability that a randomly chosen bad
    loan scores worse than a randomly chosen good one. 0.5 is a coin flip, 1.0
    is perfect. A retail credit scorecard typically lands 0.70-0.80; much above
    that on real data usually means leakage rather than genius.

  KS statistic — the maximum vertical distance between the cumulative
    distributions of goods and bads. It answers "at the single best cutoff, how
    much separation do we get?" Banks quote KS constantly because it maps onto
    an actual operating decision. Roughly: <20 weak, 20-40 usable, 40+ strong.

  Gini = 2 x AUC - 1. Just a rescaling of AUC onto 0-1 where 0 is random.
    Europe tends to quote Gini, the US tends to quote KS. Know both.

AND THE ONE THAT MATTERS MOST FOR PRICING: CALIBRATION.
A model can rank perfectly and still be useless. If it says "5% chance of
default" for a bucket of loans where 20% actually default, you have priced
those loans catastrophically wrong even though the RANKING was flawless.
Ranking metrics (AUC, KS, Gini) are all blind to this. Expected loss
calculations, capital requirements, and loan pricing all depend on the
PROBABILITY being right, not just the order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    """Discrimination and calibration for a binary classifier."""

    model: str
    n: int
    positive_rate: float
    auc: float
    gini: float
    ks: float
    ks_threshold: float
    average_precision: float
    brier: float
    calibration_bins: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{self.model}  (n={self.n:,}, positive rate {self.positive_rate:.3%})",
            f"  DISCRIMINATION  AUC={self.auc:.4f}  Gini={self.gini:.4f}  "
            f"KS={self.ks:.2f} @ threshold {self.ks_threshold:.4f}",
            f"  CALIBRATION     Brier={self.brier:.5f} (lower is better)",
        ]
        if self.average_precision is not None:
            lines.append(f"  RANKING (rare)  Average precision={self.average_precision:.4f}")
        return "\n".join(lines)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov separation, on the 0-100 scale banks quote.

    Computed from the ROC curve: KS is the largest gap between the true
    positive rate and the false positive rate, which is exactly the largest
    separation between the cumulative distributions of bads and goods.
    Returns (ks, the score threshold where it occurs).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    gaps = tpr - fpr
    idx = int(np.argmax(gaps))
    return float(gaps[idx] * 100), float(thresholds[idx])


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Compare predicted probability against observed rate, by decile.

    THE ONLY WAY TO SEE MISCALIBRATION. Sort by predicted probability, bucket,
    and ask: in the bucket where we predicted 5%, did roughly 5% actually
    default? A well-calibrated model has predicted ~= observed down the whole
    table. Systematic drift (predicted always below observed) means the model
    understates risk, which in a bank means underpricing loans.
    """
    order = np.argsort(y_prob)
    y_true, y_prob = np.asarray(y_true)[order], np.asarray(y_prob)[order]
    bins = np.array_split(np.arange(len(y_prob)), n_bins)

    rows = []
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": i,
                "n": len(idx),
                "predicted_mean": float(y_prob[idx].mean()),
                "observed_rate": float(y_true[idx].mean()),
                "difference": float(y_true[idx].mean() - y_prob[idx].mean()),
            }
        )
    return rows


def evaluate_classifier(
    y_true: np.ndarray, y_prob: np.ndarray, model: str = "model", n_bins: int = 10
) -> ClassificationMetrics:
    """Full evaluation: discrimination AND calibration, never just one."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    auc = float(roc_auc_score(y_true, y_prob))
    ks, ks_thr = ks_statistic(y_true, y_prob)

    return ClassificationMetrics(
        model=model,
        n=len(y_true),
        positive_rate=float(y_true.mean()),
        auc=auc,
        gini=2 * auc - 1,
        ks=ks,
        ks_threshold=ks_thr,
        average_precision=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        calibration_bins=calibration_table(y_true, y_prob, n_bins),
    )


# --- imbalanced classification ---------------------------------------------


@dataclass
class ThresholdAnalysis:
    """Performance across operating thresholds, with the business cost attached."""

    threshold: float
    predicted_positives: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    net_benefit: float

    def summary(self) -> str:
        return (
            f"  {self.threshold:>6.3f}  alerts={self.predicted_positives:>7,}  "
            f"caught={self.true_positives:>5,}  missed={self.false_negatives:>5,}  "
            f"precision={self.precision:>6.2%}  recall={self.recall:>6.2%}  "
            f"net=${self.net_benefit:>12,.0f}"
        )


def cost_based_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
    recovery_rate: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> list[ThresholdAnalysis]:
    """Choose an operating threshold by DOLLARS, not by 0.5.

    WHY 0.5 IS ALMOST ALWAYS WRONG
    0.5 is the default because it is the midpoint of a probability, not because
    it is optimal for anything. With 0.15% fraud, a threshold of 0.5 will fire
    on almost nothing. The right threshold depends on the relative cost of the
    two mistakes, and those costs are wildly asymmetric:

      FALSE NEGATIVE (missed fraud) — you lose the transaction amount.
      FALSE POSITIVE (false alarm)  — you pay an analyst to review it, and you
        annoy a legitimate customer.

    A missed $3,000 fraud costs far more than a $4 review. That asymmetry means
    the optimal threshold sits far below 0.5, and finding it is an arithmetic
    question, not a modelling one.

    net_benefit = (fraud dollars caught x recovery_rate) - (alerts x review_cost)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    if thresholds is None:
        thresholds = np.percentile(y_prob, np.linspace(90, 99.9, 12))

    results = []
    for t in thresholds:
        flagged = y_prob >= t
        tp = int(np.sum(flagged & (y_true == 1)))
        fp = int(np.sum(flagged & (y_true == 0)))
        fn = int(np.sum(~flagged & (y_true == 1)))

        caught_value = float(amounts[flagged & (y_true == 1)].sum()) * recovery_rate
        review_spend = int(flagged.sum()) * review_cost

        results.append(
            ThresholdAnalysis(
                threshold=float(t),
                predicted_positives=int(flagged.sum()),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=tp / (tp + fp) if (tp + fp) else 0.0,
                recall=tp / (tp + fn) if (tp + fn) else 0.0,
                net_benefit=caught_value - review_spend,
            )
        )
    return results


def why_accuracy_lies(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Demonstrate that accuracy is meaningless under extreme imbalance.

    Compares a real model against the trivial "predict nothing is positive"
    baseline. On a 0.15% base rate the trivial model is 99.85% accurate — and
    catches zero fraud. Any evaluation where that model looks good is an
    evaluation measuring the wrong thing.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    always_negative_accuracy = float((y_true == 0).mean())

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)

    return {
        "base_rate": float(y_true.mean()),
        "always_negative_accuracy": always_negative_accuracy,
        "model_accuracy": float((y_pred == y_true).mean()),
        "model_precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "model_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "pr_curve_points": len(precision_curve),
    }
