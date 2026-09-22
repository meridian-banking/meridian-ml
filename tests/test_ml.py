"""Tests for the ML modules.

TESTING ML CODE IS MOSTLY TESTING PROPERTIES, NOT NUMBERS.
Model metrics move with data and seeds, so asserting "AUC == 0.6511" is a test
that breaks for no good reason. What is worth asserting:
  - invariants that must hold (Gini == 2*AUC - 1, always)
  - structural guarantees (no training row postdates a test row)
  - guardrails firing when they should (prohibited feature detected)
  - the failure modes we actually hit while building this, so they stay fixed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from meridian_ml.churn_segmentation import (
    InsufficientPositivesError,
    choose_k,
    fit_churn_model,
    segment_customers,
)
from meridian_ml.credit_risk import (
    calculate_woe_iv,
    fit_logistic_pd,
    out_of_time_split,
    rank_features_by_iv,
)
from meridian_ml.explainability import (
    adverse_action_reasons,
    check_prohibited_features,
    compute_shap_values,
)
from meridian_ml.metrics import (
    calibration_table,
    cost_based_threshold,
    evaluate_classifier,
    ks_statistic,
    why_accuracy_lies,
)


@pytest.fixture
def credit_data() -> pd.DataFrame:
    """Synthetic loans where default genuinely depends on score and DTI."""
    rng = np.random.default_rng(42)
    n = 3000
    score = np.clip(rng.normal(690, 60, n), 300, 850)
    dti = np.clip(rng.normal(0.35, 0.12, n), 0.02, 0.9)
    # Real signal: risk rises as score falls and DTI rises.
    logit = -3.0 - 0.012 * (score - 690) + 4.0 * (dti - 0.35)
    default = rng.random(n) < 1 / (1 + np.exp(-logit))
    dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D")
    return pd.DataFrame(
        {
            "credit_score_at_origination": score,
            "dti_at_origination": dti,
            "log_principal": rng.normal(10, 1, n),
            "origination_date": dates,
            "defaulted": default.astype(int),
        }
    )


FEATURES = ["credit_score_at_origination", "dti_at_origination", "log_principal"]


# --- metric invariants ------------------------------------------------------


def test_gini_is_exactly_two_auc_minus_one(credit_data):
    """An algebraic identity, so it must hold for every input."""
    rng = np.random.default_rng(0)
    m = evaluate_classifier(credit_data["defaulted"].values, rng.random(len(credit_data)))
    assert abs(m.gini - (2 * m.auc - 1)) < 1e-10


def test_perfect_model_scores_perfectly():
    y = np.array([0, 0, 0, 1, 1, 1])
    m = evaluate_classifier(y, y.astype(float))
    assert m.auc == 1.0
    assert m.ks == pytest.approx(100.0)


def test_random_model_scores_near_chance():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    m = evaluate_classifier(y, rng.random(5000))
    assert 0.45 < m.auc < 0.55
    assert m.ks < 10


def test_ks_is_on_the_zero_to_hundred_scale():
    """Banks quote KS as 0-100, not 0-1. Getting the scale wrong makes a
    strong model look useless in a committee paper."""
    y = np.array([0] * 500 + [1] * 500)
    scores = np.concatenate([np.zeros(500), np.ones(500)])
    ks, _ = ks_statistic(y, scores)
    assert 99 <= ks <= 100


def test_calibration_table_covers_every_row(credit_data):
    rng = np.random.default_rng(2)
    probs = rng.random(len(credit_data))
    table = calibration_table(credit_data["defaulted"].values, probs, n_bins=10)
    assert sum(b["n"] for b in table) == len(credit_data)


def test_calibration_detects_a_miscalibrated_model():
    """A model that ranks perfectly can still be badly calibrated — this is
    exactly what ranking metrics cannot see."""
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.20).astype(int)
    # Ranks correctly but predicts ~4x too low.
    probs = np.where(y == 1, 0.06, 0.02) + rng.normal(0, 0.002, 2000)
    table = calibration_table(y, probs.clip(0, 1), n_bins=5)
    worst = max(table, key=lambda b: abs(b["difference"]))
    assert abs(worst["difference"]) > 0.1


# --- imbalance --------------------------------------------------------------


def test_accuracy_is_misleading_under_imbalance():
    """The headline lesson: a useless model wins on accuracy."""
    rng = np.random.default_rng(4)
    n = 100_000
    y = (rng.random(n) < 0.0015).astype(int)
    probs = np.where(y == 1, rng.beta(5, 2, n), rng.beta(1, 20, n))
    result = why_accuracy_lies(y, probs)
    assert result["always_negative_accuracy"] > 0.99
    assert result["average_precision"] > result["base_rate"] * 10


def test_cost_based_threshold_is_not_point_five():
    """With asymmetric costs the optimum sits far from the 0.5 default."""
    rng = np.random.default_rng(5)
    n = 50_000
    y = (rng.random(n) < 0.002).astype(int)
    probs = np.where(y == 1, rng.beta(6, 2, n), rng.beta(1, 25, n))
    amounts = rng.lognormal(5, 1, n)
    results = cost_based_threshold(y, probs, amounts, review_cost=4.0)
    best = max(results, key=lambda r: r.net_benefit)
    assert best.net_benefit > 0
    assert best.recall > 0.5


def test_higher_threshold_means_fewer_alerts():
    rng = np.random.default_rng(6)
    n = 20_000
    y = (rng.random(n) < 0.01).astype(int)
    probs = rng.random(n)
    results = cost_based_threshold(
        y,
        probs,
        rng.lognormal(5, 1, n),
        review_cost=2.0,
        thresholds=np.array([0.5, 0.7, 0.9]),
    )
    counts = [r.predicted_positives for r in results]
    assert counts == sorted(counts, reverse=True)


# --- WoE / IV ---------------------------------------------------------------


def test_woe_iv_finds_the_real_predictor(credit_data):
    """Score and DTI drive default by construction; principal does not."""
    ranked = rank_features_by_iv(credit_data, "defaulted", FEATURES)
    top = set(ranked.head(2)["feature"])
    assert "credit_score_at_origination" in top or "dti_at_origination" in top
    assert ranked.iloc[-1]["feature"] == "log_principal"


def test_woe_bins_partition_the_data(credit_data):
    res = calculate_woe_iv(
        credit_data["credit_score_at_origination"], credit_data["defaulted"], n_bins=5
    )
    assert sum(b.n for b in res.bins) == len(credit_data)


def test_woe_sign_follows_risk(credit_data):
    """Higher bad rate must give lower (more negative) WoE. If this inverts,
    the transformation is backwards and every downstream coefficient flips."""
    res = calculate_woe_iv(
        credit_data["credit_score_at_origination"], credit_data["defaulted"], n_bins=5
    )
    worst = max(res.bins, key=lambda b: b.bad_rate)
    best = min(res.bins, key=lambda b: b.bad_rate)
    assert worst.woe < best.woe


# --- out-of-time validation -------------------------------------------------


def test_out_of_time_split_has_no_temporal_overlap(credit_data):
    """THE structural guarantee. Every training row must precede every test
    row, or the validation is measuring a task the model will never face."""
    split = out_of_time_split(credit_data, "origination_date", "2023-06-01")
    assert split.train["origination_date"].max() < split.test["origination_date"].min()


def test_out_of_time_split_keeps_every_row(credit_data):
    split = out_of_time_split(credit_data, "origination_date", "2023-06-01")
    assert len(split.train) + len(split.test) == len(credit_data)


def test_model_learns_real_signal(credit_data):
    """Sanity: the model should beat chance on data with genuine signal."""
    split = out_of_time_split(credit_data, "origination_date", "2023-06-01")
    model = fit_logistic_pd(split.train, FEATURES)
    probs = model.predict_proba(split.test)
    m = evaluate_classifier(split.test["defaulted"].values, probs)
    assert m.auc > 0.60


def test_coefficient_signs_are_economically_sensible(credit_data):
    """A model whose coefficients contradict domain knowledge is wrong even
    if its AUC is fine — risk should FALL as credit score RISES."""
    split = out_of_time_split(credit_data, "origination_date", "2023-06-01")
    model = fit_logistic_pd(split.train, FEATURES)
    coefs = model.coefficients.set_index("feature")["coefficient"]
    assert coefs["credit_score_at_origination"] < 0
    assert coefs["dti_at_origination"] > 0


# --- fair lending -----------------------------------------------------------


def test_prohibited_features_are_blocked():
    review = check_prohibited_features(["credit_score", "applicant_race", "income"])
    assert not review.passes
    assert "applicant_race" in review.prohibited


def test_proxy_features_are_flagged_but_not_blocked():
    """Proxies need justification, not automatic rejection — some have a
    legitimate business rationale."""
    review = check_prohibited_features(["credit_score", "zip_code"])
    assert review.passes
    assert "zip_code" in review.proxy_risk


def test_clean_feature_list_passes():
    review = check_prohibited_features(["credit_score", "dti", "loan_amount"])
    assert review.passes
    assert not review.prohibited


# --- SHAP -------------------------------------------------------------------


def test_shap_background_must_differ_from_explained_rows(credit_data):
    """THE BUG WE ACTUALLY HIT.

    SHAP measures deviation from a background distribution. Passing a single
    row as both the explained row and the background yields all-zero values —
    silently, with no error — and an adverse action notice with no reasons on
    it. In a bank that ships as a compliance failure.
    """
    split = out_of_time_split(credit_data, "origination_date", "2023-06-01")
    model = fit_logistic_pd(split.train, FEATURES)
    applicant = split.test.head(1)

    bad, _, _ = compute_shap_values(model.pipeline, applicant[FEATURES])
    assert np.allclose(bad, 0), "single-row background should produce zeros"

    good, _, _ = compute_shap_values(
        model.pipeline, applicant[FEATURES], background=split.train[FEATURES]
    )
    assert not np.allclose(good, 0), "a real background must produce real values"


def test_adverse_action_returns_ranked_reasons():
    shap_row = np.array([0.8, -0.3, 0.2])
    values = pd.Series([600, 0.5, 11.0], index=FEATURES)
    medians = pd.Series([690, 0.35, 10.0], index=FEATURES)
    reasons = adverse_action_reasons(shap_row, values, medians, FEATURES, top_n=4)
    assert len(reasons) == 2, "only risk-INCREASING features are reasons"
    assert reasons[0].contribution > reasons[1].contribution
    assert reasons[0].rank == 1


def test_adverse_action_reasons_are_human_readable():
    """Regulation B expects reasons an applicant can act on, not feature names."""
    shap_row = np.array([0.9, 0.1, 0.05])
    values = pd.Series([580, 0.6, 12.0], index=FEATURES)
    medians = pd.Series([690, 0.35, 10.0], index=FEATURES)
    reasons = adverse_action_reasons(shap_row, values, medians, FEATURES, top_n=1)
    assert "credit_score_at_origination" not in reasons[0].explanation
    assert len(reasons[0].explanation.split()) > 3


# --- churn and segmentation -------------------------------------------------


def test_churn_refuses_to_train_without_positives():
    """A DATA finding, not a bug — and it must fail loudly, never quietly.

    Our own generator produces zero churners (the longest account silence is
    42 days), so this path is exercised for real, not hypothetically.
    """
    df = pd.DataFrame(
        {"a": np.random.default_rng(7).random(200), "churned": np.zeros(200, dtype=int)}
    )
    with pytest.raises(InsufficientPositivesError):
        fit_churn_model(df, ["a"])


def test_segmentation_assigns_every_customer():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    result = segment_customers(df, ["x", "y"], k=3)
    assert len(result.labels) == len(df)
    assert set(np.unique(result.labels)) == {0, 1, 2}


def test_inertia_always_falls_as_k_rises():
    """Which is precisely why inertia alone cannot choose k — at k=n it is 0."""
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"x": rng.normal(0, 1, 400), "y": rng.normal(0, 1, 400)})
    table = choose_k(df, range(2, 6))
    inertias = table["inertia"].tolist()
    assert inertias == sorted(inertias, reverse=True)
