"""Model explainability, and the legal requirement behind it.

WHY THIS MODULE IS NOT OPTIONAL IN BANKING

Under the Equal Credit Opportunity Act (ECOA) and its implementing
Regulation B, a lender who declines an application — or approves it on worse
terms — must provide the applicant with the SPECIFIC PRINCIPAL REASONS. Not
"you did not meet our criteria." Not "the model declined you." Specific,
actionable reasons, typically the top three or four.

This single requirement explains a great deal about how banks build models:

  - It is why logistic scorecards survive. The coefficients ARE the reasons.
  - It is why "black box" is a deployment blocker, not an aesthetic complaint.
  - It is why SHAP matters here beyond curiosity: it makes a complex model
    produce per-applicant reasons that can actually be printed on a letter.

FAIR LENDING GOES FURTHER THAN EXPLAINABILITY
Regulation B also prohibits using protected characteristics — race, sex,
religion, national origin, marital status, age (with narrow exceptions) — in
credit decisions. And it prohibits DISPARATE IMPACT: a facially neutral
variable that acts as a proxy for a protected class can be illegal even
without intent. Zip code is the classic example, because in the US it
correlates strongly with race.

That is why `check_prohibited_features` exists below, and why it treats
proxies as seriously as direct use.

WHAT SHAP ACTUALLY IS, briefly
A SHAP value answers: "how much did this feature move THIS prediction away
from the average prediction?" It comes from cooperative game theory (Shapley
values) and has a property that matters here — ADDITIVITY. The base value plus
all the SHAP values equals exactly the prediction. That is what lets you say
"your score was reduced primarily by X, then Y, then Z" and have it be
arithmetically true rather than a plausible-sounding story.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Feature names that must never appear in a credit model, and the reason.
PROHIBITED_FEATURES = {
    "race": "ECOA prohibited basis",
    "ethnicity": "ECOA prohibited basis",
    "sex": "ECOA prohibited basis",
    "gender": "ECOA prohibited basis",
    "religion": "ECOA prohibited basis",
    "national_origin": "ECOA prohibited basis",
    "marital_status": "ECOA prohibited basis",
    "pregnancy": "ECOA prohibited basis",
}

# Features that are not themselves prohibited but commonly act as PROXIES for
# a protected class. Disparate impact does not require intent, so these need
# an affirmative justification, not just an absence of bad motive.
PROXY_RISK_FEATURES = {
    "zip_code": "strong proxy for race in the US; classic redlining vector",
    "postcode": "strong proxy for race; classic redlining vector",
    "zipcode": "strong proxy for race; classic redlining vector",
    "branch_id": "geographic proxy — may encode neighbourhood demographics",
    "home_branch_id": "geographic proxy — may encode neighbourhood demographics",
    "city": "geographic proxy",
    "surname": "proxy for ethnicity and national origin",
    "last_name": "proxy for ethnicity and national origin",
    "first_name": "proxy for sex and ethnicity",
    "age": "age is a protected basis under ECOA with narrow exceptions",
}


@dataclass
class FairLendingReview:
    """Outcome of screening a feature list for fair-lending risk."""

    features: list[str]
    prohibited: dict[str, str] = field(default_factory=dict)
    proxy_risk: dict[str, str] = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        """Prohibited features are a hard fail. Proxies require review, not
        automatic rejection — some have a legitimate business justification."""
        return len(self.prohibited) == 0

    def summary(self) -> str:
        lines = [f"Fair lending review of {len(self.features)} features"]
        if self.prohibited:
            lines.append("  PROHIBITED (must remove):")
            lines.extend(f"    {f}: {why}" for f, why in self.prohibited.items())
        if self.proxy_risk:
            lines.append("  PROXY RISK (requires documented justification):")
            lines.extend(f"    {f}: {why}" for f, why in self.proxy_risk.items())
        if not self.prohibited and not self.proxy_risk:
            lines.append("  No prohibited features or known proxies detected.")
        elif not self.prohibited:
            lines.append("  No prohibited features. Proxies flagged for review.")
        return "\n".join(lines)


def check_prohibited_features(features: list[str]) -> FairLendingReview:
    """Screen a feature list against ECOA prohibited bases and known proxies.

    THIS IS A SCREEN, NOT A COMPLIANCE SIGN-OFF. It catches obvious names. It
    cannot detect that `avg_transaction_at_merchant_X` happens to correlate
    with a protected class in your particular population. Real fair-lending
    testing measures OUTCOMES across groups (adverse impact ratio, etc.), not
    just variable names — that requires the protected attributes themselves,
    held by compliance, precisely so modellers cannot use them.
    """
    prohibited, proxies = {}, {}
    for f in features:
        key = f.lower()
        for bad, reason in PROHIBITED_FEATURES.items():
            if bad in key:
                prohibited[f] = reason
        for risky, reason in PROXY_RISK_FEATURES.items():
            if risky in key and f not in prohibited:
                proxies[f] = reason
    return FairLendingReview(features=features, prohibited=prohibited, proxy_risk=proxies)


@dataclass
class AdverseActionReason:
    """One principal reason for a credit decline."""

    rank: int
    feature: str
    contribution: float
    customer_value: float
    population_median: float
    explanation: str

    def summary(self) -> str:
        return (
            f"  {self.rank}. {self.explanation}\n"
            f"     ({self.feature}={self.customer_value:,.2f}, "
            f"typical={self.population_median:,.2f}, "
            f"impact={self.contribution:+.4f})"
        )


# Plain-language templates. Regulation B expects reasons an APPLICANT can act
# on, not feature names. "credit_score_at_origination was 0.42 SDs below the
# mean" is not a reason; "your credit score is lower than we require" is.
REASON_TEMPLATES = {
    "credit_score_at_origination": "Credit score is below our threshold for this product",
    "credit_score": "Credit score is below our threshold for this product",
    "dti_at_origination": "Debt-to-income ratio is higher than we can accept",
    "dti": "Debt-to-income ratio is higher than we can accept",
    "log_principal": "Requested loan amount is high relative to your profile",
    "principal": "Requested loan amount is high relative to your profile",
    "apr": "The pricing for your risk profile exceeds product limits",
    "term_months": "Requested loan term is outside our accepted range",
    "tenure_days_at_origination": "Length of relationship with the bank is short",
    "loan_to_score_ratio": "Requested amount is high relative to your credit profile",
    "is_secured": "This product requires collateral",
    "segment_code": "Account relationship does not meet product requirements",
    "age": "Insufficient credit file depth",
}


def compute_shap_values(
    model,
    X: pd.DataFrame,
    background: pd.DataFrame | None = None,
    max_samples: int = 1000,
):
    """Compute SHAP values, choosing the right explainer for the model type.

    TreeExplainer is exact and fast for tree models. For linear models the
    coefficients already give a global explanation, but SHAP still gives
    per-applicant attributions, which is what adverse action needs.

    THE `background` ARGUMENT IS NOT OPTIONAL IN PRACTICE, AND GETTING IT
    WRONG PRODUCES SILENTLY USELESS OUTPUT.

    A SHAP value measures how far a feature pushes THIS prediction away from
    what the model would predict for a TYPICAL applicant. "Typical" is defined
    by the background distribution. If you pass a single row as both the thing
    being explained and the background, every SHAP value comes out as exactly
    zero — the row does not deviate from itself. No error is raised; you just
    get a page of zeros and, if you are not paying attention, an adverse
    action notice with no reasons on it.

    So: background should be a representative sample of the population the
    model scores, normally the training set. It defaults to X only for the
    convenience case where X is already a large sample.
    """
    import shap

    X_sample = X.head(max_samples) if len(X) > max_samples else X
    bg = background if background is not None else X_sample
    bg_sample = bg.head(max_samples) if len(bg) > max_samples else bg

    # Unwrap a sklearn Pipeline to find the estimator and any scaler.
    estimator = model
    scaler = None
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "predict_proba"):
                estimator = step
            elif hasattr(step, "transform"):
                scaler = step

    X_transformed = pd.DataFrame(
        scaler.transform(X_sample) if scaler is not None else X_sample.values,
        columns=X_sample.columns,
        index=X_sample.index,
    )
    bg_transformed = pd.DataFrame(
        scaler.transform(bg_sample) if scaler is not None else bg_sample.values,
        columns=bg_sample.columns,
        index=bg_sample.index,
    )

    if hasattr(estimator, "get_booster") or estimator.__class__.__name__.startswith(
        ("XGB", "LGBM", "RandomForest", "GradientBoosting")
    ):
        explainer = shap.TreeExplainer(estimator)
    else:
        # Background comes from the population, NOT from the rows being explained.
        explainer = shap.LinearExplainer(estimator, bg_transformed)

    values = explainer.shap_values(X_transformed)
    if isinstance(values, list):  # some explainers return one array per class
        values = values[1]
    return values, X_sample, X_transformed


def global_feature_importance(shap_values: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP per feature — the global view.

    Note this differs from a tree model's built-in `feature_importances_`,
    which measures split frequency or impurity reduction. SHAP measures actual
    contribution to predictions and is consistent across model types, which is
    why it is the defensible number to put in a model card.
    """
    return (
        pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def adverse_action_reasons(
    shap_row: np.ndarray,
    feature_values: pd.Series,
    population_medians: pd.Series,
    features: list[str],
    top_n: int = 4,
) -> list[AdverseActionReason]:
    """Generate the principal reasons for one applicant's decline.

    THE MECHANISM: take the SHAP values for this specific applicant, keep only
    the ones that pushed the risk UP (positive contribution to default
    probability), rank by magnitude, and translate the top few into plain
    language.

    Because SHAP is additive, these genuinely are the largest drivers of THIS
    decision — not a generic list, and not a post-hoc rationalisation. That
    distinction is exactly what Regulation B is asking for.
    """
    contributions = pd.Series(shap_row, index=features)
    adverse = contributions[contributions > 0].sort_values(ascending=False)

    reasons = []
    for rank, (feature, contribution) in enumerate(adverse.head(top_n).items(), start=1):
        reasons.append(
            AdverseActionReason(
                rank=rank,
                feature=feature,
                contribution=float(contribution),
                customer_value=float(feature_values.get(feature, np.nan)),
                population_median=float(population_medians.get(feature, np.nan)),
                explanation=REASON_TEMPLATES.get(
                    feature, f"{feature.replace('_', ' ').title()} is outside our criteria"
                ),
            )
        )
    return reasons


def explain_decision(
    model,
    applicant: pd.DataFrame,
    features: list[str],
    population: pd.DataFrame,
    top_n: int = 4,
) -> dict:
    """Full explanation for a single applicant: score plus principal reasons."""
    # The population is the background: it defines what "typical" means.
    shap_values, X_sample, _ = compute_shap_values(
        model, applicant[features], background=population[features]
    )
    probability = float(model.predict_proba(applicant[features])[:, 1][0])
    medians = population[features].median()

    return {
        "predicted_default_probability": probability,
        "reasons": adverse_action_reasons(
            shap_values[0], applicant[features].iloc[0], medians, features, top_n
        ),
    }
