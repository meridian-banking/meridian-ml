# ADR 0013: Interpretability is a deployment requirement, not a preference

## Status
Accepted — 2026-07

## Context
XGBoost usually outperforms logistic regression on tabular data. The obvious
engineering decision is to deploy the better model. In consumer credit that
decision is constrained by law.

Under ECOA and Regulation B, a declined applicant must receive the SPECIFIC
PRINCIPAL REASONS for the decision. "The model declined you" is not compliant.

## Decision
Logistic regression is the primary credit model. XGBoost is fitted as a
challenger to quantify what interpretability costs. SHAP provides per-applicant
adverse action reasons for whichever model is used.

## Rationale
On our data the trade-off did not even arise — XGBoost overfit badly
(train 0.967, out-of-time 0.539) while logistic generalised better. But the
decision would stand even if XGBoost had won on AUC, because:

- A model that cannot produce defensible adverse action reasons cannot be
  deployed, regardless of accuracy.
- Logistic coefficients are directly auditable: a risk manager can check that
  higher credit score reduces predicted risk, and reject the model outright if
  the sign is wrong. We test exactly this
  (`test_coefficient_signs_are_economically_sensible`).
- Model risk management (SR 11-7) requires independent validation. Validating
  a model a validator can read is materially cheaper and faster.

Fitting the challenger anyway is the point: it converts "we use logistic
because we always have" into a measured statement about what the constraint
costs. If gradient boosting bought 0.08 AUC, that would be a real conversation
with compliance. Here it bought −0.11, so there is nothing to discuss.

## Consequences
+ The model is deployable, auditable, and explainable per applicant.
+ Adverse action reasons are arithmetically derived, not rationalised.
- Potentially lower predictive performance in settings where the non-linear
  model genuinely generalises.
- SHAP adds a computation step and a correctness hazard: passing the wrong
  background distribution silently yields all-zero values and an empty adverse
  action notice. Documented in `compute_shap_values` and covered by a test.
