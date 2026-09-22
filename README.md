# meridian-ml

Credit risk, fraud detection, churn, and segmentation models for the Meridian platform. This is the sprint where the ground-truth labels planted back in Sprint 1 finally get used — and where the validation discipline from Sprint 7 gets applied under regulatory constraints.

## What makes this *banking* ML rather than generic ML

| | |
|---|---|
| **Out-of-time validation is mandatory** | Random splits leak *and* ignore population drift |
| **Accuracy is actively misleading** | Fraud is 0.15% — "predict never" is 99.85% accurate |
| **Explainability is a legal requirement** | ECOA/Reg B: declined applicants must get specific reasons |
| **Calibration matters, not just ranking** | `Expected Loss = PD × LGD × EAD` needs a real probability |

## Measured results

**Credit PD, out-of-time (train pre-2024, test 2024, 772 loans / 43 defaults):**

| Model | Train AUC | Test AUC | Gap |
|---|---|---|---|
| Logistic regression | 0.608 | **0.651** | −0.04 |
| XGBoost | 0.967 | **0.539** | **+0.43** |

XGBoost memorised the training set and landed barely above chance on unseen loans. The scorecard is modest but *stable* — which is the entire argument for interpretable models in credit, made with numbers rather than assertion.

**Why a single random split tells you nothing here:** across 25 seeds, random-split AUC ranged **0.527 to 0.671**. You could honestly report almost anything. The out-of-time split is deterministic and answers the question that matters.

**Fraud, 180k held-out transactions (291 fraud):**

```
"predict never fraud" accuracy:  99.8383%   <- catches ZERO fraud
our model accuracy:              99.5711%   <- WORSE on accuracy
our model recall:               100.00%     <- catches ALL fraud
average precision:                0.8153    (vs 0.0016 random)
```

If you evaluated on accuracy, you would pick the useless model.

**Threshold selection by dollars, not by 0.5** (review cost $4/alert):

| Threshold | Alerts | Precision | Recall | Net benefit |
|---|---|---|---|---|
| 0.50 (default) | 1,063 | 27.4% | 100.0% | $94,504 |
| **0.95 (optimal)** | **670** | **41.6%** | **95.9%** | **$95,097** |

The net-benefit curve is flat between 0.5 and 0.95, which is the practical finding: you can cut analyst workload 37% for essentially no cost.

## Adverse action notices (Regulation B)

SHAP produces per-applicant principal reasons that are arithmetically true, not rationalised:

```
Predicted default probability: 78.64%   Decision: DECLINE

1. Debt-to-income ratio is higher than we can accept
   (dti=0.62, typical=0.34, impact=+0.9356)
2. This product requires collateral
3. Requested loan amount is high relative to your profile
```

**A hazard worth knowing:** passing the wrong background distribution to SHAP produces **all-zero values silently** — no error, just an adverse action notice with no reasons on it. In a bank that ships as a compliance failure. Documented in `compute_shap_values()` and covered by `test_shap_background_must_differ_from_explained_rows`.

## An honest negative result

The churn model **does not train**, and that is the correct behaviour. Our generator gives every account steady activity — median gap between transactions is 3 days, and the longest silence across 10,700 accounts is 42 days. The 90-day churn definition yields zero positives, so `fit_churn_model` raises `InsufficientPositivesError`.

The tempting fix is to drop the window to 30 days until positives appear. That defines churn to fit the data rather than the business question, and produces a model that predicts "was briefly quiet" instead of "left us". The honest conclusion is that this synthetic source does not model attrition.

## Quick start

```bash
pip install -e ".[dev]"
make test    # 25 tests, no database required
make lint
```

## Modules

| Module | Purpose |
|---|---|
| `metrics` | AUC, **KS**, **Gini**, calibration tables, cost-based thresholds |
| `features` | Leakage-safe feature construction with velocity aggregates |
| `credit_risk` | WoE/IV, out-of-time splits, logistic + XGBoost PD models |
| `explainability` | SHAP, fair-lending screening, adverse action reasons |
| `churn_segmentation` | Churn with window design, k-means with honest k selection |

See [`docs/model_cards/`](docs/model_cards/) for the PD model card and [`docs/adr/`](docs/adr/) for the validation and interpretability decisions.

Part of the 8-repository Meridian platform.
