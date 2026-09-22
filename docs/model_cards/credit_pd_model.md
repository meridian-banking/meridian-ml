# Model Card: Credit Risk PD Model

> A model card documents what a model is for, how it was built, how it
> performs, and — most importantly — where it should NOT be used. Under
> SR 11-7 (US model risk management guidance) this kind of documentation is
> expected, not optional, for any model used in credit decisions.

## Intended use

**Purpose:** estimate probability of default (PD) for retail loan applications,
feeding `Expected Loss = PD x LGD x EAD` for pricing and provisioning.

**In scope:** auto, personal, mortgage, and student loans to retail customers.

**Explicitly out of scope:**
- Commercial or business lending (different risk drivers entirely)
- Automated decline without human review — the model informs, it does not decide
- Any population materially different from the training window

## Model details

| | |
|---|---|
| Type | Logistic regression (primary), XGBoost (challenger) |
| Features | 10 origination-time attributes — see `CREDIT_FEATURES` |
| Target | Binary default indicator |
| Validation | Out-of-time: train pre-2024, test 2024 |
| Class handling | `class_weight='balanced'` |

**Why logistic regression is primary:** the coefficients are directly
interpretable, adverse action reasons fall out of them, and a regulator can
read the model. XGBoost is fitted as a challenger to quantify what
interpretability costs.

## Performance (out-of-time test, 772 loans, 43 defaults)

| Model | Train AUC | Test AUC | Test KS | Gap |
|---|---|---|---|---|
| Logistic regression | 0.608 | **0.651** | 29.4 | −0.04 |
| XGBoost | 0.967 | **0.539** | 15.4 | **+0.43** |

**The XGBoost result is the important finding.** A +0.43 train-test gap means
it memorised the training loans rather than learning risk. On unseen loans it
is barely better than chance. The logistic model is modest but *stable*, and
stability is what matters for a model that will score applicants for years.

An AUC of 0.65 is at the low end of typical retail scorecards (0.70-0.80). Two
honest reasons: only 175 defaults in total, and synthetic data whose default
mechanism is simpler than reality.

## Calibration

`class_weight='balanced'` improves ranking but **systematically inflates
predicted probabilities**. Acceptable for rank-ordering (who to decline);
**not acceptable for pricing or expected-loss** without recalibration. See
`calibration_table()`.

## Fair lending

- Screened with `check_prohibited_features()`: no ECOA-prohibited attributes.
- `age` is flagged as a proxy risk and would require documented justification
  or removal before production use.
- Adverse action reasons generated per-applicant via SHAP.

**Not yet done:** outcome-based disparate impact testing. That requires
protected attributes held by compliance and is a prerequisite for deployment.

## Known limitations

1. Trained on synthetic data with a simpler default mechanism than reality.
2. Small default count (175) — metrics have wide confidence intervals.
3. No macroeconomic features, so it cannot anticipate a downturn.
4. No LGD or EAD component; PD alone is not expected loss.
5. Calibration not corrected after class balancing.

## Monitoring requirements

- **PSI** on input features, monthly — population drift is why out-of-time
  validation was required in the first place.
- **Score distribution** stability.
- **Backtest actual vs predicted** default rates by score band, quarterly.
- Retrain trigger: PSI > 0.25 on any major feature, or AUC decline > 0.05.
