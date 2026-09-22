# ADR 0012: Out-of-time validation is mandatory for credit models

## Status
Accepted — 2026-07

## Context
Random train/test splitting is the ML default. For credit risk it is wrong for
two compounding reasons:

1. **Temporal leakage** (Sprint 7, ADR 0011) — random splits let the model
   train on data from after the test period.
2. **Population drift** — the people who borrowed in 2022 are not the people
   who borrow in 2024. The economy moved, marketing moved, underwriting moved.
   A model validated on a random shuffle of its own era is tested on a
   population it will never see again.

## Decision
All credit models are validated out-of-time: train on an earlier window, test
on a strictly later one. `out_of_time_split()` enforces this and a test asserts
no temporal overlap.

## Rationale
Measured on our data with the logistic model: out-of-time test AUC 0.651.
A random split over 25 seeds gave mean 0.586 with a range of **0.527 to 0.671**
— a 0.144 swing driven purely by the seed.

That instability is the real argument, and it is a subtler one than "random
splits are optimistic". At 175 defaults, a single random split tells you almost
nothing; you could report anything from 0.53 to 0.67 honestly. The out-of-time
split is deterministic AND answers the question that matters: how will this
perform on next year's applicants?

## Consequences
+ Reported performance reflects real deployment conditions.
+ Train-test degradation becomes visible, and that degradation IS the drift.
+ Deterministic, so results are reproducible across runs and reviewers.
- Less training data (the most recent period is held out) and a single test
  window rather than an averaged estimate.
- Requires a reliable date on every row — which is why Sprint 1 generated
  `origination_date` and Sprint 3 built SCD Type 2 for point-in-time joins.

## Evidence this matters
XGBoost scored 0.967 on training data and 0.539 out-of-time. Under a random
split its apparent performance would have been far higher, and a model that is
barely better than a coin flip could have reached a validation committee
looking excellent.
