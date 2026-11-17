# Two controlled preprocessing experiments — 2026-11-16

The evaluation corrections were pushed as `6c90b8c` on `data-preprocessing` before these
experiments. Their code is in `src/experiments.py`; notebook 03 section 15 and notebook 06
section 12 display the numerical record and can replay the calculations.

## Protocol

- Same configured models, predictors, five expanding temporal folds and 90-day embargo.
  No hyperparameter tuning, statistical row removal, holdout scoring or Kaggle evaluation.
- Classification: logistic regression, seed 42. Regression: random forest with seeds 42, 1, 2.
  The target encoder retains its configured seed. Repetitions vary the forest seed only.
- Regression comparisons first average the three seeds **within each temporal fold**, then
  describe five paired fold differences. The 15 scores are not treated as independent folds.
- Tables below use the configured recipe as reference: history + mode in classification,
  history + Missing in regression, and no ticket feature in experiment 2.
- A positive advantage means the candidate is better. No significance or equivalence claim
  follows from the fold summaries. The all-fold consistency screen is descriptive.

The regression baseline here is **1938.5399**, the average across three seeds. The previous
configured seed-42-only result was **1937.3293**. Those are different summaries, not a change
to the dataset or a decline caused by these experiments.

## 1. Purchase-category history versus mode or Missing

Test a 2 × 2 design: last observed category on/off, crossed with mode/Missing fallback.
The previous-purchase-date recovery stays enabled in every candidate. Region, island and
municipality retain the configured customer-profile imputation. New optional per-column
`imputation.categorical_overrides` prevents changing those profile categories accidentally.
The default config has no overrides.

| Purchase-category policy | Classification accuracy | Folds better vs current | Regression RMSE (€), 3-seed mean | Folds better vs current |
|---|---:|---:|---:|---:|
| History + mode | **0.709505 (current)** | — | 1938.8268 | 3/5 |
| History + Missing | 0.712132 | 4/5 | **1938.5399 (current)** | — |
| No category history + mode | 0.710220 | 3/5 | 1937.2717 | 4/5 |
| No category history + Missing | 0.710460 | 3/5 | 1938.2395 | 2/5 |

Classification's best mean gain is **+0.002628 accuracy** (0.263 percentage points), with
one fold worse. Regression's best mean gain is **1.2682 euros** (about 0.065% of baseline),
with one fold worse after seed averaging. For the latter, individual seed mean gains are
0.0861, 0.4561 and 3.2625 euros; each seed improves on only 2–3 of its five folds.
None passes the all-fold consistency screen.

**Decision:** retain the current per-task policies provisionally. The alternatives have
mixed temporal effects and small gains. This does not prove the current policy is optimal.
In particular, transport is an estimate from a previous purchase, not an immutable customer
attribute. A field-specific policy remains a possible later experiment, not a result of
these purchase-category-wide comparisons.

### Reconstruction diagnostic

Within each validation block, hide 10% of known cells in purchase-category fields that
actually have development gaps. Every method receives the same masks. Fit both the frozen
history source and the fallback imputer on the training period only.

Classification development folds, pooled over the fixed masks:

| Field | Masked cells | History + mode correct | Mode only correct | History available | History correct when available |
|---|---:|---:|---:|---:|---:|
| Payment | 404 | 88.86% | 71.29% | 79.46% | 96.26% |
| Salesperson | 405 | 82.47% | 67.41% | 78.02% | 91.46% |
| Transport | 404 | 56.44% | 17.82% | 74.26% | 70.67% |

History reconstructs more of these hidden observed values, but it is far from a fixed
transport attribute. Missing-category encoding has zero exact reconstruction accuracy
when it intentionally outputs Missing; that is a representation of uncertainty, not
an attempted category estimate. Its value for the prediction task must be evaluated
through downstream CV, not reconstruction accuracy alone.

Random masking of observed values **does not establish accuracy on the original gaps**,
identify the missingness mechanism, or show that better reconstruction improves target
prediction. The regression reconstruction table is saved separately.

## 2. A coherent monetary basis for ticket_vs_usual

In **4428/4428** checked consecutive pairs from classification **development predictors**,
the historical-spend increment agrees with the previous purchase's `net_total + total_tax`
within 0.05 euros. Net-only value agrees in just **96/4428** pairs. This verifies the amount
basis on these observed pairs; it is not a test of prediction quality.

The candidate definition now uses:

```python
ticket_vs_usual = (net_total + total_tax) / prior_mean_purchase_value
```

Zero historical mean produces NaN, subsequently handled by the fitted imputer. Compare
no ticket feature, the old net-only ratio, and the corrected tax-inclusive ratio, leaving
all other predictors and preprocessing choices unchanged.

| Ticket option | Classification accuracy | Folds better vs no ticket | Regression RMSE (€), 3-seed mean | Folds better vs no ticket |
|---|---:|---:|---:|---:|
| Absent (current) | 0.709505 | — | 1938.5399 | — |
| Legacy net-only ratio | 0.710221 | 2/5, 2 ties | 1938.7122 | 3/5 |
| Tax-inclusive ratio | 0.709743 | 1/5, 2 ties | 1938.6456 | 3/5 |

The corrected ratio gains only **0.000238 accuracy** over absence. It is **0.1057 euros
worse** in mean regression RMSE. Against the legacy ratio, classification is worse on
two folds and tied on three; regression mean gains vary by seed (+0.8182, −0.7624, +0.1442
euros for seeds 42, 1, 2). This does not demonstrate a predictive improvement.

**Decision:** keep the candidate's corrected definition, but leave it out of both model
feature lists. A coherent feature definition and a useful additional predictor are
separate questions. No default model, feature list or category policy changed.

## Evidence, verification and replay

- Pipeline suite: **34/34 passed**, including per-column profile isolation/order,
  independent candidate recipes and tax-inclusive/zero-history ratio handling.
- [Numerical evidence](experiments/): per-fold scores for every model seed, paired results,
  masked reconstruction counts, monetary audit and a manifest/config snapshot.
  The snapshot records versions and fingerprints; notebook displays check semantic
  configuration and data fingerprints rather than depending on Windows line endings.
- Classification has 35 scored fits (20 categorical + 15 ticket); regression has 105
  (60 categorical + 45 ticket). Reconstruction fits history/imputation separately,
  without training target-prediction models. No holdout scores were inspected.

Run from the repository root:

```powershell
python tests/test_pipeline.py
python -m src.experiments --experiment both --output-dir results/preprocessing_experiments
```

The appended notebook sections default to displaying the saved evidence. Set their
`RECOMPUTE = True` to train and repeat the corresponding experiment on the current config
and data. Earlier notebook sections retain the execution-scope notes from the evaluation
correction; these experiments do not rerun the full historical feature/model searches.
