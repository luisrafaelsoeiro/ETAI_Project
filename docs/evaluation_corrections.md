# Evaluation corrections — 2026-11-15

Branch: `data-preprocessing`. Starting commit: `535981dafcb7b6acf22537b055e127327ae46d09`.

## 1. Original units before outlier transformations

Notebook 04 previously applied log1p or winsorisation to the raw dataframe before the
numeric imputer. That changed inputs to formulas such as purchase count × mean value
and mixed transformed and original units in row bounds.

The common numeric pipeline now runs:

`RowBoundedImputer → optional numeric_transform → scale`

Notebook 04 imports `SkewLog1p`/`Winsorizer` from `src/outliers.py` and supplies the optional
transform through its recipe. Both learn from the imputed training fold only. The default
recipe still has no outlier transformation. Row identities apply in original units;
logarithms/clipping in model space need not preserve those identities afterwards.

## 2. Freeze history at each training cutoff

History filling used to precede the internal temporal split. An earlier validation
purchase could therefore provide a missing category or a previous-purchase date to a
later purchase in the same validation block.

`prepare_task_data` now returns statelessly prepared, unfilled matrices. The first pipeline
step, `CustomerHistoryFiller`, freezes the prepared train.csv reference at the latest date
of the rows passed to that particular `fit`. The source keeps only lookup predictors,
not target values. Each query additionally requires a same-customer, strictly earlier
source purchase. This excludes the embargo, validation period and future locked period.

Earlier purchases without a usable regression target remain valid predictor-only context.
Restricting the source to labelled returners would unnecessarily discard known past purchases.
The implemented policy is deliberately frozen per fit, not an online simulation in which
newly observed validation purchases can update history. Random-split diagnostics must
disable history; a date-cutoff policy is intended for temporal splits.

IDs, customer IDs and dates accompany X as lookup context. They are excluded from the
model matrix by explicit feature lists, with a validation error if listed as predictors.
Saved models retain their frozen history, so prediction needs no history file reload.
`predict_new_records(..., history=...)` retains its old argument for compatibility but
ignores it; only refitting can change the model's source. Disable history in the config
before fitting for a no-history experiment. A development-fitted model also uses its
development cutoff for test.csv; a later refit can use a later cutoff.

Because this step is inside the estimator, ordinary CV, Optuna trials, nested inner/outer
fits and notebook manual fit/predict loops all rebuild their own source. Notebooks 01
and 05 now use the common task preparation instead of prefilling their matrices.
Notebook 07 explanations also apply the fitted history step before preprocessing.

## 3. Descriptive paired comparisons

The old `abs(mean difference) > 2 × SE` rule treated dependent fold results too strongly.
It no longer produces a `decisive` verdict. `paired_comparison` reports mean advantage,
standard deviation, range, wins and ties; positive means A is better even for RMSE.
It rejects missing/non-finite or mismatched scores instead of silently ignoring failed folds.

`consistent_gain` means a strict improvement on every fold, with at least two folds.
This is a conservative screening heuristic, not a statistical test. Tiny improvements
still need practical justification. Notebook 07's provisional selection code retains
the current model unless its challenger passes this screen; the recommendation still
needs review of magnitude, reasoning and complexity.

The dependence problem is discussed in
[scikit-learn's statistical comparison example](https://scikit-learn.org/1.7/auto_examples/model_selection/plot_grid_search_stats.html).
Its corrected repeated-CV test was not applied as if it automatically covered expanding
temporal folds. Earlier numerical ablations are historical until rerun; previous claims
of significance/equivalence were withdrawn. Notebook 06's permutation error bars now
show between-fold SD, not a naive SE interpreted as uncertainty of an independent mean.

## Verification

- Pipeline suite: 31/31 passed. Added adversarial history-boundary/batch checks, original-unit
  formula/transform checks, training-only quantile checks, and descriptive comparison checks.
  The actual nested tuning test also records fits and checks every expected inner/outer cutoff.
- Both CLI tasks completed on the five configured temporal folds, with the same models,
  feature lists, hyperparameters and 90-day gap. Holdout evaluation was disabled in the
  validation run; the historical holdout has already been evaluated in earlier work.
- Both saved/reloaded models produced identical predictions and all 986 ordered test.csv IDs.

| Configured model | Previous run | Fold-local history | Difference |
|---|---:|---:|---:|
| Classification accuracy | 0.709743857 | 0.709504909 | −0.000238949 |
| Regression RMSE (€) | 1937.702591 | 1937.329331 | −0.373260 |

These are observed changes on development folds, not evidence of a predictive improvement.
The objective of these edits is correct transformation order and explicit evaluation boundaries.

The previous scores are recorded in the local `history_independence_run/runs.csv`;
the corrected CLI run is `20261115_232351`, recorded in `fold_local_run/runs.csv` under
the temporary review environment. The refreshed scores are also in the README.

Notebook 01 (29 code cells) and notebook 04 (13 code cells) were executed in full.
Notebook 04's refreshed comparisons show classification `log1p` accuracy 0.7160
(+0.0065, 4/5 folds), and regression robust-scaling RMSE 1936.7117 (0.62 euros lower,
5/5 folds). The latter passes the consistency screen but the relative effect is only
0.032%; no default recipe was changed. Isolation Forest removal gains 21.60 euros
on average in regression but only on 3/5 folds, with advantages from −68.85 to +183.52.
IQR removal worsens regression RMSE by 227.56 euros, on every fold.

Notebook execution scope is recorded at the top of each affected notebook. Full multi-model
Optuna tuning was not rerun or replaced with a smaller experiment presented as equivalent.
Outputs not refreshed under this protocol were cleared. The two proposed improvement
experiments (categorical history alternatives and tax-consistent `ticket_vs_usual`) are
not part of these corrections.

Reproduce the checks from the repository root with the dependencies in `requirements.txt`:

```powershell
python tests/test_pipeline.py
python main.py --task both
```

For a development-only verification run, copy config.yaml and set `test_set.evaluate: false`;
use `--config` to point to that copy and separate output directories. The checked environment
used Python 3.11, pandas 2.3.3, NumPy 2.4.6 and scikit-learn 1.7.2.
