# Refreshed encoding/scaling and expanded KNN comparisons — 2026-11-16

## Protocol and scope

These are development-set comparisons using the current configured features, imputations, five expanding temporal folds and 90-day gap. History is frozen at each fit's training cutoff. No holdout scores or Kaggle results enter the comparisons. The input data and semantic configuration match the preceding controlled experiments.

The numerical record is in [experiments/recipe_comparisons](experiments/recipe_comparisons/): configuration snapshot, runtime/data/source fingerprints, per-fold scores, paired summaries and the forest diagnostic search. Notebook [05 section 2](../notebooks/05_pipeline_and_models.ipynb) contains the encoder/scaler grid; notebook [03 sections 3 and 12](../notebooks/03_missing_values.ipynb) contains the corrected association labels and KNN sensitivity analysis. Only the recorded sections were executed; other pending searches remain pending.

Paired differences are descriptive: positive advantage means the candidate is better, including for RMSE. Folds share training history and are not independent observations. No significance or equivalence claim follows from these tables. An all-fold improvement is a descriptive consistency screen, not an automatic selection rule.

## 1. Encoder × scaler

Four encoders (one-hot, ordinal, count, target) crossed with four numeric scalers (none, standard, minmax, robust), for 16 recipes per task and 160 scored fold fits in total. The comparison models are configured logistic regression and Ridge, without new hyperparameter tuning. All other recipe settings remain fixed; scaling applies to the numeric branch, while categorical encoding and flags have separate branches.

| Task / comparison model | Recipe | Mean score | Difference from current | Folds better |
|---|---|---:|---:|---:|
| Classification / logistic regression | target + standard, current | accuracy 0.709505 | reference | — |
| Classification / logistic regression | target + minmax | accuracy 0.714041 | +0.004536 | 2/5 |
| Classification / logistic regression | onehot + minmax | accuracy 0.712372 | +0.002867 | 3/5, 1 tie |
| Classification / logistic regression | target + robust | accuracy 0.709745 | +0.000240 | 3/5, 1 tie |
| Regression / Ridge | target + minmax, current | RMSE 2032.6187 euros | reference | — |
| Regression / Ridge | ordinal + minmax | RMSE 2044.2233 euros | 11.6046 euros worse | 1/5 |
| Regression / Ridge | target + standard | RMSE 2186.2858 euros | 153.6670 euros worse | 0/5 |
| Regression / Ridge | target + robust | RMSE 2179.7935 euros | 147.1748 euros worse | 0/5 |

**Decision.** Retain the current recipes as references. Classification target + minmax has the best mean but loses three folds against target + standard. Its paired advantages range from -0.005974 to +0.017900 accuracy. Treat it as a candidate for the model comparisons, without claiming that standard is superior or that the pairs are equivalent. Ridge target + minmax has the best mean in its grid and beats target + standard/robust on all five folds; this supports retaining that Ridge reference.

These scores are not an assessment of the configured regression forest. Scaling changes the effective regularisation at fixed C/alpha, so revisit recipe choices alongside hyperparameters for shortlisted models. The best observed grid score is development-selection evidence, not an independent estimate of the selection procedure's final performance. Production configuration was not changed.

## 2. Missingness association labels

The notebook's diagnostic table now reports `association_screen`: `none detected` or `detected; assess indicator`. It no longer labels these test outcomes as MCAR or MAR/MNAR. The table and the raw-tenure check were re-executed on the 5024 classification development rows, excluding holdout labels. The raw-tenure p-value is 0.834; it does not establish MCAR. Missingness-mechanism and indicator choices need separate reasoning; an association alone does not automatically select an imputer or justify a flag. See the [decision register](preprocessing_decisions.md).

## 3. Expanded KNN sensitivity analysis

All 210 scored comparison fits completed. The experiment uses median plus KNN with 3/5/10/20 neighbours and uniform/distance weighting. Classification uses the configured logistic regression. Regression uses three seeds of a forest selected by a 30-trial Optuna search with median, plus two no-formula diagnostics. The selected forest has 234 trees, depth 16, minimum leaf size 11 and max_features 1.0. Its median-reference scores are not the scores of the configured 300-tree forest.

This is a sensitivity comparison conditional on a median-tuned forest. Tuning and the comparison reuse development folds; it is not nested evaluation and does not establish an unbiased tuned-model performance estimate or joint optimisation of every imputer/model recipe. Seeds are averaged within each fold before calculating five paired differences. Runtime is the median of five serial fit-and-transform timings of the full recipe, including frozen history, after a warm-up; it is descriptive for this machine.

| Task | Variant | Mean score | Advantage over median reference | Folds better / tied |
|---|---|---:|---:|---:|
| Classification | Median | accuracy 0.709505 | reference | — |
| Classification | KNN k=3 uniform | accuracy 0.711655 | +0.002151 | 4 / 1 |
| Classification | KNN k=3 distance | accuracy 0.711178 | +0.001673 | 4 / 1 |
| Classification | KNN k=5 uniform | accuracy 0.710939 | +0.001434 | 3 / 2 |
| Regression, three seeds | Median | RMSE 1937.0712 euros | reference | — |
| Regression, three seeds | KNN k=3 distance | RMSE 1936.8424 euros | 0.2288 euros | 3 / 0 |
| Regression, three seeds | KNN k=5 uniform | RMSE 1937.2316 euros | -0.1603 euros | 2 / 0 |
| Regression, three seeds | Median without spend formula | RMSE 1928.0286 euros | 9.0426 euros | 4 / 0 |
| Regression, three seeds | KNN k=5 uniform without formula | RMSE 1929.1666 euros | 7.9046 euros | 4 / 0 |

**Classification decision.** KNN k=3 uniform is a candidate: it improves four folds and ties one, with none worse. Its strict all-fold `consistent_gain` flag is false because of the tie; this must not be interpreted as a reason to discard it. Keep median as the reference while carrying KNN into model/recipe comparisons. This experiment does not support a claim that median is superior.

**Regression decision.** The best KNN mean gain with the current formula is about 0.23 euros, with mixed temporal effects. Retain median as the reference. Removing the spend formula is a more promising candidate in this conditional experiment: it improves the mean for each forest seed, and four of five seed-averaged folds. The paired fold advantages range from -1.6990 to +37.2700 euros, with fold SD 16.2577. The effect is uneven across periods; it does not establish a final choice or the cause of the change. Reassess this option with shortlisted models. Do not cite this run as evidence that the current formula improves downstream prediction.

**Cost.** Classification median fit-and-transform took a median 1.2827 s; KNN k=3 uniform took 1.3387 s (1.044 times). Regression medians ranged around 1.22–1.36 s across these variants. This benchmark does not support the historical claim of a large KNN slowdown. It measures the fitted history/preprocessing recipe on development data, excluding stateless input preparation and model training; small timing differences are not portable performance guarantees.

The earlier five-neighbour comparison with the configured forest remains a separate experiment (notebook 03 section 4). The classification minmax and KNN candidates have not been combined in this run. KNN distances use the configured scaler; evaluate combinations alongside regularisation/model tuning rather than extrapolating these conditional results.

## Verification and replay

- Existing pipeline suite: **34/34 passed** after the notebook changes.
- The production configuration values are unchanged; only its explanatory comments were updated. No source module, model default or feature list was changed.
- Per-fold CSVs allow reconstruction of all ranking/paired summaries. The manifest records data/config/source fingerprints, runtime, selected sections and the diagnostic tuning scope. A snapshot is evidence of this run, not an automatically refreshed manifest for future notebook runs.
- To repeat the grid, execute notebook 05 setup and section 2. For KNN, execute notebook 03 setup/helpers and section 12; section 3 can be repeated independently after setup. These runs write the experiment CSVs. Preserve the recorded output folder or use a copy before replaying, and regenerate provenance for a new experiment.
- Broader model rankings, joint nested tuning, feature searches and full spend-estimate comparisons were not executed by these selected sections. Their pending status remains explicit.
