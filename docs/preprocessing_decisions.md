# Preprocessing decisions: rationale, evidence and open comparisons

Updated 2026-11-16. This register describes the configured recipe and the scope of its evidence. The refreshed grid and expanded KNN sensitivity results are in [the comparison record](recipe_comparisons.md); they do not establish a final optimal recipe.

## How to read a decision

For each choice, distinguish:

1. **Data definition:** a value follows from the available columns or a statistic is not applicable. Record the condition, formula and checks on observed rows.
2. **Modelling hypothesis:** a plausible representation or estimate, such as carrying forward a category, using a median or selecting a scaler. Explain the motivation and compare alternatives.
3. **Evidence status:** refreshed comparison, historical comparison awaiting rerun, or untested proposal. A configured option is not automatically the best one.

A domain rule needs its assumptions and checks. A modelling hypothesis needs assessment of the full prediction pipeline. Reconstruction of artificially hidden values and downstream prediction quality answer different questions.

## Current decisions

### 1. Recover values before statistical imputation

**Reason.** Use available information before estimating an unknown quantity. For example, with one previous purchase, its amount equals the historical total, minimum and maximum; with tenure at most 365 days, the full historical spend belongs to the 365-day window. These are conditional identities, not competing statistical imputers.

**Evidence and location.** Notebook [01](../notebooks/01_data_cleaning.ipynb), section 5, and notebook [03](../notebooks/03_missing_values.ipynb), section 6, document the conditions and observed-value checks. The rules are in `diagnostics.recompute_rules`. Checks on observed records support their use on this dataset; they do not verify the unavailable generating process.

### 2. Structural absence versus an unknown value

**Reason.** A first purchase has no previous-purchase gap. The configured structural fills use zero for specified undefined history fields, with `is_first_purchase` and `has_gap_history` to distinguish short histories. Zero is a modelling representation of “not applicable”; it is not an observed historical measurement. Missing future-spend targets are not filled.

**Evidence and status.** Notebook 01 section 5 and notebook 03 sections 1–2 describe the definitions. Excluding first-purchase rows when estimating the medians of `spend_365d` and `prior_max_purchase_value` is an additional modelling choice: their no-history values should not determine a fallback for established customers. Its full downstream comparison in notebook 03 sections 10–12 remains pending rerun.

### 3. Use the customer's earlier category before the fallback

**Reason.** A customer's last payment method, salesperson or transport method may be informative. This is a carry-forward estimate, not an immutable customer attribute. A date difference can recover the previous-purchase gap when that purchase is identified.

**Alternatives and evidence.** Notebook 03 section 15 and [the controlled experiments](preprocessing_experiments.md) compare category history on/off crossed with mode/explicit Missing, with profile-category treatment held fixed. History is frozen at each training cutoff; query batches cannot supply history. Reconstruction checks hide observed validation cells and are reported separately from target-prediction scores.

**Decision.** Keep history provisionally. The alternatives have mixed fold effects. The comparisons do not establish the best policy for every field or model.

### 4. Numeric fallback: median rather than KNN

**Reason.** Median is a simple reference estimator that does not depend on nearest-neighbour distance choices. Simplicity motivates the baseline; it does not establish predictive superiority.

**Alternatives and evidence.** Notebook 03 section 4 contains a refreshed comparison of median, mean, zero, KNN with five neighbours and iterative imputation. Each option is fitted within the same five temporal folds using the configured model for its task.

| Numeric fallback | Classification accuracy | Regression RMSE, euros |
|---|---:|---:|
| Median | 0.7095 | 1937.3293 |
| KNN | 0.7109 | 1937.2830 |

KNN improves three of five folds in each task. Its mean gain is about 0.14 percentage points of accuracy and 0.046 euros of RMSE. These are stored comparison results, not new runs for this document. The regression comparison uses the configured seed 42, not the three-seed mean from the later category experiment.

**Expanded evidence.** Section 12 is now executed. Classification KNN k=3 uniform reaches 0.711655, improving four folds and tying one against median 0.709505, with none worse. It is a candidate to carry forward, not something to reject because the strict all-fold screen treats ties as false. With three seeds of the median-tuned forest, the best regression KNN mean gain is 0.2288 euros, improving three of five seed-averaged folds. Complete-recipe timings do not show a large KNN slowdown in this run.

**Decision.** Keep median as the reference for model comparison and carry classification KNN k=3 as a candidate. Do not claim that median is best/equivalent or that KNN is universally settled. The forest sensitivity reuses tuning folds and is not final nested model assessment. Full scope and tables: [recipe comparisons](recipe_comparisons.md).

### 5. Estimate `spend_365d` from related columns in regression

**Reason.** `purchase_count_365d * prior_mean_purchase_value` incorporates customer-specific information. It is an estimate: the historical average need not equal the average in the last year. Median remains the fallback when formula inputs are unavailable.

**Evidence and decision.** Notebook 03 section 10 contains reconstruction diagnostics and a full downstream comparison awaiting rerun. The refreshed section 12 sensitivity test also compares removing this formula: median without the estimate improves mean RMSE by 9.0426 euros on four of five seed-averaged folds, but loses one fold and has uneven gains across periods. This makes no-formula a candidate for model-specific assessment, and does not support claiming a downstream advantage for the configured formula. Keep the current recipe as a reference while carrying the alternative forward. Good reconstruction on masked observed cells alone does not demonstrate improved spend prediction or accuracy on original missing cells.

### 6. Categorical fallback: mode in classification, Missing in regression

**Reason.** Mode estimates a likely category; explicit Missing preserves the fact that the category is unknown. They represent different hypotheses about what helps prediction. Association diagnostics motivate investigation but do not prove either choice.

**Evidence and decision.** Notebook 03 section 4 compares the fallbacks. Section 15 separates purchase-category policy from customer-profile fields in a 2 × 2 history/fallback experiment. Classification's best mean is history + Missing, 0.712132 versus the current 0.709505, improving four of five folds. Regression alternatives also have mixed effects. Keep current policies provisionally; see the full paired results in [the experiment record](preprocessing_experiments.md). Do not describe the configured choices as demonstrated winners.

### 7. Row limits and the missingness indicator

**Reason.** Limits apply to estimated numeric values using the row's available constraints. For example, 365-day spend must respect available shorter-window and all-history spend limits. Notebook 03 section 8 documents the constraints; the imputer leaves observed values unchanged.

The `spend_365d` missingness flag is a separate modelling hypothesis: after recoveries, the remaining gaps have a particular history/tenure pattern that the model may use. Its presence is not proof of MNAR, and adding a flag to every missing column is not automatically useful or harmful. The indicator policy should be reconsidered with the selected models.

### 8. Encoding and scaling

**Current recipe.** Target encoding for both tasks; StandardScaler for classification and MinMaxScaler for regression. Scaling applies to the numeric branch after imputation and row limits. Encoded categorical columns and binary flags have separate branches.

**Reason and alternatives.** Notebook [05](../notebooks/05_pipeline_and_models.ipynb), section 2, now has the refreshed grid of four encoders (one-hot, ordinal, count, target) and four scalers (none, standard, minmax, robust). Classification target + minmax has the best mean, 0.714041, but improves only two folds against target + standard 0.709505. Ridge target + minmax has the lowest mean RMSE in its grid, 2032.6187 euros; it beats target + standard and robust on all five folds. These are conditional linear-model comparisons, not universal recipe rankings or the configured forest's score.

**Decision.** Use the current pairs as references and retain classification minmax as a candidate. Reassess appropriate recipes with model hyperparameters: regularisation at fixed C/alpha interacts with scaling. Scaling parameters, imputer statistics and encoders are fitted within each training fold. The target-encoding fit uses cross-fitting; model selection still needs outer temporal evaluation. See [the refreshed numerical record](recipe_comparisons.md).

### 9. Outlier treatment

**Reason.** A high but coherent purchase is not a confirmed data error. Statistical flags identify observations to inspect; they do not alone justify deletion.

**Evidence and decision.** Notebook [04](../notebooks/04_outlier_detection.ipynb), sections 8–11, has refreshed transformation/removal comparisons. No statistical outlier transform/removal is enabled in the default recipe. Classification log1p reached about 0.7160 accuracy, improving four of five folds; this remains a candidate. IQR removal worsened regression on every fold in that comparison. See [the evaluation record](evaluation_corrections.md) for the scope and other results. Reassess treatments with shortlisted models, inside training folds.

### 10. Feature inclusion and selection

**Reason.** Definitions, redundancy and association can suggest candidate features; predictive usefulness must be assessed with the model. Correlation or low univariate association alone does not settle inclusion for every model.

**Evidence and status.** Notebook [06](../notebooks/06_feature_engineering_selection.ipynb) contains additions, ablations and selector comparisons, with earlier full searches awaiting rerun. Fixed feature lists remain provisional and automatic selection is disabled. Section 12 has refreshed evidence for the tax-inclusive `ticket_vs_usual` candidate: the monetary audit supports the corrected amount basis, but the prediction comparison does not support adding it to either current model.

## Interpreting results and completing the evidence

- Compare the full recipes on the same temporal folds and report metric direction, paired mean differences, variability and folds improved. The all-fold `consistent_gain` screen is descriptive, not a significance test or an automatic selection rule.
- Separate the quality of a reconstructed value from the final target score. Random masks of known values do not identify the mechanism or true values of original gaps.
- A non-significant association test does not establish MCAR; associations do not distinguish MAR from MNAR. Do not infer a preferred imputer from an identified mechanism that has not been established. See [van Buuren, concepts of missingness](https://stefvanbuuren.name/fimd/sec-MCAR.html).
- The expanded KNN sensitivity and encoding/scaling grid are refreshed. Other pending feature/estimate searches retain that status. Compare shortlisted model/recipe combinations before presenting final choices, and keep numerical conclusions tied to their exact model and protocol.
- This document changes the explanation and evidence status, not the configured preprocessing, features or models.
