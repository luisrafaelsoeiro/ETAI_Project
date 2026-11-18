# Mines notebook review — 2026-11-17

Branch: `data-exploration`, based on `93c3975154b98656b51bae21113154f584240d89`.

## Exploration

Filled the 19 conclusion placeholders with short interpretations and next steps. Re-read the surrounding text and corrected unsupported missingness mechanisms, past-versus-future target interpretations, ABC share denominators, and claims of out-of-time RFM validation. Calendar, geographical and campaign interpretations now distinguish observed associations from predictive or causal evidence.

The notebook preserves both original targets. Conditional spend remains unknown for non-returners and for 229 labelled returners with no observed amount. A retrospective coverage diagnostic describes the dated subset without changing labels or adding future-derived predictors. Predictor correlation screens exclude targets, IDs and dates. Missing recency is no longer automatically labelled a first purchase.

All **91 code cells** executed successfully from a clean kernel after the final changes. End-of-notebook assertions check unchanged targets, retained rows and predictor-screen exclusions. Diagnostic plots were inspected. This is exploratory analysis, not a new model-selection experiment.

## Regression

Corrected a double sign inversion: `src.evaluate` already returns positive EUR errors. Human-readable tuning aggregates and their ordering now use positive errors; sklearn/Optuna's internal negative objectives remain unchanged. Random Forest has the lowest recorded mean nested RMSE, but tuning did not improve its RMSE over the untuned forest. The text also no longer claims it wins every fold.

The **full six-model/nested search was not rerun**. Published aggregates were archived with their original notebook hash and display precision, then sign-corrected algebraically. Retained tables carry provenance notes; these are not reconstructed raw fold records. Original residual and log-target experiment outputs were retained and their interpretation reviewed.

Fresh execution covered setup/helpers, the labelled mean-metric plot, permutation importance, its plot, spend-driver summaries and guarded holdout prediction (zero-based cells 3, 4, 21, 36, 37, 39, 24). Importance fits the published fixed forest parameters afresh on five development folds. It shuffles 37 model inputs after freezing fitted history, preserves pandas dtypes, and leaves lookup IDs/dates unchanged. One shuffle per feature/fold was used; across-fold standard deviations are descriptive, not confidence intervals. Parameters were selected on development data, so this diagnostic is not independent nested evaluation.

The largest mean MAE increases concern annual, 90-day and 180-day spending; annual-spend importance varies substantially across folds. These are model-reliance diagnostics, not causal effects or a completed feature-selection decision.

The published run had already scored the holdout. That result is archived in the evidence JSON and is not an unseen test. The refreshed prediction cell respects `test_set.evaluate: false` and did not score it. A future full Run All will rerun tuning; scoring remains governed by configuration.

## Execution and evidence

Resolved committed merge-conflict markers in `requirements.txt`, retaining the pipeline dependencies and adding seaborn. Adjusted the exploration palette for the existing seaborn runtime. No shared pipeline source or configuration was changed.

Validation: **34/34 existing pipeline tests passed**; both notebooks validate and their code cells compile; no saved error outputs or conclusion placeholders remain; TOC anchors resolve. The positive-score helper and the 185 feature/fold importance records were checked.

Evidence: [execution record](experiments/mines_notebook_review_20261117/execution_record.json), [published aggregates](experiments/mines_notebook_review_20261117/recorded_regression_results.json), [fold importance](experiments/mines_notebook_review_20261117/permutation_by_fold.csv), [importance summary](experiments/mines_notebook_review_20261117/permutation_summary.csv). The execution record includes runtime versions, data/config fingerprints and artifact hashes. Published aggregates are loaded for section-only replay only while the recorded data/config fingerprints match; this check does not establish equivalence of other environments or code revisions.

The exploration was subsequently moved to `notebooks/00_data_exploration.ipynb` on 2026-11-18. To reproduce it, run that notebook with a clean kernel from either the repository root or `notebooks/`; its data paths resolve the project root. Historical execution records retain the original filename. To replay the refreshed regression diagnostics without retuning, run setup/helper cells, the mean-metric plot, importance/driver cells, then guarded prediction in that order. A full regression execution is still required before presenting all results as one fresh run.

Relocation validation (2026-11-18): all 91 exploration code cells executed successfully with `notebooks/` as the working directory; setup/data loading also passed from the repository root. Only the data-loading cell source changed; analytical code and conclusions were preserved. Final integrity assertions passed.
