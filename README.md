
# ETAI 2026/27 | Customer Repurchase & Spend Prediction

## Project Overview

This project was developed as part of the **ETAI 2026/27 Group Project**.


The project focuses on **Everything & Then Some, Ltd.**, a diversified wholesale company whose customers have different purchasing patterns. The objective is to use historical transaction data to support more targeted commercial actions.

The project aims to answer two questions for each customer purchase:

1. **Will the customer make another purchase within the next 90 days?**
2. **If the customer returns, how much will they spend during those 90 days?**

The answers to these two questions can support different commercial actions. Customers who are unlikely to return can be targeted with incentives, while customers expected to spend more can be prioritized for retention and account management. The final solution should therefore be capable of making both predictions for a new purchase.

---

## Project Members

20231693 David Carrilho
20260620 Diogo Padilha
20211536 Luís Soeiro
20231630 Maria Inês Santos


## Project Objectives

The project is divided into several main objectives:

* Explore and clean the historical transaction data.
* Develop and optimize a **classification pipeline** to predict whether a customer will purchase again within 90 days.
* Develop and optimize a **regression pipeline** to predict the amount the customer will spend during the following 90 days.
* Generate predictions for the Kaggle competition.
* Build terminal-based pipelines for training and prediction.
* Provide predictions for individual new purchases through a client interface.
* Apply explainability methods to understand the predictions of both models.

---

## How to Run

```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python main.py                          # both tasks: classification, then regression
python main.py --task classification    # or: --task regression
python tests/test_pipeline.py           # 10 automated checks of the pipeline (or: python -m pytest tests)
```

The notebooks in `notebooks/` run with the same environment and find the project root themselves.

### What `python main.py` does, per task

1. **Load and validate**: placeholder tokens → `NaN`, placeholder report, schema validation (types, ranges, allowed categories) and `customer_id` checks.
2. **Clean** with stateless rules (train.csv and test.csv alike), prepare and join the customer table (`src/customers.py`), and remove almost-empty rows and duplicates from the training file only.
3. **Lock the temporal test set**: the latest 20% of purchases, with a 90-day gap.
4. **Cross-validate** the configured pipeline (preprocessing + model) on 5 expanding-window temporal folds, against a dummy floor, on every configured metric. The out-of-fold predictions feed the report.
5. **Optional tuning:** Optuna with nested temporal cross-validation.
6. **Refit** the selected pipeline on all development rows, and add a row to the scoreboard.
7. **Optional, only for the final chosen models:** evaluate once on the locked test set.
8. **Save** the fitted pipeline, reload it, and check that the reloaded copy predicts identically. Then write the test.csv predictions and the full run report.

| Output | Content |
|---|---|
| `results/runs.csv` | scoreboard: one row per run and task (model, recipe, CV mean ± std, per-fold scores, untuned score, dummy floor) |
| `results/<task>_<model>_<run>.txt` | full report: validation summary, per-fold scores, overfitting gap, tuning, out-of-fold report (confusion matrix / errors by spend band), locked-test report when enabled |
| `results/predictions/classification.csv` | test.csv predictions in the Kaggle format (`ID`, `repeat_purchase_90d`), checked against `sample_submission.csv` |
| `results/predictions/regression.csv` | test.csv predictions (`ID`, `spend_next_90d`) |
| `models/<task>_pipeline.joblib` | fitted pipeline (preprocessing + model), ready for `src/pipeline.py` → `load_pipeline()` / `predict_new_records()` |

`results/` and `models/` are generated and git-ignored.

### Changing the experiment (`config.yaml`)

| To… | Edit |
|---|---|
| try another model | `<task>.model.type` and `params`. Options: dummy, logistic_regression / ridge, decision_tree, random_forest, gradient_boosting, hist_gradient_boosting, xgboost |
| tune it | `<task>.tuning.enabled: true` (search spaces in `<task>.tuning.search_spaces`, keyed by model type) |
| change the metrics | `<task>.metrics` and `<task>.selection_metric` (options in `<task>/evaluate.py` → `SCORING`) |
| change the recipe | `<task>.preprocessing` (encoder, scaler, imputation, extra features) or the shared `preprocessing` section |
| score the locked test set (final models only) | `test_set.evaluate: true`, plus `refit_on_all_rows: true` to refit on every labelled row afterwards |

---

## Project Structure

```text
ETAI_Project/
│
├── main.py                         # entry point: python main.py [--task classification|regression|both]
├── config.yaml                     # every rule and choice, in pipeline order (no fitted values)
├── requirements.txt
├── README.md
│
├── data/
│   ├── README.md                   # data dictionary
│   └── raw/                        # train.csv, test.csv, clientes.csv, sample_submission.csv (never modified)
│
├── notebooks/                      # evidence for every decision (they import the pipeline's own code)
│   ├── 01_preprocessing.ipynb      # diagnosis, cleaning, missingness, recipe, encoder x scaler, models, ablations, imputers
│   ├── 02_outlier_detection.ipynb  # data errors vs valid extremes, transformations and row removal in temporal CV
│   ├── 03_feature_engineering_selection.ipynb   # redundancy, relevance, permutation importance (in progress)
│   └── 04_customer_table.ipynb     # full preprocessing of clientes.csv, tested in temporal CV
│
├── src/                            # shared by both tasks
│   ├── data.py                     # loading, placeholder report, column names, schema and customer_id validation, dtypes
│   ├── cleaning.py                 # stateless cleaning rules, customer join, prepare_table, training-only row removal
│   ├── customers.py                # customer table (clientes.csv): validation, codes, hierarchy recovery, structural fill
│   ├── preprocessing.py            # (X, y, extras), locked test split, the fitted recipe (ColumnTransformer)
│   ├── pipeline.py                 # build / save / load the pipeline, prepare_task_data, predict_new_records
│   ├── evaluate.py                 # temporal folds, cross-validation, comparisons, CV report
│   ├── tuning.py                   # Optuna tuning, nested temporal cross-validation, tuning reports
│   ├── results.py                  # scoreboard, run reports, prediction files
│   └── tasks.py                    # registry of the two task packages
│
├── classification/                 # only what is specific to repeat_purchase_90d
│   ├── preprocessing.py            # training rows, (X, y, extras)
│   ├── model.py                    # classifier registry
│   └── evaluate.py                 # metrics (SCORING), probability for the reports, confusion-matrix reports
│
├── regression/                     # only what is specific to spend_next_90d
│   ├── preprocessing.py            # training rows (repeaters with an observed spend), (X, y, extras)
│   ├── model.py                    # regressor registry, optional log1p target transform
│   └── evaluate.py                 # metrics (SCORING), error reports by spend band
│
├── tests/
│   └── test_pipeline.py            # 10 checks: feature lists, indicators, leakage, cleaning, schema,
│                                   # placeholders, save/load/predict, tuning, Kaggle format
│
├── results/                        # generated by main.py (git-ignored)
└── models/                         # generated by main.py (git-ignored)
```

### How the code is organised

* **One implementation of everything the two tasks share** (`src/`): loading, validation, cleaning, the recipe, cross-validation, tuning, saving results. Each task package (`classification/`, `regression/`) holds only what genuinely differs:
  * which rows the task trains on;
  * its models;
  * its metrics and reports.

  All three have the same interface, so `main.py` runs both tasks with the same code (`src/tasks.py` → `TASKS`).
* **No per-task `tuning.py` or `results.py`.** Tuning (`src/tuning.py`) and saving results (`src/results.py`) work the same way for both tasks. What differs per task is configuration: search spaces in `<task>.tuning`, the target name in the prediction file. That lives in `config.yaml`, not in duplicated code.
* **The notebooks import the pipeline's functions instead of redefining them**, so the evidence and `main.py` always run the same code.
* **Every rule and choice is in `config.yaml`**, with a comment pointing to the notebook section that justifies it.

---

## Data

See `data/README.md` for the files and the column dictionary. Each row is a **customer purchase occasion**. Every predictor is available at the time of the purchase, so no information comes from the future or from other test observations.

---

## Pipeline Progress

| Stage | What changed | Where |
|---|---|---|
| Project setup | Repository structure, raw data, requirements | — |
| Outlier exploration | Univariate (IQR) and multivariate (DBSCAN, K-Means, SOM) exploration (first version, since replaced) | — |
| Data diagnosis | Column standardisation, invalid values, placeholder, almost-empty rows, category text, target missingness, date checks, duplicates | `notebooks/01_preprocessing.ipynb` sections 1–4 |
| Preprocessing → pipeline | Diagnosis turned into config-driven rules; identity recovery; structural missingness; missingness-mechanism tests; temporal locked test set; leak-safe `ColumnTransformer` | `src/`, `config.yaml`, notebook sections 5–8 |
| Encoder × scaler + models | Temporal cross-validation, 16-recipe comparison per task, paired comparisons, Pipeline with dummy / linear / tree / random forest | `src/evaluate.py`, `*/model.py`, notebook sections 9–10 |
| Regression target + notebook clean-up | Missing-spend mechanism tested (not MCAR), `log1p` target tested and rejected; notebook sections 1–4 call the `src/` functions | `regression/model.py`, notebook 01 |
| Data quality + additions + imputers | Accounting identity explained; consistency rules fix inflated values; purchase-time features, customer table and rare-category grouping tested by ablation; 5 numeric and 2 categorical imputers compared in CV | `src/cleaning.py`, `src/evaluate.py`, notebook sections 5.2, 11, 12 |
| Data types + outliers | Compact dtypes (`standardize_dtypes`); `prior_max_purchase_value ≤ prior_total_spend` rule; outlier analysis with temporal validation (no statistical deletion or capping) | `config.yaml` → `dtypes`, `notebooks/02_outlier_detection.ipynb` |
| Validation + pipeline checks | Placeholder-token report and NaN-on-load; schema validation; `customer_id` validation; feature-list checks; missing indicators inside the pipeline; save → load → predict new records | `src/data.py`, `src/pipeline.py`, `tests/`, notebook sections 2.2, 2.10, 4.4, 8 |
| Customer table | Full preprocessing of clientes.csv: validation, key and duplicate checks, labels → codes, exact recovery through the geographic hierarchy, island as structural `"none"`, "customer before the records" flag, profile completeness, rare-level back-off (tested, off) | `src/customers.py`, `config.yaml` → `customers`, `notebooks/04_customer_table.ipynb` |
| Repository reorganisation | Baseline-pipeline structure. Shared code split into `src/data`, `cleaning`, `preprocessing`, `pipeline`, `evaluate`, `tuning`, `results`; task packages reduced to what differs (`preprocessing`, `model`, `evaluate`). `main.py` runs the full workflow (dummy floor, temporal CV, optional Optuna + nested CV, scoreboard, run reports, optional locked-test evaluation, saved models, Kaggle predictions). Notebooks moved to `notebooks/`; redundant files removed | whole repository |

---

## Preprocessing Decisions

Every decision below is backed by evidence in `notebooks/01_preprocessing.ipynb` (section numbers in brackets), `notebooks/02_outlier_detection.ipynb` or `notebooks/04_customer_table.ipynb`, and stored as a rule in `config.yaml`. A column-by-column summary is at the end (§12).

### 1. Where each step runs

Each preprocessing step is one of three kinds, and the kind decides where it may run. This is what keeps the pipeline leak-free and usable on new data.

| Kind | Steps | Learns from data? | Runs on |
|---|---|---|---|
| **Stateless rule** | placeholder tokens → `NaN` on load; schema validation; renames; invalid values → `NaN`; consistency rules; identity recomputation; structural fill; flags; date parts; ratios; dropped columns; customer-table join; compact dtypes. All run through `prepare_table` | No | train.csv, test.csv **and** any new record |
| **Training-data decision** | almost-empty rows, duplicates (`drop_training_rows`); regression rows (`regression.select_training_rows`) | No, but removes rows | training file only, before the split |
| **Fitted step** | imputation, missing indicators, rare-category grouping, encoding, scaling (`build_preprocessor`) | Yes | inside the model `Pipeline`, fitted on training rows only |

`prepare_table()` never removes rows (same rows, same order), because every one of the 986 test.csv ids needs a prediction. It doesn't need the target either: on test.csv, `y` comes back as `None`.

### 2. Cleaning & validation

#### 2.1 Column names [1.1]

Portuguese column names are converted to English `snake_case` (e.g. `Total a Pagar` → `total_payable`, `Custo` → `cost`, `N_ Detalhes` → `line_item_count`). Explicit names come from `diagnostics.column_renames`, and a generic rule covers the rest. The function raises an error if two columns end up with the same name.

#### 2.2 Placeholder tokens: report, frequency, treatment [2.2]

`report_placeholders()` reads each raw file as plain text, before pandas converts anything, and counts every token in `diagnostics.placeholder_tokens` (blank, `-`, `--`, `?`, `N/A`, `NA`, `null`, `None`, `nan`, `missing`, `unknown`, …). It also counts the numeric placeholder value:

| Placeholder | train.csv | test.csv | clientes.csv | Treatment |
|---|---|---|---|---|
| blank cell `''` | 12,927 (37 columns) | 489 (12 columns) | 3,385 (9 columns) | → `NaN` when the file is loaded |
| `?`, `N/A`, `-`, `null`, `unknown`, … | 0 | 0 | 0 | → `NaN` when the file is loaded (`load_data(path, placeholder_tokens)`), so a future file using them can't turn a numeric column into text |
| `1,000,000,000` (in `prior_mean_purchase_value`, `spend_365d`) | 31 (16 / 15) | 4 (2 / 2) | — | → `NaN` in `clean_dataset` (`flag_invalid_values`), then recovered from identities (§2.5) |

The blank cells are the real gaps of the data. Their mechanism is analysed in §2.6–2.7.

#### 2.3 Schema validation: expected types, valid ranges, allowed categories [2.10]

`validate_schema()` is the first step of `prepare_table()`, so it runs on train.csv, test.csv and every new record. Its rules live in `config.yaml` → `schema`:

| Check | Rule | train.csv | test.csv | Treatment |
|---|---|---|---|---|
| required columns | 37 purchase columns; targets and `random_noise` optional | all present | all present | error if one is missing; extra columns reported |
| expected types | 33 numeric columns, 11 whole-number columns, `purchase_date` a date | 0 problems | 0 problems | text in a numeric column → `NaN` (counted); unparseable dates counted |
| valid ranges | every non-negative column ≥ 0; `purchase_date` 2010–2030; ≥ 1 line item and invoice; target 0/1; spend ≥ 0 | 63 out of range (the negatives of §2.4) | 8 | counted here, set to `NaN` by `clean_dataset` |
| allowed categories | each label follows `"<fixed text> <number>"`, e.g. `'Maybe Tomorrow Payment \d+'` | 0 not allowed | 0 not allowed | a label that doesn't match → `NaN` (counted). New but well-formed labels are allowed; the encoders handle them |

After cleaning, `standardize_dtypes()` stores counts and ids as nullable integers (`Int8` / `Int16` / `Int32`), categories as `category`, and keeps monetary values and calculated statistics as `float64` (`config.yaml` → `dtypes`).

#### 2.4 Invalid values → missing [2.2]

| Problem | Columns | Train | Test | Decision |
|---|---|---|---|---|
| Negative values (impossible for amounts, counts and time intervals) | `days_since_previous_purchase`, `prior_mean_purchase_value`, `prior_mean_gap_days`, `spend_365d` | 16 / 16 / 16 / 15 | 2 each | → `NaN` |
| The placeholder `1,000,000,000` | `prior_mean_purchase_value`, `spend_365d` | 16 / 15 | 2 / 2 | → `NaN` |

The non-negativity rule is applied to all 28 amount, count and interval columns (`diagnostics.non_negative_columns`). Category labels were compared after trimming spaces and ignoring case: no variants were found, so labels are kept as they are [2.3].

#### 2.5 Recovering values from identities [5.1]

Several history columns are linked by exact definitions. A missing term (originally missing, or set to `NaN` in §2.4 or §3) is recomputed **from the same row**. Nothing is learned from other rows. Each identity was kept only after checking it holds on the rows where all its terms are observed and valid; the < 1% where one fails are the inflated values of §3.

| Column | Identity | Holds on | Values filled (train) |
|---|---|---|---|
| `prior_mean_purchase_value` | `prior_total_spend / prior_purchase_count` (count > 0) | 99.2% | 272 |
| `days_since_previous_purchase` | `customer_tenure_days` (count = 1) | 100% | 22 |
| `days_since_previous_purchase` | `customer_tenure_days − prior_mean_gap_days × (count − 1)` (count ≥ 2) | 100% | 179 |
| `prior_mean_gap_days` | `(customer_tenure_days − days_since_previous_purchase) / (count − 1)` (count ≥ 2) | 100% | 203 |
| `spend_365d` | `prior_total_spend` (tenure ≤ 365 days) | 99.2% | 82 |

In total, 758 values are recovered exactly instead of being guessed by an imputer.

#### 2.6 Structural missingness: values that cannot exist [5.3]

| Columns | Undefined when | Rows (train) |
|---|---|---|
| `days_since_previous_purchase`, `prior_mean/min/max_purchase_value` | first purchase (`prior_purchase_count = 0`) | 603 |
| `prior_std_purchase_value`, `prior_mean_gap_days` | fewer than 2 prior purchases | 1,048 |
| `prior_std_gap_days` | fewer than 3 prior purchases (a std of gaps needs two gaps) | 1,413 |

These gaps aren't MCAR, MAR or MNAR: the value is *not applicable*. A median would invent a purchase history for a first-time customer. So they are set to **0** (`diagnostics.structural_fill`), a rule from the column definitions, not a fitted value. Two flags let a model tell "0 because undefined" apart from a real 0: `is_first_purchase` (`prior_purchase_count == 0`) and `has_gap_history` (`prior_purchase_count >= 2`).

#### 2.7 Mechanism of the remaining gaps [6]

For each remaining gap we tested whether *being missing* is related to the target (χ²) or to any other feature (Mann-Whitney U / χ² with categories under 30 rows grouped), with a Bonferroni threshold.

| Column | Missing (train) | Evidence | Mechanism | Treatment (fitted on training rows) |
|---|---|---|---|---|
| `days_since_previous_purchase`, `prior_mean_gap_days` | 8 rows (0.1%) | no relation to target or features | MCAR | median |
| `payment_type`, `salesperson`, `transport_method`, `warehouse` | ~3% each, never on the same rows | target rate 0.55–0.64 vs 0.56, no significant relation | MCAR | most frequent (regression: `"Missing"`, §10) |
| `spend_365d` | 182 rows (2.8%) | MCAR in the raw data (tenure p = 0.84); after the identity fill only tenure > 365 days stays missing | MAR | median **+ missing indicator** (`MissingIndicator` in the pipeline) |
| `prior_max_purchase_value` | 45 rows (0.7%) | the inflated maxima set to `NaN` in §3; repurchase rate 0.53 vs 0.56 (p = 0.84); only associated with `transport_method` (80 levels) | gap marks a corrupted value | median, **no indicator**: a flag would teach the model to recognise the corruption |
| `commercial_region`, `island`, `municipality` (customer table, regression only) | 0.5% / 0.5% / 3.4% | what is left after the customer-table preprocessing (§13): a blank island is structural (`"none"`), and regions are recovered exactly. The rest are incomplete customer profiles, related to other geography (`municipality` also to the target) | MAR | explicit `"Missing"` category, which acts as the indicator for a categorical column (§10) |

An automatic `SimpleImputer(add_indicator=True)` would flag every column with gaps, including the MCAR ones, where a flag only adds noise. We flag only where the evidence says the gap carries information (`preprocessing.mnar_indicator_sources`).

#### 2.8 Targets [2.6, 10]

* **`repeat_purchase_90d`** is complete. It is never imputed, and the pipeline raises an error if a label is missing.
* **`spend_next_90d`** is missing for 100% of non-repeaters (structural) and for 196 of 3,621 repeaters (5.4%), and those 5.4% are **not missing at random**:
  * They are concentrated at the edges of the data: 16.8% of 2015, 11.6% of 2024 and 49% of 2025 purchases, against 1.3–4.9% in between.
  * They are related to 13 of 32 features: shorter, lower-value histories (median 8.5 vs 17 prior purchases).

  Targets are never imputed. The regressor trains only on repeat purchasers with an observed spend (`regression.row_filter`), and its preprocessing is fitted on those rows only.
* **Target transform:** fitting on `log1p(spend)` (skew 2.4 → −0.6) is decisively worse in RMSE for ridge (+818) and the chosen random forest (+377, all 5 folds). A log-scale model predicts close to the median and underestimates large spenders. So the target stays in euros (`target_transform: none`).

### 3. Inconsistencies [5.2]

17 rules (`diagnostics.consistency_rules`) tie columns of the same row together. Each is checked only where all its terms are observed:

* **Accounting identities.** `gross_total − line_discount_total − global_discount_total = net_total` holds on 100% of rows.

  `net_total + total_tax + shipping_cost = total_payable` holds on only 83.7%. This is **not a data error**: `total_payable = net_total + total_tax` holds on 100%, and the rows that "break" are exactly those with a shipping cost. Shipping isn't part of `total_payable`.
* **Logic.** Counts and spend over 30 ≤ 90 ≤ 180 ≤ 365 days ≤ all prior purchases. Other rules: `prior_min ≤ prior_max`, `days_since_previous_purchase ≤ customer_tenure_days`, at least 1 line item and 1 invoice. **0 violations** for all of these.
* **Four rules are broken:**

| Rule | Violations (train / test) | What the values look like | Decision |
|---|---|---|---|
| `prior_mean_purchase_value = prior_total_spend / prior_purchase_count` | 46 / 7 | 8–25× too large (median 16×); 45 of 46 above the customer's own maximum | → `NaN`, recomputed exactly from the identity |
| `spend_365d ≤ prior_total_spend` | 39 / 7 | 1.5–23× too large | → `NaN`, recovered where tenure ≤ 365 days, otherwise median + indicator |
| `prior_max_purchase_value ≤ prior_total_spend` | 45 / 6 | 1.1–22× too large (median 4×): one purchase can't exceed all purchases together | → `NaN`, median inside the pipeline (rule added after the outlier analysis) |
| `cost ≤ net_total` | 39 / 5 | sales below cost (median cost = 1.22 × net) | **kept**: unusual but possible, and it contradicts no other column |

The corrupted rows have the same repurchase rate as the others (0.54 / 0.56 / 0.53 vs 0.56). Fixing them costs logistic regression 0.008 accuracy, a decisive drop (all 5 folds), and makes no difference to the regression (3 ± 10 RMSE) [11.1]. **We keep the rules** anyway:
* The corrupted values carry no repurchase information.
* The loss most likely comes from the inflated values stretching the scale of three columns.
* A model that depends on impossible values would behave unpredictably on new data.

### 4. Duplicates [2.1, 2.4, 4]

| Check | Result | Treatment |
|---|---|---|
| Exact duplicate rows (all columns, and all columns except `id`) | 0 in train, 0 in test | removed only if found, training file only (`drop_training_rows`) |
| Repeated `id` | 0 | same |
| Test row identical to a training row | 0 | diagnostic only |
| Almost-empty rows (every purchase column missing except ids and target) | 65 in train | removed from training only: imputing would invent a whole purchase |
| **`customer_id` validation** (`validate_customer_ids`) | always present, a positive integer, and known in clientes.csv (train and test). Within each customer, in date order, the prior purchase count only grows (by 1, or 2 on 52 occasions), the tenure never shrinks, and no customer buys twice on the same day | report only: the repeated `customer_id`s are genuine repeat purchases, not duplicates |

Training goes from 6,534 to 6,469 rows. **test.csv keeps all 986 rows**, and the notebook and `main.py` assert it. The almost-empty check ignores the columns the pipeline creates itself (flags, date parts, ratios, customer fields), so a joined customer field can't hide an empty purchase.

### 5. Feature selection [5.1, 5.2, 11]

| Feature(s) | Decision | Evidence |
|---|---|---|
| `gross_total` | removed | exact sum: net + line + global discount (100% of rows) |
| `total_payable` | removed | exact sum: net + tax (100% of rows) |
| `random_noise` | removed | noise by name and by evidence (correlation with the target −0.012) |
| `id`, `customer_id`, `purchase_date` | never model inputs (kept in `extras`) | identifiers; needed for the temporal split and the submission |
| the other task's target | removed from `X` in both tasks | target leakage |
| date parts (`purchase_month`, `purchase_weekday`) | **regression only** | 42 RMSE better on all 5 folds; no effect on classification (+0.002) |
| customer table: `country_or_region`, `commercial_region`, `island`, `municipality`, `years_as_customer` | **regression only** | 40 RMSE better. For classification it is decisively worse (0.004 better without) |
| `profile_completeness`, `customer_since_before_records` (§13) | **regression only** | 2 RMSE better than without them (3/5 folds), with a clear reason: long-standing accounts repurchase more and spend more; incomplete profiles repurchase less |
| unused `clientes.csv` columns | not used | `customer_type` 100% missing; `company_name`, `address` act as a customer id; `street`, `city` ~50% missing with hundreds of levels; postal codes covered by the kept geography |
| ratios: `margin`, `margin_rate`, `discount_rate`, `avg_line_value`, `spend_share_30d/90d` | computed, not used | no effect for either task (±0.001 accuracy, ±1 RMSE) |
| rare-category grouping (`RareCategoryGrouper`, < 10 / < 30 training rows) | off | classification −0.007 (decisive) / −0.005; regression noise. The target encoder already shrinks rare categories |

Every feature used is known at the moment of the purchase. The customer join adds no future information: every customer is found, and no purchase predates `customer_since_year`. **Final sets:**
* **classification:** 26 numeric + 6 categorical + 2 flags + 1 missing indicator = 35 model columns;
* **regression:** 31 numeric + 10 categorical + 2 flags + 1 missing indicator.

### 6. Outlier detection (`notebooks/02_outlier_detection.ipynb`)

The outlier analysis separates **data errors** from **valid extremes**. Its principles:
* The locked period is not used to choose a method.
* Transformations and detectors are fitted on each training fold only.
* Validation rows and test.csv rows are never removed.

| Method | Judgement for this dataset |
|---|---|
| Logical consistency rules | **Used.** They find impossible values from feature definitions, including `prior_max_purchase_value > prior_total_spend` (45 train / 6 test values → `NaN` → imputed in the pipeline). |
| IQR | Diagnostic only. It fails on zero-heavy columns, and removing rows flagged in 3+ columns seriously harms the regression. |
| `log1p` of skewed columns | Plausible for monetary values. Classification: +0.007 accuracy, not decisive. Regression: 0.3 RMSE better (1.9 SE), not decisive and negligible (0.02% of 1,943). |
| Winsorisation (1%–99%) | Small changes, no reliable improvement, and it discards real magnitude information. Not used. |
| Robust scaling | Not reliably better than the task-specific scalers already chosen (§9). |
| Isolation Forest | The most reasonable optional multivariate diagnostic, but the rows it flags are often valuable high-repeat or high-spend customers, and removing them is not reliably better. |
| LOF / KNN distance | Sensitive to local density and to the distance definition; no reliable validation gain. |
| DBSCAN | One broad cluster plus a noise set that changes sharply with reasonable `eps` choices. |
| K-Means distance | Depends on an arbitrary number of clusters; not a natural outlier definition, and no reliable gain. |
| SOM distance | Useful for visual exploration, but map size and threshold are arbitrary; no reliable gain. |

**Decision:**
* Genuine data errors are corrected with domain and consistency rules (§3).
* Statistically extreme but plausible observations (large purchases, long histories) are **kept**: no row-removal method gives a reliable temporal-validation improvement.
* No `log1p`, winsorisation or extra scaler is added for outliers.
* IQR and Isolation Forest are only for manual review if new data arrives.

### 7. Validation design: temporal, with a 90-day gap [3, 7]

* test.csv lies **strictly after** train.csv in time (train: 2015-07-11 → 2025-02-15; test: 2025-05-24 → 2026-10-06).
* 607 customers make the 6,469 training purchases, and 184 of the 206 test customers already appear in train.

A random split would let the model learn from a customer's later purchases and then be scored on their earlier ones. Instead:
* **Locked test set** (`test_set`): the latest 20% of purchases (2023-06 → 2025-02). Development rows from the 90 days before it are left out (`gap_days: 90`), because their labels are measured inside the test period. It is never used to fit, tune, compare or choose anything.

  | Task | Development rows | Locked test rows |
  |---|---|---|
  | Classification | 5,024 | 1,286 |
  | Regression | 2,659 | 685 |

* **Cross-validation** (`cv`): 5 expanding-window temporal folds inside the development set, with the same 90-day gap.

Every comparison uses these folds, and two options are compared **fold by fold**. A difference is *decisive* only if it is more than 2 standard errors from zero. **Rule:** change the pipeline only for a decisive gain with a sensible reason, or for a clear reason that most folds confirm. Otherwise keep the current or simpler option.

### 8. Preprocessing pipeline [8]

`build_preprocessor()` builds one `ColumnTransformer` per task from `config.yaml`. `task_preprocessing_config()` merges the shared `preprocessing` section with `<task>.preprocessing`. It is placed inside the model's `Pipeline`:

| Step | Classification | Regression | What it does |
|---|---|---|---|
| `numeric` | 26 amount, count and history columns | the same 26 + `purchase_month`, `purchase_weekday`, `years_as_customer`, `profile_completeness`, `customer_since_before_records` | median imputer → scaler (standard / min-max) |
| `categorical` | 6 transaction categories | the same 6 + 4 customer-table categories | imputer (most frequent / `"Missing"`) → rare-category grouping (off) → target encoder |
| `indicators` | `MissingIndicator` on `spend_365d` | the same | 0/1 column for each `mnar_indicator_sources` column, computed on the **original** column before the imputer fills it |
| `flags` | `is_first_purchase`, `has_gap_history` | the same | passed through |

* **Feature lists are defined in config and checked.** `check_feature_lists()` raises an error if:
  * a feature is listed twice, or in two groups (numeric / categorical / flags);
  * an indicator source isn't a feature;
  * a listed feature doesn't exist in the data (`prepare_task_data()` checks this before anything is fitted).

  `build_preprocessor()` runs the same checks without the data.
* **Every data-dependent operation is fitted inside the pipeline.** The imputers, missing indicator, rare grouper, target encoder and scaler are fitted on the training rows of each fold. Nothing before the pipeline learns from data; sections 2–5 are rules. A test checks that the medians equal those of the training rows and not of other rows, and that the encoder only knows training categories.
* **Unseen categories don't crash anything.** Target encoding gives them the overall target mean. The output has no missing values for the development set, the locked test set and test.csv.
* **Save → load → predict new records** (`src/pipeline.py`). `main.py` saves each fitted pipeline to `models/<task>_pipeline.joblib` (16 KB for classification, ~11 MB for the random-forest regression). It then loads the file back and checks that the loaded copy predicts all 986 test.csv rows identically. `predict_new_records()` also scores a single raw record through the same stateless steps and the loaded pipeline.
* **Automated checks** (`tests/test_pipeline.py`, 10 tests, all passing):
  1. feature lists exist and don't overlap;
  2. errors are caught for overlaps, missing names and orphan indicators;
  3. indicators are created inside the pipeline, before imputation;
  4. fitted steps learn from training rows only;
  5. `prepare_table` keeps every row and gives the same result for a row alone or inside a table;
  6. schema validation flags and repairs bad labels and non-numeric values, and rejects a missing column;
  7. placeholder tokens become `NaN` and are reported;
  8. both pipelines can be fitted, saved, loaded and used to predict new raw records;
  9. Optuna tuning and nested cross-validation run on temporal folds;
  10. the classification predictions match `sample_submission.csv` (columns, ids, order, 0/1 labels).

### 9. Encoder × scaler [9]

All 16 combinations (one-hot, ordinal, count, target × none, standard, min-max, robust) were scored with a scaling-sensitive model (logistic regression / ridge).

| Task | Chosen | Validation | Reason |
|---|---|---|---|
| Classification | **target + standard** | accuracy 0.707 | Target + min-max ranks first but is tied (+0.002, SE 0.007), and so is target + robust, so the current pair stays. Ahead of one-hot (+0.006, not decisive) with one column per feature. Count encoding takes the bottom 4 places; no scaling drops target encoding to 0.694. |
| Regression | **target + min-max** | RMSE 2052 | Ranks first, and min-max is the best scaler for every encoder. Ordinal + min-max is a tie (11 RMSE, SE 13) but would impose a fake order. It beats target + robust by 172 RMSE on all 5 folds. |

### 10. Alternative imputers [12]

Each imputer is fitted inside each fold. KNN and iterative impute after scaling.

| Imputer | Classification (accuracy vs median) | Regression (RMSE vs median) |
|---|---|---|
| numeric `mean` | +0.0005 | 2.1 better |
| numeric `constant` (0) | +0.0010 (decisive, 3/5 folds) | 0.2 worse |
| numeric `knn` (5 neighbours) | +0.0005 | **6.0 better (5/5 folds, decisive)** |
| numeric `iterative` | −0.0005 | 0.8 better |
| categorical: the alternative to the current one | `"Missing"`: −0.0007 | most frequent: 2.6 worse (4/5 folds) |

* **Classification: the median stays.** `constant` is decisive but worth only 0.001 accuracy, and 0 already means "not applicable" in this pipeline (§2.6). Filling real gaps with 0 would merge two situations.
* **Regression: a pending decision.** Since the customer-table preprocessing (§13), KNN is decisively better than the median: 6 RMSE, better on all 5 folds. By the rule in §7 it is a candidate to switch to (`regression.preprocessing.imputation.numeric_strategy: knn`). It is kept for confirmation after tuning, because the gain is small next to the fold-to-fold spread (±165 RMSE) and KNN makes every fit slower.
* **Categorical:** the regression keeps an explicit `"Missing"` category: the remaining customer gaps are incomplete profiles related to the target, and filling them with the most common region or municipality would hide that. Classification keeps the most frequent value, since its categorical gaps are MCAR.

### 11. Recipe + model = one Pipeline [10]

Mean temporal-CV scores on the development set, with each task's final recipe:

| Model | Classification accuracy (train / validation) | Regression RMSE (train / validation) |
|---|---|---|
| Dummy (floor to beat) | 0.553 / 0.553 | 1940 / 2472 |
| Logistic regression / ridge | 0.712 / **0.707** | 1487 / 2052 |
| Decision tree (depth 5) | 0.739 / 0.644 | 1219 / 2522 |
| Random forest (300 trees) | 0.851 / 0.707 | 988 / **1943** |

* **Classification → logistic regression.** It beats the dummy on every fold (+0.154), with no train/validation gap. The random forest ties it (0.0002 ahead, not decisive) and overfits, so the simpler model stays.
* **Regression → random forest, target in euros.** It has the lowest RMSE (R² ≈ 0.33) and beats the dummy on every fold (−528 RMSE). Its edge over ridge is not decisive (108 ± 77, 3 of 5 folds), and it overfits, which tuning should address.

The main metric is accuracy for classification and RMSE for regression (`<task>.metric`). F1, ROC-AUC, MAE and R² are also reported.

### 12. Column-by-column summary

| Column | Problem found | Mechanism / evidence | Action | Why |
|---|---|---|---|---|
| `id` | — | unique per row | kept aside (`extras`) | identifier, needed for the submission |
| `customer_id` | 607 customers, many purchases each | valid ids, consistent histories; 184 of 206 test customers are in train | kept aside; join key for `clientes.csv` | identifier; temporal split instead of a random one |
| `purchase_date` | test.csv lies after train.csv | train ≤ 2025-02-15 < 2025-05-24 ≤ test | kept aside; temporal split; month and weekday features (regression) | validate on the future, like Kaggle |
| `total_payable` | "breaks" net + tax + shipping on 16% of rows | = `net_total + total_tax` on 100% | dropped | exact sum of two features |
| `gross_total` | — | = net + line + global discount on 100% | dropped | exact sum of three features |
| `net_total`, `total_tax`, `shipping_cost`, `global_discount_total`, `line_discount_total`, `line_item_count`, `invoice_count_current_purchase` | none | all rules pass | median → scaler | — |
| `cost` | 39 rows above `net_total` | sales below cost, plausible | kept as is | contradicts no other column |
| `customer_tenure_days`, `prior_purchase_count`, `prior_total_spend` | none | consistent with every rule and across a customer's purchases | median → scaler | — |
| `days_since_previous_purchase` | 16 negatives; undefined on first purchases | structural (603) + MCAR (8 left) | → `NaN`; recomputed (201); 0 + flag when undefined; median | exact recovery first, then a mechanism-matched fill |
| `prior_mean_purchase_value` | 16 negatives, 16 placeholders, 46 inflated 8–25× | identity with total / count | → `NaN`, recomputed exactly (272); 0 + flag on first purchases | the identity holds on 99.2% of valid rows |
| `prior_max_purchase_value` | undefined on first purchases; 45 above all prior spend | structural (603) + corrupted (45) | 0 + `is_first_purchase`; corrupted → `NaN` → median, no indicator | one purchase can't exceed the total |
| `prior_min_purchase_value` | undefined on first purchases | structural (603) | 0 + `is_first_purchase` | not applicable ≠ missing |
| `prior_std_purchase_value` | undefined with < 2 purchases | structural (1,048) | 0 + `has_gap_history` | a std needs two values |
| `prior_mean_gap_days` | 16 negatives; undefined with < 2 purchases | structural (1,048) + MCAR (8 left) | → `NaN`; recomputed (203); 0 + flag; median | as above |
| `prior_std_gap_days` | undefined with < 3 purchases | structural (1,413) | 0 + `has_gap_history` | a std of gaps needs two gaps |
| `purchase_count_30/90/180/365d`, `spend_30/90/180d` | none | nested windows consistent on 100% | median → scaler | — |
| `spend_365d` | 15 negatives, 15 placeholders, 39 above all prior spend | MAR after the identity fill | → `NaN`; recomputed when tenure ≤ 365 (82); median + `MissingIndicator` | the gap carries information about tenure |
| `payment_type`, `salesperson`, `transport_method`, `warehouse` | ~3% blank each; unseen test categories; `transport_method` 80 levels | MCAR; every label allowed | most frequent (regression: `"Missing"`) → target encoder; no rare grouping | the encoder handles unseen and rare categories |
| `commercial_zone`, `document_series` | unseen test categories (`commercial_zone`) | every label allowed | target encoder | handles unseen categories |
| `random_noise` | — | correlation with target −0.012 | dropped | pure noise |
| `repeat_purchase_90d` | — | complete | classification target, never imputed | — |
| `spend_next_90d` | missing for all non-repeaters and 5.4% of repeaters | structural + not MCAR | regression trains on observed rows only; no log transform | a target is never imputed; log1p was worse |
| `country_or_region`, `commercial_region`, `island`, `municipality` (`clientes.csv`) | labels with filler text; region 10% missing; island 51% blank; 183 municipalities | strict geographic hierarchy; island undefined outside 2 regions | labels → codes; region recovered exactly (71); island → `"none"` (535); regression only: `"Missing"` → target encoder | correct values first; lowers RMSE |
| `customer_since_year` (`clientes.csv`) | 664 of 1,143 customers dated 2015, the first year of the records | never after the purchase year; 2015 = "before the records start" | → `years_as_customer` at the purchase + `customer_since_before_records` flag (regression) | known at purchase time; the flag separates long-standing accounts |
| `profile_completeness` (derived) | — | customers with 1–2 address fields repurchase less | regression feature | data-quality signal |
| other `clientes.csv` columns | `customer_type` 100% missing; company name / address unique per customer; city, postal codes, street hundreds of levels | — | not used | no usable signal (id-like) |

### 13. Customer table (`clientes.csv`) [notebook 04]

`clientes.csv` is master data: one row per customer, no target, no purchase dates. Every step is a rule on that static table, so it runs identically for training, test.csv and new records (`src/customers.py` → `prepare_customers`, rules in `config.yaml` → `customers`). The cleaned table is joined to every purchase by `add_customer_features`.

| Step | Finding | Treatment |
|---|---|---|
| Load | blank cells are the only placeholder (3,385) | → `NaN` on load |
| Validate | all 13 columns present; ids and years whole numbers; years 2015–2026; every label follows its pattern | `validate_schema` with `customers.schema` |
| Key | `customer_id` present and unique (1,143) | `check_customer_key` stops the pipeline otherwise (a repeated key would duplicate purchases) |
| Duplicates | no exact duplicates. 2 company names twice: "Ltd 384" is two accounts at the same address, both with purchases; "Ltd 664" is a probable duplicate profile without purchases. 33 customers share 16 addresses | report only: different ids are different accounts in the purchase files, so never merged |
| Empty columns | `customer_type` 100% missing | dropped (any fully empty column is) |
| Labels → codes | every label is "<prefix> <code>: <filler>", one label per code; `#REF!` in postal codes is filler text, not an error | keep the code (`RG17`, `MU14`, …): no information lost |
| Profile completeness | customers with only 1–2 address fields repurchase less (≈ 0.40 vs ≈ 0.55) | `profile_completeness` = known address fields, counted before recovery |
| Hierarchy | strict: every island, municipality, postal area and postal code belongs to one region, every region to one country (6 postal areas span two municipalities: reported, never used) | exact recovery: **region 71 of 114 gaps** (from the postal area / code), **municipality 5** |
| Island | islands exist only in regions RG17 and RG29, and no region has a mix | a blank island elsewhere is **structural → `"none"`** (535); missing only when the region is unknown (43) |
| Rare levels | 129 of 154 used municipalities have < 5 customers | back-off to the region implemented; **not better** in CV (the target encoder already shrinks small groups) → off |
| Registration year | 664 customers dated 2015, the first year of the records, vs 13–82 in later years; many first buy years later | `customer_since_before_records` flag; `years_as_customer` computed at each purchase, checked ≥ 0 |
| Columns kept | — | country, region, island, municipality, `profile_completeness`, the flag, the year. City, postal area / code, street, address and company name are dropped (id-like) |

**Effect on the models** (temporal CV, regression):
* Codes + exact recovery + island `"none"`: 1.5 RMSE better (4/5 folds).
* Adding the two new features: another 2.3 RMSE (3/5 folds). In total 3.8 RMSE better than the previous customer table, not decisive.
* Municipality back-off: 3 RMSE worse → off.
* Classification still doesn't use customer features: even cleaned, they cost logistic regression 0.007 accuracy.

The main gain is **correctness**: values recovered exactly instead of imputed, and an undefined island no longer treated as unknown. That also changes the reading of the island gap from "MAR, related to the target" to "not an island region".

### Open items

* Compare the preprocessed models with the week-2 baseline.
* Leakage demo: show the optimistic gap of fitting the scaler/encoder on all rows instead of per fold.
* Decide on KNN imputation for the regression: decisively better since the customer-table change (6 RMSE, 5/5 folds); confirm after tuning (§10).
* Run the hyperparameter tuning for the final models (`<task>.tuning.enabled: true`). It is implemented and tested, but off by default because nested temporal CV with 30 trials takes a while.
* Once the final models are chosen: score the locked test set once (`test_set.evaluate: true`), then refit on every labelled row for the Kaggle submission (`refit_on_all_rows: true`).

---

## Best Model (development set, temporal CV)

| Task | Model | Recipe | Validation score | Dummy |
|---|---|---|---|---|
| Classification | Logistic regression | original 32 features; median / most frequent; target encoding + standard scaling; `spend_365d` missing indicator | accuracy 0.707 | 0.553 |
| Regression | Random forest (target in euros) | 32 features + date parts + cleaned customer table (geography codes, years as customer, profile completeness, "before the records" flag); median / `"Missing"`; target encoding + min-max scaling; `spend_365d` missing indicator | RMSE 1943 (R² 0.33) | 2472 |

The locked test set has not been scored yet. The fitted pipelines are in `models/` after `python main.py`.
