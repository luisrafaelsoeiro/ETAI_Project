
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
pip install -r requirements.txt
python main.py                          # both tasks
python main.py --task classification    # or: --task regression
```

For each task, `main.py` cleans the data, sets the locked test set aside, cross-validates the configured model against a dummy baseline on the development set, and fits the final pipeline on the development rows. Every choice is read from `config.yaml`.

---

## Project Structure

```text
ETAI_Project/
│
├── config.yaml                  # every rule and choice (no fitted values)
├── main.py                      # terminal entry point
├── preprocessing.ipynb          # diagnosis + evidence for every preprocessing decision
├── 2.Outlier_detection.ipynb    # outlier exploration (to be rewritten, see Open Items)
├── PREPROCESSING_TODO.md        # preprocessing checklist
├── README.md
├── requirements.txt
│
├── src/                         # shared by both tasks
│   ├── preprocessing.py         # cleaning rules, row decisions, split, ColumnTransformer
│   └── evaluate.py              # temporal folds, cross-validation, comparisons
│
├── classification/
│   ├── preprocessing.py         # training rows + (X, y, extras) for repeat_purchase_90d
│   ├── model.py                 # classification model factory
│   ├── evaluate.py, results.py, tuning.py   # (not used yet)
│
├── regression/
│   ├── preprocessing.py         # training rows + (X, y, extras) for spend_next_90d
│   ├── model.py                 # regression model factory
│   ├── evaluate.py, results.py, tuning.py   # (not used yet)
│
├── Self_utils/                  # plotting/outlier helpers used by 2.Outlier_detection.ipynb
│
└── data/
    └── raw/                     # train.csv, test.csv, clientes.csv, sample_submission.csv
```

### Main files

**`main.py`**
Runs the pipeline from the terminal (see *How to Run*).

**`config.yaml`**
Stores every rule and choice: column renames, validity rules, imputation, encoder, scaler, split and models. It stores rules only, never fitted values. Medians, category means and scaling statistics are learned again from the training rows of every split.

**`src/preprocessing.py`, `src/evaluate.py`**
Code shared by both tasks. `preprocessing.ipynb` imports these functions instead of redefining them, so the notebook and `main.py` always run the same code.

**`classification/`, `regression/`**
Task-specific code: which rows each task trains on, the target, and the model factory.

### Data

**`data/raw/`** contains the original project datasets:

* `train.csv`: 6,534 purchases with both targets
* `test.csv`: 986 purchases without targets (Kaggle)
* `clientes.csv`: optional customer table, joined on `customer_id` (used by the regression, see *Preprocessing Decisions* §14)
* `sample_submission.csv`: Kaggle submission format (`ID`, `repeat_purchase_90d`)

---

## Data

Each row in the dataset represents a **customer purchase occasion**.

The available variables include information about:

* Current transaction values
* Purchase details
* Categorical transaction information
* Historical purchase timing
* Historical customer value
* Rolling purchase frequency
* Rolling customer spend
* Additional numerical variables

The project also requires that all features used for prediction are available at the time of the purchase, avoiding information from the future or from other test observations.

---

## Pipeline Progress

| Stage | What changed | Where |
|---|---|---|
| Project setup | Repository structure, raw data, requirements | — |
| Outlier exploration | Univariate (IQR) and multivariate (DBSCAN, K-Means, SOM) outlier analysis | `2.Outlier_detection.ipynb` |
| Data diagnosis | Column standardisation, invalid values, placeholder, almost-empty rows, category text, target missingness, date checks, duplicates | `preprocessing.ipynb` sections 1–4 |
| Preprocessing → pipeline | Diagnosis turned into config-driven rules; identity recovery; structural missingness; missingness-mechanism tests; temporal locked test set; leak-safe `ColumnTransformer` | `src/preprocessing.py`, `config.yaml`, notebook sections 5–8 |
| Encoder × scaler + models | Temporal cross-validation, 16-recipe comparison per task, paired comparisons, Pipeline with dummy / linear / tree / random forest | `src/evaluate.py`, `*/model.py`, notebook sections 9–10 |
| Regression target + notebook clean-up | Missing-spend mechanism tested (not MCAR), `log1p` target tested and rejected. Notebook sections 1–4 now call the `src/` functions and read every list and value from `config.yaml`; duplicate check is diagnostic only (test.csv never filtered); table of contents with unique anchors; the per-feature plots replaced by a test table + verdict; executed top to bottom | `regression/model.py`, `preprocessing.ipynb` |
| Data quality + additions + imputers | Accounting identity explained (shipping is not in `total_payable`); 16 consistency rules, 2 of which fix inflated values; purchase-time features, customer table and rare-category grouping tested by ablation; 5 numeric and 2 categorical imputers compared in CV; a conclusion at the end of every notebook section | `src/preprocessing.py`, `src/evaluate.py`, `config.yaml`, notebook sections 5.2, 11, 12 |

---

## Preprocessing Decisions

Every decision below is backed by evidence in `preprocessing.ipynb` (the section numbers in brackets) and stored as a rule in `config.yaml`. A column-by-column summary is at the end (§17).

### 1. Where each step runs

Each preprocessing step is one of three kinds, and the kind decides where it may run. This is what keeps the pipeline leak-free and usable on new data.

| Kind | Steps | Learns from data? | Runs on |
|---|---|---|---|
| **Stateless rule** | renames, invalid values → `NaN`, consistency rules, identity recomputation, structural fill, flags, date parts, ratios, dropped columns (`clean_dataset`); customer-table join (`add_customer_features`). Both run through `prepare_table` | No | train.csv **and** test.csv |
| **Training-data decision** | almost-empty rows, duplicates (`drop_training_rows`); regression rows (`regression.select_training_rows`) | No, but removes rows | training file only, before the split |
| **Fitted step** | imputation, rare-category grouping, encoding, scaling (`build_preprocessor`) | Yes | inside the model `Pipeline`, fitted on training rows only |

`prepare_table()` never removes rows (same rows, same order), because every one of the 986 test.csv ids needs a prediction. It doesn't need the target either: on test.csv, `y` comes back as `None`.

### 2. Column names [1.1]

Portuguese column names are converted to English `snake_case` (e.g. `Total a Pagar` → `total_payable`, `Custo` → `cost`, `N_ Detalhes` → `line_item_count`). Explicit names come from `diagnostics.column_renames`, and a generic rule covers the rest. The function raises an error if two columns end up with the same name.

### 3. Invalid values → missing [2.2]

| Problem | Columns | Train | Test | Decision |
|---|---|---|---|---|
| Negative values (impossible for amounts, counts and time intervals) | `days_since_previous_purchase`, `prior_mean_purchase_value`, `prior_mean_gap_days`, `spend_365d` | 16 / 16 / 16 / 15 | 2 each | → `NaN` |
| The exact value `1,000,000,000`, found only in two historical-spend columns | `prior_mean_purchase_value`, `spend_365d` | 16 / 15 | 2 / 2 | placeholder → `NaN` |

The non-negativity rule is applied to all 28 amount, count and interval columns (`diagnostics.non_negative_columns`), so new data is checked the same way. Category labels were compared after trimming spaces and ignoring case: no variants were found, so labels are kept as they are [2.3].

### 4. Consistency between columns [5.2]

16 rules (`diagnostics.consistency_rules`) tie columns of the same row together. Each is checked only where all its terms are observed:

* **Accounting identities.** `gross_total − line_discount_total − global_discount_total = net_total` holds on 100% of rows.

  `net_total + total_tax + shipping_cost = total_payable` holds on only 83.7%. This is **not a data error**: `total_payable = net_total + total_tax` holds on 100% of rows, and the rows that "break" are exactly the ones with a shipping cost. Shipping is simply not included in `total_payable`.
* **Logic.** The time windows nest: counts and spend over 30 ≤ 90 ≤ 180 ≤ 365 days ≤ all prior purchases. Other rules: `prior_min ≤ prior_max`, `days_since_previous_purchase ≤ customer_tenure_days`, at least 1 line item and 1 invoice. **0 violations** for all of these.
* **Three rules are broken:**

| Rule | Violations (train / test) | What the values look like | Decision |
|---|---|---|---|
| `prior_mean_purchase_value = prior_total_spend / prior_purchase_count` | 46 / 7 | 8–25× too large (median 16×); 45 of the 46 are above the customer's own maximum, which is impossible | → `NaN`, then recomputed exactly from the identity (§5) |
| `spend_365d ≤ prior_total_spend` | 39 / 7 | 1.5–23× too large: the last 365 days can't exceed all prior spend | → `NaN`, then recovered by the identity where tenure ≤ 365 days, otherwise imputed (§10) |
| `cost ≤ net_total` | 39 / 5 | sales below cost (median cost = 1.22 × net) | **kept**: unusual but possible (promotions, clearance), and it contradicts no other column |

The corrupted rows have the same repurchase rate as the other rows (0.54 / 0.56 vs 0.56), so fixing them removes no real information. The effect on the models is discussed in §14.

### 5. Recovering values from identities [5.1]

Several history columns are linked by exact definitions. A missing term (originally missing, or set to `NaN` in §3–4) is recomputed **from the same row**. Nothing is learned from other rows, so this is a stateless rule and also runs on test.csv. Each identity was kept only after checking it holds on the rows where all its terms are observed and valid; the < 1% where one fails are the inflated values of §4.

| Column | Identity | Holds on | Values filled (train) |
|---|---|---|---|
| `prior_mean_purchase_value` | `prior_total_spend / prior_purchase_count` (count > 0) | 99.2% | 272 |
| `days_since_previous_purchase` | `customer_tenure_days` (count = 1) | 100% | 22 |
| `days_since_previous_purchase` | `customer_tenure_days − prior_mean_gap_days × (count − 1)` (count ≥ 2) | 100% | 179 |
| `prior_mean_gap_days` | `(customer_tenure_days − days_since_previous_purchase) / (count − 1)` (count ≥ 2) | 100% | 203 |
| `spend_365d` | `prior_total_spend` (tenure ≤ 365 days) | 99.2% | 82 |

In total, 758 values are recovered exactly instead of being guessed by an imputer.

### 6. Structural missingness: values that cannot exist [5.3]

The history columns are 100% missing exactly where the value is undefined, and almost never elsewhere:

| Columns | Undefined when | Rows (train) |
|---|---|---|
| `days_since_previous_purchase`, `prior_mean/min/max_purchase_value` | first purchase (`prior_purchase_count = 0`) | 603 |
| `prior_std_purchase_value`, `prior_mean_gap_days` | fewer than 2 prior purchases | 1,048 |
| `prior_std_gap_days` | fewer than 3 prior purchases (a std of gaps needs two gaps) | 1,413 |

These gaps aren't MCAR, MAR or MNAR: the value is *not applicable*. A median would invent a purchase history for a first-time customer. So they are set to **0** (`diagnostics.structural_fill`), which is a rule from the column definitions, not a fitted value. Two flags let a model tell "0 because undefined" apart from a real 0:

* `is_first_purchase` = `prior_purchase_count == 0`
* `has_gap_history` = `prior_purchase_count >= 2`

### 7. Columns removed or kept aside [5.1, 5.2]

| Column | Decision | Reason |
|---|---|---|
| `gross_total` | dropped | `gross_total = net_total + line_discount_total + global_discount_total` on 100% of rows: exact duplicate information |
| `total_payable` | dropped | `total_payable = net_total + total_tax` on 100% of rows (correlation 0.996 with `net_total`) |
| `random_noise` | dropped | noise by name and by evidence: correlation with the target −0.012 |
| `id`, `customer_id`, `purchase_date` | kept aside in `extras`, never model inputs | needed for the split, grouping and the submission file |
| `repeat_purchase_90d`, `spend_next_90d` | both removed from `X` in both tasks | the classifier must not see the spend, and the regressor must not see the repeat label (target leakage) |

### 8. Rows removed from training only [2.1, 2.4, 4]

| Rule | Rows removed | Reason |
|---|---|---|
| Almost-empty rows: every purchase column missing except `id`, `customer_id` and the target | 65 | imputing would invent a whole purchase record; test.csv has no such rows |
| Exact duplicates (ignoring `id`) | 0 | checked; none found |
| Repeated ids | 0 | checked; none found |

This runs on the labelled training file only, before the split. Training goes from 6,534 to 6,469 rows. **test.csv keeps all 986 rows**, and the notebook asserts it. The almost-empty check ignores the columns the pipeline creates itself (flags, date parts, ratios, customer fields), so a joined customer field can't hide an empty purchase. The duplicate check in section 4 is diagnostic only. It looks at all columns and at all columns except `id` (a unique `id` would otherwise hide every duplicate), and also checks whether any test row is identical to a training row: none is.

### 9. Targets [2.5, 10]

* **`repeat_purchase_90d`** is complete. It is never imputed, and the pipeline raises an error if a label is missing.
* **`spend_next_90d`** is missing for 100% of non-repeaters (no return → no spend, structural) and for 196 of 3,621 repeaters (5.4%). Those 5.4% are **not missing at random**:
  * They are concentrated at the edges of the data. The spend is missing for 16.8% of 2015 purchases, 11.6% of 2024 purchases and 49% of 2025 purchases, against 1.3–4.9% in the years in between. This looks like the 90-day spend not being fully recorded at the edges of the extract.
  * 13 of 32 features are significantly related to it. The rows come from customers with a shorter, lower-value history (median 8.5 vs 17 prior purchases, 365-day spend ≈ 1,973 € vs 4,644 €).

  Targets are never imputed: that would mean training on our own guesses. The regressor trains only on repeat purchasers with an observed spend (`regression.row_filter`), and its preprocessing is fitted on those rows only. As a consequence, the regression training rows slightly over-represent longer, higher-value histories, and the most recent months have fewer usable regression rows.
* **Target transform:** the spend is right-skewed (skew 2.4 on the development rows; −0.6 after `log1p`). Fitting on `log1p(spend)` (`TransformedTargetRegressor`, `regression.model.target_transform`) was tested on the same folds. In RMSE it is **decisively worse** for ridge (+798) and for the chosen random forest (+374, worse on all 5 folds); for a single depth-5 tree it makes no real difference. Once transformed back, a log-scale model predicts something close to the median and underestimates the large spenders that dominate RMSE. So the target stays in euros (`target_transform: none`). The option is kept for a log-scale metric such as RMSLE.

### 10. Mechanism of the remaining gaps [6]

After §3–6, only real gaps are left. For each one, we tested whether *being missing* is related to the target (χ² test) or to any other feature (Mann-Whitney U for numeric features, χ² for categorical ones, with categories under 30 rows grouped so sparse cells don't create false signals). Tests used a Bonferroni-corrected threshold.

| Column | Missing (train) | Evidence | Mechanism | Treatment (fitted on training rows) |
|---|---|---|---|---|
| `days_since_previous_purchase`, `prior_mean_gap_days` | 8 rows (0.1%) | no relation to target or features | MCAR | median |
| `payment_type`, `salesperson`, `transport_method`, `warehouse` | ~3% each, never on the same rows | target rate 0.55–0.64 vs 0.56, no significant relation | MCAR | most frequent category (regression: `"Missing"`, §15) |
| `spend_365d` | 182 rows (2.8%) | MCAR in the raw data (tenure p = 0.84), but after the identity fill only customers with > 365 days of tenure stay missing (minimum 370 days), so missingness is related to tenure and history | MAR | median **+ `spend_365d_was_missing` flag** |
| `island`, `municipality`, `commercial_region` (customer table, regression only) | 33.8% / 3.4% / 0.5% | related to other geography and transaction features; for `island` and `municipality` also to the target | MAR | explicit `"Missing"` category, which acts as the flag for a categorical column (§15) |

An automatic `SimpleImputer(add_indicator=True)` would flag every column with gaps, including the MCAR ones, where a flag only adds noise. We flag only where the evidence says the gap carries information (`preprocessing.mnar_indicator_sources`).

### 11. Validation design: temporal, with a 90-day gap [3, 7]

* test.csv lies **strictly after** train.csv in time (train: 2015-07-11 → 2025-02-15; test: 2025-05-24 → 2026-10-06).
* 607 customers make the 6,469 training purchases, and 184 of the 206 test customers already appear in train.

A random split would let the model learn from a customer's later purchases and then be scored on their earlier ones, which is the opposite of the Kaggle task and gives scores that are too optimistic. Instead:

* **Locked test set** (`test_set`): the latest 20% of training purchases (2023-06 → 2025-02). Development rows from the 90 days just before it are left out (`gap_days: 90`), because their 90-day labels are measured inside the test period. The locked test set is never used to fit, tune, compare or choose anything. It will be scored once, for the final models.

  | Task | Development rows | Locked test rows |
  |---|---|---|
  | Classification | 5,024 | 1,286 |
  | Regression | 2,659 | 685 |

* **Cross-validation** (`validation`): 5 expanding-window temporal folds inside the development set. Each fold trains on the past and validates on the next block, with the same 90-day gap.

Every comparison below uses these same 5 folds, and two options are compared **fold by fold**. A difference counts as *decisive* only if it is more than 2 standard errors from zero. **Rule:** a change is adopted when it is decisive, or when it has a clear reason and points the same way on most folds. Otherwise the simpler or current option stays.

### 12. The recipe: a leak-safe `ColumnTransformer` [8]

`build_preprocessor()` builds one `ColumnTransformer` from `config.yaml` (`task_preprocessing_config()` merges the shared `preprocessing` section with `<task>.preprocessing`). It is placed inside the model's `Pipeline`, so in every fold it is fitted on the training rows only:

| Group | Classification | Regression | Steps |
|---|---|---|---|
| numeric | 26 amount, count and history columns | the same 26 + `purchase_month`, `purchase_weekday`, `years_as_customer` | median imputer → scaler |
| categorical | `payment_type`, `salesperson`, `commercial_zone`, `transport_method`, `warehouse`, `document_series` | the same 6 + `country_or_region`, `commercial_region`, `island`, `municipality` | imputer (most frequent / `"Missing"`) → rare-category grouping (off) → target encoder |
| flags | `is_first_purchase`, `has_gap_history`, `spend_365d_was_missing` | the same | passed through |

The classification matrix has 35 model-ready columns with no missing values, for the development set, the locked test set and test.csv. test.csv contains categories never seen in training (e.g. new payment types, commercial zones, transport methods and a new warehouse). Every encoder handles them without crashing: target encoding gives them the overall target mean.

### 13. Encoder × scaler: chosen by comparison [9]

All 16 combinations (one-hot, ordinal, count, target × none, standard, min-max, robust) were scored with a scaling-sensitive model (logistic regression / ridge; trees ignore scaling).

| Task | Chosen | Validation | Reason |
|---|---|---|---|
| Classification | **target + standard** | accuracy 0.709 | Tied with target + min-max (+0.0014, SE 0.0069) and target + robust, so the current pair stays. Decisively better than one-hot + robust (+0.008). Count encoding takes the bottom 4 places, and no scaling drops target encoding to 0.698. |
| Regression | **target + min-max** | RMSE 2036 | Ranks first, and min-max is the best scaler for every encoder (ridge penalises coefficients by scale). Ordinal + min-max is a tie (5 RMSE, SE 10), but ordinal would impose a fake order on unordered categories. It beats target + robust by 142 RMSE on all 5 folds. |

### 14. Data additions: tested by ablation [11]

Each addition was scored **with and without**, using the task's chosen recipe and model:

| Addition | What it is | Classification (accuracy) | Regression (RMSE) | Decision |
|---|---|---|---|---|
| Consistency rules (§4) | fix 85 impossible values | −0.0055 (decisive) | 2 ± 10 (noise) | **kept** for both |
| Date parts | `purchase_month`, `purchase_weekday` | −0.001 without them | **41 better, 5/5 folds** | regression |
| Customer table | `country_or_region`, `commercial_region`, `island`, `municipality`, `years_as_customer` | 0.003 better *without* | **42 better** | regression |
| Ratios | `margin`, `margin_rate`, `discount_rate`, `avg_line_value`, `spend_share_30d/90d` | no effect | no effect (±1) | not used |
| Rare-category grouping | categories under 10 / 30 training rows → `rare` (`RareCategoryGrouper`, fitted per fold) | −0.006 / −0.005 | +2 / −3 (noise) | **off** |

* **Consistency rules are kept even though classification loses 0.0055.** The corrupted rows carry no repurchase information (same repurchase rate). The loss most likely comes from the inflated values stretching the scale of two columns, which logistic regression happens to score slightly better on. A model that depends on impossible values would behave unpredictably on new data.
* **Purchase-time features** use only what is known at the moment of the purchase (the date itself and ratios of the same row).
* **Customer table:** every customer is found in `clientes.csv`, and no purchase is dated before `customer_since_year`, so the join adds no future information.
  * **Dropped:**
    * `customer_type`: 100% missing.
    * `company_name`, `address`: unique per customer, so they would act as a customer id.
    * `street`, `city`: about half missing, with hundreds of levels.
    * `postal_code`, `postal_area`: hundreds of levels, and the kept geography already covers them.
* **Classification keeps its original 32 features.** None of the additions helps it, and together they are slightly worse (−0.005).
* **Regression gains 48 RMSE** from the date parts and customer table together (better on 4 of 5 folds). Spend seems to depend on season and on where the customer is, while repurchase doesn't.
* **Rare categories are not grouped.** The target encoder already shrinks a rare category's mean towards the overall mean, so grouping only throws information away (`rare_category_min_count: 0`).

### 15. Alternative imputers [12]

Each imputer is fitted inside each fold on training rows only. KNN and iterative impute after scaling, so large-valued columns don't dominate.

| Imputer | Classification (accuracy vs median) | Regression (RMSE vs median) |
|---|---|---|
| numeric `mean` | −0.0010 (decisive) | 2.5 better |
| numeric `constant` (0) | −0.0002 | 0.8 better |
| numeric `knn` (5 neighbours) | −0.0012 | 7.1 better (3/5 folds, not decisive) |
| numeric `iterative` | −0.0010 (decisive) | 1.3 better |
| categorical `"Missing"` vs most frequent | −0.0007 | **2.6 better, 5/5 folds (decisive)** |

* **Numeric: the median stays for both tasks.** With 0.1–3% gaps per column, the imputer barely matters: every difference is far below the fold-to-fold spread (±160 RMSE). The median also matches the MCAR verdict of §10.
* **Categorical: regression uses an explicit `"Missing"` category.** The customer columns have real gaps (`island` 34% missing) that are related to the target, and filling them with the most common island would hide that. Classification keeps the most frequent value: its categorical gaps are MCAR, and `"Missing"` doesn't help it.

### 16. Recipe + model = one Pipeline [10]

Preprocessing and model are a single sklearn `Pipeline`, so `fit` learns everything (medians, category means, scales, model) from the training rows of each fold. Mean temporal-CV scores on the development set, with each task's final recipe:

| Model | Classification accuracy (train / validation) | Regression RMSE (train / validation) |
|---|---|---|
| Dummy (floor to beat) | 0.553 / 0.553 | 1940 / 2472 |
| Logistic regression / ridge | 0.712 / **0.709** | 1494 / 2036 |
| Decision tree (depth 5) | 0.739 / 0.644 | 1248 / 2521 |
| Random forest (300 trees) | 0.851 / 0.707 | 989 / **1946** |

* **Classification → logistic regression.** It has the best validation accuracy, it beats the dummy on every fold (+0.156), and train ≈ validation, so it doesn't overfit. The random forest is 0.002 behind (not decisive) and overfits.
* **Regression → random forest, target in euros.** It has the lowest validation RMSE (R² ≈ 0.33) and beats the dummy on every fold (−526 RMSE). Its edge over ridge is not decisive (90 ± 65, better on 3 of 5 folds), and it overfits clearly. Hyperparameter tuning should address that. A single depth-5 tree does worse than the dummy (R² −0.16).

The main metric is accuracy for classification and RMSE for regression (`<task>.metric`). F1, ROC-AUC, MAE and R² are also reported.

### 17. Column-by-column summary

| Column | Problem found | Mechanism / evidence | Action | Why |
|---|---|---|---|---|
| `id` | — | unique per row | kept aside (`extras`) | identifier, needed for the submission |
| `customer_id` | 607 customers, many purchases each | 184 of 206 test customers are in train | kept aside; join key for `clientes.csv` | identifier; temporal split instead of a random one |
| `purchase_date` | test.csv lies after train.csv | train ≤ 2025-02-15 < 2025-05-24 ≤ test | kept aside; temporal split; month and weekday features (regression) | validate on the future, like Kaggle |
| `total_payable` | "breaks" net + tax + shipping on 16% of rows | = `net_total + total_tax` on 100% (shipping not included) | dropped | exact sum of two features |
| `gross_total` | — | = net + line + global discount on 100% | dropped | exact sum of three features |
| `net_total`, `total_tax`, `shipping_cost`, `global_discount_total`, `line_discount_total`, `line_item_count`, `invoice_count_current_purchase` | none (non-negative, no gaps after cleaning) | all consistency rules pass | median → scaler (no imputation needed) | — |
| `cost` | 39 rows above `net_total` | sales below cost, plausible | kept as is | contradicts no other column |
| `customer_tenure_days`, `prior_purchase_count`, `prior_total_spend` | none | consistent with every rule | median → scaler | — |
| `days_since_previous_purchase` | 16 negatives; undefined on first purchases | structural (603) + MCAR (8 left) | → `NaN`; recomputed from tenure / mean gap (201); 0 + flag when undefined; median | exact recovery first, then a mechanism-matched fill |
| `prior_mean_purchase_value` | 16 negatives, 16 placeholders, 46 inflated 8–25× | identity with total / count | → `NaN`, recomputed exactly (272); 0 + flag on first purchases | the identity holds on 99.2% of valid rows |
| `prior_min/max_purchase_value` | undefined on first purchases | structural (603) | 0 + `is_first_purchase` | not applicable ≠ missing |
| `prior_std_purchase_value` | undefined with < 2 purchases | structural (1,048) | 0 + `has_gap_history` | a std needs two values |
| `prior_mean_gap_days` | 16 negatives; undefined with < 2 purchases | structural (1,048) + MCAR (8 left) | → `NaN`; recomputed (203); 0 + flag; median | as above |
| `prior_std_gap_days` | undefined with < 3 purchases | structural (1,413) | 0 + `has_gap_history` | a std of gaps needs two gaps |
| `purchase_count_30/90/180/365d`, `spend_30/90/180d` | none | nested windows consistent on 100% | median → scaler | — |
| `spend_365d` | 15 negatives, 15 placeholders, 39 above all prior spend, ~3% missing | MAR after the identity fill (only tenure > 365 days stays missing) | → `NaN`; recomputed when tenure ≤ 365 (82); median + `spend_365d_was_missing` | the gap carries information about tenure |
| `payment_type`, `salesperson`, `transport_method`, `warehouse` | ~3% missing each; unseen test categories; `transport_method` 80 levels | MCAR | most frequent (regression: `"Missing"`) → target encoder; no rare grouping | the encoder handles unseen and rare categories |
| `commercial_zone`, `document_series` | unseen test categories (`commercial_zone`) | — | target encoder | handles unseen categories |
| `random_noise` | — | correlation with target −0.012 | dropped | pure noise |
| `repeat_purchase_90d` | — | complete | classification target, never imputed | — |
| `spend_next_90d` | missing for all non-repeaters and 5.4% of repeaters | structural + not MCAR (edges of the data, lower-value customers) | regression trains on observed rows only; no log transform | a target is never imputed; log1p was worse |
| `country_or_region`, `commercial_region`, `island`, `municipality` (`clientes.csv`) | up to 34% missing (`island`) | MAR, related to the target | regression only: `"Missing"` category → target encoder | lowers RMSE; the gap is informative |
| `customer_since_year` (`clientes.csv`) | — | never after the purchase year | → `years_as_customer` (regression) | known at purchase time |
| other `clientes.csv` columns | 100% missing / id-like / hundreds of levels | — | not used | no usable signal |

### Open items

Still to do in preprocessing (details in `PREPROCESSING_TODO.md`):

* Outliers and skew: decide column by column, and rewrite `2.Outlier_detection.ipynb` for this dataset (it still refers to columns from another project).
* Compare the preprocessed models with the week-2 baseline.
* Leakage demo: show the optimistic gap of fitting the scaler/encoder on all rows instead of per fold.
* Automated sanity tests for `prepare_table` (row-preserving, works without targets, no `NaN` after `transform`).

---

## Best Model (development set, temporal CV)

| Task | Model | Recipe | Validation score | Dummy |
|---|---|---|---|---|
| Classification | Logistic regression | original 32 features; median / most frequent; target encoding + standard scaling | accuracy 0.709 | 0.553 |
| Regression | Random forest (target in euros) | 32 features + date parts + customer table; median / `"Missing"`; target encoding + min-max scaling | RMSE 1946 (R² 0.33) | 2472 |

The locked test set has not been scored yet.
