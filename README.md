
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
* `clientes.csv`: optional customer table, joinable on `customer_id` (not used yet)
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

---

## Preprocessing Decisions

Every decision below is backed by evidence in `preprocessing.ipynb` and stored as a rule in `config.yaml`.

### 1. Where each step runs

Each preprocessing step is one of three kinds, and the kind decides where it may run. This is what keeps the pipeline leak-free and usable on new data.

| Kind | Steps | Learns from data? | Runs on |
|---|---|---|---|
| **Stateless rule** | renames, invalid values → `NaN`, identity recomputation, structural fill, flags, dropped columns (`clean_dataset`) | No | train.csv **and** test.csv |
| **Training-data decision** | almost-empty rows, duplicates (`drop_training_rows`); regression rows (`regression.select_training_rows`) | No, but removes rows | training file only, before the split |
| **Fitted step** | imputation, encoding, scaling (`build_preprocessor`) | Yes | inside the model `Pipeline`, fitted on training rows only |

`clean_dataset()` never removes rows (same rows, same order), because every one of the 986 test.csv ids needs a prediction. It doesn't need the target either: on test.csv, `y` comes back as `None`.

### 2. Column names

Portuguese column names are converted to English `snake_case` (e.g. `Total a Pagar` → `total_payable`, `Custo` → `cost`, `N_ Detalhes` → `line_item_count`). Explicit names come from `diagnostics.column_renames`, and a generic rule covers the rest. The function raises an error if two columns end up with the same name.

### 3. Invalid values → missing

| Problem | Columns | Train | Test | Decision |
|---|---|---|---|---|
| Negative values (impossible for amounts, counts and time intervals) | `days_since_previous_purchase`, `prior_mean_purchase_value`, `prior_mean_gap_days`, `spend_365d` | 16 / 16 / 16 / 15 | 2 each | → `NaN` |
| The exact value `1,000,000,000`, found only in two historical-spend columns | `prior_mean_purchase_value`, `spend_365d` | 16 / 15 | 2 / 2 | placeholder → `NaN` |

The non-negativity rule is applied to all 28 amount, count and interval columns (`diagnostics.non_negative_columns`), so new data is checked the same way. Category labels were compared after trimming spaces and ignoring case: no variants were found, so labels are kept as they are.

### 4. Recovering values from identities

Several history columns are linked by exact definitions. A missing term (originally missing, or set to `NaN` in step 3) is recomputed **from the same row**. Nothing is learned from other rows, so this is a stateless rule and also runs on test.csv. Each identity was kept only after checking it holds on the rows where all its terms are observed and valid:

| Column | Identity | Holds on | Values filled (train) |
|---|---|---|---|
| `prior_mean_purchase_value` | `prior_total_spend / prior_purchase_count` (count > 0) | 99.2% | 226 |
| `days_since_previous_purchase` | `customer_tenure_days` (count = 1) | 100% | 22 |
| `days_since_previous_purchase` | `customer_tenure_days − prior_mean_gap_days × (count − 1)` (count ≥ 2) | 100% | 179 |
| `prior_mean_gap_days` | `(customer_tenure_days − days_since_previous_purchase) / (count − 1)` (count ≥ 2) | 100% | 203 |
| `spend_365d` | `prior_total_spend` (tenure ≤ 365 days) | 99.2% | 68 |

In total, 698 values are recovered exactly instead of being guessed by an imputer.

### 5. Structural missingness: values that cannot exist

The history columns are 100% missing exactly where the value is undefined, and almost never elsewhere:

| Columns | Undefined when | Rows (train) |
|---|---|---|
| `days_since_previous_purchase`, `prior_mean/min/max_purchase_value` | first purchase (`prior_purchase_count = 0`) | 603 |
| `prior_std_purchase_value`, `prior_mean_gap_days` | fewer than 2 prior purchases | 1,048 |
| `prior_std_gap_days` | fewer than 3 prior purchases (a std of gaps needs two gaps) | 1,413 |

These gaps aren't MCAR, MAR or MNAR: the value is *not applicable*. A median would invent a purchase history for a first-time customer. So they are set to **0** (`diagnostics.structural_fill`), which is a rule from the column definitions, not a fitted value. Two flags let a model tell "0 because undefined" apart from a real 0:

* `is_first_purchase` = `prior_purchase_count == 0`
* `has_gap_history` = `prior_purchase_count >= 2`

### 6. Columns removed or kept aside

| Column | Decision | Reason |
|---|---|---|
| `gross_total` | dropped | `gross_total = net_total + line_discount_total + global_discount_total` on 100% of rows: exact duplicate information |
| `total_payable` | dropped | correlation 0.996 with `net_total` (multicollinearity) |
| `random_noise` | dropped | noise by name and by evidence: correlation with the target −0.012 |
| `id`, `customer_id`, `purchase_date` | kept aside in `extras`, never model inputs | needed for the split, grouping and the submission file |
| `repeat_purchase_90d`, `spend_next_90d` | both removed from `X` in both tasks | the classifier must not see the spend, and the regressor must not see the repeat label (target leakage) |

### 7. Rows removed from training only

| Rule | Rows removed | Reason |
|---|---|---|
| Almost-empty rows: everything missing except `id`, `customer_id` and the target | 65 | imputing would invent a whole purchase record; test.csv has no such rows |
| Exact duplicates (ignoring `id`) | 0 | checked; none found |
| Repeated ids | 0 | checked; none found |

This runs on the labelled training file only, before the split. Training goes from 6,534 to 6,469 rows. **test.csv keeps all 986 rows.**

### 8. Targets

* **`repeat_purchase_90d`** is complete. It is never imputed, and the pipeline raises an error if a label is missing.
* **`spend_next_90d`** is missing for 100% of non-repeaters (no return → no spend) and for 5.4% of repeaters. Targets are never imputed. The regressor trains only on repeat purchasers with an observed spend (`regression.row_filter`), and its preprocessing is fitted on those rows only.

### 9. Mechanism of the remaining gaps

After steps 3–5, only real gaps are left. For each one, we tested whether *being missing* is related to the target (χ² test) or to any other feature (Mann-Whitney U for numeric features, χ² for categorical ones, with categories under 30 rows grouped so sparse cells don't create false signals). Tests used a Bonferroni-corrected threshold.

| Column | Missing (train) | Evidence | Mechanism | Treatment (fitted on training rows) |
|---|---|---|---|---|
| `days_since_previous_purchase`, `prior_mean_gap_days` | 8 rows (0.1%) | no relation to target or features | MCAR | median |
| `payment_type`, `salesperson`, `transport_method`, `warehouse` | ~3% each, never on the same rows | target rate 0.55–0.64 vs 0.56, no significant relation | MCAR | most frequent category |
| `spend_365d` | 157 rows (2.4%) | MCAR in the raw data (tenure p = 0.84), but after the identity fill only customers with > 365 days of tenure stay missing (minimum 378 days), so missingness is related to tenure and history | MAR | median **+ `spend_365d_was_missing` flag** |

An automatic `SimpleImputer(add_indicator=True)` would flag every column with gaps, including the MCAR ones, where a flag only adds noise. We flag only where the evidence says the gap carries information (`preprocessing.mnar_indicator_sources`).

### 10. Validation design: temporal, with a 90-day gap

* test.csv lies **strictly after** train.csv in time (train: 2015-07-11 → 2025-02-15; test: 2025-05-24 → 2026-10-06).
* 607 customers make the 6,469 training purchases, and 184 of the 206 test customers already appear in train.

A random split would let the model learn from a customer's later purchases and then be scored on their earlier ones, which is the opposite of the Kaggle task and gives scores that are too optimistic. Instead:

* **Locked test set** (`test_set`): the latest 20% of training purchases (2023-06 → 2025-02). Development rows from the 90 days just before it are left out (`gap_days: 90`), because their 90-day labels are measured inside the test period. The locked test set is never used to fit, tune, compare or choose anything. It will be scored once, for the final models.

  | Task | Development rows | Locked test rows |
  |---|---|---|
  | Classification | 5,024 | 1,286 |
  | Regression | 2,659 | 685 |

* **Cross-validation** (`validation`): 5 expanding-window temporal folds inside the development set. Each fold trains on the past and validates on the next block, with the same 90-day gap.

### 11. The recipe: a leak-safe `ColumnTransformer`

`build_preprocessor()` builds one `ColumnTransformer` from `config.yaml`, placed inside the model's `Pipeline`, so in every fold it is fitted on the training rows only:

| Group | Columns | Steps |
|---|---|---|
| numeric | 26 amount, count and history columns | median imputer → scaler |
| categorical | `payment_type`, `salesperson`, `commercial_zone`, `transport_method`, `warehouse`, `document_series` | most-frequent imputer → encoder |
| flags | `is_first_purchase`, `has_gap_history`, `spend_365d_was_missing` | passed through |

The output has 35 model-ready columns with no missing values, for the development set, the locked test set and test.csv. test.csv contains categories never seen in training (e.g. new payment types, commercial zones, transport methods and a new warehouse). Every encoder handles them without crashing: target encoding gives them the overall target mean.

### 12. Encoder × scaler: chosen by comparison

All 16 combinations (one-hot, ordinal, count, target × none, standard, min-max, robust) were scored on the same 5 temporal folds with a scaling-sensitive model (logistic regression / ridge; trees ignore scaling). The top pairs were then compared **fold by fold**: a difference counts as real only if it is more than 2 standard errors from zero.

| Task | Chosen | Validation | Reason |
|---|---|---|---|
| Classification | **target + standard** | accuracy 0.715 | Ranked first. Tied with target + robust (+0.0005, SE 0.0006). Decisively better than one-hot + robust (+0.008, better on 5/5 folds). Count encoding takes the bottom 4 places. |
| Regression | **target + min-max** | RMSE 2104 | Min-max is best with every encoder (ridge penalises coefficients by scale). Ordinal + min-max was 3 RMSE better, which is noise (SE 19). The tie goes to target encoding, because ordinal encoding imposes a fake order on unordered categories. |

Each task stores its pair in `config.yaml` → `<task>.preprocessing`.

### 13. Recipe + model = one Pipeline

Preprocessing and model are a single sklearn `Pipeline`, so `fit` learns everything (medians, category means, scales, model) from the training rows of each fold. Mean temporal-CV scores on the development set:

| Model | Classification accuracy (train / validation) | Regression RMSE (train / validation) |
|---|---|---|
| Dummy (floor to beat) | 0.553 / 0.553 | 1940 / 2472 |
| Logistic regression / ridge | 0.711 / **0.715** | 1541 / 2104 |
| Decision tree (depth 5) | 0.740 / 0.649 | 1265 / 2177 |
| Random forest (300 trees) | 0.849 / 0.709 | 1008 / **2008** |

* **Classification → logistic regression.** It has the best validation accuracy, it beats the dummy on every fold (+0.16), and train ≈ validation, so it doesn't overfit. The random forest is 0.006 behind (not decisive) and overfits.
* **Regression → random forest.** It has the lowest validation RMSE (R² ≈ 0.30) and beats the dummy on every fold (−464 RMSE). Its edge over ridge is not decisive (96 ± 64, better on 3 of 5 folds), and it overfits clearly. Hyperparameter tuning should address that.

The main metric is accuracy for classification and RMSE for regression (`<task>.metric`). F1, ROC-AUC, MAE and R² are also reported.

### Open items

Still to do in preprocessing (details in `PREPROCESSING_TODO.md`):

* Fix the old notebook sections 1–4: duplicate removal is also applied to test.csv there, and some functions are duplicated from `src/`.
* Investigate `net_total + total_tax + shipping_cost = total_payable`, which holds for only 83.7% of rows.
* Group the rare levels of `transport_method` (80 levels, 40 with fewer than 10 rows).
* Decide on skew and outliers column by column, and rewrite `2.Outlier_detection.ipynb` for this dataset (it still refers to columns from another project).
* Use a `log1p` target transform for the regression (target skew 3.0).
* Try merging `clientes.csv`.
* Compare alternative imputers (KNN, iterative) inside cross-validation.

---

## Best Model (development set, temporal CV)

| Task | Model | Recipe | Validation score | Dummy |
|---|---|---|---|---|
| Classification | Logistic regression | target encoding + standard scaling | accuracy 0.715 | 0.553 |
| Regression | Random forest | target encoding + min-max scaling | RMSE 2008 (R² 0.30) | 2472 |

The locked test set has not been scored yet.
