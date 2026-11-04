# Preprocessing — To-Do List

Gap analysis of `preprocessing.ipynb` (+ `2.Outlier_detection.ipynb`, `Self_utils/`, `classification/`, `regression/`, `config.yaml`, `main.py`) against the course reference (`ETAI-Pipeline/notebooks/02_preprocessing.ipynb`, `02.1_preprocessing.ipynb`, `src/preprocessing.py`).

**Current state, in short:** the notebook does a solid *diagnosis* (column standardisation, invalid values, placeholder `1,000,000,000`, almost-empty rows, date checks, duplicate checks). What the course grades next is missing: a **leak-safe recipe** (imputation + encoding + scaling fitted inside a `Pipeline`), a **config-driven `src` module**, and evidence-based decisions written down. `config.yaml`, `main.py`, `classification/preprocessing.py` and `regression/preprocessing.py` are all **empty (0 bytes)**.

Facts below marked 🔎 were checked on the raw data while writing this list.

---

## P0 — Required by the course (missing entirely)

### 1. Move the notebook code into the pipeline ✅
- [x] `src/preprocessing.py` (shared): `load_data`, `standardize_columns`, `flag_invalid_values`, `apply_recompute_rules`, `apply_structural_fill`, `add_flags`, `clean_dataset` (row-preserving, returns a change report), `drop_training_rows` (train only), `add_missingness_indicators`, `split_features_target` (`y = None` on test.csv), `split_dev_test`, `build_preprocessor`.
- [x] `classification/preprocessing.py` / `regression/preprocessing.py`: `select_training_rows` (regression keeps repeaters with an observed spend) + `prepare_features`.
- [x] `main.py`: `python main.py [--task classification|regression|both]` runs it end to end and checks that no NaN is left and test.csv keeps its 986 rows. The model step is still missing (item 6).
- [x] The notebook imports these functions (sections 5–8) instead of redefining them, so the notebook and pipeline run the same code.
- [ ] `Self_utils/py_functions_preproc.py` (code from another project) was **not** deleted: `2.Outlier_detection.ipynb` still imports it as `pf`. Handle it together with the outlier notebook rewrite (P1).
- [ ] Old notebook functions in sections 1–4 (`standardize_columns`, `replace_invalid_values`, `validate_and_handle_duplicates`) still duplicate the module → trim them with item 9.

### 2. Fill `config.yaml` ✅
- [x] `data`, `classification`, `regression`, `test_set`, `diagnostics`, `preprocessing`. Only rules are stored, never fitted values.

### 3. Missingness mechanism diagnosis ✅ (notebook sections 5.1, 5.2, 6)
- [x] **Identity recovery** (stateless): `prior_mean_purchase_value = prior_total_spend / prior_purchase_count`, `days_since_previous_purchase` ↔ `prior_mean_gap_days` via tenure, `spend_365d = prior_total_spend` when tenure ≤ 365. Each holds for 99.2–100% of rows and together they fill 698 values.
- [x] **Structural "not applicable" gaps** → 0 + `is_first_purchase` / `has_gap_history` flags (each history column is 100% missing exactly where it is undefined).
- [x] **Remaining gaps, tested** (χ² vs target, Mann-Whitney / χ² vs every feature, Bonferroni, sparse categories grouped): `days_since_previous_purchase`/`prior_mean_gap_days` (8 rows) and the 4 categoricals → MCAR → median / most frequent. `spend_365d` → related to tenure after the identity fill → median + `spend_365d_was_missing`.
- [x] Verdict table in the notebook (section 6).

### 4. Leak-safe `build_preprocessor()` ✅ (notebook section 8)
- [x] numeric: median → scaler chosen per task in item 5 (`knn` option available); categorical: most frequent → target encoder (`target_type="auto"`, so it works for both tasks); flags passed through. 35 model-ready columns.
- [x] Fitted on development rows only; locked test and test.csv transformed with 0 NaN; unseen categories demo (they get the overall target mean).

### 5. Choose encoder × scaler with evidence ✅ (notebook section 9)
- [x] 4 encoders × 4 scalers scored with a scaling-sensitive model (logistic regression / ridge) on the **same 5 temporal folds** of the development set (`src/evaluate.py` → `compare_encoders_scalers`), with a heatmap per task.
- [x] Paired comparisons (`paired_comparison`, mean advantage vs 2·SE).
- [x] Classification → **target + standard** (0.7148 accuracy; tied with target + robust, +0.0005 ± 0.0006; decisively better than one-hot + robust).
- [x] Regression → **target + minmax** (RMSE 2104; tied with ordinal + minmax, 3 ± 19 → tie goes to target, because ordinal imposes a fake order).
- [x] Stored per task in `config.yaml` → `classification.preprocessing` / `regression.preprocessing`, merged by `task_preprocessing_config()`.

### 6. Recipe + model = one `Pipeline` ✅ (notebook section 10)
- [x] `make_pipeline()` = `Pipeline([("prep", build_preprocessor(...)), ("model", build_model(...))])`; `classification/model.py` and `regression/model.py` hold the model factories.
- [x] Dummy baselines: classification 0.553 accuracy, regression 2472 RMSE.
- [x] Train vs validation for 4 candidates per task. Chosen: **logistic regression** (0.715 val, no overfitting; random forest 0.709, overfits) and **random forest regressor** (2008 val RMSE, R² 0.30; ahead of ridge 96 ± 64, not decisive; overfits: train 1008 → tune later).
- [x] `python main.py` cross-validates the configured model vs the dummy and fits the final pipeline on the development rows. The locked test set is still untouched.

### 7. Validation split that matches how the test set was built ✅
- [x] Locked test set = latest 20% of purchases (2023-06 → 2025-02), with a 90-day gap (`test_set.gap_days`) so no development label window reaches into it.
- [x] CV inside the development set = 5 expanding-window temporal folds with the same 90-day gap (`validation`, `src/evaluate.py` → `temporal_folds`).
- [ ] Optional: compare against `GroupKFold` on `customer_id` to show how optimistic a random/grouped split would be.

### 8. Regression-specific preprocessing ✅ (notebook sections 2.5, 10)
- [x] Missing spend among repeaters (196 / 3,621 = 5.4%) tested: **not MCAR**. Concentrated at the data edges (2015 16.8%, 2024 11.6%, 2025 49% vs 1–5% in between), and related to 13 of 32 features (shorter, lower-value histories). Never imputed: the regressor trains on observed rows only; consequence documented.
- [x] Regression preprocessor fitted only on the regression rows (`regression.select_training_rows()` before the split).
- [x] `log1p` target (`TransformedTargetRegressor`, `regression.model.target_transform`) tested: decisively worse in RMSE for every model on every fold (RF +345, tree +191, ridge +1194) → `target_transform: none`.

### 9. Fix bugs / weaknesses in the current notebook ✅
- [x] Duplicate check is diagnostic only (`duplicate_diagnostics`); test.csv is never filtered and `assert len(test_clean) == 986`.
- [x] Duplicate check also ignores `id` (0 duplicates either way; 0 test rows identical to a train row).
- [x] Almost-empty rows: diagnosed in 2.1, removed by `drop_training_rows()` in the new 2.4 (training only).
- [x] Column renames, non-negative columns, placeholder value, categorical columns, preserved columns and paths all read from `config.yaml`; `SEED` removed (seeds live in config).
- [x] Notebook functions duplicated from `src/` removed (`standardize_columns`, `replace_invalid_values`); all imports and the config load moved to section 0.
- [x] Notebook executed top to bottom in a fresh kernel, outputs saved (empty cell removed).
- [x] Table of contents added, every section has a unique anchor.
- [x] The dozens of per-feature plots (old 2.4) replaced by one test table, one by-year chart and a written verdict.
- [x] `�` characters: not actually in the notebook (it was a console encoding artefact) → nothing to fix.

### 10. Drop non-informative and leaky columns (and say why)
- [x] 🔎 `random_noise` — literally noise (corr with target −0.012). Dropped.
- [x] `id`, `customer_id`, `purchase_date` → kept in `extras`, never model inputs.
- [x] Both targets are removed from `X` for both tasks (`data.target_columns`).
- [x] Redundancy: `gross_total` (exact sum) and `total_payable` (corr 0.996 with `net_total`) dropped in `config.yaml`.

---

## P1 — Data-quality checks still missing

- [ ] 🔎 **Accounting identities:** `gross − line_discount − global_discount = net` holds for 100 % of rows; `net + tax + shipping = total_payable` holds for only **83.7 %** → investigate the other 16 % (rounding? missing tax? invalid?) and decide.
- [ ] **Logical consistency asserts:** `count_30d ≤ 90d ≤ 180d ≤ 365d` and same for spend; `prior_min ≤ prior_mean ≤ prior_max`; `customer_tenure_days ≥ days_since_previous_purchase`; `spend_365d ≤ prior_total_spend` (🔎 30d ≤ 90d and min ≤ max already pass — add the rest).
- [ ] 🔎 **Rare categories:** `transport_method` has 80 levels, 40 of them with < 10 rows → group rare levels into `"Other"` (threshold fitted on train) or use count/target encoding.
- [ ] **Outliers & skew** (`2.Outlier_detection.ipynb`):
  - [ ] It reuses another project's code/columns (cell 24 `spend_vegetables_per_year`, …) and saves to `Data/cleaned_df` (wrong folder, no `.csv`) — rewrite for this dataset.
  - [ ] 🔎 Skew goes up to 27.5 (`global_discount_total`), 21.3 (`prior_max_purchase_value`), 20.4 (`spend_365d`) → per-column decision: `log1p`, winsorise with caps fitted on train, or `RobustScaler`.
  - [ ] Never drop outlier rows from test; any removal is train-only and its effect must be measured in CV.
- [ ] **`clientes.csv` merge** (optional table):
  - [ ] 🔎 Drop `customer_type` (100 % NaN); high-missing address fields (`street` 52 %, `island` 51 %, `city` 48 %) → drop or reduce to `commercial_region`/`country_or_region`.
  - [ ] Check `customer_since_year ≤ purchase year` (no future info) and handle customers not found in `clientes` (your cross-check in §2.1).
  - [ ] Compare model score with vs without the merge.
- [ ] **Purchase-time features** (no future info): month / weekday / quarter from `purchase_date`, margin (`net_total − cost`), discount rate, average line value, `spend_30d / spend_365d` ratios.

---

## P2 — Challenge / extra credit (02 notebook "Challenge")

- [ ] Alternative imputers (`KNNImputer` with scaling first, `IterativeImputer`, constant) compared **inside CV**.
- [ ] Extra data issues found beyond the EDA (items in P1) added to `clean_dataset`.
- [ ] Compare preprocessed models vs the week-2 baseline; explain why it improved (or didn't); state the current best model.
- [ ] Leakage demo: fit scaler/encoder on the full data vs per fold and show the optimistic gap.
- [ ] Sanity tests: `clean_dataset(test)` keeps 986 rows in the same order; works with targets absent; no NaN after `transform`; train/test column sets match.

---

## Documentation

- [ ] README → **Preprocessing decisions** section: one line per column (problem | mechanism | action | why).
- [ ] README → **Pipeline progress** log (week by week) and **Best model** section.
- [ ] README → fix project structure (it lists `notebooks/` and `results/`, which don't exist; notebooks are in the root).
- [ ] Every notebook section ends with a short markdown conclusion, not just output.
