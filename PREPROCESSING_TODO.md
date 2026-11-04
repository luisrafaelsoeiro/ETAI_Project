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
- [x] numeric: median → robust scaler (`knn` option available); categorical: most frequent → target encoder (`target_type="auto"`, so it works for both tasks); flags passed through. 35 model-ready columns.
- [x] Fitted on development rows only; locked test and test.csv transformed with 0 NaN; unseen categories demo (they get the overall target mean).

### 7. (done early, needed by item 1) Temporal locked test set ✅
- [x] `test_set.strategy: temporal`: the latest 20% of purchases (2023-06 → 2025-02) is the locked test set, matching test.csv lying after train.csv. CV inside the development set (`TimeSeriesSplit` / `GroupKFold`) is still to do with the model work.

### 5. Choose encoder × scaler with evidence (02 notebook Step 4–5)
- [ ] Grid of 4 encoders × 4 scalers, repeated splits, mean ± std (classification: accuracy/F1; regression: RMSE/MAE).
- [ ] Paired comparison winner vs runner-up (mean diff vs 2·SE).
- [ ] Ordinal encoder only with an explicit order (none of your categoricals is ordered → avoid with linear models).
- [ ] Write the winner and the reason into `config.yaml` (don't hardcode a guess).

### 6. Recipe + model = one `Pipeline`
- [ ] `Pipeline([("prep", ...), ("model", ...)])` for classification **and** regression.
- [ ] `DummyClassifier` / `DummyRegressor` baseline (the floor to beat).
- [ ] Report train vs validation score.

### 7. Validation split that matches how the test set was built ⚠️ biggest risk
- [ ] 🔎 `test.csv` is **strictly in the future**: train dates 2015-07-11 → 2025-02-15, test dates 2025-05-24 → 2026-10-06. Your §3 check prints this but the conclusion isn't used anywhere.
- [ ] 🔎 607 customers generate 6,469 train rows, and 184 of the 206 test customers already appear in train. A random stratified split puts the same customer (and overlapping 90-day windows) on both sides → optimistic scores.
- [ ] → Use a **temporal hold-out** (last ~15–20 % of train by `purchase_date`) as the locked test set, and `TimeSeriesSplit` (or `GroupKFold` on `customer_id`) for CV. Justify the choice in markdown and store it in config.

### 8. Regression-specific preprocessing
- [ ] 🔎 `spend_next_90d` is missing for 100 % of non-repeaters (structural, correct to exclude) and for 5.4 % of repeaters → state the verdict for those (can't impute a target; exclude and explain).
- [ ] Fit the regression preprocessor **only on the regression rows** (repeaters with observed spend), not on the classification set.
- [ ] 🔎 Target skew = 3.0 and no zeros → try `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`.

### 9. Fix bugs / weaknesses in the current notebook
- [ ] `validate_and_handle_duplicates(remove_exact=True)` drops rows from **test** too. Kaggle needs a prediction for all 986 IDs → de-duplicate train only and `assert len(test_clean) == 986`.
- [ ] Exact-duplicate check includes `id`, which is always unique, so it can never find a duplicate → also check with `subset` = all columns except `id` (🔎 result: 0 duplicates — report it).
- [ ] The 65 almost-empty rows are dropped mid-cleaning (cell 15) using a mask built on another frame (`train`) — move it into the training-only function.
- [ ] `1_000_000_000`, column lists, `SEED` → config.
- [ ] Cells 9, 15, 17, 21, 23, 25, 27, 30 have **no outputs** → *Restart & Run All* before submitting. Cell 31 is empty.
- [ ] Every section links to `#toc` but there is no Table of Contents, and every anchor has `id="1"` → add a TOC with unique anchors.
- [ ] Cells 18–19 produce one plot per feature (dozens) with no written conclusion → summarise in a table + 2–3 sentences of verdict.
- [ ] Non-ASCII characters show up as `�` in some comments (e.g. "1st–99th") → re-save the notebook as UTF-8.

### 10. Drop non-informative and leaky columns (and say why)
- [x] 🔎 `random_noise` — literally noise (corr with target −0.012). Dropped.
- [x] `id`, `customer_id`, `purchase_date` → kept in `extras`, never model inputs.
- [x] Both targets are removed from `X` for both tasks (`data.target_columns`).
- [ ] ✅ Redundancy: `gross_total` (exact sum) and `total_payable` (corr 0.996 with `net_total`) dropped in `config.yaml`.

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
