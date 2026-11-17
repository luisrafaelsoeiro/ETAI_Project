# Data

`raw/` holds the original project files, never modified. Every cleaned table is built from
them in memory by `src/cleaning.py` → `prepare_table()`; nothing derived is written here.

| File | Rows | What it is |
|---|---|---|
| `raw/train.csv` | 6,534 purchases | Predictors and both targets (2015-07-11 → 2025-02-15) |
| `raw/test.csv` | 986 purchases | Predictors only, for the Kaggle predictions (2025-05-24 → 2026-10-06) |
| `raw/clientes.csv` | 1,143 customers | Optional customer table, joined on `customer_id` (used by the regression) |
| `raw/sample_submission.csv` | 986 rows | Kaggle submission format: `ID`, `repeat_purchase_90d` |

Each row of train.csv / test.csv is one **purchase occasion** of a customer. Every predictor
is known at the moment of that purchase.

## Columns (names after `standardize_columns`)

| Group | Columns |
|---|---|
| Identifiers | `id` (`ID` in the files), `customer_id`, `purchase_date` |
| Current purchase | `total_payable`, `gross_total`, `net_total`, `total_tax`, `shipping_cost`, `global_discount_total`, `line_discount_total`, `cost`, `line_item_count`, `invoice_count_current_purchase` |
| Purchase details | `payment_type`, `salesperson`, `commercial_zone`, `transport_method`, `warehouse`, `document_series` |
| Purchase history | `days_since_previous_purchase`, `customer_tenure_days`, `prior_purchase_count`, `prior_total_spend`, `prior_mean/std/min/max_purchase_value`, `prior_mean_gap_days`, `prior_std_gap_days` |
| Rolling windows | `purchase_count_30d/90d/180d/365d`, `spend_30d/90d/180d/365d` |
| Other | `random_noise` (dropped: pure noise) |
| **Targets** (train.csv only) | `repeat_purchase_90d` (classification: another purchase within 90 days, 0/1); `spend_next_90d` (regression: spend in those 90 days, only for repeat purchasers) |
| clientes.csv | `customer_id`, `country_or_region`, `commercial_region`, `island`, `municipality`, `city`, `postal_area`, `postal_code`, `street`, `address`, `customer_type`, `company_name`, `customer_since_year` |

The original Portuguese column names (`Total a Pagar`, `Custo`, `N_ Detalhes`, …) are mapped
in `config.yaml` → `diagnostics.column_renames`. Data-quality findings and every cleaning
decision are documented in the main README (*Preprocessing Decisions*) and in
`notebooks/01_data_cleaning.ipynb` (cleaning), followed by the other notebooks of the workflow.
