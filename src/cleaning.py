"""
Cleaning rules shared by both tasks -- the diagnosis of preprocessing.ipynb turned into code:

  - stateless rules (`clean_dataset`, `add_customer_features`, both run by `prepare_table`):
    impossible values -> NaN, consistency rules, exact recovery from identities, structural
    fill, flags, date parts, ratios, redundant columns, the customer-table join. They learn
    nothing from the data and keep every row, so they run on train.csv AND test.csv.
  - history rule (`fill_from_customer_history`, run after `prepare_table`): gaps filled from
    the same customer's earlier purchases. It reads other rows of the same customer -- past
    ones only -- so it needs the purchase history, but it learns nothing and keeps every row.
  - training-data decision (`drop_training_rows`): removes almost-empty rows and duplicates
    from the labelled training file only, before the dev/test split.

All rules come from config.yaml (`diagnostics`, `customers`, `schema`, `dtypes`).
"""
import re

import numpy as np
import pandas as pd

from src.customers import prepare_customers
from src.data import load_data, standardize_columns, standardize_dtypes, validate_schema


def flag_invalid_values(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Converts impossible values to NaN **in place**: negative amounts/counts/intervals, and
    the `1,000,000,000` placeholder found in two historical-spend columns. Returns a report
    of how many values were converted per column.
    """
    report = []
    for col in diagnostics_config.get("non_negative_columns", []):
        if col in df.columns:
            mask = df[col].lt(0)
            report.append({"step": "negative -> NaN", "column": col, "values": int(mask.sum())})
            df.loc[mask, col] = np.nan

    placeholders = diagnostics_config.get("placeholder_values", [])
    for col in diagnostics_config.get("placeholder_columns", []):
        if col in df.columns:
            mask = df[col].isin(placeholders)
            report.append({"step": "placeholder -> NaN", "column": col, "values": int(mask.sum())})
            df.loc[mask, col] = np.nan
    return pd.DataFrame(report)


def _referenced_columns(expression: str, columns) -> list:
    return [c for c in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression) if c in columns]


def apply_consistency_rules(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """
    Checks logical rules between columns of the same row (e.g. 30-day spend <= 90-day
    spend). A rule is only checked on rows where every column it uses is observed. When a
    rule names a column to `invalidate`, that column is set to NaN **in place** on the
    violating rows -- the value is impossible, so it is treated as missing (and may then be
    recovered by `apply_recompute_rules`). Rules without `invalidate` are report-only.
    Returns one report row per rule.
    """
    report = []
    for rule in rules:
        expression = rule["rule"]
        used = _referenced_columns(expression + " " + rule.get("when", ""), df.columns)
        checked = df[used].notna().all(axis=1)
        if "when" in rule:
            checked &= df.eval(rule["when"]).fillna(False).astype(bool)
        violations = checked & ~df.eval(expression).fillna(False).astype(bool)
        column = rule.get("invalidate")
        if column:
            df.loc[violations, column] = np.nan
        report.append({"step": f"inconsistent -> NaN ({column})" if column else "consistency check (report only)",
                       "column": column or "", "rule": expression,
                       "rows_checked": int(checked.sum()), "values": int(violations.sum())})
    return pd.DataFrame(report)


def apply_recompute_rules(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """
    Fills NaN **in place** with values that follow from an exact identity between columns
    (e.g. prior mean purchase value = prior total spend / prior purchase count). Each rule
    only reads the same row, so nothing is learned from other rows. Returns a report.
    """
    report = []
    for rule in rules:
        col = rule["column"]
        condition = df.eval(rule["when"]) if "when" in rule else pd.Series(True, index=df.index)
        values = df.eval(rule["expression"])
        mask = df[col].isna() & condition.fillna(False).astype(bool) & values.notna()
        report.append({"step": "recomputed from identity", "column": col, "values": int(mask.sum())})
        df.loc[mask, col] = values[mask]
    return pd.DataFrame(report)


def apply_structural_fill(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """
    Fills **in place** the values that are missing because they cannot exist ("not
    applicable"): a first purchase has no previous purchase, a standard deviation needs at
    least two values, and so on. This is a rule from the column definitions, not a fitted
    statistic. Returns a report.
    """
    report = []
    for rule in rules:
        condition = df.eval(rule["when"]).fillna(False).astype(bool)
        for col in rule["columns"]:
            mask = df[col].isna() & condition
            report.append({"step": f"structural fill ({rule['when']})", "column": col, "values": int(mask.sum())})
            df.loc[mask, col] = rule["value"]
    return pd.DataFrame(report)


def add_flags(df: pd.DataFrame, flags: dict) -> pd.DataFrame:
    """Adds 0/1 columns defined by a boolean expression, e.g. is_first_purchase."""
    out = df.copy()
    for name, expression in flags.items():
        out[name] = out.eval(expression).fillna(False).astype(int)
    return out


_DATE_PARTS = {
    "month": lambda d: d.dt.month,
    "quarter": lambda d: d.dt.quarter,
    "weekday": lambda d: d.dt.weekday,
}


def add_date_features(df: pd.DataFrame, date_column: str, parts: list) -> pd.DataFrame:
    """`purchase_month`, `purchase_weekday`, ... -- known at the moment of the purchase."""
    out = df.copy()
    for part in parts:
        out[f"purchase_{part}"] = _DATE_PARTS[part](out[date_column])
    return out


def add_derived_features(df: pd.DataFrame, features: dict) -> pd.DataFrame:
    """New columns from expressions on the same row (margin, discount rate, ...).
    Divisions by zero become NaN, never +/-inf."""
    out = df.copy()
    for name, expression in features.items():
        out[name] = out.eval(expression).replace([np.inf, -np.inf], np.nan)
    return out


def add_customer_features(df: pd.DataFrame, clientes: pd.DataFrame, customers_config: dict, report: list = None) -> pd.DataFrame:
    """
    Prepares the customer table (`src/customers.py` -> `prepare_customers`) and left-joins it to
    the purchases on the customer id. Stateless and row-preserving (same rows, same order): a
    customer missing from clientes simply gets NaN, which the fitted imputers handle.
    `derived_features` are computed after the join, at the moment of each purchase (e.g. years
    as a customer), then checked with `consistency_rules` (e.g. no purchase before the customer
    existed). Appends the customer report rows to `report` when a list is given.
    """
    if not customers_config.get("use", False):
        return df
    key = customers_config["key"]
    customers, customer_report = prepare_customers(clientes, customers_config)
    out = df.join(customers.set_index(key), on=key)
    unknown = out[key].notna() & ~out[key].isin(customers[key])
    out = add_derived_features(out, customers_config.get("derived_features", {}))
    join_report = apply_consistency_rules(out, customers_config.get("consistency_rules", []))
    out = out.drop(columns=[c for c in customers_config.get("drop_after_derive", []) if c in out.columns])
    if len(out) != len(df) or not out.index.equals(df.index):
        raise AssertionError("add_customer_features must keep every row, in the same order.")
    if report is not None:
        report.extend(customer_report.to_dict("records"))
        report.append({"step": "customers: purchases whose customer is not in clientes.csv", "column": key,
                       "values": int(unknown.sum()), "rule": ""})
        report.extend({**r, "step": f"customers: {r['step']}"} for r in join_report.to_dict("records"))
    return out


def fill_from_customer_history(df: pd.DataFrame, history: pd.DataFrame, diagnostics_config: dict,
                               id_column: str = "id"):
    """
    Fills gaps from the SAME customer's EARLIER purchases (`diagnostics.customer_history`):
      - `last_observed_columns`: the customer's last observed value before this purchase
        (customers keep their payment type, salesperson, ... from one purchase to the next);
      - `previous_purchase_gap`: the date difference to the customer's previous purchase, when
        that purchase is in the data and is the immediately previous one (its prior purchase
        count is exactly one lower);
    then `recompute_rules` run again, so values that follow from a filled one (the mean gap
    between purchases) are recovered too.

    Past purchases only, so nothing comes from the future, and no target is read: the same
    information a prediction-time system has. `history` is every purchase known at prediction
    time, prepared by `prepare_table` and not yet filled (train.csv + test.csv); `df`'s own rows
    may be part of it. Rows without a date are neither filled nor used. Row-preserving.
    Returns (df, report).
    """
    rules = diagnostics_config.get("customer_history", {})
    out = df.copy()
    if not rules.get("use", False):
        return out, pd.DataFrame(columns=["step", "column", "values", "rule"])
    key, date, order = rules["key"], rules["date_column"], rules["order_column"]
    columns = [c for c in rules.get("last_observed_columns", []) if c in out.columns]
    gap_column = rules.get("previous_purchase_gap")
    needed = [id_column, key, date, order] + columns + ([gap_column] if gap_column else [])

    pool = pd.concat([history[needed], out[needed]]).drop_duplicates(subset=id_column, keep="last")
    pool = pool[pool[date].notna()].sort_values([key, date, order], kind="mergesort").reset_index(drop=True)
    groups = pool.groupby(key, sort=False)
    report = []

    for column in columns:
        # last observed value strictly before each purchase: shift one purchase back, then carry forward
        earlier = groups[column].shift(1).astype(object)
        last_observed = earlier.groupby(pool[key]).ffill()
        fill = out[id_column].map(pd.Series(last_observed.to_numpy(), index=pool[id_column]))
        mask = out[column].isna() & fill.notna()
        was_category = isinstance(out[column].dtype, pd.CategoricalDtype)
        values = out[column].astype(object)
        values[mask] = fill[mask]
        out[column] = values.astype("category") if was_category else values
        report.append({"step": "filled from the customer's last earlier purchase", "column": column,
                       "values": int(mask.sum()), "rule": "last observed value, past purchases only"})

    if gap_column:
        consecutive = (pool[order] - groups[order].shift(1)) == 1
        gap = (pool[date] - groups[date].shift(1)).dt.days.where(consecutive)
        fill = out[id_column].map(pd.Series(gap.to_numpy(), index=pool[id_column]))
        mask = out[gap_column].isna() & fill.notna()
        out.loc[mask, gap_column] = fill[mask]
        report.append({"step": "filled from the customer's previous purchase date", "column": gap_column,
                       "values": int(mask.sum()), "rule": "purchase_date - previous purchase_date (count one lower)"})

    names = lambda rule: re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join([rule["column"], rule["expression"], rule.get("when", "")]))
    usable = [r for r in diagnostics_config.get("recompute_rules", []) if set(names(r)) <= set(out.columns)]
    recomputed = apply_recompute_rules(out, usable)
    report = pd.concat([pd.DataFrame(report), recomputed.assign(step="recomputed after the history fill")],
                       ignore_index=True)
    if len(out) != len(df) or not out.index.equals(df.index):
        raise AssertionError("fill_from_customer_history must keep every row, in the same order.")
    return out, report[report["values"] > 0].reset_index(drop=True)


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict):
    """
    Applies the diagnosis as stateless rules. Row-preserving -- same rows, same order --
    and target-agnostic, so it runs unchanged on train.csv and on test.csv.

    Returns (clean_df, report) where report counts every value each step changed.
    """
    out = standardize_columns(df, diagnostics_config.get("column_renames", {}))

    date_column = diagnostics_config.get("date_column")
    if date_column and date_column in out.columns:
        out[date_column] = pd.to_datetime(out[date_column], errors="coerce")

    reports = [
        flag_invalid_values(out, diagnostics_config),
        apply_consistency_rules(out, diagnostics_config.get("consistency_rules", [])),
        apply_recompute_rules(out, diagnostics_config.get("recompute_rules", [])),
        apply_structural_fill(out, diagnostics_config.get("structural_fill", [])),
    ]
    out = add_flags(out, diagnostics_config.get("flags", {}))
    if date_column and date_column in out.columns:
        out = add_date_features(out, date_column, diagnostics_config.get("date_features", []))
    out = add_derived_features(out, diagnostics_config.get("derived_features", {}))

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=columns_to_drop)

    if len(out) != len(df):
        raise AssertionError("clean_dataset must keep every row.")
    report = pd.concat(reports, ignore_index=True)
    return out, report[report["values"] > 0].reset_index(drop=True)


def prepare_table(df: pd.DataFrame, config: dict, clientes: pd.DataFrame = None):
    """Every stateless step, in the order main.py and the notebook use for train.csv and
    test.csv: standardise names -> validate_schema -> clean_dataset -> customer-table join ->
    compact dtypes. Returns (df, report): every value the validation and cleaning changed."""
    diagnostics = config["diagnostics"]
    out = standardize_columns(df, diagnostics.get("column_renames", {}))
    out, schema_report = validate_schema(out, config.get("schema", {}), diagnostics.get("non_negative_columns", []))
    out, report = clean_dataset(out, diagnostics)
    report = pd.concat([schema_report[schema_report["values"] > 0], report], ignore_index=True)
    customers_config = config.get("customers", {})
    if customers_config.get("use", False):
        if clientes is None:
            clientes = load_data(config["data"]["clientes_path"], diagnostics.get("placeholder_tokens"))
        customer_rows = []
        out = add_customer_features(out, clientes, customers_config, customer_rows)
        report = pd.concat([report, pd.DataFrame(customer_rows)], ignore_index=True)
    out = standardize_dtypes(out, config.get("dtypes", {}))
    return out, report


def generated_columns(diagnostics_config: dict, customers_config: dict = None) -> set:
    """Columns the pipeline creates itself (flags, date parts, derived ratios, joined
    customer fields) -- i.e. every column that did not come from the raw purchase file."""
    columns = set(diagnostics_config.get("flags", {})) | set(diagnostics_config.get("derived_features", {}))
    columns |= {f"purchase_{part}" for part in diagnostics_config.get("date_features", [])}
    if customers_config and customers_config.get("use", False):
        columns |= set(customers_config["columns"]) | set(customers_config.get("derived_features", {}))
    return columns


def drop_training_rows(df: pd.DataFrame, diagnostics_config: dict, id_column: str = None, customers_config: dict = None):
    """
    TRAINING DATA ONLY, before the dev/test split. Removes:
      - almost-empty rows: every column of the purchase file except the preserved ones
        (ids, target) is missing, so imputing them would invent a whole purchase;
      - exact duplicate rows (ignoring the id) and repeated ids, keeping the first.

    Never call this on data we predict for: every Kaggle id needs a prediction.
    Returns (df, report).
    """
    preserved = set(diagnostics_config.get("almost_empty_preserved_columns", []))
    ignored = preserved | generated_columns(diagnostics_config, customers_config)
    checked = [c for c in df.columns if c not in ignored]
    almost_empty = df[checked].isna().all(axis=1)
    out = df.loc[~almost_empty]

    non_id_columns = [c for c in out.columns if c != id_column]
    exact_duplicates = out.duplicated(subset=non_id_columns, keep="first")
    out = out.loc[~exact_duplicates]

    repeated_ids = out.duplicated(subset=id_column, keep="first") if id_column in out.columns else pd.Series(False, index=out.index)
    out = out.loc[~repeated_ids]

    report = pd.DataFrame([
        {"rule": "almost-empty rows", "rows_removed": int(almost_empty.sum())},
        {"rule": "exact duplicates (ignoring id)", "rows_removed": int(exact_duplicates.sum())},
        {"rule": "repeated ids", "rows_removed": int(repeated_ids.sum())},
    ])
    return out.copy(), report
