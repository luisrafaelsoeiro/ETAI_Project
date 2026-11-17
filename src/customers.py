"""
Preprocessing of the customer table (clientes.csv), shared by both tasks.

clientes.csv is master data: one row per customer, no target, no dates of purchase. Every step
below is a rule on that static table -- it reads no purchase row and no label -- so it runs
the same for training, test.csv and any new record (`prepare_customers`). The cleaned table is
then joined to the purchases by `src/cleaning.py` -> `add_customer_features`.

Steps, in order (rules in config.yaml -> customers; evidence in notebooks/02_customer_table.ipynb):
  1. validate: required columns, types, ranges, allowed label patterns (`validate_schema`);
  2. key: customer_id present and unique -- a duplicated key would duplicate purchases on join;
  3. duplicates and near-duplicates: reported (exact rows, company names, shared addresses);
  4. columns with no usable information dropped (100% missing);
  5. labels -> codes: "Region RG17: We Will See Later" -> "RG17" (the text after the code is
     filler, identical for every customer with the same code);
  6. profile completeness: how many address fields the customer record has (before recovery);
  7. exact recovery through the geographic hierarchy: a missing region is the region of the
     customer's postal area / postal code / municipality when that child belongs to one region;
  8. structural fill: `island` is undefined outside the island regions -> "none";
  9. rare-level backoff: a level with too few customers is replaced by its parent level
     (municipality -> its region) or by "other";
 10. flags (e.g. customer registered before the records start) and the final column selection.
"""
import numpy as np
import pandas as pd

from src.data import standardize_columns, validate_schema


def _report(step: str, column: str, values: int, rule: str = "") -> dict:
    return {"step": f"customers: {step}", "column": column, "values": int(values), "rule": rule}


def check_customer_key(table: pd.DataFrame, key: str) -> None:
    """The join key must be present and unique, or the join would duplicate purchases."""
    if table[key].isna().any():
        raise ValueError(f"clientes.csv has {int(table[key].isna().sum())} rows without {key}.")
    duplicated = table[key].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(f"clientes.csv has repeated {key}: {sorted(table.loc[duplicated, key].unique())[:10]}")


def duplicate_report(table: pd.DataFrame, key: str, near_duplicate_columns: list) -> list:
    """Exact duplicate customers (ignoring the key) and customers sharing a value that should be
    individual (company name, address). Report only: different ids are different accounts in
    the purchase files, so they are never merged."""
    rows = [_report("exact duplicate customers (ignoring id)", "",
                    table.drop(columns=[key]).duplicated().sum())]
    for column in near_duplicate_columns:
        if column in table.columns:
            shared = table[column].notna() & table[column].duplicated(keep=False)
            rows.append(_report("customers sharing a value", column, shared.sum(), "report only"))
    return rows


def labels_to_codes(table: pd.DataFrame, columns: list, pattern: str):
    """'Region RG17: We Will See Later' -> 'RG17'. Checks first that each code always comes with
    the same label (so nothing is lost); a label without a code is not allowed -> NaN."""
    out, rows = table.copy(), []
    for column in columns:
        if column not in out.columns:
            continue
        codes = out[column].astype("string").str.extract(pattern, expand=False)
        labels_per_code = out[column].groupby(codes).nunique()
        if (labels_per_code > 1).any():
            raise ValueError(f"{column}: some codes have more than one label -- codes would merge different values.")
        unreadable = out[column].notna() & codes.isna()
        rows.append(_report("label -> code", column, out[column].notna().sum(), pattern))
        if unreadable.any():
            rows.append(_report("label without a code -> NaN", column, unreadable.sum(), pattern))
        out[column] = codes.astype(object).where(codes.notna(), np.nan)
    return out, rows


def profile_completeness(table: pd.DataFrame, columns: list) -> pd.Series:
    """Number of address fields the customer record actually has (before any recovery)."""
    return table[[c for c in columns if c in table.columns]].notna().sum(axis=1)


def hierarchy_report(table: pd.DataFrame, pairs: list) -> list:
    """For each (child, parent) pair: how many child codes belong to more than one parent."""
    rows = []
    for child, parent in pairs:
        both = table[[child, parent]].dropna()
        conflicting = (both.groupby(child)[parent].nunique() > 1).sum()
        rows.append(_report("hierarchy conflicts (child codes with >1 parent)", f"{child} -> {parent}", conflicting))
    return rows


def recover_from_hierarchy(table: pd.DataFrame, rules: list):
    """
    Fills a missing value from a lower geographic level that belongs to exactly one value of
    this level (e.g. the region of the customer's postal area). The lookup is built from the
    customers where both levels are known; child codes with conflicting parents are skipped.
    """
    out, rows = table.copy(), []
    for rule in rules:
        column = rule["column"]
        for child in rule["from"]:
            both = out[[child, column]].dropna()
            parents = both.groupby(child)[column].nunique()
            lookup = both[both[child].isin(parents[parents == 1].index)].drop_duplicates(child).set_index(child)[column]
            fill = out[column].isna() & out[child].isin(lookup.index)
            out.loc[fill, column] = out.loc[fill, child].map(lookup)
            rows.append(_report("recovered from hierarchy", column, fill.sum(), f"from {child}"))
    return out, rows


def structural_fill(table: pd.DataFrame, rules: dict):
    """
    A level that only exists inside some parent values (islands only exist in two regions):
    where the parent is known and has no customer with this level, the value is undefined, not
    missing -> `value` (e.g. "none"). Where the parent is unknown it stays missing.
    """
    out, rows = table.copy(), []
    for column, rule in rules.items():
        parent = rule["parent"]
        parents_with_level = set(out.loc[out[column].notna(), parent].dropna())
        fill = out[column].isna() & out[parent].notna() & ~out[parent].isin(parents_with_level)
        out.loc[fill, column] = rule["value"]
        rows.append(_report(f"structural fill -> '{rule['value']}'", column, fill.sum(),
                            f"{parent} has no {column}"))
    return out, rows


def rare_level_backoff(table: pd.DataFrame, rules: dict):
    """
    A level with fewer than `min_customers` customers in the customer table acts like a
    customer id and can't be learned from: it is replaced by its parent level
    (`fallback_column`, prefixed so the values don't mix) or by a fixed `fallback` label.
    Counts come from the static customer table, never from labelled purchases.
    """
    out, rows = table.copy(), []
    for column, rule in rules.items():
        counts = out[column].value_counts()
        rare = out[column].notna() & out[column].map(counts).lt(rule["min_customers"])
        if "fallback_column" in rule:
            replacement = out.loc[rare, rule["fallback_column"]].map(lambda v: f"other in {v}" if pd.notna(v) else np.nan)
        else:
            replacement = rule["fallback"]
        out.loc[rare, column] = replacement
        rows.append(_report(f"rare level (< {rule['min_customers']} customers) -> backoff", column, rare.sum(),
                            rule.get("fallback_column", rule.get("fallback", ""))))
    return out, rows


def prepare_customers(clientes: pd.DataFrame, customers_config: dict):
    """
    Runs the whole customer-table preprocessing (module docstring) and returns
    (customer table with `key` + `columns`, report). Row-preserving: one row per customer.
    """
    key = customers_config["key"]
    table = standardize_columns(clientes, {})
    table, schema_report = validate_schema(table, customers_config.get("schema", {}))
    rows = [_report(r.step, r.column, r.values, r.rule) for r in schema_report.itertuples() if r.values > 0]
    check_customer_key(table, key)
    rows += duplicate_report(table, key, customers_config.get("near_duplicate_columns", []))

    empty = [c for c in table.columns if c != key and table[c].isna().all()]
    rows += [_report("100% missing -> dropped", c, len(table)) for c in empty]
    table = table.drop(columns=empty)

    table, step = labels_to_codes(table, customers_config.get("code_columns", []), customers_config["code_pattern"])
    rows += step
    table["profile_completeness"] = profile_completeness(table, customers_config.get("profile_columns", []))
    rows += hierarchy_report(table, [tuple(p) for p in customers_config.get("hierarchy_checks", [])])
    table, step = recover_from_hierarchy(table, customers_config.get("hierarchy_recovery", []))
    rows += step
    table, step = structural_fill(table, customers_config.get("structural_fill", {}))
    rows += step
    table, step = rare_level_backoff(table, customers_config.get("rare_level_backoff", {}))
    rows += step
    for name, expression in customers_config.get("flags", {}).items():
        table[name] = table.eval(expression).fillna(False).astype(int)
        rows.append(_report("flag", name, table[name].sum(), expression))

    columns = [key] + customers_config["columns"]
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"customers.columns not produced by the customer pipeline: {missing}")
    return table[columns].reset_index(drop=True), pd.DataFrame(rows, columns=["step", "column", "values", "rule"])

