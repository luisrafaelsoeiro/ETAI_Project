"""
Loading and validating raw tables, shared by both tasks: reading a CSV with placeholder
tokens turned into NaN, the placeholder report, English snake_case column names, schema
validation (required columns, expected types, valid ranges, allowed categories), the
customer_id checks and the compact storage dtypes.

Everything here is stateless and row-preserving: it learns nothing from the data, so it runs
unchanged on train.csv, test.csv and any new record.
"""
import re
import unicodedata

import numpy as np
import pandas as pd


def load_data(path, placeholder_tokens=None) -> pd.DataFrame:
    """Load a raw CSV file. Every placeholder token (blank cell, "?", "N/A", ...) becomes
    NaN on load, so a numeric column can't silently turn into text because of one token."""
    tokens = [t for t in (placeholder_tokens or []) if t != ""]   # blank cells are NaN already
    return pd.read_csv(path, na_values=tokens or None)


def report_placeholders(path, diagnostics_config: dict) -> pd.DataFrame:
    """
    Missing-value report straight from the raw file (read as text, so nothing is converted
    yet): how often each placeholder token appears in each column, plus the numeric
    placeholder values (e.g. 1,000,000,000). Report only -- `load_data` turns the tokens
    into NaN and `flag_invalid_values` handles the numeric placeholders.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    tokens = diagnostics_config.get("placeholder_tokens", [])
    rows = []
    for column in raw.columns:
        stripped = raw[column].str.strip()
        for token in tokens:
            count = int(stripped.eq(token).sum())
            if count:
                rows.append({"column": column, "placeholder": repr(token), "count": count,
                             "treatment": "NaN on load (load_data)"})
        numeric = pd.to_numeric(stripped, errors="coerce")
        for value in diagnostics_config.get("placeholder_values", []):
            count = int(numeric.eq(value).sum())
            if count:
                rows.append({"column": column, "placeholder": f"{value:,}", "count": count,
                             "treatment": "NaN in clean_dataset (flag_invalid_values)"})
    return pd.DataFrame(rows, columns=["column", "placeholder", "count", "treatment"])


def to_snake_case(name) -> str:
    name = unicodedata.normalize("NFKD", str(name))
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return re.sub(r"_+", "_", name)


def standardize_columns(df: pd.DataFrame, column_renames: dict) -> pd.DataFrame:
    """Portuguese column names -> English snake_case (explicit map first, then a generic rule)."""
    out = df.copy()
    out.columns = [column_renames.get(c, to_snake_case(c)) for c in out.columns]
    if out.columns.duplicated().any():
        raise ValueError("Column standardisation created duplicate names.")
    return out


def validate_schema(df: pd.DataFrame, schema_config: dict, non_negative_columns=()):
    """
    Validates a table with standardised column names against config.yaml -> `schema`, and
    returns (df, report). Stateless and row-preserving.
      - required columns: an error if one is missing; unexpected extra columns are reported;
      - expected data types: numeric columns that are not numeric are coerced (values that
        can't be read as numbers -> NaN, counted); dates that don't parse are counted;
      - valid ranges: values outside [min, max] are counted (`clean_dataset` turns them
        into NaN). Every non-negative column gets min 0 automatically;
      - allowed categories: each categorical column has a pattern for its labels; a label
        that doesn't match is not a category at all -> NaN, counted. Labels that match but
        were never seen in training are fine: the encoders handle them.
    """
    out = df.copy()
    rows = []

    def add(step, column, values, rule=""):
        rows.append({"step": step, "column": column, "values": int(values), "rule": rule})

    missing = [c for c in schema_config.get("required_columns", []) if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    known = set(schema_config.get("required_columns", [])) | set(schema_config.get("optional_columns", []))
    for column in [c for c in out.columns if known and c not in known]:
        add("schema: unexpected column", column, len(out))

    types = schema_config.get("expected_types", {})
    for column in types.get("numeric", []):
        if column in out.columns and not pd.api.types.is_numeric_dtype(out[column]):
            coerced = pd.to_numeric(out[column], errors="coerce")
            add("schema: non-numeric -> NaN", column, coerced.isna().sum() - out[column].isna().sum(), "numeric")
            out[column] = coerced
    for column in types.get("date", []):
        if column in out.columns:
            parsed = pd.to_datetime(out[column], errors="coerce")
            add("schema: unparseable date", column, parsed.isna().sum() - out[column].isna().sum(), "date")
    for column in types.get("integer", []):
        if column in out.columns:
            values = out[column].dropna()
            add("schema: non-integer value", column, (values != np.round(values)).sum(), "integer")

    ranges = {c: {"min": 0} for c in non_negative_columns}
    for column, bounds in schema_config.get("ranges", {}).items():
        ranges[column] = {**ranges.get(column, {}), **bounds}
    for column, bounds in ranges.items():
        if column not in out.columns:
            continue
        values = out[column]
        if pd.api.types.is_datetime64_any_dtype(values) or column in types.get("date", []):
            values = pd.to_datetime(values, errors="coerce")
            low, high = (pd.Timestamp(bounds[k]) if k in bounds else None for k in ("min", "max"))
        else:
            low, high = bounds.get("min"), bounds.get("max")
        outside = pd.Series(False, index=values.index)
        if low is not None:
            outside |= values < low
        if high is not None:
            outside |= values > high
        add("schema: outside valid range", column, outside.fillna(False).sum(),
            f"[{bounds.get('min', '-inf')}, {bounds.get('max', 'inf')}]")

    for column, pattern in schema_config.get("allowed_categories", {}).items():
        if column in out.columns:
            labels = out[column].astype("string").str.strip()
            invalid = labels.notna() & ~labels.str.fullmatch(pattern).fillna(False)
            add("schema: label not allowed -> NaN", column, invalid.sum(), pattern)
            out.loc[invalid, column] = np.nan

    return out, pd.DataFrame(rows, columns=["step", "column", "values", "rule"])


def validate_customer_ids(df: pd.DataFrame, clientes: pd.DataFrame, data_config: dict) -> pd.DataFrame:
    """
    Report on the customer id of every purchase: present, a positive integer, known in
    clientes.csv, and -- per customer, in date order -- a history that never goes
    backwards (prior purchase count and tenure can't decrease; no two purchases of the
    same customer on the same day). Report only: nothing is removed.
    """
    key, date = data_config["group_column"], data_config["date_column"]
    ids = df[key]
    numeric_ids = pd.to_numeric(ids, errors="coerce")
    known = set(standardize_columns(clientes, {})[key])
    rows = [
        ("missing customer_id", ids.isna().sum()),
        ("customer_id not a positive integer", (ids.notna() & ~((numeric_ids > 0) & (numeric_ids == np.round(numeric_ids)))).sum()),
        ("customer_id not in clientes.csv", (ids.notna() & ~ids.isin(known)).sum()),
    ]
    ordered = df.assign(_date=pd.to_datetime(df[date], errors="coerce")).sort_values([key, "_date"])
    by_customer = ordered.groupby(key)
    for column in ["prior_purchase_count", "customer_tenure_days"]:
        if column in ordered.columns:
            rows.append((f"{column} decreases over a customer's purchases", (by_customer[column].diff() < 0).sum()))
    dated = ordered[ordered["_date"].notna()]                 # rows without a date can't collide
    rows.append(("same customer, same purchase date", dated.duplicated([key, "_date"]).sum()))
    return pd.DataFrame([{"check": check, "rows": int(count)} for check, count in rows])


def standardize_dtypes(df: pd.DataFrame, dtype_config: dict) -> pd.DataFrame:
    """Apply the compact pandas data types listed in config.yaml.

    Missing integer values use pandas' nullable integer types. Monetary values and
    calculated statistics are deliberately left as float64.
    """
    out = df.copy()
    for dtype, columns in dtype_config.items():
        for column in columns:
            if column in out.columns:
                out[column] = out[column].astype(dtype)
    return out
