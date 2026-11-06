"""
Preprocessing shared by the classification and regression pipelines: raw data in,
model-ready matrix out.

Every step is one of three kinds, and the kind decides where it may run:

  - stateless rule (`clean_dataset`, `split_features_target`): learns nothing from the
    data, keeps every row -> safe on train AND on the Kaggle test file.
  - training-data decision (`drop_training_rows`): removes rows -> labelled training
    file only, before the dev/test split. Never on data we predict for.
  - fitted step (`build_preprocessor`): imputation, encoding, scaling. Lives inside the
    model's sklearn Pipeline, so it is fitted on training rows only.

All column lists, rules and choices come from config.yaml -- nothing is hardcoded here.
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 -- makes IterativeImputer importable
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)
from category_encoders import CountEncoder


def load_data(path) -> pd.DataFrame:
    """Load a raw CSV file."""
    return pd.read_csv(path)


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


def add_customer_features(df: pd.DataFrame, clientes: pd.DataFrame, customers_config: dict) -> pd.DataFrame:
    """
    Left-joins the chosen columns of clientes.csv on the customer id. Stateless and
    row-preserving (same rows, same order): a customer missing from clientes simply gets
    NaN, which the fitted imputers handle. `derived_features` are computed after the join
    (e.g. years as a customer at the moment of the purchase).
    """
    if not customers_config.get("use", False):
        return df
    key = customers_config["key"]
    customers = standardize_columns(clientes, {}).set_index(key)[customers_config["columns"]]
    out = df.join(customers, on=key)
    out = add_derived_features(out, customers_config.get("derived_features", {}))
    out = out.drop(columns=[c for c in customers_config.get("drop_after_derive", []) if c in out.columns])
    if len(out) != len(df) or not out.index.equals(df.index):
        raise AssertionError("add_customer_features must keep every row, in the same order.")
    return out


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


def prepare_table(df: pd.DataFrame, config: dict, clientes: pd.DataFrame = None):
    """clean_dataset + the optional customer-table join: every stateless step, in the order
    main.py and the notebook use for train.csv and test.csv. Returns (df, cleaning report)."""
    out, report = clean_dataset(df, config["diagnostics"])
    customers_config = config.get("customers", {})
    if customers_config.get("use", False):
        if clientes is None:
            clientes = load_data(config["data"]["clientes_path"])
        out = add_customer_features(out, clientes, customers_config)
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


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds `<col>_was_missing` for each column whose missingness carries information,
    before the column is imputed -- the fill value alone can't carry that signal."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, target: str, mnar_indicator_sources: list):
    """
    Returns (X, y, extras).
      - y is None when the target column is absent (test.csv) -- nothing needs it.
      - extras keeps id / customer_id / purchase_date row-aligned with X: needed for the
        temporal split, grouping and the submission file, never as model inputs.
      - every target column is removed from X, so neither task can see the other's label.
    """
    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in data_config.get("extras_columns", []) if c in df.columns]
    extras = df[extras_cols].copy()

    always_drop = set(data_config.get("drop_columns", [])) | set(data_config["target_columns"]) | set(extras_cols)
    X = df[[c for c in df.columns if c not in always_drop]]
    return X, y, extras


def split_dev_test(X, y, extras, test_set_config: dict, date_column: str = None):
    """
    Sets the locked test set aside. Never used to fit, tune, compare or choose anything.

    strategy "temporal" (default): the latest `size` share of purchases becomes the test
    set, because test.csv itself lies strictly after train.csv in time -- a random split
    would let the model see the future of the very customers it is evaluated on.
    `gap_days` leaves out the development rows of the last days before the test period:
    their 90-day labels are measured inside it.
    strategy "stratified": a random stratified split (kept for comparison only).
    """
    strategy = test_set_config.get("strategy", "temporal")
    size = test_set_config["size"]

    if strategy == "temporal":
        dates = extras[date_column]
        cutoff = dates.quantile(1 - size)
        is_test = (dates > cutoff).to_numpy()
        is_dev = (dates <= cutoff - pd.Timedelta(days=test_set_config.get("gap_days", 0))).to_numpy()
        return X[is_dev], X[is_test], y[is_dev], y[is_test], extras[is_dev], extras[is_test]

    if strategy == "stratified":
        return train_test_split(
            X, y, extras, test_size=size, random_state=test_set_config["random_state"], stratify=y,
        )
    raise ValueError(f"Unknown test_set strategy: {strategy}. Options: ['temporal', 'stratified']")


def task_preprocessing_config(config: dict, task: str) -> dict:
    """The shared `preprocessing` section, overridden by `<task>.preprocessing` (each task
    keeps the encoder/scaler pair, imputation and extra features its own comparisons
    chose). `extra_numeric_features` / `extra_categorical_features` are appended to the
    shared feature lists; `imputation` keys override the shared ones."""
    shared = config["preprocessing"]
    override = dict(config[task].get("preprocessing") or {})
    merged = {**shared, **override}
    merged["numeric_features"] = shared["numeric_features"] + override.pop("extra_numeric_features", [])
    merged["categorical_features"] = shared["categorical_features"] + override.pop("extra_categorical_features", [])
    merged["imputation"] = {**shared.get("imputation", {}), **override.get("imputation", {})}
    merged.pop("extra_numeric_features", None)
    merged.pop("extra_categorical_features", None)
    return merged


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
# Each entry takes a random_state (only the target encoder uses it). sklearn's TargetEncoder
# cross-fits during fit -- a training row is encoded with category means from the OTHER
# internal folds, never its own label -- and with target_type="auto" it works for both the
# binary (classification) and the continuous (regression) target.
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(target_type="auto", cv=5, shuffle=True, random_state=seed),
}


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Fitted step: learns, from the training rows only, which categories of each column have
    at least `min_count` rows; every other category -- rare in training, or never seen --
    becomes `label` at transform time. `min_count=0` disables it.
    """

    def __init__(self, min_count: int = 0, label: str = "rare"):
        self.min_count = min_count
        self.label = label

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.frequent_ = [set(X[c].value_counts()[lambda s: s >= self.min_count].index) for c in X.columns]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.min_count:
            for c, frequent in zip(X.columns, self.frequent_):
                X[c] = X[c].where(X[c].isin(frequent), self.label)
        return X.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _numeric_imputer(imputation: dict):
    strategy = imputation.get("numeric_strategy", "median")
    if strategy == "knn":
        return KNNImputer(n_neighbors=imputation.get("n_neighbors", 5))
    if strategy == "iterative":
        return IterativeImputer(max_iter=imputation.get("max_iter", 10), random_state=imputation.get("random_state", 0))
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=imputation.get("numeric_fill_value", 0))
    return SimpleImputer(strategy=strategy)


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Factory: the leak-safe ColumnTransformer described by config.yaml's `preprocessing`
    section. Nothing is fitted here -- it is fitted later, inside the model's Pipeline, on
    training rows only. Every encoder tolerates categories never seen in training.

    Numeric imputation: "median" / "mean" / "constant" impute then scale; "knn" and
    "iterative" scale first, so distances / regressions are not dominated by the
    large-valued columns. Categorical: impute, group rare categories, encode.
    """
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    flag_features = preprocessing_config.get("flag_features", [])
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))

    imputer = _numeric_imputer(imputation)
    if imputation.get("numeric_strategy", "median") in ("knn", "iterative"):
        numeric_pipeline = Pipeline([("scale", scaler), ("impute", imputer)])
    else:
        numeric_pipeline = Pipeline([("impute", imputer), ("scale", scaler)])

    categorical_strategy = imputation.get("categorical_strategy", "most_frequent")
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(
            strategy=categorical_strategy,
            fill_value=imputation.get("categorical_fill_value", "Missing") if categorical_strategy == "constant" else None,
        )),
        ("group_rare", RareCategoryGrouper(min_count=preprocessing_config.get("rare_category_min_count", 0))),
        ("encode", encoder),
    ])

    passthrough = flag_features + [f"{c}_was_missing" for c in mnar_indicator_sources]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("flags", "passthrough", passthrough),
    ])
