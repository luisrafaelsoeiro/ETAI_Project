"""
From a cleaned table to a model-ready matrix, shared by both tasks:

  - `split_features_target`: (X, y, extras), with y None on data without labels;
  - `split_dev_test`: the locked temporal test set;
  - `build_preprocessor`: the fitted recipe -- imputation, missing indicators, rare-category
    grouping, encoding and scaling in one ColumnTransformer. It lives inside the model's
    Pipeline (src/pipeline.py), so it is fitted on the training rows of every fold only.

Feature lists and choices come from config.yaml (`preprocessing` and `<task>.preprocessing`).
"""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 -- makes IterativeImputer importable
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)
from category_encoders import CountEncoder


def split_features_target(df: pd.DataFrame, data_config: dict, target: str):
    """
    Returns (X, y, extras).
      - y is None when the target column is absent (test.csv) -- nothing needs it.
      - extras keeps id / customer_id / purchase_date row-aligned with X: needed for the
        temporal split, grouping and the submission file, never as model inputs.
      - every target column is removed from X, so neither task can see the other's label.
    Missingness indicators are not added here: the `indicators` step of
    `build_preprocessor` creates them inside the pipeline.
    """
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


def _referenced_names(expression: str) -> set:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(expression)))


class RowBoundedImputer(BaseEstimator, TransformerMixin):
    """
    Fitted step: imputes the numeric columns with `imputer`, then keeps every IMPUTED value
    within the limits its own row implies (`bounds`, e.g. 365-day spend between the 180-day
    spend and all prior spend). A single median ignores the rest of the row and can produce
    impossible values; observed values are never changed.

    `bounds`: {column: {"min": expression, "max": expression}}, evaluated on the imputed row
    with DataFrame.eval; a limit that can't be computed for a row (NaN) is skipped, and `max`
    wins if the two limits cross. `scaler` (KNN / iterative imputation): impute in scaled space,
    so large-valued columns don't dominate distances, then return to the original units so the
    limits can be compared -- the pipeline's own scaler follows this step.
    """

    def __init__(self, imputer, feature_names, bounds=None, scaler=None):
        self.imputer = imputer
        self.feature_names = feature_names
        self.bounds = bounds
        self.scaler = scaler

    def _frame(self, X) -> pd.DataFrame:
        frame = pd.DataFrame(X).astype("float64")
        frame.columns = list(self.feature_names)
        return frame

    def fit(self, X, y=None):
        X = self._frame(X)
        self.scaler_ = clone(self.scaler).fit(X) if self.scaler is not None else None
        self.imputer_ = clone(self.imputer).fit(X if self.scaler_ is None else self.scaler_.transform(X))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._frame(X)
        missing = X.isna()
        filled = self.imputer_.transform(X if self.scaler_ is None else self.scaler_.transform(X))
        if self.scaler_ is not None:
            filled = self.scaler_.inverse_transform(filled)
        out = pd.DataFrame(filled, columns=X.columns, index=X.index).where(missing, X)  # observed: exact
        for column, limits in (self.bounds or {}).items():
            for side in ("min", "max"):
                if side not in limits:
                    continue
                limit = out.eval(str(limits[side]))
                beyond = out[column] < limit if side == "min" else out[column] > limit
                beyond &= missing[column] & limit.notna()
                out.loc[beyond, column] = limit[beyond]
        return out.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names, dtype=object)


def _numeric_imputer(imputation: dict):
    strategy = imputation.get("numeric_strategy", "median")
    if strategy == "knn":
        return KNNImputer(n_neighbors=imputation.get("n_neighbors", 5))
    if strategy == "iterative":
        return IterativeImputer(max_iter=imputation.get("max_iter", 10), random_state=imputation.get("random_state", 0))
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=imputation.get("numeric_fill_value", 0))
    return SimpleImputer(strategy=strategy)


def check_feature_lists(preprocessing_config: dict, available_columns=None) -> None:
    """
    Raises ValueError when the feature lists of a preprocessing config are inconsistent:
      - a feature listed twice, or in more than one group (numeric / categorical / flags);
      - a missingness-indicator source that is not itself a numeric or categorical feature;
      - (when `available_columns` is given) a listed feature that doesn't exist in the data.
    """
    groups = {
        "numeric_features": list(preprocessing_config["numeric_features"]),
        "categorical_features": list(preprocessing_config["categorical_features"]),
        "flag_features": list(preprocessing_config.get("flag_features", [])),
    }
    problems = []
    for name, features in groups.items():
        repeated = sorted({f for f in features if features.count(f) > 1})
        if repeated:
            problems.append(f"{name} lists {repeated} more than once")
    names = list(groups)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            overlap = sorted(set(groups[first]) & set(groups[second]))
            if overlap:
                problems.append(f"{overlap} are in both {first} and {second}")
    indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    orphan = sorted(set(indicator_sources) - set(groups["numeric_features"]) - set(groups["categorical_features"]))
    if orphan:
        problems.append(f"mnar_indicator_sources {orphan} are not numeric or categorical features")
    for column, limits in preprocessing_config.get("imputation", {}).get("row_bounds", {}).items():
        names = {column}.union(*(_referenced_names(e) for e in limits.values()))
        outside = sorted(names - set(groups["numeric_features"]))
        if outside:
            problems.append(f"row_bounds of {column} use {outside}, which are not numeric features")
    if available_columns is not None:
        absent = sorted(set().union(*groups.values()) - set(available_columns))
        if absent:
            problems.append(f"features not found in the data: {absent}")
    if problems:
        raise ValueError("Invalid preprocessing feature lists: " + "; ".join(problems))


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Factory: the leak-safe ColumnTransformer described by config.yaml's `preprocessing`
    section. Nothing is fitted here -- it is fitted later, inside the model's Pipeline, on
    training rows only. Every encoder tolerates categories never seen in training.

    Numeric: `RowBoundedImputer` (impute -- "median" / "mean" / "constant" directly, "knn" /
    "iterative" in scaled space so distances / regressions are not dominated by the
    large-valued columns -- then keep imputed values within their row's limits,
    `imputation.row_bounds`), then scale. Categorical: impute, group rare categories, encode.
    Indicators: `MissingIndicator` on the raw (not yet imputed) MNAR/MAR columns.
    """
    check_feature_lists(preprocessing_config)
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    flag_features = preprocessing_config.get("flag_features", [])
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))

    scale_before_imputing = imputation.get("numeric_strategy", "median") in ("knn", "iterative") and scaler != "passthrough"
    numeric_pipeline = Pipeline([
        ("impute", RowBoundedImputer(
            _numeric_imputer(imputation), list(numeric_features), imputation.get("row_bounds", {}),
            scaler=clone(scaler) if scale_before_imputing else None,
        )),
        ("scale", scaler),
    ])

    categorical_strategy = imputation.get("categorical_strategy", "most_frequent")
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(
            strategy=categorical_strategy,
            fill_value=imputation.get("categorical_fill_value", "Missing") if categorical_strategy == "constant" else None,
        )),
        ("group_rare", RareCategoryGrouper(min_count=preprocessing_config.get("rare_category_min_count", 0))),
        ("encode", encoder),
    ])

    # every transformer sees the ORIGINAL columns, so the indicators see the gaps before
    # the numeric/categorical imputers fill them
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", MissingIndicator(features="all"), mnar_indicator_sources),
        ("flags", "passthrough", flag_features),
    ])
