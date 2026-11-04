"""
Classification task (will the customer buy again within 90 days?): turns the cleaned
training table into (X, y, extras) for `repeat_purchase_90d`. The shared rules live in
src/preprocessing.py.
"""
import pandas as pd

from src.preprocessing import split_features_target


def select_training_rows(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Every cleaned training row has a classification label -- nothing to filter."""
    target = config["classification"]["target"]
    if df[target].isna().any():
        raise ValueError(f"{target} has missing labels; they must never be imputed.")
    return df


def prepare_features(df: pd.DataFrame, config: dict):
    """(X, y, extras) for the classifier. y is None on test.csv."""
    return split_features_target(
        df, config["data"], config["classification"]["target"],
        config["preprocessing"].get("mnar_indicator_sources", []),
    )
