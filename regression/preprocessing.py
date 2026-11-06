"""
Regression task (how much will a returning customer spend in the next 90 days?): turns
the cleaned training table into (X, y, extras) for `spend_next_90d`. The shared rules live
in src/preprocessing.py.
"""
import pandas as pd

from src.preprocessing import split_features_target


def select_training_rows(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Training-data decision: only repeat purchasers with an observed spend. Non-repeaters
    have no spend to learn from (100% missing), and the target is never imputed. The
    regression preprocessor is therefore fitted on these rows only.
    """
    return df.query(config["regression"]["row_filter"], engine="python").copy()


def prepare_features(df: pd.DataFrame, config: dict):
    """(X, y, extras) for the regressor. y is None on test.csv."""
    return split_features_target(
        df, config["data"], config["regression"]["target"],
    )
