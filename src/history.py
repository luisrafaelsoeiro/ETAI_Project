"""Fold-local purchase history. Query batches never become a history source."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.cleaning import fill_from_customer_history, prepare_table
from src.data import load_data


class CustomerHistoryFiller(TransformerMixin, BaseEstimator):
    """Freeze prepared train.csv context at the last date passed to fit.

    Earlier unlabelled purchases/non-returners remain usable for regression: the
    reference file supplies predictors, never targets. The temporal training cutoff
    excludes the embargo and validation period, including in nested CV. Dates/IDs
    remain context columns; the subsequent ColumnTransformer selects predictors only.
    Loading happens at fit; prediction uses the stored target-free history only.
    """

    def __init__(self, config):
        self.config = config

    def fit(self, X, y=None):
        cfg = self.config
        rules = cfg["diagnostics"]["customer_history"]
        date = rules["date_column"]
        if X.empty or X[date].isna().any():
            raise ValueError("History fitting requires non-missing training dates.")
        self.cutoff_ = X[date].max()
        raw = load_data(cfg["data"]["train_path"], cfg["diagnostics"].get("placeholder_tokens"))
        source, _ = prepare_table(raw, cfg)
        columns = list(dict.fromkeys([
            cfg["data"]["id_column"], rules["key"], date, rules["order_column"],
            *rules.get("last_observed_columns", []),
            *([rules["previous_purchase_gap"]] if rules.get("previous_purchase_gap") else []),
        ]))
        self.history_ = source.loc[source[date] <= self.cutoff_, columns].copy()
        self.feature_names_in_ = X.columns.to_numpy()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "history_")
        return fill_from_customer_history(
            X, self.history_, self.config["diagnostics"], self.config["data"]["id_column"],
        )[0]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else input_features
