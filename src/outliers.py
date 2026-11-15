"""Optional numeric transforms, fitted after imputation/bounds and before scaling."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SkewLog1p(TransformerMixin, BaseEstimator):
    def __init__(self, columns, minimum_skew=2):
        self.columns = columns
        self.minimum_skew = minimum_skew

    def fit(self, X, y=None):
        frame = pd.DataFrame(np.asarray(X), columns=self.columns)
        self.columns_ = [c for c in self.columns
                         if frame[c].min() >= 0 and abs(frame[c].skew()) > self.minimum_skew]
        self.positions_ = [list(self.columns).index(c) for c in self.columns_]
        return self

    def transform(self, X):
        check_is_fitted(self, "positions_")
        result = np.asarray(X, dtype=float).copy()
        if np.any(result[:, self.positions_] < 0):
            raise ValueError("log1p experiment expects non-negative values in selected columns.")
        result[:, self.positions_] = np.log1p(result[:, self.positions_])
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns if input_features is None else input_features)


class Winsorizer(TransformerMixin, BaseEstimator):
    def __init__(self, columns, lower=0.01, upper=0.99):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower <= self.upper <= 1:
            raise ValueError("Winsorisation quantiles must satisfy 0 <= lower <= upper <= 1.")
        self.lower_limits_ = np.quantile(X, self.lower, axis=0)
        self.upper_limits_ = np.quantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_limits_")
        return np.clip(np.asarray(X, dtype=float), self.lower_limits_, self.upper_limits_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns if input_features is None else input_features)
