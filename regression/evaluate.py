"""
What is specific to evaluating the regressor (how much will a returning customer spend in the
next 90 days?): its metrics, what the reports predict (the spend in euros), and the reports on
the out-of-fold and locked-test predictions -- overall errors, bias, and errors by spend band.
The cross-validation itself is shared (src/evaluate.py).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# metrics that config.yaml -> regression.metrics may list: {name: sklearn scoring string}
# (errors are negated by sklearn so that greater is better; reports show them positive)
SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "median_ae": "neg_median_absolute_error",
    "r2": "r2",
}


def predict_for_report(model, X) -> np.ndarray:
    """Predicted spend in euros."""
    return model.predict(X)


def _prediction_report(y_true, y_pred, title: str) -> str:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    bands = pd.qcut(y_true, 4, labels=["lowest 25%", "25-50%", "50-75%", "highest 25%"])
    by_band = (pd.DataFrame({"true spend band": bands, "true": y_true, "error": errors})
               .groupby("true spend band", observed=True)
               .agg(rows=("true", "size"), mean_true=("true", "mean"), mean_error=("error", "mean"),
                    mae=("error", lambda e: e.abs().mean())))
    return (f"{title} ({len(y_true)} rows):\n"
            f"  rmse: {root_mean_squared_error(y_true, y_pred):.1f}   mae: {mean_absolute_error(y_true, y_pred):.1f}   "
            f"r2: {r2_score(y_true, y_pred):.4f}   mean error (bias): {errors.mean():+.1f}\n\n"
            f"Errors by true spend band (mean_error > 0: over-prediction):\n"
            f"{by_band.to_string(float_format=lambda v: f'{v:.1f}')}\n")


def oof_report(y_true, oof_prediction: pd.Series) -> str:
    """Errors of the out-of-fold predictions (the rows some fold validated -- temporal folds
    never validate the first block)."""
    validated = oof_prediction.notna()
    return _prediction_report(pd.Series(y_true)[validated.to_numpy()], oof_prediction[validated],
                              "Out-of-fold predictions, development set")


def holdout_report(y_true, y_pred) -> str:
    """The same report on the locked test set, evaluated once."""
    return _prediction_report(y_true, y_pred, "Locked test set, evaluated once")
