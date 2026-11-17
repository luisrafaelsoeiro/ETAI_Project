"""
What is specific to evaluating the regressor (how much will the customer spend in the next
90 days?): its metrics, what the reports predict (spend in EUR), and the reports on the
out-of-fold and locked-test predictions. The cross-validation itself is shared
(src/evaluate.py).

All reports are computed on customers who actually returned, in euros.

"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error

# metrics that config.yaml -> regression.metrics may list: {name: sklearn scoring string}
# (errors are negated by sklearn so that greater is better; reports show them positive)
SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "median_ae": "neg_median_absolute_error",
    "r2": "r2",
}


def predict_for_report(model, X) -> np.ndarray:
    """Predicted spend in euros. """
    return np.clip(np.asarray(model.predict(X), dtype=float), 0.0, None)

def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, median AE, R2 and mean error (bias: >0 means over-prediction)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }

def _prediction_report(y_true, prediction, title: str) -> str:
    y_true = pd.Series(y_true).astype(float).reset_index(drop=True)
    prediction = pd.Series(np.asarray(prediction, dtype=float))
    m = regression_metrics(y_true, prediction)

    # error by spend band: shows whether the model is only good on small or on large spenders
    bands = pd.qcut(y_true, q=4, duplicates="drop")
    by_band = (pd.DataFrame({"band": bands, "abs_error": (prediction - y_true).abs(),
                             "error": prediction - y_true})
               .groupby("band", observed=True)
               .agg(rows=("abs_error", "size"), mae=("abs_error", "mean"), bias=("error", "mean")))

    return (f"{title} ({len(y_true)} rows, EUR):\n"
            f"  rmse: {m['rmse']:.2f}   mae: {m['mae']:.2f}   median_ae: {m['median_ae']:.2f}   "
            f"r2: {m['r2']:.4f}   bias: {m['bias']:+.2f}\n"
            f"  spend: true mean {y_true.mean():.2f} / predicted mean {prediction.mean():.2f}\n\n"
            f"Error by true-spend quartile:\n{by_band.round(2).to_string()}\n")


def oof_report(y_true, oof_prediction: pd.Series) -> str:
    """Metrics of the out-of-fold predictions (the rows some fold validated -- temporal
    folds never validate the first block)."""
    validated = oof_prediction.notna()
    return _prediction_report(pd.Series(y_true)[validated.to_numpy()], oof_prediction[validated],
                              "Out-of-fold predictions, development set")


def holdout_report(y_true, y_pred) -> str:
    """The same report on the locked test set, evaluated once."""
    return _prediction_report(y_true, y_pred, "Locked test set, evaluated once")
