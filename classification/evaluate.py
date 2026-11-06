"""
What is specific to evaluating the classifier (will the customer buy again within 90 days?):
its metrics, what the reports predict (the probability of a repurchase), and the reports on
the out-of-fold and locked-test predictions. The cross-validation itself is shared
(src/evaluate.py).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# metrics that config.yaml -> classification.metrics may list: {name: sklearn scoring string}
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def predict_for_report(model, X) -> np.ndarray:
    """Probability of a repurchase (class 1); labels when the model has no probabilities."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(float)


def _prediction_report(y_true, probability, title: str, threshold: float) -> str:
    y_true = pd.Series(y_true).astype(int).to_numpy()
    y_pred = (np.asarray(probability) >= threshold).astype(int)
    matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0, 1]),
                          index=["true 0 (no repurchase)", "true 1 (repurchase)"],
                          columns=["predicted 0", "predicted 1"])
    auc = roc_auc_score(y_true, probability) if len(np.unique(y_true)) == 2 else float("nan")
    return (f"{title} ({len(y_true)} rows, threshold {threshold}):\n"
            f"  accuracy: {accuracy_score(y_true, y_pred):.4f}   roc_auc: {auc:.4f}   "
            f"repurchase rate: true {y_true.mean():.3f} / predicted {y_pred.mean():.3f}\n\n"
            f"{matrix.to_string()}\n\n{classification_report(y_true, y_pred, zero_division=0)}")


def oof_report(y_true, oof_probability: pd.Series, threshold: float = 0.5) -> str:
    """Confusion matrix and classification report of the out-of-fold predictions (the rows
    some fold validated -- temporal folds never validate the first block)."""
    validated = oof_probability.notna()
    return _prediction_report(pd.Series(y_true)[validated.to_numpy()], oof_probability[validated],
                              "Out-of-fold predictions, development set", threshold)


def holdout_report(y_true, probability, threshold: float = 0.5) -> str:
    """The same report on the locked test set, evaluated once."""
    return _prediction_report(y_true, probability, "Locked test set, evaluated once", threshold)
