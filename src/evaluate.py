"""
Evaluation shared by both tasks: temporal cross-validation folds on the development set,
cross-validated scoring of a full Pipeline (preprocessing + model), the encoder x scaler
comparison, and the paired comparison of two recipes.

The locked test set never enters any function here.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

# sklearn scorer names per task; scores are "greater is better" (errors come negated)
SCORERS = {
    "classification": {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"},
    "regression": {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
}


def temporal_folds(dates: pd.Series, n_splits: int, gap_days: int = 0):
    """
    Expanding-window folds: the development rows are ordered by date and cut into
    `n_splits + 1` consecutive blocks; fold k trains on blocks 0..k and validates on block
    k + 1 -- always predicting the future from the past, like test.csv.

    `gap_days` drops training rows from the last days before each validation block: their
    90-day labels are measured inside the validation period, so keeping them would let the
    label of a training row "see" the period being scored.

    Returns a list of (train_positions, validation_positions), usable as `cv=` in sklearn.
    """
    order = np.argsort(dates.to_numpy(), kind="stable")
    blocks = np.array_split(order, n_splits + 1)
    values = dates.to_numpy()
    folds = []
    for k in range(1, n_splits + 1):
        validation = blocks[k]
        start = values[validation].min()
        train = np.concatenate(blocks[:k])
        train = train[values[train] < start - np.timedelta64(gap_days, "D")]
        folds.append((np.sort(train), np.sort(validation)))
    return folds


def make_pipeline(preprocessing_config: dict, model) -> Pipeline:
    """Preprocessing + model as one estimator: `fit` learns everything from training rows only."""
    return Pipeline([("prep", build_preprocessor(preprocessing_config)), ("model", model)])


def cross_validate_pipeline(pipeline, X, y, folds, task: str) -> pd.DataFrame:
    """One row per fold with train and validation scores for every metric of the task.
    Error metrics are reported positive (RMSE, MAE)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = cross_validate(pipeline, X, y, cv=folds, scoring=SCORERS[task], return_train_score=True)
    rows = {}
    for metric, scorer in SCORERS[task].items():
        sign = -1 if scorer.startswith("neg_") else 1
        rows[f"train_{metric}"] = sign * result[f"train_{metric}"]
        rows[f"validation_{metric}"] = sign * result[f"test_{metric}"]
    return pd.DataFrame(rows).rename_axis("fold")


def summarize(fold_scores: pd.DataFrame) -> pd.Series:
    """Mean and std over folds of every column."""
    return pd.concat([fold_scores.mean().add_suffix("_mean"), fold_scores.std(ddof=1).add_suffix("_std")])


def compare_encoders_scalers(preprocessing_config, model_factory, X, y, folds, task, metric,
                             encoders, scalers) -> tuple:
    """
    Scores every encoder x scaler pair on the SAME folds. Returns
    (ranking table, {(encoder, scaler): per-fold validation scores}) -- the per-fold scores
    make a paired comparison possible.
    """
    higher_is_better = not SCORERS[task][metric].startswith("neg_")
    per_fold, rows = {}, []
    for encoder, scaler in itertools.product(encoders, scalers):
        config = {**preprocessing_config, "encoder": encoder, "scaler": scaler}
        scores = cross_validate_pipeline(make_pipeline(config, model_factory()), X, y, folds, task)
        per_fold[(encoder, scaler)] = scores[f"validation_{metric}"].to_numpy()
        rows.append({"encoder": encoder, "scaler": scaler,
                     f"mean_{metric}": scores[f"validation_{metric}"].mean(),
                     f"std_{metric}": scores[f"validation_{metric}"].std(ddof=1)})
    ranking = pd.DataFrame(rows).sort_values(f"mean_{metric}", ascending=not higher_is_better).reset_index(drop=True)
    return ranking, per_fold


def paired_comparison(scores_a, scores_b, higher_is_better: bool = True) -> dict:
    """
    Same folds -> subtract fold by fold. A mean difference more than 2 standard errors
    from zero is a real edge; within 2 SE it could just be the luck of the folds.
    """
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    if not higher_is_better:
        diffs = -diffs                      # positive = A is better, for every metric
    mean, se = diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(diffs))
    return {"mean_advantage": mean, "se": se, "folds_a_better": int((diffs > 0).sum()),
            "n_folds": len(diffs), "decisive": bool(abs(mean) > 2 * se)}
