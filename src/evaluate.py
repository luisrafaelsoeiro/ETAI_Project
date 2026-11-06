"""
Evaluation shared by both tasks, on the development set only (the locked test set never
enters any function here):

  - temporal cross-validation folds (expanding window, with a gap);
  - cross-validated scoring of a full Pipeline on several metrics, with out-of-fold
    predictions for the reports;
  - the encoder x scaler comparison and other preprocessing comparisons on the same folds;
  - the paired comparison of two options, fold by fold.

Metrics are passed in as `scoring`: {name: sklearn scoring string}, chosen per task in
config.yaml (`<task>.metrics`) from the task's `evaluate.SCORING`. Scores are reported
"natural side up": errors such as RMSE come out positive.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate

from src.pipeline import build_pipeline


def select_scoring(available: dict, metrics: list) -> dict:
    """{name: sklearn scoring string} for the metrics listed in config.yaml, checked
    against the ones a task supports (`<task>.evaluate.SCORING`)."""
    unknown = [m for m in metrics if m not in available]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}. Options: {list(available)}")
    return {m: available[m] for m in metrics}


def greater_is_better(scoring: str) -> bool:
    """sklearn error scorers are negated ("neg_..."): lower is better for the metric itself."""
    return not scoring.startswith("neg_")


def temporal_folds(dates: pd.Series, n_splits: int, gap_days: int = 0):
    """
    Expanding-window folds: the development rows are ordered by date and cut into
    `n_splits + 1` consecutive blocks; fold k trains on blocks 0..k and validates on block
    k + 1 -- always predicting the future from the past, like test.csv.

    `gap_days` drops training rows from the last days before each validation block: their
    90-day labels are measured inside the validation period, so keeping them would let the
    label of a training row "see" the period being scored. A fold left without training rows
    is skipped.

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
        if len(train):
            folds.append((np.sort(train), np.sort(validation)))
    return folds


def cross_validate_with_oof(pipeline, X, y, folds, scoring: dict, predict=None):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training rows and
    scores it on that fold's validation rows, on every metric in `scoring`.

    Returns (fold_scores, oof):
      - fold_scores: one row per fold, `train_<metric>` and `validation_<metric>` columns
        (errors reported positive);
      - oof: out-of-fold predictions (`predict(model, X)`, e.g. a probability) for every
        validated row, from the fold model that did not train on it. Temporal folds never
        validate the first block, so those rows stay NaN. None when `predict` is None.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = cross_validate(pipeline, X, y, cv=folds, scoring=scoring, return_train_score=True,
                                return_estimator=predict is not None)
    rows = {}
    for metric, scorer in scoring.items():
        sign = 1 if greater_is_better(scorer) else -1
        rows[f"train_{metric}"] = sign * result[f"train_{metric}"]
        rows[f"validation_{metric}"] = sign * result[f"test_{metric}"]
    fold_scores = pd.DataFrame(rows).rename_axis("fold")

    if predict is None:
        return fold_scores, None
    oof = pd.Series(np.nan, index=X.index, dtype=float)
    for model, (_, validation) in zip(result["estimator"], folds):
        oof.iloc[validation] = predict(model, X.iloc[validation])
    return fold_scores, oof


def cross_validate_pipeline(pipeline, X, y, folds, scoring: dict) -> pd.DataFrame:
    """`cross_validate_with_oof` without the out-of-fold predictions: one row per fold."""
    return cross_validate_with_oof(pipeline, X, y, folds, scoring)[0]


def summarize(fold_scores: pd.DataFrame) -> pd.Series:
    """Mean and std over folds of every column."""
    return pd.concat([fold_scores.mean().add_suffix("_mean"), fold_scores.std(ddof=1).add_suffix("_std")])


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str,
              higher_is_better: bool = True) -> str:
    """Per-fold validation table, mean and std of every metric, and the overfitting gap of the
    selection metric (how much better the training score is than the validation score), as text."""
    validation = fold_scores[[f"validation_{m}" for m in metrics]].rename(columns=lambda c: c.replace("validation_", ""))
    gap = fold_scores[f"train_{selection_metric}"] - fold_scores[f"validation_{selection_metric}"]
    gap = gap if higher_is_better else -gap
    return "\n".join([
        f"{title} ({len(fold_scores)} temporal folds, selection metric: {selection_metric})",
        "",
        validation.to_string(float_format=lambda v: f"{v:.4f}"),
        "",
        validation.agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.4f}"),
        "",
        f"{selection_metric}: train mean = {fold_scores[f'train_{selection_metric}'].mean():.4f}, "
        f"overfitting gap (train better than validation by) = {gap.mean():+.4f}",
    ])


def compare_encoders_scalers(preprocessing_config, model_factory, X, y, folds, scoring: dict, metric: str,
                             encoders, scalers) -> tuple:
    """
    Scores every encoder x scaler pair on the SAME folds. Returns
    (ranking table, {(encoder, scaler): per-fold validation scores}) -- the per-fold scores
    make a paired comparison possible.
    """
    variants = {(encoder, scaler): {**preprocessing_config, "encoder": encoder, "scaler": scaler}
                for encoder, scaler in itertools.product(encoders, scalers)}
    table, per_fold = compare_preprocessing_variants(variants, model_factory, X, y, folds, scoring, metric)
    pairs = table.pop("variant")
    table.insert(0, "encoder", [encoder for encoder, _ in pairs])
    table.insert(1, "scaler", [scaler for _, scaler in pairs])
    return table, per_fold


def compare_preprocessing_variants(variants: dict, model_factory, X, y, folds, scoring: dict, metric: str) -> tuple:
    """
    Scores several preprocessing configurations (name -> preprocessing config) with the same
    model on the SAME folds. Returns (table, {name: per-fold validation scores}) for paired
    comparisons -- used for ablations (with vs without a feature group) and imputers.
    """
    per_fold, rows = {}, []
    for name, config in variants.items():
        scores = cross_validate_pipeline(build_pipeline(config, model_factory()), X, y, folds, scoring)
        per_fold[name] = scores[f"validation_{metric}"].to_numpy()
        rows.append({"variant": name, f"mean_{metric}": per_fold[name].mean(), f"std_{metric}": per_fold[name].std(ddof=1)})
    ascending = not greater_is_better(scoring[metric])
    table = pd.DataFrame(rows).sort_values(f"mean_{metric}", ascending=ascending).reset_index(drop=True)
    return table, per_fold


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
