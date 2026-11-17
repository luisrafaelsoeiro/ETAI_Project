"""
Hyperparameter tuning with Optuna, and nested cross-validation, shared by both tasks.

Tuning uses the development set only and scores every candidate by temporal cross-validation
of the whole pipeline on the task's selection metric. The best trial's score is the maximum of
many noisy scores, so it is optimistic; the honest estimate of "tune, then refit" comes from
`nested_cross_validate`: every outer fold is tuned on its own training rows (with inner
temporal folds) and scored on its validation rows, which took no part in the tuning.

Search spaces live in config.yaml (`<task>.tuning.search_spaces.<model type>`), keyed by
Pipeline parameter names (`model__max_depth`).
"""
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

from src.evaluate import greater_is_better, temporal_folds


def resolve_search_space(pipeline, search_space: dict) -> dict:
    """
    Adapts parameter names to the pipeline: when the model is wrapped (a regressor inside a
    TransformedTargetRegressor), `model__max_depth` becomes `model__regressor__max_depth`.
    Raises when a name matches neither.
    """
    valid = pipeline.get_params(deep=True)
    resolved = {}
    for name, spec in search_space.items():
        wrapped = name.replace("model__", "model__regressor__", 1)
        if name in valid:
            resolved[name] = spec
        elif wrapped in valid:
            resolved[wrapped] = spec
        else:
            raise ValueError(f"Search-space parameter '{name}' is not a parameter of the pipeline.")
    return resolved


def suggest_params(trial, search_space: dict) -> dict:
    """One value per search-space entry (`type`: int / float / categorical)."""
    params = {}
    for name, spec in search_space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search-space type for {name}: {spec['type']}. Options: int, float, categorical")
    return params


def tune_pipeline(pipeline, X, y, folds, scoring: str, search_space: dict, n_trials: int, random_state: int):
    """
    Runs an Optuna study (seeded TPE sampler) on (X, y): each trial sets one draw of
    `search_space` on a fresh copy of `pipeline` and returns its mean CV score over `folds`
    with `scoring` (an sklearn scoring string, maximised: errors are negated).

    Returns (best_pipeline, study): an unfitted copy of `pipeline` with the best trial's
    hyperparameters, and the study with every trial (mean score and fold std).
    """
    search_space = resolve_search_space(pipeline, search_space)

    def objective(trial):
        candidate = clone(pipeline).set_params(**suggest_params(trial, search_space))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            scores = cross_val_score(candidate, X, y, cv=folds, scoring=scoring)
        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    return clone(pipeline).set_params(**study.best_params), study


def nested_cross_validate(pipeline, X, y, dates: pd.Series, outer_folds, inner_splits: int, gap_days: int,
                          scoring: dict, selection_metric: str, search_space: dict, n_trials: int,
                          random_state: int, predict=None):
    """
    Honest estimate of the tuning procedure. For every outer fold: tune on the outer-training
    rows (inner temporal folds built from their dates, same gap), refit the best pipeline on
    them, and score it on the outer-validation rows on every metric in `scoring`.

    Returns (fold_scores, oof) like `cross_validate_with_oof`: `train_<metric>` and
    `validation_<metric>` columns (errors positive), plus `inner_best` (the tuning score inside
    that fold, same sign) and the hyperparameters chosen in each fold.
    """
    rows = []
    oof = pd.Series(np.nan, index=X.index, dtype=float)
    selection_sign = 1 if greater_is_better(scoring[selection_metric]) else -1
    for train, validation in outer_folds:
        X_train, y_train = X.iloc[train], y.iloc[train]
        inner = temporal_folds(dates.iloc[train], inner_splits, gap_days)
        best, study = tune_pipeline(pipeline, X_train, y_train, inner, scoring[selection_metric],
                                    search_space, n_trials, random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            best.fit(X_train, y_train)
        row = {}
        for metric, name in scoring.items():
            scorer, sign = get_scorer(name), (1 if greater_is_better(name) else -1)
            row[f"train_{metric}"] = sign * scorer(best, X_train, y_train)
            row[f"validation_{metric}"] = sign * scorer(best, X.iloc[validation], y.iloc[validation])
        row["inner_best"] = selection_sign * study.best_value
        rows.append({**row, **study.best_params})
        if predict is not None:
            oof.iloc[validation] = predict(best, X.iloc[validation])
    return pd.DataFrame(rows).rename_axis("fold"), (oof if predict is not None else None)


def tuning_report(study, nested_scores: pd.DataFrame, scoring: dict, selection_metric: str, top: int = 5) -> str:
    """
    Best hyperparameters and top trials of the final study (on the whole development set),
    plus the optimism check (tuning score inside the outer folds vs the honest outer score),
    as text.
    """
    sign = 1 if greater_is_better(scoring[selection_metric]) else -1
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials = trials.rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", ""))
    trials = trials.rename(columns={"number": "trial", "value": selection_metric, "std": "fold_std"})
    trials[selection_metric] = sign * trials[selection_metric]
    top_trials = trials.dropna(subset=[selection_metric]).sort_values(selection_metric, ascending=sign < 0).head(top)

    inner = nested_scores["inner_best"].mean()
    outer = nested_scores[f"validation_{selection_metric}"]
    return "\n".join([
        f"Optuna tuning on the whole development set ({len(trials)} trials, metric: {selection_metric})",
        f"Best trial: {study.best_trial.number}   best mean CV {selection_metric}: {sign * study.best_value:.4f}"
        "   (the best of many noisy scores -- optimistic)",
        f"Best hyperparameters: {study.best_params}",
        "",
        f"Top {len(top_trials)} trials:",
        top_trials.to_string(index=False, float_format=lambda v: f"{v:.4g}"),
        "",
        "Optimism check (nested CV):",
        f"  tuning score inside the outer folds (inner_best) mean = {inner:.4f}",
        f"  honest score on the outer folds (validation)      mean = {outer.mean():.4f} ± {outer.std(ddof=1):.4f}",
        f"  optimism (inner - outer, in the metric's direction)  = {sign * (inner - outer.mean()):+.4f}",
        "Report the outer validation score as the tuned model's performance.",
    ])


def nested_cv_report(nested_scores: pd.DataFrame, metrics: list) -> str:
    """The hyperparameters chosen in each outer fold, as text."""
    parameters = [c for c in nested_scores.columns
                  if not c.startswith(("train_", "validation_")) and c != "inner_best"]
    if not parameters:
        return "No hyperparameters were recorded."
    return "Hyperparameters chosen in each outer fold:\n" + nested_scores[parameters].to_string(
        float_format=lambda v: f"{v:.4g}")
