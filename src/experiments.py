"""Controlled development-only preprocessing experiments, shared by notebooks 03/06.

python -m src.experiments --experiment both
No holdout scoring, model tuning, row removal or changes to the configured feature lists.
"""
import argparse
import copy
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluate import cross_validate_pipeline, greater_is_better, paired_comparison, select_scoring, temporal_folds
from src.history import CustomerHistoryFiller
from sklearn.ensemble import RandomForestClassifier

from src.pipeline import build_pipeline, prepare_task_data
from src.preprocessing import build_categorical_imputer, build_preprocessor
from src.tasks import TASKS


def categorical_history_variants(recipe):
    """2 x 2: last observed category on/off x mode/Missing fallback.

    The previous-purchase-date recovery stays on. Customer-profile categories keep
    their configured imputation, so only purchase-category handling changes.
    """
    columns = [c for c in recipe["customer_history"]["diagnostics"]["customer_history"]["last_observed_columns"]
               if c in recipe["categorical_features"]]
    variants = {}
    for history in [True, False]:
        for strategy, label in [("most_frequent", "mode"), ("constant", "Missing")]:
            candidate = copy.deepcopy(recipe)
            if not history:
                candidate["customer_history"]["diagnostics"]["customer_history"]["last_observed_columns"] = []
            overrides = candidate["imputation"].setdefault("categorical_overrides", {})
            overrides.update({c: {"strategy": strategy, "fill_value": "Missing"} for c in columns})
            variants[f"{'history' if history else 'no category history'} + {label}"] = candidate
    return variants, columns


def _evaluate_variants(config, task, data, variants, reference, regression_seeds):
    package = TASKS[task]
    metric = config[task]["selection_metric"]
    scoring = select_scoring(package.evaluate.SCORING, config[task]["metrics"])
    higher = greater_is_better(scoring[metric])
    folds = temporal_folds(data["extras_dev"][config["data"]["date_column"]],
                           config["cv"]["n_splits"], config["cv"]["gap_days"])
    seeds = [42] if task == "classification" else list(regression_seeds)
    rows, paired = [], []
    for seed in seeds:
        by_variant = {}
        for name, recipe in variants.items():
            model_cfg = copy.deepcopy(config[task]["model"])
            model_cfg.setdefault("params", {})["random_state"] = seed
            pipeline = build_pipeline(recipe, package.model.build_model(model_cfg))
            scores = cross_validate_pipeline(pipeline, data["X_dev"], data["y_dev"], folds, scoring)
            values = scores[f"validation_{metric}"].to_numpy()
            by_variant[name] = values
            for fold, score in scores.iterrows():
                rows.append({"task": task, "variant": name, "seed": seed, "fold": fold + 1,
                             "metric": metric, **score.to_dict()})
            print(f"{task}, seed {seed}, {name}: {metric} {values.mean():.6f}", flush=True)
        for name, values in by_variant.items():
            paired.append({"task": task, "variant": name, "reference": reference, "seed": seed,
                           **paired_comparison(values, by_variant[reference], higher)})
    scores = pd.DataFrame(rows)
    # Average repeated seeds WITHIN each temporal fold, not 15 independent folds.
    fold_means = scores.groupby(["variant", "fold"], sort=False)[f"validation_{metric}"].mean().unstack(0)
    summary = []
    for name in variants:
        values = fold_means[name].to_numpy()
        summary.append({"task": task, "variant": name, "metric": metric,
                        "mean_score": values.mean(), "fold_std": values.std(ddof=1), "seeds": len(seeds),
                        **paired_comparison(values, fold_means[reference].to_numpy(), higher)})
    return {"scores": scores, "paired_by_seed": pd.DataFrame(paired), "summary": pd.DataFrame(summary)}, folds


def _masked_category_reconstruction(data, variants, columns, folds, fraction=0.1, seed=42):
    """Same masked observed validation cells for every variant, fold-local fitting.

    Missing is a representation of uncertainty, not an attempt at exact reconstruction.
    Reconstruction on randomly hidden observed cells does not establish accuracy on the
    original gaps or prove a missingness mechanism.
    """
    rows = []
    X = data["X_dev"]
    for fold, (train, validation) in enumerate(folds, start=1):
        original = X.iloc[validation]
        masked = original.copy()
        rng = np.random.default_rng(seed + fold)
        masks = {}
        for column in columns:
            # Test reconstruction only for fields that actually have gaps in development.
            if not X[column].isna().any():
                continue
            candidates = original.index[original[column].notna()]
            if len(candidates) == 0:
                continue
            selected = rng.choice(candidates, size=max(1, int(len(candidates) * fraction)), replace=False)
            masks[column] = selected
            masked[column] = masked[column].astype(object)
            masked.loc[selected, column] = np.nan
        for name, recipe in variants.items():
            history = CustomerHistoryFiller(recipe["customer_history"])
            train_filled = history.fit_transform(X.iloc[train])
            queries_filled = history.transform(masked)
            categories = recipe["categorical_features"]
            imputer = build_categorical_imputer(recipe["imputation"], categories).fit(train_filled[categories])
            filled = pd.DataFrame(imputer.transform(queries_filled[categories]), columns=categories,
                                  index=queries_filled.index)
            for column, selected in masks.items():
                truth = original.loc[selected, column].astype(object)
                values = filled.loc[selected, column].astype(object)
                recovered = queries_filled.loc[selected, column].notna()
                rows.append({"task": data["task"], "variant": name, "fold": fold, "column": column,
                             "masked": len(selected), "correct": int(values.eq(truth).sum()),
                             "history_available": int(recovered.sum()),
                             "history_correct": int(queries_filled.loc[selected, column].astype(object).eq(truth).sum())})
    return pd.DataFrame(rows)


def categorical_history_experiment(config, task, regression_seeds=(42, 1, 2)):
    data = prepare_task_data(config, task)
    variants, columns = categorical_history_variants(data["preprocessing_config"])
    fallback = data["preprocessing_config"]["imputation"].get("categorical_strategy", "most_frequent")
    reference = "history + mode" if fallback == "most_frequent" else "history + Missing"
    tables, folds = _evaluate_variants(config, task, data, variants, reference, regression_seeds)
    reconstruction = _masked_category_reconstruction(data, variants, columns, folds)
    tables["reconstruction_by_fold"] = reconstruction
    pooled = reconstruction.groupby(["task", "variant", "column"], sort=False)[
        ["masked", "correct", "history_available", "history_correct"]].sum().reset_index()
    pooled["accuracy"] = pooled["correct"] / pooled["masked"]
    pooled["history_coverage"] = pooled["history_available"] / pooled["masked"]
    pooled["history_accuracy_when_available"] = pooled["history_correct"] / pooled["history_available"].replace(0, np.nan)
    tables["reconstruction"] = pooled
    return tables


def ticket_basis_audit(data, tolerance=0.05):
    """Verify the historical amount basis using development predictors only."""
    rows = data["X_dev"].sort_values(["customer_id", "purchase_date", "prior_purchase_count"])
    previous = rows.groupby("customer_id", observed=True).shift(1)
    consecutive = ((rows.prior_purchase_count - previous.prior_purchase_count).eq(1)
                   & rows.purchase_date.gt(previous.purchase_date))
    delta = rows.prior_total_spend - previous.prior_total_spend
    current_basis = previous.net_total + previous.total_tax
    checked = consecutive & delta.notna() & current_basis.notna() & previous.net_total.notna()
    return pd.DataFrame([{"scope": "classification development predictors only", "pairs": int(checked.sum()),
                          "tax_inclusive_matches": int((delta.sub(current_basis).abs().le(tolerance) & checked).sum()),
                          "net_only_matches": int((delta.sub(previous.net_total).abs().le(tolerance) & checked).sum()),
                          "tolerance_eur": tolerance}])


def ticket_basis_experiment(config, task, regression_seeds=(42, 1, 2)):
    data = prepare_task_data(config, task)
    X = data["X_dev"].copy()
    X["ticket_vs_usual_legacy"] = (X.net_total / X.prior_mean_purchase_value).replace([np.inf, -np.inf], np.nan)
    # Explicit here as well: replay remains controlled even if the config later changes.
    X["ticket_vs_usual"] = ((X.net_total + X.total_tax) / X.prior_mean_purchase_value).replace([np.inf, -np.inf], np.nan)
    data["X_dev"] = X
    recipe = data["preprocessing_config"]
    if {"ticket_vs_usual", "ticket_vs_usual_legacy"} & set(recipe["numeric_features"]):
        raise ValueError("Ticket experiment requires a no-ticket configured baseline.")
    variants = {"no ticket feature": recipe,
                "+ legacy net / historical mean": {**recipe, "numeric_features": recipe["numeric_features"] + ["ticket_vs_usual_legacy"]},
                "+ tax-inclusive ticket / historical mean": {**recipe, "numeric_features": recipe["numeric_features"] + ["ticket_vs_usual"]}}
    tables, _ = _evaluate_variants(config, task, data, variants, "no ticket feature", regression_seeds)
    # Direct corrected-vs-legacy comparison, not just each against absence.
    paired = []
    metric = config[task]["selection_metric"]
    higher = greater_is_better(TASKS[task].evaluate.SCORING[metric])
    for seed, scores in tables["scores"].groupby("seed"):
        pivot = scores.pivot(index="fold", columns="variant", values=f"validation_{metric}")
        paired.append({"task": task, "seed": seed, **paired_comparison(
            pivot["+ tax-inclusive ticket / historical mean"], pivot["+ legacy net / historical mean"],
            higher)})
    tables["corrected_vs_legacy"] = pd.DataFrame(paired)
    if task == "classification":
        tables["basis_audit"] = ticket_basis_audit(data)
    return tables


# ---------------------------------------------------------------- week 10: numeric imputers
# name -> imputation overrides; "median" is the reference (the configured fallback)
IMPUTER_VARIANTS = {
    "median": {"numeric_strategy": "median"},
    "knn": {"numeric_strategy": "knn"},
    "iterative": {"numeric_strategy": "iterative"},
    "random_forest": {"numeric_strategy": "random_forest"},
    "knn+categories": {"numeric_strategy": "knn", "categorical_helpers": True},
    "iterative+categories": {"numeric_strategy": "iterative", "categorical_helpers": True},
    "random_forest+categories": {"numeric_strategy": "random_forest", "categorical_helpers": True},
    "iterative x5": {"numeric_strategy": "iterative", "n_imputations": 5},
}


def imputer_recipes(recipe: dict, names=None) -> dict:
    """{variant: recipe} -- the task's recipe with only the numeric fallback imputer changed
    (row limits, `fit_rows` and `column_imputers` stay, as in the real pipeline)."""
    names = names or list(IMPUTER_VARIANTS)
    recipes = {}
    for name in names:
        candidate = copy.deepcopy(recipe)
        candidate["imputation"] = {**candidate["imputation"], "categorical_helpers": False, "n_imputations": 1,
                                   **IMPUTER_VARIANTS[name]}
        recipes[name] = candidate
    return recipes


def imputer_model_comparison(config, task, data=None, names=None, regression_seeds=(42, 1, 2)):
    """
    Temporal CV of the configured model with each numeric imputer (A1), with the imputer also
    seeing the categories (B2) and with multiple imputation (B1). Regression scores are averaged
    over model seeds WITHIN each fold. Every variant is compared with the median on the same
    folds: descriptively (`paired_comparison`) and with the corrected paired t-test
    (`corrected_paired_test`, an approximation on expanding windows). Returns
    {"scores": per variant / seed / fold, "summary": one row per variant}.
    """
    import time
    from src.evaluate import corrected_paired_test, fold_test_train_ratio
    data = data or prepare_task_data(config, task)
    package = TASKS[task]
    metric = config[task]["selection_metric"]
    scoring = select_scoring(package.evaluate.SCORING, config[task]["metrics"])
    higher = greater_is_better(scoring[metric])
    folds = temporal_folds(data["extras_dev"][config["data"]["date_column"]],
                           config["cv"]["n_splits"], config["cv"]["gap_days"])
    seeds = [42] if task == "classification" else list(regression_seeds)
    rows = []
    for name, recipe in imputer_recipes(data["preprocessing_config"], names).items():
        for seed in seeds:
            model_cfg = copy.deepcopy(config[task]["model"])
            model_cfg.setdefault("params", {})["random_state"] = seed
            start = time.perf_counter()
            scores = cross_validate_pipeline(build_pipeline(recipe, package.model.build_model(model_cfg)),
                                             data["X_dev"], data["y_dev"], folds, scoring)
            seconds = time.perf_counter() - start
            for fold, score in scores.iterrows():
                rows.append({"task": task, "variant": name, "seed": seed, "fold": fold + 1, "metric": metric,
                             "seconds_per_cv": seconds, **score.to_dict()})
            print(f"{task}, {name}, seed {seed}: {metric} {scores[f'validation_{metric}'].mean():.6f} "
                  f"({seconds:.0f}s)", flush=True)
    scores = pd.DataFrame(rows)
    by_fold = scores.groupby(["variant", "fold"], sort=False)[f"validation_{metric}"].mean().unstack(0)
    ratio = fold_test_train_ratio(folds)
    summary = []
    for name in by_fold.columns:
        values, reference = by_fold[name].to_numpy(), by_fold["median"].to_numpy()
        corrected = corrected_paired_test(values, reference, ratio, higher)
        summary.append({"task": task, "variant": name, "metric": metric, "mean_score": values.mean(),
                        "fold_std": values.std(ddof=1), "seeds": len(seeds),
                        "seconds_per_cv": scores.loc[scores["variant"] == name, "seconds_per_cv"].mean(),
                        **paired_comparison(values, reference, higher),
                        "corrected_t": corrected["t"], "corrected_p": corrected["p_value"],
                        "test_train_ratio": ratio})
    return {"scores": scores, "summary": pd.DataFrame(summary)}


def mask_numeric_values(X: pd.DataFrame, columns, rows, n: int, seed: int):
    """Hides up to `n` observed values per column among `rows` (positions in X). Returns
    (masked copy of X, {column: true values at the hidden index labels})."""
    rng = np.random.default_rng(seed)
    masked, truth = X.copy(), {}
    for column in columns:
        candidates = X.index[rows][X.iloc[rows][column].notna().to_numpy()]
        hidden = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        truth[column] = X.loc[hidden, column].astype("float64")
        masked.loc[hidden, column] = np.nan
    return masked, truth


def proximity_impute(train: pd.DataFrame, y_train, queries: pd.DataFrame, column: str,
                     n_estimators: int = 300, min_samples_leaf: int = 5, random_state: int = 0) -> pd.Series:
    """
    B3, random-forest proximities from scratch (Breiman's idea): a forest is grown on the LABEL
    from the training rows (gaps filled with the training medians); two rows' proximity = the
    share of trees in which they land in the same leaf (`forest.apply`). A query's missing
    `column` = the proximity-weighted mean of the training rows where `column` is observed.
    The queries' own labels are never used: they are filled with the training medians and
    only pushed down the trees, so this can run on new records -- but only if the model keeps
    the forest and the whole training matrix (memory grows with the data), and the forest
    needed a label to grow, so it can only be fitted where the target exists.
    """
    medians = train.median()
    filled_train, filled_queries = train.fillna(medians), queries.fillna(medians)
    forest = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                    random_state=random_state, n_jobs=-1).fit(filled_train, y_train)
    donors = train[column].notna().to_numpy()
    leaves_train = forest.apply(filled_train)[donors]
    leaves_queries = forest.apply(filled_queries)
    proximity = np.zeros((len(queries), donors.sum()), dtype="float32")
    for tree in range(leaves_train.shape[1]):
        proximity += leaves_queries[:, [tree]] == leaves_train[:, tree]
    weights = proximity.sum(axis=1)
    estimate = (proximity @ train.loc[donors, column].to_numpy(dtype="float64")) / np.where(weights > 0, weights, 1)
    return pd.Series(np.where(weights > 0, estimate, medians[column]), index=queries.index)


def hidden_value_errors(config, task="classification", data=None, names=None, columns=None,
                        n_per_fold: int = 200, seed: int = 42, include_proximity: bool = True) -> pd.DataFrame:
    """
    Which imputer guesses best? In the validation rows of every temporal fold, hides
    `n_per_fold` observed values of each column with real gaps (`columns`, default: the numeric
    features still missing after the history fill) among rows with purchase history (the
    gaps' own population), fits each variant's numeric step -- history fill, the imputer, row
    limits -- on the training rows, and returns the mean absolute error of its guesses, in the
    column's units, per fold. With `include_proximity`: the B3 proximity imputer too.
    The hidden cells are the same for every variant. A guess close to the truth does not by
    itself make the model better (that is `imputer_model_comparison`).
    """
    data = data or prepare_task_data(config, task)
    base = data["preprocessing_config"]
    numeric = list(base["numeric_features"])
    X, y = data["X_dev"], data["y_dev"]
    folds = temporal_folds(data["extras_dev"][config["data"]["date_column"]],
                           config["cv"]["n_splits"], config["cv"]["gap_days"])
    history = CustomerHistoryFiller(base["customer_history"])
    rows = []
    for fold, (train, validation) in enumerate(folds, start=1):
        history.fit(X.iloc[train])
        train_filled, validation_filled = history.transform(X.iloc[train]), history.transform(X.iloc[validation])
        if columns is None:
            columns = [c for c in numeric if X[c].isna().any()]
        has_history = np.flatnonzero(validation_filled["prior_purchase_count"].gt(0).to_numpy())
        masked, truth = mask_numeric_values(validation_filled, columns, has_history, n_per_fold, seed + fold)
        for name, recipe in imputer_recipes(base, names).items():
            if int(recipe["imputation"].get("n_imputations", 1)) > 1:
                continue                                      # an ensemble of models, not one guess
            preprocessor = build_preprocessor(recipe).fit(train_filled, y.iloc[train])
            step = preprocessor.named_transformers_["numeric"].named_steps["impute"]
            inputs = numeric + list(step.helper_features or [])
            filled = pd.DataFrame(step.transform(masked[inputs]), columns=numeric, index=masked.index)
            for column, values in truth.items():
                rows.append({"task": task, "fold": fold, "variant": name, "column": column, "hidden": len(values),
                             "mae": float((filled.loc[values.index, column] - values).abs().mean())})
        if include_proximity:
            for column, values in truth.items():
                guess = proximity_impute(train_filled[numeric].astype("float64"), y.iloc[train],
                                         masked.loc[values.index, numeric].astype("float64"), column,
                                         random_state=seed)
                rows.append({"task": task, "fold": fold, "variant": "rf_proximity (B3)", "column": column,
                             "hidden": len(values), "mae": float((guess - values).abs().mean())})
        print(f"{task}: hidden values, fold {fold} done", flush=True)
    return pd.DataFrame(rows)


def imputer_experiment(config, task, regression_seeds=(42, 1, 2)):
    """Week 10: model scores per imputer (+ corrected paired test) and, for classification,
    the hidden-value errors (+ B3 proximities)."""
    data = prepare_task_data(config, task)
    tables = imputer_model_comparison(config, task, data, regression_seeds=regression_seeds)
    if task == "classification":
        errors = hidden_value_errors(config, task, data)
        tables["hidden_values"] = errors
        tables["hidden_values_summary"] = (errors.groupby(["column", "variant"], sort=False)["mae"]
                                           .agg(["mean", "std"]).reset_index())
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["categories", "ticket", "imputers", "both"], default="both")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("results/preprocessing_experiments"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = {"categories": categorical_history_experiment, "ticket": ticket_basis_experiment}
    if args.experiment == "imputers":
        experiments["imputers"] = imputer_experiment
    if args.experiment != "both":
        experiments = {args.experiment: experiments[args.experiment]}
    for name, function in experiments.items():
        for task in TASKS:
            for table_name, table in function(config, task).items():
                table.to_csv(args.output_dir / f"{name}_{task}_{table_name}.csv", index=False)
    metadata = {"config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                "config_semantic_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "train_sha256": hashlib.sha256(Path(config["data"]["train_path"]).read_bytes()).hexdigest(),
                "data_normalized_sha256": {key: hashlib.sha256(Path(config["data"][key]).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                           for key in ("train_path", "test_path", "clientes_path")},
                "runtime": {"python": platform.python_version(), **{name: version(name) for name in
                             ("pandas", "numpy", "scikit-learn", "category-encoders")}},
                "cv": config["cv"], "regression_model_seeds": [42, 1, 2], "classification_model_seed": 42,
                "mask_fraction": .1, "mask_seed": 42, "holdout_scored": False,
                "models": {task: config[task]["model"] for task in TASKS}}
    (args.output_dir / f"{args.experiment}_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
