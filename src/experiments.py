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
from src.pipeline import build_pipeline, prepare_task_data
from src.preprocessing import build_categorical_imputer
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["categories", "ticket", "both"], default="both")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("results/preprocessing_experiments"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = {"categories": categorical_history_experiment, "ticket": ticket_basis_experiment}
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
