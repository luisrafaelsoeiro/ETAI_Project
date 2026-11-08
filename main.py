"""
Entry point of the pipeline, for both tasks.

Run with:
    python main.py                          # classification, then regression
    python main.py --task classification    # or: --task regression

Workflow, per task (every choice comes from config.yaml):
    load (placeholder tokens -> NaN) -> validate (schema, customer ids) -> clean (stateless rules,
    customer-table join) -> training-row decisions -> (X, y, extras) -> lock the temporal test set
    -> no-model floor and temporal cross-validation of the configured pipeline (every metric)
    -> optional Optuna tuning with nested cross-validation (on the selection metric)
    -> out-of-fold report -> refit the selected pipeline on all development rows
    -> scoreboard row -> (only when test_set.evaluate is true) evaluate once on the locked test set
    -> save the fitted pipeline, reload it and check it -> predictions for test.csv -> run report
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluate import (
    cross_validate_pipeline, cross_validate_with_oof, cv_report, greater_is_better, select_scoring, temporal_folds,
)
from src.pipeline import build_pipeline, load_pipeline, predict_new_records, prepare_task_data, save_pipeline
from src.results import save_predictions, save_run, save_scores
from src.tasks import TASKS, get_task
from src.tuning import nested_cross_validate, nested_cv_report, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    """Load the project configuration."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def validation_summary(data: dict) -> str:
    """One paragraph on what loading, validation and cleaning found."""
    reports = data["reports"]
    placeholders = reports["placeholders"].groupby("placeholder")["count"].sum().to_dict()
    schema = reports["train_cleaning"]
    schema = schema[schema["step"].str.startswith("schema")]
    flagged = ", ".join(f"{r.column} {r.values}" for r in schema.itertuples()) or "none"
    removed = ", ".join(f"{r.rule} {r.rows_removed}" for r in reports["training_rows"].itertuples())
    return (f"Placeholders in train.csv: {placeholders}\n"
            f"Schema validation: {int(schema['values'].sum())} values flagged ({flagged})\n"
            f"customer_id validation: {int(reports['customer_ids']['rows'].sum())} problems\n"
            f"Training rows removed: {removed}\n"
            f"Development rows: {len(data['X_dev'])}   locked test rows: {len(data['X_test'])}   "
            f"test.csv rows: {len(data['X_kaggle'])}")


def run_task(config: dict, task: str, run: str) -> None:
    task_cfg, package = config[task], get_task(task)
    model_cfg = task_cfg["model"]
    metrics, metric = task_cfg["metrics"], task_cfg["selection_metric"]
    if metric not in metrics:
        raise ValueError(f"{task}.selection_metric '{metric}' must be one of {task}.metrics: {metrics}")
    scoring = select_scoring(package.evaluate.SCORING, metrics)
    higher = greater_is_better(scoring[metric])
    output = config.get("output", {})

    # 1. Load, validate, clean, lock the test set -----------------------------------------
    data = prepare_task_data(config, task)
    prep_cfg, X_dev, y_dev = data["preprocessing_config"], data["X_dev"], data["y_dev"]
    dates = data["extras_dev"][config["data"]["date_column"]]
    folds = temporal_folds(dates, config["cv"]["n_splits"], config["cv"]["gap_days"])
    summary = validation_summary(data)
    print(f"\n{'=' * 72}\n{task.upper()}  (target: {task_cfg['target']}, model: {model_cfg['type']}, "
          f"selection metric: {metric})\n{'=' * 72}\n{summary}")
    report = summary + "\n\n"

    # 2. No-model floor and default cross-validation ---------------------------------------
    pipeline = build_pipeline(prep_cfg, package.model.build_model(model_cfg))
    dummy_cfg = next(m for m in task_cfg["candidate_models"] if m["type"] == "dummy")
    floor_scores = cross_validate_pipeline(build_pipeline(prep_cfg, package.model.build_model(dummy_cfg)),
                                           X_dev, y_dev, folds, {metric: scoring[metric]})
    fold_scores, oof = cross_validate_with_oof(pipeline, X_dev, y_dev, folds, scoring,
                                               package.evaluate.predict_for_report)
    floor = floor_scores[f"validation_{metric}"].mean()
    untuned = fold_scores[f"validation_{metric}"].mean()
    print(f"\nTemporal CV ({len(folds)} folds), default hyperparameters:  {metric} = {untuned:.4f} "
          f"± {fold_scores[f'validation_{metric}'].std(ddof=1):.4f}   (no-model floor: {floor:.4f})")
    report += (f"Encoder: {prep_cfg['encoder']}   scaler: {prep_cfg['scaler']}   "
               f"imputation: {prep_cfg['imputation']}\n"
               f"Hyperparameters from config.yaml: {model_cfg.get('params')}\n\n"
               + cv_report(fold_scores, metrics, metric, "Temporal cross-validation (default hyperparameters)", higher)
               + f"\n\nNo-model floor (dummy, same folds): {metric} = {floor:.4f}")
    final_pipeline, final_scores, study = pipeline, fold_scores, None

    # 3. Optional Optuna tuning with nested cross-validation -------------------------------
    tuning = task_cfg.get("tuning") or {}
    if tuning.get("enabled", False):
        search_space = (tuning.get("search_spaces") or {}).get(model_cfg["type"])
        if not search_space:
            raise ValueError(f"Tuning is enabled but {task}.tuning.search_spaces has no entry for "
                             f"'{model_cfg['type']}'. Add one, or set {task}.tuning.enabled: false.")
        n_trials, seed = tuning.get("n_trials", 30), tuning.get("random_state", 42)
        inner_splits, gap = tuning.get("inner_splits", 3), config["cv"]["gap_days"]
        print(f"Tuning ({n_trials} trials, nested temporal CV) ... this takes a while", flush=True)
        final_scores, oof = nested_cross_validate(
            pipeline, X_dev, y_dev, dates, folds, inner_splits, gap, scoring, metric, search_space,
            n_trials, seed, package.evaluate.predict_for_report)
        final_pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, temporal_folds(dates, inner_splits, gap),
                                              scoring[metric], search_space, n_trials, seed)
        print(f"  nested CV {metric} = {final_scores[f'validation_{metric}'].mean():.4f} ± "
              f"{final_scores[f'validation_{metric}'].std(ddof=1):.4f}   (honest estimate of the tuned model)")
        print(f"  chosen: {study.best_params}")
        report += ("\n\n" + cv_report(final_scores, metrics, metric, "Nested cross-validation (tuning, honest estimate)", higher)
                   + "\n\n" + nested_cv_report(final_scores, metrics)
                   + "\n\n" + tuning_report(study, final_scores, scoring, metric))

    for name in metrics:
        print(f"  {name:<10s} {final_scores[f'validation_{name}'].mean():.4f} ± "
              f"{final_scores[f'validation_{name}'].std(ddof=1):.4f}")
    report += "\n\n" + package.evaluate.oof_report(y_dev, oof)

    # 4. Refit on all development rows -----------------------------------------------------
    final_model = final_pipeline.fit(X_dev, y_dev)
    refit = (f"Final model: {model_cfg['type']} {'(tuned)' if study else '(default hyperparameters)'} "
             f"refit on all {len(X_dev)} development rows.")
    report += "\n" + refit + "\n"

    # 5. Scoreboard --------------------------------------------------------------------------
    validation = final_scores[f"validation_{metric}"]
    row = {"run": run, "task": task, "model": model_cfg["type"], "tuned": bool(study),
           "encoder": prep_cfg["encoder"], "scaler": prep_cfg["scaler"],
           "params": str(model_cfg.get("params")), "tuned_params": str(study.best_params) if study else "",
           "metric": metric, "mean": validation.mean(), "std": validation.std(ddof=1),
           "untuned": untuned, "floor": floor,
           **{f"fold_{i}": score for i, score in enumerate(validation, start=1)}}
    runs = save_scores(output.get("scoreboard", "results/runs.csv"), row)
    board = (runs[runs["task"] == task]
             .sort_values("mean", ascending=not higher)
             [["run", "model", "tuned", "encoder", "scaler", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": metric}).head(10)
             .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nScoreboard ({task}, best first)\n{board}")
    report += f"\nScoreboard ({task}, best first):\n{board}\n"

    # 6. Locked test set: evaluated once, only for the final chosen model ------------------
    if config["test_set"].get("evaluate", False):
        holdout = package.evaluate.holdout_report(
            data["y_test"], package.evaluate.predict_for_report(final_model, data["X_test"]))
        print("\n" + holdout)
        report += "\n" + holdout
        if config["test_set"].get("refit_on_all_rows", False):
            final_model = final_pipeline.fit(pd.concat([X_dev, data["X_test"]]), pd.concat([y_dev, data["y_test"]]))
            refit = f"Refit on all {len(X_dev) + len(data['X_test'])} labelled rows (development + locked test)."
            report += refit + "\n"

    # 7. Save the fitted pipeline, reload it, predict test.csv -----------------------------
    model_path = save_pipeline(final_model, Path(output.get("models_dir", "models")) / f"{task}_pipeline.joblib")
    loaded = load_pipeline(model_path)
    predictions = loaded.predict(data["X_kaggle"])
    if not np.allclose(predictions, final_model.predict(data["X_kaggle"])):
        raise AssertionError("The reloaded pipeline predicts differently from the fitted one.")
    one_record = predict_new_records(data["kaggle_raw"].head(1), config, task, loaded, data["clientes"], data["history"])
    if not np.allclose(one_record, predictions[:1]):
        raise AssertionError("A single raw record is not predicted like the same row of test.csv.")
    predictions_path = Path(output.get("predictions_dir", "results/predictions")) / f"{task}.csv"
    save_predictions(predictions_path, data["extras_kaggle"][config["data"]["id_column"]], predictions,
                     config["data"]["submission_id_column"], task_cfg["target"],
                     config["data"]["sample_submission_path"])
    saved = (f"Saved {model_path} (reloaded: identical predictions) and {len(predictions)} test.csv "
             f"predictions to {predictions_path}")
    report += saved + "\n"

    # 8. Run report -------------------------------------------------------------------------
    path = save_run(output.get("results_dir", "results"), run, task, config, report)
    print(f"\n{refit}\n{saved}\nFull report: {path}")


def main():
    parser = argparse.ArgumentParser(description="ETAI customer repurchase & spend pipeline")
    parser.add_argument("--task", choices=[*TASKS, "both"], default="both")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    for task in (list(TASKS) if args.task == "both" else [args.task]):
        run_task(config, task, run)


if __name__ == "__main__":
    main()
