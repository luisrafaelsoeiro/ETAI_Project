"""
Terminal entry point: python main.py [--task classification|regression|both]

For each task:
raw train.csv -> prepare_table (schema validation, cleaning, clientes.csv join) -> drop_training_rows
-> task rows -> (X, y, extras) -> locked temporal test set -> Pipeline(build_preprocessor + build_model)
-> temporal cross-validation on the development set (train vs validation scores).

Then the fitted pipeline is saved (config.yaml -> output.models_dir), loaded back, and the
loaded copy predicts test.csv and a single raw record -- proving the saved file is all that
is needed to score new purchases.

The locked test set is set aside here and never scored by this script.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

import classification.model as classification_model
import classification.preprocessing as classification_prep
import regression.model as regression_model
import regression.preprocessing as regression_prep
from src.evaluate import cross_validate_pipeline, make_pipeline, summarize, temporal_folds
from src.preprocessing import (
    build_preprocessor, check_feature_lists, drop_training_rows, load_data, prepare_table, report_placeholders,
    split_dev_test, task_preprocessing_config, validate_customer_ids,
)

TASKS = {"classification": classification_prep, "regression": regression_prep}
MODELS = {"classification": classification_model, "regression": regression_model}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_preprocessing(config: dict, task: str, verbose: bool = True) -> dict:
    task_prep = TASKS[task]
    data_cfg, diag_cfg = config["data"], config["diagnostics"]
    prep_cfg = task_preprocessing_config(config, task)
    tokens = diag_cfg.get("placeholder_tokens")

    clientes = load_data(data_cfg["clientes_path"], tokens)
    train, train_report = prepare_table(load_data(data_cfg["train_path"], tokens), config, clientes)
    customer_report = validate_customer_ids(train, clientes, data_cfg)
    train, _ = drop_training_rows(train, diag_cfg, data_cfg["id_column"], config.get("customers"))
    train = task_prep.select_training_rows(train, config)
    X, y, extras = task_prep.prepare_features(train, config)
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras, config["test_set"], data_cfg["date_column"],
    )

    # test.csv: same stateless cleaning, no rows dropped, no target needed
    kaggle_raw = load_data(data_cfg["test_path"], tokens)
    kaggle, _ = prepare_table(kaggle_raw, config, clientes)
    X_kaggle, _, extras_kaggle = task_prep.prepare_features(kaggle, config)
    if len(X_kaggle) != len(kaggle):
        raise AssertionError("test.csv lost rows -- every id needs a prediction.")

    check_feature_lists(prep_cfg, X_dev.columns)               # every listed feature exists, no overlaps
    prep = build_preprocessor(prep_cfg).fit(X_dev, y_dev)      # fitted on development rows only
    matrices = {"dev": prep.transform(X_dev), "locked test": prep.transform(X_test), "test.csv": prep.transform(X_kaggle)}

    if verbose:
        date = data_cfg["date_column"]
        print(f"\n=== {task} (encoder: {prep_cfg['encoder']}, scaler: {prep_cfg['scaler']}) ===")
        placeholders = report_placeholders(data_cfg["train_path"], diag_cfg)
        print("train.csv placeholders:", placeholders.groupby("placeholder")["count"].sum().to_dict())
        schema_issues = train_report[train_report["step"].str.startswith("schema")]
        flagged = ", ".join(f"{r.column} {r.values}" for r in schema_issues.itertuples()) or "none"
        print(f"schema validation: {int(schema_issues['values'].sum())} values flagged ({flagged})")
        print(f"customer_id validation: {int(customer_report['rows'].sum())} problems")
        print(f"development rows: {len(X_dev)}  ({extras_dev[date].min():%Y-%m-%d} -> {extras_dev[date].max():%Y-%m-%d})")
        print(f"locked test rows: {len(X_test)}  ({extras_test[date].min():%Y-%m-%d} -> {extras_test[date].max():%Y-%m-%d})")
        for name, matrix in matrices.items():
            print(f"{name:>12}: shape {matrix.shape}, NaN left {int(np.isnan(matrix).sum())}")

    return {"preprocessing_config": prep_cfg, "X_dev": X_dev, "y_dev": y_dev, "X_test": X_test, "y_test": y_test,
            "extras_dev": extras_dev, "extras_test": extras_test, "X_kaggle": X_kaggle, "extras_kaggle": extras_kaggle,
            "kaggle_raw": kaggle_raw, "clientes": clientes}


def run_model(config: dict, task: str, data: dict):
    """Cross-validates the configured model against the dummy baseline on the development
    set, then fits the configured pipeline on all development rows."""
    model_cfg = config[task]["model"]
    folds = temporal_folds(
        data["extras_dev"][config["data"]["date_column"]],
        config["validation"]["n_splits"], config["validation"]["gap_days"],
    )
    baseline_cfg = next(m for m in config[task]["candidate_models"] if m["type"] == "dummy")

    rows = {}
    for name, cfg in {"dummy baseline": baseline_cfg, f"{model_cfg['type']} (configured)": model_cfg}.items():
        pipeline = make_pipeline(data["preprocessing_config"], MODELS[task].build_model(cfg))
        rows[name] = summarize(cross_validate_pipeline(pipeline, data["X_dev"], data["y_dev"], folds, task))
    report = pd.DataFrame(rows).T
    metric = config[task]["metric"]
    print(f"temporal CV, {len(folds)} folds, metric {metric}:")
    print(report[[f"train_{metric}_mean", f"validation_{metric}_mean", f"validation_{metric}_std"]].round(4).to_string())

    pipeline = make_pipeline(data["preprocessing_config"], MODELS[task].build_model(model_cfg))
    return pipeline.fit(data["X_dev"], data["y_dev"]), report


def predict_new_records(raw_records: pd.DataFrame, config: dict, task: str, pipeline, clientes=None):
    """Raw purchase rows (as in test.csv) -> predictions: the same stateless steps as
    training, then the fitted pipeline. Nothing is learned from the new rows."""
    table, _ = prepare_table(raw_records, config, clientes)
    X, _, _ = TASKS[task].prepare_features(table, config)
    return pipeline.predict(X)


def save_and_check(pipeline, config: dict, task: str, data: dict) -> Path:
    """Saves the fitted pipeline, loads it back, and checks that the loaded copy predicts
    test.csv and a single new raw record exactly like the pipeline in memory."""
    path = Path(config.get("output", {}).get("models_dir", "models")) / f"{task}_pipeline.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)

    in_memory = pipeline.predict(data["X_kaggle"])
    from_disk = loaded.predict(data["X_kaggle"])
    if not np.allclose(in_memory, from_disk):
        raise AssertionError("the loaded pipeline predicts differently from the saved one")
    one_record = predict_new_records(data["kaggle_raw"].head(1), config, task, loaded, data["clientes"])
    if not np.allclose(one_record, in_memory[:1]):
        raise AssertionError("a single raw record is not predicted like the same row inside test.csv")
    print(f"saved {path} ({path.stat().st_size / 1024:.0f} KB); the loaded copy predicts test.csv "
          f"({len(from_disk)} rows) and a single raw record identically (first prediction: {one_record[0]:.3f})")
    return path


def main():
    parser = argparse.ArgumentParser(description="ETAI customer repurchase & spend pipeline")
    parser.add_argument("--task", choices=["classification", "regression", "both"], default="both")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tasks = list(TASKS) if args.task == "both" else [args.task]
    for task in tasks:
        data = run_preprocessing(config, task)
        pipeline, _ = run_model(config, task, data)
        save_and_check(pipeline, config, task, data)


if __name__ == "__main__":
    main()
