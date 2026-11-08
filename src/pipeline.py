"""
The estimator and its life cycle, shared by both tasks:

  - `build_pipeline`: the preprocessing recipe and a model as ONE sklearn Pipeline, so every
    learned step is re-fitted on the training rows of each fold;
  - `prepare_task_data`: everything before fitting, for one task -- load (placeholder tokens
    -> NaN), validate, clean, fill from each customer's earlier purchases, training-row
    decisions, (X, y, extras), the locked temporal test set, and test.csv prepared with the
    same steps;
  - `save_pipeline` / `load_pipeline`: joblib persistence of a fitted pipeline;
  - `predict_new_records`: raw purchase rows (as in test.csv) -> predictions.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline

from src.cleaning import drop_training_rows, fill_from_customer_history, prepare_table
from src.data import load_data, report_placeholders, validate_customer_ids
from src.preprocessing import (
    build_preprocessor, build_selector, check_feature_lists, split_dev_test, task_preprocessing_config,
)
from src.tasks import get_task


def build_pipeline(preprocessing_config: dict, model) -> Pipeline:
    """Preprocessing (+ optional feature selection) + model as one estimator: `fit` learns
    everything from training rows only. Step names address hyperparameters from outside
    (`model__max_depth`). `preprocessing.selection.method: none` (the default) adds no step,
    so the pipeline is then exactly `prep` + `model`."""
    steps = [("prep", build_preprocessor(preprocessing_config))]
    selector = build_selector(preprocessing_config.get("selection"), is_classifier(model),
                              preprocessing_config.get("random_state"))
    if selector is not None:
        steps.append(("select", selector))
    return Pipeline(steps + [("model", model)])


def prepare_task_data(config: dict, task: str) -> dict:
    """
    Loads, validates and cleans train.csv and test.csv for one task, then sets the locked
    temporal test set aside. Nothing is fitted here. Returns a dict with:
      - X_dev / y_dev / extras_dev: the development set (everything we learn and choose on);
      - X_test / y_test / extras_test: the locked test set;
      - X_kaggle / extras_kaggle / kaggle_raw: test.csv (no target), every row kept;
      - preprocessing_config: the task's recipe (shared section + `<task>.preprocessing`);
      - clientes: the customer table; reports: placeholder, schema and customer_id reports.
    """
    task_package = get_task(task)
    data_cfg, diag_cfg = config["data"], config["diagnostics"]
    tokens = diag_cfg.get("placeholder_tokens")
    prep_cfg = task_preprocessing_config(config, task)

    clientes = load_data(data_cfg["clientes_path"], tokens)
    train, train_report = prepare_table(load_data(data_cfg["train_path"], tokens), config, clientes)
    kaggle_raw = load_data(data_cfg["test_path"], tokens)
    kaggle, kaggle_report = prepare_table(kaggle_raw, config, clientes)

    # every purchase known at prediction time; each row is filled from its own past only
    history = pd.concat([train, kaggle], ignore_index=True)
    train, train_history = fill_from_customer_history(train, history, diag_cfg, data_cfg["id_column"])
    kaggle, kaggle_history = fill_from_customer_history(kaggle, history, diag_cfg, data_cfg["id_column"])
    train_report = pd.concat([train_report, train_history], ignore_index=True)
    kaggle_report = pd.concat([kaggle_report, kaggle_history], ignore_index=True)

    customer_report = validate_customer_ids(train, clientes, data_cfg)
    train, row_report = drop_training_rows(train, diag_cfg, data_cfg["id_column"], config.get("customers"))
    train = task_package.preprocessing.select_training_rows(train, config)
    X, y, extras = task_package.preprocessing.prepare_features(train, config)
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras, config["test_set"], data_cfg["date_column"],
    )

    # test.csv: the same stateless steps and history fill (above), no rows dropped, no target needed
    X_kaggle, _, extras_kaggle = task_package.preprocessing.prepare_features(kaggle, config)
    if len(X_kaggle) != len(kaggle_raw):
        raise AssertionError("test.csv lost rows -- every id needs a prediction.")
    check_feature_lists(prep_cfg, X_dev.columns)              # every listed feature exists, no overlaps

    return {
        "task": task, "preprocessing_config": prep_cfg,
        "X_dev": X_dev, "y_dev": y_dev, "extras_dev": extras_dev,
        "X_test": X_test, "y_test": y_test, "extras_test": extras_test,
        "X_kaggle": X_kaggle, "extras_kaggle": extras_kaggle, "kaggle_raw": kaggle_raw, "clientes": clientes,
        "history": history,
        "reports": {
            "placeholders": report_placeholders(data_cfg["train_path"], diag_cfg),
            "train_cleaning": train_report, "test_cleaning": kaggle_report,
            "customer_ids": customer_report, "training_rows": row_report,
        },
    }


def save_pipeline(pipeline, path) -> Path:
    """Saves a fitted pipeline with joblib (the folder is created if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(path):
    """Loads a pipeline saved by `save_pipeline`."""
    return joblib.load(path)


def predict_new_records(raw_records: pd.DataFrame, config: dict, task: str, pipeline, clientes=None,
                        history: pd.DataFrame = None):
    """Raw purchase rows (as in test.csv) -> predictions: the same stateless steps as
    training (`prepare_table`), the history fill from the customer's earlier purchases
    (`history`: the known purchases, e.g. `prepare_task_data(...)["history"]`; without it only
    the new rows themselves are used), then the fitted pipeline. Nothing is learned from the new rows."""
    table, _ = prepare_table(raw_records, config, clientes)
    known = table if history is None else pd.concat([history, table], ignore_index=True)
    table, _ = fill_from_customer_history(table, known, config["diagnostics"], config["data"]["id_column"])
    X, _, _ = get_task(task).preprocessing.prepare_features(table, config)
    return pipeline.predict(X)
