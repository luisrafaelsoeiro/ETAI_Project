"""
The estimator and its life cycle, shared by both tasks:

  - `build_pipeline`: the preprocessing recipe and a model as ONE sklearn Pipeline, so every
    learned step is re-fitted on the training rows of each fold;
  - `prepare_task_data`: everything before fitting, for one task -- load (placeholder tokens
    -> NaN), validate, clean, training-row
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

from src.cleaning import drop_training_rows, prepare_table
from src.data import load_data, report_placeholders, validate_customer_ids
from src.preprocessing import (
    build_preprocessor, build_selector, check_feature_lists, split_dev_test, task_preprocessing_config,
)
from src.tasks import get_task
from src.history import CustomerHistoryFiller


def build_pipeline(preprocessing_config: dict, model) -> Pipeline:
    """Preprocessing (+ optional feature selection) + model as one estimator: `fit` learns
    everything from training rows only. Step names address hyperparameters from outside
    (`model__max_depth`). Optional history is frozen inside each fit, then preprocessing,
    optional selection and the model run. `selection.method: none` adds no selector."""
    steps = []
    if preprocessing_config.get("customer_history") is not None:
        steps.append(("history", CustomerHistoryFiller(preprocessing_config["customer_history"])))
    steps.append(("prep", build_preprocessor(preprocessing_config)))
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
      - history: prepared training purchases for descriptive diagnostics only; each
        estimator freezes its own source at fit time (never prefill these matrices);
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

    # The brief allows only train.csv as cross-row context for a Kaggle purchase.
    # Keep the unfilled source separate: no prediction row can enter another's history.
    history = train.copy()
    # Keep gaps here. Each fitted pipeline freezes its own temporal history source,
    # so CV, nested tuning, holdout and inference share the same protocol.

    customer_report = validate_customer_ids(train, clientes, data_cfg)
    train, row_report = drop_training_rows(train, diag_cfg, data_cfg["id_column"], config.get("customers"))
    train = task_package.preprocessing.select_training_rows(train, config)
    X, y, extras = task_package.preprocessing.prepare_features(train, config)
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras, config["test_set"], data_cfg["date_column"],
    )

    # test.csv: stateless preparation only; history filling happens at pipeline.predict.
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
    """Raw purchases -> stateless preparation -> the fitted pipeline's own history/imputers.

    Query rows never supply history to each other. The model stores its authorised,
    target-free source at fit time, including for a loaded model. `history` is a deprecated
    compatibility argument and is ignored: supplying it cannot enlarge that frozen source.
    Disable diagnostics.customer_history.use BEFORE fitting to compare against no history.
    Nothing is fitted on the new rows.
    """
    table, _ = prepare_table(raw_records, config, clientes)
    # `history` is retained for call compatibility. The fitted pipeline owns its
    # authorised source; an external dataframe must not bypass its training cutoff.
    X, _, _ = get_task(task).preprocessing.prepare_features(table, config)
    return pipeline.predict(X)
