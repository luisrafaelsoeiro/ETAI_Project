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
import json
import platform
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier
from sklearn.pipeline import Pipeline

from src.cleaning import drop_training_rows, prepare_table
from src.data import load_data, report_placeholders, standardize_columns, validate_customer_ids
from src.preprocessing import (
    MODEL_BASED_IMPUTERS, build_preprocessor, build_selector, check_feature_lists, split_dev_test,
    task_preprocessing_config,
)
from src.tasks import get_task
from src.history import CustomerHistoryFiller


class _MultipleImputation(BaseEstimator):
    """
    MICE as the theory describes it: `n_imputations` copies of `pipeline`, each with its
    iterative imputer DRAWING every gap around the regression's guess (`sample_posterior`)
    with its own seed -- so each copy is trained on a different completed dataset -- and their
    predictions averaged. The spread between copies is the imputation uncertainty a single
    imputation hides. Costs `n_imputations` times one pipeline. Hyperparameters are reached
    through the template: `pipeline__model__C`.

    The draws happen in TRAINING only: once a copy is fitted, its imputer fills new rows with
    its own best guess (no draw). Otherwise a fitted imputer keeps drawing, so the same
    purchase would get a different prediction on every call, depending on the other rows of
    the batch. The copies still disagree on a new row's gaps -- each learned its imputation
    models and its classifier from a different completed dataset -- and their average is
    reproducible and row by row.
    """

    IMPUTER = "prep__numeric__impute__imputer"

    def __init__(self, pipeline, n_imputations: int = 5, random_state: int = 0):
        self.pipeline = pipeline
        self.n_imputations = n_imputations
        self.random_state = random_state

    def fit(self, X, y):
        self.pipelines_ = []
        for i in range(self.n_imputations):
            copy = clone(self.pipeline).set_params(**{f"{self.IMPUTER}__sample_posterior": True,
                                                      f"{self.IMPUTER}__random_state": self.random_state + i})
            copy.fit(X, y)
            fitted_imputer(copy).sample_posterior = False      # trained on draws; predicts with the best guess
            self.pipelines_.append(copy)
        return self


def fitted_imputer(pipeline):
    """The fitted base numeric imputer (e.g. the IterativeImputer) inside a fitted Pipeline."""
    return pipeline.named_steps["prep"].named_transformers_["numeric"].named_steps["impute"].imputer_


class MultipleImputationClassifier(ClassifierMixin, _MultipleImputation):
    """Averages the class probabilities of the copies; predicts the most probable class."""

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = self.pipelines_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([p.predict_proba(X) for p in self.pipelines_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class MultipleImputationRegressor(RegressorMixin, _MultipleImputation):
    """Averages the predictions of the copies."""

    def predict(self, X):
        return np.mean([p.predict(X) for p in self.pipelines_], axis=0)


def build_pipeline(preprocessing_config: dict, model):
    """Preprocessing (+ optional feature selection) + model as one estimator: `fit` learns
    everything from training rows only. Step names address hyperparameters from outside
    (`model__max_depth`). Optional history is frozen inside each fit, then preprocessing,
    optional selection and the model run. `selection.method: none` adds no selector.

    `imputation.n_imputations` > 1 (needs `numeric_strategy: iterative`) returns that many
    copies of the Pipeline as one estimator (`MultipleImputationClassifier` / `...Regressor`),
    each trained on its own randomly drawn completed dataset, with averaged predictions."""
    steps = []
    if preprocessing_config.get("customer_history") is not None:
        steps.append(("history", CustomerHistoryFiller(preprocessing_config["customer_history"])))
    steps.append(("prep", build_preprocessor(preprocessing_config)))
    selector = build_selector(preprocessing_config.get("selection"), is_classifier(model),
                              preprocessing_config.get("random_state"))
    if selector is not None:
        steps.append(("select", selector))
    pipeline = Pipeline(steps + [("model", model)])

    imputation = preprocessing_config.get("imputation", {})
    n_imputations = int(imputation.get("n_imputations", 1))
    if n_imputations <= 1:
        return pipeline
    if imputation.get("numeric_strategy") != "iterative":
        raise ValueError("imputation.n_imputations > 1 draws from the iterative imputer: "
                         "set imputation.numeric_strategy: iterative.")
    ensemble = MultipleImputationClassifier if is_classifier(model) else MultipleImputationRegressor
    return ensemble(pipeline, n_imputations, imputation.get("random_state", 0))


def imputer_label(preprocessing_config: dict) -> str:
    """The numeric imputer in a few characters, for the scoreboard and the reports:
    e.g. "median", "knn+categories", "iterative x5"."""
    imputation = preprocessing_config.get("imputation", {})
    label = imputation.get("numeric_strategy", "median")
    if imputation.get("categorical_helpers", False) and label in MODEL_BASED_IMPUTERS:
        label += "+categories"
    if int(imputation.get("n_imputations", 1)) > 1:
        label += f" x{int(imputation['n_imputations'])}"
    return label


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


# the libraries a saved pipeline depends on: a model must be loaded with the versions it was saved with
_PICKLE_LIBRARIES = ("scikit-learn", "xgboost", "numpy", "pandas", "scipy", "joblib", "category-encoders")


def library_versions() -> dict:
    """{library: installed version} for the libraries a saved pipeline depends on, plus Python."""
    versions = {"python": platform.python_version()}
    for name in _PICKLE_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def versions_path(model_path) -> Path:
    """Where `save_pipeline` writes the library versions: next to the model, `<name>.versions.json`."""
    model_path = Path(model_path)
    return model_path.with_name(model_path.stem + ".versions.json")


def save_pipeline(pipeline, path) -> Path:
    """Saves a fitted pipeline with joblib (the folder is created if needed), and the library
    versions it was saved with next to it (`versions_path`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    versions_path(path).write_text(json.dumps(library_versions(), indent=2), encoding="utf-8")
    return path


def load_pipeline(path):
    """Loads a pipeline saved by `save_pipeline`. Warns when an installed library version differs
    from the one the model was saved with (its predictions may change, or loading may fail).
    Only load files you trust: a .joblib file can run code when loaded."""
    path = Path(path)
    saved = versions_path(path)
    if saved.exists():
        expected, installed = json.loads(saved.read_text(encoding="utf-8")), library_versions()
        different = {name: (expected[name], installed.get(name)) for name in expected
                     if name != "python" and expected[name] != installed.get(name)}
        if different:
            details = ", ".join(f"{name} saved {old} / installed {new}" for name, (old, new) in different.items())
            warnings.warn(f"{path.name} was saved with other library versions ({details}). "
                          "Install the saved versions or re-run main.py.", stacklevel=2)
    return joblib.load(path)


def check_input_columns(raw_records: pd.DataFrame, config: dict) -> None:
    """
    Raises ValueError naming every column the model needs that `raw_records` lacks (after the
    Portuguese -> English renames). Targets are never needed. Extra columns are allowed: the
    schema validation reports them.
    """
    columns = set(standardize_columns(raw_records.head(0), config["diagnostics"]["column_renames"]).columns)
    needed = [c for c in config["schema"]["required_columns"] if c not in config["data"]["target_columns"]]
    missing = [c for c in needed if c not in columns]
    if missing:
        raise ValueError(f"The input lacks {len(missing)} column(s) the model needs: {missing}. "
                         "It must have the columns of test.csv (original or English names).")


def prepare_new_records(raw_records: pd.DataFrame, config: dict, task: str, clientes=None) -> tuple:
    """Raw purchases (as in test.csv) -> (X, extras) with the same stateless steps as training.
    Row-preserving: no row is dropped, not even a duplicate -- every row needs a prediction."""
    check_input_columns(raw_records, config)
    table, _ = prepare_table(raw_records, config, clientes)
    X, _, extras = get_task(task).preprocessing.prepare_features(table, config)
    if len(X) != len(raw_records):
        raise AssertionError("Preparing the new records lost rows -- every row needs a prediction.")
    return X, extras


def predict_new_records(raw_records: pd.DataFrame, config: dict, task: str, pipeline, clientes=None,
                        history: pd.DataFrame = None):
    """Raw purchases -> stateless preparation -> the fitted pipeline's own history/imputers.

    Query rows never supply history to each other. The model stores its authorised,
    target-free source at fit time, including for a loaded model. `history` is a deprecated
    compatibility argument and is ignored: supplying it cannot enlarge that frozen source.
    Disable diagnostics.customer_history.use BEFORE fitting to compare against no history.
    Nothing is fitted on the new rows.
    """
    # `history` is retained for call compatibility. The fitted pipeline owns its
    # authorised source; an external dataframe must not bypass its training cutoff.
    X, _ = prepare_new_records(raw_records, config, task, clientes)
    return pipeline.predict(X)
