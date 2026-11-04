"""
Terminal entry point: python main.py [--task classification|regression|both]

Runs the preprocessing pipeline end to end for each task:
raw train.csv -> clean_dataset -> drop_training_rows -> task rows -> (X, y, extras)
-> locked temporal test set -> build_preprocessor fitted on the development rows only,
then applied unchanged to the locked test set and to test.csv.

The model step (Pipeline([("prep", ...), ("model", ...)])) is added with model.py.
"""
import argparse

import numpy as np
import yaml

import classification.preprocessing as classification_prep
import regression.preprocessing as regression_prep
from src.preprocessing import build_preprocessor, clean_dataset, drop_training_rows, load_data, split_dev_test

TASKS = {"classification": classification_prep, "regression": regression_prep}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_preprocessing(config: dict, task: str) -> dict:
    task_prep = TASKS[task]
    data_cfg, diag_cfg, prep_cfg = config["data"], config["diagnostics"], config["preprocessing"]

    train, _ = clean_dataset(load_data(data_cfg["train_path"]), diag_cfg)
    train, _ = drop_training_rows(train, diag_cfg, data_cfg["id_column"])
    train = task_prep.select_training_rows(train, config)
    X, y, extras = task_prep.prepare_features(train, config)
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras, config["test_set"], data_cfg["date_column"],
    )

    # test.csv: same stateless cleaning, no rows dropped, no target needed
    kaggle, _ = clean_dataset(load_data(data_cfg["test_path"]), diag_cfg)
    X_kaggle, _, extras_kaggle = task_prep.prepare_features(kaggle, config)

    prep = build_preprocessor(prep_cfg).fit(X_dev, y_dev)      # fitted on development rows only
    matrices = {"dev": prep.transform(X_dev), "locked test": prep.transform(X_test), "test.csv": prep.transform(X_kaggle)}

    print(f"\n=== {task} ===")
    print(f"development rows: {len(X_dev)}  ({extras_dev[data_cfg['date_column']].min():%Y-%m-%d} -> "
          f"{extras_dev[data_cfg['date_column']].max():%Y-%m-%d})")
    print(f"locked test rows: {len(X_test)}  ({extras_test[data_cfg['date_column']].min():%Y-%m-%d} -> "
          f"{extras_test[data_cfg['date_column']].max():%Y-%m-%d})")
    for name, matrix in matrices.items():
        print(f"{name:>12}: shape {matrix.shape}, NaN left {int(np.isnan(matrix).sum())}")
    if len(X_kaggle) != len(kaggle):
        raise AssertionError("test.csv lost rows -- every id needs a prediction.")

    return {"preprocessor": prep, "X_dev": X_dev, "y_dev": y_dev, "X_test": X_test, "y_test": y_test,
            "extras_dev": extras_dev, "extras_test": extras_test, "X_kaggle": X_kaggle, "extras_kaggle": extras_kaggle}


def main():
    parser = argparse.ArgumentParser(description="ETAI customer repurchase & spend pipeline")
    parser.add_argument("--task", choices=["classification", "regression", "both"], default="both")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tasks = list(TASKS) if args.task == "both" else [args.task]
    for task in tasks:
        run_preprocessing(config, task)


if __name__ == "__main__":
    main()
