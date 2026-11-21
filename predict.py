"""
Predicts new purchases with the pipelines saved by `python main.py` -- no training.

Run from the project root:
    python predict.py data/examples/new_purchases.csv predictions.csv
    python predict.py data/raw/test.csv predictions.csv --task classification

The input has the columns of test.csv (original Portuguese or English names; no target needed).
Every row goes through the same stateless cleaning as training -- placeholder tokens -> NaN,
schema checks, consistency rules, the customer-table join -- and then the saved pipeline, whose
own steps fill the gaps with what they learned in training (customer history, imputers) and
encode categories never seen in training. Nothing is learned from the new rows, and no row is
dropped, not even a duplicate: one prediction per input row, in the input's order.

Output columns (`--task both`, the default):
    ID                      the purchase id, as in the input
    repeat_probability      probability that the customer buys again within 90 days
    repeat_purchase_90d     that prediction at threshold 0.5 (0/1, as in the Kaggle file)
    spend_next_90d          predicted spend in the next 90 days IF the customer returns (euros);
                            the regression model only learned from returning customers
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

from src.data import load_data
from src.pipeline import check_input_columns, load_pipeline, prepare_new_records
from src.tasks import TASKS


def model_path(config: dict, task: str) -> Path:
    """Where main.py saves the task's fitted pipeline."""
    return Path(config.get("output", {}).get("models_dir", "models")) / f"{task}_pipeline.joblib"


def predict(raw_records: pd.DataFrame, pipelines: dict, config: dict, clientes=None) -> pd.DataFrame:
    """
    One row per input row: the id and every requested task's prediction. `pipelines` is
    {task: fitted pipeline}. Raises ValueError when the input lacks a column the model needs.
    """
    check_input_columns(raw_records, config)
    ids =raw_records[config["data"]["submission_id_column"]] if config["data"]["submission_id_column"] in raw_records \
        else raw_records.get(config["data"]["id_column"], pd.Series(range(len(raw_records))))
    out = pd.DataFrame({config["data"]["submission_id_column"]: ids.to_numpy()})
    for task, pipeline in pipelines.items():
        X, _ = prepare_new_records(raw_records, config, task, clientes)
        if task == "classification":
            probability = pipeline.predict_proba(X)[:, 1]
            out["repeat_probability"] = probability.round(6)
            out[config[task]["target"]] = (probability >= 0.5).astype(int)
        else:
            out[config[task]["target"]] = pipeline.predict(X).round(2)
    if len(out) != len(raw_records):
        raise AssertionError(f"{len(raw_records)} rows in, {len(out)} rows out.")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict a CSV of new purchases with the models saved by main.py.")
    parser.add_argument("input", help="CSV with the columns of test.csv (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--task", choices=[*TASKS, "both"], default="both")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    tasks = list(TASKS) if args.task == "both" else [args.task]

    pipelines = {}
    for task in tasks:
        path = model_path(config, task)
        if not path.exists():
            print(f"error: {path} not found. Train and save the model first: python main.py --task {task}",
                  file=sys.stderr)
            return 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipelines[task] = load_pipeline(path)
        for warning in caught:
            if "library versions" in str(warning.message):
                print(f"warning: {warning.message}", file=sys.stderr)

    tokens = config["diagnostics"].get("placeholder_tokens")
    raw = load_data(args.input, tokens)
    clientes = load_data(config["data"]["clientes_path"], tokens)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = predict(raw, pipelines, config, clientes)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"{len(raw)} rows in, {len(predictions)} rows predicted ({', '.join(tasks)}) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
