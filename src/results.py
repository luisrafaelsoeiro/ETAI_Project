"""
Saving results, shared by both tasks: one row per run in the scoreboard (`save_scores`), each
run's settings and full report (`save_run`), and the predictions for test.csv
(`save_predictions`, in the Kaggle submission format).
"""
import os

import numpy as np
import pandas as pd

MODEL_ABBREVIATIONS = {
    "dummy": "dm", "logistic_regression": "lg", "ridge": "rg", "decision_tree": "dt",
    "random_forest": "rf", "gradient_boosting": "gb", "hist_gradient_boosting": "hgb", "xgboost": "xgb",
}


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, task: str, config: dict, report_text: str) -> str:
    """
    Writes the run's settings and full report to `results_dir/<task>_<model>_<run>.txt`.
    `run` is the run id (a timestamp), the same as the run's row in the scoreboard, so the
    report and the row can be matched. Returns the file path.
    """
    os.makedirs(results_dir, exist_ok=True)
    task_cfg, cv, test_set = config[task], config["cv"], config["test_set"]
    model_type = task_cfg["model"]["type"]
    path = os.path.join(results_dir, f"{task}_{MODEL_ABBREVIATIONS.get(model_type, model_type)}_{run}.txt")
    tuning = task_cfg.get("tuning") or {}
    tuning_line = (f"Optuna, {tuning['n_trials']} trials, {tuning.get('inner_splits', cv['n_splits'])} inner "
                   f"temporal folds, nested CV, random_state={tuning.get('random_state')}"
                   if tuning.get("enabled") else "off")
    header = (
        f"Run: {run}\nTask: {task}  (target: {task_cfg['target']})\n"
        f"Model: {model_type}  params={task_cfg['model'].get('params')}"
        f"{'  target_transform=' + task_cfg['model']['target_transform'] if 'target_transform' in task_cfg['model'] else ''}\n"
        f"CV: {cv['n_splits']} expanding-window temporal folds, gap {cv['gap_days']} days\n"
        f"Metrics: {', '.join(task_cfg['metrics'])}   selection metric: {task_cfg['selection_metric']}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: latest {test_set['size']:.0%} of purchases, gap {test_set.get('gap_days', 0)} days  "
        f"({'evaluated once' if test_set.get('evaluate', False) else 'not evaluated'})\n"
        + "=" * 72 + "\n\n"
    )
    with open(path, "w", encoding="utf-8") as file:     # utf-8: the report contains "±"
        file.write(header + report_text)
    return path


def save_predictions(path: str, ids, predictions, id_column: str, target: str,
                     sample_submission_path: str = None) -> pd.DataFrame:
    """
    Writes one prediction per test.csv id. When the task's target is the Kaggle target, the
    file must match `sample_submission_path` exactly: same columns, same ids, same order.
    """
    submission = pd.DataFrame({id_column: np.asarray(ids, dtype="int64"), target: np.asarray(predictions)})
    if sample_submission_path:
        sample = pd.read_csv(sample_submission_path)
        if target in sample.columns:
            same_ids = np.array_equal(sample[id_column].to_numpy(), submission[id_column].to_numpy())
            if list(sample.columns) != list(submission.columns) or not same_ids:
                raise ValueError("Predictions don't match sample_submission.csv (columns, ids or order).")
            submission[target] = submission[target].astype(sample[target].dtype)   # e.g. 0/1 labels
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    submission.to_csv(path, index=False)
    return submission
