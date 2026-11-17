"""
Regression task -- How much will a returning customer spend in the next 90 days? (target: `spend_next_90d`).

Only what differs from the other task lives here; the shared pipeline is in src/:
  - preprocessing: which training rows the task uses, and (X, y, extras);
  - model: the model registry (`build_model`);
  - evaluate: the metrics (`SCORING`), what the reports predict, and the reports.
"""
from regression import evaluate, model, preprocessing  # noqa: F401
