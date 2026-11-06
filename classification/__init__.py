"""
Classification task -- Will the customer buy again within 90 days? (target: `repeat_purchase_90d`).

Only what differs from the other task lives here; the shared pipeline is in src/:
  - preprocessing: which training rows the task uses, and (X, y, extras);
  - model: the model registry (`build_model`);
  - evaluate: the metrics (`SCORING`), what the reports predict, and the reports.
"""
from classification import evaluate, model, preprocessing  # noqa: F401
