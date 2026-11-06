"""
The two prediction tasks. Each task package has the same interface, holding only what differs
between the tasks; everything else is shared code in src/:

  - `<task>.preprocessing`: which training rows the task uses, and (X, y, extras);
  - `<task>.model`: the model registry (`build_model`);
  - `<task>.evaluate`: the task's metrics (`SCORING`), what the reports predict, and the
    out-of-fold / locked-test reports.
"""
import classification
import regression

TASKS = {"classification": classification, "regression": regression}


def get_task(name: str):
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}. Options: {list(TASKS)}")
    return TASKS[name]
