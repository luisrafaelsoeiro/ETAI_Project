"""
Regression models (how much will a returning customer spend in the next 90 days?).
Chosen in config.yaml -> regression.model, built here.
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

_MODELS = {
    # always predicts the training mean -- the floor every real model must beat
    "dummy": DummyRegressor,
    "ridge": Ridge,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
}

# the model is fitted on the transformed target; predictions are transformed back, so every
# score stays in euros
_TARGET_TRANSFORMS = {
    "none": None,
    "log1p": (np.log1p, np.expm1),
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}
    if model_type not in _MODELS:
        raise ValueError(f"Unknown regression model: {model_type}. Options: {list(_MODELS)}")
    model = _MODELS[model_type](**params)

    transform = model_config.get("target_transform", "none")
    if transform not in _TARGET_TRANSFORMS:
        raise ValueError(f"Unknown target_transform: {transform}. Options: {list(_TARGET_TRANSFORMS)}")
    if _TARGET_TRANSFORMS[transform] is None:
        return model
    func, inverse_func = _TARGET_TRANSFORMS[transform]
    return TransformedTargetRegressor(regressor=model, func=func, inverse_func=inverse_func)
