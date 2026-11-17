"""
Regression models (how much will a returning customer spend in the next 90 days?).
Chosen in config.yaml -> regression.model, built here.
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

_MODELS = {
    # always predicts the training mean -- the floor every real model must beat
    "dummy": DummyRegressor,
    "ridge": Ridge,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "xgboost": XGBRegressor,
}


def build_model(model_config: dict):
    """Builds the model named by `model_config["type"]` with `model_config["params"]`.

    `model_config["target_transform"]` is `none` or `log1p`. With log1p the model learns
    log(1 + spend) and its predictions are mapped back to euros, so every metric stays in
    euros. The fitted inner model is then `.regressor_`, and tuned params need the prefix
    `regressor__` (e.g. model__regressor__alpha).
    """
    model_type = model_config["type"]
    params = model_config.get("params") or {}
    if model_type not in _MODELS:
        raise ValueError(f"Unknown regression model: {model_type}. Options: {list(_MODELS)}")
    model = _MODELS[model_type](**params)

    # YAML reads a bare none as the string "none", not as null
    transform = str(model_config.get("target_transform") or "none").lower()
    if transform == "log1p":
        return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    if transform != "none":
        raise ValueError(f"Unknown target_transform: {transform}. Options: none, log1p")
    return model
