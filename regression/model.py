"""
Regression models (how much will a returning customer spend in the next 90 days?).
Chosen in config.yaml -> regression.model, built here.
"""
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


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}
    if model_type not in _MODELS:
        raise ValueError(f"Unknown regression model: {model_type}. Options: {list(_MODELS)}")
    return _MODELS[model_type](**params)
