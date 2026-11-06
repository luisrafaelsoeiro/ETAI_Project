"""
Checks of the preprocessing pipeline. Run from the project root:

    python tests/test_pipeline.py        (or: python -m pytest tests, if pytest is installed)

They cover: feature lists (exist, no overlap), missingness indicators inside the pipeline,
every data-dependent step fitted inside the pipeline on training rows only, the
stateless/row-preserving cleaning, schema validation, placeholder tokens, and saving,
loading and predicting new records with the complete pipeline.
"""
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.impute import MissingIndicator, SimpleImputer  # noqa: E402

import main  # noqa: E402
from src.evaluate import make_pipeline  # noqa: E402
from src.preprocessing import (  # noqa: E402
    build_preprocessor, check_feature_lists, load_data, prepare_table, report_placeholders, validate_schema,
)


@lru_cache(maxsize=None)
def task_data(task):
    return main.run_preprocessing(main.load_config(), task, verbose=False)


def config():
    return main.load_config()


def classification_pipeline():
    cfg = config()
    return make_pipeline(task_data("classification")["preprocessing_config"],
                         main.MODELS["classification"].build_model(cfg["classification"]["model"]))


def expect_error(function, *args):
    try:
        function(*args)
    except ValueError as error:
        return str(error)
    raise AssertionError(f"{function.__name__} did not raise ValueError")


# ---------------------------------------------------------------- feature lists
def test_feature_lists_exist_and_do_not_overlap():
    for task in ["classification", "regression"]:
        data = task_data(task)
        check_feature_lists(data["preprocessing_config"], data["X_dev"].columns)   # raises if wrong


def test_feature_list_errors_are_caught():
    recipe = task_data("classification")["preprocessing_config"]
    overlap = {**recipe, "categorical_features": recipe["categorical_features"] + ["net_total"]}
    assert "both numeric_features and categorical_features" in expect_error(check_feature_lists, overlap)
    absent = {**recipe, "numeric_features": recipe["numeric_features"] + ["not_a_column"]}
    assert "not found" in expect_error(check_feature_lists, absent, task_data("classification")["X_dev"].columns)
    orphan = {**recipe, "mnar_indicator_sources": ["not_a_feature"]}
    assert "mnar_indicator_sources" in expect_error(check_feature_lists, orphan)


# ---------------------------------------------------------------- indicators + fitted steps
def test_missingness_indicators_are_created_inside_the_pipeline():
    data = task_data("classification")
    prep = build_preprocessor(data["preprocessing_config"])
    name, transformer, columns = next(t for t in prep.transformers if t[0] == "indicators")
    assert isinstance(transformer, MissingIndicator)
    assert columns == data["preprocessing_config"]["mnar_indicator_sources"]
    assert not any(c.endswith("_was_missing") for c in data["X_dev"].columns), "indicators must not be made before the pipeline"
    prep.fit(data["X_dev"], data["y_dev"])
    names = list(prep.get_feature_names_out())
    assert any(n.startswith("indicators__") for n in names)
    column = names.index(next(n for n in names if n.startswith("indicators__")))
    flags = prep.transform(data["X_dev"])[:, column]
    expected = data["X_dev"][columns[0]].isna().to_numpy()
    assert np.array_equal(flags.astype(bool), expected), "the indicator must see the gaps before imputation"


def test_data_dependent_steps_are_fitted_inside_the_pipeline_on_training_rows_only():
    data = task_data("classification")
    pipeline = classification_pipeline()
    try:
        pipeline.predict(data["X_dev"])
        raise AssertionError("an unfitted pipeline must not predict")
    except NotFittedError:
        pass

    half = len(data["X_dev"]) // 2
    X_train, y_train = data["X_dev"].iloc[:half], data["y_dev"].iloc[:half]
    pipeline.fit(X_train, y_train)
    prep = pipeline.named_steps["prep"]
    numeric_features = data["preprocessing_config"]["numeric_features"]
    imputer = prep.named_transformers_["numeric"].named_steps["impute"]
    assert isinstance(imputer, SimpleImputer)
    learned = pd.Series(imputer.statistics_, index=numeric_features)
    assert np.allclose(learned, X_train[numeric_features].median()), "medians must come from the training rows"
    assert not np.allclose(learned, data["X_dev"][numeric_features].median()), "medians must not see the other rows"

    scaler = prep.named_transformers_["numeric"].named_steps["scale"]
    encoder = prep.named_transformers_["categorical"].named_steps["encode"]
    for step in [imputer, scaler, encoder, prep.named_transformers_["indicators"]]:
        assert getattr(step, "n_features_in_", None) is not None, f"{type(step).__name__} is not fitted"
    unseen = set(data["X_dev"].iloc[half:]["transport_method"].dropna()) - set(X_train["transport_method"].dropna())
    known = set(encoder.categories_[data["preprocessing_config"]["categorical_features"].index("transport_method")])
    assert not (unseen & known), "the encoder must only know categories of the training rows"


# ---------------------------------------------------------------- stateless cleaning + validation
def test_prepare_table_keeps_every_row_and_is_stateless():
    cfg = config()
    tokens = cfg["diagnostics"]["placeholder_tokens"]
    raw, clientes = load_data(cfg["data"]["test_path"], tokens), load_data(cfg["data"]["clientes_path"], tokens)
    table, _ = prepare_table(raw, cfg, clientes)
    assert len(table) == len(raw) == 986 and table["id"].tolist() == raw["ID"].tolist()
    for position in [0, 500, 985]:
        alone, _ = prepare_table(raw.iloc[[position]], cfg, clientes)
        numeric = table.select_dtypes("number").columns
        assert np.allclose(alone[numeric].to_numpy(dtype=float), table.iloc[[position]][numeric].to_numpy(dtype=float),
                           equal_nan=True), "a row must be cleaned the same way alone or inside a table"


def test_schema_validation_flags_and_repairs():
    cfg = config()
    raw = load_data(cfg["data"]["test_path"]).head(5)
    from src.preprocessing import standardize_columns
    table = standardize_columns(raw, cfg["diagnostics"]["column_renames"])
    table["payment_type"] = table["payment_type"].astype(object)
    table.loc[table.index[0], "payment_type"] = "Cash!!"            # not an allowed label
    table["cost"] = table["cost"].astype(object)
    table.loc[table.index[1], "cost"] = "twelve"                     # not a number
    out, report = validate_schema(table, cfg["schema"], cfg["diagnostics"]["non_negative_columns"])
    counts = report.set_index(["step", "column"])["values"]
    assert counts[("schema: label not allowed -> NaN", "payment_type")] == 1 and pd.isna(out["payment_type"].iloc[0])
    assert counts[("schema: non-numeric -> NaN", "cost")] == 1 and pd.api.types.is_numeric_dtype(out["cost"])
    assert "Missing required columns" in expect_error(validate_schema, table.drop(columns=["net_total"]), cfg["schema"])


def test_placeholder_tokens_become_nan_and_are_reported():
    cfg = config()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "tokens.csv"
        path.write_text("a,b\n1,x\n?,N/A\n,y\n1000000000,z\n", encoding="utf-8")
        loaded = load_data(path, cfg["diagnostics"]["placeholder_tokens"])
        assert loaded["a"].isna().sum() == 2 and loaded["b"].isna().sum() == 1
        report = report_placeholders(path, cfg["diagnostics"]).set_index(["column", "placeholder"])["count"]
        assert report[("a", "'?'")] == 1 and report[("a", "''")] == 1 and report[("b", "'N/A'")] == 1
        assert report[("a", "1,000,000,000")] == 1


# ---------------------------------------------------------------- save / load / predict
def test_complete_pipeline_can_be_fitted_saved_loaded_and_predict_new_records():
    cfg = config()
    for task in ["classification", "regression"]:
        data = task_data(task)
        pipeline = make_pipeline(data["preprocessing_config"], main.MODELS[task].build_model(cfg[task]["model"]))
        pipeline.fit(data["X_dev"], data["y_dev"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"{task}.joblib"
            joblib.dump(pipeline, path)
            loaded = joblib.load(path)
        assert np.allclose(loaded.predict(data["X_kaggle"]), pipeline.predict(data["X_kaggle"]))
        new_records = data["kaggle_raw"].sample(3, random_state=0)
        predictions = main.predict_new_records(new_records, cfg, task, loaded, data["clientes"])
        positions = data["kaggle_raw"].index.get_indexer(new_records.index)
        assert np.allclose(predictions, pipeline.predict(data["X_kaggle"].iloc[positions]))
        assert len(predictions) == 3 and np.isfinite(predictions).all()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except Exception as error:                     # report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
