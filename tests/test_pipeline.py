"""
Checks of the preprocessing pipeline. Run from the project root:

    python tests/test_pipeline.py        (or: python -m pytest tests, if pytest is installed)

They cover: feature lists (exist, no overlap), missingness indicators inside the pipeline,
every data-dependent step fitted inside the pipeline on training rows only, the optional
feature-selection step, the excluded features, the stateless/row-preserving cleaning, schema validation, placeholder tokens, saving, loading and
predicting new records with the complete pipeline, Optuna tuning with nested temporal CV,
and the Kaggle submission format.
"""
import copy
import json
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
from sklearn.utils.validation import check_is_fitted  # noqa: E402

import yaml  # noqa: E402

from src.cleaning import prepare_table  # noqa: E402
from src.data import load_data, report_placeholders, standardize_columns, validate_schema  # noqa: E402
from src.evaluate import select_scoring, temporal_folds  # noqa: E402
from src.pipeline import build_pipeline, predict_new_records, prepare_task_data  # noqa: E402
from src.cleaning import apply_consistency_rules, clean_dataset, fill_from_customer_history
from src.preprocessing import (
    RowBoundedImputer,
    SELECTION_METHODS,
    build_preprocessor,
    build_selector,
    check_feature_lists,
    drop_features,
    task_preprocessing_config,
)
from src.results import save_predictions  # noqa: E402
from src.tasks import TASKS  # noqa: E402
from src.tuning import nested_cross_validate, tune_pipeline  # noqa: E402
from src.customers import prepare_customers  # noqa: E402


@lru_cache(maxsize=None)
def config():
    with open("config.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


@lru_cache(maxsize=None)
def task_data(task):
    return prepare_task_data(config(), task)


def task_pipeline(task, model_config=None):
    return build_pipeline(task_data(task)["preprocessing_config"],
                          TASKS[task].model.build_model(model_config or config()[task]["model"]))


def classification_pipeline():
    return task_pipeline("classification")


def expect_error(function, *args):
    try:
        function(*args)
    except ValueError as error:
        return str(error)
    raise AssertionError(f"{function.__name__} did not raise ValueError")


# ---------------------------------------------------------------- feature lists
def test_purchase_category_overrides_leave_profile_imputation_and_order_unchanged():
    from src.preprocessing import build_categorical_imputer
    columns = ["payment_type", "municipality", "transport_method"]
    train = pd.DataFrame({"payment_type": ["card", "card", "cash"], "municipality": ["A", "A", "B"],
                          "transport_method": ["van", "van", "truck"]})
    query = pd.DataFrame([[np.nan, np.nan, np.nan], ["new", "C", "other"]], columns=columns)
    settings = {"categorical_strategy": "constant", "categorical_fill_value": "Missing",
                "categorical_overrides": {"payment_type": {"strategy": "most_frequent"},
                                          "transport_method": {"strategy": "most_frequent"}}}
    imputer = build_categorical_imputer(settings, columns).fit(train)
    assert imputer.transform(query).tolist() == [["card", "Missing", "van"], ["new", "C", "other"]]
    assert list(imputer.get_feature_names_out()) == columns
    assert train.isna().sum().sum() == 0, "fitting must not alter training purchases"
    recipe = {"numeric_features": [], "categorical_features": columns, "imputation": settings}
    reduced = drop_features(recipe, ["transport_method"])
    assert "transport_method" not in reduced["imputation"]["categorical_overrides"]
    check_feature_lists(reduced)
    invalid = {**recipe, "categorical_features": ["municipality"]}
    assert "categorical_overrides" in expect_error(check_feature_lists, invalid)


def test_history_experiment_changes_only_purchase_category_choices():
    from src.experiments import categorical_history_variants
    recipe = copy.deepcopy(task_data("regression")["preprocessing_config"])
    variants, columns = categorical_history_variants(recipe)
    assert len(variants) == 4 and "municipality" not in columns
    for name, candidate in variants.items():
        rules = candidate["customer_history"]["diagnostics"]["customer_history"]
        assert rules["previous_purchase_gap"] == recipe["customer_history"]["diagnostics"]["customer_history"]["previous_purchase_gap"]
        assert rules["use"], "date-gap history must remain enabled for all candidates"
        assert candidate["imputation"]["categorical_strategy"] == recipe["imputation"]["categorical_strategy"]
        assert "municipality" not in candidate["imputation"]["categorical_overrides"]
        assert candidate["numeric_features"] == recipe["numeric_features"]
        if name.startswith("no category history"):
            assert rules["last_observed_columns"] == []
    assert recipe["customer_history"]["diagnostics"]["customer_history"]["last_observed_columns"], "recipes must be independent copies"


def test_ticket_ratio_compares_tax_inclusive_amounts_and_handles_zero_history():
    from src.cleaning import add_derived_features
    columns = {"ticket_vs_usual": config()["diagnostics"]["derived_features"]["ticket_vs_usual"]}
    purchase = pd.DataFrame({"net_total": [100., 0., 100.], "total_tax": [23., 0., 23.],
                             "prior_mean_purchase_value": [123., 0., 0.]})
    result = add_derived_features(purchase, columns)
    assert result.ticket_vs_usual.iloc[0] == 1., "a 123 euro current purchase should equal a 123 euro historical mean"
    assert result.ticket_vs_usual.iloc[1:].isna().all(), "zero history must not produce infinity"
    for task in TASKS:
        data = task_data(task)
        assert "ticket_vs_usual" not in data["preprocessing_config"]["numeric_features"], "the experiment must not enable the feature"
        for part in ["X_dev", "X_kaggle"]:
            X = data[part]
            expected = ((X.net_total + X.total_tax) / X.prior_mean_purchase_value).replace([np.inf, -np.inf], np.nan)
            assert np.allclose(X.ticket_vs_usual, expected, equal_nan=True)


def test_history_is_refitted_at_outer_inner_and_holdout_boundaries():
    from unittest.mock import patch
    from sklearn.base import clone
    from src.history import CustomerHistoryFiller

    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2021-01-01", "2021-02-01",
                            "2022-01-01", "2022-02-01"])
    source = pd.DataFrame({"id": range(6), "customer_id": [1] * 6, "purchase_date": dates,
                           "prior_purchase_count": range(6), "payment_type": ["A", "A", "B", "B", "C", "C"],
                           "spend_next_90d": [999.] * 6})
    diagnostics = {"customer_history": {"use": True, "key": "customer_id", "date_column": "purchase_date",
                                        "order_column": "prior_purchase_count", "last_observed_columns": ["payment_type"]}}
    cfg = {"data": {"train_path": "mock.csv", "id_column": "id"}, "diagnostics": diagnostics}
    queries = source.drop(columns="spend_next_90d").iloc[[4, 5]].copy()
    queries.loc[queries.index[1], "payment_type"] = np.nan
    outer_train = source.iloc[:2].drop(columns="spend_next_90d")
    with patch("src.history.load_data", return_value=source), patch("src.history.prepare_table", return_value=(source, None)):
        outer = CustomerHistoryFiller(cfg).fit(outer_train)
        assert outer.transform(queries).iloc[1].payment_type == "A", "query C must not fill the later validation query"
        assert not set(outer.history_.id) & {2, 3, 4, 5}, "embargo and future purchases stay outside the source"
        later = clone(outer).fit(source.iloc[:4].drop(columns="spend_next_90d"))
        assert later.transform(queries).iloc[1].payment_type == "B"
        inner = clone(later).fit(outer_train)
        assert inner.transform(queries).iloc[1].payment_type == "A", "inner fit must rebuild history, not reuse outer fill"
        assert "spend_next_90d" not in inner.history_, "target values must not be stored as history"
        # A raw past purchase excluded by task filtering still belongs to the time-authorised source.
        sparse_labelled_train = source.iloc[[0, 3]].drop(columns="spend_next_90d")
        past = CustomerHistoryFiller(cfg).fit(sparse_labelled_train)
        assert 2 in set(past.history_.id)
        alone = outer.transform(queries.iloc[[1]])
        assert alone.payment_type.iloc[0] == outer.transform(queries).payment_type.iloc[1]


def test_numeric_transforms_follow_original_unit_imputation_and_fit_only_training():
    from src.outliers import SkewLog1p, Winsorizer
    features = ["purchase_count_365d", "prior_mean_purchase_value", "spend_180d", "spend_365d", "prior_total_spend"]
    train = pd.DataFrame([[2., 20., 30., 40., 100.], [3., 20., 50., 60., 100.],
                          [4., 20., 60., 80., 100.], [5., 20., 60., 100., 100.]], columns=features)
    query = pd.DataFrame([[2., 20., 30., np.nan, 100.]], columns=features)
    recipe = {"numeric_features": features, "categorical_features": [], "scaler": "none", "encoder": "ordinal",
              "imputation": {"column_imputers": {"spend_365d": {"strategy": "formula", "expression": "purchase_count_365d * prior_mean_purchase_value"}},
                             "row_bounds": {"spend_365d": {"min": "spend_180d", "max": "prior_total_spend"}}}}
    log_prep = build_preprocessor({**recipe, "numeric_transform": SkewLog1p(features, minimum_skew=-1)}).fit(train)
    numeric = log_prep.named_transformers_["numeric"]
    assert list(numeric.named_steps) == ["impute", "outlier_transform", "scale"]
    raw_filled = numeric.named_steps["impute"].transform(query)
    assert raw_filled[0, 3] == 40., "2 purchases x 20 euros must be calculated before logarithms"
    assert np.isclose(numeric.transform(query)[0, 3], np.log1p(40.))
    cap_prep = build_preprocessor({**recipe, "numeric_transform": Winsorizer(features, .25, .75)}).fit(train)
    cap = cap_prep.named_transformers_["numeric"].named_steps["outlier_transform"]
    assert cap.upper_limits_[3] == 85., "quantiles come from the training fold only"
    extreme = query.copy()
    extreme["spend_365d"] = 100000.
    assert cap_prep.transform(extreme)[0, 3] == 85.
    assert cap.upper_limits_[3] == 85., "prediction must not relearn quantiles"


def test_paired_comparisons_are_descriptive_and_directional():
    from src.evaluate import paired_comparison
    mixed = paired_comparison([1., 4., 3.], [2., 2., 2.])
    assert mixed["folds_a_better"] == 2 and not mixed["consistent_gain"]
    assert mixed["min_advantage"] == -1 and mixed["max_advantage"] == 2
    assert "decisive" not in mixed and "se" not in mixed
    errors = paired_comparison([1., 2.], [2., 3.], higher_is_better=False)
    assert errors["consistent_gain"] and errors["mean_advantage"] == 1.
    assert not paired_comparison([1.], [0.])["consistent_gain"]
    assert paired_comparison([1., 1.], [1., 1.])["folds_tied"] == 2
    assert "finite" in expect_error(paired_comparison, [np.nan], [1.])
    assert "equally sized" in expect_error(paired_comparison, [1., 2.], [1.])


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
    context = {**recipe, "numeric_features": recipe["numeric_features"] + ["id"]}
    assert "history context" in expect_error(check_feature_lists, context)


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
    assert isinstance(imputer, RowBoundedImputer) and isinstance(imputer.imputer_, SimpleImputer)
    learned = pd.Series(imputer.imputer_.statistics_, index=numeric_features)

    def medians(rows):   # per column, of the rows its `fit_rows` condition keeps (structural zeros left out)
        out = rows[numeric_features].median()
        for column, condition in data["preprocessing_config"]["imputation"].get("fit_rows", {}).items():
            out[column] = rows.loc[rows.eval(condition), column].median()
        return out

    history_step = pipeline.named_steps["history"]
    filled_train = history_step.transform(X_train)
    assert np.allclose(learned, medians(filled_train)), "medians must come from history-filled training rows"
    assert history_step.cutoff_ == X_train[config()["data"]["date_column"]].max()
    assert not np.allclose(learned, medians(history_step.transform(data["X_dev"]))), "medians must not see the other rows"

    scaler = prep.named_transformers_["numeric"].named_steps["scale"]
    encoder = prep.named_transformers_["categorical"].named_steps["encode"]
    for step in [imputer, scaler, encoder, prep.named_transformers_["indicators"]]:
        assert getattr(step, "n_features_in_", None) is not None, f"{type(step).__name__} is not fitted"
    unseen = set(data["X_dev"].iloc[half:]["transport_method"].dropna()) - set(X_train["transport_method"].dropna())
    known = set(encoder.categories_[data["preprocessing_config"]["categorical_features"].index("transport_method")])
    assert not (unseen & known), "the encoder must only know categories of the training rows"


# ---------------------------------------------------------------- feature selection
def test_selection_none_adds_no_step_and_unknown_methods_are_rejected():
    for task in TASKS:
        recipe = task_data(task)["preprocessing_config"]
        assert recipe.get("selection", {}).get("method", "none") == "none", "the default must not select"
        assert list(task_pipeline(task).named_steps) == ["history", "prep", "model"], "method none must add no selector"
    assert "Unknown selection method" in expect_error(build_selector, {"method": "not_a_method"}, True)


def test_selection_step_is_fitted_inside_the_pipeline_on_training_rows_only():
    for task in TASKS:
        data = task_data(task)
        half = len(data["X_dev"]) // 2
        X_train, y_train = data["X_dev"].iloc[:half], data["y_dev"].iloc[:half]
        for method in [m for m in SELECTION_METHODS if m != "none"]:
            recipe = {**data["preprocessing_config"], "selection": {"method": method, "k": 10}}
            pipeline = build_pipeline(recipe, TASKS[task].model.build_model(config()[task]["model"]))
            assert list(pipeline.named_steps) == ["history", "prep", "select", "model"], f"{task}/{method}: step order"
            try:
                check_is_fitted(pipeline.named_steps["select"])
                raise AssertionError(f"{task}/{method}: the selector must not be fitted before fit")
            except NotFittedError:
                pass

            pipeline.fit(X_train, y_train)
            width = pipeline.named_steps["prep"].transform(X_train).shape[1]
            support = pipeline.named_steps["select"].get_support()
            assert len(support) == width and 0 < support.sum() <= width, f"{task}/{method}: bad support"
            assert len(pipeline[:-1].get_feature_names_out()) == support.sum(), f"{task}/{method}: names"
            if method == "kbest":
                assert support.sum() == 10, "kbest must keep exactly k columns"
            predictions = pipeline.predict(data["X_kaggle"])
            assert len(predictions) == len(data["X_kaggle"]) and not pd.isna(predictions).any()

# ---------------------------------------------------------------- excluded features
def config_without_exclusions():
    """config.yaml as if no column were excluded (the real exclusions are tested by the pipeline tests)."""
    plain = copy.deepcopy(config())
    plain["preprocessing"].pop("exclude_features", None)
    for task in TASKS:
        plain[task]["preprocessing"].pop("exclude_features", None)
    return plain


def test_exclude_features_removes_columns_shared_and_per_task():
    plain = config_without_exclusions()
    cfg = copy.deepcopy(plain)
    cfg["preprocessing"]["exclude_features"] = ["warehouse", "shipping_cost"]
    cfg["classification"]["preprocessing"]["exclude_features"] = ["is_first_purchase"]
    cfg["regression"]["preprocessing"]["exclude_features"] = ["cost"]
    own = {"classification": "is_first_purchase", "regression": "cost"}
    for task in TASKS:
        base, recipe = task_preprocessing_config(plain, task), task_preprocessing_config(cfg, task)
        gone = {"warehouse", "shipping_cost", own[task]}
        for key in ["numeric_features", "categorical_features", "flag_features"]:
            assert recipe[key] == [c for c in base[key] if c not in gone], f"{task}/{key}: only the excluded go"
        assert "exclude_features" not in recipe, "the key is consumed, not passed on"
        check_feature_lists(recipe, task_data(task)["X_dev"].columns)
        build_preprocessor(recipe)
    # a task's own exclusions never reach the other task
    assert "cost" in task_preprocessing_config(cfg, "classification")["numeric_features"]
    assert "is_first_purchase" in task_preprocessing_config(cfg, "regression")["flag_features"]
    bad = copy.deepcopy(plain)
    bad["classification"]["preprocessing"]["exclude_features"] = ["not_a_column"]
    assert "exclude_features" in expect_error(task_preprocessing_config, bad, "classification")


def test_dropping_a_column_removes_only_the_imputation_settings_that_read_it():
    recipe = task_preprocessing_config(config_without_exclusions(), "regression")
    before = copy.deepcopy(recipe)
    limits = lambda r, column: r["imputation"]["row_bounds"][column]

    out = drop_features(recipe, ["spend_180d"])                     # one side of a limit goes, the other stays
    assert limits(out, "spend_365d")["min"] == [
        "purchase_count_365d * prior_min_purchase_value",
        "prior_total_spend - (prior_purchase_count - purchase_count_365d) * prior_max_purchase_value"]
    assert "prior_total_spend" in limits(out, "spend_365d")["max"]
    out = drop_features(recipe, ["prior_total_spend"])              # it is read by two limits
    assert all("prior_total_spend" not in e for column in ["spend_365d", "prior_max_purchase_value"]
               for e in limits(out, column)["max"])
    assert limits(out, "prior_max_purchase_value")["max"] == [
        "prior_mean_purchase_value + prior_std_purchase_value * sqrt(abs(prior_purchase_count - 1))"]
    assert "min" in limits(out, "prior_max_purchase_value")
    out = drop_features(recipe, ["prior_max_purchase_value"])       # a column's own settings go with it
    assert "prior_max_purchase_value" not in out["imputation"]["row_bounds"]
    assert "prior_max_purchase_value" not in out["imputation"]["fit_rows"]
    out = drop_features(recipe, ["prior_mean_purchase_value"])      # read by the spend_365d estimate
    assert "spend_365d" not in out["imputation"]["column_imputers"]
    for dropped in (["spend_180d"], ["prior_total_spend"], ["prior_max_purchase_value"], ["prior_mean_purchase_value"]):
        check_feature_lists(drop_features(recipe, dropped))         # the result is always a valid recipe
    assert recipe == before, "drop_features must not change its input"
    assert drop_features(recipe, ["not_a_column"]) == recipe, "unknown names are ignored"


def test_config_exclusions_reach_the_recipe_of_each_task():
    shared = set(config()["preprocessing"].get("exclude_features", []))
    for task in TASKS:
        declared = shared | set((config()[task].get("preprocessing") or {}).get("exclude_features", []))
        recipe = task_data(task)["preprocessing_config"]
        listed = set(recipe["numeric_features"] + recipe["categorical_features"] + list(recipe.get("flag_features", [])))
        assert declared and not declared & listed, f"{task}: excluded columns still in the recipe: {sorted(declared & listed)}"
        assert declared <= set(task_data(task)["X_dev"].columns), f"{task}: an excluded column does not exist in the data"


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


def test_exact_recovery_of_prior_max_and_min():
    cfg = config()
    table = pd.DataFrame({"prior_purchase_count": [1, 1, 2, 2], "prior_total_spend": [80.0, 50.0, 100.0, 90.0],
                          "prior_max_purchase_value": [np.nan, 50.0, np.nan, 60.0],
                          "prior_min_purchase_value": [80.0, np.nan, 30.0, np.nan]})
    rules = [r for r in cfg["diagnostics"]["recompute_rules"] if r["column"].startswith(("prior_max", "prior_min"))]
    out, _ = clean_dataset(table, {"recompute_rules": rules})
    assert out["prior_max_purchase_value"].tolist() == [80.0, 50.0, 70.0, 60.0]
    assert out["prior_min_purchase_value"].tolist() == [80.0, 50.0, 30.0, 30.0]


def test_maximum_must_agree_with_history_moments():
    # A genuine history [10, 30, 50] has mean 30 and sample std 20.
    # 80 is below the total 90, but cannot be its maximum with those moments.
    table = pd.DataFrame({
        "prior_purchase_count": [3, 3, 0, 1, 3],
        "prior_total_spend": [90., 90., 0., 10., 90.],
        "prior_std_purchase_value": [20., 20., 0., np.nan, np.nan],
        "prior_max_purchase_value": [50., 80., 0., 10., 80.],
    })
    rules = [r for r in config()["diagnostics"]["consistency_rules"] if "sqrt(" in r["rule"]]
    with np.errstate(invalid="raise"):
        report = apply_consistency_rules(table, rules)
    assert table.prior_max_purchase_value.isna().tolist() == [False, True, False, False, False]
    assert report["rows_checked"].tolist() == [2] and report["values"].tolist() == [1]


def test_annual_spend_must_agree_with_window_counts_and_extremes():
    # History [20, 50, 100], of which 50 is in 180d and 20+50 in 365d.
    # Annual 160 respects total=170 and count365*max=200, but its additional
    # purchase would cost 110 > max100. Annual 55 makes it 5 < min20.
    table = pd.DataFrame({
        "prior_purchase_count": [3] * 5,
        "prior_total_spend": [170.] * 5,
        "prior_min_purchase_value": [20.] * 5,
        "prior_max_purchase_value": [100.] * 5,
        "purchase_count_180d": [1] * 5,
        "spend_180d": [50., 50., 50., 50., np.nan],
        "purchase_count_365d": [2] * 5,
        "spend_365d": [70., 160., 55., 70.03, 70.],
    })
    rules = [r for r in config()["diagnostics"]["consistency_rules"]
             if r.get("invalidate") == "spend_365d"]
    apply_consistency_rules(table, rules)
    assert table.spend_365d.isna().tolist() == [False, True, True, False, False]


def test_additional_real_data_inconsistencies_are_marked_without_dropping_rows():
    cfg = config()
    for name, bad_max, bad_spend in [("train", [10003253, 10003727, 10004737], [10005752]),
                                    ("test", [10007090], [])]:
        raw = load_data(cfg["data"][f"{name}_path"], cfg["diagnostics"]["placeholder_tokens"])
        table, _ = prepare_table(raw, cfg)
        assert table.id.tolist() == raw.ID.tolist()
        assert table.loc[table.id.isin(bad_max), "prior_max_purchase_value"].isna().all()
        assert table.loc[table.id.isin(bad_spend), "spend_365d"].isna().all()
        # Each affected row must get the same result without any other purchase.
        for identifier, column in [(i, "prior_max_purchase_value") for i in bad_max] + [(i, "spend_365d") for i in bad_spend]:
            alone, _ = prepare_table(raw[raw.ID.eq(identifier)], cfg)
            assert len(alone) == 1 and alone[column].isna().all()


def test_multiple_row_limits_skip_missing_terms_and_preserve_observed_values():
    features = ["value", "upper_a", "upper_b", "divisor", "lower"]
    train = pd.DataFrame([[500., 900., 900., 1., 0.]] * 2, columns=features)
    bounds = {"value": {"min": "lower", "max": ["upper_a", "upper_b / divisor"]}}
    imputer = RowBoundedImputer(SimpleImputer(strategy="median"), features, bounds).fit(train)
    new = pd.DataFrame([[np.nan, 100., 200., 1., 0.], [np.nan, 200., 100., 1., 0.],
                        [np.nan, 100., 0., 0., 0.], [700., 100., 100., 1., 0.]], columns=features)
    # The third row's second limit is undefined (0/0); the first still applies.
    assert imputer.transform(new)[:, 0].tolist() == [100., 100., 100., 700.]


def test_imputed_history_respects_the_new_limits_for_both_tasks():
    for task in TASKS:
        data = task_data(task)
        prep = build_preprocessor(data["preprocessing_config"]).fit(data["X_dev"], data["y_dev"])
        features = data["preprocessing_config"]["numeric_features"]
        imputer = prep.named_transformers_["numeric"].named_steps["impute"]
        for rows in [data["X_dev"], data["X_kaggle"]]:
            filled = pd.DataFrame(imputer.transform(rows[features]), columns=features, index=rows.index)
            assert np.isfinite(filled.to_numpy()).all()
            for column, limits in data["preprocessing_config"]["imputation"]["row_bounds"].items():
                missing = rows[column].isna()
                assert np.allclose(filled.loc[~missing, column], rows.loc[~missing, column])
                for side, expressions in limits.items():
                    for expression in expressions if isinstance(expressions, list) else [expressions]:
                        bound = filled.eval(str(expression))
                        checked = missing & bound.notna()
                        margin = filled[column] - bound
                        assert ((margin[checked] >= -1e-7) if side == "min" else (margin[checked] <= 1e-7)).all(), (task, column, side, expression)


def test_history_fill_uses_the_customers_past_purchases_only():
    cfg = config()
    day = pd.Timestamp("2024-01-01")
    history = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "customer_id": [7, 7, 7, 7, 8],
        "purchase_date": [day, day + pd.Timedelta(days=10), day + pd.Timedelta(days=25), day + pd.Timedelta(days=40), day],
        "prior_purchase_count": [0, 1, 2, 4, 0],                      # id 4: one purchase is not in the data
        "payment_type": ["A", None, "B", None, None],
        "days_since_previous_purchase": [0.0, 10.0, np.nan, np.nan, 0.0],
        "customer_tenure_days": [0.0, 10.0, 25.0, 40.0, 0.0],
        "prior_mean_gap_days": [0.0, 0.0, np.nan, np.nan, 0.0],
        "prior_total_spend": [0.0, 1.0, 2.0, 4.0, 0.0],
    })
    diagnostics = {**cfg["diagnostics"], "customer_history": {**cfg["diagnostics"]["customer_history"],
                                                              "last_observed_columns": ["payment_type"]}}
    out, _ = fill_from_customer_history(history, history, diagnostics, "id")
    # id 2 gets A (its past); id 4 gets B, the LAST observed value; id 1 / id 5 have no past -> stay missing
    assert out["payment_type"].tolist()[1:4] == ["A", "B", "B"] and out["payment_type"].isna().sum() == 1
    assert out.loc[2, "days_since_previous_purchase"] == 15, "date difference to the previous purchase"
    assert out.loc[2, "prior_mean_gap_days"] == 10, "then recomputed from the identity (25 - 15) / (2 - 1)"
    assert pd.isna(out.loc[3, "days_since_previous_purchase"]), "the previous purchase is missing from the data"
    later_only = fill_from_customer_history(history.iloc[[1]], history.iloc[[1, 2]], diagnostics, "id")[0]
    assert later_only["payment_type"].isna().all(), "a later purchase must never fill an earlier one"


def test_history_fill_never_uses_other_query_rows():
    day = pd.Timestamp("2024-01-01")
    allowed = pd.DataFrame({
        "id": [1, 2, 3], "customer_id": [7, 7, 8],
        "purchase_date": [day, day + pd.Timedelta(days=10), day],
        "prior_purchase_count": [0, 1, 0], "payment_type": ["A", None, "Other"],
        "days_since_previous_purchase": [0., 10., 0.],
    })
    queries = pd.DataFrame({
        "id": [4, 5, 6], "customer_id": [7, 7, 9],
        "purchase_date": [day + pd.Timedelta(days=d) for d in [20, 30, 30]],
        "prior_purchase_count": [2, 3, 0], "payment_type": ["B", None, None],
        "days_since_previous_purchase": [np.nan, np.nan, np.nan],
    }, index=[30, 10, 20])  # preserve the caller's non-sequential index/order
    diagnostics = {"customer_history": {"use": True, "key": "customer_id",
                   "date_column": "purchase_date", "order_column": "prior_purchase_count",
                   "last_observed_columns": ["payment_type"],
                   "previous_purchase_gap": "days_since_previous_purchase"}}
    snapshot = allowed.copy(deep=True)
    batch, _ = fill_from_customer_history(queries, allowed, diagnostics)
    assert batch.loc[10, "payment_type"] == "A", "earlier query B is not training history"
    assert pd.isna(batch.loc[10, "days_since_previous_purchase"]), "the immediately prior purchase is a query, not allowed history"
    assert batch.loc[30, "days_since_previous_purchase"] == 10
    assert pd.isna(batch.loc[20, "payment_type"]), "a new customer has no authorised past"
    for position in range(len(queries)):
        alone, _ = fill_from_customer_history(queries.iloc[[position]], allowed, diagnostics)
        pd.testing.assert_frame_equal(alone, batch.iloc[[position]])
    shuffled, _ = fill_from_customer_history(queries.iloc[::-1], allowed, diagnostics)
    pd.testing.assert_frame_equal(shuffled, batch.iloc[::-1])
    no_history, _ = fill_from_customer_history(queries, None, diagnostics)
    assert no_history.loc[[10, 20], "payment_type"].isna().all()
    assert no_history.days_since_previous_purchase.isna().all()
    pd.testing.assert_frame_equal(allowed, snapshot)

    # A training value on the exact query date is excluded, including the query's own id.
    same_day = allowed.iloc[[1]].copy()
    same_day["payment_type"] = None
    with_same_day = allowed.copy()
    with_same_day.loc[1, "payment_type"] = "Same-day value"
    result, _ = fill_from_customer_history(same_day, with_same_day, diagnostics)
    assert result.payment_type.iloc[0] == "A"
    # Missing keys/dates and an empty source do not manufacture a previous purchase.
    unavailable = queries.copy()
    unavailable.loc[10, "purchase_date"] = pd.NaT
    unavailable.loc[20, "customer_id"] = np.nan
    unavailable["customer_id"] = unavailable["customer_id"].astype("Int64")
    matching_types = allowed.astype({"customer_id": "Int64"})
    result, _ = fill_from_customer_history(unavailable, matching_types, diagnostics)
    assert result.loc[[10, 20], "payment_type"].isna().all()
    result, _ = fill_from_customer_history(queries, allowed.iloc[:0], diagnostics)
    pd.testing.assert_frame_equal(result, no_history)


def test_prediction_history_contains_training_rows_only():
    cfg = config()
    train_ids = set(load_data(cfg["data"]["train_path"])["ID"])
    for task in TASKS:
        data = task_data(task)
        assert set(data["history"].id) == train_ids
        assert not set(data["history"].id) & set(data["kaggle_raw"].ID)


def test_raw_predictions_are_independent_of_batch_context():
    cfg = config()
    for task in TASKS:
        data = task_data(task)
        model = task_pipeline(task).fit(data["X_dev"], data["y_dev"])
        raw = data["kaggle_raw"]
        # Exercise the missing categorical fields for which history is actually used.
        columns = cfg["diagnostics"]["customer_history"]["last_observed_columns"]
        missing = raw[columns].isna().any(axis=1)
        probes = raw.loc[missing].head(12)
        assert len(probes) == 12
        positions = raw.index.get_indexer(probes.index)
        expected = model.predict(data["X_kaggle"].iloc[positions])
        batch = predict_new_records(probes, cfg, task, model, data["clientes"], data["history"])
        reversed_batch = predict_new_records(probes.iloc[::-1], cfg, task, model, data["clientes"], data["history"])
        assert np.allclose(batch, expected) and np.allclose(reversed_batch[::-1], expected)
        without_history = predict_new_records(probes, cfg, task, model, data["clientes"])
        for position in range(len(probes)):
            one = probes.iloc[[position]]
            assert np.allclose(predict_new_records(one, cfg, task, model, data["clientes"], data["history"]), expected[position])
            assert np.allclose(predict_new_records(one, cfg, task, model, data["clientes"]), without_history[position])
        if task == "classification":
            # Compare probabilities too: equal labels alone can hide different features.
            class Probabilities:
                def predict(self, X):
                    return model.predict_proba(X)
            probabilities = predict_new_records(probes, cfg, task, Probabilities(), data["clientes"], data["history"])
            expected_probabilities = model.predict_proba(data["X_kaggle"].iloc[positions])
            assert np.allclose(probabilities, expected_probabilities)
            for position in range(len(probes)):
                alone = predict_new_records(probes.iloc[[position]], cfg, task, Probabilities(), data["clientes"], data["history"])
                assert np.allclose(alone, probabilities[[position]])


def test_row_bounds_only_change_imputed_values():
    features = ["spend_180d", "spend_365d", "prior_total_spend"]
    imputer = RowBoundedImputer(SimpleImputer(strategy="median"), features,
                                {"spend_365d": {"min": "spend_180d", "max": "prior_total_spend"}})
    train = pd.DataFrame({"spend_180d": [0, 10, 20, 30.0], "spend_365d": [500, 500, 500, 500.0],
                          "prior_total_spend": [900, 900, 900, 900.0]})
    imputer.fit(train)
    # row 3: an observed value above its limit is NOT touched (consistency rules handle those)
    new = pd.DataFrame({"spend_180d": [600, 10, 10, 1.0], "spend_365d": [np.nan, np.nan, 2000, np.nan],
                        "prior_total_spend": [1000, 100, 1500, np.nan]})
    out = imputer.transform(new)[:, 1]
    assert out.tolist() == [600, 100, 2000, 500], "raised to the 180-day spend, cut to the total, observed kept, NaN limit skipped"


def test_column_estimates_and_structural_rows_in_the_imputer():
    features = ["prior_purchase_count", "purchase_count_365d", "prior_mean_purchase_value", "spend_365d", "prior_max_purchase_value"]
    train = pd.DataFrame({
        "prior_purchase_count": [0, 0, 0, 2, 5, 12],
        "purchase_count_365d": [0, 0, 0, 2, 3, 6],
        "prior_mean_purchase_value": [0, 0, 0, 100, 200, 300.0],
        "spend_365d": [0, 0, 0, 210, 590, 1800.0],
        "prior_max_purchase_value": [0, 0, 0, 120, 400, 900.0],   # 0 on first purchases = "no history"
    })
    new = pd.DataFrame({"prior_purchase_count": [4, 8, 30], "purchase_count_365d": [5, 2, np.nan],
                        "prior_mean_purchase_value": [100, 50, 10.0], "spend_365d": [np.nan, np.nan, np.nan],
                        "prior_max_purchase_value": [np.nan, 77, np.nan]})
    formula = {"spend_365d": {"strategy": "formula", "expression": "purchase_count_365d * prior_mean_purchase_value"}}
    out = pd.DataFrame(RowBoundedImputer(SimpleImputer(strategy="median"), features, column_imputers=formula,
                                         fit_rows={"prior_max_purchase_value": "prior_purchase_count > 0"})
                       .fit(train).transform(new), columns=features)
    assert out["spend_365d"].tolist() == [500, 100, 105], "count x mean per row; a row it can't compute falls back to the median"
    assert out["prior_max_purchase_value"].tolist() == [400, 77, 400], "median of rows WITH history (not 60 with the zeros); observed kept"

    bands = {"prior_max_purchase_value": {"strategy": "group_median", "by": "prior_purchase_count", "bins": [0, 1, 10, 1000],
                                          "fit_rows": "prior_purchase_count > 0"}}
    out = pd.DataFrame(RowBoundedImputer(SimpleImputer(strategy="median"), features, column_imputers=bands)
                       .fit(train).transform(new), columns=features)
    assert out["prior_max_purchase_value"].tolist() == [260, 77, 900], "median of the row's purchase-count band"


# ---------------------------------------------------------------- week 10: imputers as a setting
def _recipe_with(imputation_overrides, task="classification"):
    recipe = copy.deepcopy(task_data(task)["preprocessing_config"])
    recipe["imputation"] = {**recipe["imputation"], **imputation_overrides}
    return recipe


def test_every_numeric_imputer_is_a_setting_and_only_fills_gaps():
    from src.preprocessing import _NUMERIC_IMPUTERS
    data = task_data("classification")
    numeric = data["preprocessing_config"]["numeric_features"]
    history = data["X_dev"].iloc[:1500]
    for strategy in _NUMERIC_IMPUTERS:
        recipe = _recipe_with({"numeric_strategy": strategy, "n_estimators": 5, "max_iter": 2})
        prep = build_preprocessor(recipe).fit(history, data["y_dev"].iloc[:1500])
        step = prep.named_transformers_["numeric"].named_steps["impute"]
        X = history[numeric].astype("float64")
        out = step.transform(X)
        assert out.shape == X.shape and not np.isnan(out).any(), f"{strategy} must fill every gap"
        observed = X.notna().to_numpy()
        assert np.allclose(out[observed], X.to_numpy()[observed]), f"{strategy} must keep observed values"
        assert (step.scaler_ is not None) == (strategy in ("knn", "iterative", "random_forest")), \
            "model-based imputers work in scaled space"
    assert "numeric_strategy" in expect_error(build_preprocessor, _recipe_with({"numeric_strategy": "magic"}))


def test_categorical_helpers_are_seen_by_model_based_imputers_but_never_output():
    data = task_data("classification")
    numeric, categories = (data["preprocessing_config"][k] for k in ("numeric_features", "categorical_features"))
    train = data["X_dev"].iloc[:1500]
    for strategy, expected in [("knn", categories), ("median", [])]:
        recipe = _recipe_with({"numeric_strategy": strategy, "categorical_helpers": True})
        prep = build_preprocessor(recipe).fit(train, data["y_dev"].iloc[:1500])
        step = prep.named_transformers_["numeric"].named_steps["impute"]
        assert list(step.helper_features) == list(expected), "a median has no use for the categories"
        out = step.transform(train[numeric + list(expected)])
        assert out.shape == (len(train), len(numeric)), "the step outputs the numeric columns only"
    # categories it never saw, and gaps in them, are accepted at prediction time
    knn = build_preprocessor(_recipe_with({"numeric_strategy": "knn", "categorical_helpers": True}))
    knn.fit(train, data["y_dev"].iloc[:1500])
    new = train.head(3).copy()
    new[categories[0]] = new[categories[0]].astype(object)
    new.loc[new.index[0], categories[0]], new.loc[new.index[1], categories[0]] = "never seen", np.nan
    assert np.isfinite(knn.transform(new)).all()


def test_multiple_imputation_trains_one_pipeline_per_drawn_dataset_and_averages():
    from src.pipeline import MultipleImputationClassifier, MultipleImputationRegressor, fitted_imputer, imputer_label
    data = task_data("classification")
    recipe = _recipe_with({"numeric_strategy": "iterative", "n_imputations": 3, "random_state": 7})
    ensemble = build_pipeline(recipe, TASKS["classification"].model.build_model(config()["classification"]["model"]))
    assert isinstance(ensemble, MultipleImputationClassifier) and imputer_label(recipe) == "iterative x3"
    ensemble.fit(data["X_dev"], data["y_dev"])
    imputers = [fitted_imputer(p) for p in ensemble.pipelines_]
    assert [i.random_state for i in imputers] == [7, 8, 9], "one seed per completed dataset"
    assert all(p.get_params()["prep__numeric__impute__imputer__sample_posterior"] for p in ensemble.pipelines_), \
        "every copy is trained on a drawn dataset"
    first = ensemble.predict_proba(data["X_kaggle"])
    members = np.mean([p.predict_proba(data["X_kaggle"]) for p in ensemble.pipelines_], axis=0)
    assert np.allclose(first, members) and np.allclose(ensemble.predict_proba(data["X_kaggle"]), first), \
        "the average of the copies, reproducible: no new draw at prediction time"
    assert np.allclose(ensemble.predict_proba(data["X_kaggle"].iloc[::-1])[::-1], first), "independent of the batch"
    assert not np.allclose(*[p.predict_proba(data["X_dev"])[:, 1] for p in ensemble.pipelines_[:2]]), \
        "the copies differ: each learned from another completed dataset"
    assert set(np.unique(ensemble.predict(data["X_kaggle"]))) <= {0, 1}
    regression = build_pipeline(_recipe_with({"numeric_strategy": "iterative", "n_imputations": 2}, "regression"),
                                TASKS["regression"].model.build_model({"type": "ridge"}))
    assert isinstance(regression, MultipleImputationRegressor)
    assert "iterative" in expect_error(build_pipeline, _recipe_with({"numeric_strategy": "knn", "n_imputations": 2}),
                                       TASKS["classification"].model.build_model({"type": "dummy"}))


def test_corrected_paired_test_inflates_the_variance_of_fold_differences():
    from src.evaluate import corrected_paired_test, fold_test_train_ratio
    a, b = [0.72, 0.70, 0.74, 0.71, 0.73], [0.70, 0.69, 0.71, 0.70, 0.70]
    plain = corrected_paired_test(a, b, 0.0)
    corrected = corrected_paired_test(a, b, 0.25)
    assert corrected["mean_advantage"] > 0 and corrected["p_value"] > plain["p_value"], "correction -> larger p"
    assert np.isclose(corrected_paired_test(b, a, 0.25, higher_is_better=False)["t"], corrected["t"]), \
        "positive = A better, for errors too"
    folds = [(np.arange(100), np.arange(100, 150)), (np.arange(150), np.arange(150, 200))]
    assert np.isclose(fold_test_train_ratio(folds), (0.5 + 1 / 3) / 2)


# ---------------------------------------------------------------- week 10: predict.py
def _saved_models(folder):
    from src.pipeline import save_pipeline
    for task in TASKS:
        data = task_data(task)
        save_pipeline(task_pipeline(task).fit(data["X_dev"], data["y_dev"]), Path(folder) / f"{task}_pipeline.joblib")
    cfg = copy.deepcopy(config())
    cfg["output"]["models_dir"] = str(folder)
    path = Path(folder) / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_predict_cli_gives_one_prediction_per_input_row():
    import predict
    with tempfile.TemporaryDirectory() as folder:
        cfg_path = _saved_models(folder)
        out = Path(folder) / "predictions.csv"
        assert predict.main(["data/examples/new_purchases.csv", str(out), "--config", str(cfg_path)]) == 0
        new = pd.read_csv("data/examples/new_purchases.csv")
        written = pd.read_csv(out)
        assert len(written) == len(new) == 20, "rows out = rows in"
        assert written["ID"].tolist() == new["ID"].tolist(), "same ids, same order, duplicates kept"
        assert written.notna().all().all() and set(written["repeat_purchase_90d"]) <= {0, 1}
        assert written.iloc[14].equals(written.iloc[15]), "the same purchase twice gets the same prediction"
        # test.csv through predict.py = the pipeline on the prepared test.csv matrix
        test_out = Path(folder) / "test_predictions.csv"
        assert predict.main(["data/raw/test.csv", str(test_out), "--config", str(cfg_path)]) == 0
        loaded = joblib.load(Path(folder) / "classification_pipeline.joblib")
        expected = loaded.predict_proba(task_data("classification")["X_kaggle"])[:, 1]
        assert np.allclose(pd.read_csv(test_out)["repeat_probability"], expected, atol=1e-6)


def test_predict_stops_clearly_on_missing_columns_and_warns_on_other_library_versions():
    import predict
    from src.pipeline import load_pipeline, versions_path
    new = pd.read_csv("data/examples/new_purchases.csv").drop(columns=["Custo", "payment_type"])
    message = expect_error(predict.predict, new, {}, config())
    assert "cost" in message and "payment_type" in message, "names every missing column"
    with tempfile.TemporaryDirectory() as folder:
        cfg_path = _saved_models(folder)
        broken = Path(folder) / "broken.csv"
        new.to_csv(broken, index=False)
        assert predict.main([str(broken), str(Path(folder) / "out.csv"), "--config", str(cfg_path)]) == 1
        model = Path(folder) / "classification_pipeline.joblib"
        versions = json.loads(versions_path(model).read_text(encoding="utf-8"))
        assert versions["scikit-learn"] and versions["xgboost"], "the versions are saved next to the model"
        versions["scikit-learn"] = "0.0.1"
        versions_path(model).write_text(json.dumps(versions), encoding="utf-8")
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_pipeline(model)
        assert any("library versions" in str(w.message) for w in caught)


def test_schema_validation_flags_and_repairs():
    cfg = config()
    raw = load_data(cfg["data"]["test_path"]).head(5)
    table = standardize_columns(raw, cfg["diagnostics"]["column_renames"])
    table["payment_type"] = table["payment_type"].astype(object)
    table.loc[table.index[0], "payment_type"] = "Cash!!"            # not an allowed label
    table["cost"] = table["cost"].astype(object)
    table.loc[table.index[1], "cost"] = "twelve"                     # not a number
    table["purchase_date"] = table["purchase_date"].astype(object)
    table.loc[table.index[2], "purchase_date"] = str((pd.Timestamp.today() + pd.Timedelta(days=1)).date())  # in the future
    out, report = validate_schema(table, cfg["schema"], cfg["diagnostics"]["non_negative_columns"])
    counts = report.set_index(["step", "column"])["values"]
    assert counts[("schema: outside valid range", "purchase_date")] == 1, "a future purchase date is out of range"
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
        pipeline = task_pipeline(task)
        pipeline.fit(data["X_dev"], data["y_dev"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"{task}.joblib"
            joblib.dump(pipeline, path)
            loaded = joblib.load(path)
        assert np.allclose(loaded.predict(data["X_kaggle"]), pipeline.predict(data["X_kaggle"]))
        new_records = data["kaggle_raw"].sample(3, random_state=0)
        predictions = predict_new_records(new_records, cfg, task, loaded, data["clientes"], data["history"])
        positions = data["kaggle_raw"].index.get_indexer(new_records.index)
        assert np.allclose(predictions, pipeline.predict(data["X_kaggle"].iloc[positions]))
        assert len(predictions) == 3 and np.isfinite(predictions).all()


# ---------------------------------------------------------------- tuning + outputs
def test_tuning_and_nested_cv_run_on_temporal_folds():
    from unittest.mock import patch
    from src.history import CustomerHistoryFiller
    cfg = config()
    data = task_data("classification")
    scoring = select_scoring(TASKS["classification"].evaluate.SCORING, cfg["classification"]["metrics"])
    dates = data["extras_dev"][cfg["data"]["date_column"]]
    folds = temporal_folds(dates, 3, cfg["cv"]["gap_days"])
    space = cfg["classification"]["tuning"]["search_spaces"]["logistic_regression"]
    best, study = tune_pipeline(classification_pipeline(), data["X_dev"], data["y_dev"], folds,
                                scoring["accuracy"], space, n_trials=2, random_state=0)
    assert len(study.trials) == 2 and set(study.best_params) == set(space)
    original_fit, seen = CustomerHistoryFiller.fit, set()
    def record_fit(estimator, X, y=None):
        fitted = original_fit(estimator, X, y)
        seen.add((len(X), fitted.cutoff_))
        assert fitted.history_.purchase_date.le(fitted.cutoff_).all()
        assert not set(cfg["data"]["target_columns"]) & set(fitted.history_.columns)
        return fitted
    with patch.object(CustomerHistoryFiller, "fit", record_fit):
        scores, oof = nested_cross_validate(classification_pipeline(), data["X_dev"], data["y_dev"], dates, folds, 2,
                                            cfg["cv"]["gap_days"], scoring, "accuracy", space, 2, 0,
                                            TASKS["classification"].evaluate.predict_for_report)
    required = set()
    for train, _ in folds:
        outer_dates = dates.iloc[train]
        required.add((len(train), outer_dates.max()))
        for inner_train, _ in temporal_folds(outer_dates, 2, cfg["cv"]["gap_days"]):
            required.add((len(inner_train), outer_dates.iloc[inner_train].max()))
    assert required <= seen, "actual nested tuning must refit history at every inner/outer training boundary"
    assert len(scores) == len(folds) and {"inner_best", "validation_accuracy"} <= set(scores.columns)
    assert oof.notna().sum() == sum(len(v) for _, v in folds)


def test_classification_predictions_match_the_kaggle_sample_submission():
    cfg = config()
    data = task_data("classification")
    pipeline = classification_pipeline().fit(data["X_dev"], data["y_dev"])
    with tempfile.TemporaryDirectory() as folder:
        written = save_predictions(Path(folder) / "classification.csv", data["extras_kaggle"]["id"],
                                   pipeline.predict(data["X_kaggle"]), cfg["data"]["submission_id_column"],
                                   cfg["classification"]["target"], cfg["data"]["sample_submission_path"])
    sample = pd.read_csv(cfg["data"]["sample_submission_path"])
    assert list(written.columns) == list(sample.columns) and written["ID"].tolist() == sample["ID"].tolist()
    assert written["repeat_purchase_90d"].dtype == sample["repeat_purchase_90d"].dtype
    assert set(written["repeat_purchase_90d"].unique()) <= {0, 1}


# ---------------------------------------------------------------- customer table
def test_customer_table_pipeline():
    cfg = config()
    clientes = load_data(cfg["data"]["clientes_path"], cfg["diagnostics"]["placeholder_tokens"])
    table, report = prepare_customers(clientes, cfg["customers"])
    key = cfg["customers"]["key"]
    assert len(table) == len(clientes) and table[key].is_unique, "one row per customer"
    assert list(table.columns) == [key] + cfg["customers"]["columns"]
    for column in ["country_or_region", "commercial_region", "island", "municipality"]:
        assert not table[column].dropna().astype(str).str.contains(":").any(), f"{column} still holds labels"
    island_regions = set(table.loc[~table["island"].isin(["none"]) & table["island"].notna(), "commercial_region"])
    assert island_regions <= {"RG17", "RG29"}
    assert (table.loc[table["commercial_region"].notna() & ~table["commercial_region"].isin(island_regions), "island"] == "none").all()
    recovered = report[report["step"].str.contains("recovered")].groupby("column")["values"].sum()
    assert recovered["commercial_region"] == 71 and recovered["municipality"] == 5
    duplicated = pd.concat([clientes, clientes.head(1)])
    assert "repeated" in expect_error(prepare_customers, duplicated, cfg["customers"])
    # registration year: future and absurd years -> NaN, the current year is kept
    this_year = pd.Timestamp.today().year
    years = clientes.copy()
    years.loc[years.index[:3], "customer_since_year"] = [this_year + 1, 1850, this_year]
    since = prepare_customers(years, cfg["customers"])[0]["customer_since_year"]
    assert pd.isna(since.iloc[0]), "future registration year must become NaN"
    assert pd.isna(since.iloc[1]), "absurd registration year must become NaN"
    assert since.iloc[2] == this_year, "the current year is valid"


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
