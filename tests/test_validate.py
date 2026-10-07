import numpy as np
import pandas as pd
import pytest

from src.validate import DataValidationError, validate_data


def make_valid_tables(n=100):
    rng = np.random.default_rng(0)
    app = pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "TARGET": [1] * 8 + [0] * (n - 8),
        "CODE_GENDER": ["F"] * n,
        "DAYS_BIRTH": -rng.integers(8000, 24000, n),
        "DAYS_EMPLOYED": -rng.integers(100, 5000, n),
        "AMT_INCOME_TOTAL": rng.integers(50000, 300000, n).astype(float),
        "AMT_CREDIT": rng.integers(100000, 800000, n).astype(float),
        "AMT_ANNUITY": rng.integers(5000, 40000, n).astype(float),
        "EXT_SOURCE_1": rng.random(n),
        "EXT_SOURCE_2": rng.random(n),
        "EXT_SOURCE_3": rng.random(n),
    })
    bureau = pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "SK_ID_BUREAU": np.arange(n) + 1000,
        "CREDIT_ACTIVE": ["Active"] * n,
        "AMT_CREDIT_SUM_DEBT": [0.0] * n,
        "AMT_CREDIT_SUM": [1000.0] * n,
        "CREDIT_DAY_OVERDUE": [0] * n,
        "DAYS_CREDIT": [-300] * n,
    })
    prev = pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "SK_ID_PREV": np.arange(n) + 5000,
        "NAME_CONTRACT_STATUS": ["Approved"] * n,
        "AMT_APPLICATION": [1000.0] * n,
        "AMT_CREDIT": [1000.0] * n,
    })
    return app, bureau, prev


def test_valid_data_passes():
    app, bureau, prev = make_valid_tables()
    warnings = validate_data(app, bureau, prev, min_rows=10)
    assert isinstance(warnings, list)


def test_missing_column_is_caught():
    app, bureau, prev = make_valid_tables()
    app = app.drop(columns=["DAYS_BIRTH"])
    with pytest.raises(DataValidationError):
        validate_data(app, bureau, prev, min_rows=10)


def test_bad_target_value_is_caught():
    app, bureau, prev = make_valid_tables()
    app.loc[0, "TARGET"] = 2
    with pytest.raises(DataValidationError):
        validate_data(app, bureau, prev, min_rows=10)


def test_impossible_age_is_caught():
    app, bureau, prev = make_valid_tables()
    app.loc[0, "DAYS_BIRTH"] = 100
    with pytest.raises(DataValidationError):
        validate_data(app, bureau, prev, min_rows=10)


def test_duplicate_ids_are_caught():
    app, bureau, prev = make_valid_tables()
    app.loc[1, "SK_ID_CURR"] = 0
    with pytest.raises(DataValidationError):
        validate_data(app, bureau, prev, min_rows=10)


def test_absurd_default_rate_is_caught():
    app, bureau, prev = make_valid_tables()
    app["TARGET"] = 1
    with pytest.raises(DataValidationError):
        validate_data(app, bureau, prev, min_rows=10)