import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.serving import API_DATASET_COLUMNS, explain, make_model_input, risk_band, to_raw_row

MINIMAL = {"age_years": 30, "annual_income": 100000, "loan_amount": 200000, "loan_annuity": 10000}


def test_to_raw_row_converts_units():
    row = to_raw_row(dict(MINIMAL, years_employed=2))
    assert row["DAYS_BIRTH"] == -10950
    assert row["DAYS_EMPLOYED"] == -730
    assert row["AMT_CREDIT"] == 200000
    row = to_raw_row(dict(MINIMAL, no_employment_record=True, years_employed=5))
    assert row["DAYS_EMPLOYED"] == 365243


def test_to_raw_row_missing_optional_fields_are_nan():
    row = to_raw_row(MINIMAL)
    assert np.isnan(row["EXT_SOURCE_1"])
    assert np.isnan(row["DAYS_EMPLOYED"])
    assert set(row) == set(API_DATASET_COLUMNS)


def test_make_model_input_matches_training_features():
    features = [
        "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "DAYS_EMPLOYED",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "N_BUREAU_CREDITS", "N_PREV_APPS", "NO_BUREAU", "NO_PREV", "THIN_FILE",
        "DAYS_EMPLOYED_ANOM", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
        "PAYMENT_RATE", "EXT_SOURCE_MEAN", "EXT_MISSING_COUNT",
        "ORGANIZATION_TYPE", "DAYS_REGISTRATION",
    ]
    bundle = {
        "features": features,
        "categories": {"ORGANIZATION_TYPE": ["A", "B"]},
        "fill_values": {"DAYS_REGISTRATION": -4000.0},
    }
    raw = pd.DataFrame([{
        "AMT_CREDIT": 200000.0, "AMT_INCOME_TOTAL": 100000.0,
        "AMT_ANNUITY": 10000.0, "ORGANIZATION_TYPE": "A",
    }])

    X = make_model_input(raw, bundle)

    assert list(X.columns) == features
    assert X["CREDIT_INCOME_RATIO"].iloc[0] == 2.0
    assert X["THIN_FILE"].iloc[0] == 1                    # no bureau info means thin-file
    assert X["EXT_MISSING_COUNT"].iloc[0] == 3
    assert X["DAYS_REGISTRATION"].iloc[0] == -4000.0      # filled with the typical value
    assert str(X["ORGANIZATION_TYPE"].dtype) == "category"


def test_explain_returns_top_reasons():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((300, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + 0.1 * rng.random(300) > 0.6).astype(int)
    model = LGBMClassifier(n_estimators=20, min_child_samples=5, verbose=-1).fit(X, y)

    reasons = explain(model, X.iloc[[0]], top_k=3)

    assert len(reasons) == 3
    assert {r["effect"] for r in reasons} <= {"raises risk", "lowers risk"}
    assert abs(reasons[0]["impact"]) >= abs(reasons[1]["impact"]) >= abs(reasons[2]["impact"])


def test_risk_band():
    assert risk_band(0.02) == "low"
    assert risk_band(0.08) == "medium"
    assert risk_band(0.30) == "high"