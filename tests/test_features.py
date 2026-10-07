import pandas as pd

from src.features import build_features


def make_tables():
    app = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "DAYS_EMPLOYED": [-500, 365243],
        "AMT_INCOME_TOTAL": [100000.0, 200000.0],
        "AMT_CREDIT": [200000.0, 100000.0],
        "AMT_ANNUITY": [10000.0, 5000.0],
        "EXT_SOURCE_1": [0.5, None],
        "EXT_SOURCE_2": [0.6, 0.4],
        "EXT_SOURCE_3": [None, None],
    })
    bureau = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_BUREAU": [10, 11],
        "CREDIT_ACTIVE": ["Active", "Closed"],
        "AMT_CREDIT_SUM_DEBT": [1000.0, 0.0],
        "AMT_CREDIT_SUM": [5000.0, 3000.0],
        "CREDIT_DAY_OVERDUE": [0, 0],
        "DAYS_CREDIT": [-400, -900],
    })
    prev = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_PREV": [100, 101, 102],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
        "AMT_APPLICATION": [1000.0, 2000.0, 3000.0],
        "AMT_CREDIT": [1000.0, 0.0, 3000.0],
    })
    return app, bureau, prev


def row(df, applicant_id):
    return df[df["SK_ID_CURR"] == applicant_id].iloc[0]


def test_one_row_per_applicant():
    df = build_features(*make_tables())
    assert len(df) == 2
    assert df["SK_ID_CURR"].is_unique


def test_employment_placeholder_is_fixed():
    df = build_features(*make_tables())
    assert pd.isna(row(df, 2)["DAYS_EMPLOYED"])
    assert row(df, 2)["DAYS_EMPLOYED_ANOM"] == 1
    assert row(df, 1)["DAYS_EMPLOYED_ANOM"] == 0


def test_thin_file_flags():
    df = build_features(*make_tables())
    assert row(df, 1)["N_BUREAU_CREDITS"] == 2
    assert row(df, 1)["THIN_FILE"] == 0
    assert row(df, 2)["N_BUREAU_CREDITS"] == 0
    assert row(df, 2)["THIN_FILE"] == 1


def test_external_score_missing_count():
    df = build_features(*make_tables())
    assert row(df, 1)["EXT_MISSING_COUNT"] == 1
    assert row(df, 2)["EXT_MISSING_COUNT"] == 2