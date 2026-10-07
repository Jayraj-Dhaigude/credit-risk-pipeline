import numpy as np
import pandas as pd

from src import config


def load_raw():
    app = pd.read_csv(config.RAW_DIR / "application_train.csv")
    bureau = pd.read_csv(config.RAW_DIR / "bureau.csv")
    prev = pd.read_csv(config.RAW_DIR / "previous_application.csv")
    return app, bureau, prev


def aggregate_bureau(bureau):
    bureau = bureau.copy()
    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    agg = bureau.groupby("SK_ID_CURR").agg(
        N_BUREAU_CREDITS=("SK_ID_BUREAU", "size"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_OLDEST_CREDIT=("DAYS_CREDIT", "min"),
    )
    return agg.reset_index()


def aggregate_previous(prev):
    prev = prev.copy()
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    agg = prev.groupby("SK_ID_CURR").agg(
        N_PREV_APPS=("SK_ID_PREV", "size"),
        PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
        PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
        PREV_AMT_APP_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
    )
    return agg.reset_index()


def add_derived_features(df):
    """Row-wise features. Used by BOTH training and the API, so they can never drift apart."""
    df = df.copy()

    # no rows in the history tables really means zero past credits
    count_cols = ["N_BUREAU_CREDITS", "N_PREV_APPS"]
    df[count_cols] = df[count_cols].fillna(0)
    df["NO_BUREAU"] = (df["N_BUREAU_CREDITS"] == 0).astype(int)
    df["NO_PREV"] = (df["N_PREV_APPS"] == 0).astype(int)
    df["THIN_FILE"] = (df["N_BUREAU_CREDITS"] <= config.THIN_FILE_MAX_CREDITS).astype(int)

    # the 1,000-year placeholder
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # ratios (PAYMENT_RATE is the feature we called CREDIT_TERM in the notebook)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
    df["EXT_MISSING_COUNT"] = df[ext].isnull().sum(axis=1)
    return df


def build_features(app, bureau, prev):
    df = app.merge(aggregate_bureau(bureau), on="SK_ID_CURR", how="left")
    df = df.merge(aggregate_previous(prev), on="SK_ID_CURR", how="left")
    return add_derived_features(df)


def get_feature_columns(df):
    return [c for c in df.columns if c not in config.EXCLUDE_FROM_MODEL]


def to_model_matrix(df, categories=None):
    """Select model inputs and turn text columns into fixed categories.

    At training time, leave `categories` empty and it is learned from the data.
    At prediction time, pass the saved `categories` so new data is encoded
    exactly the same way as the training data.
    """
    X = df[get_feature_columns(df)].copy()
    if categories is None:
        text_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        categories = {c: sorted(X[c].dropna().unique().tolist()) for c in text_cols}
    for col, levels in categories.items():
        X[col] = pd.Categorical(X[col], categories=levels)
    return X, categories