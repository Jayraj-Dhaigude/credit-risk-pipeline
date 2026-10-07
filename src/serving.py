import numpy as np
import pandas as pd

from src import config
from src.features import add_derived_features, to_model_matrix


def _nan(x):
    return np.nan if x is None else x


def to_raw_row(a):
    """Turn the friendly API fields into dataset-style columns."""
    if a.get("no_employment_record"):
        days_employed = 365243              # the dataset's placeholder for 'no employment record'
    elif a.get("years_employed") is not None:
        days_employed = -a["years_employed"] * 365
    else:
        days_employed = np.nan

    return {
        "DAYS_BIRTH": -a["age_years"] * 365,
        "AMT_INCOME_TOTAL": a["annual_income"],
        "AMT_CREDIT": a["loan_amount"],
        "AMT_ANNUITY": a["loan_annuity"],
        "AMT_GOODS_PRICE": _nan(a.get("goods_price")),
        "DAYS_EMPLOYED": days_employed,
        "ORGANIZATION_TYPE": _nan(a.get("organization_type")),
        "NAME_EDUCATION_TYPE": _nan(a.get("education_type")),
        "OWN_CAR_AGE": _nan(a.get("own_car_age")),
        "EXT_SOURCE_1": _nan(a.get("ext_source_1")),
        "EXT_SOURCE_2": _nan(a.get("ext_source_2")),
        "EXT_SOURCE_3": _nan(a.get("ext_source_3")),
        "N_BUREAU_CREDITS": _nan(a.get("n_bureau_credits")),
        "BUREAU_ACTIVE_COUNT": _nan(a.get("bureau_active_count")),
        "BUREAU_DEBT_SUM": _nan(a.get("bureau_debt_sum")),
        "BUREAU_CREDIT_SUM": _nan(a.get("bureau_credit_sum")),
        "N_PREV_APPS": _nan(a.get("n_prev_apps")),
        "PREV_REFUSED_COUNT": _nan(a.get("prev_refused_count")),
        "PREV_APPROVED_COUNT": _nan(a.get("prev_approved_count")),
    }


# the dataset columns the API collects (kept in sync with to_raw_row automatically)
API_DATASET_COLUMNS = list(
    to_raw_row({"age_years": 30, "annual_income": 1, "loan_amount": 1, "loan_annuity": 1})
)


def make_model_input(raw, bundle):
    """Turn a table of applicant fields (dataset column names) into model-ready input.

    Columns we were not given: filled with the typical training value if the model
    never saw a gap there, otherwise left missing. Derived features are then computed
    by the same function training uses.
    """
    n = len(raw)
    fill_values = bundle.get("fill_values", {})

    columns = {}
    for col in bundle["features"]:
        if col in raw.columns:
            columns[col] = raw[col].to_numpy()
        elif col in fill_values:
            columns[col] = [fill_values[col]] * n
        else:
            columns[col] = np.full(n, np.nan)

    df = add_derived_features(pd.DataFrame(columns, index=raw.index))
    X, _ = to_model_matrix(df, categories=bundle["categories"])
    return X[bundle["features"]]


def explain(model, X, top_k=5):
    """The features that pushed this applicant's score the most, up or down.

    Uses LightGBM's built-in SHAP-style contributions (log-odds units: the sign is the
    direction, the size is the strength).
    """
    contributions = model.predict(X, pred_contrib=True)[0]   # last value is the baseline
    values = contributions[:-1]
    order = np.argsort(-np.abs(values))[:top_k]

    reasons = []
    for i in order:
        v = X.iloc[0, i]
        reasons.append({
            "feature": str(X.columns[i]),
            "value": None if pd.isna(v) else (v if isinstance(v, str) else float(v)),
            "effect": "raises risk" if values[i] > 0 else "lowers risk",
            "impact": round(float(values[i]), 3),
        })
    return reasons


def risk_band(probability):
    for limit, name in config.RISK_BANDS:
        if probability < limit:
            return name
    return "high"