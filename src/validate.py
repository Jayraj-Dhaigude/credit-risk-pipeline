import pandas as pd


class DataValidationError(Exception):
    """Raised when the input data is too broken to train on."""


REQUIRED = {
    "application": [
        "SK_ID_CURR", "TARGET", "CODE_GENDER", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    ],
    "bureau": [
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT",
    ],
    "previous_application": [
        "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS",
        "AMT_APPLICATION", "AMT_CREDIT",
    ],
}

# Largest share of missing values we accept.
# In the training data these were about 56%, 0.2% and 20%.
MAX_MISSING = {"EXT_SOURCE_1": 0.75, "EXT_SOURCE_2": 0.10, "EXT_SOURCE_3": 0.40}


def validate_data(app, bureau, prev, min_rows=10_000):
    """Check the three input tables.

    Raises DataValidationError listing every problem found.
    Returns a list of warnings (things that look odd but do not stop training).
    """
    tables = {"application": app, "bureau": bureau, "previous_application": prev}

    # 1. Required columns. Without them we cannot run the other checks.
    column_problems = []
    for name, table in tables.items():
        missing = [c for c in REQUIRED[name] if c not in table.columns]
        if missing:
            column_problems.append(f"{name} is missing columns {missing}")
    if column_problems:
        raise DataValidationError("; ".join(column_problems))

    errors, warnings = [], []

    # 2. Size and identity
    if len(app) < min_rows:
        errors.append(f"application has only {len(app)} rows (minimum {min_rows})")
    if not app["SK_ID_CURR"].is_unique:
        errors.append("application: SK_ID_CURR contains duplicates")

    # 3. The thing we predict
    if app["TARGET"].isnull().any() or not set(app["TARGET"].unique()) <= {0, 1}:
        errors.append("TARGET must contain only 0 and 1, with no gaps")
    else:
        rate = app["TARGET"].mean()
        if not 0.04 <= rate <= 0.14:
            errors.append(f"default rate is {rate:.2%}, outside the expected 4%-14%")

    # 4. Impossible values
    age_years = -app["DAYS_BIRTH"] / 365
    if age_years.min() < 18 or age_years.max() > 90:
        errors.append(
            f"ages outside 18-90 (found {age_years.min():.0f} to {age_years.max():.0f})"
        )
    if (app["AMT_INCOME_TOTAL"] <= 0).any():
        errors.append("AMT_INCOME_TOTAL has zero or negative values")
    if (app["AMT_CREDIT"] <= 0).any():
        errors.append("AMT_CREDIT has zero or negative values")
    odd = ((app["DAYS_EMPLOYED"] > 0) & (app["DAYS_EMPLOYED"] != 365243)).sum()
    if odd > 0:
        errors.append(f"DAYS_EMPLOYED has {odd} unexpected positive values")

    # 5. Too much missing data in the key columns
    for col, limit in MAX_MISSING.items():
        share = app[col].isnull().mean()
        if share > limit:
            errors.append(f"{col} is {share:.1%} missing (limit {limit:.0%})")

    # 6. Odd but not fatal: how many applicants have bureau records
    coverage = app["SK_ID_CURR"].isin(bureau["SK_ID_CURR"]).mean()
    if not 0.70 <= coverage <= 0.95:
        warnings.append(
            f"{coverage:.1%} of applicants have bureau records (training data had about 86%)"
        )

    if errors:
        raise DataValidationError("; ".join(errors))
    return warnings