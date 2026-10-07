import numpy as np

from src.features import build_features, load_raw


def make_batch(kind="no_drift", n=20_000, seed=0):
    """A pretend batch of new applicants.

    'no_drift' is a random sample of the data. 'drifted' applies changes a real lender
    might see: weaker outside scores, higher amounts (inflation), slightly younger
    applicants, and more applicants without bureau history.
    """
    if kind not in ("no_drift", "drifted"):
        raise ValueError("kind must be 'no_drift' or 'drifted'")

    app, bureau, prev = load_raw()
    rng = np.random.default_rng(seed)
    app = app.sample(n=n, random_state=seed).copy()

    if kind == "drifted":
        app["DAYS_BIRTH"] = (app["DAYS_BIRTH"] + 3 * 365).clip(upper=-18 * 365)
        for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]:
            app[col] = app[col] * 1.2
        for col in ["EXT_SOURCE_2", "EXT_SOURCE_3"]:
            app[col] = (app[col] - 0.12).clip(lower=0)
        lose_history = rng.random(len(app)) < 0.33
        bureau = bureau[~bureau["SK_ID_CURR"].isin(app.loc[lose_history, "SK_ID_CURR"])]

    return build_features(app, bureau, prev)