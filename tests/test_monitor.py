import numpy as np
import pandas as pd

from src.monitor import decide, psi_categorical, psi_numeric


def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, b) < 0.05


def test_psi_is_large_for_a_shifted_distribution():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(1, 1, 5000))
    assert psi_numeric(a, b) > 0.25


def test_psi_for_categories():
    base = pd.Series(["x"] * 700 + ["y"] * 300)
    similar = pd.Series(["x"] * 690 + ["y"] * 310)
    different = pd.Series(["x"] * 300 + ["y"] * 700)
    assert psi_categorical(base, similar) < 0.01
    assert psi_categorical(base, different) > 0.25


def make_report(score_psi, feature_psis):
    rows = [{"column": "score", "psi": score_psi}]
    rows += [{"column": f"f{i}", "psi": p} for i, p in enumerate(feature_psis)]
    return pd.DataFrame(rows)


def test_decide_triggers_on_score_drift():
    retrain, reasons = decide(make_report(0.40, [0.01, 0.02]))
    assert retrain and reasons


def test_decide_triggers_when_many_features_drift():
    retrain, _ = decide(make_report(0.02, [0.30, 0.35, 0.28, 0.01]))
    assert retrain


def test_decide_stays_quiet_when_one_feature_drifts():
    retrain, _ = decide(make_report(0.02, [0.01, 0.30, 0.05]))
    assert not retrain