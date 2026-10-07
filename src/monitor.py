import sys

import numpy as np
import pandas as pd

from src import config
from src.features import to_model_matrix
from src.gate import load_production
from src.simulate import make_batch

EPS = 1e-4


def _psi(ref_share, cur_share):
    ref_share = np.clip(ref_share, EPS, None)
    cur_share = np.clip(cur_share, EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_numeric(reference, current, bins=10):
    """PSI for a number column: 10 equal-sized buckets from the reference, plus one for 'missing'."""
    ref_all = reference.to_numpy(dtype=float)
    cur_all = current.to_numpy(dtype=float)
    ref = ref_all[~np.isnan(ref_all)]
    cur = cur_all[~np.isnan(cur_all)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    n_bins = len(inner) + 1
    ref_counts = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=n_bins)
    cur_counts = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=n_bins)

    ref_counts = np.append(ref_counts, len(ref_all) - len(ref))      # the 'missing' bucket
    cur_counts = np.append(cur_counts, len(cur_all) - len(cur))
    return _psi(ref_counts / len(ref_all), cur_counts / len(cur_all))


def psi_categorical(reference, current):
    """PSI for a text column: one bucket per category."""
    ref = reference.astype("string").fillna("MISSING").value_counts(normalize=True)
    cur = current.astype("string").fillna("MISSING").value_counts(normalize=True)
    categories = ref.index.union(cur.index)
    return _psi(
        ref.reindex(categories, fill_value=0).to_numpy(),
        cur.reindex(categories, fill_value=0).to_numpy(),
    )


def status(psi):
    if np.isnan(psi):
        return "unknown"
    if psi >= config.DRIFT_PSI_ALERT:
        return "DRIFT"
    if psi >= config.DRIFT_PSI_WARN:
        return "watch"
    return "stable"


def drift_report(reference, current):
    rows = []
    for col in config.MONITORED_FEATURES + ["score"]:
        ref, cur = reference[col], current[col]
        numeric = pd.api.types.is_numeric_dtype(ref)
        psi = psi_numeric(ref, cur) if numeric else psi_categorical(ref, cur)
        rows.append({
            "column": col,
            "psi": round(psi, 4),
            "status": status(psi),
            "missing_shift_pp": round((cur.isna().mean() - ref.isna().mean()) * 100, 1),
            "ref_mean": round(float(ref.mean()), 3) if numeric else None,
            "cur_mean": round(float(cur.mean()), 3) if numeric else None,
        })
    return pd.DataFrame(rows)


def decide(report):
    """Retrain if the score drifted, or if enough features drifted."""
    score_psi = float(report.loc[report["column"] == "score", "psi"].iloc[0])
    features = report[report["column"] != "score"]
    drifted = features.loc[features["psi"] >= config.DRIFT_PSI_ALERT, "column"].tolist()

    reasons = []
    if score_psi >= config.DRIFT_PSI_ALERT:
        reasons.append(f"score distribution drifted (PSI {score_psi:.2f})")
    if len(drifted) >= config.DRIFT_MIN_FEATURES:
        reasons.append(f"{len(drifted)} features drifted: {', '.join(drifted)}")
    return bool(reasons), reasons


def load_reference():
    path = config.PRODUCTION_DIR / "reference.csv"
    if not path.exists():
        raise FileNotFoundError("No reference.csv in models/production. Run the training pipeline first.")
    return pd.read_csv(path)


def check_batch(kind):
    bundle = load_production()
    if bundle is None:
        raise RuntimeError("No production model yet. Run the training pipeline first.")
    reference = load_reference()

    batch = make_batch(kind)
    X, _ = to_model_matrix(batch, categories=bundle["categories"])
    scores = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]

    current = batch[config.MONITORED_FEATURES].copy()
    current["score"] = scores

    report = drift_report(reference, current)
    retrain, reasons = decide(report)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    report.to_csv(config.REPORTS_DIR / f"drift_{kind}.csv", index=False)
    return report, retrain, reasons


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "no_drift"
    report, retrain, reasons = check_batch(kind)
    print(f"\nDrift report for batch '{kind}':\n")
    print(report.to_string(index=False))
    print("\nRetrain needed:", retrain)
    for reason in reasons:
        print(" -", reason)


if __name__ == "__main__":
    main()