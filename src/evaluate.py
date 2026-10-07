import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

AGE_BINS = [0, 30, 40, 50, 60, 100]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]


def audit_table(df_test, probs, reject_share=0.20, min_group_size=100):
    """Compare groups: actual vs predicted default, and who gets wrongly rejected.

    df_test needs the columns TARGET, CODE_GENDER, DAYS_BIRTH and THIN_FILE.
    Gender is used here for auditing only. It is never a model input.
    """
    a = pd.DataFrame({
        "y": df_test["TARGET"].to_numpy(),
        "risk": probs,
        "gender": df_test["CODE_GENDER"].to_numpy(),
        "age_band": pd.cut(-df_test["DAYS_BIRTH"].to_numpy() / 365,
                           AGE_BINS, labels=AGE_LABELS),
        "thin_file": df_test["THIN_FILE"].map({0: "not thin-file", 1: "thin-file"}).to_numpy(),
    })

    cutoff = np.quantile(a["risk"], 1 - reject_share)
    a["rejected"] = (a["risk"] >= cutoff).astype(int)

    rows = []
    for col in ["gender", "age_band", "thin_file"]:
        for name, g in a.groupby(col, observed=True):
            if len(g) < min_group_size:
                continue
            good = g[g["y"] == 0]
            rows.append({
                "attribute": col,
                "group": str(name),
                "n": int(len(g)),
                "actual_default_pct": round(g["y"].mean() * 100, 2),
                "avg_predicted_pct": round(g["risk"].mean() * 100, 2),
                "rejected_pct": round(g["rejected"].mean() * 100, 1),
                "good_wrongly_rejected_pct": round(good["rejected"].mean() * 100, 1),
                "roc_auc": round(roc_auc_score(g["y"], g["risk"]), 4)
                if g["y"].nunique() == 2 else np.nan,
            })
    return pd.DataFrame(rows)


def summary_metrics(audit):
    """Boil the audit table down to a few numbers we can track run after run.

    'pp' means percentage points.
    """
    thin = audit[audit["attribute"] == "thin_file"].set_index("group")
    age = audit[audit["attribute"] == "age_band"]
    gender = audit[audit["attribute"] == "gender"]
    return {
        "thin_roc_auc": float(thin.loc["thin-file", "roc_auc"]),
        "thin_calibration_gap_pp": float(
            thin.loc["thin-file", "avg_predicted_pct"] - thin.loc["thin-file", "actual_default_pct"]),
        "thin_wrongly_rejected_gap_pp": float(
            thin.loc["thin-file", "good_wrongly_rejected_pct"]
            - thin.loc["not thin-file", "good_wrongly_rejected_pct"]),
        "age_wrongly_rejected_gap_pp": float(
            age["good_wrongly_rejected_pct"].max() - age["good_wrongly_rejected_pct"].min()),
        "gender_wrongly_rejected_gap_pp": float(
            gender["good_wrongly_rejected_pct"].max() - gender["good_wrongly_rejected_pct"].min()),
    }

def score_model(model, X, y, df_test, reject_share=0.20):
    """Score one model on a test set: overall metrics plus the group audit."""
    probs = model.predict_proba(X)[:, 1]
    audit = audit_table(df_test, probs, reject_share=reject_share)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y, probs)), 4),
        "pr_auc": round(float(average_precision_score(y, probs)), 4),
        **{k: round(v, 3) for k, v in summary_metrics(audit).items()},
    }
    return metrics, audit