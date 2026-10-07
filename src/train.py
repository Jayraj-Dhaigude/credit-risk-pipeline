import json
import shutil

import joblib
import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.evaluate import score_model
from src.features import build_features, load_raw, to_model_matrix
from src.gate import load_production, run_gate
from src.validate import validate_data


def prepare(app, bureau, prev):
    """Build the features and split into train and test sets."""
    df = build_features(app, bureau, prev)
    X, categories = to_model_matrix(df)
    y = df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "df_test": df.loc[X_test.index],
        "categories": categories,
    }


def fit(data):
    """Train the candidate model."""
    model = lgb.LGBMClassifier(**config.LGBM_PARAMS) # type: ignore
    model.fit(data["X_train"], data["y_train"])
    return model


def evaluate_and_decide(model, data):
    """Score candidate and production on the SAME test rows, then run the gate."""
    candidate, audit = score_model(
        model, data["X_test"], data["y_test"], data["df_test"], config.REJECT_SHARE
    )

    production_bundle = load_production()
    production = None
    if production_bundle is not None:
        X_prod, _ = to_model_matrix(
            data["df_test"], categories=production_bundle["categories"]
        )
        X_prod = X_prod[production_bundle["features"]]
        production, _ = score_model(
            production_bundle["model"], X_prod, data["y_test"],
            data["df_test"], config.REJECT_SHARE,
        )

    approved, checks = run_gate(candidate, production)
    return {
        "candidate": candidate,
        "production": production,
        "audit": audit,
        "approved": approved,
        "checks": checks,
    }


def print_report(result):
    print("\nCandidate:", result["candidate"])
    print("Production:", result["production"] if result["production"] else "none yet")
    print("\nAudit by group (candidate):")
    print(result["audit"].to_string(index=False))
    print("\nEvaluation gate:")
    for c in result["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    print("\nDecision:", "PROMOTED" if result["approved"] else "BLOCKED")

def compute_fill_values(X_train):
    """Typical values for columns the model has (almost) never seen with a gap.

    LightGBM silently treats a missing value in such a column as 0, or follows a rule
    learned from a handful of rows, which can be very unnatural. The API fills these
    columns with the typical (median or most common) value instead.
    """
    fill = {}
    for col in X_train.columns:
        s = X_train[col]
        if s.isnull().mean() >= 0.01:     # a real share of gaps: the model has learned what missing means here
            continue
        if isinstance(s.dtype, pd.CategoricalDtype):
            fill[col] = s.mode().iloc[0]
        else:
            fill[col] = float(s.median())
    return fill

def build_reference(model, X_train):
    """A sample of the training data (monitored columns plus scores) that new data is compared against."""
    sample = X_train.sample(n=min(config.REFERENCE_ROWS, len(X_train)), random_state=config.SEED)
    reference = sample[config.MONITORED_FEATURES].copy()
    reference["score"] = model.predict_proba(sample)[:, 1]
    return reference


def save_and_promote(model, data, result):
    """Save the candidate; copy it to production only if the gate approved it."""
    decision = "promoted" if result["approved"] else "blocked"
    config.CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "features": list(data["X_train"].columns),
        "categories": data["categories"],
        "fill_values": compute_fill_values(data["X_train"]),
        "params": config.LGBM_PARAMS,
    }
    joblib.dump(bundle, config.CANDIDATE_DIR / "model.joblib")
    result["audit"].to_csv(    build_reference(model, data["X_train"]).to_csv(config.CANDIDATE_DIR / "reference.csv", index=False))
    (config.CANDIDATE_DIR / "metrics.json").write_text(
        json.dumps(result["candidate"], indent=2)
    )
    report = {
        "decision": decision,
        "checks": result["checks"],
        "candidate": result["candidate"],
        "production": result["production"],
    }
    (config.CANDIDATE_DIR / "gate_report.json").write_text(json.dumps(report, indent=2))

    if result["approved"]:
        shutil.copytree(config.CANDIDATE_DIR, config.PRODUCTION_DIR, dirs_exist_ok=True)
    return decision


def record_run(data, result, decision):
    """Write the run into MLflow."""
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params({
            **config.LGBM_PARAMS,
            "n_features": data["X_train"].shape[1],
            "n_train": len(data["X_train"]),
            "n_test": len(data["X_test"]),
            "thin_file_max_credits": config.THIN_FILE_MAX_CREDITS,
            "reject_share": config.REJECT_SHARE,
        })
        mlflow.log_metrics(result["candidate"])
        mlflow.log_metric("gate_approved", int(result["approved"]))
        mlflow.set_tag("gate_decision", decision)
        for name in ["model.joblib", "audit.csv", "gate_report.json"]:
            mlflow.log_artifact(str(config.CANDIDATE_DIR / name))


def main():
    print("Loading data...")
    app, bureau, prev = load_raw()

    print("Validating data...")
    for warning in validate_data(app, bureau, prev):
        print("  WARNING:", warning)
    print("  Data looks fine.")

    print("Building features and training...")
    data = prepare(app, bureau, prev)
    model = fit(data)

    result = evaluate_and_decide(model, data)
    print_report(result)

    decision = save_and_promote(model, data, result)
    record_run(data, result, decision)
    print("Done.")


if __name__ == "__main__":
    main()