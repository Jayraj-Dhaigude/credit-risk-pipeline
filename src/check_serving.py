import numpy as np
from sklearn.metrics import roc_auc_score

from src import train
from src.features import load_raw, to_model_matrix
from src.gate import load_production
from src.serving import API_DATASET_COLUMNS, make_model_input


def report(name, probs, y):
    print(f"{name:<18} ROC-AUC {roc_auc_score(y, probs):.4f} | "
          f"average predicted risk {probs.mean():.2%} | actual default rate {y.mean():.2%}")


def main():
    bundle = load_production()
    model = bundle["model"] # type: ignore

    app, bureau, prev = load_raw()
    data = train.prepare(app, bureau, prev)
    df_test, y_test = data["df_test"], data["y_test"]

    # 1) every model input available (what the offline evaluation used)
    X_full, _ = to_model_matrix(df_test, categories=bundle["categories"]) # type: ignore
    full = model.predict_proba(X_full[bundle["features"]])[:, 1] # type: ignore

    # 2) only the fields the API collects
    raw = df_test[API_DATASET_COLUMNS].copy()
    raw.loc[df_test["DAYS_EMPLOYED_ANOM"] == 1, "DAYS_EMPLOYED"] = 365243
    api = model.predict_proba(make_model_input(raw, bundle))[:, 1]

    report("all inputs", full, y_test)
    report("API inputs only", api, y_test)
    print("Correlation between the two scores:", round(float(np.corrcoef(full, api)[0, 1]), 3))


if __name__ == "__main__":
    main()