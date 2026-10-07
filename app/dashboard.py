import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# make `src` importable when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.gate import load_production  # noqa: E402
from src.monitor import decide  # noqa: E402
from src.serving import explain, make_model_input, risk_band, to_raw_row  # noqa: E402

AVERAGE_RATE = 0.0807   # share of applicants who defaulted in the training data

st.set_page_config(page_title="Thin-file credit risk", layout="wide")
st.title("Thin-file credit risk")
st.caption(
    "Portfolio demo built on the Home Credit Default Risk dataset. "
    "Not for real lending decisions."
)


@st.cache_resource
def get_bundle():
    return load_production()


bundle = get_bundle()
if bundle is None:
    st.error("No production model found in models/production. Run `python -m src.train` first.")
    st.stop()

tab_score, tab_perf, tab_drift = st.tabs(
    ["Score an applicant", "Model and fairness", "Drift monitoring"]
)


def fmt_value(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "not provided"
    if isinstance(v, str):
        return v
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3f}"


# ---------------------------------------------------------------- score
with tab_score:
    st.write("Enter what you know about an applicant. Unknown fields are handled exactly as in the API.")
    org_options = ["(unknown)"] + bundle["categories"]["ORGANIZATION_TYPE"]
    edu_options = ["(unknown)"] + bundle["categories"]["NAME_EDUCATION_TYPE"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Applicant and loan")
        age = st.number_input("Age (years)", min_value=18, max_value=90, value=29)
        income = st.number_input("Annual income", min_value=1.0, value=150000.0, step=5000.0)
        loan = st.number_input("Loan amount", min_value=1.0, value=450000.0, step=10000.0)
        annuity = st.number_input("Repayment amount (annuity)", min_value=1.0, value=22000.0, step=500.0)
        goods = st.number_input("Price of goods financed (0 = unknown)", min_value=0.0, value=400000.0, step=10000.0)
        employed = st.number_input("Years employed (-1 = unknown)", min_value=-1.0, max_value=60.0, value=3.0, step=0.5)
        org = st.selectbox("Organization type", org_options)
        edu = st.selectbox("Education type", edu_options)

    with c2:
        st.subheader("Outside credit scores")
        ext = {}
        for i in (1, 2, 3):
            if st.checkbox(f"External score {i} known", value=(i != 1), key=f"known{i}"):
                ext[i] = st.slider(f"External score {i}", 0.0, 1.0, 0.5, 0.01, key=f"ext{i}")
            else:
                ext[i] = None

    with c3:
        st.subheader("Credit history")
        has_bureau = st.checkbox("Has credit bureau history", value=True)
        if has_bureau:
            n_bureau = st.number_input("Number of past credits", min_value=0, max_value=100, value=1)
            active = st.number_input("Currently active credits", min_value=0, max_value=100, value=1)
            debt = st.number_input("Total outstanding debt", min_value=0.0, value=30000.0, step=1000.0)
            credit_sum = st.number_input("Total credit amount", min_value=0.0, value=120000.0, step=1000.0)
        else:
            n_bureau = active = debt = credit_sum = None   # this applicant becomes thin-file
        n_prev = st.number_input("Previous applications", min_value=0, max_value=100, value=1)
        refused = st.number_input("Of which refused", min_value=0, max_value=100, value=0)
        approved = st.number_input("Of which approved", min_value=0, max_value=100, value=1)

    if st.button("Score applicant", type="primary"):
        applicant = {
            "age_years": age,
            "annual_income": income,
            "loan_amount": loan,
            "loan_annuity": annuity,
            "goods_price": goods or None,
            "years_employed": employed if employed >= 0 else None,
            "organization_type": None if org == "(unknown)" else org,
            "education_type": None if edu == "(unknown)" else edu,
            "ext_source_1": ext[1],
            "ext_source_2": ext[2],
            "ext_source_3": ext[3],
            "n_bureau_credits": n_bureau,
            "bureau_active_count": active,
            "bureau_debt_sum": debt,
            "bureau_credit_sum": credit_sum,
            "n_prev_apps": n_prev,
            "prev_refused_count": refused,
            "prev_approved_count": approved,
        }
        X = make_model_input(pd.DataFrame([to_raw_row(applicant)]), bundle)
        probability = float(bundle["model"].predict_proba(X)[0, 1])

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated default probability", f"{probability:.1%}")
        m2.metric("Risk band", risk_band(probability).upper())
        m3.metric("Thin-file applicant", "Yes" if X["THIN_FILE"].iloc[0] == 1 else "No")
        st.caption(f"For comparison, the average applicant in the training data defaulted {AVERAGE_RATE:.1%} of the time.")

        reasons = pd.DataFrame(explain(bundle["model"], X))
        reasons["value"] = reasons["value"].apply(fmt_value)
        st.write("**Main factors behind this score** (impact is in log-odds: positive raises risk, negative lowers it)")
        st.dataframe(reasons, hide_index=True)
        st.caption(
            "Reasons are model features and can overlap, for example the average of the outside "
            "scores and the individual scores."
        )

# ---------------------------------------------------------------- performance
with tab_perf:
    metrics = json.loads((config.PRODUCTION_DIR / "metrics.json").read_text())
    audit = pd.read_csv(config.PRODUCTION_DIR / "audit.csv")

    st.subheader("Held-out test performance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC-AUC", metrics["roc_auc"])
    k2.metric("PR-AUC", metrics["pr_auc"])
    k3.metric("Thin-file ROC-AUC", metrics["thin_roc_auc"])
    k4.metric("Thin-file calibration gap (points)", metrics["thin_calibration_gap_pp"])

    st.subheader("Fairness audit")
    st.caption(
        "Simulates rejecting the riskiest 20% of applicants. "
        "`good_wrongly_rejected_pct` is the share of people who would have repaid but were rejected."
    )
    for attribute, group in audit.groupby("attribute", sort=False):
        st.markdown(f"**{attribute}**")
        table = group.drop(columns="attribute")
        st.dataframe(table, hide_index=True)
        st.bar_chart(table.set_index("group")["good_wrongly_rejected_pct"])

# ---------------------------------------------------------------- drift
with tab_drift:
    files = sorted(config.REPORTS_DIR.glob("drift_*.csv"))
    if not files:
        st.info("No drift reports yet. Run `python -m src.monitor drifted` and refresh.")
    else:
        choice = st.selectbox("Batch", [f.stem.replace("drift_", "") for f in files])
        report = pd.read_csv(config.REPORTS_DIR / f"drift_{choice}.csv")
        retrain, reasons_list = decide(report)
        if retrain:
            st.error("Retraining would be triggered: " + "; ".join(reasons_list))
        else:
            st.success("No retraining needed: the new data looks like the training data.")

        def color_rows(row):
            colors = {
                "DRIFT": "background-color: #ffd6d6; color: #111",
                "watch": "background-color: #fff3cd; color: #111",
            }
            return [colors.get(row["status"], "")] * len(row)

        st.dataframe(report.style.apply(color_rows, axis=1), hide_index=True)
        st.caption("PSI below 0.10 is stable, 0.10 to 0.25 is worth watching, 0.25 or more is significant drift.")