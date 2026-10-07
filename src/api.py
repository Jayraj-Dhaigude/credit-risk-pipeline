import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.gate import load_production
from src.serving import explain, make_model_input, risk_band, to_raw_row

state = {}


@asynccontextmanager
async def lifespan(app):
    bundle = load_production()          # loaded once, when the server starts
    if bundle is None:
        raise RuntimeError("No production model found. Run the training pipeline first.")
    state["bundle"] = bundle
    yield
    state.clear()


app = FastAPI(title="Thin-file credit risk API", version="1.0", lifespan=lifespan)


class Applicant(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "age_years": 29,
        "annual_income": 150000,
        "loan_amount": 450000,
        "loan_annuity": 22000,
        "goods_price": 400000,
        "years_employed": 3,
        "organization_type": "Business Entity Type 3",
        "education_type": "Secondary / secondary special",
        "ext_source_2": 0.5,
        "ext_source_3": 0.45,
        "n_bureau_credits": 1,
        "bureau_active_count": 1,
        "bureau_debt_sum": 30000,
        "bureau_credit_sum": 120000,
        "n_prev_apps": 1,
        "prev_approved_count": 1,
        "prev_refused_count": 0,
    }})

    # required
    age_years: float = Field(..., ge=18, le=90, description="Age in years")
    annual_income: float = Field(..., gt=0, description="Total yearly income")
    loan_amount: float = Field(..., gt=0, description="Loan amount requested")
    loan_annuity: float = Field(..., gt=0, description="Regular repayment amount")

    # optional application details
    goods_price: float | None = Field(None, gt=0)
    years_employed: float | None = Field(None, ge=0, le=60)
    no_employment_record: bool = False
    organization_type: str | None = None
    education_type: str | None = None
    own_car_age: float | None = Field(None, ge=0, le=80)
    ext_source_1: float | None = Field(None, ge=0, le=1)
    ext_source_2: float | None = Field(None, ge=0, le=1)
    ext_source_3: float | None = Field(None, ge=0, le=1)

    # optional credit-history summary (leave out if the person has none)
    n_bureau_credits: int | None = Field(None, ge=0)
    bureau_active_count: int | None = Field(None, ge=0)
    bureau_debt_sum: float | None = Field(None, ge=0)
    bureau_credit_sum: float | None = Field(None, ge=0)
    n_prev_apps: int | None = Field(None, ge=0)
    prev_refused_count: int | None = Field(None, ge=0)
    prev_approved_count: int | None = Field(None, ge=0)


class Reason(BaseModel):
    feature: str
    value: float | str | None
    effect: str
    impact: float


class Prediction(BaseModel):
    default_probability: float
    risk_band: str
    thin_file: bool
    top_reasons: list[Reason]
    notes: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    path = config.PRODUCTION_DIR / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    return {"n_features": len(state["bundle"]["features"]), "test_metrics": metrics}


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant):
    bundle = state["bundle"]
    row = to_raw_row(applicant.model_dump())

    for col in ["ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE"]:
        value = row[col]
        if isinstance(value, str) and value not in bundle["categories"][col]:
            raise HTTPException(
                status_code=422,
                detail=f"{col} must be one of {bundle['categories'][col]}",
            )

    X = make_model_input(pd.DataFrame([row]), bundle)
    probability = float(bundle["model"].predict_proba(X)[0, 1])

    notes = [
        "Only the fields in this request were used; other model inputs were filled with "
        "typical values, so this estimate is less precise than the offline evaluation."
    ]
    if applicant.n_bureau_credits is None:
        notes.append("No bureau history was sent, so the applicant is treated as having none (thin-file).")

    return Prediction(
        default_probability=round(probability, 4),
        risk_band=risk_band(probability),
        thin_file=bool(X["THIN_FILE"].iloc[0] == 1),
        top_reasons=explain(bundle["model"], X),
        notes=notes,
    )