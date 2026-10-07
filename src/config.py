from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

# CODE_GENDER is kept in the data for fairness audits but never given to the model
EXCLUDE_FROM_MODEL = ["TARGET", "SK_ID_CURR", "CODE_GENDER"]

THIN_FILE_MAX_CREDITS = 1   # thin-file = at most this many bureau credits

TEST_SIZE = 0.2
SEED = 42

LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    colsample_bytree=0.5,
    subsample=0.8,
    subsample_freq=1,
    min_child_samples=50,
    random_state=SEED,
    verbose=-1,
)

REJECT_SHARE = 0.20          

MLFLOW_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()
EXPERIMENT_NAME = "credit-risk-thin-file"

CANDIDATE_DIR = MODELS_DIR / "candidate"
PRODUCTION_DIR = MODELS_DIR / "production"

# Evaluation gate rules (policy choices: we'll explain them in the README)
GATE_MIN_ROC_AUC = 0.72                  # below this a model is never accepted
GATE_MIN_IMPROVEMENT = 0.0               # candidate must be at least this much better
GATE_MAX_THIN_ROC_DROP = 0.01            # thin-file ROC-AUC may not drop more than this
GATE_MAX_THIN_CALIBRATION_GAP_PP = 2.0   # predicted vs actual thin-file default, in points
GATE_MAX_GAP_GROWTH_PP = 2.0             # fairness gaps may not widen by more than this


# risk bands for the API (illustrative: the average applicant in the data defaults about 8% of the time)
RISK_BANDS = [(0.05, "low"), (0.12, "medium")]   # below 5% low, below 12% medium, otherwise high

REPORTS_DIR = ROOT / "reports"

# monitoring
MONITORED_FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "PAYMENT_RATE", "CREDIT_INCOME_RATIO",
    "N_BUREAU_CREDITS", "BUREAU_DEBT_SUM", "PREV_REFUSED_COUNT",
    "ORGANIZATION_TYPE",
]
REFERENCE_ROWS = 20000        # size of the training sample new data is compared against
DRIFT_PSI_WARN = 0.10
DRIFT_PSI_ALERT = 0.25
DRIFT_MIN_FEATURES = 3        # this many drifted features triggers retraining