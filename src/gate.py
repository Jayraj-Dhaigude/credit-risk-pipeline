import joblib

from src import config


def load_production():
    """Load the current production model bundle, or None if there isn't one yet."""
    path = config.PRODUCTION_DIR / "model.joblib"
    return joblib.load(path) if path.exists() else None


def run_gate(candidate, production):
    """Decide whether the candidate model may replace the production model.

    candidate and production are metric dictionaries, both measured on the
    SAME test rows. production is None when no model has been promoted yet.
    Returns (approved, list_of_checks).
    """
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # Rules every model must pass, even the very first one
    add("ROC-AUC above minimum",
        candidate["roc_auc"] >= config.GATE_MIN_ROC_AUC,
        f"{candidate['roc_auc']:.4f} (minimum {config.GATE_MIN_ROC_AUC})")
    add("thin-file calibration",
        abs(candidate["thin_calibration_gap_pp"]) <= config.GATE_MAX_THIN_CALIBRATION_GAP_PP,
        f"predicted minus actual = {candidate['thin_calibration_gap_pp']:+.2f} points "
        f"(allowed +/-{config.GATE_MAX_THIN_CALIBRATION_GAP_PP})")

    # Rules that compare against the current production model
    if production is not None:
        add("ROC-AUC at least as good as production",
            candidate["roc_auc"] >= production["roc_auc"] + config.GATE_MIN_IMPROVEMENT,
            f"candidate {candidate['roc_auc']:.4f} vs production {production['roc_auc']:.4f}")
        add("thin-file ROC-AUC not worse",
            candidate["thin_roc_auc"] >= production["thin_roc_auc"] - config.GATE_MAX_THIN_ROC_DROP,
            f"candidate {candidate['thin_roc_auc']:.3f} vs production {production['thin_roc_auc']:.3f}")
        for key in ["thin_wrongly_rejected_gap_pp",
                    "age_wrongly_rejected_gap_pp",
                    "gender_wrongly_rejected_gap_pp"]:
            add(f"{key} not much wider",
                candidate[key] <= production[key] + config.GATE_MAX_GAP_GROWTH_PP,
                f"candidate {candidate[key]:.1f} vs production {production[key]:.1f}")

    approved = all(c["passed"] for c in checks)
    return approved, checks