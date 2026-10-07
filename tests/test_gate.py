from src.gate import run_gate


def good():
    return {
        "roc_auc": 0.773, "pr_auc": 0.267, "thin_roc_auc": 0.760,
        "thin_calibration_gap_pp": -0.2,
        "thin_wrongly_rejected_gap_pp": 6.9,
        "age_wrongly_rejected_gap_pp": 26.0,
        "gender_wrongly_rejected_gap_pp": 7.2,
    }


def test_first_model_is_approved_if_good_enough():
    approved, _ = run_gate(good(), None)
    assert approved


def test_first_model_below_minimum_is_blocked():
    approved, _ = run_gate(dict(good(), roc_auc=0.60), None)
    assert not approved


def test_better_model_is_approved():
    approved, _ = run_gate(dict(good(), roc_auc=0.776), good())
    assert approved


def test_identical_model_is_approved():
    approved, _ = run_gate(good(), good())
    assert approved


def test_worse_model_is_blocked():
    approved, checks = run_gate(dict(good(), roc_auc=0.771), good())
    assert not approved
    assert any(not c["passed"] for c in checks)


def test_thin_file_regression_is_blocked():
    # better overall, but much worse for thin-file borrowers
    approved, _ = run_gate(dict(good(), roc_auc=0.780, thin_roc_auc=0.70), good())
    assert not approved


def test_wider_fairness_gap_is_blocked():
    approved, _ = run_gate(
        dict(good(), roc_auc=0.780, age_wrongly_rejected_gap_pp=30.0), good()
    )
    assert not approved