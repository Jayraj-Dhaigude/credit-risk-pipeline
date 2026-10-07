import numpy as np
import pandas as pd

from src.evaluate import audit_table


def test_audit_table_covers_everyone():
    n = 400
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "TARGET": rng.integers(0, 2, n),
        "CODE_GENDER": rng.choice(["F", "M"], n),
        "DAYS_BIRTH": -rng.integers(7500, 25000, n),
        "THIN_FILE": rng.integers(0, 2, n),
    })
    probs = rng.random(n)

    table = audit_table(df, probs, min_group_size=1)

    assert set(table["attribute"]) == {"gender", "age_band", "thin_file"}
    for attribute in ["gender", "age_band", "thin_file"]:
        assert table.loc[table["attribute"] == attribute, "n"].sum() == n