import pandas as pd

from forecast import seed_backtest


def test_rows_keep_only_lightgbm_with_the_fold_cutoff_as_origin():
    cutoff = pd.Timestamp("2026-10-26")
    preds = pd.DataFrame(
        {"unique_id": ["a", "a"], "ds": [cutoff, cutoff], "yhat": [1.0, 2.0], "q10": [0.5, None],
         "q50": [1.0, None], "q90": [2.0, None], "model": ["lightgbm", "seasonal_naive"], "cutoff": [cutoff, cutoff]}
    )
    out = seed_backtest.rows(preds)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.source == "backtest" and row.model_version == "backtest-lightgbm"
    assert row.origin == pd.Timestamp("2026-10-26", tz="UTC") == row.issued_at
