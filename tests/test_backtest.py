from itertools import pairwise

import numpy as np
import pandas as pd

from forecast import backtest, baselines
from forecast.data import fill_history, to_grid
from tests.forecast_fixtures import START, weekly_panel


def test_cutoffs_are_mondays_with_a_full_last_week():
    data_end = pd.Timestamp("2026-11-01 22:00")  # Sunday, last hour of the week missing
    cuts = backtest.fold_cutoffs(data_end, n_folds=8)
    assert len(cuts) == 8
    assert all(c.day_of_week == 0 and c.hour == 0 for c in cuts)
    assert cuts[-1] == pd.Timestamp("2026-10-19")
    assert cuts[-1] + pd.Timedelta(hours=167) <= data_end
    assert all(b - a == pd.Timedelta(weeks=1) for a, b in pairwise(cuts))
    # A complete week ending at Sunday 23:00 is usable.
    assert backtest.fold_cutoffs(pd.Timestamp("2026-11-01 23:00"), 1) == [pd.Timestamp("2026-10-26")]


def test_to_grid_turns_gaps_into_rows():
    df = pd.DataFrame(
        {"unique_id": "a", "ds": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 03:00"]), "y": [1.0, 4.0]}
    )
    grid = to_grid(df, end=pd.Timestamp("2026-01-01 05:00"))
    assert len(grid) == 6
    assert grid["y"].isna().sum() == 4


def test_fill_history_uses_earlier_weeks_only():
    df = weekly_panel(weeks=3, counters=1)
    hole = START + pd.Timedelta(weeks=2, hours=5)
    truth = df.loc[df.ds == hole, "y"].item()
    df.loc[df.ds == hole, "y"] = np.nan
    # Make the earlier week distinctive so the source of the fill is visible.
    df.loc[df.ds == hole - pd.Timedelta(weeks=1), "y"] = truth + 100
    filled = fill_history(df)
    assert filled.loc[filled.ds == hole, "y"].item() == truth + 100


def test_fill_history_without_earlier_week_uses_median():
    df = weekly_panel(weeks=1, counters=1)
    df.loc[0, "y"] = np.nan
    filled = fill_history(df)
    assert filled.loc[0, "y"] == df["y"].median()


def test_seasonal_naive_recovers_a_pure_weekly_pattern():
    data = weekly_panel(weeks=6)
    cutoff = START + pd.Timedelta(weeks=5)
    for factory in baselines.MODELS.values():
        fc = factory().fit_predict(data, cutoff, 168)
        actual = data[data.ds >= cutoff]
        merged = actual.merge(fc, on=["unique_id", "ds"])
        assert len(merged) == len(actual)
        assert np.allclose(merged.y, merged.yhat)


def test_run_scores_every_fold_and_model():
    data = weekly_panel(weeks=7, noise=1.0)
    data.loc[data.sample(frac=0.02, random_state=1).index, "y"] = np.nan
    cuts = backtest.fold_cutoffs(data.ds.max(), n_folds=2)
    models = [f() for f in baselines.MODELS.values()]
    result = backtest.run(models, data, cuts)
    assert [f["fold"] for f in result["folds"]] == [1, 2]
    for fold in result["folds"]:
        assert set(fold["models"]) == set(baselines.MODELS)
    # Missing actuals are not scored: fewer pairs than counters x hours, the same for every model.
    scored = [f["models"]["seasonal_naive"]["n"] for f in result["folds"]]
    assert sum(scored) < 2 * 3 * 168
    assert scored == [f["models"]["seasonal_naive_4w"]["n"] for f in result["folds"]]
    assert set(result["overall"]) == set(baselines.MODELS)
    assert result["overall"]["seasonal_naive"]["n"] == sum(
        f["models"]["seasonal_naive"]["n"] for f in result["folds"]
    )
