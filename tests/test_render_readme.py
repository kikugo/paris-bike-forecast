import pytest

from scripts import render_readme as rr

SOURCES = {
    "checked_at": "2026-11-02T00:00:00+00:00",
    "window_start": "2023-01-01",
    "sources": {
        "live": {"name": "Live", "page": "https://x", "licence": "ODbL", "first": "2025-10-02",
                 "last": "2026-11-01", "counters": 3, "rows": 1200},
        "weather": {"name": "Weather", "page": "https://y", "licence": "CC BY 4.0",
                    "first": "1940-01-01", "last": "yesterday"},
    },
}
GAPS = {
    "window_start": "2023-01-01T00:00:00+00:00", "window_end": "2026-11-01T22:00:00+00:00",
    "rule": {"zero_run_hours": 24, "max_missing_share": 0.1},
    "counters_total": 5, "counters_kept": 4, "counters_excluded": 1, "hours_in_window": 33622,
}


def test_tables_come_from_results():
    assert "| [Live](https://x) | ODbL | 2025-10-02 to 2026-11-01 | 3 counters, 1,200 hourly rows |" in rr.data_table(SOURCES)
    table = rr.gap_table(GAPS)
    assert "| 33,622 | 5 | 4 | 1 |" in table
    assert "more than 10%" in table


def test_replace_block_is_idempotent():
    text = "a\n<!-- DATA:START -->\nold\n<!-- DATA:END -->\nb"
    once = rr.replace_block(text, "DATA", "new")
    assert once == "a\n<!-- DATA:START -->\nnew\n<!-- DATA:END -->\nb"
    assert rr.replace_block(once, "DATA", "new") == once


def test_missing_block_raises():
    with pytest.raises(ValueError):
        rr.replace_block("no markers", "DATA", "x")


def test_fills_an_empty_block():
    text = "<!-- GAPS:START -->\n<!-- GAPS:END -->\n"
    assert rr.replace_block(text, "GAPS", "t") == "<!-- GAPS:START -->\nt\n<!-- GAPS:END -->\n"


BACKTEST = {
    "snapshot": {"start": "2024-01-01T00:00:00+00:00", "end": "2026-11-01T22:00:00+00:00", "counters": 2},
    "horizon_hours": 168,
    "models": {"seasonal_naive": "Seasonal naive (same hour last week)", "lightgbm": "LightGBM (median)"},
    "folds": [
        {"fold": 1, "cutoff": "2026-10-19T00:00:00", "counters": 2,
         "models": {"seasonal_naive": {"mae": 20.0, "rmse": 30.0, "smape": 40.0, "n": 5},
                    "lightgbm": {"mae": 15.0, "rmse": 25.0, "smape": 35.0, "n": 5,
                                 "coverage_80": 0.8123, "interval_width": 50.04, "pinball": 5.0}}},
    ],
    "overall": {"seasonal_naive": {"mae": 20.0, "rmse": 30.0, "smape": 40.0, "n": 5},
                "lightgbm": {"mae": 15.0, "rmse": 25.0, "smape": 35.0, "n": 5,
                             "coverage_80": 0.8123, "interval_width": 50.04, "pinball": 5.0}},
}


def test_backtest_table():
    table = rr.backtest_table(BACKTEST)
    assert "| Fold | Week from | Counters | Seasonal naive MAE | LightGBM MAE | LightGBM 80% coverage | Interval width |" in table
    assert "| 1 | 2026-10-19 | 2 | 20.00 | 15.00 | 81% | 50.0 |" in table
    assert "| **All folds** |  |  | **20.00** | **15.00** | **81%** | **50.0** |" in table


def test_overall_table_has_change_against_seasonal_naive():
    table = rr.overall_table(BACKTEST)
    assert "| Seasonal naive (same hour last week) | 20.00 | 30.00 | 40.0 |  |" in table
    assert "| LightGBM (median) | 15.00 | 25.00 | 35.0 | -25.0% |" in table


def test_champion_line():
    first = {"promote": True, "reason": "no champion yet", "candidate_mae": 19.8, "candidate_version": "1"}
    assert rr.champion_line(first) == (
        "Last training run: model version 1, backtest MAE 19.80, promoted to champion (no champion yet)."
    )
    later = first | {"promote": False, "reason": "does not beat the champion", "candidate_version": "2",
                     "champion_version": "1", "champion_mae": 19.8}
    assert rr.champion_line(later).endswith(" Champion: version 1, MAE 19.80.")
    promoted = later | {"promote": True, "reason": "beats the champion", "candidate_mae": 19.1}
    assert rr.champion_line(promoted).endswith("Previous champion: version 1, MAE 19.80.")
