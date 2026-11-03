import pandas as pd

from ingest.gaps import counter_gaps

START = pd.Timestamp("2026-01-01T00:00Z")


def _hours(n, start=START):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def _counts(values, start=START):
    return pd.DataFrame({"ts": _hours(len(values), start), "count": values})


def test_absent_rows_are_missing():
    counts = _counts([5] * 100).drop(index=range(10, 15))
    stats, runs = counter_gaps("a", counts, START, START + pd.Timedelta(hours=99))
    assert stats.expected_hours == 100
    assert stats.absent_hours == 5
    assert stats.zero_run_hours == 0
    assert runs.empty


def test_night_zeros_are_demand_not_missing():
    values = ([0] * 6 + [10] * 18) * 4
    stats, _runs = counter_gaps("a", _counts(values), START, START + pd.Timedelta(hours=95))
    assert stats.zero_run_hours == 0
    assert stats.missing_share == 0


def test_run_of_24_zero_hours_is_missing_but_23_is_not():
    for length, flagged in [(23, 0), (24, 24)]:
        values = [10] * 10 + [0] * length + [10] * 10
        end = START + pd.Timedelta(hours=len(values) - 1)
        stats, runs = counter_gaps("a", _counts(values), START, end, zero_run_hours=24)
        assert stats.zero_run_hours == flagged
        assert len(runs) == (1 if flagged else 0)


def test_absent_hours_inside_a_zero_run_join_it():
    # 5 normal hours, 12 zeros, 6 absent hours, 12 zeros, 5 normal hours.
    values = [10] * 5 + [0] * 12 + [None] * 6 + [0] * 12 + [10] * 5
    counts = _counts(values).dropna()
    end = START + pd.Timedelta(hours=len(values) - 1)
    stats, runs = counter_gaps("a", counts, START, end, zero_run_hours=24)
    assert stats.absent_hours == 6
    assert stats.zero_run_hours == 24
    assert len(runs) == 1


def test_ten_percent_threshold():
    exactly_ten = _counts([5] * 100).drop(index=range(10))
    stats, _ = counter_gaps("a", exactly_ten, START, START + pd.Timedelta(hours=99))
    assert stats.missing_share == 0.10
    assert not stats.excluded
    eleven = _counts([5] * 100).drop(index=range(11))
    stats, _ = counter_gaps("a", eleven, START, START + pd.Timedelta(hours=99))
    assert stats.excluded
