import pandas as pd

from ingest.expectations import validate
from ingest.sources import school_holidays

RECORDS = [
    # Toussaint 2026: Saturday 17 Oct to Sunday 1 Nov, school restarts Monday 2 Nov (Paris dates).
    {"start_date": "2026-10-16T22:00:00+00:00", "end_date": "2026-11-01T23:00:00+00:00",
     "description": "Vacances de la Toussaint", "population": "-"},
    # Summer has a pupils' and a teachers' record; the pupils' one is kept.
    {"start_date": "2026-07-03T22:00:00+00:00", "end_date": "2026-08-31T22:00:00+00:00",
     "description": "Vacances d'Été", "population": "Élèves"},
    {"start_date": "2026-07-03T22:00:00+00:00", "end_date": "2026-08-29T22:00:00+00:00",
     "description": "Vacances d'Été", "population": "Enseignants"},
]


def test_markers_without_days_are_dropped():
    marker = {"start_date": "2028-07-03T22:00:00+00:00", "end_date": "2028-07-03T22:00:00+00:00",
              "description": "Début des Vacances d'Été", "population": "-"}
    assert len(school_holidays.parse([*RECORDS, marker])) == 2


def test_parse_uses_paris_dates_and_pupils_summer():
    df = school_holidays.parse(RECORDS)
    assert len(df) == 2
    toussaint = df[df.description.str.contains("Toussaint")].iloc[0]
    assert str(toussaint.first_day) == "2026-10-17"
    assert str(toussaint.last_day) == "2026-11-01"
    summer = df[df.description.str.contains("Été")].iloc[0]
    assert str(summer.last_day) == "2026-08-31"


def test_days_cover_each_break_inclusive():
    days = school_holidays.days(school_holidays.parse(RECORDS[:1]))
    assert len(days) == 16
    assert days[0] == pd.Timestamp("2026-10-17") and days[-1] == pd.Timestamp("2026-11-01")


def test_suite_rejects_a_break_that_ends_before_it_starts():
    df = school_holidays.parse(RECORDS)
    assert validate("school_holidays", df).success
    df.loc[0, "last_day"] = pd.Timestamp("2026-01-01").date()
    assert not validate("school_holidays", df).success
