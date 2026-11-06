"""School holidays for Paris (Zone C), from the national school calendar
(data.education.gouv.fr, "Le calendrier scolaire", Licence Ouverte 2.0).

Dates are published years ahead, so they are known at any forecast cutoff. Each record gives
the start and end as midnight Paris time stored in UTC: the break covers Paris days from the start
date up to, not including, the end date (the day school restarts). Summer has one record for
pupils and one for teachers; pupils' dates are kept.
"""

import httpx
import pandas as pd

URL = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/exports/json"
TZ = "Europe/Paris"


def parse(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df[df["population"].isin(["-", "Élèves"])]
    first = pd.to_datetime(df["start_date"], utc=True).dt.tz_convert(TZ).dt.date
    restart = pd.to_datetime(df["end_date"], utc=True).dt.tz_convert(TZ).dt.date
    out = pd.DataFrame(
        {
            "first_day": first,
            "last_day": (pd.to_datetime(restart) - pd.Timedelta(days=1)).dt.date,
            "description": df["description"].str.strip(),
        }
    )
    # A few records have the same start and end: markers such as "Début des Vacances d'Été" for a
    # year whose end date is not published yet. They cover no day and are dropped.
    out = out[out["last_day"] >= out["first_day"]]
    return out.drop_duplicates("first_day").sort_values("first_day").reset_index(drop=True)


def fetch(client: httpx.Client) -> pd.DataFrame:
    resp = client.get(URL, params={"where": 'location="Paris"'})
    resp.raise_for_status()
    return parse(resp.json())


def days(periods: pd.DataFrame) -> pd.DatetimeIndex:
    """Every school-holiday day (Paris dates) covered by the periods."""
    if periods is None or periods.empty:
        return pd.DatetimeIndex([])
    ranges = [pd.date_range(a, b, freq="D") for a, b in zip(periods["first_day"], periods["last_day"], strict=True)]
    return pd.DatetimeIndex(sorted(set().union(*ranges)))
