"""French public holidays, metropolitan France (calendrier.api.gouv.fr, Licence Ouverte 2.0).

The endpoint returns 20 years back and 5 years ahead of the current year.
"""

import httpx
import pandas as pd

URL = "https://calendrier.api.gouv.fr/jours-feries/metropole.json"


def parse(payload: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(sorted(payload.items()), columns=["day", "name"])
    df["day"] = pd.to_datetime(df["day"], format="%Y-%m-%d").dt.date
    return df


def fetch(client: httpx.Client) -> pd.DataFrame:
    resp = client.get(URL)
    resp.raise_for_status()
    return parse(resp.json())
