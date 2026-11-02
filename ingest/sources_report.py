"""Licence and history length of every source, read from the sources themselves.

Writes results/data_sources.json, which the README data table is generated from.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ingest import config
from ingest.sources import holidays, open_meteo, paris_counters

PARIS_PAGE = "https://opendata.paris.fr/explore/dataset/{}/information/"
DATAGOUV_HOLIDAYS = "https://www.data.gouv.fr/api/1/datasets/jours-feries-en-france/"
DATAGOUV_LICENSES = "https://www.data.gouv.fr/api/1/datasets/licenses/"
# Open-Meteo states its licence on a web page, not in the API.
OPEN_METEO_LICENCE = {"licence": "CC BY 4.0", "page": "https://open-meteo.com/en/licence"}
OPEN_METEO_EARLIEST = "1940-01-01"


def _paris_meta(client: httpx.Client, dataset: str) -> dict:
    resp = client.get(f"{paris_counters.API}/{dataset}")
    resp.raise_for_status()
    meta = resp.json()["metas"]["default"]
    return {"licence": meta["license"], "page": PARIS_PAGE.format(dataset), "modified": meta["modified"]}


def _live_coverage(client: httpx.Client) -> dict:
    resp = client.get(
        f"{paris_counters.API}/{paris_counters.LIVE_DATASET}/records",
        params={"select": "min(date) as first, max(date) as last, count(distinct id_compteur) as counters, count(*) as rows", "limit": 1},
    )
    resp.raise_for_status()
    row = resp.json()["results"][0]
    return {"first": row["first"][:10], "last": row["last"][:10], "counters": row["counters"], "rows": row["rows"]}


def build(client: httpx.Client) -> dict:
    live = _paris_meta(client, paris_counters.LIVE_DATASET) | _live_coverage(client)
    years = list(paris_counters.history_attachments(client))
    history = _paris_meta(client, paris_counters.HISTORY_DATASET) | {
        "first": f"{years[0]}-01-01",
        "last": f"{years[-1]}-12-31",
        "files": len(years),
    }

    # Ask the archive for its earliest day to check the documented start still holds.
    early = open_meteo.archive(
        client,
        datetime.fromisoformat(OPEN_METEO_EARLIEST).date(),
        datetime.fromisoformat(OPEN_METEO_EARLIEST).date(),
    )
    weather = OPEN_METEO_LICENCE | {"first": early["ts"].min().date().isoformat(), "last": "yesterday"}

    hol = holidays.fetch(client)
    meta = client.get(DATAGOUV_HOLIDAYS).json()
    licences = {lic["id"]: lic["title"] for lic in client.get(DATAGOUV_LICENSES).json()}
    calendar = {
        "licence": licences.get(meta["license"], meta["license"]),
        "page": meta["page"],
        "first": hol["day"].min().isoformat(),
        "last": hol["day"].max().isoformat(),
    }
    return {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window_start": config.WINDOW_START.isoformat(),
        "sources": {
            "counters_live": {"name": "Comptage vélo - Données compteurs (13 rolling months)", **live},
            "counters_history": {"name": "Comptage vélo - Historique (yearly files)", **history},
            "weather": {"name": "Open-Meteo historical weather API", **weather},
            "holidays": {"name": "Jours fériés, metropolitan France (calendrier.api.gouv.fr)", **calendar},
        },
    }


def main() -> None:
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        report = build(client)
    path = Path(config.RESULTS_DIR) / "data_sources.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
