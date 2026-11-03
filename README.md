# Paris bike forecast

Hourly forecasts of bike traffic at the City of Paris permanent bike counters, evaluated with
rolling-origin backtests against seasonal-naive baselines, with quantile intervals and drift
monitoring.

This is my own rewrite of a bike-demand forecasting project I did as coursework. The original
course repo lives under a classmate's account and nothing here is copied from it: the data
source, pipeline, models and evaluation were rebuilt from scratch.

Work in progress. Results are added phase by phase.

## Architecture

```
opendata.paris.fr ──┐
Open-Meteo ─────────┼─> Airflow DAGs ─> Great Expectations ─> Postgres ─> Grafana
calendrier.api ─────┘    (ingest/)        (every load)        (bikes)    (data quality)
                                                                 │
                                                       MLflow (models, later phases)
```

- `counters_backfill` (manual): yearly history files, then the live dataset, then the gap report.
- `counters_daily`: the last days from the live dataset, then the gap report.
- `weather_hourly`: one-off archive backfill, a daily archive refresh, and the latest hours.
- `holidays_yearly`: public holidays for metropolitan France.

Each load is validated before it is written: no negative or missing counts, no duplicate
counter-hour, timestamps on the hour, weather values in physical ranges. Results land in the
`dq_results` table and on the Grafana "Data quality" dashboard.

## Data

<!-- DATA:START -->
| Source | Licence | Coverage | Notes |
| --- | --- | --- | --- |
| [Comptage vélo - Données compteurs (13 rolling months)](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/) | Open Database License (ODbL) | 2025-10-02 to 2026-11-01 | 113 counters, 1,036,713 hourly rows |
| [Comptage vélo - Historique (yearly files)](https://opendata.paris.fr/explore/dataset/comptage-velo-historique-donnees-compteurs/information/) | Open Database License (ODbL) | 2016-01-01 to 2025-12-31 | 10 yearly files |
| [Open-Meteo historical weather API](https://open-meteo.com/en/licence) | CC BY 4.0 | 1940-01-01 to yesterday |  |
| [Jours fériés, metropolitan France (calendrier.api.gouv.fr)](https://www.data.gouv.fr/datasets/jours-feries-en-france) | Licence Ouverte / Open Licence version 2.0 | 2006-01-01 to 2031-12-25 |  |

Checked 2026-11-03 by `ingest/sources_report.py`. This project ingests from 2024-01-01 onwards.
<!-- DATA:END -->

Weather data by [Open-Meteo.com](https://open-meteo.com/). Bike counter data © Ville de Paris,
under the ODbL.

### Counters kept after the gap rule

<!-- GAPS:START -->
| Window | Hours | Counters | Kept | Excluded |
| --- | --- | --- | --- | --- |
| 2024-01-01 to 2026-11-01 | 24,863 | 187 | 95 | 92 |

A counter is excluded when more than 10% of its hours are missing (no row, or inside a run of at least 24 zero hours).
<!-- GAPS:END -->

Counters break and get replaced, so a long run of zeros is treated as missing data rather than
as zero demand. Night-time zeros are kept.

## Run it

```bash
cp .env.example .env          # set local passwords
uv sync
uv run pytest
docker compose up -d --build  # Airflow, Postgres, MLflow, Grafana (ports in docker-compose.yml)
```

Trigger `holidays_yearly`, `weather_hourly` and `counters_backfill` once from the Airflow UI,
then refresh the generated parts of this README:

```bash
uv run python -m ingest.sources_report
uv run python -m scripts.render_readme
```
