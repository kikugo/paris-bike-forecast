# Paris bike forecast

Hourly forecasts of bike traffic at the City of Paris permanent bike counters, evaluated with
rolling-origin backtests against seasonal-naive baselines, with quantile intervals and drift
monitoring.

This is my own rewrite of a bike-demand forecasting project I did as coursework. The original
course repo lives under a classmate's account and nothing here is copied from it: the data
source, pipeline, models and evaluation were rebuilt from scratch.

![CI](https://github.com/kikugo/paris-bike-forecast/actions/workflows/ci.yml/badge.svg)

Work in progress. Results are added phase by phase.

## Results

<!-- BACKTEST:START -->
| Fold | Week from | Counters | Seasonal naive (same hour last week) MAE | RMSE | sMAPE % | Seasonal naive, 4-week average MAE | RMSE | sMAPE % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-08-31 | 94 | 37.07 | 67.80 | 45.3 | 47.13 | 84.70 | 50.6 |
| 2 | 2026-09-07 | 95 | 16.73 | 32.18 | 27.7 | 41.52 | 73.83 | 43.1 |
| 3 | 2026-09-14 | 95 | 14.76 | 25.82 | 25.7 | 30.89 | 53.93 | 35.0 |
| 4 | 2026-09-21 | 95 | 17.94 | 39.38 | 29.1 | 20.22 | 39.55 | 29.6 |
| 5 | 2026-09-28 | 95 | 22.37 | 47.45 | 31.9 | 17.45 | 35.78 | 25.5 |
| 6 | 2026-10-05 | 95 | 20.47 | 40.12 | 30.6 | 18.63 | 35.95 | 28.2 |
| 7 | 2026-10-12 | 95 | 20.69 | 46.01 | 30.1 | 18.77 | 44.16 | 26.8 |
| 8 | 2026-10-19 | 95 | 35.65 | 59.52 | 46.0 | 38.10 | 66.20 | 46.5 |
| **All folds** |  |  | **23.04** | **46.37** | **33.1** | **29.27** | **57.37** | **35.7** |

8 rolling-origin folds, each trained on every hour before its cutoff and scored on the next 168 hours. Data snapshot 2024-01-01 to 2026-11-01, 95 counters. Scores pool every counter-hour with a known count.
<!-- BACKTEST:END -->

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

## How it's evaluated

- `forecast/snapshot.py` freezes the clean counts, weather and holidays from Postgres into
  parquet files and records their range and a content hash in `results/snapshot.json`, so a rerun
  scores exactly the same data.
- `forecast/backtest.py` runs a rolling-origin backtest: weekly cutoffs on Mondays, each fold
  trains on all data before its cutoff and forecasts the next week hour by hour.
- Metrics are MAE, RMSE and sMAPE. Plain MAPE is undefined here because counts hit zero at night;
  sMAPE counts an hour where both the actual and the forecast are zero as a perfect hour.
- Missing hours in the training history are filled from the same hour of an earlier week, so the
  weekly season never shifts. Missing actuals are not scored.
- `tests/test_leakage.py` checks every model: a forecast made at a cutoff must not change when all
  data after the cutoff is deleted.

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

Backtest (with the stack running):

```bash
uv run python -m forecast.snapshot
uv run python -m forecast.backtest
uv run python -m scripts.render_readme
```
