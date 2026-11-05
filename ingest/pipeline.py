"""Load steps called by the Airflow DAGs. Each step validates, then upserts, then logs."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from sqlalchemy import Engine

from ingest import config, db, schema
from ingest.expectations import Validation, require, validate
from ingest.sources import holidays, open_meteo, paris_counters


def _client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(60, read=600), follow_redirects=True)


def record_validation(engine: Engine, validation: Validation, batch: str) -> None:
    row = pd.DataFrame(
        [
            {
                "suite": validation.suite,
                "batch": batch,
                "run_at": datetime.now(UTC),
                "success": validation.success,
                "rows": validation.rows,
                "evaluated": validation.evaluated,
                "failed": validation.failed,
                "failures": validation.failures,
            }
        ]
    )
    with engine.begin() as conn:
        conn.execute(schema.dq_results.insert(), row.to_dict("records"))


def _checked(engine: Engine, suite: str, df: pd.DataFrame, batch: str) -> Validation:
    validation = validate(suite, df)
    record_validation(engine, validation, batch)
    return require(validation)


def history_years(window_start: date = config.WINDOW_START) -> list[int]:
    with _client() as client:
        years = paris_counters.history_attachments(client)
    return [y for y in years if y >= window_start.year]


def load_history_year(engine: Engine, year: int, force: bool = False) -> int:
    """Download one yearly file, validate each chunk, upsert counts and counter metadata."""
    key = str(year)
    if not force and db.already_loaded(engine, "counters_history", key):
        return 0
    with _client() as client:
        url = paris_counters.history_attachments(client)[year]
        path = paris_counters.download(client, url, Path(config.DATA_DIR) / f"counters_{year}.zip")
    total, counters, summary = 0, [], None
    try:
        for chunk in paris_counters.read_history_zip(path):
            counts, meta = paris_counters.split(chunk, config.WINDOW_START)
            if counts.empty:
                continue
            v = validate("counts_hourly", counts)
            summary = v if summary is None else summary.merge(v)
            if not v.success:
                break
            db.upsert(engine, schema.counts_hourly, counts, ["counter_id", "ts"])
            counters.append(meta)
            total += len(counts)
    finally:
        path.unlink(missing_ok=True)
    if summary is not None:
        record_validation(engine, summary, f"history {year}")
        require(summary)
    if counters:
        _upsert_counters(engine, pd.concat(counters))
    db.log_ingest(engine, "counters_history", key, total)
    return total


def _upsert_counters(engine: Engine, meta: pd.DataFrame) -> None:
    meta = meta.drop_duplicates("counter_id", keep="last")
    db.upsert(engine, schema.counters, meta, ["counter_id"])


def load_live(engine: Engine, since: date, until: date | None = None) -> int:
    """Counts from the live dataset in [since, until)."""
    with _client() as client:
        raw = paris_counters.live_export(client, since, until)
    counts, meta = paris_counters.split(raw, config.WINDOW_START)
    label = f"live {since} to {until or 'now'}"
    _checked(engine, "counts_hourly", counts, label)
    db.upsert(engine, schema.counts_hourly, counts, ["counter_id", "ts"])
    _upsert_counters(engine, meta)
    db.log_ingest(engine, "counters_live", label, len(counts))
    return len(counts)


def load_live_full(engine: Engine) -> int:
    """All 13 months of the live dataset, one month per request to keep memory flat."""
    first = (date.today() - timedelta(days=400)).replace(day=1)
    total = 0
    for start in pd.date_range(first, date.today(), freq="MS").date:
        end = (pd.Timestamp(start) + pd.offsets.MonthBegin(1)).date()
        total += load_live(engine, start, end)
    return total


def load_recent_counts(engine: Engine, days: int = config.RECENT_DAYS) -> int:
    return load_live(engine, date.today() - timedelta(days=days))


def backfill_weather(engine: Engine, force: bool = False) -> int:
    """Archive weather from the window start to yesterday, one request per year."""
    if not force and db.already_loaded(engine, "weather_archive", "backfill"):
        return 0
    end = date.today() - timedelta(days=1)
    total = 0
    with _client() as client:
        for year in range(config.WINDOW_START.year, end.year + 1):
            start = max(config.WINDOW_START, date(year, 1, 1))
            stop = min(end, date(year, 12, 31))
            df = open_meteo.archive(client, start, stop)
            _checked(engine, "weather_hourly", df, f"archive {year}")
            db.upsert(engine, schema.weather_hourly, df, ["ts"])
            total += len(df)
    db.log_ingest(engine, "weather_archive", "backfill", total)
    return total


def refresh_weather_archive(engine: Engine, days: int = 10) -> int:
    """Re-pull the last days from the archive once a day so final values replace forecast ones."""
    key = date.today().isoformat()
    if db.already_loaded(engine, "weather_archive", key):
        return 0
    with _client() as client:
        df = open_meteo.archive(client, date.today() - timedelta(days=days), date.today())
    _checked(engine, "weather_hourly", df, f"archive refresh {key}")
    db.upsert(engine, schema.weather_hourly, df, ["ts"])
    db.log_ingest(engine, "weather_archive", key, len(df))
    return len(df)


def load_recent_weather(engine: Engine) -> int:
    with _client() as client:
        df = open_meteo.recent(client)
    _checked(engine, "weather_hourly", df, "forecast api recent")
    # Forecast-API values only fill hours the archive has not covered yet.
    db.upsert(engine, schema.weather_hourly, df, ["ts"], only_if="weather_hourly.source = 'forecast'")
    return len(df)


def _load_previous_runs(engine: Engine, start: date, end: date, batch: str) -> int:
    with _client() as client:
        df = open_meteo.previous_runs(client, start, end)
    df = df[df["ts"] <= pd.Timestamp.now(tz="UTC").floor("h")]
    _checked(engine, "weather_forecast", df, batch)
    db.upsert(engine, schema.weather_forecast, df, ["ts", "lead_day"])
    return len(df)


def backfill_weather_forecasts(engine: Engine, force: bool = False) -> int:
    """Issued weather forecasts from the window start to yesterday, one request per month."""
    if not force and db.already_loaded(engine, "weather_forecast", "backfill"):
        return 0
    end = date.today() - timedelta(days=1)
    total = 0
    for start in pd.date_range(config.WINDOW_START, end, freq="MS").date:
        stop = min(end, (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).date())
        total += _load_previous_runs(engine, start, stop, f"previous runs {start:%Y-%m}")
    db.log_ingest(engine, "weather_forecast", "backfill", total)
    return total


def refresh_weather_forecasts(engine: Engine, days: int = 3) -> int:
    """Once a day, the last days of issued forecasts."""
    key = date.today().isoformat()
    if db.already_loaded(engine, "weather_forecast", key):
        return 0
    rows = _load_previous_runs(
        engine, date.today() - timedelta(days=days), date.today(), f"previous runs refresh {key}"
    )
    db.log_ingest(engine, "weather_forecast", key, rows)
    return rows


def load_holidays(engine: Engine) -> int:
    with _client() as client:
        df = holidays.fetch(client)
    _checked(engine, "holidays", df, "metropole")
    db.upsert(engine, schema.holidays, df, ["day"])
    db.log_ingest(engine, "holidays", date.today().isoformat(), len(df))
    return len(df)
