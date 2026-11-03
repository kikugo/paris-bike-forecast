"""Hourly Paris weather from Open-Meteo: backfill once, then recent hours every hour."""

from datetime import datetime, timedelta

from airflow.sdk import dag, task


@dag(
    dag_id="weather_hourly",
    schedule="@hourly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["weather"],
)
def weather_hourly():
    @task
    def backfill() -> int:
        from ingest import db, pipeline

        engine = db.get_engine()
        db.init_db(engine)
        return pipeline.backfill_weather(engine)

    @task
    def archive_refresh() -> int:
        from ingest import db, pipeline

        return pipeline.refresh_weather_archive(db.get_engine())

    @task
    def recent() -> int:
        from ingest import db, pipeline

        return pipeline.load_recent_weather(db.get_engine())

    backfill() >> archive_refresh() >> recent()


weather_hourly()
