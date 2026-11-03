"""Paris bike counter ingestion: a manual backfill and a daily incremental load."""

from datetime import datetime, timedelta

from airflow.sdk import dag, task

DEFAULTS = {"retries": 2, "retry_delay": timedelta(minutes=10)}


@dag(
    dag_id="counters_backfill",
    schedule=None,
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_tasks=1,  # one 1-2 GB yearly file at a time keeps memory flat
    default_args=DEFAULTS,
    tags=["counters"],
)
def counters_backfill():
    @task
    def init_db():
        from ingest import db

        db.init_db(db.get_engine())

    @task
    def list_years() -> list[int]:
        from ingest import pipeline

        return pipeline.history_years()

    @task
    def load_year(year: int) -> int:
        from ingest import db, pipeline

        return pipeline.load_history_year(db.get_engine(), year)

    @task
    def load_live_full() -> int:
        from ingest import db, pipeline

        return pipeline.load_live_full(db.get_engine())

    @task
    def gap_report() -> dict:
        from ingest import db, gaps

        summary = gaps.refresh(db.get_engine())
        return {k: summary[k] for k in ("counters_total", "counters_kept", "counters_excluded")}

    years = list_years()
    loaded = load_year.expand(year=years)
    init_db() >> years
    loaded >> load_live_full() >> gap_report()


@dag(
    dag_id="counters_daily",
    schedule="0 7 * * *",  # the live dataset is refreshed early morning with J-1 data
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args=DEFAULTS,
    tags=["counters"],
)
def counters_daily():
    @task
    def load_recent() -> int:
        from ingest import db, pipeline

        engine = db.get_engine()
        db.init_db(engine)
        return pipeline.load_recent_counts(engine)

    @task
    def gap_report() -> dict:
        from ingest import db, gaps

        summary = gaps.refresh(db.get_engine())
        return {k: summary[k] for k in ("counters_total", "counters_kept", "counters_excluded")}

    load_recent() >> gap_report()


counters_backfill()
counters_daily()
