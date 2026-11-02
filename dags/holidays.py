"""French public holidays, refreshed once a year (the API covers 20 years back, 5 ahead)."""

from datetime import datetime, timedelta

from airflow.sdk import dag, task


@dag(
    dag_id="holidays_yearly",
    schedule="@yearly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["calendar"],
)
def holidays_yearly():
    @task
    def load() -> int:
        from ingest import db, pipeline

        engine = db.get_engine()
        db.init_db(engine)
        return pipeline.load_holidays(engine)

    load()


holidays_yearly()
