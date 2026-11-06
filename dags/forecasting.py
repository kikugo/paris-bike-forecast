"""Weekly forecast issue and daily scoring.

`forecast_weekly` issues the champion's forecast for the week (Monday 00:00 UTC, 168 hours), the
setup the model was backtested on. `score_daily` scores issued forecasts against the counts and
computes feature drift.
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, task

MODEL_NAME = "paris-bike-lgbm"


@dag(
    dag_id="forecast_weekly",
    schedule="0 9 * * 1",  # after Monday's 07:00 counter load
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=15)},
    tags=["forecast"],
)
def forecast_weekly():
    @task
    def issue() -> dict:
        import httpx
        import mlflow
        import pandas as pd
        from airflow.sdk.exceptions import AirflowSkipException
        from mlflow import MlflowClient

        from forecast import live
        from ingest import db
        from ingest.sources import open_meteo

        engine = db.get_engine()
        db.init_db(engine)
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        trained_until = live.trained_until(version.tags["snapshot_end"])
        origin = live.week_origin(pd.Timestamp.now(tz="UTC").tz_localize(None))
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version.version}")
        with httpx.Client(timeout=60) as client:
            weather = open_meteo.forecast_at_issue(client, origin, live.HORIZON)
        try:
            rows = live.issue(engine, model, version.version, trained_until, origin, weather)
        except live.OriginTooEarly as exc:
            # Not an error: the champion was trained on part of this week, so there is no fair
            # forecast to issue until next Monday.
            raise AirflowSkipException(str(exc)) from exc
        return {"origin": str(origin), "model_version": version.version, "rows": rows}

    issue()


@dag(
    dag_id="score_daily",
    schedule="30 9 * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["forecast"],
)
def score_daily():
    @task
    def score_and_drift() -> dict:
        from forecast import monitor
        from ingest import db

        engine = db.get_engine()
        db.init_db(engine)
        return monitor.run(engine)

    score_and_drift()


forecast_weekly()
score_daily()
