"""Load the backtest fold forecasts into the `forecasts` table as source = 'backtest'.

Each fold's forecast was made by a model trained only on data before that fold's cutoff, so these
rows are honest out-of-sample history for the monitoring dashboard while live forecasts build up.

Run (stack up, after forecast.backtest): uv run python -m forecast.seed_backtest
"""

import pandas as pd
from sqlalchemy import delete

from forecast.backtest import PREDICTIONS
from forecast.snapshot import database_url
from ingest import db, schema


def rows(predictions: pd.DataFrame, model: str = "lightgbm") -> pd.DataFrame:
    p = predictions[predictions["model"] == model]
    cutoff = pd.to_datetime(p["cutoff"]).dt.tz_localize("UTC")
    return pd.DataFrame(
        {
            "source": "backtest",
            "origin": cutoff,
            "counter_id": p["unique_id"],
            "ts": pd.to_datetime(p["ds"]).dt.tz_localize("UTC"),
            "q10": p["q10"],
            "q50": p["q50"],
            "q90": p["q90"],
            "model_version": f"backtest-{model}",
            "issued_at": cutoff,
        }
    ).reset_index(drop=True)


def main() -> None:
    engine = db.get_engine(database_url())
    db.init_db(engine)
    df = rows(pd.read_parquet(PREDICTIONS))
    with engine.begin() as conn:
        conn.execute(delete(schema.forecasts).where(schema.forecasts.c.source == "backtest"))
    print(f"seeded {db.upsert(engine, schema.forecasts, df, ['source', 'origin', 'counter_id', 'ts'])} rows")


if __name__ == "__main__":
    main()
