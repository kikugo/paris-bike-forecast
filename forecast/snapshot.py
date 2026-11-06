"""Freeze the clean data the backtest reads.

The database keeps growing every day, so results come from a snapshot: parquet files under
data/snapshot/ (not committed) and results/snapshot.json (committed) with the end hour, sizes and
a content hash, so anyone can check they are scoring the same data.

Run: uv run python -m forecast.snapshot
"""

import hashlib
import json
import os

import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import create_engine

from forecast.backtest import RESULTS, ROOT, SNAPSHOT_DIR


def database_url() -> str:
    if url := os.getenv("BIKES_DATABASE_URL"):
        return url
    env = dotenv_values(ROOT / ".env")
    return f"postgresql+psycopg://bikes:{env['BIKES_DB_PASSWORD']}@localhost:5433/bikes"


def content_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def main() -> None:
    engine = create_engine(database_url())
    counts = pd.read_sql(
        "SELECT counter_id, ts, count FROM counts_clean ORDER BY counter_id, ts", engine
    )
    weather = pd.read_sql("SELECT * FROM weather_hourly ORDER BY ts", engine)
    holidays = pd.read_sql("SELECT day, name FROM holidays ORDER BY day", engine)
    weather_fc = pd.read_sql("SELECT * FROM weather_forecast ORDER BY ts, lead_day", engine)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_parquet(SNAPSHOT_DIR / "counts.parquet", index=False)
    weather.to_parquet(SNAPSHOT_DIR / "weather.parquet", index=False)
    holidays.to_parquet(SNAPSHOT_DIR / "holidays.parquet", index=False)
    weather_fc.to_parquet(SNAPSHOT_DIR / "weather_forecast.parquet", index=False)

    info = {
        "start": counts["ts"].min().isoformat(),
        "end": counts["ts"].max().isoformat(),
        "counters": int(counts["counter_id"].nunique()),
        "rows": len(counts),
        "rows_with_count": int(counts["count"].notna().sum()),
        "counts_sha256": content_hash(counts),
        "weather_forecast_rows": len(weather_fc),
        "weather_forecast_sha256": content_hash(weather_fc),
    }
    (RESULTS / "snapshot.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
