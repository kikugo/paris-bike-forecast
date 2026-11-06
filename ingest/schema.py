"""Tables of the bikes database. Shared by Postgres and the sqlite test database."""

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
)

metadata = MetaData()

counters = Table(
    "counters",
    metadata,
    Column("counter_id", String, primary_key=True),
    Column("counter_name", String),
    Column("site_id", String),
    Column("site_name", String),
    Column("installed_on", Date),
    Column("lat", Float),
    Column("lon", Float),
)

counts_hourly = Table(
    "counts_hourly",
    metadata,
    Column("counter_id", String, primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("count", Integer, nullable=False),
    Index("ix_counts_hourly_ts", "ts"),
)

weather_hourly = Table(
    "weather_hourly",
    metadata,
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("temperature_2m", Float),
    Column("relative_humidity_2m", Float),
    Column("precipitation", Float),
    Column("rain", Float),
    Column("wind_speed_10m", Float),
    Column("cloud_cover", Float),
    Column("source", String, nullable=False),
)

weather_forecast = Table(
    "weather_forecast",
    metadata,
    Column("ts", DateTime(timezone=True), primary_key=True),
    # Forecast issued lead_day x 24 hours before ts.
    Column("lead_day", Integer, primary_key=True),
    Column("temperature_2m", Float),
    Column("precipitation", Float),
    Column("wind_speed_10m", Float),
    Column("cloud_cover", Float),
    Column("relative_humidity_2m", Float),
)

holidays = Table(
    "holidays",
    metadata,
    Column("day", Date, primary_key=True),
    Column("name", String, nullable=False),
)

school_holidays = Table(
    "school_holidays",
    metadata,
    Column("first_day", Date, primary_key=True),
    Column("last_day", Date, nullable=False),
    Column("description", String, nullable=False),
)

ingest_log = Table(
    "ingest_log",
    metadata,
    Column("source", String, primary_key=True),
    Column("key", String, primary_key=True),
    Column("rows", Integer),
    Column("loaded_at", DateTime(timezone=True)),
)

dq_results = Table(
    "dq_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("suite", String, nullable=False),
    Column("batch", String, nullable=False),
    Column("run_at", DateTime(timezone=True), nullable=False),
    Column("success", Boolean, nullable=False),
    Column("rows", Integer),
    Column("evaluated", Integer),
    Column("failed", Integer),
    Column("failures", JSON),
)

gap_report = Table(
    "gap_report",
    metadata,
    Column("counter_id", String, primary_key=True),
    Column("first_ts", DateTime(timezone=True)),
    Column("last_ts", DateTime(timezone=True)),
    Column("expected_hours", Integer),
    Column("observed_hours", Integer),
    Column("absent_hours", Integer),
    Column("zero_run_hours", Integer),
    Column("missing_share", Float),
    Column("excluded", Boolean),
    Column("computed_at", DateTime(timezone=True)),
)

zero_runs = Table(
    "zero_runs",
    metadata,
    Column("counter_id", String, primary_key=True),
    Column("start_ts", DateTime(timezone=True), primary_key=True),
    Column("end_ts", DateTime(timezone=True), nullable=False),
    Column("hours", Integer, nullable=False),
)

# Counts from kept counters inside the window, with hours inside long zero runs set to NULL.
COUNTS_CLEAN_VIEW = """
    CREATE OR REPLACE VIEW counts_clean AS
    SELECT c.counter_id, c.ts,
           CASE WHEN z.counter_id IS NULL THEN c.count END AS count
    FROM counts_hourly c
    JOIN gap_report g ON g.counter_id = c.counter_id AND NOT g.excluded
    LEFT JOIN zero_runs z
      ON z.counter_id = c.counter_id AND c.ts BETWEEN z.start_ts AND z.end_ts
    WHERE c.ts >= :window_start
"""
