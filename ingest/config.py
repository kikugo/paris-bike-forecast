"""Settings for the ingestion layer, read from the environment with local defaults."""

import os
from datetime import date

WINDOW_START = date.fromisoformat(os.getenv("WINDOW_START", "2024-01-01"))

# One weather point for the whole city: every counter sits within ~10 km of it,
# which is smaller than the weather model grid.
PARIS_LAT = 48.8566
PARIS_LON = 2.3522

# A run of at least this many consecutive zero hours is a dead sensor, not zero demand.
ZERO_RUN_HOURS = int(os.getenv("ZERO_RUN_HOURS", "24"))
# Counters missing more than this share of hours in the window are dropped.
MAX_MISSING_SHARE = float(os.getenv("MAX_MISSING_SHARE", "0.10"))

RECENT_DAYS = int(os.getenv("RECENT_DAYS", "3"))

DATA_DIR = os.getenv("DATA_DIR", "data/raw")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")


def database_url() -> str:
    return os.getenv(
        "BIKES_DATABASE_URL", "postgresql+psycopg://bikes:bikes@localhost:5433/bikes"
    )
