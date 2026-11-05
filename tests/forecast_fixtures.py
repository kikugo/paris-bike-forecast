import numpy as np
import pandas as pd

START = pd.Timestamp("2026-01-05")  # a Monday


def weekly_panel(weeks: int = 6, counters: int = 3, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Counters with a fixed weekly profile (scaled per counter), naive UTC hours."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range(START, periods=168 * weeks, freq="h")
    profile = 10 + 5 * np.sin(np.arange(168) * 2 * np.pi / 24) + np.arange(168) / 10
    parts = []
    for c in range(counters):
        y = np.tile(profile * (c + 1), weeks) + rng.normal(0, noise, len(ds))
        parts.append(pd.DataFrame({"unique_id": f"c{c}", "ds": ds, "y": np.maximum(y, 0)}))
    return pd.concat(parts, ignore_index=True)
