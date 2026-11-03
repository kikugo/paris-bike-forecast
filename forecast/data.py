"""Panel helpers shared by every model.

Inside `forecast/` the panel uses statsforecast's long format: `unique_id` (counter id),
`ds` (hour, naive UTC) and `y` (count, NaN when missing).
"""

import pandas as pd

HOUR = pd.Timedelta(hours=1)


def to_grid(df: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Reindex each counter to every hour from its first row up to `end` (inclusive).

    Seasonal models read rows by position, so a missing row would shift the season; here every
    absent hour becomes a row with y = NaN instead.
    """
    parts = []
    for uid, g in df.groupby("unique_id", sort=True):
        hours = pd.date_range(g["ds"].min(), end, freq="h")
        part = g.set_index("ds")["y"].reindex(hours).rename_axis("ds").reset_index()
        part.insert(0, "unique_id", uid)
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])
    return pd.concat(parts, ignore_index=True)


def fill_history(grid: pd.DataFrame) -> pd.DataFrame:
    """Impute missing training hours using earlier data only.

    First the value at the same hour of an earlier week (latest available), then the counter's
    median over its observed history. Only ever applied to history before a cutoff.
    """
    out = grid.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    hour_of_week = out["ds"].dt.day_of_week * 24 + out["ds"].dt.hour
    filled = out.groupby([out["unique_id"], hour_of_week])["y"].ffill()
    median = out.groupby("unique_id")["y"].transform("median")
    out["y"] = filled.fillna(median).fillna(0.0)
    return out
