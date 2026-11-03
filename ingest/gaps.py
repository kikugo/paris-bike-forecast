"""Per-counter gap report and the drop rule.

An hour counts as missing for a counter when there is no row for it, or when it sits in a
run of at least `zero_run_hours` consecutive zero (or absent) hours: counters break and
report zeros, and a day of zeros is a dead sensor, not a day without cyclists. Counters with
more than `max_missing_share` of the window missing are excluded from modelling.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, delete, func, select

from ingest import config, db, schema


@dataclass
class CounterGaps:
    counter_id: str
    first_ts: pd.Timestamp | None
    last_ts: pd.Timestamp | None
    expected_hours: int
    observed_hours: int
    absent_hours: int
    zero_run_hours: int
    missing_share: float
    excluded: bool


def hourly_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", tz="UTC")


def zero_runs(series: pd.Series, min_hours: int) -> pd.DataFrame:
    """Runs of >= min_hours consecutive hours that are zero or absent and hold at least one zero.

    `series` is indexed by the full hourly grid, NaN where no row exists.
    Returns start_ts, end_ts, hours (zero hours inside the run).
    """
    dead = series.fillna(0).eq(0)
    run_id = (dead != dead.shift()).cumsum()
    runs = []
    for _, block in series[dead].groupby(run_id[dead]):
        if len(block) >= min_hours and block.eq(0).any():
            runs.append(
                {
                    "start_ts": block.index[0],
                    "end_ts": block.index[-1],
                    "hours": int(block.eq(0).sum()),
                }
            )
    return pd.DataFrame(runs, columns=["start_ts", "end_ts", "hours"])


def counter_gaps(
    counter_id: str,
    counts: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    zero_run_hours: int = config.ZERO_RUN_HOURS,
    max_missing_share: float = config.MAX_MISSING_SHARE,
) -> tuple[CounterGaps, pd.DataFrame]:
    """Gap stats for one counter over [start, end]. `counts` has columns ts, count."""
    grid = hourly_grid(start, end)
    in_window = counts[(counts["ts"] >= start) & (counts["ts"] <= end)]
    series = in_window.set_index("ts")["count"].astype(float).reindex(grid)
    observed = int(series.notna().sum())
    absent = len(grid) - observed
    runs = zero_runs(series, zero_run_hours)
    zero_hours = int(runs["hours"].sum()) if not runs.empty else 0
    share = (absent + zero_hours) / len(grid) if len(grid) else 1.0
    stats = CounterGaps(
        counter_id=counter_id,
        first_ts=in_window["ts"].min() if observed else None,
        last_ts=in_window["ts"].max() if observed else None,
        expected_hours=len(grid),
        observed_hours=observed,
        absent_hours=absent,
        zero_run_hours=zero_hours,
        missing_share=round(share, 6),
        excluded=share > max_missing_share,
    )
    runs.insert(0, "counter_id", counter_id)
    return stats, runs


def refresh(engine: Engine, results_dir: str = config.RESULTS_DIR) -> dict:
    """Recompute the gap report for every counter, store it, and write results/gap_report.json."""
    c = schema.counts_hourly
    start = pd.Timestamp(config.WINDOW_START, tz="UTC")
    with engine.connect() as conn:
        end = conn.execute(select(func.max(c.c.ts))).scalar()
        # Counters retired before the window start are not part of the report.
        ids = conn.execute(
            select(c.c.counter_id).where(c.c.ts >= start).distinct().order_by(c.c.counter_id)
        ).scalars().all()
    if end is None:
        raise RuntimeError("counts_hourly is empty")
    end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")

    reports, all_runs = [], []
    for counter_id in ids:
        query = select(c.c.ts, c.c.count).where(c.c.counter_id == counter_id, c.c.ts >= start)
        counts = pd.read_sql(query, engine)
        counts["ts"] = pd.to_datetime(counts["ts"], utc=True)
        stats, runs = counter_gaps(counter_id, counts, start, end)
        reports.append(stats)
        all_runs.append(runs)

    now = datetime.now(UTC)
    report_df = pd.DataFrame([asdict(r) for r in reports])
    report_df["computed_at"] = now
    runs_df = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    with engine.begin() as conn:
        conn.execute(delete(schema.zero_runs))
        conn.execute(delete(schema.gap_report))
    db.upsert(engine, schema.gap_report, report_df, ["counter_id"])
    if not runs_df.empty:
        db.upsert(engine, schema.zero_runs, runs_df, ["counter_id", "start_ts"])

    summary = summarise(reports, start, end)
    path = Path(results_dir) / "gap_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def summarise(reports: list[CounterGaps], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    kept = [r for r in reports if not r.excluded]
    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "rule": {
            "zero_run_hours": config.ZERO_RUN_HOURS,
            "max_missing_share": config.MAX_MISSING_SHARE,
        },
        "counters_total": len(reports),
        "counters_kept": len(kept),
        "counters_excluded": len(reports) - len(kept),
        "hours_in_window": reports[0].expected_hours if reports else 0,
        "counters": [
            {
                "counter_id": r.counter_id,
                "missing_share": r.missing_share,
                "absent_hours": r.absent_hours,
                "zero_run_hours": r.zero_run_hours,
                "excluded": r.excluded,
            }
            for r in sorted(reports, key=lambda r: r.missing_share)
        ],
    }


if __name__ == "__main__":
    summary = refresh(db.get_engine())
    print(
        f"{summary['counters_kept']} of {summary['counters_total']} counters kept "
        f"({summary['window_start']} to {summary['window_end']})"
    )
