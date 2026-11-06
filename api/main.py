"""Forecast API: the next hours for a counter, with an 80% interval.

Forecasts are issued once a week by the `forecast_weekly` DAG and stored in Postgres; this service
only reads them, so every number it returns is one the daily scoring DAG later checks.

Run locally: BIKES_DATABASE_URL=... uv run uvicorn api.main:app --port 8090
"""

import os
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, select, text

from ingest import schema

DEFAULT_URL = "postgresql+psycopg://bikes:bikes@localhost:5433/bikes"

app = FastAPI(title="Paris bike forecast", version="1.0")
_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = create_engine(os.getenv("BIKES_DATABASE_URL", DEFAULT_URL), pool_pre_ping=True)
    return _engine


def now() -> datetime:
    return datetime.now(UTC)


class Hour(BaseModel):
    ts: datetime
    q10: float
    q50: float
    q90: float


class Forecast(BaseModel):
    counter_id: str
    counter_name: str | None
    model_version: str
    origin: datetime
    issued_at: datetime
    hours: list[Hour]


@app.get("/health")
def health() -> dict:
    with engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/counters")
def counters() -> list[dict]:
    f, c = schema.forecasts.c, schema.counters.c
    query = (
        select(f.counter_id, c.counter_name)
        .distinct()
        .select_from(schema.forecasts.outerjoin(schema.counters, f.counter_id == c.counter_id))
        .where(f.source == "live")
        .order_by(f.counter_id)
    )
    with engine().connect() as conn:
        return [{"counter_id": r.counter_id, "counter_name": r.counter_name} for r in conn.execute(query)]


@app.get("/forecast/{counter_id}", response_model=Forecast)
def forecast(
    counter_id: Annotated[str, Path(max_length=64)],
    hours: Annotated[int, Query(ge=1, le=168)] = 24,
) -> Forecast:
    start = now().replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=hours)
    f, c = schema.forecasts.c, schema.counters.c
    with engine().connect() as conn:
        known = conn.execute(select(c.counter_name).where(c.counter_id == counter_id)).first()
        if known is None:
            raise HTTPException(status_code=404, detail=f"unknown counter {counter_id}")
        # The most recently issued live forecast that covers the current hour.
        issue = conn.execute(
            select(f.origin, f.issued_at, f.model_version)
            .where(f.source == "live", f.counter_id == counter_id, f.ts == start)
            .order_by(f.issued_at.desc())
            .limit(1)
        ).first()
        if issue is None:
            raise HTTPException(
                status_code=503,
                detail="no issued forecast covers the current hour yet (the weekly forecast is issued on Mondays)",
            )
        rows = conn.execute(
            select(f.ts, f.q10, f.q50, f.q90)
            .where(f.source == "live", f.counter_id == counter_id, f.origin == issue.origin, f.ts >= start, f.ts < end)
            .order_by(f.ts)
        ).all()
    return Forecast(
        counter_id=counter_id,
        counter_name=known.counter_name,
        model_version=issue.model_version,
        origin=issue.origin,
        issued_at=issue.issued_at,
        hours=[Hour(ts=r.ts, q10=r.q10, q50=r.q50, q90=r.q90) for r in rows],
    )
