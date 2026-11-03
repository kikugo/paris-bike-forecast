"""Engine creation and idempotent upserts."""

from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import Engine, Table, create_engine, select, text
from sqlalchemy.dialects import postgresql, sqlite

from ingest import config, schema

CHUNK_ROWS = 5_000


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or config.database_url(), pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    schema.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            # DDL takes no bind parameters, so the window start is inlined as a date literal.
            sql = schema.COUNTS_CLEAN_VIEW.replace(
                ":window_start", f"'{config.WINDOW_START.isoformat()}'::timestamptz"
            )
            conn.execute(text(sql))


def _records(df: pd.DataFrame) -> list[dict]:
    out = df.astype(object).where(df.notna(), None)
    return out.to_dict("records")


def upsert(
    engine: Engine,
    table: Table,
    df: pd.DataFrame,
    keys: Iterable[str],
    only_if: str | None = None,
) -> int:
    """Insert rows, updating non-key columns on key conflict.

    `only_if` is a SQL condition on the existing row; when given, conflicting rows are
    updated only if it holds (used so forecast weather never overwrites archive weather).
    """
    if df.empty:
        return 0
    keys = list(keys)
    dialect = postgresql if engine.dialect.name == "postgresql" else sqlite
    rows = _records(df)
    with engine.begin() as conn:
        for start in range(0, len(rows), CHUNK_ROWS):
            stmt = dialect.insert(table).values(rows[start : start + CHUNK_ROWS])
            update = {c: stmt.excluded[c] for c in df.columns if c not in keys}
            if update:
                where = text(only_if) if only_if else None
                stmt = stmt.on_conflict_do_update(index_elements=keys, set_=update, where=where)
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=keys)
            conn.execute(stmt)
    return len(rows)


def log_ingest(engine: Engine, source: str, key: str, rows: int) -> None:
    df = pd.DataFrame(
        [{"source": source, "key": key, "rows": rows, "loaded_at": datetime.now(UTC)}]
    )
    upsert(engine, schema.ingest_log, df, ["source", "key"])


def already_loaded(engine: Engine, source: str, key: str) -> bool:
    t = schema.ingest_log
    with engine.connect() as conn:
        row = conn.execute(select(t.c.rows).where(t.c.source == source, t.c.key == key)).first()
    return row is not None
