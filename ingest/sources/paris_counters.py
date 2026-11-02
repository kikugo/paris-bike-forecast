"""Paris bike counters ("Comptage vélo", opendata.paris.fr, ODbL).

Two sources:
- the historical dataset, one zipped CSV per year. Separator, encoding and timestamp
  format change from year to year, and each file is 1-2 GB because of a photo-URL column,
  so files are streamed in chunks and only the needed columns are kept;
- the live dataset (13 rolling months, updated daily), read through the Explore API export.
"""

import io
import re
import zipfile
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

import httpx
import pandas as pd

API = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"
HISTORY_DATASET = "comptage-velo-historique-donnees-compteurs"
LIVE_DATASET = "comptage-velo-donnees-compteurs"

COUNT_COLUMNS = ["counter_id", "ts", "count"]
COUNTER_COLUMNS = ["counter_id", "counter_name", "site_id", "site_name", "installed_on", "lat", "lon"]

# Positions in the historical CSV: id, name, site id, site name, count, timestamp,
# installation date, (photo links), coordinates.
_HISTORY_POSITIONS = [0, 1, 2, 3, 4, 5, 6, 8]
_HISTORY_NAMES = [
    "counter_id",
    "counter_name",
    "site_id",
    "site_name",
    "count",
    "ts",
    "installed_on",
    "coordinates",
]
_TITLE = re.compile(r"^(\d{4})[-_]comptage-velo-donnees-compteurs")
# "2024-01-02 19:00:00.000 +0100" -> "2024-01-02 19:00:00.000+01:00"
_SPACED_OFFSET = re.compile(r"\s*([+-]\d{2}):?(\d{2})$")


def history_attachments(client: httpx.Client) -> dict[int, str]:
    """Year -> download URL of the per-counter hourly file for that year."""
    resp = client.get(f"{API}/{HISTORY_DATASET}/attachments")
    resp.raise_for_status()
    out = {}
    for item in resp.json()["attachments"]:
        meta = item["metas"]
        m = _TITLE.match(meta["title"])
        if m:
            out[int(m.group(1))] = meta["url"]
    return dict(sorted(out.items()))


def download(client: httpx.Client, url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for block in resp.iter_bytes(1 << 20):
                fh.write(block)
    return dest


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse local timestamps with a UTC offset (both formats seen in the files) to UTC."""
    cleaned = values.astype(str).str.strip().str.replace(_SPACED_OFFSET, r"\1:\2", regex=True)
    return pd.to_datetime(cleaned, utc=True, format="ISO8601")


def _sniff(head: bytes) -> tuple[str, str]:
    try:
        head.decode("utf-8")
        encoding = "utf-8-sig"
    except UnicodeDecodeError as exc:
        # A multi-byte character cut at the end of the sample is still UTF-8.
        encoding = "utf-8-sig" if exc.start >= len(head) - 3 else "cp1252"
    first_line = head.split(b"\n", 1)[0]
    sep = ";" if first_line.count(b";") > first_line.count(b",") else ","
    return encoding, sep


def _coords(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = values.astype(str).str.split(",", n=1, expand=True)
    if parts.shape[1] < 2:
        nan = pd.Series(float("nan"), index=values.index)
        return nan, nan
    lat = pd.to_numeric(parts[0].str.strip(), errors="coerce")
    lon = pd.to_numeric(parts[1].str.strip(), errors="coerce")
    return lat, lon


def read_history_csv(raw: io.BufferedIOBase, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Stream one historical CSV as normalised chunks (UTC timestamps, numeric counts)."""
    buffered = io.BufferedReader(raw) if not hasattr(raw, "peek") else raw
    encoding, sep = _sniff(buffered.peek(65536)[:65536])
    reader = pd.read_csv(
        buffered,
        sep=sep,
        encoding=encoding,
        usecols=_HISTORY_POSITIONS,
        dtype=str,
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk.columns = _HISTORY_NAMES
        chunk["count"] = pd.to_numeric(chunk["count"], errors="coerce")
        chunk["ts"] = parse_timestamps(chunk["ts"])
        chunk["installed_on"] = pd.to_datetime(
            chunk["installed_on"].str.slice(0, 10), errors="coerce"
        ).dt.date
        chunk["lat"], chunk["lon"] = _coords(chunk.pop("coordinates"))
        yield chunk


def read_history_zip(path: Path, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as raw:
            yield from read_history_csv(raw, chunksize)


def live_export(client: httpx.Client, since: date | None = None) -> pd.DataFrame:
    """Hourly counts from the live dataset, optionally from `since` (UTC date) onwards."""
    params = {
        "select": "id_compteur,nom_compteur,id,name,sum_counts,date,installation_date,coordinates",
        "timezone": "UTC",
        "delimiter": ";",
    }
    if since is not None:
        params["where"] = f"date >= date'{since.isoformat()}'"
    resp = client.get(f"{API}/{LIVE_DATASET}/exports/csv", params=params)
    resp.raise_for_status()
    return parse_live_csv(resp.content)


def parse_live_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), sep=";", dtype=str)
    df = df.rename(
        columns={
            "id_compteur": "counter_id",
            "nom_compteur": "counter_name",
            "id": "site_id",
            "name": "site_name",
            "sum_counts": "count",
            "date": "ts",
            "installation_date": "installed_on",
        }
    )
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df["ts"] = parse_timestamps(df["ts"])
    df["installed_on"] = pd.to_datetime(df["installed_on"], errors="coerce").dt.date
    df["lat"], df["lon"] = _coords(df.pop("coordinates"))
    return df


def split(df: pd.DataFrame, window_start: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split normalised rows into (counts, counter metadata), keeping rows in the window."""
    if window_start is not None:
        df = df[df["ts"] >= pd.Timestamp(datetime.combine(window_start, datetime.min.time()), tz="UTC")]
    # Rows without a count carry no information; they show up as absent hours in the gap report.
    # The 2024 file also holds one full-year series with no counter id, name or site; it
    # cannot be attributed to a counter, so it is dropped too.
    df = df[df["count"].notna() & df["counter_id"].notna()]
    counts = df[COUNT_COLUMNS].copy()
    counts["count"] = counts["count"].round().astype("int64")
    counters = (
        df[COUNTER_COLUMNS]
        .sort_values("counter_id")
        .drop_duplicates("counter_id", keep="last")
        .reset_index(drop=True)
    )
    return counts.reset_index(drop=True), counters
