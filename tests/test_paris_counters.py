import io

import pandas as pd

from ingest.sources import paris_counters as pc

HEADER = (
    "Identifiant du compteur;Nom du compteur;Identifiant du site de comptage;"
    "Nom du site de comptage;Comptage horaire;Date et heure de comptage;"
    "Date d'installation du site de comptage;Lien vers photo du site de comptage;"
    "Coordonnées géographiques"
)

# 2023 style: UTF-8 with BOM, ';', ISO timestamps with offset.
FILE_2023 = (
    "﻿" + HEADER + "\n"
    "100003096-353242251;97 avenue Denfert Rochereau SO-NE;100003096;97 avenue Denfert Rochereau;"
    "4;2023-01-01T07:00:00+01:00;2012-02-22;https://x/1.jpg;48.83504,2.33314\n"
).encode("utf-8")

# 2024 style: quoted, ';', float counts, offset after a space.
FILE_2024 = (
    '"' + HEADER.replace(";", '";"') + '"\n'
    '"100044493-101044493";"67 boulevard Voltaire SE-NO";"100044493";"67 boulevard Voltaire";'
    "155.0;2024-01-02 19:00:00.000 +0100;2018-06-27 00:00:00.000;{'https://x/2.jpg'};"
    '"48.86, 2.37"\n'
).encode("utf-8")

# 2025 style: cp1252, ',', the DST fall-back hour appears twice in local time.
FILE_2025 = (
    '"' + HEADER.replace(";", '","') + '"\n'
    '"100003096-101003096","97 avenue Denfert Rochereau SO-NE","100003096","Denfert",'
    "3,2025-10-26 02:00:00.000 +0200,2012-02-22 00:00:00.000,\"['https://x/3.jpg']\",\"48.83504, 2.33314\"\n"
    '"100003096-101003096","97 avenue Denfert Rochereau SO-NE","100003096","Denfert",'
    "2,2025-10-26 02:00:00.000 +0100,2012-02-22 00:00:00.000,\"['https://x/3.jpg']\",\"48.83504, 2.33314\"\n"
).encode("cp1252")


def _read(content: bytes) -> pd.DataFrame:
    return pd.concat(pc.read_history_csv(io.BytesIO(content)))


def test_reads_2023_format():
    df = _read(FILE_2023)
    row = df.iloc[0]
    assert row.counter_id == "100003096-353242251"
    assert row.ts == pd.Timestamp("2023-01-01T06:00:00Z")
    assert row["count"] == 4
    assert (row.lat, row.lon) == (48.83504, 2.33314)
    assert str(row.installed_on) == "2012-02-22"


def test_reads_2024_format():
    df = _read(FILE_2024)
    row = df.iloc[0]
    assert row.ts == pd.Timestamp("2024-01-02T18:00:00Z")
    assert row["count"] == 155
    assert row.lat == 48.86


def test_reads_cp1252_and_keeps_both_dst_hours():
    df = _read(FILE_2025)
    assert list(df.ts) == [pd.Timestamp("2025-10-26T00:00:00Z"), pd.Timestamp("2025-10-26T01:00:00Z")]
    assert df.counter_name.iloc[0] == "97 avenue Denfert Rochereau SO-NE"


def test_live_csv():
    content = (
        b"id_compteur;nom_compteur;id;name;sum_counts;date;installation_date;coordinates\n"
        b"100050132-101050132;243 bd Saint Germain NO-SE;100050132;243 bd Saint Germain;40.0;"
        b"2026-10-31T00:00:00+00:00;2018-12-12;48.86179, 2.32014\n"
    )
    df = pc.parse_live_csv(content)
    assert df.ts.iloc[0] == pd.Timestamp("2026-10-31T00:00:00Z")
    assert df.site_id.iloc[0] == "100050132"
    assert df.lon.iloc[0] == 2.32014


def test_split_filters_window_drops_empty_counts_and_rounds():
    df = pd.DataFrame(
        {
            "counter_id": ["a", "a", "a"],
            "counter_name": ["A"] * 3,
            "site_id": ["1"] * 3,
            "site_name": ["S"] * 3,
            "installed_on": [None] * 3,
            "lat": [48.8] * 3,
            "lon": [2.3] * 3,
            "count": [5.0, None, 7.0],
            "ts": pd.to_datetime(
                ["2022-12-31T23:00Z", "2023-01-01T00:00Z", "2023-01-01T01:00Z"], utc=True
            ),
        }
    )
    counts, meta = pc.split(df, pd.Timestamp("2023-01-01").date())
    assert counts["count"].tolist() == [7]
    assert counts["count"].dtype == "int64"
    assert meta.counter_id.tolist() == ["a"]


def test_split_drops_rows_without_counter_id():
    df = pd.DataFrame(
        {
            "counter_id": ["a", None],
            "counter_name": ["A", None],
            "site_id": ["1", None],
            "site_name": ["S", None],
            "installed_on": [None, None],
            "lat": [48.8, None],
            "lon": [2.3, None],
            "count": [5.0, 2048.0],
            "ts": pd.to_datetime(["2024-11-03T19:00Z"] * 2, utc=True),
        }
    )
    counts, meta = pc.split(df)
    assert counts.counter_id.tolist() == ["a"]
    assert meta.counter_id.tolist() == ["a"]
