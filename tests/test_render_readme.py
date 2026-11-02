import pytest

from scripts import render_readme as rr

SOURCES = {
    "checked_at": "2026-11-02T00:00:00+00:00",
    "window_start": "2023-01-01",
    "sources": {
        "live": {"name": "Live", "page": "https://x", "licence": "ODbL", "first": "2025-10-02",
                 "last": "2026-11-01", "counters": 3, "rows": 1200},
        "weather": {"name": "Weather", "page": "https://y", "licence": "CC BY 4.0",
                    "first": "1940-01-01", "last": "yesterday"},
    },
}
GAPS = {
    "window_start": "2023-01-01T00:00:00+00:00", "window_end": "2026-11-01T22:00:00+00:00",
    "rule": {"zero_run_hours": 24, "max_missing_share": 0.1},
    "counters_total": 5, "counters_kept": 4, "counters_excluded": 1, "hours_in_window": 33622,
}


def test_tables_come_from_results():
    assert "| [Live](https://x) | ODbL | 2025-10-02 to 2026-11-01 | 3 counters, 1,200 hourly rows |" in rr.data_table(SOURCES)
    table = rr.gap_table(GAPS)
    assert "| 33,622 | 5 | 4 | 1 |" in table
    assert "more than 10%" in table


def test_replace_block_is_idempotent():
    text = "a\n<!-- DATA:START -->\nold\n<!-- DATA:END -->\nb"
    once = rr.replace_block(text, "DATA", "new")
    assert once == "a\n<!-- DATA:START -->\nnew\n<!-- DATA:END -->\nb"
    assert rr.replace_block(once, "DATA", "new") == once


def test_missing_block_raises():
    with pytest.raises(ValueError):
        rr.replace_block("no markers", "DATA", "x")


def test_fills_an_empty_block():
    text = "<!-- GAPS:START -->\n<!-- GAPS:END -->\n"
    assert rr.replace_block(text, "GAPS", "t") == "<!-- GAPS:START -->\nt\n<!-- GAPS:END -->\n"
