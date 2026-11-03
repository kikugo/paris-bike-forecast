"""Fill the generated blocks of README.md from results/*.json.

Every number in the README comes from here; nothing between the markers is edited by hand.
Run: uv run python -m scripts.render_readme
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "results"


def _load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def data_table(sources: dict) -> str:
    rows = [
        "| Source | Licence | Coverage | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for s in sources["sources"].values():
        notes = []
        if "counters" in s:
            notes.append(f"{s['counters']} counters, {s['rows']:,} hourly rows")
        if "files" in s:
            notes.append(f"{s['files']} yearly files")
        rows.append(
            f"| [{s['name']}]({s['page']}) | {s['licence']} | {s['first']} to {s['last']} | {'; '.join(notes)} |"
        )
    rows.append("")
    rows.append(
        f"Checked {sources['checked_at'][:10]} by `ingest/sources_report.py`. "
        f"This project ingests from {sources['window_start']} onwards."
    )
    return "\n".join(rows)


def gap_table(report: dict) -> str:
    rule = report["rule"]
    return "\n".join(
        [
            "| Window | Hours | Counters | Kept | Excluded |",
            "| --- | --- | --- | --- | --- |",
            (f"| {report['window_start'][:10]} to {report['window_end'][:10]} | {report['hours_in_window']:,} "
            f"| {report['counters_total']} | {report['counters_kept']} | {report['counters_excluded']} |"),
            "",
            (f"A counter is excluded when more than {rule['max_missing_share']:.0%} of its hours are missing "
            f"(no row, or inside a run of at least {rule['zero_run_hours']} zero hours)."),
        ]
    )


def backtest_table(result: dict) -> str:
    names = list(result["models"])
    header = ["Fold", "Week from", "Counters"]
    for name in names:
        label = result["models"][name]
        header += [f"{label} MAE", "RMSE", "sMAPE %"]
    rows = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]

    def cells(scores: dict) -> list[str]:
        out = []
        for name in names:
            s = scores[name]
            out += [f"{s['mae']:.2f}", f"{s['rmse']:.2f}", f"{s['smape']:.1f}"]
        return out

    for fold in result["folds"]:
        row = [str(fold["fold"]), fold["cutoff"][:10], str(fold["counters"])] + cells(fold["models"])
        rows.append("| " + " | ".join(row) + " |")
    total = ["**All folds**", "", ""] + [f"**{c}**" for c in cells(result["overall"])]
    rows.append("| " + " | ".join(total) + " |")
    snap = result["snapshot"]
    note = (
        f"{len(result['folds'])} rolling-origin folds, each trained on every hour before its cutoff "
        f"and scored on the next {result['horizon_hours']} hours. Data snapshot "
        f"{snap['start'][:10]} to {snap['end'][:10]}, {snap['counters']} counters. "
        "Scores pool every counter-hour with a known count."
    )
    rows += ["", note]
    return "\n".join(rows)


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:START -->\n).*?(<!-- {name}:END -->)", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"README has no {name} block")
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def render(text: str) -> str:
    sources = _load("data_sources.json")
    if sources:
        text = replace_block(text, "DATA", data_table(sources))
    backtest = _load("backtest.json")
    if backtest:
        text = replace_block(text, "BACKTEST", backtest_table(backtest))
    gaps = _load("gap_report.json")
    if gaps:
        text = replace_block(text, "GAPS", gap_table(gaps))
    return text


def main() -> None:
    README.write_text(render(README.read_text()))
    print(f"updated {README}")


if __name__ == "__main__":
    main()
