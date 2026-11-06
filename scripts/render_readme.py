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


SHORT = {"seasonal_naive": "Seasonal naive", "seasonal_naive_4w": "4-week average", "lightgbm": "LightGBM"}


def _short(name: str, result: dict) -> str:
    return SHORT.get(name, result["models"][name])


def backtest_table(result: dict) -> str:
    """Per-fold MAE for every model, plus interval coverage and width for quantile models."""
    names = list(result["models"])
    interval = [n for n in names if "coverage_80" in result["overall"][n]]
    header = ["Fold", "Week from", "Counters"] + [f"{_short(n, result)} MAE" for n in names]
    for n in interval:
        header += [f"{_short(n, result)} 80% coverage", "Interval width"]
    rows = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]

    def cells(scores: dict, bold: bool = False) -> list[str]:
        out = [f"{scores[n]['mae']:.2f}" for n in names]
        for n in interval:
            out += [f"{scores[n]['coverage_80']:.0%}", f"{scores[n]['interval_width']:.1f}"]
        return [f"**{c}**" for c in out] if bold else out

    for fold in result["folds"]:
        rows.append("| " + " | ".join([str(fold["fold"]), fold["cutoff"][:10], str(fold["counters"])] + cells(fold["models"])) + " |")
    rows.append("| " + " | ".join(["**All folds**", "", ""] + cells(result["overall"], bold=True)) + " |")
    snap = result["snapshot"]
    note = (
        f"{len(result['folds'])} rolling-origin folds, each trained on every hour before its cutoff "
        f"and scored on the next {result['horizon_hours']} hours. Data snapshot "
        f"{snap['start'][:10]} to {snap['end'][:10]}, {snap['counters']} counters. "
        "Scores pool every counter-hour with a known count."
    )
    return "\n".join([*rows, "", note])


def overall_table(result: dict) -> str:
    """MAE, RMSE and sMAPE over all folds, and each model's MAE change against seasonal naive."""
    names = list(result["models"])
    base = result["overall"].get("seasonal_naive")
    rows = [
        "| Model | MAE | RMSE | sMAPE % | MAE vs seasonal naive |",
        "| --- | --- | --- | --- | --- |",
    ]
    for n in names:
        s = result["overall"][n]
        change = "" if base is None or n == "seasonal_naive" else f"{(s['mae'] - base['mae']) / base['mae']:+.1%}"
        rows.append(f"| {result['models'][n]} | {s['mae']:.2f} | {s['rmse']:.2f} | {s['smape']:.1f} | {change} |")
    return "\n".join(rows)


def champion_line(decision: dict) -> str:
    verdict = "promoted to champion" if decision["promote"] else "not promoted"
    line = (
        f"Last training run: model version {decision['candidate_version']}, backtest MAE "
        f"{decision['candidate_mae']:.2f}, {verdict} ({decision['reason']})."
    )
    if "champion_version" in decision:
        role = "Previous champion" if decision["promote"] else "Champion"
        line += f" {role}: version {decision['champion_version']}, MAE {decision['champion_mae']:.2f}."
    return line


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
        text = replace_block(text, "OVERALL", overall_table(backtest))
    champion = _load("champion.json")
    if champion:
        text = replace_block(text, "CHAMPION", champion_line(champion))
    gaps = _load("gap_report.json")
    if gaps:
        text = replace_block(text, "GAPS", gap_table(gaps))
    return text


def main() -> None:
    README.write_text(render(README.read_text()))
    print(f"updated {README}")


if __name__ == "__main__":
    main()
