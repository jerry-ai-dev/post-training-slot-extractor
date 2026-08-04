from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _metric(card: dict[str, Any], name: str) -> str:
    value = card.get("aggregate_dimensions", {}).get(name, {}).get("score")
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _effective(card: dict[str, Any]) -> str:
    value = card.get("effective_pass", {})
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    return "n/a" if numerator is None or denominator is None else f"{numerator}/{denominator}"


def _timing(card: dict[str, Any], name: str) -> str:
    value = card.get("timing", {}).get(name)
    return "n/a" if value is None else f"{float(value):.0f}ms"


def _table(cards: list[dict[str, Any]]) -> str:
    lines = [
        "| run_id | status | protocol | task_correctness | effective_pass | mean | p95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for card in cards:
        lines.append(
            f"| `{card['run_id']}` | {card.get('status', 'unknown')} | "
            f"{_metric(card, 'protocol')} | {_metric(card, 'task_correctness')} | "
            f"{_effective(card)} | {_timing(card, 'total_ms_mean')} | "
            f"{_timing(card, 'total_ms_p95')} |"
        )
    return "\n".join(lines)


def _baseline_table(path: Path) -> str:
    if not path.is_file():
        return "M0 baseline file unavailable."
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    lines = [
        "| M0 model | protocol | task_correctness | effective_pass |",
        "|---|---:|---:|---:|",
    ]
    for model, value in models.items():
        lines.append(
            f"| `{model}` | {value['protocol'] * 100:.1f}% | "
            f"{value['task_correctness'] * 100:.1f}% | {value['effective_pass']} |"
        )
    return "\n".join(lines)


def render_reports(
    cards: list[dict[str, Any]],
    selection: dict[str, Any],
    reports_root: Path,
    *,
    diffs: list[dict[str, Any]],
    baseline_path: Path = Path("reports/baseline-m0/comparison.json"),
) -> tuple[Path, Path]:
    latency_caveat = (
        "LLaMA-Factory CPU latency is not directly comparable to M0 llama.cpp latency."
    )
    sft = sorted((card for card in cards if card.get("stage") == "sft"), key=lambda x: x["run_id"])
    dpo = sorted((card for card in cards if card.get("stage") == "dpo"), key=lambda x: x["run_id"])
    m1 = reports_root / "m1-sft" / "README.md"
    m2 = reports_root / "m2-dpo" / "README.md"
    m1.parent.mkdir(parents=True, exist_ok=True)
    m2.parent.mkdir(parents=True, exist_ok=True)
    m1.write_text(
        "# M1 SFT comparison\n\n"
        "## M0 baseline\n\n"
        f"{_baseline_table(baseline_path)}\n\n"
        "## SFT runs\n\n"
        f"{_table(sft)}\n\n"
        f"> {latency_caveat}\n",
        encoding="utf-8",
        newline="\n",
    )
    diff_lines = []
    for diff in diffs:
        diff_lines.append(
            f"- `{diff['left_run_id']}` → `{diff['right_run_id']}`: "
            f"net effective pass {diff['net_effective_pass']:+d}"
        )
    m2.write_text(
        "# M2 DPO comparison\n\n"
        f"Winner: `{selection['winner']}`\n\n"
        "## SFT parents and DPO runs\n\n"
        f"{_table([*sft, *dpo])}\n\n"
        "## Selection rule application\n\n"
        f"Contenders within one effective pass: "
        f"{', '.join(selection['selection']['contenders_within_one'])}.\n\n"
        "Ineligible and failed runs remain in the table above.\n\n"
        "## Parent diffs\n\n"
        f"{chr(10).join(diff_lines) if diff_lines else 'No real diffs available yet.'}\n\n"
        f"> {latency_caveat}\n",
        encoding="utf-8",
        newline="\n",
    )
    return m1, m2


def _load_cards(runs_root: Path) -> list[dict[str, Any]]:
    cards = []
    for path in sorted(runs_root.glob("phase04-*/scorecard.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        run_id = card["run_id"]
        card.setdefault("stage", "dpo" if "-dpo-" in run_id else "sft")
        cards.append(card)
    return cards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render phase04 M1 and M2 reports.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--diff", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        diffs = [json.loads(path.read_text(encoding="utf-8")) for path in args.diff]
        m1, m2 = render_reports(
            _load_cards(args.runs_root), selection, args.reports_root, diffs=diffs
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"report error: {error}", file=sys.stderr)
        return 2
    print(m1)
    print(m2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
