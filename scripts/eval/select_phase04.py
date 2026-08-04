from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROTOCOL_REGRESSION_LIMIT = 0.02


def _score(card: dict[str, Any], dimension: str) -> float:
    return float(card["aggregate_dimensions"][dimension]["score"])


def select_winner(cards: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {card["run_id"]: card for card in cards}
    runs: dict[str, dict[str, Any]] = {}
    for card in cards:
        run_id = card["run_id"]
        eligible = card.get("status") == "evaluated"
        reasons = [] if eligible else [f"status={card.get('status')}"]
        parent_id = card.get("parent_run_id")
        if eligible and parent_id:
            parent = by_id.get(parent_id)
            if parent is None:
                eligible = False
                reasons.append(f"missing parent {parent_id}")
            else:
                regression = _score(parent, "protocol") - _score(card, "protocol")
                if regression > PROTOCOL_REGRESSION_LIMIT:
                    eligible = False
                    reasons.append(f"protocol regression {regression:.6f} > 0.02")
        runs[run_id] = {"eligible": eligible, "reasons": reasons}
    eligible_cards = [card for card in cards if runs[card["run_id"]]["eligible"]]
    if not eligible_cards:
        raise ValueError("no eligible runs")
    maximum = max(card["effective_pass"]["numerator"] for card in eligible_cards)
    contenders = [
        card for card in eligible_cards if maximum - card["effective_pass"]["numerator"] <= 1
    ]
    contenders.sort(
        key=lambda card: (
            -_score(card, "task_correctness"),
            float(card["parameter_billions"]),
            card["run_id"],
        )
    )
    winner = contenders[0]["run_id"]
    return {
        "winner": winner,
        "runs": runs,
        "selection": {
            "max_effective_pass": maximum,
            "contenders_within_one": [card["run_id"] for card in contenders],
            "tie_break": "task_correctness, parameter_billions, run_id",
        },
    }


def _load_cards(runs_root: Path) -> list[dict[str, Any]]:
    cards = []
    for scorecard in sorted(runs_root.glob("phase04-*/scorecard.json")):
        card = json.loads(scorecard.read_text(encoding="utf-8"))
        run_id = card["run_id"]
        card["parameter_billions"] = 0.6 if "0.6b" in run_id else 1.7
        if "-dpo-" in run_id:
            card["parent_run_id"] = run_id.split("-dpo-")[0] + "-sft"
        cards.append(card)
    return cards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select the phase04 winning run.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = select_winner(_load_cards(args.runs_root))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"selection error: {error}", file=sys.stderr)
        return 2
    print(result["winner"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
