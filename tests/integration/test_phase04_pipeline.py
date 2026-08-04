import json
from pathlib import Path

from scripts.eval.render_phase04_reports import render_reports
from scripts.eval.select_phase04 import select_winner


def _card(run_id: str, effective: int, *, status: str = "evaluated") -> dict:
    stage = "dpo" if "-dpo-" in run_id else "sft"
    card = {
        "run_id": run_id,
        "status": status,
        "stage": stage,
        "effective_pass": {"numerator": effective, "denominator": 51},
        "aggregate_dimensions": {
            "protocol": {"score": 0.95},
            "task_correctness": {"score": effective / 51},
        },
        "timing": {"total_ms_mean": 100.0, "total_ms_p95": 120.0},
        "scenario_slices": {},
        "parameter_billions": 0.6 if "0.6b" in run_id else 1.7,
    }
    if stage == "dpo":
        card["parent_run_id"] = run_id.split("-dpo-")[0] + "-sft"
    return card


def test_offline_phase04_reports_include_every_run_and_selection(tmp_path: Path) -> None:
    cards = [
        _card("qwen3-0.6b-sft", 20),
        _card("qwen3-1.7b-sft", 25),
        _card("qwen3-0.6b-dpo-b01", 22),
        _card("qwen3-0.6b-dpo-b03", 0, status="failed"),
        _card("qwen3-1.7b-dpo-b01", 28),
        _card("qwen3-1.7b-dpo-b03", 27),
    ]
    selection = select_winner(cards)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    m1, m2 = render_reports(cards, selection, tmp_path / "reports", diffs=[])
    m1_text = m1.read_text(encoding="utf-8")
    m2_text = m2.read_text(encoding="utf-8")
    caveat = "LLaMA-Factory CPU latency is not directly comparable to M0 llama.cpp latency"
    assert caveat in m1_text
    assert caveat in m2_text
    assert "qwen3-0.6b-sft" in m1_text and "qwen3-1.7b-sft" in m1_text
    for card in cards:
        if card["stage"] == "dpo":
            assert card["run_id"] in m2_text
    assert "failed" in m2_text
    assert selection["winner"] in m2_text
