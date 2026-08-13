import json
from pathlib import Path

from scripts.eval.phase05_artifacts import write_failure, write_model_result
from scripts.eval.phase05_reports import render_phase05_reports


def test_report_is_observational_and_has_no_selection_or_gate_fields(tmp_path: Path):
    write_model_result(
        tmp_path,
        "model-a",
        {
            "status": "complete",
            "quality": {"score": 0.5},
            "workloads": {"short": {"cold": {"total_ms": {"mean": 10}}}},
            "manifest": {"model_id": "model-a"},
        },
    )
    write_failure(tmp_path, "model-b", RuntimeError("failed"))
    summary_json, summary_md = render_phase05_reports(tmp_path)
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    text = summary_md.read_text(encoding="utf-8").lower()
    assert payload["comparison_mode"] == "observational"
    assert "winner" not in payload and "threshold" not in payload
    assert "pass/fail" not in text
    assert "cold" in payload["models"][0]["workloads"]["short"]
    assert "## 质量结果" in summary_md.read_text(encoding="utf-8")
    assert "| Model | Protocol | Task correctness | Cases |" in summary_md.read_text(
        encoding="utf-8"
    )
    assert "## 场景切片" in summary_md.read_text(encoding="utf-8")
    assert "## 速度" in summary_md.read_text(encoding="utf-8")
