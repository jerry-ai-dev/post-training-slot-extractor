# tests/integration/test_cli_smoke.py
from pathlib import Path

from scripts.eval.run_eval import main


def test_cli_smoke_writes_report(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--backend-config",
            "configs/inference/mock.yaml",
            "--cases",
            "tests/fixtures/phase01_eval.jsonl",
            "--report-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Appointment-Agent 评估分数卡" in captured.out
    assert "任务正确性" in captured.out
    assert (tmp_path / "scorecard-mock-phase02-v08.json").exists()
