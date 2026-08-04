import json
from pathlib import Path

from scripts.eval.run_phase04_local import (
    PHASE04_RUN_IDS,
    build_api_config,
    is_complete,
    postprocess,
)


def test_build_api_config_uses_manifest_base_and_downloaded_adapter(tmp_path: Path) -> None:
    run = tmp_path / "phase04-qwen3-0.6b-sft"
    (run / "adapter").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"base_model": "Qwen/Qwen3-0.6B", "adapter_path": "adapter"}),
        encoding="utf-8",
    )

    config = build_api_config(run)

    assert config == {
        "model_name_or_path": "Qwen/Qwen3-0.6B",
        "adapter_name_or_path": str((run / "adapter").resolve()),
        "finetuning_type": "lora",
        "template": "qwen3",
        "enable_thinking": False,
        "infer_backend": "huggingface",
    }


def test_run_order_is_two_sft_then_four_dpo() -> None:
    assert PHASE04_RUN_IDS[:2] == ("qwen3-0.6b-sft", "qwen3-1.7b-sft")
    assert all("-dpo-" in run_id for run_id in PHASE04_RUN_IDS[2:])


def test_complete_requires_all_evaluation_artifacts(tmp_path: Path) -> None:
    for name in ("predictions.jsonl", "scorecard.json"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert not is_complete(tmp_path)
    (tmp_path / "server.log").write_text("", encoding="utf-8")
    assert is_complete(tmp_path)


def test_postprocess_writes_four_diffs_selection_and_reports(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for index, run_id in enumerate(PHASE04_RUN_IDS):
        run = runs / f"phase04-{run_id}"
        run.mkdir(parents=True)
        row = {
            "id": "case-1",
            "run_id": run_id,
            "model_output": "{}",
            "dimensions": {
                "protocol": {"score": 1.0},
                "task_correctness": {"score": index / 10},
            },
            "effective_pass": index > 1,
            "failure_reasons": [] if index > 1 else ["task_correctness"],
            "scenario_labels": [],
        }
        (run / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        card = {
            "run_id": run_id,
            "status": "evaluated",
            "effective_pass": {"numerator": int(index > 1), "denominator": 1},
            "aggregate_dimensions": {
                "protocol": {"score": 1.0},
                "task_correctness": {"score": index / 10},
            },
            "timing": {},
        }
        (run / "scorecard.json").write_text(json.dumps(card), encoding="utf-8")
        (run / "server.log").write_text("ready", encoding="utf-8")

    selection = postprocess(runs, tmp_path / "reports")

    assert selection["winner"] in PHASE04_RUN_IDS
    assert len(list((tmp_path / "reports" / "phase04-diffs").glob("*.json"))) == 4
    assert (tmp_path / "reports" / "phase04-selection.json").is_file()
    assert (tmp_path / "reports" / "m1-sft" / "README.md").is_file()
    assert (tmp_path / "reports" / "m2-dpo" / "README.md").is_file()
