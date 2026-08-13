"""Render observational Phase 05 reports without model selection."""

import json
from pathlib import Path


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _seconds(value: float | None) -> str:
    return "N/A" if value is None else f"{value / 1000:.2f}s"


def render_phase05_reports(reports_root: Path) -> tuple[Path, Path]:
    models = []
    for directory in sorted(path for path in reports_root.iterdir() if path.is_dir()):
        result_path = directory / "result.json"
        failure_path = directory / "failure.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            models.append(
                {
                    "model_id": directory.name,
                    "status": "complete",
                    "quality": result.get("quality"),
                    "workloads": result.get("workloads"),
                    "provenance": result.get("manifest"),
                    "source": str(result_path),
                }
            )
        elif failure_path.is_file():
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            models.append(
                {
                    "model_id": directory.name,
                    "status": "failed",
                    "error": failure.get("error"),
                    "quality": None,
                    "workloads": None,
                    "provenance": None,
                    "source": str(failure_path),
                }
            )
    payload = {"comparison_mode": "observational", "models": models, "observations": []}
    summary_json = reports_root / "summary.json"
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    lines = [
        "# 阶段五本地量化模型评估报告",
        "",
        "评估模式：observational。本报告不自动选择模型，也不设置质量门槛。",
        "",
    ]
    lines.extend(f"- `{model['model_id']}`: {model['status']}" for model in models)
    lines.extend(
        [
            "",
            "## 质量结果",
            "",
            "| Model | Protocol | Task correctness | Cases |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in models:
        quality = model.get("quality") or {}
        dimensions = quality.get("aggregate_dimensions", {})
        protocol = dimensions.get("protocol", {}).get("score")
        task = dimensions.get("task_correctness", {}).get("score")
        lines.append(
            f"| {model['model_id']} | {_percent(protocol)} | {_percent(task)} | "
            f"{quality.get('n', 'N/A')} |"
        )
    scenario_names = (
        "missing_information",
        "tool_call",
        "tool_result",
        "multi_turn",
        "confirmation",
        "unrelated",
    )
    lines.extend(
        [
            "",
            "## 场景切片",
            "",
            "| Model | Missing info | Tool call | Tool result | Multi-turn | "
            "Confirmation | Unrelated |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        slices = (model.get("quality") or {}).get("scenario_slices", {})
        values = [
            _percent(slices.get(name, {}).get("task_correctness"))
            for name in scenario_names
        ]
        lines.append(f"| {model['model_id']} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            "## 速度",
            "",
            "| Model | Quality mean | Quality P95 | tok/s | Short hot | 4K hot | Size GiB |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        quality = model.get("quality") or {}
        timing = quality.get("aggregate_timing") or {}
        workloads = model.get("workloads") or {}
        short = workloads.get("short", {}).get("hot", {}).get("total_ms", {}).get("mean")
        long = workloads.get("4k", {}).get("hot", {}).get("total_ms", {}).get("mean")
        size = (
            workloads.get("short", {})
            .get("hot", {})
            .get("file_size_bytes", {})
            .get("mean")
        )
        tps = timing.get("tokens_per_s_mean")
        lines.append(
            f"| {model['model_id']} | {_seconds(timing.get('total_ms_mean'))} | "
            f"{_seconds(timing.get('total_ms_p95'))} | "
            f"{'N/A' if tps is None else f'{tps:.2f}'} | {_seconds(short)} | "
            f"{_seconds(long)} | {'N/A' if size is None else f'{size / 1024**3:.2f}'} |"
        )
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- `summary.json`：机器可读的全模型汇总。",
            "- `<model-id>/quality.json`：51 条逐样本输出、评分和场景切片。",
            "- `<model-id>/workloads.json`：short/medium/2K/4K 冷热统计。",
            "- `<model-id>/manifest.json`：量化产物与 lineage。",
        ]
    )
    summary_md = reports_root / "summary.md"
    report = "\n".join(lines) + "\n"
    summary_md.write_text(report, encoding="utf-8", newline="\n")
    (reports_root / "report.md").write_text(report, encoding="utf-8", newline="\n")
    return summary_json, summary_md
