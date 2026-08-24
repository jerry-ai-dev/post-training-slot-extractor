"""Run Phase 06 Round 008 Q4 CPU, prompt-cache, and compact-prompt experiments."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.eval.run_phase06_llamacpp import stop, wait_ready
from slot_extractor.evaluation.runner import default_scorers
from slot_extractor.inference.llama_server import LlamaServerBackend, LlamaServerConfig
from slot_extractor.prompts.rules import (
    FINAL_SCHEMA_HINT,
    TOOL_SCHEMA_HINT,
    render_tool_descriptions,
)
from slot_extractor.prompts.template import PromptBuilder
from slot_extractor.schemas.sample import Sample, load_samples

MODEL = Path("models/gguf/phase06-round006-local/r004-qwen3-0.6b-sft-Q4_K_M.gguf")
SERVER = Path("deployment/llama_cpp/bin/llama-server.exe")

COMPACT_RULES = (
    "你是按摩预约Agent，只输出一个JSON对象，不解释、不用Markdown。\n"
    "状态：最新明确修改覆盖旧值，未修改字段继承current_state；只清空被修改条件的直接依赖项。"
    "相对时间按当前时间换算；仅有上午/下午等模糊时段时start_time=null。"
    "按摩类型或部位写preferences。\n"
    "字段：gender_preference是用户性别要求；technician_gender是工具核实的技师性别。"
    "二者仅female/male/null。start_time为YYYY-MM-DD HH:MM/null；"
    "duration_minutes为正整数/null；missing_info只含start_time、duration_minutes并按此顺序。"
    "info_complete仅表示时间和时长齐全。\n"
    "决策：无关输入用handoff且unrelated=true。缺时间/时长则final并分别用"
    "ask_start_time_and_duration、ask_start_time、ask_duration。"
    "信息完整且无当前有效工具结果时，有工具就tool_call。工具结果available用"
    "confirm_available并请求确认；unavailable/not_found/no_match分别用"
    "inform_unavailable/inform_not_found/inform_no_match。用户确认available才用"
    "booking_authorized且confirmation=true；拒绝用appointment_paused；"
    "明确知悉失败结果用acknowledge_result。\n"
    "工具证据：可用性只能来自最新tool消息，查询条件变化后旧结果失效并重新查询。"
    "specific结果保留请求姓名；search匹配复制唯一candidate，no_match姓名为null。"
    "工具返回gender只写technician_gender，不自动改gender_preference。"
    "回复必须准确说明技师、时间、时长和结果，不得编造。"
)


class CompactPromptBuilder(PromptBuilder):
    def build_messages(
        self, sample: Sample, *, include_tool_descriptions: bool = True
    ) -> list[dict[str, object]]:
        available = sample.input.get("available_tools")
        tools = render_tool_descriptions(available if isinstance(available, list) else None)
        tool_block = f"{TOOL_SCHEMA_HINT}\n{tools}\n" if tools else ""
        system = (
            f"{COMPACT_RULES}\n{FINAL_SCHEMA_HINT}\n{tool_block}"
            f"当前时间：{sample.input.get('current_time', '')}\n"
            f"当前状态：{self._compact_json(sample.input.get('current_state'))}"
        )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": system, "_sample_id": sample.id}
        ]
        messages.extend(self._history_turns(sample))
        user = sample.input.get("user_input")
        if isinstance(user, str) and user:
            messages.append({"role": "user", "content": user})
        return messages


def parameter_matrix(output: Path, port: int) -> list[dict[str, Any]]:
    settings = [
        (4, 4, 2048, 512),
        (8, 8, 2048, 512),
        (16, 16, 2048, 512),
        (8, 16, 2048, 512),
        (8, 8, 512, 128),
        (8, 8, 1024, 256),
        (8, 8, 2048, 256),
    ]
    rows = []
    sample = load_samples(Path("data/eval/phase06_holdout_v0.3.jsonl"))[-1]
    messages = PromptBuilder().build_messages(sample)
    for regular, batch_threads, batch, ubatch in settings:
        params = {
            "threads": regular,
            "threads_batch": batch_threads,
            "batch": batch,
            "ubatch": ubatch,
        }
        label = f"params-t{regular}-tb{batch_threads}-b{batch}-ub{ubatch}"
        process, stream = start_server(output / f"{label}.log", port, cache=False, params=params)
        try:
            backend = LlamaServerBackend(
                LlamaServerConfig(label, f"http://127.0.0.1:{port}/v1", max_tokens=512)
            )
            result = backend.generate(messages)
        finally:
            stop(process)
            stream.close()
        rows.append(
            {
                **params,
                "input_tokens": result.input_tokens,
                "prefill_ms": result.prefill_ms,
                "prefill_tokens_per_s": result.prefill_tokens_per_s,
                "decode_tokens_per_s": result.decode_tokens_per_s,
                "total_ms": result.total_ms,
            }
        )
        print(f"parameters: t={regular} tb={batch_threads} b={batch} ub={ubatch}", flush=True)
    return rows


def start_server(log: Path, port: int, *, cache: bool, params: dict[str, int]):
    command = [
        str(SERVER),
        "-m",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--threads",
        str(params["threads"]),
        "--threads-batch",
        str(params["threads_batch"]),
        "--batch-size",
        str(params["batch"]),
        "--ubatch-size",
        str(params["ubatch"]),
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "0",
        "--parallel",
        "1",
        "--jinja",
        "--cache-prompt" if cache else "--no-cache-prompt",
    ]
    stream = log.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    wait_ready(process, f"http://127.0.0.1:{port}/v1", timeout_s=180)
    return process, stream


def selected_samples() -> list[Sample]:
    samples = load_samples(Path("data/eval/phase06_holdout_v0.3.jsonl"))
    indices = [0, 3, 6, 10, 14, 17, 20, 23]
    return [samples[index] for index in indices]


def latency_experiment(
    output: Path, port: int, cache: bool, compact: bool, params: dict[str, int]
) -> dict[str, Any]:
    label = f"{'compact' if compact else 'full'}-{'cache' if cache else 'cold'}"
    process, stream = start_server(output / f"{label}.log", port, cache=cache, params=params)
    builder = CompactPromptBuilder() if compact else PromptBuilder()
    backend = LlamaServerBackend(
        LlamaServerConfig(label, f"http://127.0.0.1:{port}/v1", max_tokens=512)
    )
    rows = []
    try:
        for sample in selected_samples():
            result = backend.generate(builder.build_messages(sample))
            rows.append(
                {
                    "id": sample.id,
                    "input_tokens": result.input_tokens,
                    "prefill_ms": result.prefill_ms,
                    "ttft_ms": result.first_token_ms,
                    "decode_ms": result.decode_ms,
                    "total_ms": result.total_ms,
                }
            )
    finally:
        stop(process)
        stream.close()
    return {
        "label": label,
        "rows": rows,
        "median": {
            key: statistics.median(row[key] for row in rows if row[key] is not None)
            for key in ("input_tokens", "prefill_ms", "ttft_ms", "decode_ms", "total_ms")
        },
    }


def quality_experiment(
    output: Path,
    port: int,
    compact: bool,
    params: dict[str, int],
    *,
    cache: bool = False,
) -> dict[str, Any]:
    label = f"{'compact' if compact else 'full'}-{'cache' if cache else 'cold'}"
    process, stream = start_server(
        output / f"quality-{label}.log", port, cache=cache, params=params
    )
    builder = CompactPromptBuilder() if compact else PromptBuilder()
    backend = LlamaServerBackend(
        LlamaServerConfig(label, f"http://127.0.0.1:{port}/v1", max_tokens=512)
    )
    scorers = default_scorers()
    records = []
    try:
        for sample in load_samples(Path("data/eval/phase06_holdout_v0.3.jsonl")):
            generation = backend.generate(builder.build_messages(sample))
            dimensions = {
                scorer.dimension: scorer.score(sample, generation)
                for scorer in scorers
                if scorer.applies_to(sample)
            }
            passed = all(score.passed is not False for score in dimensions.values())
            records.append({"id": sample.id, "passed": passed, "output": generation.text})
    finally:
        stop(process)
        stream.close()
    return {
        "label": label,
        "passed": sum(row["passed"] for row in records),
        "n": len(records),
        "rows": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/phase06/round-008/results")
    )
    parser.add_argument("--port", type=int, default=8038)
    parser.add_argument("--cache-quality-only", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache_quality_only:
        params = {"threads": 8, "threads_batch": 16, "batch": 2048, "ubatch": 512}
        result = quality_experiment(args.output, args.port, False, params, cache=True)
        target = args.output / "cache-quality.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(target)
        return 0
    parameters = parameter_matrix(args.output, args.port)
    best = max(parameters, key=lambda row: row["prefill_tokens_per_s"])
    params = {key: int(best[key]) for key in ("threads", "threads_batch", "batch", "ubatch")}
    latency = [
        latency_experiment(args.output, args.port, cache, compact, params)
        for compact in (False, True)
        for cache in (False, True)
    ]
    quality = [
        quality_experiment(args.output, args.port, compact, params) for compact in (False, True)
    ]
    document = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": str(MODEL),
        "parameter_matrix": parameters,
        "selected_parameters": params,
        "latency": latency,
        "quality": quality,
    }
    target = args.output / "round008-results.json"
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
